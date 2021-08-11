from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import json
import random

import os,argparse,time

from tqdm import tqdm, trange

from networks.POS_utils import BiLSTMPOSTagger, Discriminator, categorical_accuracy

from dataloaders.TP_POS import POS

from pathlib import Path

def train(model, 
        discriminator,
        iterator, 
        adv_iterator,
        optimizer, 
        criterion, 
        adv_criterion, 
        tag_pad_idx,
        LAMBDA,
        device
        ):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch, p_batch in zip(iterator, adv_iterator):
        
        text, tags, _ = batch
        p_text, _, p_tags = p_batch

        text = text.long()
        tags = tags.long()
        p_text = p_text.long()
        p_tags = p_tags.long()
        
        text = text.to(device)
        tags = tags.to(device)
        p_text = p_text.to(device)
        p_tags = p_tags.to(device)
        
        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions, tags)

        # adv loss
        hs = model.hidden_state(p_text)
        adv_predictions = discriminator(hs)
        adv_loss = adv_criterion(adv_predictions, p_tags)
                
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        # merge main task loss with adv loss
        loss = loss - LAMBDA*adv_loss
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion, tag_pad_idx, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, tags, p_tags = batch

            text = text.long()
            tags = tags.long()
            
            text = text.to(device)
            tags = tags.to(device)
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# train a discriminator 1 epoch
def adv_train_epoch(model, discriminator, adv_iterator, adv_optimizer, adv_criterion, device):
    """"
    Train the discriminator to get a meaningful gradient
    """
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    discriminator.train()
    
    for batch in adv_iterator:
        
        text, tags, p_tags = batch

        text = text.long()
        tags = tags.long()
        p_tags = p_tags.long()
            
        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        adv_optimizer.zero_grad()
        
        hs = model.hidden_state(text).detach()
        
        adv_predictions = discriminator(hs)
        adv_loss = adv_criterion(adv_predictions, p_tags)
        
                        
        adv_loss.backward()

        adv_optimizer.step()
        epoch_loss += adv_loss.item()
        
    return epoch_loss / len(adv_iterator)


# train a discriminator 1 epoch
def adv_eval_epoch(model, discriminator, adv_iterator, adv_criterion, device):
    """"
    Train the discriminator to get a meaningful gradient
    """
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    discriminator.eval()

    with torch.no_grad():
        for batch in adv_iterator:
            
            text, tags, p_tags = batch

            text = text.long()
            tags = tags.long()
            p_tags = p_tags.long()
                
            text = text.to(device)
            tags = tags.to(device)
            p_tags = p_tags.to(device)
                        
            hs = model.hidden_state(text).detach()
            
            adv_predictions = discriminator(hs)
            adv_loss = adv_criterion(adv_predictions, p_tags)
                                        
            epoch_loss += adv_loss.item()

    return epoch_loss / len(adv_iterator)




def evaluate_bias(model, iterator, tag_pad_idx, return_value = False):
    
    epoch_acc = 0
    g0_epoch_acc = 0
    g1_epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, tags, p_tags = batch

            text = text.long()
            tags = tags.long()
            p_tags = p_tags.long()
            
            text = text.to(device)
            tags = tags.to(device)
            p_tags = p_tags.to(device)
            
            predictions = model(text)

            g0 = torch.ByteTensor(p_tags.byte().cpu())
            g1 = torch.ByteTensor(((p_tags-1)**2).byte().cpu())

            g0_predictions = predictions[g0]
            g1_predictions = predictions[g1]

            g0_tags = tags[g0]
            g1_tags = tags[g1]
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            g0_predictions = g0_predictions.view(-1, g0_predictions.shape[-1])
            g0_tags = g0_tags.view(-1)

            g1_predictions = g1_predictions.view(-1, g1_predictions.shape[-1])
            g1_tags = g1_tags.view(-1)
                        
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            g0_acc = categorical_accuracy(g0_predictions, g0_tags, tag_pad_idx)
            g1_acc = categorical_accuracy(g1_predictions, g1_tags, tag_pad_idx)

            epoch_acc += acc.item()
            g0_epoch_acc += g0_acc.item()
            g1_epoch_acc += g1_acc.item()

    # print("Accuracy overall: {}".format(epoch_acc / len(iterator)))
    # print("Accuracy group 0: {}".format(g0_epoch_acc / len(iterator)))
    # print("Accuracy group 1: {}".format(g1_epoch_acc / len(iterator)))

    if return_value:
        g0_tags = g0_tags.detach().cpu().numpy()
        g1_tags = g1_tags.detach().cpu().numpy()
        
        # argmax
        g0_predictions = np.argmax(g0_predictions.detach().cpu().numpy(), axis = 1)
        g1_predictions = np.argmax(g1_predictions.detach().cpu().numpy(), axis = 1)

        non_padding_g0_predictions = []
        non_padding_g0_tags = []
        for i, tag in enumerate(g0_tags):
            if tag != tag_pad_idx:
                non_padding_g0_predictions.append(g0_predictions[i])
                non_padding_g0_tags.append(g0_tags[i])

        non_padding_g1_predictions = []
        non_padding_g1_tags = []
        for i, tag in enumerate(g1_tags):
            if tag != tag_pad_idx:
                non_padding_g1_predictions.append(g1_predictions[i])
                non_padding_g1_tags.append(g1_tags[i])

        label_preds = {
            "g0_label":non_padding_g0_tags,
            "g1_label":non_padding_g1_tags,
            "g0_pred":non_padding_g0_predictions,
            "g1_pred":non_padding_g1_predictions,
        }

        return epoch_acc / len(iterator), g0_epoch_acc / len(iterator), g1_epoch_acc / len(iterator), label_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_type', type=int, default = 1)
    args = parser.parse_args()

    # Dataset path
    data_path = "/home/path/Project/adv_decorrelation/data/POS/"

    # model path
    model_path = Path("/home/path/Project/adv_decorrelation/model/")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # different sampleing methods for trust polit SA dataset.
    sampling_method_list = [None, "random", "largest_leakage", "smallest_leakage", "absolute_leakage"]
    
    ####################################
    #         Hyper-parameters         #
    ####################################
    BATCH_SIZE = 64 # different batch size
    LEARNING_RATE = 1e-3
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 100
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    NUM_EPOCHS = 50
    SEED = 960925
    MIN_FREQ = 2
    SAMPLING_INDEX = 10
    LAMBDA = 1e-3
    SAMPLING_METHOD = sampling_method_list[args.sampling_type]

    log_path = "/home/xudongh1/Project/adv_decorrelation/logs"
    log_name = "POS_ADV_{}3.log".format(SAMPLING_METHOD)
    LAMBDA_ub = -6
    LAMBDA_lb = -9
    LAMBDA_times = 15
    repeating_times = 5
    
    ## Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # dataset for the main task model
    we_train_dataset = POS(data_path, "we_train", private_label = "age")
    we_valid_dataset = POS(data_path, "we_valid", private_label = "age")
    we_test_dataset = POS(data_path, "we_test", private_label = "age")
    
    TP_SA_train_dataset = POS(data_path, 
                            "TP_SA_train", 
                            private_label = "age", 
                            sampling = True,
                            target_number = 13463,
                            sampling_method = SAMPLING_METHOD,
                            sampling_threshold = 0,
                            silence = True
                            )
    TP_SA_valid_dataset = POS(data_path, 
                            "TP_SA_valid", 
                            private_label = "age", 
                            sampling = True,
                            target_number = 2000,
                            sampling_method = SAMPLING_METHOD,
                            sampling_threshold = 0,
                            silence = True
                            )
    
    TP_POS_dataset = POS(data_path, "TP_POS", private_label = "age")
    
    # DataLoader Parameters
    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0}

    test_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0}
    
    we_train_iter = torch.utils.data.DataLoader(we_train_dataset, **train_params)
    we_valid_iter = torch.utils.data.DataLoader(we_valid_dataset, **test_params)
    we_test_iter = torch.utils.data.DataLoader(we_test_dataset, **test_params)
    TP_SA_train_iter = torch.utils.data.DataLoader(TP_SA_train_dataset, **train_params)
    TP_SA_valid_iter = torch.utils.data.DataLoader(TP_SA_valid_dataset, **train_params)
    TP_POS_iter = torch.utils.data.DataLoader(TP_POS_dataset, **{'batch_size': 600,'shuffle': False,'num_workers': 0})

    # Parameters
    INPUT_DIM = 19319
    TAG_OUTPUT_DIM = 12
    AGE_OUTPUT_DIM = 2
    PAD_IDX = 0
    TAG_PAD_IDX = -1

    # init logger
    log_file_path = Path(log_path) / log_name
    f = open(log_file_path, "a+")

    section = (LAMBDA_ub - LAMBDA_lb) / LAMBDA_times
    # iterate hyperparameters
    for tune_lambda in trange(LAMBDA_times):
        # current Lambda
        exponent_LAMBDA = (LAMBDA_lb + section*tune_lambda)
        LAMBDA = 10**exponent_LAMBDA
        f.write("#"*20+"\n")
        f.write("LAMBDA: {}\n".format(LAMBDA))
        f.write("#"*20+"\n")

        for model_repeating in trange(repeating_times):
            f.write("$"*20+"\n")
            f.write("Repeating time: {}\n".format(model_repeating))
            f.write("$"*20+"\n")

            # Create a Tagger instance
            model = BiLSTMPOSTagger(INPUT_DIM, 
                                    EMBEDDING_DIM, 
                                    HIDDEN_DIM, 
                                    TAG_OUTPUT_DIM, 
                                    N_LAYERS, 
                                    BIDIRECTIONAL, 
                                    DROPOUT, 
                                    PAD_IDX)
            # model.apply(init_weights)
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

            model = model.to(device)
            criterion = criterion.to(device)
            
            # Create a Gender Discriminator instance
            discriminator = Discriminator(input_size=HIDDEN_DIM*2,
                                            hidden_size=HIDDEN_DIM,
                                            num_classes=2)
            discriminator.to(device)
            adv_criterion = nn.CrossEntropyLoss().to(device)
            adv_optimizer = optim.Adam(discriminator.parameters(), lr=1e-1*LEARNING_RATE)
            
            best_valid_loss = float('inf')
            best_epoch = -1
                
            saved_model = model_path / "POS_adv_debiased_{}_{}.pt".format(SAMPLING_METHOD, exponent_LAMBDA)
            saved_adv_model = model_path / "POS_discriminator_{}_{}.pt".format(SAMPLING_METHOD, exponent_LAMBDA)
                
            for epoch in range(NUM_EPOCHS):
                train_loss, train_acc = train(model = model, 
                                            discriminator = discriminator,
                                            iterator = we_train_iter, 
                                            adv_iterator = TP_SA_train_iter,
                                            optimizer = optimizer, 
                                            criterion = criterion, 
                                            adv_criterion = adv_criterion, 
                                            tag_pad_idx = TAG_PAD_IDX,
                                            LAMBDA = LAMBDA,
                                            device = device
                                            )
                valid_loss, valid_acc = evaluate(model, we_valid_iter, criterion, TAG_PAD_IDX, device)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), saved_model)
                    # print("best_epoch:", best_epoch)
                else:
                    if best_epoch + 5 <=epoch:
                        break
                
                # train the discriminator
                best_adv_valid_loss = adv_eval_epoch(model, discriminator, TP_SA_valid_iter, adv_criterion, device)
                best_adv_epoch = -1
                for j in range(NUM_EPOCHS):
                    adv_train_loss = adv_train_epoch(model, discriminator, TP_SA_train_iter, adv_optimizer, adv_criterion, device)

                    adv_valid_loss = adv_eval_epoch(model, discriminator, TP_SA_valid_iter, adv_criterion, device)

                    if adv_valid_loss < best_adv_valid_loss:
                        best_adv_valid_loss = adv_valid_loss
                        best_adv_epoch = j
                        torch.save(discriminator.state_dict(), saved_adv_model)
                    else:
                        if best_adv_epoch + 5 <=j:
                            break
                discriminator.load_state_dict(torch.load(saved_adv_model))

            model.load_state_dict(torch.load(saved_model))
            
            Acc_overall, Acc_group_0, Acc_group_1, results = evaluate_bias(
                model = model, 
                iterator = TP_POS_iter, 
                tag_pad_idx = TAG_PAD_IDX, 
                return_value = True
                )

            g1_accuracy_score = accuracy_score(results["g1_label"], results["g1_pred"])
            g0_accuracy_score = accuracy_score(results["g0_label"], results["g0_pred"])
            g1_f1_score = f1_score(results["g1_label"], results["g1_pred"], average="macro")
            g0_f1_score = f1_score(results["g0_label"], results["g0_pred"], average="macro")

            overall_f1_score = f1_score(results["g0_label"]+results["g1_label"], results["g0_pred"]+results["g1_pred"], average="macro")
            overall_accuracy_score = accuracy_score(results["g0_label"]+results["g1_label"], results["g0_pred"]+results["g1_pred"])
            
            f.write("Accuracy Overall: {}\n".format(overall_accuracy_score))
            f.write("Accuracy O45: {}\n".format(g0_accuracy_score))
            f.write("Accuracy U35: {}\n".format(g1_accuracy_score))
            f.write("Accuracy GAP: {}\n".format(abs(g1_accuracy_score-g0_accuracy_score)))

            f.write("F1 Overall: {}\n".format(overall_f1_score))
            f.write("F1 O45: {}\n".format(g0_f1_score))
            f.write("F1 U35: {}\n".format(g1_f1_score))
            f.write("F1 GAP: {}\n".format(abs(g1_f1_score-g0_f1_score)))