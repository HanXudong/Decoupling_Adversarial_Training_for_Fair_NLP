import os,argparse,time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed
import argparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from networks.deepmoji_sa import DeepMojiModel
from networks.discriminator import Discriminator

from networks.eval_metrices import leakage_evaluation
from networks.eval_metrices import group_evaluation

from networks.eval_metrices import group_evaluation
from networks.evaluator import eval

from dataloaders.hate_speech import HateSpeechDataset

from tqdm import tqdm, trange
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pathlib import Path

# train a discriminator 1 epoch
def adv_train_epoch(model, discriminator, adv_iterator, adv_optimizer, criterion, clipping_value, device, args):
    """"
    Train the discriminator to get a meaningul gradient
    """

    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    # model.eval()
    discriminator.train()


    # deactivate gradient reversal layer
    discriminator.GR = False
    
    for batch in adv_iterator:
        
        text = batch[0]
        tags = batch[1].long()
        # tags = batch[2].long() # Reverse
        p_tags = batch[2].long()
        # p_tags = batch[1]

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        adv_optimizer.zero_grad()
        
        hs = model.hidden(text)#.detach()

        adv_predictions = discriminator(hs)

        
        loss = criterion(adv_predictions, p_tags)
                        
        loss.backward()

        torch.nn.utils.clip_grad_norm(discriminator.parameters(), clipping_value)
        
        adv_optimizer.step()
        # print(loss.item())
        epoch_loss += loss.item()
        
    return epoch_loss / len(adv_iterator)

# evaluate the discriminator
def adv_eval_epoch(model, discriminator, adv_iterator, criterion, device, args):
    """"
    Train the discriminator to get a meaningul gradient
    """

    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    discriminator.eval()

    # deactivate gradient reversal layer
    discriminator.GR = False
    

    preds = []
    labels = []
    private_labels = []

    for batch in adv_iterator:
        
        text = batch[0]

        tags = batch[1]
        # tags = batch[2] #Reverse
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).long()
        
        # extract hidden state from the main model
        hs = model.hidden(text)
        # let discriminator make predictions
        adv_predictions = discriminator(hs)
        
        loss = criterion(adv_predictions, p_tags)
                        
        epoch_loss += loss.item()
        
        adv_predictions = adv_predictions.detach().cpu()
        tags = tags.cpu().numpy()

        preds += list(torch.argmax(adv_predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
        # private_labels += list(batch[1].cpu().numpy()) # Reverse

    
    return ((epoch_loss / len(adv_iterator)), preds, labels, private_labels)


# train the main modle with adv loss
def train_epoch(model, 
                discriminator, 
                iterator, 
                adv_iterator, 
                optimizer, 
                criterion, 
                clipping_value, 
                device, 
                args, 
                staring_adv = True):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    discriminator.train()

    # activate gradient reversal layer
    discriminator.GR = True
    
    for batch, p_batch in zip(iterator, adv_iterator):

        if not args.decor:
            text = p_batch[0]
            tags = p_batch[1].long()
        else:
            text = batch[0]
            tags = batch[1].long()
        
        p_text = p_batch[0]
        p_tags = p_batch[2].float()
        
        text = text.to(device)
        tags = tags.to(device)
        p_text = p_text.to(device)
        p_tags = p_tags.to(device)
        
        optimizer.zero_grad()
        # main model predictions
        predictions = model(text)
        # main tasks loss
        loss = criterion(predictions, tags)

        if staring_adv:
            # discriminator predictions
            p_tags = p_tags.long()

            hs = model.hidden(p_text)
            adv_predictions = discriminator(hs)
            
            loss = loss + criterion(adv_predictions, p_tags)
                        
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        
        optimizer.step()
        # print(loss.item())
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# to evaluate the main model
def eval(model, iterator, criterion, device, args):
    
    epoch_loss = 0
    
    model.eval()
    
    preds = []
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]

        tags = batch[1]
        # tags = batch[2] #Reverse
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).float()
        
        predictions = model(text)
        
        loss = criterion(predictions, tags)
                        
        epoch_loss += loss.item()
        
        predictions = predictions.detach().cpu()
        tags = tags.cpu().numpy()

        preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
        # private_labels += list(batch[1].cpu().numpy()) # Reverse

    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--gender_balanced', action='store_true')
    parser.add_argument('--cuda', type=str, default = "cuda")
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 400)
    parser.add_argument('--num_classes', type=int, default = 2)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--adv_level', type=int, default = -1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--LAMBDA', type=float, default=1)
    parser.add_argument('--n_discriminator', type=int, default = 1)
    parser.add_argument('--adv_units', type=int, default = 256)
    parser.add_argument('--DL', action='store_true')
    parser.add_argument('--diff_LAMBDA', type=float, default=1000)
    parser.add_argument('--data_path', type=str, default = "/path/path/path/Dataset/hate_speech")
    parser.add_argument('--model_path', type=str, default = "/path/path/path/adv_decorrelation/model/")
    parser.add_argument('--log_path', type=str, default = "/path/path/path/adv_decorrelation/logs/")
    parser.add_argument('--log_name', type=str, default = "hs_log")
    parser.add_argument('--main_model_name', type=str, default = "HS_decor_{}.pt")
    parser.add_argument('--adv_model_name', type=str, default = "HS_adv_decor_{}.pt")
    parser.add_argument('--batch_size', type=int, default = 512)
    parser.add_argument('--epoch', type=int, default = 100)
    parser.add_argument('--n_hidden', type=int, default = 1)
    parser.add_argument('--dropout', type=float, default = 0.5)
    # tuning hyperparameters
    parser.add_argument('--author_private_label', type=str, default="age")
    parser.add_argument('--subsampling_ratio', type=float, default=1.0)
    parser.add_argument('--decor', action='store_true')
    parser.add_argument('--LAMBDA_lb', type=float, default=0)
    parser.add_argument('--LAMBDA_ub', type=float, default=3)
    parser.add_argument('--LAMBDA_times', type=int, default=15)
    # repeating times
    parser.add_argument('--repeating_times', type=int, default=5)

    args = parser.parse_args()
    
    # DataLoader Parameters
    params = {'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 0}
    # Device
    device = torch.device("cuda")

    data_path = args.data_path
    model_path = Path(args.model_path)
    
    main_model_path = model_path / args.main_model_name.format(args.subsampling_ratio)
    adv_model_path = model_path / args.adv_model_name.format(args.subsampling_ratio)
    
    args.device = device

    # Load data
    train_data = HateSpeechDataset(args, 
                                data_path, 
                                "train", 
                                full_label_instances = True, 
                                upsampling = False,
                                private_label = args.author_private_label
                                )

    p_train_data = HateSpeechDataset(args, 
                                data_path, 
                                "train", 
                                full_label_instances = True, 
                                upsampling = False,
                                private_label = args.author_private_label,
                                subsampling = True,
                                subsampling_ratio = args.subsampling_ratio
                                )

    dev_data = HateSpeechDataset(args, 
                            data_path, 
                            "valid", 
                            full_label_instances = True, 
                            upsampling = False,
                            private_label = args.author_private_label
                            )

    p_dev_data = HateSpeechDataset(args, 
                            data_path, 
                            "valid", 
                            full_label_instances = True, 
                            upsampling = False,
                            private_label = args.author_private_label
                            )

    test_data = HateSpeechDataset(args, 
                            data_path, 
                            "test", 
                            full_label_instances = True, 
                            upsampling = False,
                            private_label = args.author_private_label
                            )

    
    p_test_data = HateSpeechDataset(args, 
                                data_path, 
                                "test", 
                                full_label_instances = True, 
                                upsampling = False,
                                private_label = args.author_private_label
                                )

    # init dataloader
    training_generator = torch.utils.data.DataLoader(train_data, **params)
    validation_generator = torch.utils.data.DataLoader(dev_data, **params)
    test_generator = torch.utils.data.DataLoader(test_data, **params)

    training_plabel_generator = torch.utils.data.DataLoader(p_train_data, **params)
    validation_plabel_generator = torch.utils.data.DataLoader(p_dev_data, **params)
    test_plabel_generator = torch.utils.data.DataLoader(p_test_data, **params)

    # init logger
    log_file_path = Path(args.log_path) / args.log_name
    f = open(log_file_path, "a+")

    section = (args.LAMBDA_ub - args.LAMBDA_lb) / args.LAMBDA_times
    # iterate hyperparameters
    for tune_lambda in trange(args.LAMBDA_times):
        # current Lambda
        args.LAMBDA = 10**(args.LAMBDA_lb + section*tune_lambda)
        f.write("#"*20+"\n")
        f.write("LAMBDA: {}\n".format(args.LAMBDA))
        f.write("#"*20+"\n")

        for model_repeating in trange(args.repeating_times):
            f.write("$"*20+"\n")
            f.write("Repeating time: {}\n".format(model_repeating))
            f.write("$"*20+"\n")

            # init model
            model = DeepMojiModel(args)
            model = model.to(device)

            # init adversary
            discriminator = Discriminator(args, args.hidden_size, 2)
            discriminator = discriminator.to(device)
            
            # optimizer
            LEARNING_RATE = args.lr

            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

            adv_optimizer = Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=1e-1*LEARNING_RATE)

            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)

            criterion = torch.nn.CrossEntropyLoss()

            
            best_loss = 1e+5
            best_adv_accuracy = 1
            best_epoch = args.epoch

            # save untrained model
            torch.save(discriminator.state_dict(), adv_model_path)
            torch.save(model.state_dict(), main_model_path)

            for i in trange(args.epoch):

                train_epoch(model, discriminator, training_generator, training_plabel_generator, optimizer, criterion, 1, device, args)

                valid_loss, valid_preds, valid_labels, _ = eval(model, validation_generator, criterion, device, args)
                valid_acc = accuracy_score(valid_preds, valid_labels)
                # learning rate scheduler
                scheduler.step(valid_loss)
                """
                Early stopping
                """
                if i >= 10: # and valid_loss < best_loss:
                    if best_loss > valid_loss:
                        best_acc = valid_acc
                        best_loss = valid_loss
                        best_epoch = i
                        torch.save(model.state_dict(), main_model_path)
                    else:
                        if best_epoch+5<=i:
                            break

                # Train discriminator until converged
                # evaluate discriminator 
                best_adv_loss, _, _, _ = adv_eval_epoch(model, discriminator, validation_plabel_generator, criterion, device, args)
                best_adv_epoch = -1
                for j in range(50):
                    adv_train_epoch(model, discriminator, training_plabel_generator, adv_optimizer, criterion, 1, device, args)
                    adv_valid_loss, _, _, _ = adv_eval_epoch(model, discriminator, validation_plabel_generator, criterion, device, args)
                    
                    if adv_valid_loss < best_adv_loss:
                            best_adv_loss = adv_valid_loss
                            best_adv_epoch = j
                            torch.save(discriminator.state_dict(), adv_model_path)
                    else:
                        if best_adv_epoch + 5 <= j:
                            break
                discriminator.load_state_dict(torch.load(adv_model_path))

            model.load_state_dict(torch.load(main_model_path))
            
            # Evaluation
            test_loss, preds, labels, p_labels = eval(model, test_generator, criterion, device, args)

            preds = np.array(preds)
            labels = np.array(labels)
            p_labels = np.array(p_labels)

            # Accuracy and GAP
            eval_results = group_evaluation(preds, labels, p_labels, silence=False)

            f.write("Accuracy 0: {}\n".format(eval_results["Accuracy_0"]))
            f.write("Accuracy 1: {}\n".format(eval_results["Accuracy_1"]))
            f.write("TPR 0: {}\n".format(eval_results["TPR_0"]))
            f.write("TPR 1: {}\n".format(eval_results["TPR_1"]))
            f.write("TNR 0: {}\n".format(eval_results["TNR_0"]))
            f.write("TNR 1: {}\n".format(eval_results["TNR_1"]))
            f.write("TPR gap: {}\n".format(eval_results["TPR_gap"]))
            f.write("TNR gap: {}\n".format(eval_results["TNR_gap"]))