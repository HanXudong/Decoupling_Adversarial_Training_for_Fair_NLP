import torch
import torch.optim as optim
import torch.nn as nn

# fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens

class RoBERTa_SentimentAnalysis(nn.Module):
    def __init__(self, args):
        super(RoBERTa_SentimentAnalysis,self).__init__()
        self.device = cuda = torch.device(args.cuda)
        self.model_type = 'roberta'
        self.pretrained_model_name = 'roberta-base'
        self.model_class = RobertaForSequenceClassification
        self.tokenizer_class = RobertaTokenizer 
        self.config_class = RobertaConfig
        print("Loading RoBERTa Preprocesser")
        self.transformer_tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_model_name, n_cpus = 1)
        self.transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = self.transformer_tokenizer, model_type = self.model_type)
        self.fastai_tokenizer = Tokenizer(tok_func = self.transformer_base_tokenizer, pre_rules=[], post_rules=[])
        self.transformer_vocab =  TransformersVocab(tokenizer = self.transformer_tokenizer)
        self.numericalize_processor = NumericalizeProcessor(vocab=self.transformer_vocab)
        self.tokenize_processor = TokenizeProcessor(tokenizer=self.fastai_tokenizer, include_bos=False, include_eos=False)
        print("Done")
        self.pad_idx = self.transformer_tokenizer.pad_token_id
        print("Initialising Configures")
        self.config = self.config_class.from_pretrained(self.pretrained_model_name)
        self.config.num_labels = args.num_labels
        self.config.use_bfloat16 = args.use_fp16
        print(self.config)
        print("\nLoading pretrained models")
        self.transformer = self.model_class.from_pretrained(self.pretrained_model_name, config = self.config)
        self.transformer.to(self.device)
        print("Done")
        self.roberta_layers = [self.transformer.roberta.embeddings, 
                                self.transformer.roberta.encoder.layer[0],
                                self.transformer.roberta.encoder.layer[1],
                                self.transformer.roberta.encoder.layer[2],
                                self.transformer.roberta.encoder.layer[3],
                                self.transformer.roberta.encoder.layer[4],
                                self.transformer.roberta.encoder.layer[5],
                                self.transformer.roberta.encoder.layer[6],
                                self.transformer.roberta.encoder.layer[7],
                                self.transformer.roberta.encoder.layer[8],
                                self.transformer.roberta.encoder.layer[9],
                                self.transformer.roberta.encoder.layer[10],
                                self.transformer.roberta.encoder.layer[11],
                                self.transformer.roberta.pooler]
        
    def forward(self, input_ids):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=self.pad_idx).type(input_ids.type())
        attention_mask.to(self.device)
        # print(attention_mask.is_cuda, input_ids.is_cuda)

        
        # logits = self.transformer(input_ids, attention_mask = attention_mask)[0]
        outputs = self.transformer.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.transformer.classifier(sequence_output)

        return logits

    def roberta_hidden(self, input_ids):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=self.pad_idx).type(input_ids.type())
        attention_mask.to(self.device)
        # print(attention_mask.is_cuda, input_ids.is_cuda)

        
        outputs = self.transformer.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        return sequence_output[:,0,:]

    def RoBerta_encoding(self, samples, pad_first:bool=False):
        "Function that collect samples and adds padding. Flips token order if needed"
        pad_idx = self.pad_idx
        samples = [self.numericalize_processor.process_one(self.tokenize_processor.process_one(i)) for i in samples]
        max_len = 512
        res = torch.zeros(len(samples), max_len).long() + pad_idx
        for i,s in enumerate(samples):
            if pad_first: res[i,-len(s):] = LongTensor(s)
            else:         res[i,:len(s):] = LongTensor(s)
        return res
    
    def predict(self, sentences):
        # roberta encoding
        sentences_encodings = self.RoBerta_encoding(sentences).to(self.device)
        return self.forward(sentences_encodings)

    def freeze_roberta_layers(self, number_of_layers):
        "number of layers: the first number of layers to be freezed"
        assert (number_of_layers < 14 and number_of_layers > -14), "beyound the total number of RoBERTa layer groups(14)."
        for target_layer in self.roberta_layers[:number_of_layers]:
                for param in target_layer.parameters():
                    param.requires_grad = False
        for target_layer in self.roberta_layers[number_of_layers:]:
                for param in target_layer.parameters():
                    param.requires_grad = True
        

    def trainable_parameter_counting(self):
        model_parameters = filter(lambda p: p.requires_grad, self.transformer.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

if __name__ == "__main__":
    class Args():
        use_fp16 = False
        cuda = "cuda"
    args = Args()
    testmodel = RoBERTa_SentimentAnalysis(args)
    print(testmodel.predict(["You are the apple of my eye."]))
    print("The number of trainable of parameters: {}".format(testmodel.trainable_parameter_counting()))
    print("freezing the first 10 layers of RoBERTa")
    testmodel.freeze_roberta_layers(10)
    print("The number of trainable of parameters: {}".format(testmodel.trainable_parameter_counting()))
