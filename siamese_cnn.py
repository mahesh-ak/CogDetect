from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from typing import Union, Tuple, List, Optional, Dict, Any
from lingpy import rc
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.functional import F
import os
import argparse
from lingpy.algorithm.clustering import flat_cluster
from lingpy.algorithm.extra import infomap_clustering
from lingpy.evaluate.acd import _get_bcubed_score as bcs
from sklearn.metrics import f1_score
from scipy.special import expit
import math
from src.charactertokenizer.charactertokenizer import CharacterTokenizer



DELIM = '|'
MAX_WORD_LENGTH = 10
MAX_LENGTH = 100
VOCAB_SIZE = 64
LANG_VOCAB_SIZE = 512
BATCH_SIZE = 512
NUM_EPOCHS = 3
LR_RATE = 1e-3

prefix = ""
num_runs = 5

parser = argparse.ArgumentParser()
parser.add_argument("--prop50", help="run on train proportion 0.5",
                    action="store_true")
parser.add_argument("--prop0", help="run on train proportion 0",
                    action="store_true")

args = parser.parse_args()
if args.prop50:
    prefix = "prop_50"
if args.prop0:
    prefix = "prop_0"
    num_runs = 1

if args.prop50 and args.prop0:
    print("Conflicting arguments: --prop0 and --prop50")
    quit()
    
device = torch.device('cuda:0')
asjp = rc('asjp')
def ASJP(seqs: List[List[str]]) -> List[List[str]]:
    out = []
    for seq in seqs:
        asjp_seq = []
        for char in seq:
            if char in asjp.converter:
                asjp_seq.append(asjp.converter[char])
            else:
                chars = list(char)
                for c in chars:
                    if c in asjp.converter:
                        asjp_seq.append(asjp.converter[char])
        out.append(asjp_seq)

    return out


def load_data_CNN(path: str) -> Tuple[Union[Dataset, DatasetDict], List[str]]:
    
    print(f"Loading training data from '{path}' ...")
    files_list = os.listdir(path)
    files_df = []
    vocab = []
    for file in tqdm(files_list):
        f_pth= os.path.join(path, file)
        file_name = file.replace('_train','').replace('_test','')
        family = file_name.split('_')[-1].split('-')[0]
        df = pd.read_csv(f_pth,sep='\t')
        df = df[['DOCULECT','TOKENS','COGID','CONCEPT']]
        df.dropna(inplace=True)
        df = df.astype({'COGID':'int32'})
        df.rename(columns={'DOCULECT': 'lang', 'TOKENS':'tokens', 'COGID':'cogid', 'CONCEPT': 'concept'}, inplace=True)
        df['langs'] = df.apply(lambda x: f"[{family};{x['lang']}]", axis=1)
        df = df.groupby(['concept'], as_index=False).agg(list)
        df['tokens'] = df.apply(lambda x: ASJP(x['tokens']), axis=1)
        df['tokens'] = df.apply(lambda x: [DELIM.join(y) for y in x['tokens']], axis=1)
        df.drop(columns=['lang'],inplace=True)
        files_df.append(df)
    files_df = pd.concat(files_df)
    vocab = files_df.apply(lambda x: sum([algn.split(DELIM) for algn in x['tokens']],[]), axis=1).agg(lambda x: sum(x,[]))
    vocab = list(set(vocab))
    lang_vocab = files_df.apply(lambda x: x['langs'], axis=1).agg(lambda x: sum(x,[]))
    lang_vocab = list(set(lang_vocab))
    return Dataset.from_pandas(files_df, preserve_index=False), vocab, lang_vocab


def tokenize_pairwise(row, tokenizer, lang_tokenizer, mode= 'test'):

    result = {'input_ids': [], 'labels': [], 'langs': []}
    

    for i, tok_i in enumerate(row['tokens']):
        tokens_i = {'input_ids': [], 'langs':[]}
        tokenize_i = tokenizer(tok_i)['input_ids'][1:-1]
        lng_i = lang_tokenizer(row['langs'][i])['input_ids'][1:-1]
        
        for j, tok_j in enumerate(row['tokens']):
            
            tokenize_j = tokenizer(tok_j)['input_ids'][1:-1]
            lng_j = lang_tokenizer(row['langs'][j])['input_ids'][1:-1]
            
            tokens_i['input_ids'].append([tokenize_i,tokenize_j])
            tokens_i['langs'].append([lng_i,lng_j])
        
        for key in result:
            if key == 'labels':
                continue
            result[key].append(tokens_i[key])
    
            
    num_seq = len(row['tokens'])
    result['labels'] = []
    for i in range(num_seq):
        label_i = []
        for j in range(num_seq):
            if row['cogid'][i] == row['cogid'][j]:
                label_i.append(1)
            else:
                label_i.append(0)
        result['labels'].append(label_i)
    
    if mode == 'train':
        new_result = {'input_ids': [], 'labels': [], 'langs': []}
        for key, val in result.items():
            for row in val:
                for ent in row:
                    new_result[key].append(ent)
        result = new_result
    elif mode == 'test':
        new_result = {'input_ids': [], 'langs': []}
        for key in new_result:
            val = result[key]
            for row in val:
                for ent in row:
                    new_result[key].append(ent)
        new_result['labels'] = result['labels']
        result = new_result
    
    return result


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

@dataclass
class DataCollatorForCharCNN(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"


    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        
        batch_size = len(no_labels_features)

        
        batch = {}

        batch["input_ids"] = torch.empty(
            (
                batch_size,
                2,
                MAX_WORD_LENGTH,
            ),
            dtype=torch.int64,
        )
        batch["input_ids"].fill_(self.tokenizer.pad_token_id)

        for i, msa in enumerate(no_labels_features):
            for j in range(2):
                lim = min(MAX_WORD_LENGTH, len(msa['input_ids'][j]))
                batch['input_ids'][i, j,:lim] = torch.Tensor(msa['input_ids'][j][:lim])

        
        batch['langs'] = torch.tensor([x['langs'] for x in no_labels_features], dtype= torch.int64)
        if labels is None:
            return batch

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)


        batch[label_name] = labels
        

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch



class SiameseCNNConfig(PretrainedConfig):

    model_type = "siamesecnn"

    def __init__(
        self,
        pad_token_id=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, mask_token_id=None, **kwargs)

        self.initializer_range = initializer_range


class CharCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, (VOCAB_SIZE, 2), padding= 'same')
        self.conv2 = nn.Conv2d(10, 10, (2, 3))
        self.max_pool = nn.MaxPool2d((1,2))
    
    def forward(
        self,
        inp: torch.Tensor
    ) -> torch.Tensor:
        #print('cnn in',inp.size())
        x = self.conv1(inp)
        #print('conv1 out',x.size())
        x = self.conv2(x)
        #print('conv2 out',x.size())
        x = self.max_pool(x)
        #print('max_pool out', x.size())
        
        return x.flatten(start_dim=1)

class SiameseCNN(PreTrainedModel):
    
    config_class = SiameseCNNConfig
    base_model_prefix = "siameseCNN"
    _no_split_modules = []
    
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        
        self.charcnn = CharCNN()
        self.fc = nn.Linear(3032,128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128,1)
        
        self.init_weights()
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        langs: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, dict]:

        return_dict = return_dict if return_dict is not None else True
        
        mask = torch.where(input_ids == self.pad_token_id, 0, 1)
                           
        inp1 = nn.functional.one_hot(input_ids[:,0,:], num_classes= VOCAB_SIZE)
        inp2 = nn.functional.one_hot(input_ids[:,1,:], num_classes= VOCAB_SIZE)
        
        inp1 = inp1*mask[:,0,:].unsqueeze(-1).type(torch.Tensor).to(device)
        inp2 = inp2*mask[:,1,:].unsqueeze(-1).type(torch.Tensor).to(device)
        
        emb1 = self.charcnn(inp1.unsqueeze(1).permute(0,1,3,2))
        emb2 = self.charcnn(inp2.unsqueeze(1).permute(0,1,3,2))
        #print('cnn out', emb1.size())
        
        batch_size = langs.size(0)
        lngs = nn.functional.one_hot(langs, num_classes= LANG_VOCAB_SIZE).sum(dim=1).squeeze(1)
        
        emb = torch.abs(emb1-emb2)
        emb = torch.cat([emb, lngs], dim=-1)
        #print('concat out', emb.size())
        out = self.fc(emb)
        out = F.relu(out)
        out = self.dropout(out)
        logits = self.classifier(out).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            labels = labels.type_as(logits)
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss= loss,
            logits= logits,
            hidden_states= None,
            attentions= None,
        )

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = expit(logits)
    predictions = (logits > 0.5).astype(np.int)
    result = {}
    result['F1'] = f1_score(labels,predictions,average='macro')
    result['Acc'] = f1_score(labels,predictions,average='micro')
    for key, val in result.items():
        result[key] = round(val,3)
    return result



def concat_lists(ll):
    out = []
    for l in ll:
        out += l.tolist()
    return out


class Cluster:
    
    def __init__(self, method= 'upgma', threshold= 0.5):
        self.method = method
        self.threshold = threshold
        self.labels_ = None
        self.n_clusters_ = None
    
    def fit(self, sim_matrix):
        if self.method == 'infomap':
            clusters = infomap_clustering(self.threshold, 1-sim_matrix)
        else:
            clusters = flat_cluster(self.method, self.threshold, 1-sim_matrix)
        
        self.labels_ = np.zeros(sim_matrix.shape[0], dtype=int)

        for cluster, inds in clusters.items():
            if self.method == 'infomap':
                indxs = (np.array(inds)-1).tolist()
                self.labels_[indxs] = cluster-1
            else:
                self.labels_[inds] = cluster
        self.n_clusters_ = self.labels_.max()
        
device = torch.device('cuda:0')
def evaluate(model, dataset, lang_tokenizer, data_collator):
    result = {'BCPrec':[], 'BCRec': [], 'BCF': []}
    clustering = Cluster(method='upgma', threshold= 0.5)
    label_clustering =  Cluster(method='upgma', threshold= 0.5)
    
    overall_preds = {}
    p_idx = 0
    overall_labels = {}
    l_idx = 0
    for i, x in tqdm(enumerate(dataset)):
        seq_len = len(x['labels'])
        if seq_len == 1:
            continue
        y = []
        for i in range(len(x['input_ids'])):
            y.append({k:x[k][i] for k in x if k!='labels'})
        preds = []
        step_size = BATCH_SIZE
        num_loops = math.ceil(len(y) / step_size)
        for i in range(num_loops):
            if (i+1)*step_size < len(y):
                inp = data_collator(y[i*step_size: (i+1)*step_size])
            else:
                inp = data_collator(y[i*step_size:])
                
            inp = {k:v.to(device) for k,v in inp.items()}
            with torch.no_grad():
                outputs = model(input_ids= inp['input_ids'], langs= inp['langs'])
                logits = outputs['logits']
                preds_batch = logits.detach().cpu().numpy()
            preds_batch = expit(preds_batch)
            preds.append(preds_batch)
        preds = np.concatenate(preds).reshape((seq_len,seq_len))
        labels = np.array(x['labels'])
        lang_family = lang_tokenizer.convert_ids_to_tokens(x['langs'][0][0][0]).split(';')[0].replace('[','')
        
        if lang_family not in overall_preds:
            overall_preds[lang_family] = []
            overall_labels[lang_family] = []
        
        clustering.fit(preds[:seq_len,:seq_len])
        p_clusters = p_idx + clustering.labels_
        overall_preds[lang_family] += p_clusters.tolist()
        p_idx += (clustering.n_clusters_ + 1)
        
        label_clustering.fit(labels[:seq_len,:seq_len])
        l_clusters = l_idx + label_clustering.labels_
        overall_labels[lang_family] += l_clusters.tolist()
        l_idx += (label_clustering.n_clusters_ + 1)
        
    for lang_family in overall_labels:    
        p, r = bcs(overall_preds[lang_family], overall_labels[lang_family]), bcs(overall_labels[lang_family], overall_preds[lang_family])
        if p + r != 0:
            f = 2 * (p * r) / (p + r)
        else:
            f = 0
        
        result[lang_family] = round(f,3)
        result['BCPrec'].append(p)
        result['BCRec'].append(r)
        result['BCF'].append(f)
    
    for key, val in result.items():
        result[key] = round(np.mean(np.array(val)),3)
        
    return result
 
headers = ['BCPrec', 'BCRec', 'BCF', 'SinoTibetan', 'Tujia', 'AustroAsiatic', 'Huon', 'IndoEuropean',\
            'Chinese', 'Uralic', 'PamaNyungan', 'Austronesian', 'Romance', 'Bahnaric']
header_txt = '\t'.join(headers)
lines = [f"SplitID\t{header_txt}"]


    
prefix_pth = prefix
if prefix != "":
    prefix_pth = prefix + '/'
file_prefix = prefix

if prefix != "":
    file_prefix = '_' + prefix

for i in range(num_runs):
    print("Running split #",i)
    
    dataset, vocab, lang_vocab = load_data_CNN(path='data/train/')
    
    if not args.prop0:
        dataset0, vocab0, lang_vocab0 = load_data_CNN(path=f"data/{prefix_pth}split_{i}/train/")

        dataset = concatenate_datasets([dataset, dataset0])
        vocab = list(set(vocab).union(set(vocab0)))
        lang_vocab = list(set(lang_vocab).union(set(lang_vocab0)))
    
    if not args.prop0:
        dataset1, vocab1, lang_vocab1 = load_data_CNN(path=f"data/{prefix_pth}split_{i}/test/")
    else:
        dataset1, vocab1, lang_vocab1 = load_data_CNN(path='data/test1/')
        dataset2, vocab2, lang_vocab2 = load_data_CNN(path='data/test2/')
        dataset1 = concatenate_datasets([dataset1, dataset2])
        vocab1 = list(set(vocab1).union(set(vocab2)))
        lang_vocab1 = list(set(lang_vocab1).union(set(lang_vocab2)))


    tokenizer = CharacterTokenizer(vocab, MAX_LENGTH, delim= DELIM)
    lang_tokenizer = CharacterTokenizer(lang_vocab, 1, delim= DELIM)

    vocab_new = set(vocab).union(set(vocab1+['-']))
    lang_vocab_new = set(lang_vocab).union(set(lang_vocab1))
    tokenizer.add_tokens(list(vocab_new.difference(vocab)))
    lang_tokenizer.add_tokens(list(lang_vocab_new.difference(lang_vocab)))

    dataset = dataset.train_test_split(test_size= 0.05)
    tokenized_dataset = dataset.map(lambda x: tokenize_pairwise(x, tokenizer=tokenizer, lang_tokenizer= lang_tokenizer, mode= 'train'), remove_columns=['tokens','concept','cogid'])
    tokenized_dataset['train'] = Dataset.from_pandas(tokenized_dataset['train'].to_pandas().explode(column=['input_ids','langs','labels'],ignore_index=True), preserve_index=False)
    tokenized_dataset['test'] = Dataset.from_pandas(tokenized_dataset['test'].to_pandas().explode(column=['input_ids','langs','labels'],ignore_index=True), preserve_index=False)
    val_dataset = dataset['test'].map(lambda x: tokenize_pairwise(x, tokenizer=tokenizer, lang_tokenizer= lang_tokenizer, mode= 'test'), remove_columns=['tokens','concept','cogid'])


    test_dataset = dataset1.map(lambda x: tokenize_pairwise(x, tokenizer=tokenizer, lang_tokenizer=lang_tokenizer, mode= 'test'), remove_columns=['tokens','concept','cogid'])


    data_collator = DataCollatorForCharCNN(tokenizer=tokenizer)


    config = SiameseCNNConfig(pad_token_id= tokenizer.pad_token_id)
    model = SiameseCNN(config=config)


    training_args = TrainingArguments(
                    output_dir=f"models/CharCNN_{file_prefix}/",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=LR_RATE,
                    per_device_train_batch_size=BATCH_SIZE,
                    per_device_eval_batch_size=2*BATCH_SIZE,
                    weight_decay=0,
                    save_total_limit=1,
                    num_train_epochs=NUM_EPOCHS,
                #   fp16=True,
                )


    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset['train'],
                    eval_dataset=tokenized_dataset['test'],
                    data_collator=data_collator,
                    compute_metrics= compute_metrics,
                )
    trainer.train()
    if not os.path.exists(f"models/CharCNN_{file_prefix}/tokenizer"):
        os.mkdir(f"models/CharCNN_{file_prefix}/tokenizer")
        os.mkdir(f"models/CharCNN_{file_prefix}/lang_tokenizer")
    tokenizer.save_pretrained(f"models/CharCNN_{file_prefix}/tokenizer")
    lang_tokenizer.save_pretrained(f"models/CharCNN_{file_prefix}/lang_tokenizer")
    eval_dict = evaluate(model, test_dataset, lang_tokenizer, data_collator)
    scores = [str(i)]
    for head in headers:
        scores.append(str(eval_dict[head]))
    lines.append('\t'.join(scores))

print('\n'.join(lines))
with open(f"results/results_crossval_CharCNN_{file_prefix}.csv",'w') as fp:
    fp.write('\n'.join(lines))

