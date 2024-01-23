from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from lingpy import Multiple, rc, prosodic_string
from typing import Union, Tuple, List, Optional, Dict, Any
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import TokenClassifierOutput, MaskedLMOutput
from transformers import Trainer, TrainingArguments
from src.charactertokenizer.charactertokenizer import CharacterTokenizer
from src.modelling_cogtran import *
from torch import nn
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
from lingpy.algorithm.clustering import flat_cluster
from lingpy.algorithm.extra import infomap_clustering
from lingpy.evaluate.acd import _get_bcubed_score as bcs
from scipy.special import softmax
import math
from sklearn.metrics import f1_score
import argparse



tqdm.pandas()
DELIM = '|'
MAX_POS_EMBED_PER_MSA = 64
MAX_POS_EMBED = 6
VOCAB_SIZE = 1024
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_ATTENTION_HEADS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 3
LR_RATE = 1e-3
INPUT_SCA = True

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


sca = rc('asjp')
def SCA(seqs: List[List[str]]) -> List[List[str]]:

    out = []
    for seq in seqs:
        sca_seq = []
        for char in seq:
            if char in sca.converter:
                sca_seq.append(sca.converter[char])
            else:
                sca_seq.append('~')
        out.append(sca_seq)
    return out

def Align(seqs: Union[List[List[str]], List[str]],
          trim: bool= False,
          input_sca: bool= False) -> List[List[str]]:
    
    mult = Multiple(seqs)
    mult.prog_align()
    mat = mult.alm_matrix
    if input_sca:
        mat = SCA(seqs= mat)
    return mat

def make_pairs(row):
    result = {'tokens': [], 'vocab': set()}
    n_per_msa = 1

    for i, tok_i in enumerate(zip(row['lang'],row['tokens'])):
        tokens_i = []
        for j, tok_j in enumerate(zip(row['lang'], row['tokens'])):
            lng_i, txt_i = tok_i
            lng_j, txt_j = tok_j
            seqs = Align([txt_i.split(DELIM), txt_j.split(DELIM)], input_sca= INPUT_SCA)
            seqs = [[lng_i] + seqs[0], [lng_j] + seqs[1]]
            tokens_i.append(seqs)
            result['vocab'] = result['vocab'].union(set(seqs[0]+seqs[1]))
        result['tokens'].append(tokens_i)
    result['vocab'] = list(result['vocab'])
    return result

def load_data_pair(path: str) -> Tuple[Union[Dataset, DatasetDict], List[str]]:
    
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
        df['lang'] = df.apply(lambda x: f"[{family};{x['lang']}]", axis=1)
        df = df.groupby(['concept'], as_index=False).agg(list)
        df['tokens'] = df.apply(lambda x: [DELIM.join(y.split()) for y in x['tokens']], axis=1)
        files_df.append(df)
    files_df = pd.concat(files_df)
    files_df['tokens_vocab'] = files_df.progress_apply(make_pairs, axis=1)
    files_df['tokens'] = files_df.apply(lambda x: x['tokens_vocab']['tokens'], axis=1)
    files_df['vocab'] = files_df.apply(lambda x: x['tokens_vocab']['vocab'], axis=1)
    vocab = files_df['vocab'].agg(lambda x: sum(x,[]))
    vocab = list(set(vocab))
    files_df.drop(columns=['lang','vocab','tokens_vocab'], inplace=True)

    return Dataset.from_pandas(files_df, preserve_index=False), vocab


def tokenize_pairwise(row, tokenizer, mode= 'test'):

    result = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}

    for pairs in row['tokens']:
        row_tokens =  {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        for pair in pairs:
            d = [tokenizer(DELIM.join(pair[i])) for i in range(2)]
            for key in row_tokens:
                row_tokens[key].append([d[0][key], d[1][key]])
        for key in row_tokens:
            result[key].append(row_tokens[key])
    num_seq = len(row['cogid'])
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
        new_result = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
        for key, val in result.items():
            for row in val:
                for ent in row:
                    new_result[key].append(ent)
        result = new_result
    elif mode == 'test':
        new_result = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
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
class DataCollatorForMSATPairwise(DataCollatorMixin):
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
        max_alignments = max(len(msa["input_ids"]) for msa in no_labels_features)
        max_seqlen = max(len(msa["input_ids"][0]) for msa in no_labels_features)
        
        batch = {}

        batch["input_ids"] = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen,
            ),
            dtype=torch.int64,
        )
        batch["input_ids"].fill_(self.tokenizer.pad_token_id)
        
        if "attention_mask" in no_labels_features[0]:
            batch["attention_mask"] = torch.empty(
                (
                    batch_size,
                    max_alignments,
                    max_seqlen,
                ),
                dtype=torch.int64,
            )
            batch["attention_mask"].fill_(0)
            
        if "token_type_ids" in no_labels_features[0]:
            batch["token_type_ids"] = torch.empty(
                (
                    batch_size,
                    max_alignments,
                    max_seqlen,
                ),
                dtype=torch.int64,
            )
            batch["token_type_ids"].fill_(self.tokenizer.pad_token_type_id)
        

        for i, msa in enumerate(no_labels_features):
            msa_seqlens = set(len(seq) for seq in msa["input_ids"])
            
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
        
            for feat in batch:
                batch[feat][i, : len(msa[feat]), : len(msa[feat][0])] = torch.Tensor(msa[feat])


        if labels is None:
            return batch

        sequence_length = max_alignments
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)


        batch[label_name] = labels
        

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch
 

          
class MSATForPairs(MSATPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size
        self.config = config
        self.args = MSATargs(config)
        self.alphabet = MSATalphabet(config)
        self.msat = MSATransformer(args=self.args,alphabet=self.alphabet)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps= config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size,2)
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, dict]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.msat(
            tokens= input_ids,
            repr_layers= list(range(self.config.num_hidden_layers+1)),
        )
        
        
        last_hidden = outputs["representations"][self.config.num_hidden_layers]*attention_mask.unsqueeze(-1) # B x R x C x D
        last_hidden = last_hidden.sum(dim=1)
        last_hidden = self.layer_norm(last_hidden.sum(dim=1))
        logits = self.classifier(last_hidden)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
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
    predictions = np.argmax(logits, axis=-1)
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
def evaluate(model, dataset, tokenizer, data_collator):
    result = {'BCPrec':[], 'BCRec': [], 'BCF': []}
    clustering = Cluster(method='upgma', threshold= 0.6)
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
        step_size = 8*BATCH_SIZE
        num_loops = math.ceil(len(y) / step_size)
        for i in range(num_loops):
            if (i+1)*step_size < len(y):
                inp = data_collator(y[i*step_size: (i+1)*step_size])
            else:
                inp = data_collator(y[i*step_size:])
                
            inp = {k:v.to(device) for k,v in inp.items()}
            with torch.no_grad():
                outputs = model(input_ids= inp['input_ids'], attention_mask= inp['attention_mask'])
                logits = outputs['logits']
                preds_batch = logits.detach().cpu().numpy()
            preds_batch = softmax(preds_batch, axis=-1)[:,1]
            preds.append(preds_batch)
        preds = np.concatenate(preds).reshape((seq_len,seq_len))
        labels = np.array(x['labels'])
        lang_family = tokenizer.convert_ids_to_tokens(x['input_ids'][0][0][1]).split(';')[0].replace('[','')
        
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
    
    dataset, vocab = load_data_pair(path='data/train/')
    
    if not args.prop0:
        dataset0, vocab0 = load_data_pair(path=f"data/{prefix_pth}split_{i}/train/")

        dataset = concatenate_datasets([dataset, dataset0])
        vocab = list(set(vocab).union(set(vocab0)))


    
    if not args.prop0:
        dataset1, vocab1 = load_data_pair(path=f"data/{prefix_pth}split_{i}/test/")
    else:
        dataset1, vocab1 = load_data_pair(path='data/test1/')
        dataset2, vocab2 = load_data_pair(path='data/test2/')
        dataset1 = concatenate_datasets([dataset1, dataset2])
        vocab1 = list(set(vocab1).union(set(vocab2)))

    tokenizer = CharacterTokenizer(vocab, MAX_POS_EMBED_PER_MSA, delim= DELIM)

    vocab_new = set(vocab).union(set(vocab1+['-','~']))
    tokenizer.add_tokens(list(vocab_new.difference(vocab)))

    dataset = dataset.train_test_split(test_size= 0.05)
    tokenized_dataset = dataset.map(lambda x: tokenize_pairwise(x, tokenizer=tokenizer, mode= 'train'), remove_columns=['tokens','concept','cogid'])

    tokenized_dataset['train'] = Dataset.from_pandas(tokenized_dataset['train'].to_pandas().explode(column=['input_ids','attention_mask','token_type_ids','labels'],ignore_index=True), preserve_index=False)
    tokenized_dataset['test'] = Dataset.from_pandas(tokenized_dataset['test'].to_pandas().explode(column=['input_ids','attention_mask','token_type_ids','labels'],ignore_index=True), preserve_index=False)



    test_dataset = dataset1.map(lambda x: tokenize_pairwise(x, tokenizer=tokenizer, mode= 'test'), remove_columns=['tokens','concept','cogid'])

    tokenizer.save_pretrained('models/MSAT-pairwise')
    tokenized_dataset.save_to_disk('models/MSAT-pairwise/tokenized_dataset')

    test_dataset.save_to_disk('models/MSAT-pairwise/test_dataset')
    
    #tokenizer = CharacterTokenizer.from_pretrained('models/MSAT-pairwise')
    #tokenized_dataset = load_from_disk('models/MSAT-pairwise/tokenized_dataset/')
    #test_dataset = load_from_disk('models/MSAT-pairwise/test_dataset/')

    data_collator = DataCollatorForMSATPairwise(tokenizer=tokenizer)

    config = MSATConfig(
                            vocab_size=VOCAB_SIZE,
                            mask_token_id=tokenizer.mask_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            cls_token_id=tokenizer.cls_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            hidden_size=EMBED_DIM,
                            num_hidden_layers=NUM_LAYERS,
                            num_attention_heads=NUM_ATTENTION_HEADS,
                            intermediate_size=HIDDEN_DIM,
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            max_position_embeddings=MAX_POS_EMBED,
                            max_position_embeddings_per_msa=MAX_POS_EMBED_PER_MSA,
                            layer_norm_eps=1e-12,
                        )


    model = MSATForPairs(config=config)




    training_args = TrainingArguments(
                    output_dir=f"models/cogtranpair_{file_prefix}/",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=LR_RATE*0.1,
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
    tokenizer.save_pretrained(f"models/cogtranpair_{file_prefix}")
    eval_dict = evaluate(model, test_dataset, tokenizer, data_collator)
    scores = [str(i)]
    for head in headers:
        scores.append(str(eval_dict[head]))
    lines.append('\t'.join(scores))

print('\n'.join(lines))
with open(f"results/results_crossval_cogtranpair_{file_prefix}.csv",'w') as fp:
    fp.write('\n'.join(lines))

