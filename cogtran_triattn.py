from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from lingpy import Multiple, rc
from lingpy.evaluate.acd import _get_bcubed_score as bcs
from lingpy.algorithm.clustering import flat_cluster
from scipy.special import softmax
from typing import Union, Tuple, List, Optional, Dict, Any
from transformers import PreTrainedTokenizerBase, DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import Trainer, TrainingArguments
from src.charactertokenizer.charactertokenizer import CharacterTokenizer
from src.modelling_cogtran import *
from alphafold2.alphafold2_pytorch.alphafold2 import PairwiseAttentionBlock
from einops import rearrange
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
import os
import argparse

DELIM = '|'
MAX_POS_EMBED_PER_MSA = 256
MAX_POS_EMBED = 256
VOCAB_SIZE = 2500
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_ATTENTION_HEADS = 2
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR_RATE = 1e-3
INPUT_SCA = True

if INPUT_SCA: 
    VOCAB_SIZE = 768

prefix = ""
num_runs = 5

parser = argparse.ArgumentParser()
parser.add_argument("--prop50", help="run on train proportion 0.5",
                    action="store_true")
parser.add_argument("--prop0", help="run on train proportion 0",
                    action="store_true")
parser.add_argument("--trioff", help="switch off pairwise module",
                    action="store_true")
parser.add_argument("--addlayers", help="test with num of layers = 2",
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

if args.addlayers:
    NUM_LAYERS += 2
    if not args.trioff:
        print("Warning: number of layers doubled for both MSA and Pairwise modules, use --trioff to turn off Pairwise module to avoid CUDA out of memory")
    
    
sca = rc('asjp')


def SCA(seqs: List[List[str]]) -> List[List[str]]:

    out = []
    for seq in seqs:
        sca_seq = []
        for char in seq:
            if char in sca.converter:
                sca_seq.append(sca.converter[char])
            elif len(char) > 1:
                com = ''
                for c in list(char):
                    if c in sca.converter:
                        com += sca.converter[c]
                if com != '':
                    sca_seq.append(com)
                else:
                    sca_seq.append('~')
            else:
                sca_seq.append('~')
        out.append(sca_seq)

    return out

def Align(seqs: Union[List[List[str]], List[str]],
          input_sca: bool= INPUT_SCA) -> List[List[str]]:
    
    mult = Multiple(seqs)
    mult.prog_align()
    mat = mult.alm_matrix
    if input_sca:
        mat = SCA(seqs= mat)
    return mat


def load_data(path: str, 
              aligned: bool= False) -> Tuple[Union[Dataset, DatasetDict], List[str]]:
    
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
        df.rename(columns={'DOCULECT': 'langs', 'TOKENS':'tokens', 'COGID':'cogid', 'CONCEPT': 'concept'}, inplace=True)
        df['old_tokens'] = df['tokens']
        df['tokens'] = df.apply(lambda x: x['tokens'].replace(" ",""), axis=1)
        df = df.groupby(['concept'], as_index=False).agg(list)
        if aligned:
            df['tokens'] = df.apply(lambda x: Align(x['tokens']), axis=1)
        df['tokens'] = df.apply(lambda x: [[f"[{family};{lng}]"]+algn for lng, algn in zip(x['langs'], x['tokens'])], axis=1)
        df['tokens'] = df.apply(lambda x: [DELIM.join(y) for y in x['tokens']], axis=1)
        files_df.append(df)
    files_df = pd.concat(files_df)
    vocab = files_df.apply(lambda x: sum([algn.split(DELIM) for algn in x['tokens']],[]), axis=1).agg(lambda x: sum(x,[]))
    vocab = list(set(vocab))
    return Dataset.from_pandas(files_df, preserve_index=False), vocab



def tokenize(row, tokenizer, return_tensors=None):

    result = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
    

    for algn in row['tokens']:
        tokenize_lng = tokenizer(algn, return_tensors=return_tensors)
        for key in tokenize_lng:
            result[key].append(tokenize_lng[key])
    
    num_msa = len(row['cogid'])
    result['labels'] = []
    for i in range(num_msa):
        label_i = []
        for j in range(num_msa):
            if row['cogid'][i] == row['cogid'][j]:
                label_i.append(1)
            else:
                label_i.append(0)
        result['labels'].append(label_i)
        
    if return_tensors == 'pt':
        for i in range(num_msa):
            result['labels'][i] = torch.Tensor(result['labels'][i])
        for key in result:
            result[key] = torch.stack(result[key])
    
    return result


class DataCollatorForMSAContactPred(DataCollatorForTokenClassification):
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


        batch[label_name] = [
            [
                to_list(label_row) + [self.label_pad_token_id] * (sequence_length - len(label_row)) for label_row in label
            ] + [[self.label_pad_token_id]*sequence_length]*(sequence_length - len(label)) for label in labels
        ]
        

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch


class OuterMean(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)

        if hidden_dim is None:
            hidden_dim = dim

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')

        if mask is not None:
            # masked mean, if there are padding in the rows of the MSA
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask.to(torch.bool), 0.)
            outer = outer.sum(dim = 1) / (mask.sum(dim = 1) + self.eps)
        else:
            outer = outer.mean(dim = 1)

        return self.proj_out(outer)
    
class PairwiseModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.PairwiseAttnBlock = PairwiseAttentionBlock(dim = config.hidden_size,
                                                                 seq_len = config.max_position_embeddings, 
                                                                 heads = config.num_attention_heads,
                                                                 dim_head = config.intermediate_size, 
                                                                 dropout = config.attention_probs_dropout_prob,
                                                                 global_column_attn = True)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps= config.layer_norm_eps)
        
        
    def forward(self, inputs, mask):
        res = self.PairwiseAttnBlock(inputs,mask=mask)
        x = self.layer_norm(res)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        output = x + res
        
        return output
          
class MSATForEdgePred(MSATPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size
        self.config = config
        self.args = MSATargs(config)
        self.alphabet = MSATalphabet(config)
        self.msat = MSATransformer(args=self.args,alphabet=self.alphabet)
    
        self.outer_mean = OuterMean(config.hidden_size, hidden_dim= config.intermediate_size,eps=config.layer_norm_eps)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        if not config.no_pairwise_module:
            self.tri_layers = nn.ModuleList([ PairwiseModule(config=config) for _ in range(config.num_hidden_layers)])
            self.layer_norm_2 = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 2)
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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

        need_head_weights = False
        if output_attentions:
            need_head_weights = True
        out_attns = None
            
        outputs = self.msat(
            tokens= input_ids,
            repr_layers= list(range(self.config.num_hidden_layers+1)),
            need_head_weights = need_head_weights,
            return_contacts = False,
        )

        last_hidden = outputs["representations"][self.config.num_hidden_layers] # B x S*R x C X D
        if output_attentions:
            out_attns = outputs["col_attentions"]
        
        last_hidden = last_hidden.permute(0,2,1,3) # B x C X R X D
        attention_mask = attention_mask.permute(0,2,1) # B x C x R
        
        mask = attention_mask[:,0,:]
        mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
        mask = mask.to(torch.bool)
        
        tri_output = self.outer_mean(last_hidden, mask= attention_mask)
        tri_output = self.layer_norm_1(tri_output)
        
        if not self.config.no_pairwise_module:
            for tri_layer in self.tri_layers:
                tri_output = tri_layer(tri_output, mask=mask)
        
            tri_output = self.layer_norm_2(tri_output)
        
        logits = self.classifier(tri_output)
        
        loss = None
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.permute(0,3,1,2), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss= loss,
            logits= logits,
            hidden_states= None,
            attentions= out_attns,
        )


class Cluster:
    
    def __init__(self, method= 'upgma', threshold= 0.55):
        self.method = method
        self.threshold = threshold
        self.labels_ = None
        self.n_clusters_ = None
    
    def fit(self, sim_matrix):
        clusters = flat_cluster(self.method, self.threshold, 1-sim_matrix)
        self.labels_ = np.zeros(sim_matrix.shape[0], dtype=int)

        for cluster, inds in clusters.items():
            self.labels_[inds] = cluster
        self.n_clusters_ = self.labels_.max()

def compute_metrics_BC(eval_preds, dataset= None, save_to_file= None):
    preds, labels = eval_preds
    result = {'BCPrec':[], 'BCRec': [], 'BCF': []}
    preds = softmax(preds, axis=-1)[:,:,:,1]
        
    clustering = Cluster(method='upgma', threshold= 0.6)
    label_clustering =  Cluster(method='upgma', threshold= 0.5)
    
    
    overall_preds = {}
    p_idx = 0
    overall_labels = {}
    l_idx = 0
    lines = {}
    for i, pr in tqdm(enumerate(zip(preds,labels))):
        p, l = pr
        lang_family = dataset[i]['tokens'][0].split(DELIM)[0].split(';')[0].replace('[','')

        base =  l.tolist()[0]
        if -100 in base:
            seq_len = base.index(-100)
        else:
            seq_len = len(base)
        if seq_len == 1:
            continue
        
        if lang_family not in overall_preds:
            overall_preds[lang_family] = []
            overall_labels[lang_family] = []
            lines[lang_family] = ["DOCULECT\tCONCEPT\tTOKENS\tCOGID\tPRED"]
        
 
        clustering.fit(p[:seq_len,:seq_len])
        p_clusters = p_idx + clustering.labels_
        overall_preds[lang_family] += p_clusters.tolist()
        p_idx += (clustering.n_clusters_ + 1)
        
        label_clustering.fit(l[:seq_len,:seq_len])
        l_clusters = l_idx + label_clustering.labels_
        overall_labels[lang_family] += l_clusters.tolist()
        l_idx += (label_clustering.n_clusters_ + 1)
        
        for j in range(p_clusters.shape[0]):
            lines[lang_family].append(f"{dataset[i]['langs'][j]}\t{dataset[i]['concept']}\t{dataset[i]['old_tokens'][j]}\t{l_clusters[j]}\t{p_clusters[j]}")
        
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
        if save_to_file:
            with open(os.path.join(save_to_file,f"{lang_family}.tsv"), 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(lines[lang_family]))
    
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

if args.trioff:
    file_prefix = file_prefix + '_no_triattn'

if args.addlayers:
    file_prefix = file_prefix + '_layersx2'

for i in range(num_runs):
    print("Running split #",i)
    dataset, vocab = load_data(path='data/train/', aligned= True)
    
    if not args.prop0:
        dataset0, vocab0 = load_data(path=f"data/{prefix_pth}split_{i}/train/", aligned= True)

        dataset = concatenate_datasets([dataset, dataset0])
        vocab = list(set(vocab).union(set(vocab0)))
    
    if not args.prop0:
        dataset1, vocab1 = load_data(path=f"data/{prefix_pth}split_{i}/test/", aligned= True)
    else:
        dataset1, vocab1 = load_data(path='data/test1/', aligned= True)
        dataset2, vocab2 = load_data(path='data/test2/', aligned= True)
        dataset1 = concatenate_datasets([dataset1, dataset2])
        vocab1 = list(set(vocab1).union(set(vocab2)))



    tokenizer = CharacterTokenizer(vocab, MAX_POS_EMBED_PER_MSA, delim= DELIM)


    vocab_new = set(vocab).union(set(vocab1))#.union(set(vocab2))
    tokenizer.add_tokens(list(vocab_new.difference(vocab)))

    dataset = dataset.train_test_split(test_size=0.05)
    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer), remove_columns=['tokens','concept','cogid', 'langs', 'old_tokens'])


    tokenized_test = dataset1.map(lambda x: tokenize(x, tokenizer=tokenizer), remove_columns=['tokens','concept','cogid', 'langs', 'old_tokens'])


    data_collator = DataCollatorForMSAContactPred(tokenizer=tokenizer)

    test1 = Dataset.from_dict({k: v.tolist() for k,v in data_collator([x for x in tokenized_test]).items()})

    test = Dataset.from_dict({k: v.tolist() for k,v in data_collator([x for x in tokenized_dataset['test']]).items()})



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
                            no_pairwise_module= args.trioff,
                        )



    model = MSATForEdgePred(config=config)
    print("num of parameters: {:.1f}M".format(model.num_parameters()/1000000))

    training_args = TrainingArguments(
                    output_dir=f"models/cogtran_{file_prefix}",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=LR_RATE,
                    per_device_train_batch_size=BATCH_SIZE,
                    per_device_eval_batch_size=BATCH_SIZE//2,
                    weight_decay=0,
                    save_total_limit=1,
                    num_train_epochs=NUM_EPOCHS,
                    fp16=False,
                )


    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset['train'],
                    eval_dataset=test,
                    data_collator=data_collator,
                    compute_metrics= lambda x: compute_metrics_BC(x, dataset = dataset['test']),
                )


    trainer.train()
    tokenizer.save_pretrained(f"models/cogtran_{file_prefix}/")
    save_to_file = f"results/cogtran_{file_prefix}/split{i}"
    os.makedirs(save_to_file,exist_ok= True)
    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset['train'],
                    eval_dataset=test1,
                    data_collator=data_collator,
                    compute_metrics= lambda x: compute_metrics_BC(x, dataset = dataset1, save_to_file= save_to_file),
                )


    eval_dict = trainer.evaluate()
    scores = [str(i)]
    for head in headers:
        scores.append(str(eval_dict[f"eval_{head}"]))
    lines.append('\t'.join(scores))





print('\n'.join(lines))
with open(f"results/results_crossval_cogtran_{file_prefix}.csv",'w') as fp:
    fp.write('\n'.join(lines))


