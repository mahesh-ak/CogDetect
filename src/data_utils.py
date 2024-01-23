#!/usr/bin/env python
# coding: utf-8

from datasets import Dataset, DatasetDict
from lingpy import Multiple
from typing import Union, Tuple, List, Optional
from transformers import PreTrainedTokenizerBase, DataCollatorForTokenClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os

DELIM = '|'
MAX_POS_EMBED_PER_MSA = 128
MAX_POS_EMBED = 128
VOCAB_SIZE = 2500



def align(seqs: Union[List[List[str]], List[str]],
          trim: bool= False) -> List[List[str]]:
    
    mult = Multiple(seqs)
    mult.prog_align()
    return mult.alm_matrix


def load_data(path: str, 
              aligned: bool= False) -> Tuple[Union[Dataset, DatasetDict], List[str]]:
    
    print(f"Loading training data from '{path}' ...")
    files_list = os.listdir(path)
    files_df = []
    vocab = []
    for file in tqdm(files_list):
        f_pth= os.path.join(path, file)
        family = file.split('_')[-1].split('-')[0]
        df = pd.read_csv(f_pth,sep='\t')
        df = df[['DOCULECT','TOKENS','COGID','CONCEPT']]
        df.dropna(inplace=True)
        df = df.astype({'COGID':'int32'})
        df.rename(columns={'DOCULECT': 'langs', 'TOKENS':'tokens', 'COGID':'cogid', 'CONCEPT': 'concept'}, inplace=True)
        df['tokens'] = df.apply(lambda x: x['tokens'].replace(" ",""), axis=1)
        df = df.groupby(['concept'], as_index=False).agg(list)
        if aligned:
            df['tokens'] = df.apply(lambda x: align(x['tokens']), axis=1)
        df['tokens'] = df.apply(lambda x: [[f"[{family};{lng}]"]+algn for lng, algn in zip(x['langs'], x['tokens'])], axis=1)
        df.drop(columns=['langs'],inplace=True)
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
    
    num_msa = len(row['tokens'])
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
