## CogTran2 - Deep Neural Cognate Detection

The original version of this repository is available on the [`legacy` branch](../../tree/legacy).

Install the package and dependencies:

```bash
python -m pip install .
```

## Cognate prediction example

Provide IPA transcriptions for each form. If starting from ordinary spelling,
convert it to IPA with a language-aware G2P tool first; this package does not
perform orthography-to-IPA conversion. The example uses unequal-sized
“bird” (12 forms) and “dog” (3 forms) MSAs to demonstrate batch collation.

```python
import torch

from cogtran2 import (
    CogTranDataCollator,
    CogTranTokenizer,
    CogTranTriAttn,
    align_msa_sca,
    upgma_clusters,
)

checkpoint = "mahesh27/CogTran2"
meanings = {
    "bird": [
        "cɛːm", "sɛːm", "sɛːmpar", "ceːm", "seːp", "cim",
        "sɨm", "ci̤m", "cem", "seːm", "com", "kaˀjleːmʔ",
    ],
    "dog": ["cɔː", "kɔʔ", "cɑː"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CogTranTokenizer.from_pretrained(checkpoint)
model = CogTranTriAttn.from_pretrained(checkpoint).to(device).eval()

features = []
for meaning, ipa_forms in meanings.items():
    msa = align_msa_sca([tuple(form) for form in ipa_forms])
    unknown = sorted({
        segment
        for row in msa
        for segment in row
        if segment != "-"
        and tokenizer.convert_tokens_to_ids(segment) == tokenizer.unk_token_id
    })
    if unknown:
        raise ValueError(f"{meaning}: unknown ASJP segments: {unknown}")
    features.append({
        "input_ids": [
            tokenizer.encode_form(row, max_length=model.config.max_columns)
            for row in msa
        ]
    })

collator = CogTranDataCollator(
    tokenizer,
    max_length=model.config.max_columns,
    max_rows=model.config.max_rows,
)
batch = {
    name: tensor.to(device)
    for name, tensor in collator(features).items()
}

with torch.inference_mode():
    logits = model(**batch).logits

for batch_index, (meaning, ipa_forms) in enumerate(meanings.items()):
    pair_scores = (
        logits[batch_index, :len(ipa_forms), :len(ipa_forms)]
        .softmax(dim=-1)[..., 1]
        .cpu()
        .tolist()
    )
    clusters = upgma_clusters(pair_scores, threshold=0.4)
    print(meaning)
    for ipa, cluster in zip(ipa_forms, clusters):
        print(f"  {ipa} -> cognate cluster {cluster}")
```

The collator pads the two MSAs together to `[2, 12, 9]` (batch, forms,
aligned tokens). With the Hub checkpoint, “bird” yields clusters including
`cɛːm` → 0, `sɛːmpar` → 1, and `kaˀjleːmʔ` → 2; “dog” includes `cɔː` → 0
and `kɔʔ` → 1. Cluster IDs are local to each meaning and have no intrinsic
meaning.


## Training from scratch

Download [fast-cognate-detection.zip](https://zenodo.org/records/3237509) to the repository root, then rebuild the training, test, and proportion data with:

> `python scripts/generate_proportion_splits.py`


Run the training suite with [run_prop_splits.sh](./run_prop_splits.sh):

```bash
bash run_prop_splits.sh
```

## Test scores

The no-language-token model accepts ASJP representations without language
identifiers, so it can be used with ASJP data from any source. The pretrained
plain-ASJP checkpoint trained for 90% proportion is available at
[`mahesh27/CogTran2`](https://huggingface.co/mahesh27/CogTran2).

Comparison with language-tokens models is given as follows. 
Mean ± sample standard deviation over five splits where available.

| Proportion | Input | Runs | BCPrec | BCRec | BCF |
|---:|---|---:|---:|---:|---:|
| 0 | Language tokens | 1 | 0.8588 ± — | 0.8017 ± — | 0.8186 ± — |
| 0 | Plain ASJP | 1 | 0.8253 ± — | 0.8191 ± — | 0.8167 ± — |
| 12.5 | Language tokens | 5 | 0.8651 ± 0.0108 | 0.8184 ± 0.0127 | 0.8370 ± 0.0034 |
| 12.5 | Plain ASJP | 5 | 0.8461 ± 0.0042 | 0.8197 ± 0.0041 | 0.8270 ± 0.0024 |
| 50 | Language tokens | 5 | 0.8653 ± 0.0126 | 0.8361 ± 0.0083 | 0.8477 ± 0.0017 |
| 50 | Plain ASJP | 5 | 0.8483 ± 0.0135 | 0.8253 ± 0.0074 | 0.8320 ± 0.0048 |
| 90 | Language tokens | 1 | 0.8805 ± — | 0.8485 ± — | 0.8614 ± — |
| 90 | Plain ASJP | 1 | 0.8431 ± — | 0.8422 ± — | 0.8389 ± — |

This specific version implements its MSA attention and pairwise triangle-attention stack
in native PyTorch rather than importing Fair-ESM or the original AlphaFold
modules at runtime. These implementation differences can affect results;
in particular, the language-token scores may differ from the original implementation.



## Citation

```
@inproceedings{akavarapu-bhattacharya-2024-automated,
    title = "Automated Cognate Detection as a Supervised Link Prediction Task with Cognate Transformer",
    author = "Akavarapu, V.S.D.S.Mahesh  and
      Bhattacharya, Arnab",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.58/",
    doi = "10.18653/v1/2024.eacl-long.58",
    pages = "965--975"
}
```