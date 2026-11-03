## Automated Cognate Detection with Cognate Transformer
V.S.D.S.Mahesh Akavarapu, Arnab Bhattacharya


The primary model uses the 12.5% training split by default and can be run by the command:

> `python -m scripts.cogtran_triattn`

The model is stored in `models/` and the results are generated in `results/`. For running on other train proportions or on ablated architecture check with the following command:

> `python -m scripts.cogtran_triattn --help`

For running on word-pair input only model i.e. on **CogTranPair** similarly run `python -m scripts.cogtran_pair` from the repository root.

The model scripts expect cross-validation data under `data/prop_125/split_<n>/` by default, with separate `train/` and `test/` directories. The `--prop50` option selects `data/prop_50/split_<n>/`; `--prop0` uses the standalone `data/train/`, `data/test1/`, and `data/test2/` directories. Generate the data folders before running an experiment.

Download [fast-cognate-detection.zip](https://zenodo.org/records/3237509) to the repository root, then rebuild the training, test, and proportion data with:

> `python scripts/generate_proportion_splits.py`

The script recreates `data/train/`, `data/test1/`, and `data/test2/` from the archive and adds stable row IDs to test datasets that lack them. It also generates 12.5% (`prop_125`) and 50% (`prop_50`) training proportions with five folds each, plus one 90% (`prop_90`) split. Each dataset is divided by `CONCEPT`, keeping a concept's rows together in either train or test. Pass `--archive` to use a ZIP stored elsewhere, or `--data-root` to write the generated data to another directory.

## Reproducing results for baseline models

* For running Siamese CNN run `python -m scripts.siamese_cnn` from the repository root.

* For running LexStat-Infomap and SCA follow the procedures in `lexstat.py` and `sca.py` respectively from the repository of Rama et. al (2018) [link](https://github.com/PhyloStar/AutoCogPhylo)
