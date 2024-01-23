# svmcc

This repository accompanies the paper **Using support vector machines and
state-of-the-art algorithms for phonetic alignment to identify cognates in
multi-lingual wordlists** by Jäger, List and Sofroniev. The repository contains
both the data and the source code used in the paper's experiment.


## data

### datasets

The datasets are located in the ``data/datasets`` directory.

| dataset            | language families            | entries  | source                       |
|--------------------|------------------------------|---------:|------------------------------|
| `abvd`             | Austronesian                 |    12414 | Greenhill et al, 2008        |
| `afrasian`         | Afro-Asiatic                 |      790 | Militarev, 2000              |
| `bai`              | Sino-Tibetan                 |     1028 | Wang, 2006                   |
| `central_asian`    | Turkic, Indo-European        |    15903 | Manni et al, 2016            |
| `chinese_2004`     | Sino-Tibetan                 |     2789 | Hóu, 2004                    |
| `chinese_1964`     | Sino-Tibetan                 |     3653 | Běijīng Dàxué, 1964          |
| `huon`             | Trans-New Guinea             |     1176 | McElhanon, 1967              |
| `ielex`            | Indo-European                |    11572 | Dunn, 2012                   |
| `japanese`         | Japonic                      |     1986 | Hattori, 1973                |
| `kadai`            | Tai-Kadai                    |      400 | Peiros, 1998                 |
| `kamasau`          | Torricelli                   |      271 | Sanders, 1980                |
| `lolo_burmese`     | Sino-Tibetan                 |      570 | Peiros, 1998                 |
| `mayan`            | Mayan                        |     2841 | Brown, 2008                  |
| `miao_yao`         | Hmong-Mien                   |      208 | Peiros, 1998                 |
| `mixe_zoque`       | Mixe-Zoque                   |      961 | Cysouw et al, 2006           |
| `mon_khmer`        | Austroasiatic                |     1424 | Peiros, 1998                 |
| `ob_ugrian`        | Uralic                       |     2006 | Zhivlov, 2011                |
| `tujia`            | Sino-Tibetan                 |      498 | Starostin, 2013              |

Each dataset is stored in a
[tsv](https://en.wikipedia.org/wiki/Tab-separated_values) file where each row is
a word and the columns are as follows:

| column          | info                                                     |
|-----------------|----------------------------------------------------------|
| `language`      | The word's doculect.                                     |
| `iso_code`      | The ISO 639-3 code of the word's doculect; can be empty. |
| `gloss`         | The word's meaning as described in the dataset.          |
| `global_id`     | The Concepticon ID of the word's gloss.                  |
| `local_id`      | The dataset's ID of the word's gloss.                    |
| `transcription` | The word's transcription in either IPA or ASJP.          |
| `cognate_class` | The ID of the set of cognates the word belongs to.       |
| `tokens`        | The word's phonological segments, space-separated.       |
| `notes`         | Field for additional information; can be empty.          |

The datasets are published under a [Creative Commons Attribution-ShareAlike 4.0
International License](https://creativecommons.org/licenses/by-sa/4.0/) and can
also be found in Zenodo (URL pending).


### vectors

The `data/vectors` directory contains the samples and targets (in the machine
learning sense) derived from the datasets, in csv format. With the exception of
`central_asian`, which is split into two because its size exceeds 100 MB, there
is a single vector file per dataset (note that the code will not split this file
for you). In these files each row comprises a pair of words from different
languages but with the same meaning. The features are described in section 4.3
of the paper.


### inferred

The `data/inferred` directory contains the SVM-inferred cognate classes for each
dataset, one `.svmCC.csv` file per dataset. It also contains the cognacies
inferred using the LexStat algorithm, one `.lsCC.csv` file per dataset.


### params

The `data/params` directory contains the parameters used for inferring the PMI
features of the aforementioned feature vectors. For more information, refer to
Jäger (2015).


## code

The `code` directory contains the source code used to run the study's
experiment. It is Python 3 code and needs
[NumPy](https://github.com/numpy/numpy),
[LingPy](https://github.com/lingpy/lingpy),
[scikit-learn](https://github.com/scikit-learn/scikit-learn),
[biopython](https://github.com/biopython/biopython), and
[pandas](https://github.com/pandas-dev/pandas) as direct dependencies. You
should use `requirements.txt` to install the dependencies, as the code is only
guaranteed to work with the specified versions of those.


### setup and usage

```bash
# clone this repository
git clone https://github.com/evolaemp/svmcc

# you do not need to create a virtual environment if you know what you are
# doing; remember that the code is written in python3
virtualenv path/to/my/venv
source path/to/my/venv/bin/activate

# install the dependencies
# it is important to use the versions specified in the requirements file
pip install -r requirements.txt

# this ensures the reproducibility of the results
export PYTHONHASHSEED=0

# use manage.py to invoke the commands
python manage.py --help
```


### commands

`python manage.py prepare <dataset>` reads a dataset, generates its samples and
targets, and writes a vector file ready for svm consumption; `data/vectors` is
the default output directory.

`python manage.py infer --svmcc` reads a directory of vector files, runs
svm-based automatic cognate detection, and writes the inferred classes into an
output directory; the default input and output directories are `data/vectors`
and `data/inferred`, respectively.

`python manage.py test` runs some unit tests.


### licence

The source code (but not the data) is published under the MIT Licence (see the
`LICENCE` file).


## links

* Zenodo URL here
