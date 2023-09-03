# biquality-learn

[![main](https://github.com/biquality-learn/biquality-learn/actions/workflows/main.yml/badge.svg)](https://github.com/biquality-learn/biquality-learn/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/biquality-learn/biquality-learn/branch/main/graph/badge.svg)](https://codecov.io/gh/biquality-learn/biquality-learn)
[![versions](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)
[![pypi](https://img.shields.io/pypi/v/biquality-learn?color=blue)](https://pypi.org/project/biquality-learn/)

**biquality-learn** (or bqlearn in short) is a library à la [scikit-learn](https://github.com/scikit-learn/scikit-learn) for Biquality Learning. 

## Biquality Learning

Biquality Learning is a machine learning framework to train classifiers on Biquality Data, where the dataset is split into a trusted and an untrusted part:

* The ***trusted*** dataset contains trustworthy samples with clean labels and proper feature distribution.
* The ***untrusted*** dataset contains potentially corrupted samples from label noise or covariate shift (distribution shift).

**biquality-learn** aims at making well-known and proven biquality learning algorithms *accessible* and *easy to use* for everyone and enabling researchers to experiment in a *reproducible* way on biquality data.

## Install

biquality-learn requires multiple dependencies:

- numpy>=1.17.3
- scipy>=1.5.0
- scikit-learn>=1.3.0
- scs>=3.2.2

The package is available on [PyPi](https://pypi.org). To install biquality-learn, run the following command :

```bash
pip install biquality-learn
```

A dev version is available on [TestPyPi](https://test.pypi.org) :

```bash
pip install --index-url https://test.pypi.org/simple/ biquality-learn
```

## Quick Start

For a quick example, we are going to train one of the available biquality classifiers, ``KPDR``, on the ``digits`` dataset with synthetic asymmetric label noise.

### Loading Data

First, we must load the dataset with **scikit-learn** and split it into a ***trusted*** and ***untrusted*** dataset.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

X, y = load_digits(return_X_y=True)

trusted, untrusted = next(StratifiedShuffleSplit(train_size=0.1).split(X, y))
```

### Simulating Label Noise

Then we generate label noise on the untrusted dataset.

```python
from bqlearn.corruption import make_label_noise

y[untrusted] = make_label_noise(y[untrusted], "flip", noise_ratio=0.8)
```

### Training Biquality Classifier

Finally, we train ``KKMM`` on the biquality dataset by providing the ``sample_quality`` metadata, indicating if a sample is trusted or untrusted.

```python
from sklearn.linear_models import LogisticRegression
from bqlearn.density_ratio import KKMM

bqclf = KKMM(LogisticRegression(), kernel="rbf")

sample_quality = np.ones(X.shape[0])
sample_quality[untrusted] = 0

bqclf.fit(X, y, sample_quality=sample_quality)
bqclf.predict(X)
```

## Citation

If you use **biquality-learn** in your research, please consider citing us :

```
@misc{nodet2023biqualitylearn,
      title={biquality-learn: a Python library for Biquality Learning}, 
      author={Pierre Nodet and Vincent Lemaire and Alexis Bondu and Antoine Cornuéjols},
      year={2023},
      eprint={2308.09643},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgment

This work has been funded by Orange Labs.

[<img src="https://c.woopic.com/logo-orange.png" alt="Orange Logo" width="100"/>](https://orange.com)
