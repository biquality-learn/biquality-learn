"""
===========================
Leveraging Metadata Routing
===========================

This example shows how to leverage metadata routing capabilities of
scikit-learn estimators with biquality-learn.
"""

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from bqlearn.corruptions import make_label_noise
from bqlearn.irbl import IRBL

seed = 2

base_classifier = LogisticRegression()

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
    n_classes=2,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed
)

trusted, untrusted = next(
    StratifiedShuffleSplit(train_size=0.1, random_state=1).split(X_train, y_train)
)
y_train[untrusted] = make_label_noise(
    y_train[untrusted], noise_matrix="flip", noise_ratio=0.8, random_state=seed
)

bc = BaggingClassifier(base_classifier, random_state=seed).fit(X_train, y_train)
clf = clone(base_classifier).fit(X_train, y_train)

sample_quality = np.ones_like(y_train)
sample_quality[untrusted] = 0
bq = IRBL(base_classifier, base_classifier).fit(
    X_train, y_train, sample_quality=sample_quality
)
# TODO : Uncomment when https://github.com/scikit-learn/scikit-learn/pull/24250 merged
# bq_bc = BaggingClassifier(
#     IRBL(base_classifier, base_classifier), random_state=seed
# ).fit(X, y, sample_quality=sample_quality)

print(
    clf.score(X_test, y_test), bc.score(X_test, y_test), bq.score(X_test, y_test)
)  # , bq_bc.score(X_test, y_test))
