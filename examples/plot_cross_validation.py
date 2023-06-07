"""
======================================
Cross-Validating Biquality Classifiers
======================================

Any cross-validators working for usual Supervised Learning can work
in the case of Biquality Learning. However, when splitting the data into
a train and test set, untrusted samples need to be removed from the test set
to avoid computing supervised metrics on corrupted labels.
That is why ``make_biquality_cv`` is provided by biquality-learn
to post-process any scikit-learn compatible cross-validators.
"""

import warnings
from time import time

import numpy as np
import scipy.stats as stats
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.utils.fixes import loguniform

from bqlearn.corruptions import make_label_noise
from bqlearn.density_ratio import KPDR
from bqlearn.model_selection import make_biquality_cv

# get some data
X, y = load_digits(return_X_y=True, n_class=2)

trusted, untrusted = next(
    StratifiedShuffleSplit(train_size=0.2, random_state=1).split(X, y)
)
y[untrusted] = make_label_noise(
    y[untrusted], noise_matrix="flip", noise_ratio=0.9, random_state=2
)

sample_quality = np.ones_like(y)
sample_quality[untrusted] = 0

# build a classifier
clf = SGDClassifier(loss="log_loss", penalty="elasticnet", fit_intercept=True)
bq_clf = KPDR(clf, n_jobs=-1)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {
    "estimator__average": [True, False],
    "estimator__l1_ratio": stats.uniform(0, 1),
    "estimator__alpha": loguniform(1e-5, 1e0),
}

# run randomized search
n_iter_search = 10
random_search = RandomizedSearchCV(
    bq_clf,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=make_biquality_cv(X, sample_quality, cv=3),
)

start = time()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    random_search.fit(X, y, sample_quality=sample_quality)
print(
    "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    % ((time() - start), n_iter_search)
)
report(random_search.cv_results_)
