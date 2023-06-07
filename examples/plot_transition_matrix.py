"""
======================================
Transition Matrices Estimation on NIST
======================================

This example illustrates the different algorithms to estimate
transition matrices on biquality datasets.
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from bqlearn.corruptions import make_label_noise
from bqlearn.metrics import (
    anchor_transition_matrix,
    gold_transition_matrix,
    iterative_anchor_transition_matrix,
)

X, y = datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

y_corrupted = make_label_noise(y_train, "background", noise_ratio=0.3, random_state=1)

true = confusion_matrix(y_train, y_corrupted, normalize="true")

clf = MLPClassifier(max_iter=30, random_state=1).fit(X_train, y_corrupted)

glc = gold_transition_matrix(y_test, clf.predict_proba(X_test))
anchor = anchor_transition_matrix(clf.predict_proba(X_train), quantile=0.95)
iterative_anchor = iterative_anchor_transition_matrix(
    clf.predict_proba(X_train), quantile=0.95, n_iter=100
)
confusion = confusion_matrix(y_test, clf.predict(X_test), normalize="true")

transition_matrices = [
    ("True", true),
    ("Confusion", confusion),
    ("Anchor", anchor),
    ("Iterative Anchor", iterative_anchor),
    ("GLC", glc),
]

plt.figure(figsize=(10, 4))

for i, (name, this_tm) in enumerate(transition_matrices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(this_tm, interpolation="nearest", vmin=0, vmax=1, cmap="Blues")
    plt.xticks(())
    plt.yticks(())
    plt.title("%s\ntransition matrix" % name)

plt.tight_layout(pad=2)
plt.show()
