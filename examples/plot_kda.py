"""
============================================
Label Noise as a K-Domain Adaptation problem
============================================
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from bqlearn.corruptions import make_label_noise

n_samples = 150
n_classes = 2

seed = 0

X, y = make_blobs(n_samples=n_samples, centers=n_classes, random_state=seed)

figure = plt.figure(figsize=(12, 3.5))

X = StandardScaler().fit_transform(X)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

trusted, untrusted = next(StratifiedShuffleSplit(train_size=0.2).split(X, y))
is_trusted = np.isin(np.arange(n_samples), trusted)

y[untrusted] = make_label_noise(
    y[untrusted], "uniform", noise_ratio=0.6, random_state=seed
)
cmap = ListedColormap(["purple", "orange"])

ax = plt.subplot(1, 4, 1)

ax.scatter(
    X[trusted, 0], X[trusted, 1], c=y[trusted], edgecolors="k", marker="s", cmap=cmap
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Trusted Dataset")

ax = plt.subplot(1, 4, 2)

ax.scatter(
    X[untrusted, 0],
    X[untrusted, 1],
    c=y[untrusted],
    alpha=0.5,
    edgecolors="k",
    cmap=cmap,
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Untrusted Dataset")

ax = plt.subplot(1, 4, 3)

ax.scatter(
    X[is_trusted & (y == 0), 0],
    X[is_trusted & (y == 0), 1],
    c=y[is_trusted & (y == 0)],
    edgecolors="k",
    marker="s",
    cmap=cmap,
)
ax.scatter(
    X[~is_trusted & (y == 0), 0],
    X[~is_trusted & (y == 0), 1],
    c=y[~is_trusted & (y == 0)],
    edgecolors="k",
    alpha=0.4,
    cmap=cmap,
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Combined Dataset\nof samples of class 0")

ax = plt.subplot(1, 4, 4)

ax.scatter(
    X[is_trusted & (y == 1), 0],
    X[is_trusted & (y == 1), 1],
    c=y[is_trusted & (y == 1)],
    edgecolors="k",
    marker="s",
    cmap=ListedColormap(["orange"]),
)
ax.scatter(
    X[~is_trusted & (y == 1), 0],
    X[~is_trusted & (y == 1), 1],
    c=y[~is_trusted & (y == 1)],
    edgecolors="k",
    alpha=0.4,
    cmap=ListedColormap(["orange"]),
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Combined Dataset\nof samples of class 1")

plt.tight_layout()
plt.show()
