"""
===============
K-Density Ratio
===============
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from bqlearn.corruptions import (
    make_cluster_imbalance,
    make_instance_dependent_label_noise,
    noisy_leaves_probability,
)
from bqlearn.density_ratio import kmm, pdr

seed = 2

clf = GradientBoostingClassifier(random_state=seed)

names = ["PDR", "KMM"]
reweightings = [partial(pdr, estimator=clf), kmm]

n_samples = 1000

X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=seed)
X = StandardScaler().fit_transform(X)
y = LabelEncoder().fit_transform(y)

K = len(np.unique(y))

figure, axs = plt.subplots(2, 3, figsize=(12, 8))

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

trusted, untrusted = next(
    StratifiedShuffleSplit(train_size=0.05, random_state=seed).split(X, y)
)
sample_quality = np.ones_like(y)
sample_quality[untrusted] = 0

_, _, subsampled = make_cluster_imbalance(
    X,
    y,
    range(n_samples),
    per_class_n_clusters=4,
    majority_ratio=10,
    imbalance_distribution="linear",
    random_state=seed,
)

y[untrusted] = make_instance_dependent_label_noise(
    noisy_leaves_probability(
        X,
        y,
        noise_ratio=0.3,
        purity="ascending",
        min_samples_leaf=40,
        random_state=seed,
    )[untrusted],
    y[untrusted],
    noise_matrix="permutation",
    random_state=seed,
)

selected = np.hstack([trusted, subsampled])

colors = ["purple", "orange"]

axs[0, 0].scatter(
    X[subsampled, 0],
    X[subsampled, 1],
    c=[colors[yy] for yy in y[subsampled]],
    edgecolors="k",
    alpha=0.4,
)
axs[0, 0].scatter(
    X[trusted, 0],
    X[trusted, 1],
    c=[colors[yy] for yy in y[trusted]],
    edgecolors="k",
    marker="s",
)

axs[0, 0].set_xlim(x_min, x_max)
axs[0, 0].set_ylim(y_min, y_max)
axs[0, 0].set_xticks(())
axs[0, 0].set_yticks(())

axs[1, 0].remove()

for i, (name, reweighting) in enumerate(zip(names, reweightings)):
    i, j = divmod(i, 3)

    axt = axs[i, j + 1]
    axb = axs[i + 1, j + 1]

    w = reweighting(
        X[subsampled],
        X[trusted],
    )

    axt.scatter(
        X[subsampled, 0],
        X[subsampled, 1],
        c=[colors[yy] for yy in y[subsampled]],
        s=20 * w,
        edgecolors="k",
        alpha=0.4,
    )

    for k in range(K):
        w_k = reweighting(
            X[subsampled][y[subsampled] == k],
            X[trusted][y[trusted] == k],
        )

        axb.scatter(
            X[subsampled][y[subsampled] == k, 0],
            X[subsampled][y[subsampled] == k, 1],
            color=colors[k],
            s=20 * w_k,
            edgecolors="k",
            alpha=0.4,
        )

    for ax in [axt, axb]:
        ax.scatter(
            X[trusted, 0],
            X[trusted, 1],
            c=[colors[yy] for yy in y[trusted]],
            marker="s",
            edgecolors="k",
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

    axt.set_title(name)
    axb.set_title(f"$K$-{name}")

plt.show()
