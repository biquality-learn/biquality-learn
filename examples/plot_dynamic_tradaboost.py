# %%
"""
=============================
Dynamic Updates in TrAdaBoost
=============================

This example illustrates the TrAdaBoost correction from
"Adaptive boosting for transfer learning using dynamic updates"
by Al-Stouhi and al. to avoid weight drift for untrusted samples.
It has been extended to work with a different learning rate and number of classes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from bqlearn.corruptions import make_label_noise
from bqlearn.tradaboost import TrAdaBoostClassifier

n_samples = 1000
n_classes = 3
learning_rate = 0.5

seed = 0

X, y = make_blobs(n_samples=n_samples, centers=n_classes, random_state=seed)

X = StandardScaler().fit_transform(X)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

trusted, untrusted = next(
    StratifiedShuffleSplit(train_size=0.1, random_state=seed).split(X, y)
)
sample_quality = np.ones(n_samples)
sample_quality[untrusted] = 0

y[untrusted] = make_label_noise(
    y[untrusted], "background", noise_ratio=0.4, random_state=seed
)

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
tsc = ax.scatter(X[trusted, 0], X[trusted, 1], marker="s", c=y[trusted])
usc = ax.scatter(X[untrusted, 0], X[untrusted, 1], alpha=0.3, c=y[untrusted])
txt = ax.text(0.01, 0.01, "iteration 0", transform=ax.transAxes)

trada = TrAdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=1,
    learning_rate=learning_rate,
)
trada.fit(X, y, sample_quality=sample_quality)

sample_weight = np.ones(n_samples)
sample_weights = []
max_iter = 30
for i in range(max_iter + 1):
    sample_weight /= sample_weight.sum()
    sample_weights.append(sample_weight)
    sample_weight, _, _ = trada._boost(
        i, X, y, np.copy(sample_weight), sample_quality, seed, {}
    )


def init():
    return ax


def update(d):
    i, sample_weight = d
    tsc.set_sizes(sample_weight[trusted] * 20 * n_samples)
    usc.set_sizes(sample_weight[untrusted] * 20 * n_samples)
    txt.set_text(f"iteration {i}")
    return ax


ani = FuncAnimation(
    fig,
    update,
    frames=zip(range(max_iter + 1), sample_weights),
    init_func=init,
    blit=False,
)
plt.tight_layout()
plt.show()

# %%
# We can verify that the sum of the trusted and untrusted weights
# is constant given the weight drift correction

sample_weights = np.vstack(sample_weights).T

fig, ax = plt.subplots(figsize=(5, 3))

ax.plot(
    np.arange(max_iter + 1),
    np.sum(sample_weights[sample_quality == 0], axis=0),
    label="untrusted",
)
ax.plot(
    np.arange(max_iter + 1),
    np.sum(sample_weights[sample_quality == 1], axis=0),
    label="trusted",
)

ax.set_xlabel("Iterations")
ax.set_ylabel("Sum of Weights")
ax.legend()

plt.tight_layout()
plt.show()

# %%
# After few iterations, they both converge to a constant value.
