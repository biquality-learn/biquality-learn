"""
============================================
Iterative Density Ratio based on Sample Loss
============================================
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from bqlearn.corruptions import make_label_noise
from bqlearn.density_ratio import IPDR

# %%
seed = 1
rng = np.random.RandomState(seed)
X = rng.randn(1000, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

trusted, untrusted = next(
    StratifiedShuffleSplit(train_size=0.05, random_state=seed).split(X_train, y_train)
)

sample_quality = np.ones(X_train.shape[0])
sample_quality[untrusted] = 0

y_corrupted = np.copy(y_train)
y_corrupted[untrusted] = make_label_noise(
    y_corrupted[untrusted],
    noise_matrix="permutation",
    noise_ratio=0.4,
    random_state=seed,
)
noisy = y_corrupted[untrusted] != y_train[untrusted]
clean = y_corrupted[untrusted] == y_train[untrusted]

Y_corrupted = LabelBinarizer().fit_transform(y_corrupted)
if Y_corrupted.shape[1] == 1:
    Y_corrupted = np.hstack((1 - Y_corrupted, Y_corrupted))

# %%
gbm = GradientBoostingClassifier(
    n_estimators=1, warm_start=True, learning_rate=0.05, max_depth=3, random_state=seed
)
n_estimators = 100
ipdr = IPDR(gbm, exploit_iterative_learning=True, n_estimators=n_estimators)
ipdr.fit(X_train, y_corrupted, sample_quality=sample_quality)

# %%
staged_sample_weights_untrusted = ipdr.sample_weights_[untrusted, :]
fig = plt.figure()
plt.plot(staged_sample_weights_untrusted[clean, :].mean(axis=0), label="clean")
plt.fill_between(
    np.arange(n_estimators),
    staged_sample_weights_untrusted[clean, :].mean(axis=0)
    - staged_sample_weights_untrusted[clean, :].var(axis=0),
    staged_sample_weights_untrusted[clean, :].mean(axis=0)
    + staged_sample_weights_untrusted[clean, :].var(axis=0),
    alpha=0.2,
    color="blue",
)
plt.plot(staged_sample_weights_untrusted[noisy, :].mean(axis=0), label="noisy")
plt.fill_between(
    np.arange(n_estimators),
    staged_sample_weights_untrusted[noisy, :].mean(axis=0)
    - staged_sample_weights_untrusted[noisy, :].var(axis=0),
    staged_sample_weights_untrusted[noisy, :].mean(axis=0)
    + staged_sample_weights_untrusted[noisy, :].var(axis=0),
    alpha=0.2,
    color="orange",
)
plt.xlabel("Iterations")
plt.ylabel("Weight")
plt.title("Evolution of the weights for clean and noisy samples")
plt.legend()
plt.tight_layout()
plt.show()

# %%
fig = plt.figure()
staged_losses_test = []
for probs in ipdr.estimator_.staged_predict_proba(X_test):
    losses = log_loss(y_test, probs)
    staged_losses_test.append(losses)
staged_losses_test = np.stack(staged_losses_test)

plt.plot(staged_losses_test)
plt.xlabel("Iterations")
plt.ylabel("Test Log-Loss")
plt.tight_layout()
plt.show()
