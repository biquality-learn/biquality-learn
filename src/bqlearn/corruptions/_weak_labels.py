import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_random_state, check_scalar
from sklearn.utils.multiclass import type_of_target, unique_labels

from ..utils import categorical


def make_weak_labels(
    X,
    y,
    estimator=None,
    *,
    train_size=0.1,
    stratify=None,
    discrete=True,
    random_state=None,
):
    """Generate weak labels for a given dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    y : array-like of shape (n_samples, )
        The targets.

    estimator : object, default=None
        The estimator used to generate weak labels.
        If None, LogisticRegression is used as the estimator.

    train_size : float or int, default=0.1
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset that the estimator will be fitted on.
        If int, represents the absolute number of samples that the estimator
        will be fitted on.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    discrete : boolean, default=True
        Determines if corrupted labels are the predicted label
        or sampled from the predicted probability distribution.
        If False, estimator must support predict_proba.

    random_state : int or RandomState, default=None
        Controls the random_state of the estimator.

    Returns
    -------
    y_corrupt : ndarray of shape (n_samples, )
        The untrusted targets as predicitions from the fitted estimator.
    """

    check_scalar(train_size, "train_size", float, min_val=0, max_val=1)

    random_state = check_random_state(random_state)

    classes = unique_labels(y)

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if train_size == 0:
        estimator_ = DummyClassifier(random_state=random_state)
        estimator_.fit(X, y)

    else:
        if estimator is None:
            estimator_ = LogisticRegression()
        else:
            if not discrete and not hasattr(estimator, "predict_proba"):
                raise TypeError(
                    "make_weak_labels requires that the weak learner "
                    "supports the calculation of class "
                    "probabilities with a predict_proba method "
                    "if discrete is set to False.\n"
                    "Please change the base estimator or set discrete to True."
                )
            estimator_ = clone(estimator)

        if hasattr(estimator_, "random_state"):
            estimator_.set_params(random_state=random_state)

        if train_size == 1.0:
            estimator_.fit(X, y)

        else:
            (X_train, _, y_train, _) = train_test_split(
                X,
                y,
                train_size=train_size,
                random_state=random_state,
                shuffle=True,
                stratify=stratify,
            )

            estimator_.fit(X_train, y_train)

    if discrete:
        probabilities = label_binarize(estimator_.predict(X), classes=classes)
        if probabilities.shape[1] == 1:
            probabilities = np.hstack([1 - probabilities, probabilities])
    else:
        probabilities = estimator_.predict_proba(X)

    return classes[categorical(probabilities, random_state=random_state)]
