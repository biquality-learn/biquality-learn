from sklearn.utils import check_random_state


def categorical(pvals, random_state=None):
    """Broadcasted multinomial distribution over pvals on the first axis.
    The results is the outcome of one experiment.

    Parameters
    ----------
    pvals : array-like of shape (n, m)
        The probabilities array of the multiple multinomial distributions.

    random_state : int or RandomState, default=None
        Controls the random seed given.

    Returns
    -------
    output : ndarray of shape (n,)
        The outcome of the multinomial experiments.
    """
    random_state = check_random_state(random_state)

    return (
        pvals.cumsum(-1) >= random_state.uniform(size=pvals.shape[0])[..., None]
    ).argmax(-1)
