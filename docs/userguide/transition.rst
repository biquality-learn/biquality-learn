.. _bqlearn_userguide_transition:

=========================
Noise Transition Matrices
=========================

In the Robust Learning to Label Noise literature, label noise models can
be represented using to their transition matrices :math:`\mathbf{T}`,
which represent the probability of seeing a noisy label
:math:`\tilde{Y}` given the true label :math:`Y` and the features
:math:`X`:

.. math:: \forall (i,j) \in [\![1,K]\!]^2, \forall x \in \mathcal{X}, \mathbf{T}_{i,j}(x) = \mathbb{P}(\tilde{Y}=j | Y=i, X=x)

Thus, an essential part of the literature has sought to estimate these
transition matrices to correct the classifier’s training procedure from
noisy data.

Indeed, given the law of total probability:

.. math:: \mathbb{P}(\tilde{Y} | X) = \sum_{k=1}^K\mathbb{P}(\tilde{Y} | X, Y=k)\mathbb{P}(Y = k | X)

However, none of the implemented algorithms and estimators use the instance-dependent noise transition
matrix :math:`\mathbf{T}(x)` but instead use the instance-independent form :math:`\mathbf{T}`.

Moreover, these approaches do not take into account shifts in the feature distribution.

Correcting Classifiers with Transition Matrices.
================================================

Plug-in
-------

The most straightforward approach is implemented in :class:`bqlearn.plugin.PluginCorrection` and uses the transition matrix to correct a classifier learn
on untrusted data at prediction time [ZLA2021]_.

.. math:: \forall x \in \mathcal{X}, f_T(x)=(\mathbf{T}^{t})^{-1}\cdot f_U(x)

By inverting the transition matrix, we can estimate the trusted classifier from the untrusted classifier.
However, this approach does not allow to derive probability predictions as the inverse transition matrix
can be non-stochastic.

Reweighting
-----------

Another approach is based on instance reweighting implemented in :class:`bqlearn.irlnl.IRLNL` (see :ref:`bqlearn_userguide_rnd`) [LT2015]_.

.. math:: 
    \frac{\mathbb{P}(Y=y|X)}{\mathbb{P}(\tilde{Y}=y|X)} 
    = \frac{\mathbb{P}(\tilde{Y}=y|X) - \mathbb{1}_{\tilde{Y}=y} \times \mathbb{P}(\tilde{Y}= y | Y\neq y ) - \mathbb{1}_{\tilde{Y}\neq y} \times \mathbb{P}(\tilde{Y}\neq y | Y =y )}
    {\left(1 -  \mathbb{P}(\tilde{Y}= y | Y\neq y ) - \mathbb{P}(\tilde{Y}\neq y | Y =y )\right)\mathbb{P}(\tilde{Y}=y|X)}

Each term used for this reweighting can be computed using to the noise transition matrix :math:`\mathbf{T}` and the clean and noisy priors :math:`\mathbb{P}(\tilde{Y})` and :math:`\mathbb{P}(Y)`:

.. math:: 
    \begin{aligned}
    \mathbb{P}(\tilde{Y}\neq y | Y =y ) &= 1 - \mathbf{T}_{yy}\\
    \mathbb{P}(\tilde{Y}= y | Y \neq y ) &= \frac{\mathbb{P}(\tilde{Y}=y) - \mathbf{T}_{yy} \mathbb{P}(Y=y)}{1 - \mathbb{P}(Y=y)}\\
    \end{aligned}

It is implemented for multiclass classification in a One versus Rest fashion [WLT2018]_.

Loss
----

A final known approach is implemented in :class:`bqlearn.unbiased.LossCorrection` named the method of unbiased estimator [NDRT2013]_.
It constructs a new loss :math:`\tilde{L}` from the loss of interest :math:`L` such that :math:`\mathbb{E}_{\tilde{y}}[\tilde{L}(f(x), \tilde{y})] = L(f(x), y)`.

.. math:: \tilde{L}(f(x), y) = \frac{ ( 1 - \mathbb{P}(\tilde{Y}= y | Y\neq y )) L(f(x), y) - \mathbb{P}(\tilde{Y}\neq y | Y =y ) L(f(x), -y) }{1 -  \mathbb{P}(\tilde{Y}= y | Y\neq y ) - \mathbb{P}(\tilde{Y}\neq y | Y =y )}

It is implemented in a way which is classifier agnostic by duplicating all samples :math:`(X,y)` into :math:`(X,y,w_y)` and :math:`(X,-y,w_{-y})` and weighting them by :math:`w_y = \frac{ 1 - \mathbb{P}(\tilde{Y}= y | Y\neq y )}{1 -  \mathbb{P}(\tilde{Y}= y | Y\neq y ) - \mathbb{P}(\tilde{Y}\neq y | Y =y )}`
and :math:`w_{-y}=\frac{-\mathbb{P}(\tilde{Y}\neq y | Y =y )}{1 -  \mathbb{P}(\tilde{Y}= y | Y\neq y ) - \mathbb{P}(\tilde{Y}\neq y | Y =y )}`.

.. figure:: ../auto_examples/images/sphx_glr_plot_unbiased_xp_001.png
   :target: ../auto_examples/plot_unbiased_xp.html
   :align: center
   :alt: unbiased xp

   Reproduced Figures 1 and 2 from "Learning with Noisy Labels" [NDRT2013]_.

For multiclass classification, it uses a One versus Rest scheme.

.. topic:: References:

 .. [NDRT2013] N. Natarajan, I. S. Dhillon, P. Ravikumar, and A. Tewari, "Learning with Noisy Labels", NeurIPS, 2013.

 .. [ZLA2021] M. Zhang, J. Lee, and S. Agarwal. "Learning from noisy labels with no change to the training process.", ICML, 2021.

 .. [LT2015] T. Liu and D. Tao, "Classification with noisy labels by importance reweighting.", in IEEE Transactions on pattern analysis and machine intelligence, 2015

 .. [WLT2018] R. Wang, T. Liu and D. Tao, "Multiclass Learning With Partially Corrupted Labels", in IEEE Transactions on Neural Networks and Learning Systems, 2018.

Estimating Noise Transition Matrices
====================================

In order to estimate noise transition matrices, algorithms have been relying on the availability of trusted data (or anchor points).

If no trusted data is available, anchor points :math:`A` are heuristically chosen with the following algorithm:

.. math::

    \forall i \in [\![1,K]\!], A_i =
    \operatorname*{argmax}_{x \in D} \mathbb{P}(Y=i|X=x)

Then it uses predictions of a model learned on untrusted data to estimate the transition matrix.

.. math::

    \forall (i,j) \in [\![1,K]\!]^2, \hat{T}_{(i,j)} = \mathbb{P}(\tilde{Y}=j|X=A_i)

This algorithm is available with :func:`bqlearn.metrics.anchor_transition_matrix`:

.. code-block:: python

    >>> from bqlearn.metrics import anchor_transition_matrix
    >>> y_prob = [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]]
    >>> anchor_transition_matrix(y_prob, quantile=1.0)
    array([[0.9, 0.1],
           [0.2, 0.8]])

In these example, the first sample has been selected as the anchor point of class :math:`0` and the the second sample has been selected as the anchor point of class :math:`1`.
Then the transition matrix has been constructed using these two points.

If there is trusted data available, they can be used as anchor points. Thus, the classifier learned on untrusted data can be evaluated on them
and the noise matrix can be estimated with :func:`sklearn.metrics.confusion_matrix` with row normalization if the
classifier does not support probability predictions, or with :func:`bqlearn.metrics.gold_transition_matrix` otherwise, as it has been shown to be
empirically more efficient:

.. code-block:: python

    >>> from bqlearn.metrics import gold_transition_matrix
    >>> y = [0, 1, 0]
    >>> y_prob = [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]]
    >>> gold_transition_matrix(y, y_prob)
    array([[0.65, 0.35],
           [0.2 , 0.8 ]])

Below is an illustration of these algorithms on the digitis dataset corrupted with :func:`bqlearn.corruptions.noise_matrices.background_noise_matrix`:

.. figure:: ../auto_examples/images/sphx_glr_plot_transition_matrix_001.png
   :target: ../auto_examples/plot_transition_matrix.html
   :align: center
   :alt: transition matrices

   Noise transition matrices estimation on the digits dataset with background label noise.