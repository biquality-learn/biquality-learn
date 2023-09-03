.. _bqlearn_userguide_rnd:

========================
Radon-Nikodym Derivative
========================

In the Empirical Risk Minimization (ERM) framework, when learning a
classifier :math:`f:\mathcal{X}\rightarrow \mathcal{Y}`, we seek to minimize
its risk :math:`R(f)` given a loss function
:math:`L:\mathcal{Y}^2\rightarrow \mathbb{R}`:

.. math:: R(f)={\mathbb {E}}[L(f(X),Y)]=\int L(f(X),Y)\,\text{d}\mathbb{P}(X,Y)

However, when the true distribution :math:`T` cannot be observed because of
weaknesses of supervision or dataset shifts, we can rewrite given the observed distribution
:math:`U`:

.. math:: R(f)=\int \frac{\text{d}\mathbb{P}_T(X,Y)}{\text{d}\mathbb{P}_U(X,Y)}L(f(X),Y)\,\text{d}\mathbb{P}_U(X,Y)

Thus, in order to minimize the true risk of the classifier on the true but
unobserved distribution, we need to reweight the loss function on the observed
distrbution by :math:`\mathbb{P}_T(X,Y)/\mathbb{P}_U(X,Y)`. This measure
is called the Radon-Nikodym Derivative (RND) of :math:`\mathbb{P}_U(X,Y)` with respect
to :math:`\mathbb{P}_U(X,Y)`. Assuming that :math:`\mathbb{P}_T(X,Y)` is absolutely continuous with
respect to :math:`\mathbb{P}_U(X,Y)`, then the RND exists and is unique.

This reweighting scheme has particularly inspired the literature of
covariate shift, and many algorithms have been implemented to estimate the RND,
regrouped in an umbrella named Density Ratio Estimators.

However, estimating the RND can be a difficult task, especially in the
case of distribution shift where the joint distribution ratio needs to be estimated.
Proposals have been made to ease this estimation.

Importance Reweighting for Biquality Learning
=============================================

A *first approach* is to focus on the concept drift between datasets using the Bayes Formula:

.. math::

    \beta(X,Y) =
    \frac{\mathbb{P}_T(X,Y)}{\mathbb{P}_U(X,Y)} = \frac{\mathbb{P}_T(Y \mid X)\mathbb{P}_T(X)}{\mathbb{P}_U(Y \mid X)\mathbb{P}_U(X)}

This approach is implemented in :class:`bqlearn.irbl.IRBL` where both conditional 
distributions are estimated using two calibrated classifiers [NLBC2021]_.
It is also implemented in :class:`bqlearn.irlnl.IRLNL` (see :ref:`bqlearn_userguide_transition`).

To be noted that none of the implemented algorithms estimates the ratio of features distribution.

.. topic:: References:

 .. [NLBC2021] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols, "Importance Reweighting for Biquality Learning", IJCNN, 2021.

K-Density Ratio
===============

.. currentmodule:: bqlearn.density_ratio

A *second approach* is to focus on the class-conditional covariate shift between datasets using the Bayes Formula differently:

.. math::

    \beta(X,Y) =
    \frac{\mathbb{P}_T(X,Y)}{\mathbb{P}_U(X,Y)} = \frac{\mathbb{P}_T(X \mid Y)\mathbb{P}_T(Y)}{\mathbb{P}_U(X \mid Y)\mathbb{P}_U(Y)}

With this approach, the joint density ratio estimation task has been decomposed into
:math:`K`-tasks where :math:`K` is the number of classes to predict.
For each class, only samples of the given class :math:`y` are selected on both
datasets, such that the samples are drawn from the
:math:`\mathbb{P}(X \mid Y=y)` distribution. Then, a density ratio
estimation procedure is used on these sub-datasets to estimate
:math:`\mathbb{P}_T(X \mid Y=y)/\mathbb{P}_U(X \mid Y=y)`. When repeated on all classes, this
approach does handle distribution shifts.

This approach is implemented in :class:`KPDR` [NLBC2023]_ and :class:`KKMM` [FNS2020]_ with
two different density ratio estimation algorithms, :func:`pdr` [BBS2007]_ and :func:`kmm` [HSGBS2006]_ respectively.

.. figure:: ../auto_examples/images/sphx_glr_plot_kdr_001.png
   :target: ../auto_examples/plot_kdr.html
   :align: center
   :alt: K-Density Ratio

   Illustration of the reweighting mechanisms behind K-Density Ratio.

The above figures shows how :func:`pdr` and :func:`kmm` by design can't take into account label noise, only subsampling biais.
However, :class:`KPDR` and :class:`KKMM` are able to deal with distribution shifts on this toy dataset.

Moreover, :class:`KKMM` implement a batched version for scalability [YAM2015]_.

.. topic:: References:

 .. [NLBC2023] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols, "Biquality Learning: a Framework to Design Algorithms Dealing with Closed-Set Distribution Shifts.", Machine Learning, 2023.

 .. [FNS2020]  Fang, T., Lu, N., Niu, G., and Sugiyama, M. "Rethinking importance weighting for deep learning under distribution shift.", NeurIPS 2020

 .. [BBS2007] S. Bickel, M. Bruckner, T. Scheffer, "Discriminative Learning for Differing Training and Test Distributions", 2007

 .. [HSGBS2006] Huang, J. and Smola, A. and Gretton, A. and Borgwardt, KM. and Schölkopf, B., "Correcting Sample Selection Bias by Unlabeled Data", 2006

 .. [YAM2015]  Miao Y., Farahat A. and Kamel M. "Ensemble Kernel Mean Matching", 2015

Loss based Density Ratio
========================

.. currentmodule:: bqlearn.density_ratio

A *third approach* is to focus on the density ratio estimation task by finding a deterministic and invertible transformation :math:`f`:

.. math::

    \beta(X,Y) =
    \frac{\mathbb{P}_T(X,Y)}{\mathbb{P}_U(X,Y)} =
    \frac{\mathbb{P}_T(Z)}{\mathbb{P}_U(Z)},\quad Z = f(X,Y)

An example of such transformation :math:`f` is the classification loss of a model learned on the biquality data.

This approach is implemented in :class:`IPDR` [L2018]_ and :class:`IKMM` [FNS2020]_.

.. figure:: ../auto_examples/images/sphx_glr_plot_idr_001.png
   :target: ../auto_examples/plot_idr.html
   :align: center
   :alt: Iterative Density Ratio

   Evolution of the weights computed by :class:`IPDR` for clean and noisy samples on a toy dataset corrupted with label noise.

:class:`IPDR` and :class:`IKMM` use a windowing approach where losses from previous iterations can be used
in addition to the current loss to estimate the density ratio. It allows to recover algorithms such as MentorNet [L2018]_ with :class:`IPDR`.

.. topic:: References:

 .. [FNS2020]  Fang, T., Lu, N., Niu, G., and Sugiyama, M. "Rethinking importance weighting for deep learning under distribution shift.", NeurIPS 2020

 .. [L2018] Jiang, Lu, et al. "Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels." International conference on machine learning. PMLR, 2018.