============
Introduction
============

Biquality Learning is a machine learning framework to train classifiers on Biquality Data,
where the dataset is split into a trusted and an untrusted part:

* The *trusted* dataset contains trustworthy samples with clean labels and proper feature distribution.
* The *untrusted* dataset contains potentially corrupted samples from label noise or covariate shift (distribution shift).

We designed the **biquality-learn** library following the general design
principles of **scikit-learn**, meaning that it provides a consistent
interface for training and using biquality learning algorithms with an
easy way to compose building blocks provided by the library with other
blocks from libraries sharing these design principles.
It includes various reweighting algorithms, plugin correctors,
and functions for simulating label noise and generating sample data
to benchmark biquality learning algorithms.

**biquality-learn** and its dependencies can be easily installed through pip:

::

   pip install biquality-learn

Overall, the goal of **biquality-learn** is to make well-known and
proven biquality learning algorithms accessible and easy to use for
everyone and to enable researchers to experiment in a reproducible way
on biquality data.

-  Source Code: https://github.com/biquality-learn/biquality-learn

-  Documentation: https://biquality-learn.readthedocs.io/

-  License: BSD 3-Clause

.. _design:

Design of the API
=================

`scikit-learn <https://scikit-learn.org/stable/>`_ is a machine learning
library for Python with a design philosophy emphasizing consistency,
simplicity, and performance. The library provides a consistent interface
for various algorithms, making it easy for users to switch between
models. It also aims to make machine learning easy to get started with
through user-friendly API and precise documentation. Additionally, it is
built on top of efficient numerical libraries (`numpy <https://numpy.org/>`_,
and `SciPy <https://scipy.org/>`_) to ensure that models can be
trained and used on large datasets in a reasonable amount of time.

In **biquality-learn**, we followed the same principle, implementing a
similar API with :meth:`fit`, :meth:`transform`, and :meth:`predict` methods.
In addition to passing the input features :math:`X` and the labels
:math:`Y` as in **scikit-learn**, in **biquality-learn**, we need to
provide information regarding whether each sample comes from the trusted or
untrusted dataset: the additional *sample_quality* parameter serves to specify
from which dataset the sample originates where a value of 0 indicates an untrusted
sample, and 1 a trusted one.

Which algorithms are implemented in biquality-learn ?
=====================================================

In **biquality-learn**, we purposely implemented only a specific class
of algorithms centered on approaches for tabular data and classifiers,
thus restricting approaches that are genuinely classifier agnostic or
implementable within **scikit-learn**\ ’s API. We did so not to break
the design principles shared with **scikit-learn** and not impose a
particular deep learning library such as `PyTorch <https://pytorch.org/>`_,
or `TensorFlow <https://www.tensorflow.org/?hl=fr>`_ on the user.

We summarized all implemented algorithms and what kind of corruption
they can handle in the following Table.

.. table:: Algorithms implemented in biquality-learn

   +-----------------------+--------------------+-----------------------+
   | **Algorithms**        | **Dataset Shifts** | **Weaknesses of       |
   |                       |                    | Supervision**         |
   +=======================+====================+=======================+
   | EasyAdapt [H2007]_    | :math:`\checkmark` | :math:`\times`        |
   +-----------------------+--------------------+-----------------------+
   | TrAdaBoost [DYXY2008]_| :math:`\checkmark` | :math:`\times`        |
   +-----------------------+--------------------+-----------------------+
   | Unhinged [RMW2015]_   | :math:`\times`     | :math:`\checkmark`    |
   | (Linear/Kernel)       |                    |                       |
   +-----------------------+--------------------+-----------------------+
   | Backward [NDRT2013]_  | :math:`\times`     | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+
   | IRLNL [LT2015]_       | :math:`\times`     | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+
   | Plugin [ZLA2021]_     | :math:`\times`     | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+
   | KKMM [FNS2020]_       | :math:`\checkmark` | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+
   | IKMM [FNS2020]_       | :math:`\checkmark` | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+
   | IRBL [NLBC2021]_      | :math:`\times`     | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+
   | KPDR [NLBC2023]_      | :math:`\checkmark` | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+
   | IPDR [L2018]_         | :math:`\checkmark` | :math:`\checkmark`    |
   +-----------------------+--------------------+-----------------------+

Refer to the documentation and examples provided in the library for more information :
:ref:`bqlearn_api_reference`.

Training Biquality Learning Classifiers
=======================================

Training a biquality learning algorithm using **biquality-learn** is the
same procedure as training a supervised algorithm with **scikit-learn**
thanks to the library\ ’s design. The features :math:`X` and the targets
:math:`Y` of samples belonging to the trusted
dataset :math:`D_T` and untrusted dataset :math:`D_U` must be provided
as one global dataset :math:`D`. Additionally, the indicator
representing if a sample is trusted or not has to be provided:
:math:`\textit{sample_quality}=\mathbb{1}_{X\in D_T}`.

Here is an example of how to train a biquality classifier using the
:class:`bqlearn.density_ratio.KKMM` (K-Kernel Mean Matching) algorithm from **biquality-learn**:

.. code:: python 

   from sklearn.linear_models import LogisticRegression
   from bqlearn.density_ratio import KKMM

   kkmm = KKMM(LogisticRegression(), kernel="rbf")
   kkmm.fit(X, y, sample_quality=sample_quality)
   kkmm.predict(X_new)   

scikit-learn's metadata routing
===============================

**scikit-learn**\ ’s metadata routing is a Scikit Learn Enhancement
Proposal (SLEP006) describing a system that can be used to seamlessly
incorporate various metadata in addition to the required features and
targets in estimators, scorers and transformers.
**biquality-learn** uses this design to integrate the *sample_quality*
property into the training and prediction process of biquality learning
algorithms. It allows one to use **biquality-learn**\ ’s algorithms in a
similar way to **scikit-learn**\ ’s algorithms by passing the
*sample_quality* property as an additional argument to the :meth:`fit`,
:meth:`predict`, and other methods.

Currently, the main components provided by **scikit-learn** support this
design and is already usable for cross-validators. However, it will be
extended to all components in the future, and **biquality-learn** will
significantly benefit from many “free” features. When
https://github.com/scikit-learn/scikit-learn/pull/24250 will be merged,
it will be possible to make a bagging ensemble of biquality classifiers
thanks to the :class:`sklearn.ensemble.BaggingClassifier` without
overriding its behavior on biquality data.

.. code:: python 

   from sklearn.ensemble import BaggingClassifier

   bag = BaggingClassifier(kkmm).fit(X, y, sample_quality=sample_quality)

Cross-Validating Biquality Classifiers
======================================

Any cross-validators working for usual Supervised Learning can work in
the case of Biquality Learning. However, when splitting the data into a
train and test set, untrusted samples need to be removed from the test
set to avoid computing supervised metrics on corrupted labels. That is
why :func:`bqlearn.model_selection.BiqualityCrossValidator` is provided
by **biquality-learn** to accommodate any **scikit-learn** compatible
cross-validators to biquality data.

Here is an example of how to use **scikit-learn**\ ’s
:class:`sklearn.model_selection.RandomizedSearchCV` function
to perform hyperparameter validation for a
biquality learning algorithm in **biquality-learn**:

.. code:: python 

   from sklearn.model_selection import RandomizedSearchCV
   from sklearn.utils.fixes import loguniform
   from bqlearn.model_selection import BiqualityCrossValidator

   param_dist = {"final_estimator__C": loguniform(1e3, 1e5)}
   n_iter=20

   random_search = RandomizedSearchCV(
      kkmm,
      param_distributions=param_dist,
      n_iter=n_iter,
      cv=BiqualityCrossValidator(cv=3)
   )
   random_search.fit(X, y, groups=sample_quality, sample_quality=sample_quality)

.. topic:: References

 .. [NDRT2013] N. Natarajan, I. S. Dhillon, P. Ravikumar, and A. Tewari, "Learning with Noisy Labels", NeurIPS, 2013.

 .. [ZLA2021] M. Zhang, J. Lee, and S. Agarwal. "Learning from noisy labels with no change to the training process.", ICML, 2021.

 .. [LT2015] T. Liu and D. Tao, "Classification with noisy labels by importance reweighting.", in IEEE Transactions on pattern analysis and machine intelligence, 2015

 .. [DYXY2008] Wenyuan Dai, Qiang Yang, Gui-Rong Xue, Yong Yu. "Boosting for Transfer Learning", 2007.
 
 .. [NLBC2021] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols, "Importance Reweighting for Biquality Learning", IJCNN, 2021.

 .. [FNS2020]  Fang, T., Lu, N., Niu, G., and Sugiyama, M. "Rethinking importance weighting for deep learning under distribution shift.", NeurIPS 2020

 .. [RMW2015] B. Rooyen, A. Menon and R. Williamson. "Learning with Symmetric Label Noise: The Importance of Being Unhinged.", NeurIPS, 2015

 .. [H2007] Daumé III, Hal. "Frustratingly Easy Domain Adaptation."
            Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics. 2007.

 .. [L2018] Jiang, Lu, et al. "Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels." International conference on machine learning. PMLR, 2018.

 .. [NLBC2023] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols, "Biquality Learning: a Framework to Design Algorithms Dealing with Closed-Set Distribution Shifts.", Machine Learning, 2023.