.. _bqlearn_api_reference:

=============
API Reference
=============

The complete ``biquality-learn`` project is automatically documented for every module.

Biquality Classifiers
=====================

.. currentmodule:: bqlearn

.. autosummary::
   :toctree: api
   :template: class.rst

   baseline.BiqualityBaseline
   density_ratio.KKMM
   density_ratio.KPDR
   density_ratio.IKMM
   density_ratio.IPDR
   irbl.IRBL
   irlnl.IRLNL
   unbiased.LossCorrection
   plugin.PluginCorrection
   tradaboost.TrAdaBoostClassifier
   ea.EasyADAPT
   unhinged.LinearUnhinged
   unhinged.KernelUnhinged

Transition Matrix Estimators
----------------------------

.. currentmodule:: bqlearn.metrics

.. autosummary::
   :toctree: api
   :template: functions.rst

   anchor_transition_matrix
   iterative_anchor_transition_matrix
   gold_transition_matrix

Biquality Cross-Validation
--------------------------

.. currentmodule:: bqlearn.model_selection

.. autosummary::
   :toctree: api
   :template: functions.rst

   BiqualityCrossValidator

Corruptions
===========

.. currentmodule:: bqlearn.corruptions

.. autosummary::
   :toctree: api
   :template: functions.rst

   make_label_noise
   make_instance_dependent_label_noise
   uncertainty_noise_probability
   noisy_leaves_probability
   make_weak_labels
   make_feature_dependent_label_noise
   make_imbalance
   make_cluster_imbalance
   make_sampling_biais

Noise Matrices
--------------

.. currentmodule:: bqlearn.corruptions.noise_matrices

.. autosummary::
   :toctree: api
   :template: functions.rst

   uniform_noise_matrix
   flip_noise_matrix
   background_noise_matrix

