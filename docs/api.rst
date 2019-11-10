.. currentmodule:: pyhdfe


API Documentation
=================

Algorithms for absorbing fixed effects should be created with the following function.

.. autosummary::
   :toctree: _api

   create

Algorithm classes contain information about the fixed effects.

.. autosummary::
   :nosignatures:
   :toctree: _api
   :template: class_without_signature.rst

   Algorithm

They can be used to absorb fixed effects (i.e., residualize matrices).

.. autosummary::
   :toctree: _api

   Algorithm.residualize
