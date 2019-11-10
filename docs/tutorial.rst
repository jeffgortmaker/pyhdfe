Tutorial
========

This section uses a series of `Jupyter Notebooks <https://jupyter.org/>`_ to demonstrate how pyhdfe can be used together with regression routines from other packages. Each notebook employs the Frisch-Waugh-Lovell (FWL) theorem of :ref:`references:Frisch and Waugh (1933)` and :ref:`references:Lovell (1963)` to run a fixed effects regression by residualizing (projecting) the variables of interest.

This tutorial is just meant to demonstrate how pyhdfe can be used in the simplest of applications. For detailed information about the different algorithms supported by pyhdfe, refer to :doc:`API Documentation </api>`.

.. toctree::
   :maxdepth: 2

   _notebooks/sklearn.ipynb
   _notebooks/statsmodels.ipynb
