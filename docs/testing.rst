Testing
=======

Testing is done with the `tox <https://tox.readthedocs.io/en/latest/>`_ automation tool, which runs a `pytest <https://docs.pytest.org/en/latest/>`_-backed test suite in the ``tests`` module. This `FAQ <https://tox.readthedocs.io/en/latest/developers.html>`_ contains some useful information about how to use tox on Windows.


Testing Requirements
--------------------

In addition to the installation requirements for the package itself, running tests and building documentation requires additional packages specified by the ``tests`` and ``docs`` extras in ``setup.py``, along with any other explicitly specified ``deps`` in ``tox.ini``.


Running Tests
-------------

Defined in ``tox.ini`` are environments that test the package under different python versions, check types, enforce style guidelines, verify the integrity of the documentation, and release the package. The following command can be run in the top-level pyfwl directory to run all testing environments::

    tox

You can choose to run only one environment, such as the one that builds the documentation, with the ``-e`` flag::

    tox -e docs


Test Organization
-----------------

Fixtures, which are defined in ``tests.conftest``, configure the testing environment and load data according to a range of specifications.

Tests in ``tests.test_hdfe`` verify that different algorithms yield the same solutions.
