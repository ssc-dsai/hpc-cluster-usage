WSGE
----

Some tools to make pretty outputs of GridEngine job information. A few of
the new commands:

* ``cluster_stat``

  * An overview of jobs currently running on the cluster and who is
    running them.

* ``wstat``

  * A tweaked qstat that shows the full job name and has age of jobs instead
    of time.


Feature Requests
================

The scripts are still in early development. Use the issue tracker on
bitbucket https://bitbucket.org/tdaff/wsge/issues to make requests for new
features.

Installation
============

Setup as a Python package so that the modules are in your ``PYTHONPATH`` and
individual scripts are in your ``PATH``.

.. code-block::

    python setup.py install