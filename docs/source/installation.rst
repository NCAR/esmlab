.. highlight:: shell

============
Installation
============


Conda
-------

To install esmlab, run this command in your terminal:

.. code-block:: console

    $ conda install -c cisl-iowa esmlab

This is the preferred method to install esmlab, as it will always install
the most recent stable release.

Pip
----

Support is coming soon, please use conda for the time being.


From sources
------------
You can install specific git commit/tag with pip by running:

.. code-block:: console

    $ pip install git+git://github.com/NCAR/esmlab.git


The sources for esmlab can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/NCAR/esmlab

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/NCAR/esmlab/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/NCAR/esmlab
.. _tarball: https://github.com/NCAR/esmlab/tarball/master
