Installing
==========

You can install esmlab with ``pip``, ``conda``, or by installing from source.

Pip
---

Pip can be used to install esmlab::

   pip install esmlab

Conda
-----

To install the latest version of esmlab from the
`conda-forge <https://conda-forge.github.io/>`_ repository using
`conda <https://www.anaconda.com/downloads>`_::

    conda install -c conda-forge esmlab

Install from Source
-------------------

To install esmlab from source, clone the repository from `github
<https://github.com/NCAR/esmlab>`_::

    git clone https://github.com/NCAR/esmlab.git
    cd esmlab
    pip install -e .

You can also install directly from git master branch::

    pip install git+https://github.com/NCAR/esmlab


Test
----

To run esmlab's tests with ``pytest``::

    git clone https://github.com/NCAR/esmlab.git
    cd esmlab
    pytest - v
