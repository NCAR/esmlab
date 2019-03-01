Release Procedure
-----------------

Our current policy for releasing is to aim for a release every two weeks. 
The idea is to get fixes and new features out instead of trying to cram a ton of 
features into a release and by consequence taking a lot of time to make a new one.

#. Create a new branch ``release-yyyy.mm.dd`` with the version for the release 

 * Update `CHANGELOG.rst` 
 * Make sure all new changes, features are reflected in the documentation.

#. Open a new pull request for this branch targeting `master` 


#. After all tests pass and the PR has been approved, merge the PR into ``master`` 

#. Tag a release and push to github::

    $ git tag -a v2019.1.1 -m "Version 2019.1.1"
    $ git push origin master --tags

#. Build and publish release on PyPI::

    $ git clean -xfd  # remove any files not checked into git
    $ python setup.py sdist bdist_wheel --universal  # build package
    $ twine upload dist/*  # register and push to pypi

#. Update esmlab conda-forge feedstock

 * Fork `esmlab-feedstock repository <https://github.com/conda-forge/esmlab-feedstock>`_ 
 * Clone this fork and edit recipe::

        $ git clone git@github.com:username/esmlab-feedstock.git
        $ cd esmlab-feedstock
        $ cd recipe
        $ # edit meta.yaml 

 - Update version 
 - Get sha256 from pypi.org for `esmlab <https://pypi.org/project/esmlab/#files>`_
 - Fill in the rest of information as described `here <https://github.com/conda-forge/esmlab-feedstock#updating-esmlab-feedstock>`_

 * Commit and submit a PR


