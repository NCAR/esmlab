''' The configuration script: set global settings. '''

from __future__ import absolute_import, print_function

import ast
import os
import threading
from collections import Mapping

import yaml

no_default = '__no_default__'

paths = [os.path.join(os.path.expanduser('~'), '.esmlab')]

if 'ESMLAB_CONFIG' in os.environ:
    PATH = os.environ['ESMLAB_CONFIG']
    paths.append(PATH)

elif os.path.exists(os.path.join(os.getcwd(), '.esmlab')):
    PATH = os.path.join(os.getcwd(), '.esmlab')
    paths.append(PATH)

else:
    PATH = os.path.join(os.path.expanduser('~'), '.esmlab')


global_config = config = {}

config_lock = threading.Lock()

defaults = []


def update(old, new, priority='new'):
    """ Update a nested dictionary with values from another
    This is like dict.update except that it smoothly merges nested values
    This operates in-place and modifies old
    Parameters
    ----------
    priority: string {'old', 'new'}
        If new (default) then the new dictionary has preference.
        Otherwise the old dictionary does.

    """

    for k, v in new.items():
        if k not in old and isinstance(v, Mapping):
            old[k] = {}
        if isinstance(v, Mapping):
            if old[k] is None:
                old[k] = {}

            update(old[k], v, priority=priority)

        else:
            if priority == 'new' or k not in old:
                old[k] = v

    return old


def merge(*dicts):
    """ Update a sequence of nested dictionaries

    This prefers the values in the latter dictionaries to those in the former
    """

    result = {}
    for d in dicts:
        update(result, d)
    return result


def normalize_key(key):
    """ Replaces underscores with hyphens in string keys

    Parameters
    ----------
    key : string, int, or float
        Key to assign.
    """
    if isinstance(key, str):
        key = key.replace('_', '-')
    return key


def normalize_nested_keys(config):
    """ Replaces underscores with hyphens for keys for a nested Mapping

    Examples
    --------
    >>> a = {'x': 1, 'y_1': {'a_2': 2}}
    >>> normalize_nested_keys(a)
    {'x': 1, 'y-1': {'a-2': 2}}
    """
    config_norm = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            value = normalize_nested_keys(value)
        key_norm = normalize_key(key)
        config_norm[key_norm] = value

    return config_norm


def collect_yaml(paths=paths):
    """ Collect configuration from yaml files

    This searches through a list of paths, expands to find all yaml or json
    files, and then parses each file.
    """
    # Find all paths
    file_paths = []
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    file_paths.extend(
                        sorted(
                            [
                                os.path.join(path, p)
                                for p in os.listdir(path)
                                if os.path.splitext(p)[1].lower() in ('.yaml', '.yml')
                            ]
                        )
                    )
                except OSError:
                    # Ignore permission errors
                    pass
            else:
                file_paths.append(path)

    configs = []

    # Parse yaml files
    for path in file_paths:
        try:
            with open(path) as f:
                data = yaml.safe_load(f.read()) or {}
                data = normalize_nested_keys(data)
                data = expand_environment_variables(data)
                configs.append(data)
        except (OSError, IOError):
            # Ignore permission errors
            pass

    return configs


def collect_env(env=None):
    """ Collect config from environment variables

    This grabs environment variables of the form "ESMLAB_FOO__BAR_BAZ=123" and
    turns these into config variables of the form ``{"foo": {"bar-baz": 123}}``
    It transforms the key and value in the following way:

    -  Lower-cases the key text
    -  Treats ``__`` (double-underscore) as nested access
    -  Replaces ``_`` (underscore) with a hyphen.
    -  Calls ``ast.literal_eval`` on the value
    """

    if env is None:
        env = os.environ

    d = {}
    for name, value in env.items():
        if name.startswith('ESMLAB_'):
            varname = name[7:].lower().replace('__', '.')
            varname = normalize_key(varname)
            try:
                d[varname] = ast.literal_eval(value)

            except (SyntaxError, ValueError):
                d[varname] = value

    result = {}
    set(d, config=result)
    return result


def ensure_file(source, destination=None, comment=True):
    """
    Copy file to default location if it does not already exist

    This tries to move a default configuration file to a default location if
    if does not already exist.  It also comments out that file by default.

    This is to be used by downstream modules (like dask.distributed) that may
    have default configuration files that they wish to include in the default
    configuration path.

    Parameters
    ----------
    source : string, filename
        Source configuration file, typically within a source directory.
    destination : string, directory
        Destination directory. Configurable by ``ESMLAB_CONFIG`` environment
        variable, falling back to ~/.esmlab.
    comment : bool, True by default
        Whether or not to comment out the config file when copying.
    """
    if destination is None:
        destination = PATH

    # destination is a file and already exists, never overwrite
    if os.path.isfile(destination):
        return

    # If destination is not an existing file, interpret as a directory,
    # use the source basename as the filename
    directory = destination
    destination = os.path.join(directory, os.path.basename(source))

    try:
        if not os.path.exists(destination):
            if not os.path.isdir(directory):
                os.makedirs(directory)
            else:
                pass

            # Atomically create destination.  Parallel testing discovered
            # a race condition where a process can be busy creating the
            # destination while another process reads an empty config file.
            tmp = '%s.tmp.%d' % (destination, os.getpid())
            with open(source) as f:
                lines = list(f)

            if comment:
                lines = [
                    '# ' + line if line.strip() and not line.startswith('#') else line
                    for line in lines
                ]

            with open(tmp, 'w') as f:
                f.write(''.join(lines))

            try:
                os.rename(tmp, destination)
            except OSError:
                os.remove(tmp)
    except OSError:
        pass


class set(object):
    """ Temporarily set configuration values within a context manager

    Examples
    --------
    >>> import esmlab
    >>> with esmlab.config.set({'foo': 123}):
    ...     pass

    See Also
    --------
    esmlab.config.get
    """

    def __init__(self, arg=None, config=config, lock=config_lock, **kwargs):
        if arg and not kwargs:
            kwargs = arg

        with lock:
            self.config = config
            self.old = {}

            for key, value in kwargs.items():
                self._assign(key.split('.'), value, config, old=self.old)

    def __enter__(self):
        return self.config

    def __exit__(self, type, value, traceback):
        for keys, value in self.old.items():
            if value == '--delete--':
                d = self.config
                try:
                    while len(keys) > 1:
                        d = d[keys[0]]
                        keys = keys[1:]
                    del d[keys[0]]
                except KeyError:
                    pass
            else:
                self._assign(keys, value, self.config)

    @classmethod
    def _assign(cls, keys, value, d, old=None, path=[]):
        """ Assign value into a nested configuration dictionary

        Optionally record the old values in old

        Parameters
        ----------
        keys: Sequence[str]
            The nested path of keys to assign the value, similar to toolz.put_in
        value: object
        d: dict
            The part of the nested dictionary into which we want to assign the
            value
        old: dict, optional
            If provided this will hold the old values
        path: List[str]
            Used internally to hold the path of old values
        """
        key = normalize_key(keys[0])
        if len(keys) == 1:
            if old is not None:
                path_key = tuple(path + [key])
                if key in d:
                    old[path_key] = d[key]
                else:
                    old[path_key] = '--delete--'
            d[key] = value
        else:
            if key not in d:
                d[key] = {}
                if old is not None:
                    old[tuple(path + [key])] = '--delete--'
                old = None
            cls._assign(keys[1:], value, d[key], path=path + [key], old=old)


def collect(paths=paths, env=None):
    """
    Collect configuration from paths and environment variables

    Parameters
    ----------
    paths : List[str]
        A list of paths to search for yaml config files

    env : dict
        The system environment variables

    Returns
    -------
    config: dict

    See Also
    --------
    esmlab.config.refresh: collect configuration and update into primary config
    """
    if env is None:
        env = os.environ
    configs = []

    if yaml:
        configs.extend(collect_yaml(paths=paths))

    configs.append(collect_env(env=env))

    return merge(*configs)


def refresh(config=config, defaults=defaults, **kwargs):
    """
    Update configuration by re-reading yaml files and env variables

    This mutates the global esmlab.config.config, or the config parameter if
    passed in.

    This goes through the following stages:

    1.  Clearing out all old configuration
    2.  Updating from the stored defaults from downstream libraries
        (see update_defaults)
    3.  Updating from yaml files and environment variables

    Note that some functionality only checks configuration once at startup and
    may not change behavior, even if configuration changes.  It is recommended
    to restart your python process if convenient to ensure that new
    configuration changes take place.

    See Also
    --------
    esmlab.config.collect: for parameters
    esmlab.config.update_defaults
    """
    config.clear()

    for d in defaults:
        update(config, d, priority='old')

    update(config, collect(**kwargs))


def get(key, default=no_default, config=config):
    """
    Get elements from global config

    Use '.' for nested access

    Examples
    --------
    >>> from esmlab import config
    >>> config.get('foo')  # doctest: +SKIP
    {'x': 1, 'y': 2}

    >>> config.get('foo.x')  # doctest: +SKIP
    1

    >>> config.get('foo.x.y', default=123)  # doctest: +SKIP
    123

    See Also
    --------
    esmlab.config.set
    """
    keys = key.split('.')
    result = config
    for k in keys:
        k = normalize_key(k)
        try:
            result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


def rename(aliases, config=config):
    """ Rename old keys to new keys

    This helps migrate older configuration versions over time
    """
    old = list()
    new = dict()
    for o, n in aliases.items():
        value = get(o, None, config=config)
        if value is not None:
            old.append(o)
            new[n] = value

    for k in old:
        del config[k]  # TODO: support nested keys

    set(new, config=config)


def update_defaults(new, config=config, defaults=defaults):
    """ Add a new set of defaults to the configuration

    It does two things:

    1.  Add the defaults to a global collection to be used by refresh later
    2.  Updates the global config with the new configuration
        prioritizing older values over newer ones
    """
    defaults.append(new)
    update(config, new, priority='old')


def expand_environment_variables(config):
    ''' Expand environment variables in a nested config dictionary

    This function will recursively search through any nested dictionaries
    and/or lists.

    Parameters
    ----------
    config : dict, iterable, or str
        Input object to search for environment variables

    Returns
    -------
    config : same type as input

    Examples
    --------
    >>> expand_environment_variables({'x': [1, 2, '$USER']})  # doctest: +SKIP
    {'x': [1, 2, 'my-username']}
    '''
    if isinstance(config, Mapping):
        return {k: expand_environment_variables(v) for k, v in config.items()}
    elif isinstance(config, str):
        return os.path.expanduser(os.path.expandvars(config))
    elif isinstance(config, (list, tuple, set)):
        return type(config)([expand_environment_variables(v) for v in config])
    else:
        return config


fn = os.path.join(os.path.dirname(__file__), 'config.yaml')
ensure_file(source=fn, comment=False)

with open(fn) as f:
    defaults = yaml.safe_load(f)

refresh()
