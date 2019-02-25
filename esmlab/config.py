""" The configuration script: set global settings.
"""

from __future__ import absolute_import, print_function

import os

import yaml

GRIDFILE_DIRECTORY = "gridfile_directory"
GRID_DEFITIONS_FILE = "grid_defitions_file"

_here = os.path.abspath(os.path.dirname(__file__))
_config_dir = os.path.join(os.path.expanduser("~"), ".esmlab")
_path_config_yml = os.path.join(_config_dir, "config.yml")

if os.path.exists(".config-esmlab.yml"):
    _path_config_yml = os.path.join(".config-esmlab.yml")

SETTINGS = {
    GRIDFILE_DIRECTORY: os.path.join(_config_dir, "esmlab-grid-files"),
    GRID_DEFITIONS_FILE: os.path.join(_here, "grid_definitions.yml"),
}

for key in [GRIDFILE_DIRECTORY]:
    if not os.path.exists(SETTINGS[key]):
        os.makedirs(SETTINGS[key])


def _check_path_write_access(value):
    value = os.path.abspath(os.path.expanduser(value))
    if os.path.exists(value):
        if not os.access(value, os.W_OK):
            print("no write access to: {0}".format(value))
            return False
        return True

    try:
        os.makedirs(value)
        return True
    except (OSError, PermissionError) as err:
        print("could not make directory: {0}".format(value))
        raise err


def _full_path(value):
    return os.path.abspath(os.path.expanduser(value))


def _check_exists(value):
    return os.path.exists(value)


_VALIDATORS = {GRIDFILE_DIRECTORY: _check_path_write_access, GRID_DEFITIONS_FILE: _check_exists}

_SETTERS = {GRIDFILE_DIRECTORY: _full_path}


class set_options(object):
    """Set configurable settings."""

    def __init__(self, **kwargs):
        self.old = {}
        for key, val in kwargs.items():
            if key not in SETTINGS:
                raise ValueError(
                    "{key} is not in the set of valid settings:\n {set}".format(
                        key=key, set=set(SETTINGS)
                    )
                )
            if key in _VALIDATORS and not _VALIDATORS[key](val):
                raise ValueError("{val} is not a valid value for {key}".format(key=key, val=val))
            self.old[key] = SETTINGS[key]
        self._apply_update(kwargs)

    def _apply_update(self, settings_dict):
        for key, val in settings_dict.items():
            if key in _SETTERS:
                settings_dict[key] = _SETTERS[key](val)
        SETTINGS.update(settings_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)


def get_options():
    return SETTINGS


if os.path.exists(_path_config_yml):
    with open(_path_config_yml) as f:
        dot_file_settings = yaml.load(f)
    if dot_file_settings:
        set_options(**dot_file_settings)
