# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"Data store implementation with backend HDF5 file."
import logging
import os
import warnings

from ..common import six

if six.PY2:
    from collections import Mapping
    from collections import MutableMapping
else:
    from collections.abc import Mapping
    from collections.abc import MutableMapping


logger = logging.getLogger(__name__)


def _h5set(grp, key, value):
    """Set a key in an h5py container, recursively converting Mappings to h5py
    groups and transparently handling None."""

    if key in grp:
        del grp[key]
    if isinstance(value, Mapping):
        subgrp = grp.create_group(key)
        for k, v in value.items():
            _h5set(subgrp, k, v)
    elif value is None:
        grp.create_dataset(key, data=None, shape=None, dtype='f')
    else:
        grp[key] = value


def _h5get(grp, key):
    """Retrieve the underlying data for a key from its h5py container."""

    result = grp[key]
    try:
        shape = result.shape
        if shape is None:
            return None
        else:
            return result.value
    except AttributeError:
        if isinstance(result, MutableMapping):
            return H5Group(result)
        else:
            return result


def _validate_key(key):
    "Emit a warning or raise an exception if key is invalid. Returns key."
    if '.' in key:
        from ..warnings import SignacDeprecationWarning
        warnings.warn(
            "\nThe use of '.' (dots) in keys is deprecated and may lead to "
            "unexpected behavior!\nSee http://www.signac.io/document-wide-migration/ "
            "for a recipe on how to replace dots in all keys.",
            SignacDeprecationWarning)
    return key


class H5Group(MutableMapping):
    """An abstraction layer over h5py's Group objects, to manage and return data."""
    _PROTECTED_KEYS = ('_group')

    def __init__(self, group):
        self._group = group

    def __getitem__(self, key):
        return _h5get(self._group, key)

    def __setitem__(self, key, value):
        _h5set(self._group, key, value)

    def __delitem__(self, key):
        del self._group[key]

    def __getattr__(self, name):
        try:
            return super(H5Group, self).__getattribute__(name)
        except AttributeError:
            if name.startswith('__'):
                raise
            try:
                return self.__getitem__(name)
            except KeyError as e:
                raise AttributeError(e)

    def __setattr__(self, key, value):
        if key.startswith('__') or key in self.__getattribute__('_PROTECTED_KEYS'):
            super(H5Group, self).__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __iter__(self):
        # The generator below should be refactored to use 'yield from'
        # once we drop Python 2.7 support.
        for key in self._group.keys():
            yield key

    def __len__(self):
        return len(self._group)


class H5Store(MutableMapping):
    """An HDF5 backend for storing arbitrary array-like and dictionary-like data.
    Both attribute-based and index-based operations are supported.

    Example:

    .. code-block:: python

        with H5Store('file.h5') as h5s:
            h5s['foo'] = 'bar'
            assert h5s.foo == 'bar'

    """
    _PROTECTED_KEYS = ('_filename', '_file')

    def __init__(self, filename):
        if not (isinstance(filename, six.string_types) and len(filename) > 0):
            raise ValueError('H5Store filename must be a non-empty string.')
        self._filename = os.path.realpath(filename)
        self._file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def ensure_open(self):
        if self._file is None:
            raise RuntimeError("To access data in an H5Store, it must be open. "
            "Use `with` or call open() and close() explicitly.")

    def open(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self._filename)
        return self

    def close(self):
        try:
            self._file.close()
            self._file = None
        except AttributeError:  # If _file is None, AttributeError is raised
            pass

    def __getitem__(self, key):
        self.ensure_open()
        return _h5get(self._file, key)

    def __setitem__(self, key, value):
        self.ensure_open()
        _h5set(self._file, _validate_key(key), value)

    def __delitem__(self, key):
        self.ensure_open()
        del self._file[key]

    def __getattr__(self, name):
        try:
            return super(H5Store, self).__getattribute__(name)
        except AttributeError:
            if name.startswith('__') or name in self.__getattribute__('_PROTECTED_KEYS'):
                raise
            try:
                return self.__getitem__(name)
            except KeyError as e:
                raise AttributeError(e)

    def __setattr__(self, key, value):
        if key.startswith('__') or key in self.__getattribute__('_PROTECTED_KEYS'):
            super(H5Store, self).__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __iter__(self):
        self.ensure_open()
        # The generator below should be refactored to use 'yield from'
        # once we drop Python 2.7 support.
        for key in self._file.keys():
            yield key

    def __len__(self):
        self.ensure_open()
        return len(self._file)
