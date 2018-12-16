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

try:
    import pandas as pd

    def _group_is_pandas_type(group):
        return 'pandas_type' in group.attrs

    def _is_pandas_type(value):
        return isinstance(value, pd.core.generic.PandasObject)

except ImportError:

    def _is_pandas_object(value):
        return False

    def _group_is_pandas_type(group):
        if 'pandas_type' in group.attrs:
            logger.warning(
                "The object stored under group '{}' appears to be a pandas object, but pandas "
                "is not installed!".format(group))


def _requires_pytables():
    try:
        import tables  # noqa
    except ImportError:
        raise ImportError(
            "Storing and loading pandas objects requires the PyTables package.")


logger = logging.getLogger(__name__)


def _h5set(file, grp, key, value, path=''):
    """Set a key in an h5py container, recursively converting Mappings to h5py
    groups and transparently handling None."""
    path = path + '/' + key if path else key

    if key in grp:
        del grp[key]
    if isinstance(value, Mapping):
        subgrp = grp.create_group(key)
        for k, v in value.items():
            _h5set(file, subgrp, k, v, path)
    elif value is None:
        grp.create_dataset(key, data=None, shape=None, dtype='f')
    elif _is_pandas_type(value):
        _requires_pytables()
        file.close()
        with pd.HDFStore(file._filename) as store:
            store[path] = value
        file.open()
    else:
        grp[key] = value


def _h5get(file, grp, key, path=''):
    """Retrieve the underlying data for a key from its h5py container."""
    path = path + '/' + key if path else key
    result = grp[key]

    if _group_is_pandas_type(result):
        _requires_pytables()
        grp.file.flush()
        with pd.HDFStore(grp.file.filename) as store:
            return store[path]

    try:
        shape = result.shape
        if shape is None:
            return None
        else:
            return result.value
    except AttributeError:
        if isinstance(result, MutableMapping):
            return H5Group(result, file, path)
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
    _PROTECTED_KEYS = ('_file', '_path')

    def __init__(self, group, file, path):
        self._file = file
        self._path = path

    @property
    def _group(self):
        return self._file._file[self._path]

    def __getitem__(self, key):
        return _h5get(self._file, self._group, key, self._path)

    def __setitem__(self, key, value):
        _h5set(self._file, self._group, key, value, self._path)
        return value

    def __delitem__(self, key):
        del self._group[key]

    def __getattr__(self, name):
        if name in self._group.keys():
            return self.__getitem__(name)
        else:
            return getattr(self._group, name)

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

    def __eq__(self, other):
        if type(other) == type(self):
            return self._group == other._group
        else:
            return super(H5Group, self).__eq__(other)


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

    def _ensure_open(self):
        """Raises an error if the datastore is not open.

        :raises RuntimeError: If the datastore is not open.
        """
        if self._file is None:
            raise RuntimeError(
                "To access data in an H5Store, it must be open. "
                "Use `with` or call open() and close() explicitly.")

    def open(self):
        """Opens the datastore file."""
        if self._file is None:
            import h5py
            self._file = h5py.File(self._filename)
        return self

    def close(self):
        """Closes the datastore file."""
        try:
            self._file.close()
            self._file = None
        except AttributeError:  # If _file is None, AttributeError is raised
            pass

    def flush(self):
        self._file.flush()

    def __getitem__(self, key):
        key = key if key.startswith('/') else '/' + key
        self._ensure_open()
        return _h5get(self, self._file, key)

    def __setitem__(self, key, value):
        self._ensure_open()
        _h5set(self, self._file, _validate_key(key), value)
        return value

    def __delitem__(self, key):
        self._ensure_open()
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
        self._ensure_open()
        # The generator below should be refactored to use 'yield from'
        # once we drop Python 2.7 support.
        for key in self._file.keys():
            yield key

    def __len__(self):
        self._ensure_open()
        return len(self._file)
