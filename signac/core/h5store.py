# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"Data store implementation with backend HDF5 file."
import os
import logging

from ..common import six

if six.PY2:
    from collections import Mapping
    from collections import MutableMapping
else:
    from collections.abc import Mapping
    from collections.abc import MutableMapping


logger = logging.getLogger(__name__)


def h5set(grp, key, value):
    if key in grp:
        del grp[key]
    if isinstance(value, Mapping):
        subgrp = grp.create_group(key)
        for k, v in value.items():
            h5set(subgrp, k, v)
    elif value is None:
        grp.create_dataset(key, data=None, shape=None, dtype='f')
    else:
        grp[key] = value


def h5get(grp, key):
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


class H5Group(MutableMapping):

    def __init__(self, group):
        self._group = group

    def __getitem__(self, key):
        return h5get(self._group, key)

    def __setitem__(self, key, value):
        h5set(self._group, key, value)

    def __delitem__(self, key):
        del self._group[key]

    def __iter__(self):
        # The generator below should be refactored to use 'yield from'
        # once we drop Python 2.7 support.
        for key in self._group.keys():
            yield key

    def __len__(self):
        return len(self._group)


class H5Store(MutableMapping):

    def __init__(self, filename):
        assert isinstance(filename, six.string_types) and len(filename) > 0, \
            'H5Store filename must be a non-empty string.'
        self._filename = os.path.realpath(filename)
        self._file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        self._load()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def _load(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self._filename, libver='latest', swmr=True)

    def close(self):
        try:
            self._file.close()
        except AttributeError:
            pass

    def __getitem__(self, key):
        self._load()
        return h5get(self._file, key)

    def __setitem__(self, key, value):
        self._load()
        h5set(self._file, key, value)

    def __delitem__(self, key):
        self._load()
        del self._file[key]

    def __iter__(self):
        self._load()
        # The generator below should be refactored to use 'yield from'
        # once we drop Python 2.7 support.
        for key in self._file.keys():
            yield key

    def __len__(self):
        self._load()
        return len(self._file)
