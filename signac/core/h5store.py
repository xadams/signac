# Copyright (c) 2017 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
"Data store implementation with backend HDF5 file."
import os
import logging
import h5py

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
        grp[key] = h5py.Empty('f')
    else:
        grp[key] = value


def h5get(grp, key):
    result = grp[key]
    if isinstance(result, h5py._hl.dataset.Dataset):
        if isinstance(result.value, h5py._hl.base.Empty):
            return None
        else:
            return result.value
    elif isinstance(result, h5py._hl.group.Group):
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
        yield from self._group.keys()

    def __len__(self):
        return len(self._group)


class H5Store(MutableMapping):

    def __init__(self, filename=None):
        self._filename = None if filename is None else os.path.realpath(filename)
        self._file = None

    def __del__(self):
        self._close()

    def __enter__(self):
        self._load()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._close()

    def _load(self):
        assert self._filename is not None
        if self._file is None:
            self._file = h5py.File(self._filename, libver='latest', swmr=True)

    def _close(self):
        if self._file is not None:
            self._file.close()

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
        yield from self._file.keys()

    def __len__(self):
        self._load()
        return len(self._file)
