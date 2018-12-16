# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
import os
import unittest
import uuid
import string
from itertools import chain
from array import array

from signac.core.h5store import H5Store
from signac.common import six

if six.PY2:
    from tempdir import TemporaryDirectory
    from collections import Mapping
else:
    from tempfile import TemporaryDirectory
    from collections.abc import Mapping

try:
    import h5py    # noqa
    H5PY = True
except ImportError:
    H5PY = False

try:
    import pandas   # noqa
    import tables   # noqa
    PANDAS_AND_TABLES = True
except ImportError:
    PANDAS_AND_TABLES = False

try:
    import numpy    # noqa
    NUMPY = True
except ImportError:
    NUMPY = False

FN_STORE = 'signac_test_h5store.h5'


class BaseH5StoreTest(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = TemporaryDirectory(prefix='signac_test_h5store_')
        self._fn_store = os.path.join(self._tmp_dir.name, FN_STORE)
        self.addCleanup(self._tmp_dir.cleanup)


@unittest.skipIf(not H5PY, 'test requires the h5py package')
class H5StoreTest(BaseH5StoreTest):

    def get_h5store(self):
        return H5Store(filename=self._fn_store)

    def get_testdata(self):
        return str(uuid.uuid4())

    def test_init(self):
        self.get_h5store()

    def test_invalid_filenames(self):
        with self.assertRaises(ValueError):
            H5Store(None)
        with self.assertRaises(ValueError):
            H5Store('')
        with self.assertRaises(ValueError):
            H5Store(123)

    def test_access_closed_file(self):
        h5s = self.get_h5store()
        with self.assertRaises(RuntimeError):
            h5s['a']
        with self.assertRaises(RuntimeError):
            h5s['a'] = 'b'
        with self.assertRaises(RuntimeError):
            del h5s['a']
        with self.assertRaises(RuntimeError):
            for key in h5s['a']:
                pass
        with self.assertRaises(RuntimeError):
            len(h5s)

    def test_set_get(self):
        with self.get_h5store() as h5s:
            key = 'setget'
            d = self.get_testdata()
            h5s.clear()
            self.assertFalse(bool(h5s))
            self.assertEqual(len(h5s), 0)
            self.assertNotIn(key, h5s)
            with self.assertRaises(KeyError):
                h5s[key]
            d_ = h5s[key] = d
            self.assertEqual(d_, d)
            self.assertTrue(bool(h5s))
            self.assertEqual(len(h5s), 1)
            self.assertIn(key, h5s)
            self.assertEqual(h5s[key], d)
            self.assertEqual(h5s.get(key), d)
            self.assertEqual(h5s.get('nonexistent', 'default'), 'default')

    def test_set_get_explicit_nested(self):
        with self.get_h5store() as h5s:
            key = 'setgetexplicitnested'
            d = self.get_testdata()
            h5s.setdefault('a', dict())
            child1 = h5s['a']
            child2 = h5s['a']
            self.assertEqual(child1, child2)
            self.assertEqual(type(child1), type(child2))
            self.assertFalse(child1)
            self.assertFalse(child2)
            child1[key] = d
            self.assertTrue(child1)
            self.assertTrue(child2)
            self.assertIn(key, child1)
            self.assertIn(key, child2)
            self.assertEqual(child1, child2)
            self.assertEqual(child1[key], d)
            self.assertEqual(child2[key], d)

    def test_copy_value(self):
        with self.get_h5store() as h5s:
            key = 'copy_value'
            key2 = 'copy_value2'
            d = self.get_testdata()
            self.assertNotIn(key, h5s)
            self.assertNotIn(key2, h5s)
            h5s[key] = d
            self.assertIn(key, h5s)
            self.assertEqual(h5s[key], d)
            self.assertNotIn(key2, h5s)
            h5s[key2] = h5s[key]
            self.assertIn(key, h5s)
            self.assertEqual(h5s[key], d)
            self.assertIn(key2, h5s)
            self.assertEqual(h5s[key2], d)

    def test_iter(self):
        with self.get_h5store() as h5s:
            key1 = 'iter1'
            key2 = 'iter2'
            d1 = self.get_testdata()
            d2 = self.get_testdata()
            d = {key1: d1, key2: d2}
            h5s.update(d)
            self.assertIn(key1, h5s)
            self.assertIn(key2, h5s)
            for i, key in enumerate(h5s):
                self.assertIn(key, d)
                self.assertEqual(d[key], h5s[key])
            self.assertEqual(i, 1)

    def test_delete(self):
        with self.get_h5store() as h5s:
            key = 'delete'
            d = self.get_testdata()
            h5s[key] = d
            self.assertEqual(len(h5s), 1)
            self.assertEqual(h5s[key], d)
            del h5s[key]
            self.assertEqual(len(h5s), 0)
            with self.assertRaises(KeyError):
                h5s[key]

    def test_update(self):
        with self.get_h5store() as h5s:
            key = 'update'
            d = {key: self.get_testdata()}
            h5s.update(d)
            self.assertEqual(len(h5s), 1)
            self.assertEqual(h5s[key], d[key])

    def test_clear(self):
        with self.get_h5store() as h5s:
            h5s.clear()
            key = 'clear'
            d = self.get_testdata()
            h5s[key] = d
            self.assertEqual(len(h5s), 1)
            self.assertEqual(h5s[key], d)
            h5s.clear()
            self.assertEqual(len(h5s), 0)

    def test_reopen(self):
        with self.get_h5store() as h5s:
            key = 'reopen'
            d = self.get_testdata()
            h5s[key] = d
        with self.get_h5store() as h5s:
            self.assertEqual(len(h5s), 1)
            self.assertEqual(h5s[key], d)

    def test_reopen_explicit_open_close(self):
        h5s = self.get_h5store().open()
        key = 'reopen'
        d = self.get_testdata()
        h5s[key] = d
        h5s.close()
        h5s.open()
        self.assertEqual(len(h5s), 1)
        self.assertEqual(h5s[key], d)
        h5s.close()

    def test_write_valid_types(self):
        with self.get_h5store() as h5s:
            valid_types = {
                'int': 123,
                'float': 123.456,
                'string': 'foobar',
                'none': None,
                'float_array': array('f', [-1.5, 0, 1.5]),
                'double_array': array('d', [-1.5, 0, 1.5]),
                'int_array': array('i', [-1, 0, 1]),
                'uint_array': array('I', [0, 1, 2]),
                'dict': {
                    'a': 1,
                    'b': None,
                    'c': 'test',
                },
            }
            for k, v in valid_types.items():
                h5s[k] = v
                if k.endswith('array'):
                    self.assertTrue((h5s[k] == v).all())
                else:
                    self.assertEqual(h5s[k], v)

    def test_write_invalid_type(self):
        class Foo(object):
            pass

        with self.get_h5store() as h5s:
            key = 'write_invalid_type'
            d = self.get_testdata()
            h5s[key] = d
            self.assertEqual(len(h5s), 1)
            self.assertEqual(h5s[key], d)
            d2 = Foo()
            with self.assertRaises(TypeError):
                h5s[key + '2'] = d2
            self.assertEqual(len(h5s), 1)
            self.assertEqual(h5s[key], d)

    def test_keys_with_dots(self):
        with self.get_h5store() as h5s:
            key = 'a.b'
            d = self.get_testdata()
            h5s[key] = d
            self.assertEqual(h5s[key], d)

    def test_keys_with_slashes(self):
        # HDF5 uses slashes for nested keys internally
        with self.get_h5store() as h5s:
            key = 'a/b'
            d = self.get_testdata()
            h5s[key] = d
            self.assertEqual(h5s[key], d)
            self.assertEqual(h5s['a']['b'], d)

    def test_value_none(self):
        with self.get_h5store() as h5s:
            key = 'a'
            d = None
            h5s[key] = d
            self.assertEqual(h5s[key], d)

    def test_set_get_attr_sync(self):
        with self.get_h5store() as h5s:
            self.assertEqual(len(h5s), 0)
            self.assertNotIn('a', h5s)
            with self.assertRaises(AttributeError):
                h5s.a
            a = 0
            h5s.a = a
            self.assertEqual(len(h5s), 1)
            self.assertIn('a', h5s)
            self.assertEqual(h5s.a, a)
            self.assertEqual(h5s['a'], a)
            a = 1
            h5s.a = a
            self.assertEqual(len(h5s), 1)
            self.assertIn('a', h5s)
            self.assertEqual(h5s.a, a)
            self.assertEqual(h5s['a'], a)

            def check_nested(a, b):
                self.assertEqual(len(h5s), 1)
                self.assertEqual(len(h5s.a), 1)
                self.assertIn('a', h5s)
                self.assertIn('b', h5s.a)
                self.assertEqual(h5s.a, a)
                self.assertEqual(h5s['a']['b'], b)
                self.assertEqual(h5s.a.b, b)
                self.assertEqual(h5s['a'], a)

            h5s.a = {'b': 0}
            check_nested({'b': 0}, 0)
            h5s.a.b = 1
            check_nested({'b': 1}, 1)
            h5s['a'] = {'b': 2}
            check_nested({'b': 2}, 2)
            h5s['a']['b'] = 3
            check_nested({'b': 3}, 3)

    def test_modify_nested(self):
        with self.get_h5store() as h5s:
            h5s.a = dict(b=True)
            a = h5s.a
            a['b'] = False
            assert not h5s.a['b']

    def test_attr_reference_modification(self):
        with self.get_h5store() as h5s:
            self.assertEqual(len(h5s), 0)
            self.assertNotIn('a', h5s)
            with self.assertRaises(AttributeError):
                h5s.a
            pairs = [(0, 1), (0.0, 1.0), ('0', '1'), (False, True)]
            dict_pairs = [(dict(c=a), dict(c=b)) for a, b in pairs]
            for A, B in chain(pairs, dict_pairs):
                h5s.a = A
                a = h5s.a
                self.assertEqual(a, A)
                self.assertEqual(h5s.a, A)
                a = B
                self.assertEqual(a, B)
                self.assertEqual(h5s.a, A)
                a = h5s['a']
                self.assertEqual(a, A)
                self.assertEqual(h5s.a, A)
                a = B
                self.assertEqual(a, B)
                self.assertEqual(h5s.a, A)

                # with nested values
                h5s['a'] = dict(b=A)
                self.assertEqual(h5s.a.b, A)
                b = h5s.a.b
                self.assertEqual(b, A)
                self.assertEqual(h5s.a.b, A)
                b = B
                self.assertEqual(b, B)
                self.assertEqual(h5s.a.b, A)
                b = h5s['a']['b']
                self.assertEqual(b, A)
                self.assertEqual(h5s.a.b, A)
                b = B
                self.assertEqual(b, B)
                self.assertEqual(h5s.a.b, A)
                b = h5s['a'].b
                self.assertEqual(b, A)
                self.assertEqual(h5s.a.b, A)
                b = B
                self.assertEqual(b, B)
                self.assertEqual(h5s.a.b, A)


class H5StoreNestedDataTest(H5StoreTest):

    def get_testdata(self):
        return dict(a=super(H5StoreNestedDataTest, self).get_testdata())


@unittest.skipIf(not PANDAS_AND_TABLES, 'requires pandas and pytables')
@unittest.skipIf(not NUMPY, 'requires numpy package')
class H5StorePandasDataTest(H5StoreTest):

    def get_testdata(self):
        return pandas.DataFrame(
            numpy.random.rand(8, 2), index=[string.ascii_letters[i] for i in range(8)])

    def assertEqual(self, a, b):
        try:
            return (a == b).all()
        except (AttributeError, ValueError):
            return super(H5StorePandasDataTest, self).assertEqual(a, b)
        else:
            assert isinstance(a, pandas.DataFrame)


@unittest.skipIf(not PANDAS_AND_TABLES, 'requires pandas and pyables')
@unittest.skipIf(not NUMPY, 'requires numpy package')
class H5StoreNestedPandasDataTest(H5StorePandasDataTest):

    def get_testdata(self):
        return dict(df=pandas.DataFrame(
            numpy.random.rand(8, 2), index=[string.ascii_letters[i] for i in range(8)]))

    def assertEqual(self, a, b):
        try:
            super(H5StoreNestedPandasDataTest, self).assertEqual(len(a), len(b))
            if six.PY2:
                super(H5StoreNestedPandasDataTest, self).assertEqual(
                    list(map(str, sorted(a.keys()))),
                    list(map(str, sorted(b.keys()))))
            else:
                super(H5StoreNestedPandasDataTest, self).assertEqual(a.keys(), b.keys())
            for key in a:
                super(H5StoreNestedPandasDataTest, self).assertEqual(a[key], b[key])
        except (TypeError, AttributeError):
            super(H5StoreNestedPandasDataTest, self).assertEqual(a, b)
        else:
            assert isinstance(a, Mapping) and isinstance(b, Mapping)


if __name__ == '__main__':
    unittest.main()
