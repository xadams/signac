import logging
import os
import datetime

import pymongo

from ..core.config import load_config
from ..core.dbclient_connector import DBClientConnector
from . import database

# namespace extension
from .conversion import DBMethod, BasicFormat, Adapter
from . import formats

logger = logging.getLogger(__name__)

PYMONGO_3 = pymongo.version_tuple[0] == 3

def access_compmatdb(host = None, config = None):
    if config is None:
        config = load_config()
    if host is None:
        host = config['compmatdb_host']
    connector = DBClientConnector(config)
    connector.connect(host)
    connector.authenticate()
    db = connector.client[config['database_compmatdb']]
    return database.Database(db = db, config = config)

class StorageFileCursor(object):

    def __init__(self, cursor, fn):
        self._cursor = cursor
        self._fn = fn

    def __getitem__(self, key):
        return self._cursor[key]

    def read(self):
        with open(self._fn, 'rb') as file:
            return file.read()

class Storage(object):
    
    def __init__(self, collection, fs_dir):
        self._collection = collection
        self._storage_path = fs_dir

    def _filename(self, file_id):
        return os.path.join(self._storage_path, str(file_id))
    
    def open(self, file_id, *args, ** kwargs):
        return open(self._filename(file_id), * args, ** kwargs)

    def new_file(self, ** kwargs):
        kwargs.update({
            '_fs_dir': self._storage_path,
            '_fs_timestamp': datetime.datetime.now(),
        })
        if PYMONGO_3:
            file_id = self._collection.insert_one(kwargs).inserted_id
        else:
            file_id = self._collection.insert(kwargs)
        return self.open(file_id, 'wb')

    def find(self, spec = {}, *args, ** kwargs):
        docs = self._collection.find(spec, ['_id'], * args, ** kwargs)
        for doc in docs:
            file_id = doc['_id']
            fn = self._filename(file_id)
            if not os.path.isfile(fn):
                fn2 = os.path.join(doc['_fs_dir'], str(file_id))
                if not os.path.isfile(fn2):
                    raise FileNotFoundError(file_id)
                else:
                    fn = fn2
            yield StorageFileCursor(doc, fn)

#    def find_recent(self, spec = {}, * args, ** kwargs):
#        spec.update({
#            {'$orderby': {'_fs_timestamp': -1}})
#        return find

    def delete(self, file_id):
        os.remove(self._filename(file_id))
        if PYMONGO_3:
            self._collection.delete_one({'_id': file_id})
        else:
            self._collection.remove({'_id': file_id})
