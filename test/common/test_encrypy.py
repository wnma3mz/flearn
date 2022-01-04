# coding: utf-8
import pickle

from flearn.common import Encrypt

if __name__ == "__main__":
    e = Encrypt()
    data = {"a": 1}
    assert pickle.loads(e.decode(e.encode(pickle.dumps(data)))) == data
