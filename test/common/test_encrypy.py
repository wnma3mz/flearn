# coding: utf-8
import pickle
import unittest

from flearn.common import Encrypt


class TestEncrypt(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.e = Encrypt()

    def test_encrypt(self):
        data = {"a": 1}
        self.assertEqual(pickle.loads(self.e.decode(self.e.encode(pickle.dumps(data)))), data)


if __name__ == "__main__":
    t = TestEncrypt("test_encrypt")
    t.test_encrypt()
