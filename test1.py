import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        import torch

        print(torch.cuda.is_available()) # add assertion here


if __name__ == '__main__':
    unittest.main()


