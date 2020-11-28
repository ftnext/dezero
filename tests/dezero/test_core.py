from unittest import TestCase

import numpy as np

from dezero.core import numerical_diff, square, Variable


class SquareTestCase(TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))

        y = square(x)

        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)

        y.backward()

        expected = np.array(6.0)  # 2x (x=3.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))  # dim=1 ndarray
        y = square(x)

        y.backward()

        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
