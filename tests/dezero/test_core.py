from unittest import TestCase

import numpy as np

from dezero.core import square, Variable


class SquareTestCase(TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))

        y = square(x)

        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
