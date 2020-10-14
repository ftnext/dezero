import numpy as np


class Variable:
    """「箱」（データを持つ存在）としての変数を表すクラス

    data属性にデータが保持される

    >>> data = np.array(1.0)
    >>> x = Variable(data)
    >>> print(x.data)
    1.0
    >>> x.data = np.array(2.0)
    >>> print(x.data)
    2.0
    """

    def __init__(self, data):
        self.data = data


class Function:
    """ある変数から別の変数への対応関係を定めるものである、関数の基底クラス

    関数の入出力をVariableインスタンスで統一している（関数の連結が可能になる -> Exp関数で例示）
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data  # actual data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        """具体的な計算

        Functionクラスを継承した具体的な関数クラスで定義する
        """
        raise NotImplementedError


class Square(Function):
    """入力された値を2乗する、具体的な関数

    >>> x = Variable(np.array(10))
    >>> f = Square()
    >>> y = f(x)
    >>> print(type(y))
    <class 'core.Variable'>
    >>> print(y.data)
    100
    """

    def forward(self, x):
        return x ** 2


class Exp(Function):
    """e（自然対数の底）の'入力された値'乗を返す、具体的な関数

    関数を連結した使用例: (e ** (x**2)) ** 2
    >>> A, B, C = Square(), Exp(), Square()
    >>> x = Variable(np.array(0.5))
    >>> a = A(x)
    >>> b = B(a)
    >>> y = C(b)
    >>> print(y.data)
    1.648721270700128
    """

    def forward(self, x):
        # https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        return np.exp(x)
