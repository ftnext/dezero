import numpy as np  # NOQA (for doctest)


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
    """ある変数から別の変数への対応関係を定めるものである、関数を表すクラス

    >>> x = Variable(np.array(10))
    >>> f = Function()
    >>> y = f(x)
    >>> print(type(y))
    <class 'core.Variable'>
    >>> print(y.data)
    100
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data  # actual data
        y = x ** 2
        output = Variable(y)
        return output
