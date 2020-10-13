class Variable:
    """「箱」（データを持つ存在）としての変数を表すクラス

    data属性にデータが保持される

    >>> import numpy as np
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
