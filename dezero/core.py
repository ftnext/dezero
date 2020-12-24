"""
連鎖率 dy/dx = ((dy/dy * dy/db) * db/da) * da/dx

順伝播
x -> A -> a -> B -> b -> C -> y

逆伝播
dy/dx <- A'(x) <- dy/da <- B'(a) <- dy/db <- C'(b) <- dy/dy(=1)
※ A'(x) は A'(x)（=変数）の乗算 を表す（簡略化して表記）
⇔
x.grad <- A.backward <- a.grad <- B.backward <- b.grad
                                                    <- C.backward <- y.grad(=1)
※ A.backwardに「A'(x)（=変数）の乗算」を内包している
"""

from typing import Iterable, List, Union

import numpy as np


def as_array(x):
    """xがスカラ型（scalar type＝ndarrayでない）であれば、ndarrayに変換して返す関数

    0次のndarrayの計算結果（スカラ型＝ndarrayでない）をndarrayに変換するのに使う
    （X次のndarrayは[]をX個使ったリストをnp.arrayに渡している）

    >>> x = np.array([1.0])
    >>> y = x ** 2
    >>> print(type(x), x.ndim)
    <class 'numpy.ndarray'> 1
    >>> print(type(y))
    <class 'numpy.ndarray'>
    >>> x = np.array(1.0)
    >>> y = x ** 2
    >>> print(type(x), x.ndim)
    <class 'numpy.ndarray'> 0
    >>> print(type(y))
    <class 'numpy.float64'>

    >>> np.isscalar(np.float64(1.0))
    True
    >>> np.isscalar(2.0)
    True
    >>> np.isscalar(np.array(1.0))
    False
    >>> np.isscalar(np.array([1, 2, 3]))
    False
    """

    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    """「箱」（データを持つ存在）としての変数を表すクラス

    data属性にデータが保持される（データはNoneかnp.ndarrayに限る）
    creator属性に計算グラフのつながりの情報が保持される

    backwardメソッドにより（関数のbackwardメソッドと連携して）逆伝播を求める

    >>> data = np.array(1.0)
    >>> x = Variable(data)
    >>> print(x.data)
    1.0
    >>> x.data = np.array(2.0)
    >>> print(x.data)
    2.0

    >>> x = Variable(None)
    >>> x = Variable(1.0)
    Traceback (most recent call last):
      ...
    TypeError: <class 'float'> is not supported
    """

    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data = data  # 通常値
        self.grad = None  # 通常値に対応する微分値
        self.creator = None  # 変数の生みの親となる関数（関数以外が生み出した変数の場合はNone）

    def set_creator(self, func: "Function"):
        self.creator = func

    def backward(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html
        if self.grad is None:
            # grad is ones because of dy/dy(= 1)
            # data and grad are the same dtype
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # get Function
            # handle output Variables (variable arguments) of the Function
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            # gxs[i] is grad of f.inputs[i] (variable arguments)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:  # when set grad for the first time
                    x.grad = gx
                else:  # add grad for the same Variable
                    # x.grad is created NEWLY
                    x.grad = x.grad + gx
                    # if wrote x.grad += gx (in-place)
                    # ex. y = add(x, x) gxs[0], gxs[1], y.grad are same ID
                    # (because of add's backward)
                    # and x.grad is the same ID (x.grad = gx)
                    # then x.grad += gx updates x.grad AND y.grad
                    # In the result, y.grad is not 1 (BUG!)

                # stop when x (Variable) is not created by Function
                if x.creator is not None:
                    funcs.append(x.creator)  # append the prior Function


class Function:
    """ある変数から別の変数への対応関係を定めるものである、関数の基底クラス

    関数の入出力をVariableインスタンスで統一している（関数の連結が可能になる -> Exp関数で例示）

    入力されたVariableを覚えることで、変数の通常値も微分値も参照できる
    順伝播が計算されるときに、箱（変数）に計算グラフのつながりを記録する
    """

    def __call__(
        self, *inputs: Iterable["Variable"]
    ) -> Union[List["Variable"], "Variable"]:
        xs = [x.data for x in inputs]  # actual data
        ys = self.forward(*xs)  # unpack and pass to the method
        # when forward method returns the only 1 element
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # make output Variable remember the creator Function
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs  # store input Variables
        self.outputs = outputs  # store output Variables
        # If the number of elements is 1, return the first Variable
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        """通常の具体的な計算（順伝播）

        Functionクラスを継承した具体的な関数クラスで定義する
        """
        raise NotImplementedError

    def backward(self, gy):
        """微分を求めるための具体的な計算（逆伝播）

        計算グラフの簡略化の部分（A'(x)の乗算）を内包していて、
        計算グラフではbackward関数の適用で済むようにしている（後述の逆伝播で微分を計算する例）。
        オブジェクト呼び出し(__call__)で設定されるself.inputを用いる
        """
        raise NotImplementedError


class Add(Function):
    """入力された2つの値を足し算する、具体的な関数

    >>> x0 = Variable(np.array(2))
    >>> x1 = Variable(np.array(3))
    >>> f = Add()
    >>> y = f(x0, x1)
    >>> print(y.data)
    5
    """

    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        """足し算の逆伝播は、上流の微分をそのまま流す

        y = x0 + x1 のとき dy/dx0 = 1, dy/dx1 = 1 より1をかける＝そのまま
        """
        return gy, gy


def add(x0: "Variable", x1: "Variable"):
    """足し算の関数のクラスをPythonの関数として利用できるようにする

    >>> x0 = Variable(np.array(2))
    >>> x1 = Variable(np.array(3))
    >>> y = add(x0, x1)
    >>> print(y.data)
    5

    z = x**2 + y**2
    >>> x = Variable(np.array(2.0))
    >>> y = Variable(np.array(3.0))
    >>> z = add(square(x), square(y))
    >>> z.backward()
    >>> print(z.data)
    13.0
    >>> print(x.grad)
    4.0
    >>> print(y.grad)
    6.0

    同じ変数を使って足し算を行う
    >>> x = Variable(np.array(3.0))
    >>> y = add(x, x)
    >>> print("y", y.data)
    y 6.0
    >>> y.backward()  # y = 2x より dy/dx = 2
    >>> print("x.grad", x.grad)
    x.grad 2.0

    >>> x = Variable(np.array(3.0))
    >>> y = add(add(x, x), x)  # y = 3x
    >>> y.backward()
    >>> print(x.grad)
    3.0
    """
    return Add()(x0, x1)


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
        """y = x ** 2"""
        return x ** 2

    def backward(self, gy):
        """dy/dx * gy

        dy/dx = 2 * x
        gyは出力側から渡される微分（ndarray）
        """
        # concrete function knows the number of inputs
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


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

    自動で作られた計算グラフを逆向きに辿る（Define-by-Run）
    >>> assert y.creator == C
    >>> assert y.creator.inputs[0] == b
    >>> assert y.creator.inputs[0].creator == B
    >>> assert y.creator.inputs[0].creator.inputs[0] == a
    >>> assert y.creator.inputs[0].creator.inputs[0].creator == A
    >>> assert y.creator.inputs[0].creator.inputs[0].creator.inputs[0] == x

    逆伝播を求める
    >>> y.backward()
    >>> print(x.grad)
    3.297442541400256
    """

    def forward(self, x):
        """y = e ** x"""
        # https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        return np.exp(x)

    def backward(self, gy):
        """dy/dx * gy

        dy/dx = e ** x
        """
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    """中心差分近似を使って数値微分を求める関数

    数値微分：微小な差異 eps を用いて関数の変化量を求める手法。
    1. 桁落ちによる誤差、2. 変数ごとに微分を求める（高い計算コスト）により、
    バックプロパゲーションの実装の正しさの確認に使われる（勾配確認）

    >>> f = Square()
    >>> x = Variable(np.array(2.0))
    >>> dy = numerical_diff(f, x)
    >>> print(dy)
    4.000000000004
    """

    # if x is dim=0 ndarray, (x.data - eps) is np.float64
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def f(x: Variable) -> Variable:
    """(e ** (x**2)) ** 2を返す合成関数

    合成関数の数値微分を求める
    >>> x = Variable(np.array(0.5))
    >>> dy = numerical_diff(f, x)
    >>> print(dy)
    3.2974426293330694

    上の結果は、xを0.5から微小な値だけ変化させたら、
    yは微小な値の3.297...倍だけ変化するということ (dy/dx=f'(x)=3.297)
    """
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


def square(x):
    """2乗するFunctionをPythonの関数として利用できるようにする

    Squareクラスのインスタンスを作って、そのインスタンスを呼び出すという手間を省略している
    """
    return Square()(x)


def exp(x):
    """指数関数のFunctionをPythonの関数として利用できるようにする

    例：通常の数値計算を行うような感覚で計算できる
    >>> x = Variable(np.array(0.5))
    >>> y = square(exp(square(x)))  # 関数を連続して適用
    >>> y.backward()
    >>> print(x.grad)
    3.297442541400256
    """
    return Exp()(x)
