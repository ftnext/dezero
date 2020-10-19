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
        self.data = data  # 通常値
        self.grad = None  # 通常値に対応する微分値


class Function:
    """ある変数から別の変数への対応関係を定めるものである、関数の基底クラス

    関数の入出力をVariableインスタンスで統一している（関数の連結が可能になる -> Exp関数で例示）

    入力されたVariableを覚えることで、変数の通常値も微分値も参照できる
    """

    def __call__(self, input: Variable) -> Variable:
        x = input.data  # actual data
        y = self.forward(x)
        output = Variable(y)
        self.input = input  # store input Variable
        return output

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
        x = self.input.data
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

    上記例で、逆伝播によってyの微分を求める
    >>> y.grad = np.array(1.0)
    >>> b.grad = C.backward(y.grad)
    >>> a.grad = B.backward(b.grad)
    >>> x.grad = A.backward(a.grad)
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
        x = self.input.data
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

    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
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
