import math
import warnings

import numpy as np

"""
封装OPNs为类
"""


# zero = (0, 0)
# one = (0, -1)
# oneNeg = (0, 1)

# 工具
def tran(x, m):
    if x < 0:
        return -math.pow(-x, m)
    else:
        return math.pow(x, m)


class OPNs:
    """
    OPNs 类
    """

    def __init__(self, a, b):
        # 【修正】将传入的参数强制转换为float类型，防止数据类型溢出
        self.a = float(a)
        self.b = float(b)

    def __str__(self):
        """
        打印输出OPNs
        """
        return f"({self.a}, {self.b})"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        """
        拷贝一个新的
        用于计算结果不变时返回。
        如果不返回新的，会因为修改结果的值而改变原始的值
        """
        return OPNs(self.a, self.b)

    def __len__(self):
        # 这里假设OPNs对象的长度为2，因为它有两个属性a和b
        return 2
    #添加迭代协议
    def __iter__(self):
        yield self.a
        yield self.b

    """
    四则运算
    """

    def __neg__(self):
        """
        负号取反
        -OPNs
        """
        return OPNs(-self.a, -self.b)

    def __add__(self, other):
        """
        +, 两个OPNs相加
        """
        if isinstance(other, (int, float)):
            return OPNs(self.a + other, self.b)
        elif isinstance(other, OPNs):
            return OPNs(self.a + other.a, self.b + other.b)
        elif hasattr(other, '__radd__'):
            return other.__radd__(self)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'OPNs' and '{type(other).__name__}'")

    def __radd__(self, other):
        """
        被加操作，当使用内置函数求和等操作时，会用0加OPNs
        """
        if isinstance(other, (int, float)) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        """
        - 两个OPNs 相减
        """
        if isinstance(other, (int, float)):
            return OPNs(self.a - other, self.b)
        elif isinstance(other, OPNs):
            return OPNs(self.a - other.a, self.b - other.b)
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'OPNs' and '{type(other).__name__}'")

    def __mul__(self, other):
        """
        * OPNs相乘，OPNs与数乘
        (a, b) * (c, d) = (-ad-bc, -ac-bd)
        """
        if isinstance(other, (int, float)):
            return OPNs(self.a * other, self.b * other)
        elif isinstance(other, OPNs):
            e = -self.a * other.b - self.b * other.a
            f = -self.a * other.a - self.b * other.b
            return OPNs(e, f)
        elif hasattr(other, '__rmul__'):
            return other.__rmul__(self)
        else:
            raise TypeError(f"unsupported operand type(s) for *: 'OPNs' and '{type(other).__name__}'")

    def __rmul__(self, other):
        """
        被乘
        """
        if isinstance(other, (int, float)) and other == 1:
            return self
        return self.__mul__(other)

    def __neg_power(self):
        """
        OPNs的倒数
        :return:
        """
        if self.a == self.b or self.a == -self.b:
            raise ZeroDivisionError(f'The multiplicative inverse of this OPNs {self} does not exist')
        else:
            c = self.a / (self.a ** 2 - self.b ** 2)
            d = self.b / (self.b ** 2 - self.a ** 2)
            return OPNs(c, d)

    def __truediv__(self, other):
        """
        / 除法
        OPNs1/OPNs2: OPNs1 * OPNs2倒数
        """
        if isinstance(other, (int, float)):
            return OPNs(self.a / other, self.b / other)
        elif isinstance(other, OPNs):
            return self.__mul__(other.__neg_power())
        elif hasattr(other, '__rdiv__'):
            return other.__rdiv__(self)
        else:
            raise TypeError(f"unsupported operand type(s) for /: 'OPNs' and '{type(other).__name__}'")

    def __rtruediv__(self, other):
        """
        被除以
        实数/OPNs: 实数 * OPNs倒数
        OPNs1/OPNs2: OPNs1 * OPNs2倒数
        :param other:
        :return:
        """
        if isinstance(other, (int, float, OPNs)):
            return self.__neg_power().__mul__(other)
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(other).__name__}' and 'OPNs'")
    """
    比较
    """

    def __eq__(self, other):
        """
        ==
        """
        return self.a == other.a and self.b == other.b

    def __gt__(self, other):
        """
        > 大于
        """
        sub_OPNs = self.__sub__(other)
        return (sub_OPNs.a + sub_OPNs.b < 0) or (sub_OPNs.a + sub_OPNs.b == 0 and sub_OPNs.a > 0)

    def __lt__(self, other):
        """
        < 小于
        """
        sub_OPNs = self.__sub__(other)
        return (sub_OPNs.a + sub_OPNs.b) > 0 or (sub_OPNs.a + sub_OPNs.b == 0 and sub_OPNs.a < 0)

    def __gl__(self, other):
        """
        >= 大于等于
        """
        return self.__gt__(other) or self.__eq__(other)

    def __le__(self, other):
        """
        <= 小于等于
        """
        return self.__lt__(other) or self.__eq__(other)

    def __abs__(self):
        if self.a + self.b > 0 or (self.a + self.b == 0 and self.a < 0):
            return OPNs(-self.a, -self.b)
        return self.__copy__()

    '''2. Power and nth root of OPNss'''

    """
    次方和开方
    """

    def __pow__(self, other):
        """
        幂运算 不支持分数次幂(但倒数后整数次开方可以)
        :param other:
        :return:
        """
        if other == 0:  # 指数为0
            return OPNs(0, -1)
        if other == 1:  # 指数为1
            return self.__copy__()
        if other > 1:  # 指数大于0, 次方
            # if self.a == -self.b or self.a == self.b:
            #     raise ZeroDivisionError(f'The multiplicative inverse of this OPNs {self} does not exist')
            # else:
            head = (((-1) ** (other + 1)) / 2) * ((self.a + self.b) ** other)
            tail = 0.5 * ((self.a - self.b) ** other)
            c = head + tail
            d = head - tail
            return OPNs(c, d)
        # 开方运算
        if other < 0:  # 指数other小于0, 返回1/OPNs^(|other|)
            return self.__pow__(-other).__neg_power()
        if other % 1 != 0:
            n = 1 / other
            if n % 1 != 0:
                raise ValueError("不规范的开根\'{}\': 该运算规则root()仅支持开整数根!".format(n))
            elif n % 2 == 1:
                head = 0.5 * tran(self.a + self.b, 1 / n)
                tail = 0.5 * tran(self.a - self.b, 1 / n)
                first_entry = head + tail
                second_entry = head - tail
                new_OPNs = OPNs(first_entry, second_entry)
                return new_OPNs
            elif n % 2 == 0 and self.a + self.b <= 0 and self.a >= self.b:
                head = 0.5 * ((-self.a - self.b) ** (1 / n))
                tail = 0.5 * ((self.a - self.b) ** (1 / n))
                first_entry = head + tail
                second_entry = head - tail
                new_OPNs = OPNs(-first_entry, -second_entry)
                return new_OPNs
            else:
                raise Exception(
                    "Error: When n is even, if OPNs is negative, or the first term of OPNs is smaller than the second term,"
                    "the OPNs {} cannot open roots!".format(self))

    def _exp(self):
        # 使用catch_warnings来管理警告的上下文环境
        with warnings.catch_warnings():
            # 使用filterwarnings来控制警告的行为
            warnings.filterwarnings('error')  # 将警告转换成异常

            try:
                # head = (math.e ** self.a - math.e ** (-self.a)) / (2 * (math.e ** self.b))
                # tail = -(math.e ** self.a + math.e ** (-self.a)) / (2 * (math.e ** self.b))
                head = 0.5 * (math.e ** (self.a - self.b) - math.e ** (-self.a - self.b))
                tail = -0.5 * (math.e ** (self.a - self.b) + math.e ** (-self.a - self.b))
                return OPNs(head, tail)
            except RuntimeWarning:
                # 这里处理RuntimeWarning的逻辑
                print(f"运行时异常，指数太大 当前: {self.b}")

        # head = (math.e ** self.a - math.e ** (-self.a)) / (2 * (math.e ** self.b))
        # tail = -(math.e ** self.a + math.e ** (-self.a)) / (2 * (math.e ** self.b))
        # return OPNs(head, tail)

    def __rpow__(self, other):
        """
        实数的OPNs次方
        """
        if other > 0:
            return (self.__mul__(math.log(other)))._exp()
        else:
            raise ValueError('Error: the real number \'{}\' should be greater than 0'.format(other))


