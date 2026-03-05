import copy
import math
import operator

import numpy as np
from scipy.constants import sigma

from .opn import OPNs

from . import opn_math
from decimal import Decimal, getcontext
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 设置Decimal模块的精度
getcontext().prec = 28


def _check_broadcast(shape1, shape2):
    # 以下功能判断两个形状是否可以进行广播操作
    # shape1, other_shape = shape1, other.shape
    min_dim = len(shape1) if len(shape1) < len(shape2) else len(shape2)
    for i in range(min_dim):
        # print(f"self.shape={self_shape}, other.shape={other_shape}, i={i}")
        # 维度上的数，从后往前个个相同或双方有一个等于1，则可以广播
        if not (shape1[-min_dim + i] == shape2[-min_dim + i] or (
                shape1[-min_dim + i] == 1 or shape2[-min_dim + i] == 1)):
            raise ValueError(f"operands could not be broadcast together with shapes {shape1} {shape2}")
    # 形状相同，返回True
    return True


class ndarray:
    def __abstract_arithmetic(self, arithmetic, other):
        def _recursive_arithmetic(mat1, mat2):
            data = []
            if isinstance(mat1, list):
                list_len = len(mat1)
                for i in range(list_len):
                    if isinstance(mat2, list):
                        if len(mat1) != len(mat2):
                            raise ValueError("List lengths do not match for arithmetic operation.")
                        data.append(_recursive_arithmetic(mat1[i], mat2[i]))
                    else:
                        data.append(_recursive_arithmetic(mat1[i], mat2))
                return data
            else:
                if not isinstance(mat1, (OPNs, int, float)):
                    raise TypeError(f"Expected OPNs, int, or float, got {type(mat1)}")
                if isinstance(mat2, list):
                    raise TypeError("Expected OPNs, int, float, or ndarray, got list")
                if isinstance(mat2, np.ndarray):
                    raise TypeError("Expected OPNs object or ndarray, got numpy.ndarray")
                if not isinstance(mat2, (OPNs, int, float)):
                    raise TypeError(f"Expected OPNs, int, or float, got {type(mat2)}")
                return arithmetic(mat1, mat2)

        if isinstance(other, ndarray):
            new1, new2 = self.elements, other.elements
            if len(new1) != len(new2):
                raise ValueError("Array lengths do not match for arithmetic operation.")
            data = _recursive_arithmetic(new1, new2)
            opns_ndarray = ndarray()
            opns_ndarray.elements = data
            return opns_ndarray
        elif isinstance(other, np.ndarray):
            raise TypeError("Expected OPNs object or ndarray, got numpy.ndarray")
        elif isinstance(other, (OPNs, int, float)):
            result = _recursive_arithmetic(self.elements, other)
        else:
            raise TypeError(f'unsupported operand type(s) for {arithmetic}: OPNs ndarray and {type(other)}')

        opns_ndarray = ndarray()
        opns_ndarray.elements = result
        return opns_ndarray

    def sum(self, axis=None):
        def _recursive_ndim_sum(arr, dim_index):
            if dim_index == axis:
                data = None
                for i in range(len(arr)):
                    if not isinstance(arr[i], (OPNs, int, float)):
                        raise TypeError(f"Expected OPNs, int, or float, got {type(arr[i])}")
                    if data is None:
                        data = arr[i]
                    else:
                        data += arr[i]
                return data
            else:
                data = []
                for i in range(len(arr)):
                    total = _recursive_ndim_sum(arr[i], dim_index + 1)
                    if isinstance(total, ndarray):
                        data.append(total.elements)
                    else:
                        data.append(total)
                opns_ndarray = ndarray()
                opns_ndarray.elements = data
                return opns_ndarray

        if axis is None:
            def _recursive_sum(arr):
                if isinstance(arr, OPNs):
                    return arr
                else:
                    total = OPNs(0, 0)
                    for j in range(len(arr)):
                        total += _recursive_sum(arr[j])
                    return total

            return _recursive_sum(self)
        else:
            axis = axis if axis >= 0 else len(self.shape) + axis
            return _recursive_ndim_sum(self, 0)

    def mean(self, axis=None):
        def _recursive_ndim_mean(arr, dim_index):
            if dim_index == axis:
                data = None
                arr_len = len(arr)
                for i in range(arr_len):
                    if not isinstance(arr[i], (OPNs, int, float)):
                        raise TypeError(f"Expected OPNs, int, or float, got {type(arr[i])}")
                    if data is None:
                        data = arr[i]
                    else:
                        data += arr[i]
                return data / arr_len
            else:
                data = []
                arr_len = len(arr)
                for i in range(arr_len):
                    total = _recursive_ndim_mean(arr[i], dim_index + 1) / arr_len
                    if isinstance(total, ndarray):
                        data.append(total.elements)
                    else:
                        data.append(total)
                opns_ndarray = ndarray()
                opns_ndarray.elements = data
                return opns_ndarray

        if axis is None:
            def _recursive_mean(arr):
                if isinstance(arr, OPNs):
                    return arr
                else:
                    total = OPNs(0, 0)
                    for j in range(len(arr)):
                        total += _recursive_mean(arr[j])
                    return total

            return _recursive_mean(self) / self.size
        else:
            axis = axis if axis >= 0 else len(self.shape) + axis
            return _recursive_ndim_mean(self, 0)

    def __init__(self, elements=None):
        self.elements = elements

    # elements 是直接的（多维）数组里OPNs

    @property
    def ndim(self):#返回矩阵的维度。通过递归计算elements的形状太确定维度
        return len(self.shape)

    @property
    def shape(self):#返回矩阵的形状。通过递归遍历 elements 列表，拼接每个维度的长度来确定形状
        def _recursive_compute_shape(mat):
            ele_len = len(mat)
            # if ele_len == 0:
            #     return (0, )
            # print(f"长度 : {ele_len}")
            if isinstance(mat[0], list):  # 元素还是opns数组，说明是高维
                return (ele_len,) + _recursive_compute_shape(mat[0])  # 拼接shape
            elif isinstance(mat[0], OPNs):  # 如果元素是OPNs, 说明已经到最低维了
                return (ele_len,)

        return _recursive_compute_shape(self.elements)
        # return self.elements.shape

    @property
    def size(self):
        """
        矩阵元素个数
        :return:
        """
        shape = self.shape
        product = 1  # 乘积
        for i in shape:
            product *= i
        return product
        # return self.elements.size

    @property
    def T(self):
        """
        矩阵转置，目前只能转置2维矩阵
        :return:
        """
        if self.ndim == 1:
            return self.__copy__()
        elif self.ndim == 2:
            transpose_array = ndarray()
            tmp_array = []
            for j in range(self.shape[1]):
                tmp_array.append(self[:, j].elements)
            transpose_array.elements = tmp_array
            return transpose_array
        else:
            raise NotImplementedError(f"转置非二维矩阵，暂未实现: {self.shape}")
        # return self.elements.T

    def __copy__(self):
        """
        必须是深拷贝才能确保不会被新对象修改原对象内容
        """
        new_obj = self.__class__()
        new_obj.elements = copy.copy(self.elements)
        return new_obj

    def __len__(self):
        return len(self.elements)

    def __str__(self):
        """
        返回数组的字符串表示
        打印结果
        全部输出
        :return:
        """
        text = ""
        text += "["
        ndim = self.ndim  # 一维，直接输出
        ele_len = len(self.elements)
        for i in range(ele_len):
            text += str(self.elements[i])  # 递归
            if i != ele_len - 1:  # 不是最后一个元素，要输出一个空格或换行
                if ndim == 1:
                    text += ' '
                else:
                    text += '\n'
        text += "]"
        return text

    def __getitem__(self, item):
        # 采用不同的索引方式从 self.elements 中获取元素。若结果是列表或 numpy 数组，
        # 就将其封装成 ndarray 对象返回；若结果是 OPNs 对象，则直接返回。
        # return self.elements[item]
        # 需要考虑item是 支持切片、数字、数组、元组等多种索引方式
        # print(f"item: {item}, self.elements length: {len(self.elements)}")  # 添加调试信息
        result = None
        if isinstance(item, (slice, int)):  # 切片 数字
            result = self.elements[item]
        elif isinstance(item, (list, np.ndarray)):  # 数组
            result = []
            for i in item:
                result.append(self.elements[i])
        elif isinstance(item, tuple):  # 元组 [ , , ]
            result = []
            data = self.elements

            if len(item) == 1:  # (1, ), ([0, 2], )
                # print(item[0], type(item[0]))
                if isinstance(item[0], (list, np.ndarray)):
                    for i in item[0]:
                        result.append(self.elements[i])
                else:
                    result = self.elements[item[0]]
            elif isinstance(item[0], slice):  # [ :, 2] 或 [ :, :]
                start, stop, step = item[0].start or 0, item[0].stop or len(data), item[0].step or 1
                for n in range(start, stop, step):  # 针对[: ,1]取 1 列，把所有行的1列拿出来
                    data = self[n][item[1:]]
                    if isinstance(data, ndarray):
                        result.append(data.elements)
                    else:
                        result.append(data)
            elif isinstance(item[0], (list, np.ndarray)):  # 针对[[1, 2] ,1] 取两行一列，把两行的1列拿出来
                for n in item[0]:
                    result.append(self[n][item[1:]])
            else:  # [1, :], [1, 1](递归之后判断 len(item)==1), [1, [1, 2]]
                # print(f"剩下的 {item[1:]}")
                result = self[item[0]][item[1:]]

        if isinstance(result, (list, np.ndarray)):  # 结果是数组，就封装
            opns_ndarray = ndarray()
            opns_ndarray.elements = result
            return opns_ndarray
        else:  # 结果是OPNs，直接返回
            return result



    def __setitem__(self, key, value):
        # self.elements[key] = value
        if isinstance(key, tuple):  # 如果传入的是元组 m[ , ]
            row_key, col_key = key
            if isinstance(row_key, slice):
                start_row, stop_row, step_row = row_key.start or 0, row_key.stop or len(
                    self.elements), row_key.step or 1
                if isinstance(col_key, (list, np.ndarray)):  # [:, [1, 2]]
                    v_i = 0  # value 行数指示器
                    for i in range(start_row, stop_row, step_row):
                        v_j = 0
                        for j in col_key:  # value 行数指示器
                            self.elements[i][j] = value[v_i][v_j]
                            v_j += 1
                        v_i += 1
                else:
                    n = 0  # value行数指示器
                    for i in range(start_row, stop_row, step_row):
                        self.elements[i][col_key] = value[n]
                        n += 1
            elif isinstance(row_key, (list, np.ndarray)) and isinstance(col_key, (list, np.ndarray)):  # [[1,2], [1, 3]]
                v_i = 0
                for i in row_key:
                    v_j = 0
                    for j in col_key:
                        self.elements[i][j] = value[v_i][v_j]
                        v_j += 1
                    v_i += 1
            elif isinstance(row_key, (list, np.ndarray)):  # [[1, 2], 4] 或 [[1, 2], :]
                v_i = 0
                for i in row_key:
                    self.elements[i][col_key] = value[v_i]
                    v_i += 1
            else:  # [1, 2]
                self.elements[row_key][col_key] = value
        elif isinstance(key, (list, np.ndarray)):  # [[ ]]
            v_i = 0
            for i in key:
                self.elements[i] = value[v_i]
                v_i += 1
        else:  # [1]
            self.elements[key] = value

    def __abstract_compare(self, compare, other):
        """
        比较大小的函数几乎一样，所以抽象出来
        :param compare: >, <, = ...
        :param other:
        :return:
        """

        def _recursive_compare(mat1, mat2):
            result = []
            ele_len = len(mat1)
            for i in range(ele_len):
                if isinstance(mat1[i], list):  # 如果元素依旧是数组，说明还没到最底层
                    if isinstance(mat2, list):
                        result.append(_recursive_compare(mat1[i], mat2[i]))  # 递归
                    else:
                        result.append(_recursive_compare(mat1[i], mat2))
                else:  # 当递归到一维数组时
                    if isinstance(mat2, list):
                        result.append(compare(mat1[i], mat2[i]))  # 不同. 取elements的值就不是调用getter了
                    else:
                        result.append(compare(mat1[i], mat2))
            return result

        if isinstance(other, ndarray):  # 判断是不是 opns ndarray
            shape1, shape2 = self.shape, other.shape
            _check_broadcast(shape1, shape2)  # 判断形状是否相同, 不同就抛出异常
            # 由于相应功能没实现，所以虽然形状上可以广播但还是抛出异常
            if shape1 != shape2:  # 形状不相同
                raise NotImplementedError("该功能尚未实现：数组形状广播")
            return _recursive_compare(self.elements, other.elements)
            # elif isinstance(other, OPNs):
            #     ele_len = len(self.elements)
            #     for i in range(ele_len):
            #         if isinstance(self.elements[i], list):  # 如果元素依旧是数组，说明还没到最底层
            #             result.append(compare(self[i], other))
            #         else:  # 当递归到一维数组时
            #             result.append(compare(self.elements[i], other))  # 不同. 取elements的值就不是调用getter了
            #     return np.array(result)
        elif isinstance(other, (OPNs, int, float)):
            return _recursive_compare(self.elements, other)
        else:
            raise TypeError(f"'==' not supported between instances of 'OPNs' and {type(other)}")

    def __abstract_arithmetic(self, arithmetic, other):
        """
        算数运算也是几乎一样，所以抽象出来
        :param other:
        :return:
        """

        def _recursive_arithmetic(mat1, mat2):
            data = []
            if isinstance(mat1, list):  # 递归没到底
                list_len = len(mat1)
                for i in range(list_len):
                    if isinstance(mat2, list):
                        data.append(_recursive_arithmetic(mat1[i], mat2[i]))  # 把内容取出来
                    else:
                        data.append(_recursive_arithmetic(mat1[i], mat2))
                return data
            else:  # 到最底层
                return arithmetic(mat1, mat2)

        if isinstance(other, ndarray):
            # shape1, shape2 = self.shape, other.shape
            # _check_broadcast(shape1, shape2)  # 判断形状是否相同, 不同就抛出异常
            # # 由于相应功能没实现，所以虽然形状上可以广播但还是抛出异常
            # 懒得写了，直接用numpy吧
            new1, new2 = np.array(self.elements), np.array(other.elements)
            data = arithmetic(new1, new2)

            opns_ndarray = ndarray()
            opns_ndarray.elements = data.tolist()
            return opns_ndarray
        elif isinstance(other, np.ndarray):
            result = arithmetic(np.array(self.elements), other)
        elif isinstance(other, (OPNs, int, float)):
            result = _recursive_arithmetic(self.elements, other)
        else:
            raise TypeError(f'unsupported operand type(s) for {arithmetic}: OPNs ndarray and {type(other)}')

        opns_ndarray = ndarray()
        opns_ndarray.elements = result
        return opns_ndarray

    def __eq__(self, other):
        """
        等于 ==
        :param other:
        :return:
        """
        # return self.__abstract_compare(operator.eq, other)
        return self.elements == other.elements

    def __lt__(self, other):
        """
        小于 <
        """
        # return self.__abstract_compare(operator.lt, other)
        return self.elements < other.elements

    def __le__(self, other):
        """
        小于等于 <=
        """
        return self.__abstract_compare(operator.le, other)
        # return self.elements <= other.elements

    def __gt__(self, other):
        """
        大于 >
        """
        return self.__abstract_compare(operator.gt, other)
        # return self.elements > other.elements

    def __ge__(self, other):
        """
        大于等于 >=
        """
        return self.__abstract_compare(operator.ge, other)
        # return self.elements >= other.elements

    def __add__(self, other):
        """
        加 +
        :param other:
        :return:
        """
        return self.__abstract_arithmetic(operator.add, other)
        # return self.elements + other.elements

    def __radd__(self, other):
        """
        因为sum()函数是从0开始的，所以在第一次调用__add__方法时，它实际上是在尝试
        将0（一个整数）和你的自定义对象相加。这通常会导致错误，因为Python不知道如何
        将一个整数和一个自定义对象相加。

        为了解决这个问题，你需要在你的类中定义一个__radd__方法1。这个方法定义了当你
        的对象在右边，而一个不同类型的对象（在这种情况下是一个整数）在左边时，如何进
        行加法运算。一个常见的做法是在__radd__方法中检查other是否为0，如果是，则返
        回当前对象的一个副本1。这样，当sum()函数试图将0和你的对象相加时，就会得到你
        的对象本身，从而避免了错误。
        :param other:
        :return:
        """
        if isinstance(other, (int, float)) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        return self.__abstract_arithmetic(operator.sub, other)
        # if isinstance(other, ndarray):
        #     return self.elements - other.elements
        # else:
        #     return self.elements - other

    def __neg__(self):
        """
        一元负操作调用该方法
        :return:
        """
        return self.__abstract_arithmetic(operator.mul, -1)
        # return -self.elements

    def __mul__(self, other):
        """
        点乘与数乘 *
        对应位置乘，注意不是矩阵乘法
        :param other:
        :return:
        """
        return self.__abstract_arithmetic(operator.mul, other)
        # return self.elements * other.elements

    def __rmul__(self, other):
        if isinstance(other, int) and other == 1:
            return self
        return self.__mul__(other)

    def __reciprocal(self):
        """
        OPNs 矩阵 所有元素取倒数
        即 1/matrix
        :return:
        """

        def _recursive_reciprocal(mat):
            data = []
            list_len = len(mat)
            if isinstance(mat[0], list):  # 递归没到最底层
                for i in range(list_len):
                    data.append(_recursive_reciprocal(mat[i]))  # 把内容取出来
            else:  # 到最底层
                for i in range(list_len):
                    data.append(1 / mat[i])
            return data

        result = _recursive_reciprocal(self.elements)
        opns_ndarray = ndarray()
        opns_ndarray.elements = result
        return opns_ndarray
        # return 1 / self.elements

    def __truediv__(self, other):
        """
        除法
        :param other:
        :return:
        """
        return self.__abstract_arithmetic(operator.truediv, other)
        # if isinstance(other, ndarray):
        #     return self.elements / other.elements
        # else:
        #     return self.elements / other

    def __rtruediv__(self, other):
        """
        被除的时候调用
        """
        if isinstance(other, (int, float, OPNs)):  # 被数字除，就数字乘上它的倒数
            return other * self.__reciprocal()
        return other.__truediv__(self)

    def __matmul__(self, other):
        """
        矩阵乘法运算符 @
        """
        return self.dot(other)

    def __pow__(self, other):
        """
        次方
        :param other:
        :return:
        """
        return self.__abstract_arithmetic(operator.pow, other)
        # return self.elements ** other

    def __abs__(self):
        # return abs(self.elements)
        def _recursive_abs(mat):
            data = []
            list_len = len(mat)
            if isinstance(mat[0], list):  # 递归没到最底层
                for i in range(list_len):
                    data.append(_recursive_abs(mat[i]))  # 把内容取出来
            else:  # 到最底层
                for i in range(list_len):
                    data.append(abs(mat[i]))
            return data

        result = _recursive_abs(self.elements)
        opns_ndarray = ndarray()
        opns_ndarray.elements = result
        return opns_ndarray

    def add(self, other):
        return self.__add__(other)

    def sub(self, other):
        return self.__sub__(other)

    def multi(self, other):
        return self.__mul__(other)

    def dot(self, other):
        # return self.elements.dot(other)
        # 判断能不能做矩阵乘法
        # 1. 最低两维度必须满足矩阵乘法
        # 2. 剩余维度必须满足广播的条件
        # def kahan_sum(arr):
        #     left_arr = []
        #     right_arr = []
        #     for num in arr:
        #         left_arr.append(num.a)
        #         right_arr.append(num.b)
        #     return OPNs(math.fsum(left_arr), math.fsum(right_arr))

        a_shape, b_shape = self.shape, other.shape
        if a_shape[-1] != b_shape[-2]:
            raise ValueError(f"matmul: Mismatch in its dimension {a_shape} and {b_shape}")

        a_other_shape, b_other_shape = a_shape[:-2], b_shape[:-2]
        _check_broadcast(a_other_shape, b_other_shape)  # 判断形状是否相同, 不同就抛出异常
        # 高维尚未实现，所以先抛出异常
        if len(a_shape) > 2 or len(b_shape) > 2:
            raise NotImplementedError(f"高维矩阵（张量）乘法尚未实现 {a_shape} @ {b_shape}")

        data = []
        # Aij = A的第i行 与 B的第j列 乘积之和
        for i in range(a_shape[0]):
            row_data = []
            for j in range(b_shape[1]):
                tmp = self[i] * other[:, j]
                row_data.append(tmp.sum())
            data.append(row_data)
        opns_ndarray = ndarray()
        opns_ndarray.elements = data
        return opns_ndarray

    def flatten(self):
        """
        高维拉平成一维
        :return:
        """

        # return self.elements.flatten()

        def _recursive_flatten(mat):
            if isinstance(mat[0], OPNs):  # 最低维了，元素是OPNs
                return mat
            else:
                data = []
                for sub in mat:
                    data += _recursive_flatten(sub)
                return data

        opns_ndarray = ndarray()
        opns_ndarray.elements = _recursive_flatten(self.elements)
        return opns_ndarray

    def reshape(self, shape):
        data = np.array(self.elements)
        opns_ndarray = ndarray()
        opns_ndarray.elements = data.reshape(shape).tolist()
        return opns_ndarray

    def sum(self, axis=None):
        def _recursive_ndim_sum(arr, dim_index):
            if dim_index == axis:  # 找到目标层 把里面的元素加起来
                data = None
                for i in range(len(arr)):
                    # print(arr[i])
                    if data is None:
                        data = arr[i]
                    else:
                        data += arr[i]
                return data  # data或许是一个ndarray，或许是一个OPNs
            else:  # 还没到，继续往下翻
                data = []
                for i in range(len(arr)):
                    total = _recursive_ndim_sum(arr[i], dim_index + 1)
                    if isinstance(total, ndarray):
                        data.append(total.elements)
                    else:
                        data.append(total)
                opns_ndarray = ndarray()
                opns_ndarray.elements = data
                return opns_ndarray

        if axis is None:
            def _recursive_sum(arr):
                if isinstance(arr, OPNs):
                    return arr
                else:
                    total = OPNs(0, 0)
                    for j in range(len(arr)):
                        total += _recursive_sum(arr[j])
                    return total

            return _recursive_sum(self)  # 这里返回的是OPNs
        else:
            # 这里处理axis 全部转成正的
            axis = axis if axis >= 0 else len(self.shape) + axis
            return _recursive_ndim_sum(self, 0)

    def mean(self, axis=None):
        """
        平均值
        axis = None, 计算所有元素的总均值
        axis = 0, 计算矩阵中每一列的均值
        axis = 1, 计算矩阵中每一行的均值
        或数组的均值
        :param axis:
        :return:
        """

        def _recursive_ndim_mean(arr, dim_index):
            if dim_index == axis:  # 找到目标层 把里面的元素加起来
                data = None
                arr_len = len(arr)
                for i in range(arr_len):
                    if data is None:
                        data = arr[i]
                    else:
                        data += arr[i]
                return data / arr_len  # data或许是一个ndarray，或许是一个OPNs
            else:  # 还没到，继续往下翻
                data = []
                arr_len = len(arr)
                for i in range(arr_len):
                    total = _recursive_ndim_mean(arr[i], dim_index + 1) / arr_len
                    if isinstance(total, ndarray):
                        data.append(total.elements)
                    else:
                        data.append(total)
                opns_ndarray = ndarray()
                opns_ndarray.elements = data
                return opns_ndarray

        if axis is None:
            def _recursive_mean(arr):
                if isinstance(arr, OPNs):
                    return arr
                else:
                    total = OPNs(0, 0)
                    for j in range(len(arr)):
                        total += _recursive_mean(arr[j])
                    return total

            return _recursive_mean(self) / self.size  # 这里返回的是OPNs
        else:
            # 这里处理axis 全部转成正的
            axis = axis if axis >= 0 else len(self.shape) + axis
            return _recursive_ndim_mean(self, 0)

    def var(self, axis=None):
        """
        计算数组的方差
        :param axis: 0 返回列数个方差。1返回行数个方差
        """
        # return self.elements.var(axis)
        mat_mean = self.mean(axis)  # 计算均值
        # print(type(mat_mean))
        # print(mat_mean)
        # print(self - mat_mean)
        # print((self - mat_mean) ** 2)
        variance = ((self - mat_mean) ** 2).mean(axis)  # 方差
        # 由于会有(-2.0, -2.0)**2 The multiplicative inverse of this OPNs (-2.0, -2.0) does not exist
        # variance = ((self - mat_mean + OPNs(0, -1e-10)) ** 2).mean(axis)
        return variance

    def std(self, axis=None):
        """
        计算数组的标准差
        :param axis: 0 返回列数个方差。1返回行数个方差
        """
        # return self.elements.std(axis)
        variance = self.var(axis)
        std_dev = opn_math.sqrt(variance)
        return std_dev


def array(arr=None):
    if isinstance(arr, ndarray):
        return copy.copy(arr)

    elif isinstance(arr, (np.ndarray, list)):
        def _check_opns_or_number(sub_arr):  # 检查元素类型 OPNs是True, Number是False
            if isinstance(sub_arr, (np.ndarray, list)):  # 还没到底
                return _check_opns_or_number(sub_arr[0])
            elif isinstance(sub_arr, OPNs):
                return True
            else:
                return False

        if _check_opns_or_number(arr):  # 如果元素是OPNs, 直接放进对象
            data = arr
        else:
            def _create_opns_array(sub_arr):
                sub_data = []
                if isinstance(sub_arr[0], (np.ndarray, list)):  # 还没到底
                    for ele in sub_arr:
                        sub_data.append(_create_opns_array(ele))
                else:  # 到底了
                    arr_len = len(sub_arr)
                    for i in range((arr_len if arr_len % 2 == 0 else arr_len + 1) // 2):
                        if i * 2 < arr_len - 1:
                            sub_data.append(OPNs(sub_arr[i * 2], sub_arr[i * 2 + 1]))
                        else:
                            sub_data.append(OPNs(sub_arr[i * 2], 0))
                return sub_data

            data = _create_opns_array(arr)
        opns_ndarray = ndarray()
        opns_ndarray.elements = data
        return opns_ndarray

def sum(arr, axis=None):
    """
    求和
    :param arr:
    :param axis:
    :return:
    """
    return arr.sum(axis=axis)


def mean(arr, axis=None):
    """
        平均值
        axis = None, 计算所有元素的总均值
        axis = 0, 计算矩阵中每一列的均值
        axis = 1, 计算矩阵中每一行的均值
        或数组的均值
        :param arr:
        :param axis:
        :return:
        """
    return arr.mean(axis=axis)
if __name__ == '__main__':
    a=array([1,2,3,4])
    b=array([OPNs(1,2),OPNs(3,2)])
    print('mknklnjk')
    print(mean(a))
    print(mean(b))


def sqrt(arr):
    return np.sqrt(arr)
    # if isinstance(arr, ndarray):
    #     def _recursive_sqrt(mat):
    #         data = []
    #         list_len = len(mat)
    #         if isinstance(mat[0], list):  # 递归没到最底层
    #             for i in range(list_len):
    #                 data.append(_recursive_sqrt(mat[i]))  # 把内容取出来
    #         else:  # 到最底层
    #             for i in range(list_len):
    #                 data.append(sqrt(mat[i]))
    #         return data
    #
    #     result = _recursive_sqrt(arr.elements)
    #     opns_ndarray = ndarray()
    #     opns_ndarray.elements = result
    #     return opns_ndarray
    # else:
    #     return np.sqrt(arr)


def vstack(array1, array2):
    """
    垂直堆叠
    :param array1:
    :param array2:
    :return:
    """
    return np.vstack((array1, array2))
    # 条件：第一个维度可以不同，后面的必须相同
    # array1_shape, array2_shape = array1.shape, array2.shape
    # if len(array1_shape) != len(array2_shape) or array1_shape[1:] != array2_shape[1:]:
    #     raise ValueError(f"vstack: Mismatch in its dimension {array1_shape} and {array2_shape}")
    # new_arr = ndarray()
    # new_arr.elements = copy.copy(array1.elements)
    # for i in range(len(array2.elements)):
    #     new_arr.elements.append(array2.elements[i])
    # return new_arr


def transpose(arr):
    return arr.T


def dot(array1, array2):
    return array1.dot(array2)


def eye(n):
    """
    创建单位矩阵
    :param n: 行列的数量
    :return: OPNss单位矩阵
    """
    opns_ndarray = ndarray()
    opns_ndarray.elements = [[OPNs(0, -1) if i == j else OPNs(0, 0) for j in range(n)] for i in range(n)]
    return opns_ndarray


def ones(shape):
    """
    创建每个元素都是(0, -1)的矩阵
    :param shape:
    :return:
    """
    if isinstance(shape, int):  # 一维
        opns_ndarray = ndarray()
        opns_ndarray.elements = [OPNs(0, -1) for _ in range(shape)]
        return opns_ndarray
    else:  # 高维
        def _create_mat(ndim):  # 维度下标和该维度上的元素个数
            if ndim == len(shape) - 1:
                return [OPNs(0, -1) for _ in range(shape[ndim])]
            else:
                return [_create_mat(ndim + 1) for _ in range(shape[ndim])]

        opns_ndarray = ndarray()
        opns_ndarray.elements = _create_mat(0)
        return opns_ndarray



def zeros(shape):
    """
    创建每个元素都是(0, 0)的矩阵
    :param shape:
    :return:
    """
    if isinstance(shape, int):  # 一维
        opns_ndarray = ndarray()
        opns_ndarray.elements = [OPNs(0, 0) for _ in range(shape)]
        return opns_ndarray
    else:  # 高维
        def _create_mat(ndim):  # 维度下标和该维度上的元素个数
            if ndim == len(shape) - 1:
                return [OPNs(0, 0) for _ in range(shape[ndim])]
            else:
                return [_create_mat(ndim + 1) for _ in range(shape[ndim])]

        opns_ndarray = ndarray()
        opns_ndarray.elements = _create_mat(0)
        return opns_ndarray


def std(arr, axis=None):
    """
    标准差
    :param arr:
    :param axis:
    :return:
    """
    if isinstance(arr, ndarray):
        return arr.std(axis=axis)
    else:
        return np.std(arr, axis=axis)


def argsort(arr, axis=-1, kind=None, order=None):
    """
    对OPNs数组进行升序比较，返回新顺序的下标
    目前只实现了一维和二维数组的排序
    **网上说axis默认是None, 然后把二位数组拉平来排序，实际不是，默认是-1, 每行排序**
    kind和order尚未实现
    :param arr:
    :param axis:
    :param kind:
    :param order:
    :return:
    """
    # if kind is not None or order is not None:
    #     raise NotImplementedError("argsort args: kind and order are not implemented")
    #
    # if axis is None:
    #     return argsort(arr.flatten())
    # # 这里处理axis
    # axis = axis if axis >= 0 else len(arr.shape) + axis

    # 太复杂了，用numpy吧
    if isinstance(arr, ndarray):
        return np.argsort(arr.elements, axis=axis, kind=kind, order=order)
    else:
        return np.argsort(arr, axis=axis, kind=kind, order=order)


def argmax(arr, *args, **kwargs):
    if isinstance(arr, ndarray):
        return np.argmax(arr.elements, *args, **kwargs)
    else:
        return np.argmax(arr, *args, **kwargs)

def min(arr, *args, **kwargs):
    if isinstance(arr, (int, float)):  # 处理标量比较
        return np.min(arr, *args, **kwargs)
    if isinstance(arr, ndarray):
        result = np.min(arr.elements, *args, **kwargs)
        if isinstance(result, np.ndarray):
            opns_ndarray = ndarray()
            opns_ndarray.elements = result.tolist()
            return opns_ndarray
        else:
            return result
    else:
        return np.min(arr, *args, **kwargs)

def max(arr, *args, **kwargs):
    if isinstance(arr, (int, float)):
        return np.max(arr, *args, **kwargs)
    if isinstance(arr, ndarray):
        result = np.max(arr.elements, *args, **kwargs)
        if isinstance(result, np.ndarray):
            opns_ndarray = ndarray()
            opns_ndarray.elements = result.tolist()
            return opns_ndarray
        else:
            return result
    else:
        return np.max(arr, *args, **kwargs)

def diag(arr):
    """
    对于二维矩阵，取出对角线元素为数组
    对于一维数组，创建对角矩阵，对角元素为该数组
    该函数不处理高维
    :param arr:
    :return:
    """
    arr_shape = arr.shape
    arr_dim = len(arr_shape)
    if arr_dim > 2:
        raise ValueError(f"Input must be 1- or 2-d, but got {arr_dim}")

    if isinstance(arr, ndarray):
        data = []
        if arr_dim == 1:
            for i in range(arr_shape[0]):
                row_data = []
                for j in range(arr_shape[0]):
                    if i == j:
                        row_data.append(arr[i])
                    else:
                        row_data.append(OPNs(0.0, 0.0))
                data.append(row_data)

        else:
            for i in range(arr_shape[0] if arr_shape[0] < arr_shape[1] else arr_shape[1]):
                data.append(arr[i][i])

        opns_ndarray = ndarray()
        opns_ndarray.elements = data
        return opns_ndarray
    else:
        return np.diag(arr)


# OPNs 特有方法 ----------------------
"""
使用雅可比方法求解矩阵的特征值和特征向量。
通过迭代更新矩阵 D 和特征向量矩阵 X，
直到收敛。
"""
def jacobi(A, iterations=5000):  # 5000
    n = len(A)
    # last_X = eye(n)  # 当连续两次迭代得到的解向量之间的差异小于一个阈值，迭代停止
    last_X = None
    X = eye(n)
    D = copy.copy(A)

    for sss in range(iterations):
        # while True:
        # max_val = OPNs(0.0, 0.0)
        max_val = D[0][1]
        # print(f"max_val = {max_val}")
        max_pos = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(D[i, j]) > max_val:
                    # max_val = abs(D[i, j])
                    max_val = abs(D[i, j])
                    max_pos = (i, j)
        i, j = max_pos
        # print(i, j)
        # 为了防止 D[i, i] == D[j, j] , 添加一个epsilon = 1e-10
        # epsilon = OPNs(1e-8, -1e-8)
        epsilon = OPNs(1, -1)
        # OPNs(1, -1) 接近 OPNs(0, -1e-16)
        # OPNs(1e-6, -1e-6) 接近 OPNs(0, -1e-21)
        # theta = arctan(2 * D[i, j] / (D[i, i] - D[j, j] + epsilon)) / 2
        if D[i, i] - D[j, j] < epsilon:
            theta = OPNs(0, -np.pi / 4)
        else:
            theta = opn_math.atan((2 * D[i, j]) / (D[i, i] - D[j, j] + epsilon)) / 2

        # print(f"theta = {theta}")

        # theta = div(arctan(div(multi(2, D[i, j]), sub(D[i, i], D[j, j]))), 2)
        U = eye(n)
        U[i, i] = opn_math.cos(theta)
        U[j, j] = opn_math.cos(theta)
        U[i, j] = -opn_math.sin(theta)
        U[j, i] = opn_math.sin(theta)
        # U 都一样

        # D = np.dot(np.dot(U.T, D), U)
        # print(U)
        D = U.T.dot(D).dot(U)
        # D 大体一致
        X = X.dot(U)
        # print(X)
        # stime = time.time()
        # if last_X is None:
        #     last_X = X
        # else:
        #     sum_change = (0, 0)
        #     change_matrix = X - last_X
        #     for qq in range(len(X)):
        #         for ww in range(len(X)):
        #             sum_change = add(sum_change, abs_opn(change_matrix[qq, ww]))
        #     if lt(sum_change, (0, -1e-8)):
        #         print(f"雅可比循环了 {sss}")
        #         break
        #     last_X = X
        # etime = time.time()
        # print(f"花费了{etime - stime}")
        #
        # stime = time.time()
        if last_X is not None:
            change_matrix = X - last_X
            # print(change_matrix)
            # if change_matrix.is_abs_sum_lt_threshold(epsilon):
            # print((abs(change_matrix)).sum())
            if (abs(change_matrix)).sum() <= epsilon:
                # print(f"jacobi迭代了 {sss} 次")
                break
        last_X = X
        # 循环的时间代价高于numpy直接操作矩阵
        # etime = time.time()
        # print(f"花费了{etime - stime}")
        # print(f"一轮结束")
    return diag(D), X


def approximate_opn2num(target, lower_bound, upper_bound, tolerance=OPNs(0, -1e-12)):
    """
    通过逼近算法寻找目标对象的大致值。

    参数:
    - target: 目标对象。
    - lower_bound: 下界。
    - upper_bound: 上界。
    - tolerance: 允许的误差范围。

    返回:
    - 大致的值范围 (lower, upper)。
    """
    while upper_bound - lower_bound > tolerance:
        # print(f"up: {upper_bound}, low: {lower_bound}")
        mid_point = (lower_bound + upper_bound) / 2
        # print(mid_point)
        if target < mid_point:
            # print("小于")
            upper_bound = mid_point
        else:
            # print("其他")
            lower_bound = mid_point

    return -((lower_bound + upper_bound) / 2).b

def validate_opn(x):
    if not isinstance(x, OPNs):
        x = OPNs(float(x), 0) if isinstance(x, (int, float)) else OPNs(0, 0)
    return x


def opn_distance(opn1, opn2):
    """
    计算两个OPNs对象之间的距离 (定义11)
    参数:
        opn1: OPNs对象
        opn2: OPNs对象
    返回:
        OPNs对象表示的距离
    """
    if not isinstance(opn1, OPNs) or not isinstance(opn2, OPNs):
        raise TypeError("opn_distance函数只接受OPNs对象作为参数")

    diff = opn1 - opn2
    return abs(diff)  # 直接调用OPNs的__abs__方法取绝对值


def generalized_metric(vec1, vec2):
    """
    计算两个OPNs向量之间的广义距离 (定义12)
    参数:
        vec1: OPNs向量（列表/数组）
        vec2: OPNs向量（列表/数组）
    返回:
        OPNs对象表示的总距离
    """
    # 输入验证和转换
    if isinstance(vec1, np.ndarray):
        vec1 = vec1.tolist()
    if isinstance(vec2, np.ndarray):
        vec2 = vec2.tolist()

    if len(vec1) != len(vec2):
        raise ValueError("向量长度必须相同")

    # 确保所有元素都是OPNs对象
    vec1 = [validate_opn(x) for x in vec1]
    vec2 = [validate_opn(x) for x in vec2]

    # 初始化累加器
    sum_sq = OPNs(0, 0)

    # 计算每个维度的距离平方和
    for i, (opn1, opn2) in enumerate(zip(vec1, vec2)):
        #print(f"\n--- 计算维度 {i} ---")
        #print(f"OPNs1: {opn1}")
        #print(f"OPNs2: {opn2}")
        dist = opn_distance(opn1, opn2)
        dist_sq = dist ** 2  # 使用OPNs的幂运算
        sum_sq += dist_sq
    #print("\n=== 最终累加结果 ===")
    #print(f"总平方和: {sum_sq}")
    # 计算平方根并取绝对值（双重保证）
    try:
        result = abs(sum_sq ** 0.5)  # 关键修改：对最终结果取绝对值
        #print(f"最终结果（取绝对值后）: {result}")

        # 验证结果是否满足OPNs正数定义（a + b < 0）
        if not (result.a + result.b <= 0):  # 注意OPNs的"正数"定义
            raise ValueError(f"结果不符合OPNs正数定义: {result}")
        return result
    except Exception as e:
        raise ValueError(f"计算失败: {str(e)}. 输入: sum_sq={sum_sq}")





if __name__ == '__main__':
    #opns_ndarray = ndarray()
    #opns_ndarray.elements = [[OPNs(0, -1), OPNs(0, -7), OPNs(0, -2)], [OPNs(0, -3), OPNs(0, -5), OPNs(0, -10)], [OPNs(0, -4), OPNs(0, 8), OPNs(0, -1)]]
    #
    #print("------------矩阵------------")
    #print(opns_ndarray)
    #print("---------------------------")
    #print(f"维度 {opns_ndarray.ndim}")
    #print(f"形状 {opns_ndarray.shape}")
    #print(f"大小 {opns_ndarray.size}")
    #print("-------------转置------------")
    #print(opns_ndarray.T)
    #print("----------------------------")
    # print(f"取单行 [1]: {opns_ndarray[1]}")
    # print(f"取单元素 [0, 1]: {opns_ndarray[0, 1]}, [0][1]: {opns_ndarray[0][1]}")
    # print(f"切片 [:-2]: {opns_ndarray[:-2]}")
    # print(f"切片 [:, -2]: {opns_ndarray[:,-2]}")
    # print(f"切片 [-2, :]: {opns_ndarray[-2, :]}")
    # print(f"切片矩阵 [1:, 1:]: {opns_ndarray[1:, 1:]}")
    # print(opns_ndarray[1, [0, 2]])
    # opns_ndarray[2, 1] = OPNs(0, -9)
    # opns_ndarray[:,1] = [OPNs(0, -5), OPNs(0, -7), OPNs(0, -2)]
    # opns_ndarray[1, :] = [OPNs(0, -5), OPNs(0, -5), OPNs(0, -5)]
    # opns_ndarray[:,[0, 1]] = [[OPNs(0, -5), OPNs(0, -3)],[ OPNs(0, -3), OPNs(0, -11)], [OPNs(0, -44), OPNs(0, -66)]]
    # opns_ndarray[[0, 1], :] = [[OPNs(0, -5), OPNs(0, -3), OPNs(0, -3)], [OPNs(0, -11), OPNs(0, -44), OPNs(0, -66)]]
    # print(opns_ndarray)

    # oa1 = ndarray()
    # oa1.elements = [[OPNs(0, -1), OPNs(0, -6), OPNs(0, -2)], [OPNs(0, -3), OPNs(0, -5), OPNs(0, -10)],
    #                 [OPNs(0, -4), OPNs(0, 8), OPNs(0, -1)]]
    # oa2 = ndarray()
    # oa2.elements = [[OPNs(0, -2), OPNs(0, -7), OPNs(0, -2)], [OPNs(0, -8), OPNs(0, -5), OPNs(0, -10)],
    #                 [OPNs(0, -4), OPNs(0, 8), OPNs(0, -1)]]
    # #
    # oa3 = ndarray()
    # oa3.elements = [OPNs(0, -1), OPNs(0, -5), OPNs(0, -2)]
    # #
    oa4 = ndarray()#矩阵
    oa4.elements = [
         [[OPNs(0, -4), OPNs(0, -3)], [OPNs(0, -1), OPNs(0, -3)], [OPNs(0, -8), OPNs(0, -5)]],
         [[OPNs(0, -2), OPNs(0, -12)], [OPNs(0, -19), OPNs(0, -6)], [OPNs(0, -20), OPNs(0, -2)]]
     ]
    oa4.elements = [
        [OPNs(0, -4), OPNs(0, -3)], [OPNs(0, -1), OPNs(0, -3)], [OPNs(0, -8), OPNs(0, -5)],
        [OPNs(0, -2), OPNs(0, -12)], [OPNs(0, -19), OPNs(0, -6)], [OPNs(0, -20), OPNs(0, -2)]
    ]
    print("oa4第一个OPNs元素是：",oa4.elements[0])
    print("oa4：", oa4.elements)
    print("oa4：", oa4.elements[0])
    # print(oa1)
    # print("-------")
    # print(diag(oa3))
    # print("-----")
    # print(oa2)
    # print("-----")
    # print(oa1.dot(oa2))

    # print(oa1.flatten())

    print("测试基本运算")
    opn1 = OPNs(3, 4)
    opn2 = OPNs(1, 2)

    print("加减法测试")
    print(opn1 + opn2) # OPNs(4, 6)
    print(opn1 - opn2) #OPNs(2, 2)

    # 乘法测试
    print(opn1 * opn2)#OPNs(-(3 * 2 + 4 * 1), -(3 * 1 + 4 * 2))  # 根据乘法公式

    print("距离测试")
    print(opn_distance(OPNs(5, 3), OPNs(2, 1)) )#OPNs(3, 2)
    print(opn_distance(OPNs(1, 2), OPNs(3, 1))) #OPNs(2, -1)


    print("test_generalized_metric()")
    # 测试向量距离
    vec1 = [OPNs(7.0, 1.4), OPNs(3.2, 4.7)]
    vec2 = [OPNs(5.0, 1.4), OPNs(3.2, 3.7)]

    # 计算广义距离
    dist = generalized_metric(vec1, vec2)
    print("dist:",dist)
    # 验证结果
    print("expected:",(OPNs(2.0, 0.0) ** 2 + OPNs(0.0, 1.0) ** 2) ** 0.5)
    print("test_edge_cases")
    # 测试边界情况
    zero = OPNs(0, 0)

    print("零距离测试")
    print("dist0:",opn_distance(zero, zero))#zero

    # 测试无效输入
    try:
        opn_distance(1, OPNs(0, 0))
    except Exception as e:
        logger.error(f"类型: {str(e)}")

    try:
        generalized_metric([1, 2], [OPNs(0, 0), OPNs(1, 1)])
    except Exception as e:
        logger.error(f"类型: {str(e)}")
