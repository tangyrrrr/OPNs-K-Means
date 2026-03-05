import sys
from .opn import OPNs

'''a unique additive identity and a unique multiplicative identity'''
zero = (0, 0)
one = (0, -1)
oneNeg = (0, 1)


'''初始值设置'''
'''nSlope、ksi均为φ函数中的参数'''
initialValue = [1, 0.5, 200, -101, True]
nSlope = initialValue[0]
ksi = initialValue[1]
default_precision = initialValue[2]
sign = initialValue[3]
# flag = initialValue[4]

'''Simplified basic φ-operations of φ-scalar-multiplication, φ-addition and φ-multiplication'''


def scalar_multi_fun(real_number, interval_number):
    return real_number * interval_number


def add_fun(interval_number1, interval_number2):
    return interval_number1 + interval_number2


def multi_fun(interval_number1, interval_number2):
    return interval_number1 * interval_number2


'''Elementary operations of OPNs'''
'''为了直观表现opn，函数中的参数opn以元组的数据类型定义'''
'''1. Arithmetic operations of OPNs'''


def scalar_multi(real_number, opn=()):  # 实数与OPNs的标量乘
    head = scalar_multi_fun(real_number, opn[0])
    tail = scalar_multi_fun(real_number, opn[1])
    new_opn = (head, tail)
    return new_opn


def add(*args):  # 传入两个及以上的opns或一个包含两个及以上opns的数组，返回相加后的OPNs
    if len(args) == 1:
        opn_list = args[0]
    else:
        opn_list = args
    first_entry = opn_list[0][0]
    second_entry = opn_list[0][1]
    for i in range(len(opn_list) - 1):
        first_entry = add_fun(first_entry, opn_list[i + 1][0])
        second_entry = add_fun(second_entry, opn_list[i + 1][1])
    new_opn = (first_entry, second_entry)
    return new_opn


def multi(*args):  # 传入两个及以上的opns或一个包含两个及以上opns的数组，返回相乘后的OPNs
    if len(args) == 1:
        opn_list = args[0]
    else:
        opn_list = args
    first_entry, second_entry = opn_list[0][0], opn_list[0][1]
    for i in range(len(opn_list) - 1):
        temp = first_entry
        # first_entry = add_fun(multi_fun(first_entry, 1 - opn_list[i + 1][1]),
        #                       multi_fun(second_entry, 1 - opn_list[i + 1][0]))
        # second_entry = add_fun(multi_fun(temp, 1 - opn_list[i + 1][0]),
        #                        multi_fun(second_entry, 1 - opn_list[i + 1][1]))
        first_entry = add_fun(multi_fun(first_entry, -opn_list[i + 1][1]),
                              multi_fun(second_entry, -opn_list[i + 1][0]))
        second_entry = add_fun(multi_fun(temp, -opn_list[i + 1][0]),
                               multi_fun(second_entry, -opn_list[i + 1][1]))
    new_opn = (first_entry, second_entry)
    return new_opn


def sub(opn1=(), opn2=()):  # 传入两个OPNs，返回相减后的OPNs
    new_opn = add(opn1, scalar_multi(-1, opn2))
    return new_opn


def neg_power(opn=()):  # 求一个OPNs的导数
    if opn[0] == opn[1] or opn[0] == -opn[1]:
        sys.exit('The multiplication inverse of this OPNs {} does not exist'.format(opn))
    else:
        first_entry = opn[0] / (opn[0] ** 2 - opn[1] ** 2)
        second_entry = opn[1] / (opn[1] ** 2 - opn[0] ** 2)
        new_opn = (first_entry, second_entry)
        return new_opn


def div(opn1=(), opn2=()):  # 除法运算
    new_opn = multi(opn1, neg_power(opn2))
    return new_opn


'''Total order on the set of OPNss'''


def compare(opn1=(), opn2=()):
    if isinstance(opn1, OPNs) and isinstance(opn2, OPNs):
        sum1 = opn1.a + opn1.b
        sum2 = opn2.a + opn2.b
        if sum1 > sum2 or (sum1 == sum2 and opn1.a > opn2.a):
            return [opn2, opn1]
        else:
            return [opn1, opn2]
    else:
        raise TypeError("compare expects two OPNs objects")

def max(*args):
    if len(args) == 1:
        opn_list = args[0]
        max_opn = opn_list[0]
        for i, v in enumerate(opn_list):
            max_opn = compare(max_opn, v)[1]
        return max_opn
    else:
        max_opn = args[0]
        for i, v in enumerate(args):
            max_opn = compare(max_opn, v)[1]
        return max_opn


def min(*args):
    if len(args) == 1:
        opn_list = args[0]
        min_opn = opn_list[0]
        for i, v in enumerate(opn_list):
            min_opn = compare(min_opn, v)[0]
        return min_opn
    else:
        min_opn = args[0]
        for i, v in enumerate(args):
            min_opn = compare(min_opn, v)[0]
        return min_opn


def sorted(opn_list, start=0, end=sign, reverse=False, flag=True):  # 传入一组opns，并在原始数组上按照升序对数组[start:end]进行排序
    try:
        if end == sign:
            end = len(opn_list) - 1
        if flag:
            if start < 0 or start >= len(opn_list) - 1 or end >= len(opn_list) or start >= end or end <= 0:
                sys.exit(
                    'Error: Array start position \'{}\' or end position \'{}\' is not standard!'.format(start, end))
        flag = False

        def partition(sublist, low, high):
            pivot = sublist[low]
            while low < high:
                while low < high and sublist[high] == compare(sublist[high], pivot)[1]:
                    high = high - 1
                sublist[low] = sublist[high]
                while low < high and sublist[low] == compare(sublist[low], pivot)[0]:
                    low = low + 1
                sublist[high] = sublist[low]
            sublist[low] = pivot
            return low

        if start < end:
            pivotpos = partition(opn_list, start, end)
            sorted(opn_list, start, pivotpos - 1, flag=flag)
            sorted(opn_list, pivotpos + 1, end, flag=flag)
            if reverse:
                opn_list.reverse()
                return opn_list
            else:
                return opn_list
    except Exception as e:
        sys.exit(e)


'''Distance of OPNss'''
def distance(opn1: tuple, opn2: tuple):  # 求两个opn的距离
    dis = sub(opn1, opn2) if max(sub(opn1, opn2), zero) == sub(opn1, opn2) else sub(opn2, opn1)
    return dis






