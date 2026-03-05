import itertools
import random
from collections import deque

import numpy as np

"""
OPNs 配对
"""


def gen_random_opns_pairs(k, n):
    """
    随机生成n个不同的配对结果
    注意个数n不能超过可能出现的配对数的最大值 max_n = [1, k]的所有奇数的乘积
    :param k: 传入数字k，返回[0-k)的所有数字的配对
    :param n: 传入n表示随机挑选配对结果, n用于控制一共挑选多少个
    """
    k = k if k % 2 == 0 else k + 1
    # random.seed(random.seed())
    # 计算n是否大于最大的可能数，是的话就生成所有可能。因为n大了就会找不到更多不同的可能，就会卡住
    product = 1
    for i in range(1, k + 1):
        if i % 2 != 0:
            product *= i
    if n > product:
        return list(gen_all_opns_pairs(k))

    results = set()
    while len(results) < n:
        # 有[0, k)的数，将其任意打乱顺序，然后从头开始依次每两个组成一对（元组），然后把组合好的放在数组里返回
        seq = list(range(k))  # 创建一个包含[0, k)的列表
        random.shuffle(seq)  # 随机打乱列表
        pairs = [tuple(sorted((seq[i], seq[i + 1]))) for i in range(0, len(seq), 2)]  # 从头开始依次每两个组成一对
        results.add(tuple(sorted(pairs)))
    return [list(result) for result in results]


#
# def gen_random_opns_pairs(k, n):
#     permutations = list(itertools.permutations([2, 5, 7, 8, 9]))
#
#     b = [0, 1, 3, 4, 6]
#     rr = []
#     for p in permutations:
#         # 用zip将b和p配对起来，然后用两层循环将结果输出
#         result = [f for r in zip(b, p) for f in r]
#
#         # 添加b和p末尾不匹配时多余的元素
#         result.extend(b[len(p):])
#         result.extend(p[len(b):])
#         rr.append(result)
#     return rr

def seq_all_pairs(seq):
    """
    使用生成器计算所有可能的配对
    """
    if len(seq) < 2:
        return
    elif len(seq) == 2:
        yield [tuple(seq)]
    else:
        first = seq[0]
        rest = seq[1:]
        for i in range(len(rest)):
            pair = (first, rest[i])
            for p in seq_all_pairs(rest[:i] + rest[i + 1:]):
                yield [pair] + p

def seq_all_pairs_list(seq):
    """
    返回所有可能的配对列表
    """
    result = []
    if len(seq) < 2:
        return result
    elif len(seq) == 2:
        result.append([tuple(seq)])
    else:
        first = seq[0]
        rest = seq[1:]
        for i in range(len(rest)):
            pair = (first, rest[i])
            for p in seq_all_pairs(rest[:i] + rest[i + 1:]):
                result.append([pair] + p)
    return result

def gen_all_opns_pairs(k):
    """
    :param k: 传入数字k，返回[0-k)的所有数字的配对
    :return: 生成器
    """
    k = k if k % 2 == 0 else k + 1
    return seq_all_pairs(list(range(k)))


def seq_all_pairs_list1(seq):
    """
    返回所有可能的配对列表（包括顺序不同的组合）
    例如：对于[0,1,2,3]，会生成[[(0,1),(2,3)], [(0,2),(1,3)], [(0,3),(1,2)],
                           [(1,0),(2,3)], [(1,2),(0,3)], [(1,3),(0,2)],
                           ...]等所有排列组合
    """
    result = []
    if len(seq) < 2:
        return result
    elif len(seq) == 2:
        # 对于两个元素，返回两种顺序的组合
        result.append([(seq[0], seq[1])])
        result.append([(seq[1], seq[0])])
    else:
        first = seq[0]
        rest = seq[1:]
        for i in range(len(rest)):
            # 对于每个可能的配对，生成两种顺序
            pair1 = (first, rest[i])
            pair2 = (rest[i], first)

            # 递归处理剩余元素
            remaining = rest[:i] + rest[i + 1:]
            sub_pairs1 = seq_all_pairs_list(remaining)
            sub_pairs2 = seq_all_pairs_list(remaining)

            # 添加两种顺序的组合
            for p in sub_pairs1:
                result.append([pair1] + p)
            for p in sub_pairs2:
                result.append([pair2] + p)
    return result


def seq_all_pairs_list3(seq):
    """
    迭代生成所有可能的配对组合（包括顺序不同的组合）
    返回一个生成器，每次生成一种配对方案
    """
    if len(seq) < 2:
        return

    from itertools import permutations

    # 生成所有可能的排列
    for perm in permutations(seq):
        pairs = []
        # 每次取两个元素形成一对
        for i in range(0, len(perm), 2):
            if i + 1 < len(perm):
                pairs.append((perm[i], perm[i + 1]))
                # 生成两种顺序的配对
                yield pairs.copy()
                pairs[-1] = (perm[i + 1], perm[i])
                yield pairs.copy()
                pairs.pop()  # 移除当前配对以便尝试下一种
            else:
                # 处理奇数个元素的情况，最后一个元素不配对
                yield pairs.copy()


def gen_all_opns_pairs1(k):
    """
    生成所有可能的配对组合（包括顺序不同的组合）
    :param k: 特征数量
    :return: 所有配对组合的列表
    """
    k = k if k % 2 == 0 else k + 1
    indices = list(range(k))
    return seq_all_pairs_list(indices)


def gen_random_pairs_(k, n):
    """
    随机生成n波配对结果，每波包含k个配对，每个数字在每波中可以重复出现。
    每波中所有数字[0, k)都要出现。

    :param k: 传入数字k，表示每波中配对的总数（也即数字的范围是[0, k)）
    :param n: 传入n表示生成n波配对结果
    :return: 一个列表，包含n波配对结果，每波是一个包含k个配对的列表
    """
    k = k if k % 2 == 0 else k + 1
    # random.seed(random.seed())
    # 计算n是否大于最大的可能数，是的话就生成所有可能。因为n大了就会找不到更多不同的可能，就会卡住
    # product = 1
    # for i in range(1, k + 1):
    #     if i % 2 != 0:
    #         product *= i
    # if n > product:
    #     return list(gen_all_opns_pairs(k))
    if k == 4:
        product = 3
    elif k == 6:
        product = 70
    elif k == 8:
        product = 3507
    else:
        product = 9999
    if n > product:
        n = product

    results = set()
    while len(results) < n:
        # 有[0, k)的数，将其任意打乱顺序，然后从头开始依次每两个组成一对（元组），然后把组合好的放在数组里返回
        seq_1 = list(range(k))  # 创建一个包含[0, k)的列表
        random.shuffle(seq_1)  # 随机打乱列表
        seq_2 = list(range(k))
        random.shuffle(seq_2)

        if not np.any(np.equal(seq_1, seq_2)):  # 两个数相等的OPNs是0
            pair_tmp = set()
            for i in range(k):
                pair_tmp.add(tuple(sorted((seq_1[i], seq_2[i]))))
            if len(pair_tmp) == k:
                # print(list(pair_tmp))
                results.add(tuple(sorted(list(pair_tmp))))
        # print(len(results))
    return [list(result) for result in results]


def seq_all_pairs_list2(seq):
    """
    生成所有可能的配对组合（每个元素只出现一次，考虑顺序）
    返回一个生成器，每次生成一种配对方案
    """
    if len(seq) < 2:
        return
    def backtrack(remaining, current_pairs):
        if len(remaining) == 0:
            yield current_pairs.copy()
            return

        first = remaining.popleft()
        for i in range(len(remaining)):
            second = remaining[i]
            # 生成两种顺序的配对
            for pair in [(first, second), (second, first)]:
                # 移除已配对的元素
                remaining.remove(second)
                # 递归处理剩余元素
                yield from backtrack(remaining.copy(), current_pairs + [pair])
                # 恢复状态（回溯）
                remaining.insert(i, second)
        # 恢复 first
        remaining.appendleft(first)

    # 使用双端队列提高性能
    yield from backtrack(deque(seq), [])


from itertools import combinations
from collections import defaultdict


def seq_all_pairs_with_repeats(seq):
    """
    生成所有可能的配对组合（允许特征在不同配对中重复使用），
    并且只返回包含重复特征值的组合（即至少有一个特征出现在多个配对中）

    规则：
    1. 每个配对内部元素不重复（如 (0,0) 无效）
    2. 不同配对之间可以共享特征（如 (0,1) 和 (0,2) 是合法组合）
    3. 配对顺序无关（组合中的配对顺序不影响结果）
    4. 仅返回至少有一个特征重复出现的组合
    """
    if len(seq) < 2:
        return

    # 首先生成所有可能的有效配对（无序）
    all_possible_pairs = list({tuple(sorted(pair)) for pair in combinations(seq, 2)})

    # 生成所有可能的非空子集（大小≥2）
    for k in range(2, len(all_possible_pairs) + 1):
        for subset in combinations(all_possible_pairs, k):
            # 统计特征出现次数
            feature_counts = defaultdict(int)
            for pair in subset:
                a, b = pair
                feature_counts[a] += 1
                feature_counts[b] += 1
            # 检查是否有特征重复（出现次数>1）
            if max(feature_counts.values()) > 1:
                yield list(subset)
if __name__ == '__main__':
    #pairs = gen_random_opns_pairs(100, 100)
    # # pairs = gen_random_pairs_(10, 10)
    # for pair in pairs:
    #     print(pair)
    #pass
    # all_pairs=seq_all_pairs_list([0,1,2,3])
    # #[[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
    # print(all_pairs)
    # for pair in all_pairs:
    #     print(pair)
    # all_pairs1 = seq_all_pairs_list1([0, 1, 2, 3])
    # #[[(0, 1), (2, 3)], [(1, 0), (2, 3)],
    # # [(0, 2), (1, 3)], [(2, 0), (1, 3)],
    # # [(0, 3), (1, 2)], [(3, 0), (1, 2)]]
    # print(all_pairs1)
    # for pair1 in all_pairs1:
    #     print(pair1)
    # all_pairs1 = gen_all_opns_pairs1(4)#[[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
    # print(all_pairs1)
    # for pair1 in all_pairs1:
    #     print(pair1)

    # all_pairs2 = gen_random_pairs_(10,4)
    # for pair in all_pairs2:
    #     print(pair)
    # 测试生成前6个配对组合
    seq = [0, 1, 2, 3]
    count = 0
    for pairs in seq_all_pairs_list2(seq):
        print(pairs)
        count += 1
    print(count)
    # seq = [0, 1, 2, 3,4]
    # count = 0
    # for pairs in seq_all_pairs_new(seq):
    #     print(pairs)
    #     count += 1
    # print(f"总共有 {count} 种配对组合")

