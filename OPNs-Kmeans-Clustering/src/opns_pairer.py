import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .common import opnpy as op 


class OPNsPairer(BaseEstimator, TransformerMixin):
    def __init__(self, pair=None):
        self.pair = pair
        self.flat_pair = None
        self.required_n_features_ = None  # 记录需要的最大特征数

    def __str__(self):
        return f'OPNsPairer: {self.pair}'

    def fit(self, X, y=None):
        if self.pair is None:
            raise ValueError("Feature pair cannot be None")

        # 计算需要的最大特征索引（补零前）
        max_index = max([idx for sublist in self.pair for idx in sublist])
        self.required_n_features_ = max_index + 1
        self.flat_pair = [item for sublist in self.pair for item in sublist]
        return self

    def transform(self, X):
        # 补零到至少满足 required_n_features_，且为偶数维度
        n_pad = max(
            self.required_n_features_ - X.shape[1],
            1 if X.shape[1] % 2 != 0 else 0
        )
        if n_pad > 0:
            new_X = np.column_stack((X, np.zeros((X.shape[0], n_pad))))
        else:
            new_X = X.copy()  # 避免修改原始数据
        result = op.array(new_X[:, self.flat_pair])
        # 打印输出
        return result


    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    # 测试数据：3列
    X = np.array([
        [1.0, 2.0, 3.0],
        [5.0, 6.0, 7.0]
    ])

    print("=== 测试手动配对 ===")
    pairer = OPNsPairer([(0, 2), (1, 3)])  # 需要4列，自动补零
    try:
        result = pairer.fit_transform(X)
        print(pairer)
        print("转换成功！维度:", len(result.elements), "x", len(result.elements[0]))
        print("第一个样本:", result.elements[0])  # 应为 [OPNs(1.0,3.0), OPNs(2.0,0.0)]
        print("第二个样本:", result.elements[1])  # 应为 [OPNs(5.0,7.0), OPNs(6.0,0.0)]
    except Exception as e:
        print("转换失败:", str(e))