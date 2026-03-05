import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets, logger
from sklearn.preprocessing import LabelEncoder
import scipy.io
import logging

class DatasetLoader:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    @staticmethod
    def iris():
        iris = sk_datasets.load_iris()
        return iris.data, iris.target, "iris", iris.data.shape[1], len(np.unique(iris.target))
    
    @staticmethod
    def balance():
        file_path = 'data/balance-scale.data'
        data = pd.read_csv(file_path)
        name='balance'
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    
    @staticmethod
    def weather():
        file_path = r'data/weather.mat'  # 确保路径正确
        name = "weather"
        data = scipy.io.loadmat(file_path)
        # 调试：打印MAT文件中的变量名
        logger.info(f"MAT文件变量: {list(data.keys())}")
        data = data['weather']
        # 提取特征和标签（假设最后一列是标签）
        X = data[:, :-1]
        y = data[:, -1]
        # 编码标签（如果非数值）
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def hayes_roth():
        file_path = r'data/hayes-roth.dat'
        name = "hayes_roth"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def phoneme():  # k=5
        file_path = r'data/phoneme.dat'
        name = "phoneme"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)  # 返回数据和元信息
    @staticmethod
    def monk_2():
        file_path = 'data/monk-2.dat'
        name='monk_2'
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def led7digit():  # k=7
        file_path = r'data/led7digit.dat'
        name = "led7digit"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)  # 返回数据和元信息
    @staticmethod
    def appendicitis():
        file_path = r'data/appendicitis.dat'
        name = "appendicitis"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)  # 返回数据和元信息
    @staticmethod
    def ecoli():
        file_path = r'data/ecoli.dat'
        name = "ecoli"
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values  # 特征列
        y = data.iloc[:, -1].values  # 原始标签（如'cp'、'im'等字符串）
        # 添加标签编码，确保统一为数值类型
        le = LabelEncoder()
        y = le.fit_transform(y)  # 这里重新赋值给y，确保是编码后的数值类型
        return X, y, name, X.shape[1], len(le.classes_)  # 返回编码后的标签和类别数
    @staticmethod
    def pima():
        file_path = r'data/pima.dat'
        name = "pima"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values
        y = data.iloc[:, -1].values # 提取除最后一列之外的所有列作为X
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def cars():
        file_path = r'data/cars.mat'  # 确保路径正确
        name = "cars"
        data = scipy.io.loadmat(file_path)
        # 调试：打印MAT文件中的变量名
        logger.info(f"MAT文件变量: {list(data.keys())}")
        # 假设数据存储在变量'X'中（根据实际MAT文件调整）
        if 'X' in data:
            cars_data = data['X']
        else:
            raise KeyError(f"MAT文件中未找到预期的数据变量。可用变量: {list(data.keys())}")
        # 提取特征和标签（假设最后一列是标签）
        X = cars_data[:, :-1]
        y = cars_data[:, -1]
        # 编码标签（如果非数值）
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def saheart():  # k=7
        file_path = r'data/saheart.dat'
        name = "saheart"
        data = pd.read_csv(file_path)
        # Convert categorical feature to numerical
        data['Famhist'] = data['Famhist'].map({'Present': 1, 'Absent': 0})

        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values  # Target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def heart():  # k=4
        file_path = r'data/heart.dat'
        name = "heart"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_) # 返回数据和元信息    
    @staticmethod
    def cleve():
        file_path = r'data/cleve.mat'  # 确保路径正确
        name = "cleve"
        data = scipy.io.loadmat(file_path)
        # 调试：打印MAT文件中的变量名
        logger.info(f"MAT文件变量: {list(data.keys())}")
        data = data['cleve']
        # 提取特征和标签（假设最后一列是标签）
        X = data[:, :-1]
        y = data[:, -1]
        # 编码标签（如果非数值）
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)


    @staticmethod
    def cleveland():
        file_path = r'data/cleveland.dat'
        name = "cleveland"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def wine():
        file_path = r'data/wine.data'
        name = "wine"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def vowel():
        file_path = r'data/vowel.dat'
        name = "vowel"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)
    @staticmethod
    def penbased():
        file_path = r'data/penbased.dat'
        name = "penbased"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)  # 返回数据和元信息   
    @staticmethod
    def vehicle():
        file_path = r'data/vehicle.dat'
        name = "vehicle"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)  # 返回数据和元信息 
    @staticmethod
    def hepatitis():
        file_path = r'data/hepatitis.dat'
        name = "hepatitis"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)

    @staticmethod
    def segment():
        file_path = r'data/segment.dat'
        name = "segment"
        data = pd.read_csv(file_path)
        X = data.iloc[:, : -1].values  # 提取除最后一列之外的所有列作为X
        y = data.iloc[:, -1].values  # 提取最后一列作为Y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)

    @staticmethod
    def sonar():
        file_path = r'data/sonar.mat'
        name = "sonar"
        data = scipy.io.loadmat(file_path)
        # 查看MAT文件中的变量名（调试用）
        print(f"MAT文件中的变量: {list(data.keys())}")
        # 获取实际数据变量（根据输出可知为'sonar'）
        sonar_data = data['sonar']
        # 查看sonar_data的类型和结构（调试用）
        print(f"sonar_data的类型: {type(sonar_data)}")
        print(f"sonar_data的形状: {sonar_data.shape if hasattr(sonar_data, 'shape') else 'N/A'}")
        # 根据实际结构提取特征和标签
        # 假设sonar_data是二维数组，最后一列是标签
        if isinstance(sonar_data, np.ndarray) and sonar_data.ndim == 2:
            X = sonar_data[:, :-1]  # 所有行，除最后一列外的所有列
            y = sonar_data[:, -1]  # 所有行，最后一列
        else:
            # 如果sonar_data是结构体或其他格式，需要进一步解析
            print(f"sonar_data的内容: {sonar_data}")
            # 示例：假设sonar_data是字典，包含'features'和'labels'键
            X = sonar_data['features']
            y = sonar_data['labels']
        # 确保数据维度正确
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)  # 展平多维特征
        if y.ndim > 1:
            y = y.flatten()  # 确保标签是一维数组
        # 编码标签（如果标签是字符串）
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)

    @staticmethod
    def air():
        file_path = r'data/air.mat'
        name = "air"
        data = scipy.io.loadmat(file_path)
        # 查看MAT文件中的变量名（调试用）
        print(f"MAT文件中的变量: {list(data.keys())}")
        # 获取实际数据变量（根据输出可知为'air'）
        air_data = data['air']
        # 查看air_data的类型和结构（调试用）
        print(f"air_data的类型: {type(air_data)}")
        print(f"air_data的形状: {air_data.shape if hasattr(air_data, 'shape') else 'N/A'}")
        # 根据实际结构提取特征和标签
        # 假设air_data是二维数组，最后一列是标签
        if isinstance(air_data, np.ndarray) and air_data.ndim == 2:
            X = air_data[:, :-1]  # 所有行，除最后一列外的所有列
            y = air_data[:, -1]  # 所有行，最后一列
        else:
            # 如果air_data是结构体或其他格式，需要进一步解析
            print(f"air_data的内容: {air_data}")
            # 示例：假设air_data是字典，包含'features'和'labels'键
            X = air_data['features']
            y = air_data['labels']
        # 确保数据维度正确
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)  # 展平多维特征
        if y.ndim > 1:
            y = y.flatten()  # 确保标签是一维数组
        # 编码标签（如果标签是字符串）
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, y_encoded, name, X.shape[1], len(le.classes_)