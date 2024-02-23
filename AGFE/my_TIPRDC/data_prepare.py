"""
本模块用于对Adult数据集进行加载并进行预处理
"""
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np


class Data:
    def __init__(self, target='aac084', u='aac009') -> None:  # 离退休状态作为target属性，户口性质作为隐私属性
        """
        准备所需数据
        :param target: target任务标签，即下游任务标签
        :param u: 敏感属性，即extractor任务的标签
        """
        self.data = pd.read_excel('../医保_个人基本信息.xlsx')
        self.data.dropna(axis=1, how='any', inplace=True)  # 丢弃所有含空值的属性
        self.data = self.data.iloc[1:, :]  # 去掉第一行属性中文名
        self.data.loc[:, 'aac006'] = self.data.loc[:, 'aac006'].astype('int')  # 将出生日期转化为int
        self.u = self.data.loc[:, u]
        self.target = self.data.loc[:, target]

        cat_col = ['aaa027', 'aac003', 'bab306', 'sfz_bm', 'aac001', 'aac004', 'aac005', 'aac058', 'aac161', 'aac999',
                   'aae100', 'akc021', 'bac062']
        num_col = ['aac006']
        target_col = ['aac084']
        u_col = ['aac009']

        scaler = MinMaxScaler()
        onehot_encoder = OneHotEncoder()
        ordinal_encoder = OrdinalEncoder()

        target_pipeline = ColumnTransformer(
            [('num', scaler, num_col),
             ('cat', onehot_encoder, cat_col),
             ('target', ordinal_encoder, target_col)]
        )

        extractor_pipeline = ColumnTransformer(
            [('num', scaler, num_col),
             ('cat', onehot_encoder, cat_col),
             ('u', ordinal_encoder, u_col)]
        )

        self.dataset_target = target_pipeline.fit_transform(self.data).toarray()
        self.dataset_extractor = extractor_pipeline.fit_transform(self.data).toarray()

        # print(self.dataset_target)
        # print(self.dataset_extractor)

        self.train_dataset_target, self.test_dataset_target = train_test_split(self.dataset_target, test_size=0.2, random_state=42)
        self.train_dataset_extractor, self.test_dataset_extractor = train_test_split(self.dataset_extractor, test_size=0.2, random_state=42)

        self.train_features_target = self.train_dataset_target[:, :-1]
        self.train_label_target = self.train_dataset_target[:, -1]

        self.test_features_target = self.test_dataset_target[:, :-1]
        self.test_label_target = self.test_dataset_target[:, -1]

        self.train_features_extractor = self.train_dataset_extractor[:, :-1]
        self.train_label_extractor = self.train_dataset_extractor[:, -1]

        self.test_features_extractor = self.test_dataset_extractor[:, :-1]
        self.test_label_extractor = self.test_dataset_extractor[:, -1]

        # 获取数据基本信息
        self.num_of_train = self.train_dataset_target.shape[0]
        self.num_of_test = self.test_dataset_target.shape[0]
        self.num_of_features_target = self.train_dataset_target[:, :-1].shape[1]
        self.num_of_features_extractor = self.train_dataset_extractor[:, :-1].shape[1]  # extractor任务特征个数
        self.num_of_label_target = len(self.target.value_counts())    # target任务标签个数
        self.num_of_label_u = len(self.u.value_counts())

        # Adult数据集代码
        if "Dataset" == "Adult":
            # 读取数据
            self.train_dataset = self.get_data('../Adult/adult.data', isTest=False)
            self.test_dataset = self.get_data('../Adult/adult.test', isTest=True)
            # 确定target任务和extractor任务的标签
            self.target = target
            self.u = u
            # 数据处理
            self.train_dataset_target, self.train_dataset_extractor, self.test_dataset_target, self.test_dataset_extractor \
            = self.data_process(self.train_dataset, self.test_dataset, self.target, self.u)
            # 获取数据基本信息
            self.num_of_train = self.train_dataset.shape[0]     # 训练集大小
            self.num_of_test = self.test_dataset.shape[0]       # 测试集大小
            self.num_of_features_target = self.train_dataset_target[:, :-1].shape[1]        # target任务特征个数
            self.num_of_features_extractor = self.train_dataset_extractor[:, :-1].shape[1]  # extractor任务特征个数
            self.num_of_label_target = len(self.train_dataset[self.target].value_counts())    # target任务标签个数
            self.num_of_label_u = len(self.train_dataset[self.u].value_counts())          # extractor任务标签个数(敏感属性取值个数)

            # target任务
            self.train_features_target = self.train_dataset_target[:, :-1]      # 训练集X
            self.train_label_target = self.train_dataset_target[:, -1]          # 训练集y
            self.test_features_target = self.test_dataset_target[:, :-1]        # 测试集X
            self.test_label_target = self.test_dataset_target[:, -1]            # 测试集y

            # extractor任务
            self.train_features_extractor = self.train_dataset_extractor[:, :-1]    # 训练集X
            self.train_label_extractor = self.train_dataset_extractor[:, -1]        # 训练集y
            self.test_features_extractor = self.test_dataset_extractor[:, :-1]      # 测试集X
            self.test_label_extractor = self.test_dataset_extractor[:, -1]          # 测试集y

    @staticmethod
    def get_data(filepath, isTest=False):
        """
        读取数据
        :param filepath: adult.data/adult.test文件的存放位置
        :param isTest: 读取的数据是否是测试集
        :return: 去除空值后的数据，以dataframe形式返回
        """
        line_list = []
        with open(filepath) as lines:
            if isTest:
                next(lines)
            for line in lines:
                l = line.split(',')
                if '?' in l:
                    print('yes')
                line_list.append(l)

        data = pd.DataFrame(line_list,
                            columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                     'marital-status', 'occupation', 'relationship',
                                     'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                     'native-country', 'label'])

        # 丢弃最后一行空行
        data = data.iloc[:-1, :]

        for col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            data[col] = data[col].astype(int)

        for col in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country',
                    'sex', 'label']:
            data[col] = data[col].astype(str)
            data[col] = data[col].apply(lambda x: x.strip(' '))
            data[col] = data[col].apply(lambda x: None if x == '?' else x)

        data['label'] = data['label'].apply(lambda l: l.strip())

        data.dropna(axis='index', how='any', inplace=True)

        data['sex'] = data['sex'].apply(lambda x: 0 if x == 'Male' else 1)

        data['label'] = data['label'].apply(lambda x: 0 if x.strip('.') == '<=50K' else 1)

        return data

    @staticmethod
    def data_process(train_dataset, test_dataset, target, u):
        """
        去除target、u后，将剩余属性作为特征
        分别将target、u作为target任务和extractor的标签
        数据预处理：
        1. 对于类别型属性转化为one-hot编码
        2. 对于数值型属性进行Min-Max归一化
        :param train_dataset: 读取的训练数据，dataframe
        :param test_dataset: 读取的测试数据，dataframe
        :param target: target任务标签
        :param u: extractor任务标签
        :return: (train_dataset_target, train_dataset_extractor, test_dataset_target, test_dataset_extractor), 都是array
        """

        X_train_raw = train_dataset.drop(columns=[target, u])
        y_train_extractor = np.array(train_dataset[u])
        y_train_target = np.array(train_dataset[target])

        X_test_raw = test_dataset.drop(columns=[target, u])
        y_test_extractor = np.array(test_dataset[u])
        y_test_target = np.array(test_dataset[target])

        num_col = ['age', 'fnlwgt', 'education-num', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']
        cat_col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'label']

        for col in (target, u):
            if col in num_col:
                num_col.remove(col)
            if col in cat_col:
                cat_col.remove(col)

        full_pipeline = ColumnTransformer([
            ("num", MinMaxScaler(), num_col),
            ("cat", OneHotEncoder(), cat_col)])

        X_train = full_pipeline.fit_transform(X_train_raw).toarray()
        X_test = full_pipeline.transform(X_test_raw).toarray()

        y_encoder = OrdinalEncoder()
        y_train_extractor = y_encoder.fit_transform(y_train_extractor.reshape(-1, 1))
        y_test_extractor = y_encoder.transform(y_test_extractor.reshape(-1, 1))

        y_encoder = OrdinalEncoder()
        y_train_target = y_encoder.fit_transform(y_train_target.reshape(-1, 1))
        y_test_target = y_encoder.transform(y_test_target.reshape(-1, 1))

        train_dataset_target = np.concatenate([X_train, y_train_target], axis=1)
        train_dataset_extractor = np.concatenate([X_train, y_train_extractor], axis=1)
        test_dataset_target = np.concatenate([X_test, y_test_target], axis=1)
        test_dataset_extractor = np.concatenate([X_test, y_test_extractor], axis=1)

        return train_dataset_target, train_dataset_extractor, test_dataset_target, test_dataset_extractor

    @staticmethod
    def load_array(data_arrays, batch_size, is_train=True):
        """
        以batch加载数据
        :param data_arrays: (X_array, y_array)
        :param batch_size: batch的大小
        :param is_train: 传入的数据是否是训练集，如果是则会随机打乱顺序
        :return: 数据迭代器，每次返回一个batch的数据
        """
        dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
        if is_train:
            dataset = dataset.shuffle(buffer_size=len(data_arrays), seed=42)
        dataset = dataset.batch(batch_size)

        return dataset


if __name__ == '__main__':
    data = Data()
    # data = Data(target='label', u='sex')
    print(data.train_dataset_target.shape)
    print(data.test_dataset_target.shape)
    print(data.num_of_features_target)
    print(data.train_dataset_extractor.shape)
    print(data.test_dataset_extractor.shape)
    print(data.num_of_features_extractor)

    print(data.num_of_label_target)
    print(data.num_of_label_u)

    print(data.train_features_target)

    data_iter = data.load_array((data.train_features_extractor, data.train_label_extractor), batch_size=5, is_train=True)
    for X, y in data_iter:
        print(X)
        print(y)
        break

