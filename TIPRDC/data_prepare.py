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
    def __init__(self, df, target, u) -> None:  # 离退休状态作为target属性，户口性质作为隐私属性
        """
        准备所需数据
        :param target: target任务标签，即下游任务标签
        :param u: 敏感属性，即extractor任务的标签
        """
        self.data = df
        self.data.dropna(axis=1, how='any', inplace=True)  # 丢弃所有含空值的属性
        # self.data.loc[:, 'aac006'] = self.data.loc[:, 'aac006'].astype('int')  # 将出生日期转化为int
        self.u = self.data.loc[:, u]
        self.target = self.data.loc[:, target]

        cat_col = ['aaa027', 'aac003', 'bab306', 'sfz_bm', 'aac001', 'aac004', 'aac005', 'aac058', 'aac161', 'aac999',
                   'aae100', 'akc021', 'bac062']
        num_col = ['aac006']
        target_col = [target]
        u_col = [u]

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
