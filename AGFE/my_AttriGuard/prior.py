"""
本模块用于存储defense想要attack求得的数据标签分布(称为defense的先验)
一般使用已公布用户数据中数据标签的分布作为defense先验
要么就使用均匀分布作为defense先验
"""


class Defense_Prior:
    def __init__(self, p):
        self.p = p  # 若p=[0.5, 0.5]，则是有50%的人收入在50K以下，50%的人收入在50K以上
