#20200928
#3.1 无监督学习的类型
#数据变换与聚类
#无监督变换：降维 为了可视化将数据降为二维
#无监督变换：找到“构成”数据的各个组成部分 对文本文档集合进行主题提取 找到每个文档中讨论的未知主题 并学习每个文档中出现了哪些主题

#聚类算法：将数据划分成不同的组 每组包含相似的物项

#3.3预处理与缩放
#StanderScaler：确保每个特征的均值为0 方差为1 使所有特征都位于同一量级
#MinMaxScaler:移动数据 使所有特征都刚好位于0到1之间
#RobustScaler：确保每个特征的统计属性都位于同一范围 但使用的是中位数和四分位数 而不是平均值和方差
#Normalizer：对每个数据点进行缩放 使得特征向量的欧氏距离长度等于1 适用于只有数据方向是重要的 而特征向量的长度无关紧要

#将核SVM（svc）应用在cancer数据集上 使用MinMaxScaler来预处理数据
#加载数据集 将其分为训练集和测试集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_test.shape)

#导入实现预处理的类 将其实例化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#使用fit方法拟合缩放器Scaler 并将其应用于训练数据 对缩放器调用fit时只提供x_train 不需要y_train
scaler.fit(X_train)

#为了应用之前学习的变换 （对训练数据进行实际缩放） 使用缩放器的transform方法
#变换数据
X_train_scaled = scaler.transform(X_train)
#在缩放之前和之后分别打印数据集属性
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
    X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    X_train_scaled.max(axis=0)))

#变换后的数据形状与原始数据相同 特征知识发生了移动和缩放
#再对test集合进行变换
X_test_scaled = scaler.transform(X_test)
#在缩放之后打印测试数据的属性
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))

#MinMaxScaler（以及其他所有缩放器）总是对训练集和测试集应用完全相同的变换 
#transform方法总是减去训练集的最小集 然后除以训练集的范围 可能与测试集的最小值和范围不同

#为了使得监督模型能够在测试集上运行 对训练集和测试集应用完全相同的变换很重要

#一般要先fit一个模型 再将其transform 但有更高效的方法
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#1 依次调用fit和transform
X_scaled = scaler.fit(X).transform(X)

#2 结果相同，但计算更加高效
X_scaled_d = scaler.fit_transform(X)
'''

#预处理对监督学习的作用
#在原始数据上拟合SVC
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))

#Test set accuracy: 0.94

#先用MinMaxScaler对数据进行缩放 再拟合SVC 
#使用0-1缩放进行预处理
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#在缩放后的训练数据上学习SVM
svm.fit(X_train_scaled, y_train)

#在缩放后的测试集上计算分数
print("Scaled test set accuracy: {:.2f}".format(
    svm.score(X_test_scaled, y_test)))

#Scaled test set accuracy: 0.97

#先用StandarScaler对数据进行缩放 再拟合SVC 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#在缩放后的训练数据上学习SVM
svm.fit(X_train_scaled, y_train)

#在缩放后的测试集上计算分数
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))

#SVM test accuracy: 0.96
