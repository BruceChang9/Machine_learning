#20201020
#chapter 4 数据表示与特征工程
#4.2 分箱、离散化、线性模型与树
#数据表示的最佳方法不仅取决于数据的语义 还取决于所使用的模型种类
#线性模型与基于树（决策树、梯度提升树和随机森林）的模型是两种成员
#在处理不同的特征表示时就具有非常不同的性质
#在wave数据集上比较线性回归和决策树
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="decision tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="linear regression")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
#线性模型只能对线性关系建模 对于单个特征的情况就是直线
#决策树可以构建更为复杂的数据模型 但强烈依赖于数据表示
#特征分箱（离散化）：使得线性模型在连续数据上变得更加强大 将其划分为多个特征

#假设将特征的输入范围划分成固定个数的箱子 那么数据点就可以用它所在的箱子来表示
#用linspace函数创建11个元素 从而创建10个箱子 即两个连续边界之间的空间
bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))
#第一个箱子包含特征取值在-3到-2.4之间的所有数据点 以此类推
#记录每个数据点所属的箱子
which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])
#将wave数据集中单个连续输入特征变换为一个分类特征 用于表示数据点所在的箱子
#利用preprocessing模块的onehotencoder将离散特征变换为one-hot编码
from sklearn.preprocessing import OneHotEncoder
#使用onehotencoder进行变换
encoder = OneHotEncoder(sparse=False)
#encoder.fit找到which_bin中的唯一值
encoder.fit(which_bin)
#transform创建one-hot编码
X_binned = encoder.transform(which_bin)
print(X_binned[:5])

print("X_binned.shape: {}".format(X_binned.shape))#变换后包含十个特征

#在one-hot编码后的数据上构建新的线性模型和新的决策树模型
line_binned = encoder.transform(np.digitize(line, bins=bins))#将x轴的一个点 变成10维的点 以便进行预测

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
#在分箱特征上比较线性回归和决策树
#虚线和实现完全重合 说明线性回归和决策树做出了完全相同的预测
#对于每个箱子 均预测一个常数值
#比较分箱前后 线性模型更加灵活 而决策树灵活性降低
#决策树可以学习如何分箱对预测这些数据最为有用
#对于有些特征与输出的关系非线性的 分箱是提高建模能力的好方法