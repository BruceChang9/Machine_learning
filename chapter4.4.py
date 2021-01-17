#20201021
#chapter 4 数据表示与特征工程
#4.4 单变量非线性变换
#虽然基于树的模型只关注特征的顺序 但线性模型和神经网络依赖于每个特征的尺度和分布
#如果在特征和目标之间存在非线性关系 建模变得十分困难 尤其是回归问题
#log和exp函数可以帮助调节数据的相对比例 从而改进线性模型或神经网络的学习效果
#大部分模型都在每个特征大致遵循高斯分布时表现最好 也即每个特征的直方图应该具有类似于钟形曲线的性质
#在处理整数计数数据是 变换非常有用

#使用一个模拟的计数数据集
import numpy as np
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
#计算每个值的出现次数 数值的分布将变得更清楚（第一个特征）
print("Number of feature appearances:\n{}".format(np.bincount(X[:, 0])))
#数字2出现68次 将计数可视化
import matplotlib.pyplot as plt
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='b')
plt.ylabel("Number of appearances")
plt.xlabel("Value")

#特征2和3具有类似的性质 泊松分布
#尝试拟合一个岭回归模型
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Test score: {:.3f}".format(score))
#Ridge无法真正捕捉x和y之间的关系 应用对数变换可能有用
#由于数据中包括0 不能直接用log 而是计算log(x+1)
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
#变换后数据分布的不对称性变小 也不再具有非常大的异常值
plt.hist(np.log(X_train_log[:, 0] + 1), bins=25, color='gray')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
#对第一个特征取值进行对数变换后的直方图

#在新数据上构建一个岭回归 可以得到更好的拟合
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("Test score: {:.3f}".format(score))

#通常来说 只有一部分特征应该进行变换 有时每个特征的变换方式也各不相同
#对基于树的模型而言 这种变换并不重要 但对线性模型来说 至关重要
#对回归的目标变量y进行变换优势也是个好主意

#分箱、多项式和交互项都对模型在给定数据集上的性能有很大影响 对于复杂度较低的模型更是这样
#相反 基于树的模型通常能够自己发现重要的交互项 大多数情况下不需要显式地变换数据
#其他模型 例如 SVM 最近邻 神经网络 有时会从分箱、多项式和交互项中受益
#但其效果不如线性模型那么明显