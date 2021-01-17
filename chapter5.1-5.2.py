#20201026
#第五章 模型评估与改进
#两个方面进行模型评估
#1 交叉验证：更可靠的评估泛化性能的方法
#2 评估分类和回归性能的方法：在默认度量之外的方法
#网格搜索：调节监督模型参数以获得最佳泛化性能的有效方法
#5.1 交叉验证
#评估泛化性能的统计学方法 比单次划分训练集和测试集的方法更加稳定、全面
#数据被多次划分 并且需要训练多个模型
#常用：k折交叉验证（通常取5或者10）
#将数据划分为5部分 每一部分叫做折 接下来训练一系列模型
#使用第1折作为测试集其他折作为训练集来训练第一个模型 以此类推

#5.1.1 sckit_learn中的交叉验证
#在iris数据集上对LogisticRegression进行评估
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target,cv=5)#默认执行5折交叉验证
print("Cross-validation scores: {}".format(scores))

#总结交叉验证精度的一种常用方法是计算平均值
print("Average cross-validation score: {:.2f}".format(scores.mean()))

#5.1.2 交叉验证的优点
#模型需要对数据集中所有样本的泛化能力都很好 才能让所有的交叉验证得分都很高
#对数据进行多次划分 还可以提供我们的模型对训练集选择的敏感信息
#对数据的使用更加高效 更多的数据通常可以得到更为精确的模型

#缺点
#增加了计算成本

#交叉验证不是一种构建可应用于新数据的模型的方法 交叉验证不会返回一个模型
#目的只是评估给定算法在特定数据集上训练后的泛化性能

#5.1.3 分层k折交叉验证和其他思路
#划分数据使得每个折中类别之间的比例与整个数据集中的比例相同
#1 对交叉验证的更多控制
#想在一个分类数据集上使用标准k折交叉验证来重现别人的结果
from sklearn.datasets import load_iris
iris = load_iris()
print("Iris labels:\n{}".format(iris.target))
#前1/3是类别0 中间1/3是类别1 后1/3是类别2

#导入KFold分类器类 并用想要使用的折数来将其实例化
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

#将kflod分离器对象作为cv参数传入cross_val_score
print("Cross-validation scores:\n{}".format(
      cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

kfold = KFold(n_splits=3)
print("Cross-validation scores:\n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
#可以验证使用3折交叉验证效果很差
#在iris数据集中每个折对应一个类别 因此学不到任何内容

#解决问题的另一个办法是将数据打乱来代替分层 以打乱样本标签的排序
#通过将KFlod的shuffle参数设为True来实现 将数据打乱还需要固定random_state以获得可重复的打乱结果
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("Cross-validation scores:\n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

#2 留一法交叉验证
#看作是每折只包含单个样本的k折交叉验证
#对于每次划分，选择单个数据点作为测试集 非常耗时 对于大新数据集来说
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
logreg = LogisticRegression()
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))

#3 打乱划分交叉验证
#每次划分为训练集取样train_size个点 为测试集取样test_size个点 将这一划分方法重复n_iter次

import mglearn
mglearn.plots.plot_shuffle_split()
#对10个点进行打乱划分，其中train_size=5、test_size=2、n_iter=4

#将数据集划分为50%的训练集和50%的测试集 共运行10次迭代
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Cross-validation scores:\n{}".format(scores))

#打乱划分交叉验证可以在训练集和测试集大小之外独立控制迭代次数 还允许每次迭代中仅使用部分数据 通过控制test_size+train_size之和不等于1来实现

#4 分组交叉验证
#适用于数据中的分组高度相关时 
#eg：构建一个从人脸图片中识别情感的系统 
#目标:构建一个分类器 能够正确识别未包含在数据集中的人的情感
#为了准确评估模型对新的人脸的泛化能力 必须确保训练集和测试集中包含不同人的图像

#数据分组常用与医疗 拥有来自同一名病人的多个样本 但想要将其泛化到新的病人

#示例：用到了一个由groups数组指定分组的模拟数据集
from sklearn.model_selection import GroupKFold
from sklearn.datasets import make_blobs
# 创建模拟数据集
X, y = make_blobs(n_samples=12, random_state=0)
# assume the first three samples belong to the same group,
# then the next four, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("Cross-validation scores:\n{}".format(scores))

#对于每次划分 每个分组都是整体出现在训练集或者测试中

mglearn.plots.plot_label_kfold()
#用GroupKFold进行依赖于标签的划分
