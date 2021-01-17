#20201021
#chapter 4 数据表示与特征工程
#4.5 自动化特征选择
#在添加新特征或处理一般的高维数据集时，最好将特征的数量减少到只包含最有用的那些特征，并删除其余特征
#判断每个特征的作用有多大：单变量统计、基于模型的选择、迭代选择（监督方法）

#4.5.1 单变量统计（对于分类问题：方差分析）
#计算每个特征和目标值之间的关系是否存在统计显著性，然后选择具有最高置信度的特征
#性质：它们是单变量的 即只单独考虑每个特征
#单变量测试：计算快 不需要构建模型 完全独立于可能想要在特征选择之后应用的模型

#单变量特征选择：分类(f_classif) 回归(f_regression)
#所有舍弃参数的方法都使用阈值来舍弃所有p值过大的特征 意味着它们不可能与目标值相关

#分类特征选择应用于cancer 同时在其中加入一些没有信息量的噪声特征
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
import numpy as np

cancer = load_breast_cancer()

#获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
#向数据中添加噪声特征
#前30个特征来自数据集 后50个是噪声
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=.5)
#使用f_classif (默认值)和SelectPercentile来选择50%的特征
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
#对训练集进行变换
X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

#使用get_support来查看哪些特征被选中 会返回所选特征的mask
import matplotlib.pyplot as plt
mask = select.get_support()
print(mask)
#将遮罩可视化——黑色为True 白色为False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

#大多数所选择的特征都是原始特征 并且大部分噪声都已被删除 但原始特征的还原并不完美
#来比较Logistic回归在所有特征上的性能与仅使用所选特征的性能
from sklearn.linear_model import LogisticRegression

#对测试数据进行变换
X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(
    lr.score(X_test_selected, y_test)))

#如果特征量太大以至于无法构建模型 或者怀疑许多特征完全没有信息量 那么单变量特征选择还是很有用的

#4.5.2 基于模型的特征选择
#使用一个监督机器学习模型来判断每个特征的重要性 并且仅保留最重要的特征
#特征选择模型需要为每个特征提供某种重要性度量 以便用这个度量对特征进行排序
#决策树和基于决策树的模型提供了feture_importances_属性，可以直接编码每个特征的重要性
#与单变量选择不同 基于模型的选择同时考虑所有特征 因此可以获取交互项
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="median")
#SelectFromModel类选出重要性度量大于给定阈值的所有特征
#为了与单变量特征选择进行对比 使用中位数作为阈值 这样就可以选择一半特征
#用包含100棵树的随机森林分类器来计算特征重要性
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

#再次查看选中的特征
mask = select.get_support()
#将遮罩可视化——黑色为True 白色为False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

#查看其性能
X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Test score: {:.3f}".format(score))
#利用更好的特征选择 性能也得到提高

#4.5.3 迭代特征选择
#在迭代特征选择中 会构建一系列模型 每个模型都使用不同数量的特征
#1 在开始没有特征 然后逐个添加特征 直到满足某个终止条件
#2 从所有特征开始 诸葛删除特征 直到满足某个终止条件
#递归特征消除：从所有特征开始构建模型 并根据模型舍弃最不重要的特征 然后使用除被舍弃特征之外的所有特征来构建一个模型
#使用之前用过的同一个随机森林模型
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
             n_features_to_select=40)

select.fit(X_train, y_train)
#将选中的特征可视化:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
#使用随机森林分类器模型的递归特征消除选择的特征

#测试使用RFE做特征选择时Logistic回归模型的精度：
X_train_rfe= select.transform(X_train)
X_test_rfe= select.transform(X_test)

score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Test score: {:.3f}".format(score))
#在RFE内使用随机森林的性能与在所选特征上训练一个Logistic回归模型得到的性能相同
#只要选择了正确的特征 线性模型的表现就与随机森林一样好

#4.6 利用专家知识
#利用这种方法可以将关于任务属性的先验知识编码到特征中，以辅助机器学习算法
#利用专家知识的特例
#预测Andereas家门口的自行车出租
#任务：对于给定的日期和时间 预测有多少人将会在Andereas的家门口租一辆自行车——这样他就知道是否还有自行车留给他

#首先将这个站点2015年8月的数据加载为一个pandas数据框
#将数据重新采样为每3小时一个数据 以得到每一天的主要趋势
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
citibike = mglearn.datasets.load_citibike()
print("Citi Bike data:\n{}".format(citibike.head()))

#对于选定的Citi Bike站点 自行车出租数量随时间变化
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
                       freq='D')
plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Rentals")
#可以清楚地区分没24小时中白天和夜间 工作日和周末的模式似乎也有很大不同
#在划分训练集和数据集时，希望使用某个特定日期之前的所有数据作为训练集 之后的为测试集
#使用前184个数据点（23天）作为训练集 后64个数据点（8天）作为测试集
#输入特征为日期和时间 输出是接下来3小时内的租车数量

#提取目标值（租车数量）
y = citibike.values
#利用"%s"将时间转换为POSIX时间
X = citibike.index.astype("int64").values.reshape(-1, 1)

#首先定义一个函数 它可以将数据划分为训练集和测试集 构建模型并将结果可视化
#使用前184个数据点用于训练 剩余的数据点用于测试
n_train = 184

#对给定特征集上的回归进行评估和作图的函数
def eval_on_features(features, target, regressor):
    #将给定特征划分为训练集和测试集
    X_train, X_test = features[:n_train], features[n_train:]
    #同样划分目标数组
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))

    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,
               ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',
             label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")

#随机森林需要很少的数据预处理 因此适合作为第一个模型
#我们使用POSIX时间特征x 并将随机森林回归传入我们的eval_on_features函数
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
plt.figure()
eval_on_features(X, y, regressor)

#在训练集上的预测结果相当好 符合随机森林的通常表现 对于测试集来说 预测结果是一条常数直线
#问题在于特征和随机森林的组合
#测试集中POSIX时间特征的值超出了训练集中特征取值的范围：测试集中是据点的数据戳要晚于训练集中的所有数据点
#树以及随机森林无法外推到训练集之外的特征范围 结果就是模型只能预测训练集中最近数据点的目标值 及最后以此观测到数据的时间

#显然可以利用专家知识
#通过观察训练数据中的租车数量图像 发现两个因素十分重要：一天内的时间与一周的星期几
#因此添加这两个特征 从POSIX时间中学不到任何东西 所以删掉这个特征
#首先仅用每天的时刻 现在的预测结果对一周内的每天都具有相同的模式
X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)
#随机森林仅使用每天的时刻做出的预测
#但预测结果没有抓住每周的模式 添加一周的星期几作为特征
#随机森林使用一周的星期几和每天的时刻两个特征做出的预测
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1),
                         citibike.index.hour.values.reshape(-1, 1)])
eval_on_features(X_hour_week, y, regressor)

#预测性能很好 实际上不需要随机森林这样复杂的模型 尝试一个更简单的模型：线性回归
from sklearn.linear_model import LinearRegression
eval_on_features(X_hour_week, y, LinearRegression())

#效果差很多 而且周期性模式看起来很奇怪：在于用整数编码一周的星期几和一天内的时间 被解释为连续变量
#通过将整数解释为分类变量（OneHotEncoder进行变换）来获取模式
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
eval_on_features(X_hour_week_onehot, y, Ridge())
#线性模型使用one——hot编码过的一周的星期几和每天的时刻两个特征做出的预测
#现在线性模式为一周内的每天都学到了一个系数 为一天内的每个时刻都学到了一个系数 也即 一周七天共享“一天内每个时刻”的模式

#利用交互特征，可以让模型为星期几和时刻的每一种组合学到一个系数
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True,
                                      include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)
#线性模型使用星期几和时刻两个特征的乘积做出的预测
#优点：可以很清楚地看到学到的内容 对每个星期几和时刻的交互项学到了一个系数 可以将模型学到的系数作图

#首先，为时刻和星期几特征船舰特征名称：
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
features =  day + hour
#然后利用get_feature_name方法对PolynomialFeatures提取的所有交互特征进行命名，并保留系数不为零的那些特征
features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]
#下面将线性模型学到的系数可视化
plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature magnitude")
plt.ylabel("Feature")
#线性模型使用星期几和时刻两个特征的成绩学到的系数

#4.7 小结与展望
#如何处理不同的数据类型（分类变量）
#1 强调了使用适合机器学习算法的数据表示方式的重要性：one——hot
#2 通过特征工程生成新特征的重要性/利用专家知识从数据中创建导出特征的可能性
#注：线性模型可能从分箱、添加多项式和交互项而生成的新特征中受益/对于更复杂的非线性模型（随机森林orSVM）可能不需要


