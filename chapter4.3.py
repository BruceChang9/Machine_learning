#20201021
#chapter 4 数据表示与特征工程
#4.3 交互特征与多项书特征
#对于线性模型 添加原始数据的交互特征和多项式特征
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

bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))

which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)

encoder.fit(which_bin)

X_binned = encoder.transform(which_bin)
print(X_binned[:5])

print("X_binned.shape: {}".format(X_binned.shape))#变换后包含十个特征

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
#想要向分箱数据上的线性模型添加斜率 可以重新加入原始特征（x轴）得到11维数据集
import numpy as np
X_combined = np.hstack([X, X_binned])
print(X_combined.shape)

reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='linear regression combined')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.plot(X[:, 0], y, 'o', c='k')
#使用分箱特征和单一全局斜率的线性回归
#模型在每个箱子中都学到了一个偏移 还学到了一个斜率 只有一个x轴特征 就只有一个斜率

#更希望每个箱子都有一个不同的斜率：添加交互特征或者乘积特征 这个特征是箱子指示符与原始特征的乘积
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)
#数据集现在有20个特征
reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression product')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
#每个箱子具有不同斜率的线性回归

#分箱是扩展连续特征的一种方法 另一种是使用原始特征的多项式 对于x 可以考虑x**2、x**3...
from sklearn.preprocessing import PolynomialFeatures

#包含直到x ** 10的多项式:
#默认的"include_bias=True"添加恒等于1的常数项
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

print("X_poly.shape: {}".format(X_poly.shape))

#比较x_ploy和x的元素
print("Entries of X:\n{}".format(X[:5]))
print("Entries of X_poly:\n{}".format(X_poly[:5]))

#获取特征的语义 给出每个特征的指数
print("Polynomial feature names:\n{}".format(poly.get_feature_names()))

#具有10次多项式特征的线性回归
reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")

#多项式特征在这个一维数据上得到了非常平滑的拟合 但高次多项式在边界上或数据很少的区域可能有极端的表现

#在原始数据上学到的核SVM模型 没有任何变换
from sklearn.svm import SVR

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
#对于RBF核的SVM 使用不同gamma参数的对比
#使用更加复杂的模型（核SVM） 能够学到一个与多项式回归的复杂度类似的预测结果 且不需要进行显式的特征变换

#观察波士顿房价数据集
#首先加载数据 然后缩放到0和1之间
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split\
    (boston.data, boston.target, random_state=0)

#缩放数据
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#提取多项式特征和交互特征 次数最高为2
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)#需要由最多两个原始特征的乘积组成的所有特征
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_poly.shape: {}".format(X_train_poly.shape))
#原始数据13个特征 现在扩展到105个交互特征 
#得到输入特征和输出特征的确切对应关系
print("Polynomial feature names:\n{}".format(poly.get_feature_names()))
#第一个新特征为常数特征

#对Ridge在有交互特征的数据上和没有交互特征的数据上的性能进行对比
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled, y_train)
print("Score without interactions: {:.3f}".format(
    ridge.score(X_test_scaled, y_test)))
ridge = Ridge().fit(X_train_poly, y_train)
print("Score with interactions: {:.3f}".format(
    ridge.score(X_test_poly, y_test)))
#在使用Ridge时 交互特征和多项式特征对性能有很大的提升 如果使用更加复杂的模型（eg：随机森林） 情况稍有不同
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print("Score without interactions: {:.3f}".format(
    rf.score(X_test_scaled, y_test)))
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print("Score with interactions: {:.3f}".format(rf.score(X_test_poly, y_test)))
#即使没有额外特征 随机森林的性能要由于Ridge 添加交互特征和多项式特征实际上会略微降低其性能