#监督学习
#20200922
#分类与回归
#泛化、过拟合与欠拟合
#在在拟合模型时过分关注训练集的细节，得到了一个在训练集上表现很好、但不能泛化到新数据上的模型，那么存在过拟合
#监督学习算法
#可视化数据
import numpy as np
import matplotlib.pyplot as plt
import mglearn
x,y=mglearn.datasets.make_forge()
mglearn.discrete_scatter(x[:,0], x[:,1],y)
plt.legend(["class 0","class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("x shape；{}".format(x.shape))
#回归
x,y=mglearn.datasets.make_wave(n_samples=100)
plt.plot(x,y,'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
#威斯康星州乳腺癌数据
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
#Bunch对象与字典相似，可以用点操作符来访问对象的值
print("shape of cancer data:{}".format(cancer.data.shape))
print("feature names；\n{}".format(cancer.feature_names))
#扩展数据集，输入特征不仅包括13个测量结果，还包括这些特征之间的乘积（交互项）.像这样导出特征的方法叫做特征工程
from sklearn.datasets import load_boston
boston=load_boston()

x,y=mglearn.datasets.load_extended_boston()

#k-NN算法
mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

#1 数据分为训练集和测试集 以便评估泛化能力
from sklearn.model_selection import train_test_split
x,y=mglearn.datasets.make_forge()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

#2 导入类并实例化
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)

#3 利用训练集对分类器进行拟合
clf.fit(x_train,y_train)

#4 对测试数据进行预测
print("test set prediction；{}".format(clf.predict(x_test)))

#5 评估泛化能力的好坏
print("test set accuracy:{:.2f}".format(clf.score(x_test,y_test)))

#6 对二维数据集，可以查看决策边界，即类别0和类别1的分界线
fig,axes=plt.subplots(1,3,figsize=(10,3))
for n_neighbors,ax in zip([1,3,9],axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x,y)
    mglearn.plots.plot_2d_separator(clf,x,fill=True,eps=0.5,ax=ax,alpha=.4)
    mglearn.discrete_scatter(x[:,0],x[:,1],y,ax=ax)
    ax.set_title("{}neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
#随着邻居个数越来越多，from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(
    cancer.data,cancer.target,random_state=70)

training_accuracy=[]
test_accuracy=[]
neighbors_settings=range(1,10)#n_neighbor取值为1至10

for n_neighbors in neighbors_settings:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train,y_train)#建模
    training_accuracy.append(clf.score(x_train,y_train))
    test_accuracy.append(clf.score(x_test,y_test))
    
plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
plt.plot(neighbors_settings,test_accuracy,label="test accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()#决策边界越来越平滑，对应更简单的模型。
#k的个数选择 k越大 训练集精度下降 模型更简单 测试集精度提高 最佳位置在中间部分

#k近邻回归
mglearn.plots.plot_knn_regression(n_neighbors=1)#单一近邻
mglearn.plots.plot_knn_regression(n_neighbors=3)#多个近邻

from sklearn.neighbors import KNeighborsRegressor
x,y=mglearn.datasets.make_wave(n_samples=40)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

reg=KNeighborsRegressor(n_neighbors=3)#模型实例化

reg.fit(x_train,y_train)#利用训练数据拟合模型

print("test set predictions:\n{}".format(reg.predict(x_test)))

print("test set R^2:{:.2f}".format(reg.score(x_test,y_test)))#评估模型

fig,axes=plt.subplots(1,3,figsize=(15,4))

line=np.linspace(-3,3,1000).reshape(-1,1)

for n_neighbors,ax in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(x_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(x_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(x_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
    ax.set_title(
        "{}neighbor(s)\n train score:{:.2f}test score{:.2f}".format(
            n_neighbors,reg.score(x_train,y_train),reg.score(x_test,y_test)))
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
axes[0].legend(["model predictions","training data/target","test data/target"],loc="best")

#两个重要的参数：邻居个数和数据点之间距离的度量方法
#较小的邻居个数（3-5）可以得到较好的结果
#默认使用欧氏距离即可
#对有很多特征的数据集和大多数特征为0的数据集效果不好

#线性模型
mglearn.plots.plot_linear_regression_wave()

#线性回归 普通最小二乘
#使得对训练集的预测值与真实的回归目标值y之间的均方误差最小
from sklearn.linear_model import LinearRegression

x,y=mglearn.datasets.make_wave(n_samples=100)

x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=42)

lr=LinearRegression().fit(x_train,y_train)

print("lr.coef_:{}".format(lr.coef_))#回归系数
print("lr.intercept_:{}".format(lr.intercept_))#截距项

print("traing set score:{:.2f}".format(lr.score(x_train,y_train)))
print("test set score:{:.2f}".format(lr.score(x_test,y_test)))
#R^2均为0.6左右，说明对于一维数据可能存在欠拟合
#训练集和测试集之间的性能差异是过拟合的明显标志


#可以控制复杂度的模型：岭回归（控制过拟合）
#正则化：对模型做显式约束，以避免过拟合。岭回归用到的是L2正则化。
#惩罚系数的L2范数 也即欧式长度(L2[a,b])
#正则化就是对最小化经验误差函数上加约束，这样的约束可以解释为先验知识(正则化参数等价于对参数引入先验分布)。
#约束有引导作用，在优化误差函数的时候倾向于选择满足约束的梯度减少的方向，使最终的解倾向于符合先验知识。
import mglearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

x,y=mglearn.datasets.load_extended_boston()
x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=42)

ridge=Ridge().fit(x_train,y_train)
print("training set score:{:.2f}".format(ridge.score(x_train,y_train)))
print("test set score:{:.2f}".format(ridge.score(x_test,y_test)))
#ridge在训练集上的分数低于linearregression，但在测试集上的分数更高

mglearn.plots.plot_ridge_n_samples()
#减小alpha可以让系数受到的限制更小
#如果有足够多的训练数据，正则化变得不那么重要，岭回归和线性回归将具有同样的性能

#lasso
#L1正则化
#惩罚系数向量的L1范数 也即绝对值之和 (L1[a,b])
#使用lasso时某些系数刚好为0
#某些特征被模型完全忽略 可以看作是一种自动化的特征选择 模型更容易解释 也可以呈现模型最重要的特征
import mglearn
import numpy as np
from sklearn.linear_model import Lasso

x,y=mglearn.datasets.load_extended_boston()
x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=50)

lasso=Lasso().fit(x_train,y_train)
print("training set score:{:.2f}".format(lasso.score(x_train,y_train)))
print("test set score:{:.2f}".format(lasso.score(x_test,y_test)))
print("nummber of features used:{}".format(np.sum(lasso.coef_!=0)))
#拟合效果很差 存在欠拟合 减小alpha 同时增加max_iter的值（运行的迭代次数）
lasso001=Lasso(alpha=0.05,max_iter=1000000).fit(x_train,y_train)
print("training set score:{:.2f}".format(lasso001.score(x_train,y_train)))
print("test set score:{:.2f}".format(lasso001.score(x_test,y_test)))
print("nummber of features used:{}".format(np.sum(lasso001.coef_!=0)))

#alpha=0.1的ridge模型的预测性能与alpha=0.01的lasso模型类似 但ridge模型的所有系数都不为0

#在实践中，在两个模型中首选岭回归。
#如果特征很多，认为只有其中几个是重要的，选择Lasso更好
#如果想要一个容易解释的模型，Lasso可以给出更容易理解的模型，因为它只许选择了一部分输入特征

#用于分类的线性模型
#二分类的线性模型
#函数值小于0 预测类别-1 函数值大于0 预测类别+1
#决策边界是输入的线性函数
#线性分类算法：Logistic回归&线性支持向量机
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import mglearn

x,y=mglearn.datasets.make_forge()

fig,axes=plt.subplots(1,2,figsize=(20,10))

for model,ax in zip([LinearSVC(),LogisticRegression()],axes):
    clf=model.fit(x,y)
    mglearn.plots.plot_2d_separator(clf,x,fill=False,eps=0.5,
                                    ax=ax,alpha=.7)
    mglearn.discrete_scatter(x[:,0],x[:,1],y,ax=ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend()
#两个模型默认使用L2正则化
#对于LogisticRegression和LinearSVC，决定正则化强度的权衡参数叫做c
#c值越大 模型将尽可能将训练集拟合到最好，c值越小(强正则化) 模型更强调使系数向量接近于0
#较小的c可以让算法尽量适应大多数数据点 而较大的c值更强调每个数据点都分类正确的重要性
mglearn.plots.plot_linear_svc_regularization()

#c值非常大使得决策边界斜率很大 但可能无法掌握类别的整体分布 模型可能过拟合
#LogisticRegression
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,stratify=cancer.target,random_state=42)
logreg=LogisticRegression().fit(x_train,y_train)#默认c=1
print("training set score:{:.3f}".format(logreg.score(x_train,y_train)))
print("test set score:{:.3f}".format(logreg.score(x_test,y_test)))
#模型训练集和测试集很接近 可能存在欠拟合 增大c值

#LogisticRegression默认L2正则化 结果与Ridge相似 更强的正则化使得系数更趋向0 但不会正好为0
#如果需要解释性更强的模型 L1正则化可能更好 因为它约束模型只需要少数几个特征
#二分类的线性模型与用于回归的线性模型有很多相似之处
#模型的主要差别在于penalty参数 这个参数影响正则化 是使用所有可用特征还是只选择特征的一个子集

#用于多分类的线性模型
#“一对其余”：对每个类别都学习一个二分类模型，将这个类别与所有其他类别尽量分开，生成与类别个数一样多的二分类模型
#在对应类别上分数最高的分类器“胜出”
from sklearn.datasets import make_blobs

x,y=make_blobs(random_state=42)
mglearn.discrete_scatter(x[:,0], x[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(["class 0","class 1","class 3"])

#训练一个LinearSVC分类器：
linear_svm=LinearSVC().fit(x,y)
print("coefficient shape:",linear_svm.coef_.shape)
print("intercept shape:",linear_svm.intercept_.shape)

#直线可视化
import numpy as np
mglearn.discrete_scatter(x[:,0],x[:,1],y)
line=np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,
                                ['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim(-10,8)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(['class 0','class 1','class 2','Line class 0','Line class 1','Line class 2'],loc=(1.01,0.3))

#二维空间中所有区域的预测结果
import numpy as np
mglearn.plots.plot_2d_classification(linear_svm,x,fill=True,alpha=.7)
mglearn.discrete_scatter(x[:,0], x[:,1],y)
line=np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,
                                ['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.legend(['class 0','class 1','class 2','Line class 0','Line class 1','Line class 2'],loc=(1.01,0.3))
plt.xlabel("feature 0")
plt.ylabel("feature 1")

#线性模型：正则化参数c  回归模型：正则化参数alpha
#alpha较大 or c较小 说明模型比较简单
#数据十万或者百万样本：用LogisticRegression和Ridge模型的solver='sag'选项

#朴素贝叶斯
#朴素：特征条件独立
#训练速度更快 泛化能力比线性分类器稍差
#GaussianNB用于任意连续数据 
#文本数据分类：BernoulliNB假定输入数据为二分类数据 
#文本数据分类：MultinommialNB假定输入数据为技术数据，即每个特征代表某个对象的整数计数，比如单词在句子里出现的次数

#BernoulliNB（多重伯努利分布数据）：计算每个类别中每个特征不为0的元素个数（只有一个参数alpha 控制模型复杂度）
#MultinommialNB（多项分布数据）：计算每个类别中特征的平均值（只有一个参数alpha 控制模型复杂度）（性能优于BernoulliNB)

#alpha越大，平滑化越强，模型越简单
#alpha对模型性能不重要 但可以使精度提高

#GaussianNB（高斯分布数据）：保存每个类别中每个特征的平均值和标准差（高维数据）

#对高维稀疏数据的效果很好 对参数的鲁棒性也相对较好 常用于非常大的数据集

#BernoulliNB实现
import numpy as np
x=np.random.randint(2,size=(6,100))#6个样本100维数据
print(x[0])
y=np.array([1,2,3,4,4,5])#给出6个样本的类别

from sklearn.naive_bayes import BernoulliNB
clf=BernoulliNB()
clf.fit(x,y)
print(clf.predict(x[2:3]))#x数组的第三个样本；也可以换成x[2:4]预测x数组的第三个和第四个样本

#MultinommialNB实现
import numpy as np
x=np.random.randint(5,size=(6,100))#特征值为0-4；6个样本100维特征
print(x[0])
y=np.array([1,2,3,4,5,6])

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(x,y)
print(clf.predict(x[4:5]))

#GaussianNB实现
#鸢尾花数据为例
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x,y)
y_pred=gnb.fit(x,y).predict(iris.data)

print("Number of mislabeled points out of a total {} points:{}".format(
    iris.data.shape[0],(iris.target!=y_pred).sum()))

#wine数据集
#每行代表一种酒的样本，共有178个样本，共有14列
#第一列为类标志属性，共有三类，分别为1，2，3
#后面13列为每个样本对应属性的样本值
#对比MultinommialNB以及GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split

file_path="C:\\Users\\apple\\Desktop\Python\\Python 3 人工智能 从入门到实践\\chapter 12 从朴素贝叶斯看算法多变\\wine.data"
#以“,”分隔 选取指定的列数
x=np.loadtxt(file_path,delimiter=",",usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
y=np.loadtxt(file_path,delimiter=",",usecols=(0))
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB
Gclf=GaussianNB().fit(x_train,y_train)
Mclf=MultinomialNB().fit(x_train,y_train)
g_pred=Gclf.predict(x_test)
m_pred=Mclf.predict(x_test)

print(("GaussianNB \n Training Score:{:.2f},Testing Score:{:.2f}").format(Gclf.score(x_train,y_train),Gclf.score(x_test,y_test)))
print(("MultinomialNB \n Training Score:{:.2f},Testing Score:{:.2f}").format(Mclf.score(x_train,y_train),Mclf.score(x_test,y_test)))

#决策树
#树的每个结点代表一个问题或者包含答案的终结点
#防止过拟合：1.及早停止树的生长（预剪枝）；2.先构造树，随后删除或折叠信息很少的结点（剪枝）
#测试预剪枝效果
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(
    cancer.data,cancer.target,stratify=cancer.target,random_state=42)
tree=DecisionTreeClassifier(max_depth=4,random_state=0)#最多问四个问题 防止过拟合
tree.fit(x_train,y_train)
print("Accuary on training set:{}".format(tree.score(x_train,y_train)))
print("Accuary on test set:{}".format(tree.score(x_test,y_test)))

#可视化决策树
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file="tree.dot",class_names=["malignant","benign"],
                feature_names=cancer.feature_names,impurity=False,filled=True)

import graphviz
import numpy as np

with open("tree.dot") as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)#需要重装graphviz 功能待实现

#树的特征重要性
print("Feature importance:\n{}".format(tree.feature_importances_))

import matplotlib.pyplot as plt
def plot_feature_importances_cancer(model):
    n_features=cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
plot_feature_importances_cancer(tree)

#回归树：所有基于树的回归模型不能外推，也不能在训练数据范围之外进行预测
#树不能在训练数据的范围之外生成“新的”响应
import pandas as pd
ram_prices=pd.read_csv("C:\\Users\\apple\\Desktop\\Python\\Python机器学习基础教程\\data\\ram_price.csv")

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
#利用历史数据预测2000年以后的价格
#得到的是两个集合，分为大于2000年的测试集合和小于2000年的训练集合
data_train = ram_prices[ram_prices.date<2000]
data_test = ram_prices[ram_prices.date>2000]

#基于日期预测价格
X_train = data_train.date[:, np.newaxis]#日期
#利用对数变换得到数据和目标之间更简单的关系
y_train = np.log(data_train.price)#价格

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

#对所有数据进行预测
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

#对数变换逆运算
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

#创建的图将决策树和线性回归模型的预测结果与真实值进行对比
plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()

#线性模型对测试数据给出了很好的预测，忽略了训练数据和测试数据中更细微的变化
#树模型完美预测训练数据
#一旦输入超出了模型训练数据的范围，模型只能持续预测最后一个已知数据点

#决策树算法不需要特征预处理，比如归一化或者标准化
#特征的尺度完全不一样或者二元特征和连续特征同时存在时，决策树的效果很好

#容易过拟合，泛化能力差

#决策树集成
#集成：合并多个机器学习模型来构建更强大模型的方法
#随机森林
#对每棵树的结果取平均来降低过拟合
#在每个结点处，算法随机选择特征的一个子集，并对其中一个特征寻找最佳测试
#所有树均不同：1.自主采样：构造每棵决策树的数据集略有不同；2.特征选择：每棵树中的每次划分均是基于特征的不同子集
#五棵树组成的随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2)#五棵树
forest.fit(X_train, y_train)
#将每棵树学到的决策边界可视化 也将总预测可视化
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
                                alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
#使用更多的树 构造更平滑的边界
#一百颗树的随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

#随机森林的特征重要性
plot_feature_importances_cancer(forest)

#随机森林本质上是随机的 设置不同的随机状态可以彻底改变构建的模型
#森林的树越多，它对随机状态选择的鲁棒性越好

#梯度提升回归树（梯度提升机）
#通过合并多个决策树来构建一个更为强大的模型
#采用连续的方式构造树，每棵树试图纠正前一棵树的错误；强预剪枝
#主要思想：合并许多简单的模型，eg：深度较小的树
#相比于随机森林：对参数设置更为敏感，如果参数设置正确，模型精度更高
#学习率（learning_rate):用于控制每棵树纠正前一棵树错误的强度
#100棵树；最大深度为3；学习率：0.1
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
'''
Accuracy on training set: 1.000
Accuracy on test set: 0.965
'''
#训练集精度过高 可能存在过拟合 降低过拟合 限制最大深度来加强预剪枝或者降低学习率
#限制最大深度
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
'''
Accuracy on training set: 0.991
Accuracy on test set: 0.972
'''
#降低学习率
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
'''
Accuracy on training set: 0.988
Accuracy on test set: 0.965
'''
#特征重要性
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

plot_feature_importances_cancer(gbrt)
#先尝试随机森林，它的鲁棒性很好。
#鲁棒性：指控制系统在一定（结构，大小）的参数摄动下，维持其它某些性能的特性。
#若效果好，但预测时间长或者进一步提高精度，切换成梯度提升

#梯度提升决策树是监督学习中最强大也最常用的模型之一
#缺点：需要仔细调参，且训练时间可能较长
#不适用于高维稀疏数据
#梯度提升模型的max_depth通常设置得很小，一般不超过5

#核支持向量机
#线性模型在低维空间中可能非常受限 因为线和平面的灵活性有限
#为了使得线性模型更加灵活：添加更多的特征，例如添加输入特征的交互项或多项式

#模拟的数据集
X, y = make_blobs(centers=4, random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#线性支持向量机分类效果不理想
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#添加第二个特征的平方作为三维特征
#添加第二个特征的平方，作为一个新特征
X_new = np.hstack([X, X[:, 1:] ** 2])

from mpl_toolkits.mplot3d import Axes3D

figure = plt.figure()
#3D可视化
ax = Axes3D(figure, elev=-152, azim=-26)
#首先画出所有y==0的点，再画出所有y==1的点
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")

#在新数据的新表示中 用线性模型（三维空间的平面）将两个类别分开
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

#显示线性决策边界
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60)

ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature0 ** 2")

#对三维空间的线性SVM模型是二维空间的一个椭圆
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#向数据表示中添加非线性特征可以使得线性模型变得更强大
#核技巧：直接计算扩展特征表示中数据点之间的距离，也即是内积，而不用实际对扩展进行运算
#将数据映射到更高维空间：1.多项式核：在一定阶数内计算原始特征所有可能的多项式；2.径向基函数核（高斯核）：对应无限维的特征空间，考虑所有阶数的所有可能的多项式，但阶数越高，特征的重要性越小
#支持向量：位于类别之间边界上的那些点
#在forge数据集上训练SVM
from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
#画出支持向量
sv = svm.support_vectors_
#支持向量的类别标签由dual_coef的正负号给出
sv_labels = svm.dual_coef_.ravel()>0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#SVM调参
#gamma参数用于控制高斯核的宽度 决定了点与点之间“靠近”是指多大的距离
#C是正则化参数，限制每个点的重要性
#改变参数的变化
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
                  ncol=4, loc=(.9, 1.2))

#gamma小 高斯核的半径大 大多数点都被看作比较靠近
#小的gamma表示决策边界变化很慢 生成复杂度较低的模型
#C很小 说明每个数据点的影响范围均有限 

#RBF核SVMM应用到乳腺癌数据 默认C=1，gammma=1/n_features
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

#每个特征的max和min
import matplotlib.pyplot as plt
plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")

#每个特征有不同的数量级 对核SVM有极大的影响
#于是对每个特征进行缩放 将所有特征缩放到0-1之间


#计算训练集中每个特征的最小值
min_on_training = X_train.min(axis=0)
#计算训练集中每个特征的范围（最大值-最小值）
range_on_training = (X_train - min_on_training).max(axis=0)

#减去最小值，然后除以范围
#这样每个特征都是min=0 max=1
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))

#利用训练集的最小值和范围对测试集做相同的变换
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

#模型处于欠拟合 训练集和测试集性能非常接近
#尝试增大C或gamma来拟合更复杂的模型
svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

#SVM在低维数据和高维数据上的表现均很好 但对样本个数的缩放表现不好 10000个样本可能表现良好 但100000或更大 运行时间和内存面临挑战
#SVM的预处理数据和调参均需要很小心
#对所有特征的测量单位相似且范围差不多时可以考虑SVM
#核SVM的重要参数：正则化参数C 核的选择 核相关的参数

#神经网络
#多层感知机
#首先计算戴白哦中间过程的隐单元 再计算这些隐单元的加权求和
#在计算完每个隐单元的加权求和之后，对结果再应用一个非线性函数，将函数的结果用于加权求和
#非线性函数:校正非线性（relu） 正切双曲线（tanh）
#需要设置一个重要的参数：隐层中的结点个数 非常小的简单数据集：10 非常大复杂的数据集：10000 也可以添加多个隐层

#神经网络调参
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)#默认100个隐结点；也可以设置10个隐结点；mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
#包含100个隐单元的神经网络在two_moons数据集上都学到的决策边界

#默认的非线性是relu 可能由直线段组成
#如果想得到更加平滑的决策边界：添加更多的隐单元 添加第二个隐藏层 使用tanh非线性

#2个隐层 每个包含10个单元 relu非线性
mlp = MLPClassifier(solver='lbfgs', random_state=0,
                    hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#2个隐层 每个包含10个单元 tanh非线性
mlp = MLPClassifier(solver='lbfgs', activation='tanh',
                    random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#还可以利用L2惩罚使得权重趋于0 从而控制神经网络的复杂度
#调节L2惩罚的参数是alpha 默认值很小 （弱正则化）
#不同alpha值对two_moons数据集的影响：2个隐层的神经网络 每层包含10或者100个单元
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs', random_state=0,
                            hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                            alpha=alpha)
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(
                      n_hidden_nodes, n_hidden_nodes, alpha))
#控制神经网络复杂度的方法；隐层的个数、每个隐层中的单元个数与正则化（alpha）
#alpha越大 正则化越强 使得算法尽量适应大多数数据点
##alpha越小 正则化越弱 使得算法更强调每个数据点都分类正确的重要性

#神经网络：在开始学习之前其权重是随即设置的 这种随机初始化会影响学习到的模型
#相同参数但不同随机初始化的情况下学到的决策函数
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver='lbfgs', random_state=i,
                        hidden_layer_sizes=[100, 100])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)


#将MLPClassifier应用在乳腺癌数据上
#1 使用默认参数
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

cancer=load_breast_cancer()
print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0))) 

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

'''
Accuracy on training set: 0.94
Accuracy on test set: 0.92
'''

#MLP精度没有其他模型好：原因在于数据的缩放 类似于SVC 神经网络要求所有输入特征的变化范围相似 最理想的情况是均值为0 方差为1
#对数据进行缩放
#计算训练集中每个特征的平均值
mean_on_train = X_train.mean(axis=0)
#计算训练集中的每个特征的标准差
std_on_train = X_train.std(axis=0)

#减去平均值 然后除以标准差
#如此运算后 均值为0 标准差为1
X_train_scaled = (X_train - mean_on_train) / std_on_train
#对测试集做相同的变换 使用训练集的品均值和标准差
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

'''
Accuracy on training set: 0.991
Accuracy on test set: 0.965
'''

#给出警告 告诉我们已经达到最大迭代次数 应该增加迭代次数
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

'''
Accuracy on training set: 1.000
Accuracy on test set: 0.972
'''

#增加迭代次数提高了训练集性能 但没有提高泛化性能
#可以尝试降低模型复杂度来得到更好的泛化性能
#选择增大alpha参数（变化范围0.0001到1） 以此向权重添加更强的正则化
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

'''
Accuracy on training set: 0.988
Accuracy on test set: 0.972
'''

#显示连接输入和第一个隐层之间的权重 行对应30个输入特征 列对应100个隐单元 浅色表示较大的正值 深色代表负值
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

#如果某个特征对所有隐单元的权重均很小 那么这个特征对模型来说就“不太重要”

#优点
#能够获得大量数据中包含的信息 并构建无比复杂的模型
#给定足够的世家和数据 仔细调参 神经网络通常可以打败其他机器学习的算法

#缺点
#需要很长的训练时间 需要仔细预处理数据 
#神经网络在均匀数据上的性能最好 均匀指所有特征都具有相似的含义
#如果数据包含不同种类的特征 基于树的训练模型可能表现更好

#调参方法：
#1 创建一个大到足以过拟合的网络 确保网络可以对任务进行学习
#2 缩小网络 或者 增大alpha来增强正则化来提高泛化能力

#如何学习模型或者用来学习参数的算法：由solver决定
#默认选项是adam 大多数情况下效果很好 对数据的缩放很敏感 始终将数据缩放为均值为0 方差为1 很重要
#另一个是lbfgs 鲁棒性很好 但在大型模式或者大型数据集上的时间会比较长
#更高级的sgd 还有许多其他参数需要调节 以获得最佳结果
#建议使用adam和lbfgs

#scikit-learn重要性质：调用fit总会重置模型之前学到的所有内容
#如果在一个数据集上构建模型 之后在另一个数据集上再次调用fit 模型会忘记从第一个数据集中学到的所有内容

#5 神经网络
#5.1 神经元模型
#神经网络最基本的成分是神经元模型
#M-P神经元模型 输入信号通过带权重的链接进行传递 神经元接收到的总输入值将与神经元的阈值进行比较 然后通过激活函数处理以产生神经元的输出

#5.2 感知机与多层网络
#感知机能很容易地实现逻辑与、或、非运算
#感知机只有输出层神经元进行激活函数处理 即只拥有一层功能神经元 学习能力非常有限
#若两类模式是线性可分的 即存在一个线性超平面将它们分开；对于非线性可分问题 考虑使用多层功能神经元
#隐层/隐含层：输出层与输入层之间的一层神经元
#输入层神经元仅接受输入 不进行函数处理 隐层与输出层包含功能神经元
#神经网络:根据训练数据来调整神经元之间的 “连接权” 以及每个功能神经元的 阈值

#5.3 误差逆传播算法
#大多使用BP算法进行训练
#BP算法不仅可用于多层前馈神经网络（一般情况） 还可以用于其他类型的神经网络 例如训练递归神经网络
#BP算法基于梯度下降策略 以目标的负梯度方向对参数进行调整
#BP神经网络经常遭遇过拟合 
#1 早停：将数据分成训练集和验证集 训练集用来计算梯度 更新连接权和阈值 验证集用来估计误差 若训练集误差降低但验证集误差升高 则停止训练同时返回具有最小验证集误差的连接权和阈值
#2 正则化：在误差目标函数中增加一个用于描述网络复杂度的部分 例如：连接权与阈值的平方和

