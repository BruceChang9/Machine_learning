#第一章 引言
#20200921
#NumPy
import numpy as np
from scipy import sparse
eye=np.eye(4)
print("Numpy array:\n{}".format(eye))
#将Numpy数组转换为CSR格式的Scipy稀疏矩阵
sparse_matrix=sparse.csr_matrix(eye)
print("Scipy sparse CSR atrix:\n{}".format(sparse_matrix))
'''
Scipy sparse mCSR atrix:
  (0, 0)        1.0
  (1, 1)        1.0
  (2, 2)        1.0
  (3, 3)        1.0
'''

#matplotlib
import matplotlib.pyplot as plt
x=np.linspace(-10,10,100)
y=np.sin(x)
plt.plot(x,y,marker="x")

#pandas
import pandas as pd
from IPython.display import display
data={'Name':['John','Mike','Joe'],
      'Location':['New York','Paris','London'],
      'Age':[10,20,18]
      }
data_pandas=pd.DataFrame(data)
display(data_pandas)#IPython.display可以打印美观的DataFrame

#鸢尾花分类
import mglearn
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset=load_iris()#返回键和值 与字典相似

print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193]+"\n...")

#target names;feature names;data;target

#将收集好的带标签数据分成两部分。一部分用于构建机器学习模型，训练集，一部分用来评估模型性能，测试集。
#train_test_split函数可以打乱数据集并进行拆分 将75%作为训练集 25%作为测试集

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
    iris_dataset['data'], iris_dataset['target'],random_state=0)
#为了确保多次运行同一函数能够得到相同的输出，利用random_state参数指定随机数生成器的种子
#数据为x,标签为y
print("x_train shape:{}".format(x_train.shape))

#可视化数据 将NumPy数组转换成pandas DataFrame
iris_dataframe=pd.DataFrame(x_train,columns=iris_dataset.feature_names)
grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',
                      hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
#按y_train着色

#k近邻算法：k可以考虑训练集中与新数据点最近的任意k个邻居，而不是只考虑最近的那一个
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

#预测
import numpy as np
x_new=np.array([[5,2.9,1,0.2]])
print("X_news.shape:{}".format(x_new.shape))
#将花的测量数据转换为二维数组 

prediction=knn.predict(x_new)
print("prediction:{}".format(prediction))
print("Prediction target name:{}".format(
    iris_dataset['target_names'][prediction]))

#评估模型
y_pred=knn.predict(x_test)
print("Test set predictions:\n{}".format(y_pred))
print("Test set score:{:2f}".format(np.mean(y_pred == y_test)))

#小结展望
#1
#为了确保多次运行同一函数能够得到相同的输出，利用random_state参数指定随机数生成器的种子
#数据为x,标签为y
x_train,x_test,y_train,y_test=train_test_split(
    iris_dataset['data'], iris_dataset['target'],random_state=0)
#2 
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
#3 
print("Test set score；{:2f}".format(knn.score(x_test,y_test)))