#20201010
#3.4 降维、特征提取与流形学习
#主成分分析
#首先找到方差最大的方向 标记为“成分1” 数据中包含信息最多的方向
#再找到与第一个方向正交且包含最多信息的方向
#可以用于去除数据中的噪声影响 或者将主成分中保留的那部分信息可视化

#最常用于将高维数据集可视化

#1 将PCA应用于cancer数据集并可视化
#对每个特征分别计算两个类别的直方图
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
cancer = load_breast_cancer()
import numpy as np

fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
fig.tight_layout()

#在应用PCA之前 利用StandarScaler缩放数据 使每个特征的方差均为1：
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

#将PCA对象实例化 调用fit方法找到主成分 然后调用transform来旋转并降维
#默认下 PCA仅选装并移动数据 但保留所有的主成分
#为了降维 需要在创建PCA对象时指定想要保留的主成分个数
from sklearn.decomposition import PCA
#保留数据的前两个主成分
pca = PCA(n_components=2)
#对乳腺癌数据拟合PCA模型
pca.fit(X_scaled)

#将数据变换到前两个主成分的方向上
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

#对前两个主成分作图 按类别着色
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

#查看components的内容
print("PCA components:\n{}".format(pca.components_))

#利用热力图将系数可视化
plt.matshow(pca.components_, cmap='viridis')
#在第一个主成分中 所有特征的符号相同（均为正） 这意味着在所有特征之间存在普遍的相关性
#如果一个测量值较大的话 其他测量值可能也较大
#第二个主成分的符号有正有负 两个主成分均包含30个特征 使得坐标轴的解释变得十分困难

#2 特征提取的特征脸
#人脸的特征提取 用于人脸识别
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fix, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

#一共3023张图像 每张大小为87像素*65像素 分别属于5个不同的人
print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

#但数据有斜偏 其中包含某些人的大量图像
#计算每个目标出现的次数
import numpy as np
counts = np.bincount(people.target)
#将次数与目标名称一起打印出来
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()
        
#为了降低数据斜偏 对每个人最多只取50张图像
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

#将灰度值缩放到0到1之间 而不是在0到255之间
#以得到更好的数据稳定性
X_people = X_people / 255

#人脸识别：看某个前所未见的人脸是否属于数据库中的某个已知人物
#构建一个分类器 每个人都是单独的类别 担人脸数据库中有很多不同的人 而同一个人的图像很少
#还想要轻松添加新的人物 不需要重新训练一个大型模型
#使用单一最近邻分类器 寻找与要分类的人脸最为相似的人脸
#使用KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
#使用一个另据构建KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))
#Test set score of 1-nn: 0.42

#计算原始像素空间中的距离是很糟糕的办法
#用像素来比较两张图像时，比较的是每个像素的灰度值与另一张图像对应位置的像素灰度值
#如果使用像素距离，那么将人脸向右移动一个像素将会发生巨大的变化
#希望 使用沿着主成分方向的距离可以提高精度
#使用PCA白化选项 它将主成分缩放到相同的尺度 
#白化不仅对应于旋转数据 还对应于缩放数据使其形状是圆形而不是椭圆

#对训练数据拟合PCA对象 并提取100个主成分 然后对训练数据和测试数据进行变换
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape: {}".format(X_train_pca.shape))
#新数据有100个特征 即前100个主成分
#使用单一最近邻分类器来将图像分类
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca, y_test)))
#Test set accuracy: 0.45

#观察前几个主成分
fix, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape),
              cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))

#第一个主成分 人脸于背景的对比
#第二个主成分 人脸最半部分和右半部分的明暗程度差异
#PCA基于像素 人脸的相对位置和明暗程度对相似程度的表示很大影响
#可以将数据降维到只包含一些主成分 然后反向旋转回到原始空间

#利用10、50、100、500个成分对一些人脸进行重建并将其可视化
import mglearn
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
#在使用仅前10个主成分 仅捕捉到图片的基本特点 比如人脸方向和明暗程度

#也可以尝试使用PCA的前两个主成分 将数据集中的所有人脸在散点图中可视化
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
#如果只使用前两个主成分 数据成一团 看不到类别之间的分界

#3.4.2 非负矩阵分解
#PCA中 希望得到正交分量
#NMF得到的分量更容易解释 因为负的分量和系数可能会导致难以解释的抵消效应
#将NMF应用于模拟数据
#所有分量的地位平等
#对于两个分量的NMF，显然所有数据点都可以写成这两个分量的正数组合

#将NMF应用于人脸图像
#NMF主要参数是想要提取的分量个数
#利用越来越多分量的NMF重建三张人脸图像
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)

#反向变换数据质量与PCA相似 但稍差一些 因为PCA找到的是重建最佳方向
#尝试仅提取一部分分量15个
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fix, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("{}. component".format(i))
#上述分量均是正的 因此比PCA分量更像人脸原型

#观察分量3和7特别大的图像
compn = 3
#按第3个分量排序、绘制前10张图像
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))

compn = 7
#按第7个分量排序、绘制前10张图像
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
#分量3较大的人脸均是向右看的人脸 分量7较大的人脸均是向左看的人脸
#提取这样的模式最适合于具有叠加结构的数据：音频 基因表达 文本数据
#假设对一个信号感兴趣 由三个不同信号源合成
import mglearn
import matplotlib.pyplot as plt
S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal")
#无法观测到原始信号 只能观测到三个信号的叠加混合
#希望将混合信号分解为原始变量
#将数据混合成100维的状态
import numpy as np
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("Shape of measurements: {}".format(X.shape))

#用NMF来还原三个信号
from sklearn.decomposition import NMF
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("Recovered signal shape: {}".format(S_.shape))

#为了对比 使用PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)

#给出NMF和PCA发现的信号活动
models = [X, S, S_, H]
names = ['Observations (first three measurements)',
         'True sources',
         'NMF recovered signals',
         'PCA recovered signals']

fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                         subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
#包含来自X的100次测量中的3次 
#NMF生成的分量是没有顺序的 NMF分量的顺序与原始信号完全相同（纯属偶然）

#3.4.3 用t-SNE进行流行学习
#用于可视化的算法：流形学习算法
#允许更复杂的映射 给出更好的可视化
#很少用来生成两个以上的新特征 计算训练数据的一种新表示 但不允许变换新数据
#只能变换用于训练的数据 探索性数据分析
#思想：找到数据的一个二维表示 尽可能保持数据点之间的距离
#重点关注距离较近的点 而不是保持距离较远的点之间的

#digits数据集的示例图像
from sklearn.datasets import load_digits
digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                         subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)

#用PCA将降到二维的数据可视化 对前两个主成分作图 并按类别对数据点着色
#构建一个PCA模型
pca = PCA(n_components=2)
pca.fit(digits.data)
#将digits数据变换到前两个主成分的方向上
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    #将数据实际绘制成文本而不是散点
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
#利用前两个主成分可以将0、6和4相对较好地分开

#t-SNE不支持变换新数据 调用FIT-TRANSFORM方法替代 会构建模型并立刻返回变换后的数据
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
#使用fit_transform而不是fit,因为TSNE没有transform方法
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    #将数据实际绘制成文本而不是散点
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
#这种方法并不知道类别标签 完全是无监督的 但它能够找到数据的一种二维表示 仅根据原始空间中数据点之间的靠近程度就能够将各个类别明确分开