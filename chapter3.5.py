#20201012
#3.5 聚类
#聚类算法为每个数据点分配一个数字 表示这个点属于哪个簇
#3.5.1 k均值聚类
#1 将每个数据点分配给最近的簇中心 2 然后将每个簇中心设置为所分配的所有数据点的均值
#KMeans类实例化
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#生成模拟的二维数据
X, y = make_blobs(random_state=1)

#构建聚类模型
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

#找到分类的标签
print("Cluster memberships:\n{}".format(kmeans.labels_))

#也可以用predict方法为新数据点分配簇标签
#预测时 会将最近的簇中心分配给每个新数据点 但现有模型不会改变
print(kmeans.predict(X))

#3个簇的k均值算法找到的簇分配和簇中心
import mglearn
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
    markers='^', markeredgewidth=2)

#使用更多或更少的簇中心
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#使用2个簇中心：
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

#使用5个簇中心：
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

#k均值只能找到相对简单的形状 假设所有簇在某种程度上具有相同的“直径” 它总是将簇之间的边界刚好画在簇中心的中间位置
#k均值还假设所有方向对每个簇都同等重要

#PCA&NMF试图将数据点表示为一些分量之和 相反 k均值则尝试利用簇中心来表示每个数据点
#将其看作仅用一个分量来表示每个数据点 该分量由簇中心给出 这种观点称为矢量量化

#对比k均值的簇中心与PCA和NMF找到的分量
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

#为了降低数据斜偏 对每个人最多只取50张图像
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]

#将灰度值缩放到0到1之间 而不是在0到255之间
#以得到更好的数据稳定性
X_people = X_people / 255

from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

fig, axes = plt.subplots(3, 5, figsize=(8, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Extracted Components")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(
        axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
    ax[2].imshow(comp_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")

fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(8, 8))

#利用100个分量或簇中心的k均值、PCA和NMF的图像重建的对比--k均值的每张图像中仅使用了一个簇中心
fig.suptitle("Reconstructions")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
        axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca,
        X_reconstructed_nmf):

    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape))
    ax[3].imshow(rec_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")

#利用k均值做矢量量化：可以用比输入维度更多的簇来对数据进行编码

#对于two_moons数据 利用PCA和NMF 我们对这个数据无能为力 因为只有两个维度 若降到1维 会完全破坏数据的结构
#使用更多的簇中心 可以用k均值找到一种更具表现力的表示
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60,
            marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
print("Cluster memberships:\n{}".format(y_pred))
#使用10个簇中心 每个点都被分配了0到9之间的一个数字 我们可以将其看作10个分量表示的数据（有10个新特征）
#只有表示该点对应的簇中心的那个特征不为0 其他特征均为0
#利用这10维表示 可以用线性模型来划分两个半月形 而利用原始的两个特征无法做到

#将到每个簇中心的距离作为特征 还可以得到一种表现力更强的数据表示 可以利用kmmeans的transform方法来完成
distance_features = kmeans.transform(X)
print("Distance feature shape: {}".format(distance_features.shape))
print("Distance features:\n{}".format(distance_features))

#缺点：依赖于随机初始化 算法的输出依赖于随机种子
#默认下 用10种不同的随机初始化将算法运行10次 并返回最佳结果

#3.5.2 凝聚聚类
#凝聚聚类指的是许多基于相同原则构建的聚类算法
#先声明每个点是自己的簇 然后合并两个最相似的簇 直到满足某种停止准则为止
#链接准则 
#1 ward：挑选两个簇合并 使得所有簇中的方差增加最小 得到大小差不多相等的簇
#2 average：将簇中所有点之间平均距离最小的两个簇合并
#3 complete：将簇中点之间最大距离最小的两个簇合并
#ward适用于大多数数据集 如果簇中成员个数非常不同 average&complete会更好
import mglearn
mglearn.plots.plot_agglomerative_algorithm()#ward

#凝聚算法不能对新数据点做出预测 为了构造模型并得到训练集上簇的成员关系 可以改用fit_predict
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#凝聚聚类生成的层次化的簇分配以及带有编号的数据点
mglearn.plots.plot_agglomerative()

#从SciPy中导入dendrogram函数和ward聚类函数
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)
#将ward聚类应用于数据数组X
#SciPy的ward函数返回一个数组，指定执行凝聚聚类时跨越的距离
linkage_array = ward(X)
#现在为包含簇之间距离的linkage_array绘制树状图
dendrogram(linkage_array)

#在数中标记划分成两个簇或三个簇的位置
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
#树状图在底部显示数据点 然后以这些点作为叶节点绘制一棵树
#仍无法分离two_moons数据集这样复杂的形状

#3.5.3 DBSCAN
#具有噪声的基于密度的空间聚类应用
#优点：不需要用户先验地设置簇的个数 可以划分具有复杂形状的簇 还可以找出不属于任何簇的点
#相较于凝聚聚类和k均值稍慢 但仍可以扩展到相对较大的数据集

#原理：识别特征空间的“拥挤”区域中的点 在这些区域中许多数据点靠近在一起（密集区域）
#在密集区域内的点被称为核心样本 如果在距一个给定数据点eps的距离内至少有min_samples个数据点 那么这个数据点就是核心样本
#将彼此距离小于eps的核心样本放到同一个簇中
#如果距起始点的距离在eps之内的数据点个数小于min_samples 则这个点被标记为噪声
#如果在距一个给定数据点eps的距离内至少有min_samples个数据点 则这个点被标记为核心样本 并被分配一个新的簇标签
#然后访问该点的所有邻居（eps以内） 如果他们还没被分配一个簇 那么就将刚刚创建的新的簇标签分配给他
#然后选取另一个尚未被访问过的点 并重复相同的过程
#不允许对新的测试数据进行预测 使用fit_predict来执行聚类并返回簇标签

from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("Cluster memberships:\n{}".format(clusters))
#所有数据点被分配-1 这代表噪声 这是eps和min_samples默认参数设置的结果

#在eps和min_samples参数不同取值的情况下 DBSCAN找到的簇分配
mglearn.plots.plot_dbscan()
#属于簇的点是实心的 噪声显示为空心 核心样本显示为较大的标记 而边界点则显示为较小的标记
#增大eps 簇会变大 多个簇合并为一个 增大min_samples 核心点变少 更多的点被标记为噪声

#eps更重要 非常小 无核心样本 所有均为噪声 非常大 所有点形成单个簇

#min_samples决定簇的最小尺寸
#使用StandarScaler or MinMaxScaler对数据进行缩放 会更容易找到eps的较好取值

#在two_moons数据集上运行DBSCAN
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

#将数据缩放成平均值为0、方差为1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
#绘制簇分配
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
#利用默认值eps=0.5的DBSCAN找到的簇分配
#由于算法找到了想要的簇的个数 因此参数设置似乎很好
#如果将eps减小到0.2（默认0.5） 将会得到8个簇 显然太多 将eps增大到0.7 则会导致只有一个

#3.5.4 聚类算法的对比与评估
#1 用真实值评估聚类
#用于评估聚类算法相对于真实聚类的结果的指标：调整rand指数（ARI) & 归一化互信息（NMI）;均给出了定量的度量：最佳值为1；0表示不相关的聚类
#使用监督ARI分数来比较k均值；凝聚聚类和DBSCAN算法
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

#将数据缩放成平均值为0 方差为1
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})

#列出要使用的算法
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]

#创建一个随机的簇分配，作为参考
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

#绘制随机分配
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
                cmap=mglearn.cm3, s=60)
axes[0].set_title("Random assignment - ARI: {:.2f}".format(
        adjusted_rand_score(y, random_clusters)))

for ax, algorithm in zip(axes[1:], algorithms):
    #绘制簇分配和簇中心
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
               cmap=mglearn.cm3, s=60)
    ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,
                                           adjusted_rand_score(y, clusters)))
#DBSCAN最佳 分数为1
#用这种方式：精度问题在于要求分配的簇标签与真实值完全匹配 
    
from sklearn.metrics import accuracy_score

#这两种点标签对应于相同的聚类
clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]
#精度为0 因为两者标签完全不同
print("Accuracy: {:.2f}".format(accuracy_score(clusters1, clusters2)))
#调整rand分数为1 因为两者聚类完全相同
print("ARI: {:.2f}".format(adjusted_rand_score(clusters1, clusters2)))

#2 没有真实值的情况下评估聚类
#如果知道数据的正确聚类 可以使用这一信息构建一个监督模型，例如分类器
#使用ARI和NMI的指标通常仅有助于开发算法 但对评估应用是否成功没有帮助

#有一些聚类的评分指标不需要真实值 比如轮廓系数 但在实践中并不好
#轮廓分数计算一个簇的紧致度 越大越好 最大为1 虽然紧致的簇很好 但紧致度不允许复杂的形状
#利用无监督的轮廓分数在数据集上比较各种聚类算法
from sklearn.metrics.cluster import silhouette_score

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
#将数据缩放成平均值为0、方差为1
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})

#创建一个随机的簇分配，作为参考
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

#绘制随机分配
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
    cmap=mglearn.cm3, s=60)
axes[0].set_title("Random assignment: {:.2f}".format(
    silhouette_score(X_scaled, random_clusters)))

algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    #绘制簇分配和簇中心
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3,
               s=60)
    ax.set_title("{} : {:.2f}".format(algorithm.__class__.__name__,
                                      silhouette_score(X_scaled, clusters)))
    
#对于评估聚类，稍好的策略是使用基于鲁棒性的聚类指标
#指标先向数据中添加一些噪声，或者使用不同的参数设定，然后运行算法，并对结果进行比较
#思想：如果许多算法参数和许多数据扰动返回相同的结果 则很可能是可信的

#即使得到一个鲁棒性很好的聚类或者非常高的轮廓分数 但仍然不知道聚类中是否有任何语义含义 或者聚类是否反映了数据中感兴趣的方面
#要想知道聚类是否对应于感兴趣的内容 则对簇进行人工分析
    
#3.5.5 聚类方法小结
#聚类的应用与评估是一个非常定性的过程 通常在数据分析的探索阶段很有帮助
#k均值&凝聚聚类：允许指定想要的簇的数量；DBSCAN：允许用eps参数定义接近程度 从而间接影响簇的大小
    
#k均值：可以用簇的平均值来表示簇。可以看作一种分解方法，每个数据点均由其簇中心表示
#DBSCAN：可以检测到没有分配任何簇的“噪声点” 还可以帮助自动判断簇的数量；与其他不同，允许簇具有复杂的形状；有时会生成大小差别很大的簇，可能是优点or缺点
#凝聚聚类：可以提供数据的可能划分的整个层次结构 通过树状图轻松查看
    
#3.6 小结与展望
#一系列无监督学习算法，用于探索性数据分析和预处理
#分解，流行学习和聚类都是加深数据理解的重要工具

#分类 预处理
#回归、主成分分析 降维
#聚类 特征提取 特征选择