#20201020
#chapter 4 数据表示与特征工程
#一种常见的特征类型就是分类特征 也即离散特征
#对于某个特定应用来说 如何找到最佳数据表示 被称为特征工程
#4.1 分类变量
#4.1.1 One-Hot编码（虚拟变量）
#思想：将一个分类变量替换为一个或多个新特征 新特征取值为0和1
#将一个特征转换为n个新的特征，并删除原始的特征
#统计学中，将具有k个可能取值的分类特征编码为k-1个特征

#首先 使用pandas从逗号分隔值CSV文件中加载数据
import pandas as pd
from IPython.display import display

#文件中没有包含列名称的表头 因此我们传入header=None
#然后在“names”中显式地提供列名称
data = pd.read_csv(
    "C:/Users/apple/Desktop/Python/Python机器学习基础教程/data/adult.data", header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
#为了便于说明 之选了其中几列
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]
# IPython.display可以在Jupyter notebook中输出漂亮的格式
display(data.head())

#1 检查字符串编码的分类数据
#先检查每一列是否包含有意义的分类数据
#可能将性别填写为male或者man 希望可以用同一个类别来表示
#检查列的内容 显示唯一值及其出现次数
print(data.gender.value_counts())

#用pandas编码数据:使用get_dummies函数 自动变换所有具有对象类型（如字符串）的列或所有分类的列
print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_dummies.columns))
#连续特征没有变化 而分类特征的每个可能取值都被扩展为一个新特征

data_dummies.head()

#下面可以将DataFrame转换为Numpy数组 然后在其上训练一个机器学习模型
#在训练模型之前 注意要把目标变量（也即因变量）从数据中分离出来
#仅提取包含特征的列 这一范围包含所有特征 但不包含目标
features = data_dummies.iloc[:, 0:44]
#提取NumPy数组
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))

#现在数据的表示方式可以被scikit-learn处理
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))

#对同时包含训练数据和测试数据的数据框调用使用get_dummies函数 可以确保训练集和测试集中分类变量的表示方式相同

#4.1.2 数字可以编码分类变量
#分类特征通常用整数进行编码 他们是数字不意味着必须被视为连续特征
#如果在被编码的语义之间没有顺序关系 那么特征必须被视为离散特征
#pandas的get_dummies函数将所有数字看成是连续的 不会为其创建虚拟变量
#可以使用OneHotEncoder指定哪些变量是连续的 哪些是离散的 也可以将数据框中的数值列转换为字符串
#创建一个两列的DataFrame对象 其中一列包含字符串 另一列包含整数

#创建一个DataFrame，包含一个整数特征和一个分类字符串特征
import pandas as pd
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
display(demo_df)

#使用get_dummies只会编码字符串特征 不会改变整数特征
pd.get_dummies(demo_df)

#对数据做one-hot编码 同时编码整数特征和字符串特征
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])