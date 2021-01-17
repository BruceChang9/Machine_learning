#20200928
#2.5 小结与展望
#最近邻
#适用小数据集 易解释

#线性模型
#可靠的首选算法 适用于非常大的数据集 也适用于高维数据

#朴素贝叶斯
#只适用分类问题 比线性模型快 适用于非常大的数据集和高维数据 精度低于线性模型

#决策树
#速度很快 不需要数据缩放 可以可视化 易解释

#随机森林
#鲁棒性很好 很强大 不需要数据缩放 不适用于高维稀疏数据

#梯度提升树
#精度比随机森林略高 但训练速度更慢 但预测速度更快 需要的内存更少 需要调节更多的参数

#支持向量机
#对于特征含义相似的中等大小的数据集很强大 需要数据缩放 对参数很敏感

#神经网络
#可以构建非常复杂的模型 尤其对于大型数据集而言 对数据缩放敏感 对参数选取敏感 

#对于新数据集 先从简单模型开始 比如线性模型 朴素贝叶斯 最近邻分类器 看能得到什么样的结果
#对数据有进一步了解后 可以考虑用于构建更复杂模型的算法