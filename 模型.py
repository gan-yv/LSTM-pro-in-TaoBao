
import pandas as pd  # 导入Pandas库，用于数据处理和分析
import numpy as np  # 导入NumPy库，支持高效的数组计算
from numpy.random import rand  # 从NumPy随机模块导入rand函数
from numpy import array  # 导入array函数，用于创建数组
from numpy import hstack  # 导入hstack函数，用于水平堆叠数组
import matplotlib as mpl  # 导入Matplotlib库的基础模块
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘图
import seaborn as sns  # 导入Seaborn库，基于Matplotlib的统计数据可视化库
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于划分训练集和测试集
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 导入缩放器，用于数据标准化
from keras.preprocessing.sequence import TimeseriesGenerator  # 导入时间序列生成器，用于处理时间序列数据
import tensorflow as tf  # 导入TensorFlow库，用于构建和训练深度学习模型
from tensorflow.keras import optimizers  # 导入优化器模块
from keras.models import Sequential  # 导入顺序模型
from keras.layers import LSTM  # 导入长短期记忆层（LSTM）
from keras.layers import Bidirectional  # 导入双向层
from keras.layers import GRU  # 导入门控循环单元（GRU）
from keras.layers import Dense  # 导入全连接层
from keras.layers import RepeatVector  # 导入重复向量层
from keras.layers import TimeDistributed  # 导入时间分布层
from keras.layers import Dropout  # 导入Dropout层，用于防止过拟合
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.metrics import mean_squared_error  # 导入均方误差函数，用于评估模型性能
from tqdm import tqdm  # 导入tqdm库，用于显示进度条




# 数据读取与检查
# 读取/显示数据
df_shangai = pd.read_csv(r"E:\大学\各类比赛竞赛\商赛正大\编造数据3.csv")


print('前五行数据：')
print(df_shangai.head())  # 显示DataFrame的前五行数据

# 获取DataFrame的形状，返回行数和列数
print('获取DataFrame的形状，返回行数和列数：')
print(df_shangai.shape)

# 获取DataFrame的列名
print('获取DataFrame的列名：')
print(df_shangai.columns)

# 获取DataFrame中每一列的数据类型
print('获取DataFrame中每一列的数据类型：')
print(df_shangai.dtypes)

# 检查DataFrame中是否存在重复的行
print('检查DataFrame中是否存在重复的行：')
print(df_shangai.duplicated().sum())

# 计算DataFrame中每一列缺失值的数量
print('计算DataFrame中每一列缺失值的数量：')
print(df_shangai.isnull().sum())

# 生成DataFrame的描述性统计信息
print('生成DataFrame的描述性统计信息：')
print(df_shangai.describe())

# 将DataFrame的索引设置为'time'列，并选择多个特定列进行绘图
df_shangai.set_index('month')[['shipin','dianzan','pinglun','fenxiang','shoucang','renshu']].plot(subplots=True)

# 手动关闭绘图窗口
plt.close()

# 绘制'O3'自相关图
pd.plotting.autocorrelation_plot(df_shangai['xsl'])
plt.title('xsl autocorrelation:')
plt.show()
plt.close()  # 关闭图形窗口




# 从DataFrame中选择第1到第299行的数据，创建一个新的DataFrame
df_shangai_1 = df_shangai.iloc[1:, ]


# 使用Seaborn绘制'O3'列的折线图，x轴为'date'列
sns.lineplot(x=df_shangai['month'], y=df_shangai['xsl'], data=df_shangai)
plt.show()
plt.close()  # 关闭图形窗口


# 绘制DataFrame的相关系数热图，显示变量之间的相关性
sns.set(rc={'figure.figsize': (20, 10)})
sns.heatmap(df_shangai.corr(), annot=True, cmap='Purples')
plt.title('Heatmap of co-relation between variables', fontsize=16)
plt.show()
plt.close()  # 关闭图形窗口

# 遍历DataFrame中的每一列并显示进度条
for i in tqdm(df_shangai.columns, desc="Processing columns"):
    df_shangai.plot(kind='scatter', x=i, y='xsl')
    plt.show()
    plt.close()  # 关闭图形窗口

# 使用Seaborn绘制DataFrame的成对图，显示变量之间的关系
sns.pairplot(df_shangai)
plt.show()
plt.close()  # 关闭图形窗口

# 遍历DataFrame中的每一列并显示进度条
for i in tqdm(df_shangai.columns, desc="Plotting boxplots"):
    df_shangai.boxplot(column='xsl', by=i, figsize=(20, 12))
    plt.show()
    plt.close()  # 关闭图形窗口

# 使用Seaborn绘制'O3'的核密度估计图
sns.displot(df_shangai, x="xsl", kind="kde")
plt.show()
plt.close()  # 关闭图形窗口

# 显示DataFrame的前五行数据
print(df_shangai.head())
# 从DataFrame中删除'date'列
df_shangai = df_shangai.drop(['month'], axis=1)

# 显示更新后的DataFrame
print(df_shangai)

# 显示DataFrame的摘要信息，包括数据类型、非空值计数和内存使用情况
df_shangai.info()




























#开始训练
'''
# 假设df_shangai是已经加载的DataFrame
# 将日期时间列转换为多个数值特征列
if 'date' in df_shangai.columns:
    df_shangai['date'] = pd.to_datetime(df_shangai['date'])
    df_shangai['Year'] = df_shangai['date'].dt.year
    df_shangai['Month'] = df_shangai['date'].dt.month
    df_shangai['Day'] = df_shangai['date'].dt.day
    df_shangai['Hour'] = df_shangai['date'].dt.hour
    # 可以选择移除原始的日期时间列
    df_shangai.drop('date', axis=1, inplace=True)
'''
# 训练集X,Y
# X特征，Y目标
# 从DataFrame中提取目标变量'O3'的值，作为训练集的Y
Y_train = df_shangai["xsl"].values

# 从DataFrame中除了'O3'列，提取特征变量，作为训练集的X
X_train = df_shangai.drop(["xsl"], axis=1).values

# 展示
# 展示

print("显示Y")
print(Y_train)
print("显示X")
print(X_train)



# 归一化
# 创建MinMaxScaler对象，用于将特征缩放到指定范围
# #缩放到0-1
scaler = MinMaxScaler()

# 使用训练数据拟合缩放器，并对特征变量进行缩放
X_train = scaler.fit_transform(X_train)
# 展示
print("显示Y")
print(Y_train)
print("显示X缩放")
print(X_train)


'''
输入参数：
    xtrain：完整的训练特征集，通常是一个二维数组或矩阵，其中每一行代表一个样本，每一列代表一个特征。
    ytrain：完整的训练目标（标签）集，通常是一个一维数组，包含与 xtrain 中样本对应的目标值。
    x：一个包含0和1的数组，表示哪些特征应该被包含在模型中。1表示选择该特征，0表示不选择。
    opts：一个字典，包含交叉验证的分割信息，其中至少包含以下键值对：
    fold['xt']：训练特征集。
    fold['yt']：训练目标集。
    fold['xv']：验证特征集。
    fold['yv']：验证目标集。
输出：
    函数返回一个单一的值，即验证集上的均方根误差（RMSE），这是一个衡量模型预测误差的指标。RMSE 越小，表示模型的预测越准确。

作用：
    特征选择：通过参数 x，函数能够从完整的特征集中选择特定的特征子集进行模型训练和评估。
    模型训练：使用线性回归模型对选定的特征进行训练。
    性能评估：通过计算验证集上的 RMSE 来评估模型的性能。
    交叉验证支持：通过 opts 参数，函数支持交叉验证，允许用户传入不同的训练和验证集分割，以评估模型在不同数据子集上的表现。
'''
def error_rate(xtrain, ytrain, x, opts):
    # 参数设置
    fold = opts['fold']
    xt = fold['xt']  # 训练特征
    yt = fold['yt']  # 训练目标
    xv = fold['xv']  # 验证特征
    yv = fold['yv']  # 验证目标

    # 实例数量
    num_train = np.size(xt, 0)  # 训练集实例数量
    num_valid = np.size(xv, 0)  # 验证集实例数量

    # 定义选择的特征
    xtrain = xt[:, x == 1]  # 选择特征为1的训练特征
    ytrain = yt.reshape(num_train)  # 重新调整训练目标的形状
    xvalid = xv[:, x == 1]  # 选择特征为1的验证特征
    yvalid = yv.reshape(num_valid)  # 重新调整验证目标的形状

    # 训练模型
    mdl = LinearRegression()  # 创建线性回归模型
    mdl.fit(xtrain, ytrain)  # 拟合训练数据

    # 预测
    ypred = mdl.predict(xvalid)  # 对验证集进行预测
    error = mean_squared_error(yvalid, ypred, squared=False)  # 计算均方根误差

    return error  # 返回误差值



'''
这个 Fun 函数是一个自定义的目标函数，用于在特征选择过程中评估不同特征子集的性能。它结合了模型的预测误差和特征选择的权重，以提供一个综合的成本度量。以下是函数的输入输出、作用、关键参数和调试方法的详细说明：

输入参数：
    xtrain：完整的训练特征集，通常是一个二维数组或矩阵，其中每一行代表一个样本，每一列代表一个特征。
    ytrain：完整的训练目标（标签）集，通常是一个一维数组，包含与 xtrain 中样本对应的目标值。
    x：一个包含0和1的数组，表示哪些特征应该被包含在模型中。1表示选择该特征，0表示不选择。
    opts：一个字典，包含交叉验证的分割信息，其中至少包含以下键值对：
    fold['xt']：训练特征集。
    fold['yt']：训练目标集。
    fold['xv']：验证特征集。
    fold['yv']：验证目标集。
输出：
    函数返回一个单一的值，即计算得到的成本，这是一个综合了模型预测误差和特征选择的权重的度量。

作用：
    综合评估：通过结合模型的预测误差（error_rate）和特征选择的权重，提供一个综合的成本度量。
    特征选择：帮助确定最优的特征子集，以在保持模型性能的同时减少特征的数量。
关键参数：
    alpha：误差的权重，用于调整预测误差在成本计算中的重要性。默认值为0.99。
    beta：特征选择的权重，用于调整特征数量在成本计算中的重要性。默认值为1 - alpha。
    max_feat：原始特征的数量，用于标准化特征选择的权重。
    num_feat：选定特征的数量，用于计算特征选择的权重。
调试方法：
    检查输入数据：确保 xtrain 和 ytrain 的格式正确，且 x 是一个包含0和1的数组。
    验证 error_rate 函数：确保 error_rate 函数能够正确计算模型的预测误差。
    调整 alpha 和 beta：通过调整这两个参数，可以改变成本函数中误差和特征数量的相对重要性，以适应不同的需求。
    检查返回值：确保函数返回的成本值是合理的，可以通过打印中间变量的值来帮助调试
'''
def Fun(xtrain, ytrain, x, opts):
    # 参数设置
    alpha = 0.9  # 误差的权重
    beta = 1 - alpha  # 特征选择的权重
    
    # 原始特征的数量
    max_feat = len(x)
    
    # 选定特征的数量
    num_feat = np.sum(x == 1)
    
    # 如果没有选择特征，则成本设为1
    if num_feat == 0:
        cost = 1
    else:
        # 获取误差率
        error = error_rate(xtrain, ytrain, x, opts)
        # 目标函数
        cost = alpha * error + beta * (num_feat / max_feat)
        
    return cost  # 返回计算得到的成本


'''
这个 init_position 函数用于初始化一个优化算法中的种群位置矩阵。这通常用于进化算法，如遗传算法或粒子群优化（PSO）算法，其中需要随机生成初始解。

输入参数：
    lb：一个二维数组，表示每个维度的下界。lb[0, d] 是第 d 个维度的下界。
    ub：一个二维数组，表示每个维度的上界。ub[0, d] 是第 d 个维度的上界。
    N：整数，表示要生成的个体数量，即种群大小。
    dim：整数，表示每个个体的维度数，即问题空间的维度。
输出：
    函数返回一个 N 行 dim 列的矩阵 X，其中每个元素是一个随机数，表示种群中每个个体在每个维度上的初始位置。

作用：
    初始化种群：为优化算法生成初始种群，每个个体的位置是随机的，且在问题的可行范围内。
    随机性：确保种群的多样性，为算法的搜索提供广泛的起点。
关键参数：
    lb 和 ub：定义了每个维度的搜索空间范围，这对算法能否有效搜索解空间至关重要。
    N：种群大小影响算法的搜索能力和收敛速度。
    dim：问题空间的维度决定了每个个体的复杂性。
调试方法：
    检查边界：确保 lb 和 ub 中的值是合理的，并且 ub 中的值不小于 lb 中的值。
    验证种群大小和维度：确保 N 和 dim 是正确的，并且 lb 和 ub 的维度与 dim 匹配。
    检查随机性：确保 rand() 函数能够生成均匀分布的随机数。
    输出检查：打印或检查返回的 X 矩阵，确保其形状和值在预期范围内。
'''
def init_position(lb, ub, N, dim):
    # 创建一个N行dim列的零矩阵，用于存储初始化位置
    X = np.zeros([N, dim], dtype='float')
    
    # 对每个个体和每个维度进行初始化
    for i in range(N):
        for d in range(dim):
            # 在给定的下界和上界之间生成随机值
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    
    return X  # 返回初始化位置矩阵

'''
init_velocity 函数用于初始化粒子群优化（PSO）算法中的粒子速度。以下是函数的输入输出、作用、关键参数和调试方法的详细说明：

输入参数：
    lb：一个二维数组，表示每个维度的下界。lb[0, d] 是第 d 个维度的下界。
    ub：一个二维数组，表示每个维度的上界。ub[0, d] 是第 d 个维度的上界。
    N：整数，表示要生成的粒子数量，即种群大小。
    dim：整数，表示每个粒子的维度数，即问题空间的维度。
输出：
    函数返回三个值：

    V：一个 N 行 dim 列的矩阵，表示粒子的初始速度。
    Vmax：一个 1 行 dim 列的矩阵，表示每个维度上的最大速度。
    Vmin：一个 1 行 dim 列的矩阵，表示每个维度上的最小速度。
作用：
    初始化速度：为粒子群优化算法中的每个粒子生成初始速度，这些速度在问题的可行速度范围内随机生成。
    定义速度界限：确定每个维度上的最大和最小速度，以防止粒子速度过大或过小。
关键参数：
    lb 和 ub：定义了每个维度的位置搜索空间范围，这对确定速度界限至关重要。
    N：种群大小影响算法的搜索能力和收敛速度。
    dim：问题空间的维度决定了每个粒子的速度向量的复杂性。
调试方法：
    检查边界：确保 lb 和 ub 中的值是合理的，并且 ub 中的值不小于 lb 中的值。
    验证种群大小和维度：确保 N 和 dim 是正确的，并且 lb 和 ub 的维度与 dim 匹配。
    检查速度范围：确保计算出的最大和最小速度是合理的，并且速度范围是对称的。
    输出检查：打印或检查返回的 V、Vmax 和 Vmin 矩阵，确保其形状和值在预期范围内。
'''
def init_velocity(lb, ub, N, dim):
    # 创建一个N行dim列的零矩阵，用于存储初始化速度
    V = np.zeros([N, dim], dtype='float')
    
    # 创建最大和最小速度的零矩阵
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    
    # 计算最大和最小速度
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2  # 最大速度为上下界之差的一半
        Vmin[0, d] = -Vmax[0, d]  # 最小速度为最大速度的相反数
        
    # 对每个个体和每个维度进行初始化速度
    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()  # 在最小和最大速度之间生成随机值
            
    return V, Vmax, Vmin  # 返回初始化速度矩阵以及最大和最小速度


'''
这个 binary_conversion 函数用于将实数矩阵 X 中的每个元素转换为二进制形式（0或1），基于给定的阈值 thres。以下是函数的输入输出、作用、关键参数和调试方法的详细说明：

输入参数：
    X：一个 N 行 dim 列的实数矩阵，表示需要转换的原始数据。
    thres：一个实数，表示转换为二进制的阈值。
    N：整数，表示矩阵 X 的行数，即个体数量。
    dim：整数，表示矩阵 X 的列数，即每个个体的维度数。
输出：
    函数返回一个 N 行 dim 列的二进制矩阵 Xbin，其中每个元素是0或1。

作用：
    二进制转换：将实数矩阵转换为二进制矩阵，这在某些优化问题中是必要的，特别是在需要将连续变量离散化的场景中。
    阈值处理：使用阈值来确定每个元素的二进制值，大于阈值的转换为1，否则为0。
关键参数：
    thres：阈值是决定元素是否转换为1的关键参数。
    X：需要转换的原始实数矩阵。
调试方法：
    检查输入矩阵：确保 X 是一个实数矩阵，并且其形状为 N x dim。
    验证阈值：确保 thres 是一个合理的实数，并且可以在 X 的值范围内。
    检查输出矩阵：打印或检查返回的 Xbin 矩阵，确保其形状和值在预期范围内，即所有元素都是0或1。
'''
def binary_conversion(X, thres, N, dim):
    # 创建一个N行dim列的零矩阵，用于存储二进制转换后的结果
    Xbin = np.zeros([N, dim], dtype='int')
    
    # 对每个个体和每个维度进行二进制转换
    for i in range(N):
        for d in range(dim):
            # 如果X中对应值大于阈值，则转换为1，否则转换为0
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    
    return Xbin  # 返回二进制转换后的矩阵



'''
这个 boundary 函数用于确保一个值 x 在给定的上下界 lb 和 ub 之间。如果 x 超出了这些界限，函数会将其调整到最近的界限值。以下是函数的输入输出、作用、关键参数和调试方法的详细说明：

输入参数：
    x：一个数值，表示需要检查和调整的值。
    lb：一个数值，表示下界。
    ub：一个数值，表示上界。
输出：
    函数返回调整后的值 x，确保其在 [lb, ub] 范围内。

作用：
    边界检查：确保输入值 x 不会超出指定的上下界，这在优化算法中非常重要，以避免无效的解。
    值调整：如果 x 超出界限，函数会将其调整到最近的界限值，确保返回的值是有效的。
关键参数：
    lb：下界，限制 x 的最小值。
    ub：上界，限制 x 的最大值。
调试方法：
    检查输入类型：确保 x、lb 和 ub 都是数值类型（如整数或浮点数）。
    验证边界：确保 lb 小于或等于 ub，否则函数的逻辑可能不正确。
    输出检查：打印或检查返回的 x 值，确保其在 [lb, ub] 范围内。
    边界情况测试：测试 x 等于 lb 和 ub 的情况，确保函数能够正确处理这些边界值。
'''
def boundary(x, lb, ub):
    # 检查x是否超出下界
    if x < lb:
        x = lb  # 如果x小于下界，则将x设置为下界
        
    # 检查x是否超出上界
    if x > ub:
        x = ub  # 如果x大于上界，则将x设置为上界
    
    return x  # 返回调整后的x值

'''
这个 jfs 函数是一个完整的粒子群优化（PSO）算法实现，用于特征选择。它结合了二进制转换、适应度计算和PSO的更新规则来找到最优的特征子集。以下是函数的输入输出、作用、关键参数和调试方法的详细说明：

输入参数：
    xtrain：完整的训练特征集，通常是一个二维数组或矩阵，其中每一行代表一个样本，每一列代表一个特征。
    ytrain：完整的训练目标（标签）集，通常是一个一维数组，包含与 xtrain 中样本对应的目标值。
    opts：一个字典，包含算法的配置参数，如个体数量 N、最大迭代次数 T 以及可选参数如惯性权重 w、加速因子 c1 和 c2。
输出：
    函数返回一个字典 pso_data，包含以下内容：

    sf：选择的特征索引。
    c：适应度曲线。
    nf：选择的特征数量。
作用：
    特征选择：使用PSO算法来确定最优的特征子集，以提高模型的性能。
    二进制转换：将PSO算法中的位置信息转换为二进制形式，以便于特征选择。
    适应度评估：通过 Fun 函数计算每个粒子的适应度，即特征子集的性能。
关键参数：
    N：个体数量，影响算法的搜索能力和收敛速度。
    T：最大迭代次数，决定算法的运行时间。
    w、c1、c2：PSO算法的参数，影响粒子的更新规则。
    thres：二进制转换阈值，决定特征是否被选中。
调试方法：
    检查输入数据：确保 xtrain 和 ytrain 的格式正确，且 opts 包含必要的参数。
    验证参数范围：确保 lb 和 ub 的值是合理的，并且 thres 在 [0, 1] 范围内。
    检查适应度函数：确保 Fun 函数能够正确计算适应度。
    输出检查：打印或检查返回的 pso_data 字典，确保其包含所有预期的键和值。
'''
#参数需要重新设计
def jfs(xtrain, ytrain, opts):
    # 参数设置
    ub    = 1  # 上界
    lb    = 0  # 下界
    thres = 0.5  # 二进制转换阈值
    w     = 0.9  # 惯性权重
    c1    = 2    # 加速因子
    c2    = 2    # 加速因子
    
    N        = opts['N']  # 个体数量
    max_iter = opts['T']  # 最大迭代次数
    
    # 可选参数
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2'] 
    
    dim = np.size(xtrain, 1)  # 特征维度
    
    # 如果下界是单个值，则扩展到适当的维度
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # 初始化位置和速度
    X             = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim) 
    
    # 初始化适应度和其他变量
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')  # 全局最佳位置
    fitG  = float('inf')  # 全局最佳适应度
    Xpb   = np.zeros([N, dim], dtype='float')  # 个人最佳位置
    fitP  = float('inf') * np.ones([N, 1], dtype='float')  # 个人最佳适应度
    curve = np.zeros([1, max_iter], dtype='float')  # 适应度曲线
    t     = 0  # 迭代计数器
    
    while t < max_iter:
        # 二进制转换
        Xbin = binary_conversion(X, thres, N, dim)
        
        # 计算适应度
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)  # 计算适应度
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :]  = X[i, :]  # 更新个人最佳位置
                fitP[i, 0] = fit[i, 0]  # 更新个人最佳适应度
            if fitP[i, 0] < fitG:
                Xgb[0, :]  = Xpb[i, :]  # 更新全局最佳位置
                fitG      = fitP[i, 0]  # 更新全局最佳适应度
        
        # 记录结果
        curve[0, t] = fitG.copy()
        print("Iteration:", t + 1)
        print("Best (PSO):", curve[0, t])
        t += 1
        
        for i in range(N):
            for d in range(dim):
                # 更新速度
                r1 = rand()
                r2 = rand()
                V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xgb[0, d] - X[i, d]) 
                # 边界检查
                V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                # 更新位置
                X[i, d] = X[i, d] + V[i, d]
                # 边界检查
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
    
    # 最终最佳解的二进制转换
    Gbin = binary_conversion(Xgb, thres, 1, dim) 
    Gbin = Gbin.reshape(dim)
    
    pos = np.asarray(range(0, dim))    
    sel_index = pos[Gbin == 1]  # 选择的特征索引
    num_feat = len(sel_index)  # 选择的特征数量
    
    # 创建返回数据字典
    pso_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return pso_data  # 返回特征选择结果

































# 导入必要的库
from sklearn.model_selection import train_test_split

# 将特征数据和目标变量进行划分
xtrain, xtest, ytrain, ytest = train_test_split(X_train, Y_train, test_size=0.3, shuffle=True)

# 创建一个字典，用于存储训练集和测试集的数据
fold = {
    'xt': xtrain,  # 训练特征集
    'yt': ytrain,  # 训练目标集
    'xv': xtest,   # 测试特征集
    'yv': ytest    # 测试目标集
}

# 设置参数
c1  = 2         # 认知因子，控制个体向个人最佳位置移动的速度
c2  = 2         # 社会因子，控制个体向全局最佳位置移动的速度
w   = 0.9       # 惯性权重，影响粒子在当前位置的速度
k   = 3         # KNN中的k值，表示考虑的邻居数量
N   = 5     # 粒子群体的数量
T   = 50       # 最大迭代次数

# 创建一个选项字典，方便后续调用
opts = {
    'k': k,      # KNN的k值
    'fold': fold,# 包含训练集和测试集的数据
    'N': N,      # 粒子数量
    'T': T,      # 最大迭代次数
    'w': w,      # 惯性权重
    'c1': c1,    # 认知因子
    'c2': c2     # 社会因子
}



# 使用选择的特征进行特征选择
fmdl  = jfs(X_train, Y_train, opts)  # 调用JFS函数进行特征选择
sf    = fmdl['sf']                    # 获取选择的特征索引

# 获取训练集和验证集的样本数量
num_train = np.size(xtrain, 0)        # 训练集样本数量
num_valid = np.size(xtest, 0)         # 验证集样本数量

# 根据选择的特征索引提取训练集和验证集的数据
x_train   = xtrain[:, sf]             # 训练集特征
y_train   = ytrain.reshape(num_train)  # 训练集目标变量，解决bug
x_valid   = xtest[:, sf]               # 验证集特征
y_valid   = ytest.reshape(num_valid)   # 验证集目标变量，解决bug

# 创建线性回归模型并进行训练
mdl       = LinearRegression()
mdl.fit(x_train, y_train)              # 拟合模型
y_pred    = mdl.predict(x_valid)       # 在验证集上进行预测

# 计算均方根误差 (RMSE)
RMSE      = mean_squared_error(y_valid, y_pred, squared=False)
print("RMSE:", RMSE)                    # 打印RMSE值

# 输出选择的特征数量
num_feat = fmdl['nf']                  # 获取选择的特征数量
print("Feature Size:", num_feat)        # 打印特征数量

# 绘制收敛曲线
curve   = fmdl['c']                    # 获取适应度曲线
curve   = curve.reshape(np.size(curve, 1))  # 重新调整曲线形状
x       = np.arange(0, opts['T'], 1.0) + 1.0  # 迭代次数数组
fig, ax = plt.subplots()               # 创建绘图对象
ax.plot(x, curve, 'o-')                # 绘制适应度曲线
ax.set_xlabel('Number of Iterations')  # 设置x轴标签
ax.set_ylabel('Fitness')                # 设置y轴标签
ax.set_title('PSO')                     # 设置图表标题
ax.grid()                               # 显示网格
plt.show()                              # 显示图表


'''
fmdl['sf']
'''
'''
print(df_shangai)
'''

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()  # 初始化输入和输出列表
    for i in range(len(sequences)):
        # 找到当前模式的结束位置
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # 检查是否超出数据集范围
        if out_end_ix > len(sequences):
            break
        # 收集输入和输出部分的模式
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)  # 将输入序列添加到列表
        y.append(seq_y)  # 将输出序列添加到列表
    return array(X), array(y)  # 返回输入和输出数组



# 计算数据集的总长度
total_dataset = len(df_shangai)

# 划分训练集和测试集，训练集占总数据的80%
df_train = df_shangai[:int(total_dataset * 0.80)]
df_test = df_shangai[int(total_dataset * 0.80):total_dataset]

# 归一化数据
scaler = MinMaxScaler()
df_shangai_train_scaled = scaler.fit_transform(df_train)  # 对训练集进行归一化
df_shangai_test_scaled = scaler.fit_transform(df_test)    # 对测试集进行归一化

# 输出训练集和测试集的数据形状
print('Data for train:', df_train.shape)
print('\nData for test:', df_test.shape)

# 输出归一化后的训练集和测试集数据
print(df_shangai_train_scaled)
print(df_shangai_test_scaled)




# 设置输入序列的时间步长和输出序列的时间步长
n_steps_in, n_steps_out = 20,1

# 将数据转换为输入/输出格式
x_train, y_train = split_sequences(df_shangai_train_scaled, n_steps_in, n_steps_out)  # 训练集
x_test, y_test = split_sequences(df_shangai_test_scaled, n_steps_in, n_steps_out)    # 测试集

# 获取特征数量
n_features = x_train.shape[2]  # 获取输入数据的特征数（最后一个维度）

# 打印训练集和测试集的形状
print("x_train")
print(x_train.shape)  # 输出训练集输入数据的形状 (样本数, 时间步长, 特征数)
print("x_test")
print(x_test.shape)   # 输出测试集输入数据的形状 (样本数, 时间步长, 特征数)
print("y_train")
print(y_train.shape)  # 输出训练集输出数据的形状 (样本数, 输出步长, 特征数)
print("y_test")
print(y_test.shape)   # 输出测试集输出数据的形状 (样本数, 输出步长, 特征数)


'''
# 打印训练集的输入数据
print(x_train)  # 显示训练集的输入数据，便于分析数据结构

'''



# 导入必要的库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# 定义LSTM模型
model = Sequential()

# 添加第一个LSTM层
# 200是输出单元的数量，activation='relu'指定激活函数为ReLU
# input_shape=(n_steps_in, n_features)指定输入数据的形状
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))

# 添加RepeatVector层
# 将LSTM的输出重复n_steps_out次，以便将其作为下一个LSTM层的输入
model.add(RepeatVector(n_steps_out))

# 添加第二个LSTM层
# 200是输出单元的数量，return_sequences=True表示该层返回序列而不是最后一个输出
model.add(LSTM(200, activation='relu', return_sequences=True))

# 添加TimeDistributed层
# 用于在每个时间步上对LSTM的输出应用Dense层
# n_features是输出的特征数量
model.add(TimeDistributed(Dense(n_features)))
'''
# 可选：编译模型（定义损失函数和优化器）
model.compile(optimizer='adam', loss='mse')
'''


# 设置训练的轮数
epoch = 3

# 编译模型
model.compile(
    optimizer='adam',                 # 使用Adam优化器
    loss='mse',                       # 损失函数为均方误差（Mean Squared Error）
    metrics=[tf.metrics.MeanAbsoluteError()]  # 额外的评估指标：平均绝对误差
)



# 查看模型概况
model.summary()



# 训练模型
history = model.fit(
    x_train,                         # 训练数据的输入
    y_train,                         # 训练数据的目标输出
    epochs=epoch,                   # 训练的总轮数
    validation_data=(x_test, y_test)  # 验证数据，用于评估模型在训练期间的表现
)



model.summary()


predictions = model.predict(x_test)
print(predictions.shape)
print(y_test.shape)

'''
# 打印训练集和测试集的形状
print("x_train")
print(x_train.shape)  # 输出训练集输入数据的形状 (样本数, 时间步长, 特征数)
# 保存整个模型
model.save("F:\\finally_zhenghe\\lstm\\lstm_model.h5",save_format='h5')
'''





# 将预测结果的形状调整为 (样本数 * 步骤数, 特征数)
predictions = np.reshape(predictions, (y_test.shape[0] * y_test.shape[1], y_test.shape[2]))

# 将真实值的形状调整为 (样本数 * 步骤数, 特征数)
real = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1], y_test.shape[2]))

# 反归一化预测结果
predictions = pd.DataFrame(predictions)  # 将预测结果转换为DataFrame格式
predictions = scaler.inverse_transform(predictions)  # 反归一化到原始范围
predictions = pd.DataFrame(predictions)  # 再次转换为DataFrame格式

# 反归一化真实值
real = pd.DataFrame(real)  # 将真实值转换为DataFrame格式
real = scaler.inverse_transform(real)  # 反归一化到原始范围
real = pd.DataFrame(real)  # 再次转换为DataFrame格式

# 打印预测结果和真实值的形状
print(predictions.shape)  # 输出预测结果的形状
print(real.shape)         # 输出真实值的形状

df_final = pd.DataFrame()

# 将实际值填充到 DataFrame
df_final['xsl'] = real[0]
df_final['shipin'] = real[1]
df_final['dianzan'] = real[2]
df_final['pinglun'] = real[3]
df_final['fenxiang'] = real[4]
df_final['shoucang'] = real[5]
df_final['renshu'] = real[6]





# 将预测值填充到 DataFrame
df_final['xsl_pred'] = predictions[0]
df_final['shipin_pred'] = predictions[1]
df_final['dianzan_pred'] = predictions[2]
df_final['pinglun_pred'] = predictions[3]
df_final['fenxiang_pred'] = predictions[4]
df_final['shoucang_pred'] = predictions[5]
df_final['renshu_pred'] = predictions[6]





# 查看 DataFrame
print(df_final)


# 绘制实际值与预测值的图形
mpl.rcParams['figure.figsize'] = (20, 10)
mpl.rcParams['axes.grid'] = False

for column in ['xsl','shipin','dianzan','pinglun','fenxiang','shoucang','renshu']:
    df_final[[column, f'{column}_pred']].plot(title=f'{column} vs {column}_pred')

plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_array

# 评估指标计算
indicator = ['xsl','shipin','dianzan','pinglun','fenxiang','shoucang','renshu']
for i in indicator:
    rmse = mean_squared_error(df_final[f'{i}'], df_final[f'{i}_pred'], squared=False)
    mse = mean_squared_error(df_final[f'{i}'], df_final[f'{i}_pred'])
    mae = mean_absolute_error(df_final[f'{i}'], df_final[f'{i}_pred'])
    '''
    mape = mean_absolute_percentage_error(df_final[f'{i}'], df_final[f'{i}_pred'])
    '''
    print(f'{i} ----> RMSE: {rmse:.4f} ; MSE: {mse:.4f} ; MAE: {mae:.4f}')
    
    '''
    ; MAPE: {mape:.4f}')
    '''


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def sensitivity_analysis(model, x_test, feature_idx, perturbation=0.1):
    """
    通过对测试集中给定特征的扰动来执行敏感性分析。
    
    参数：
    model (tf.keras.Model)：训练好的模型。
    x_test (numpy.array)：测试数据集，形状为 (样本数, 时间步长, 特征数)。
    feature_idx (int)：要扰动的特征索引。
    perturbation (float)：扰动特征的量。
    
    返回值：
    change_in_output (numpy.array)：扰动后模型输出变化。
    """
    # 创建正向和负向扰动输入
    perturbed_input_plus = x_test.copy()
    perturbed_input_minus = x_test.copy()
    perturbed_input_plus[:, :, feature_idx] += perturbation
    perturbed_input_minus[:, :, feature_idx] -= perturbation

    # 一次性进行预测
    predictions = model.predict(np.concatenate([x_test, perturbed_input_plus, perturbed_input_minus]))
    original_output, perturbed_output_plus, perturbed_output_minus = np.split(predictions, 3)

    # 计算输出变化
    change_in_output = (
        np.abs(perturbed_output_plus - original_output)
        + np.abs(original_output - perturbed_output_minus)
    )
    return change_in_output


def gradient_analysis(model, x_input, target_idx=0):
    """
    执行梯度分析，评估每个输入特征对输出预测的贡献。
    
    参数：
    model (tf.keras.Model)：训练好的模型。
    x_input (numpy.array)：单个输入样本，形状为 (1, 时间步长, 特征数)。
    target_idx (int)：目标输出索引。
    
    返回值：
    gradients (numpy.array)：模型输出相对于输入特征的梯度。
    """
    x_input_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_input_tensor)
        predictions = model(x_input_tensor, training=False)
        output = predictions[:, target_idx]  # 选择目标输出
    gradients = tape.gradient(output, x_input_tensor)
    return gradients.numpy()




# --------- 绘制实际值与预测值对比的折线图 ---------
def plot_actual_vs_predicted(df, indicators):
    """
    绘制实际值与预测值的折线图。
    
    参数：
    df (pd.DataFrame)：包含实际值与预测值的数据。
    indicators (list)：需要绘制的指标名称列表。
    """
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(indicators, 1):
        plt.subplot(len(indicators) // 2 + len(indicators) % 2, 2, i)  # 自动排列子图
        plt.plot(df[column], label=f'Actual {column}', linestyle='-', marker='o', markersize=3)
        plt.plot(df[f'{column}_pred'], label=f'Predicted {column}', linestyle='--')
        plt.title(f'{column} vs {column}_pred', fontsize=14)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------- 评估指标计算与输出 ---------
def evaluate_metrics(df, indicators):
    """
    计算并打印评估指标：RMSE、MSE、MAE。
    
    参数：
    df (pd.DataFrame)：包含实际值与预测值的数据。
    indicators (list)：需要评估的指标名称列表。
    """
    for column in indicators:
        rmse = mean_squared_error(df[column], df[f'{column}_pred'], squared=False)
        mse = mean_squared_error(df[column], df[f'{column}_pred'])
        mae = mean_absolute_error(df[column], df[f'{column}_pred'])
        print(f'{column} ----> RMSE: {rmse:.4f}; MSE: {mse:.4f}; MAE: {mae:.4f}')


# --------- 敏感性分析绘图优化 ---------
def plot_sensitivity(feature_idx, change_in_output):
    """
    绘制敏感性分析的折线图。
    
    参数：
    feature_idx (int)：特征索引。
    change_in_output (numpy.array)：扰动后输出变化。
    """
    plt.figure()
    plt.plot(change_in_output.flatten(), label=f'Feature {feature_idx}', linestyle='-', color='blue')
    plt.title(f'Sensitivity Analysis: Feature {feature_idx}', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Change in Output')
    plt.grid(True)
    plt.legend()
    plt.show()


# --------- 梯度贡献绘图优化 ---------
def plot_gradient_contributions(gradients, feature_names=None):
    """
    绘制梯度贡献的柱状图。
    
    参数：
    gradients (numpy.array)：梯度数据。
    feature_names (list)：特征名称列表，可选。
    """
    n_features = gradients.shape[-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_features), np.abs(gradients[0, 0, :]), color='orange', alpha=0.7)
    if feature_names:
        plt.xticks(range(n_features), feature_names, rotation=45, fontsize=10)
    else:
        plt.xticks(range(n_features), [f'Feature {i}' for i in range(n_features)], fontsize=10)
    plt.title('Gradient Analysis: Feature Contributions', fontsize=14)
    plt.xlabel('Feature Index')
    plt.ylabel('Gradient Magnitude')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --------- 汇总敏感性分析结果的折线图 ---------
def plot_summary_sensitivity(sensitivity_results):
    """
    绘制所有特征的敏感性分析结果的折线图。
    
    参数：
    sensitivity_results (numpy.array)：敏感性分析的变化结果。
    """
    plt.figure(figsize=(12, 8))
    for i, change_in_output in enumerate(sensitivity_results):
        plt.plot(change_in_output.flatten(), label=f'Feature {i}', alpha=0.8)
    plt.title('Summary of Sensitivity Analysis', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Change in Output')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------- 汇总梯度贡献结果的折线图 ---------
def plot_summary_gradients(gradient_results):
    """
    绘制所有特征的梯度分析结果的折线图。
    
    参数：
    gradient_results (numpy.array)：梯度分析的贡献结果。
    """
    plt.figure(figsize=(12, 8))
    for i, gradients in enumerate(gradient_results):
        plt.plot(np.abs(gradients[0, 0, :]), label=f'Feature {i}', alpha=0.8)
    plt.title('Summary of Gradient Analysis', fontsize=14)
    plt.xlabel('Feature Index')
    plt.ylabel('Gradient Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------- 代码调用示例 ---------
# 假设以下变量已经定义：
# df_final (DataFrame): 实际值与预测值数据。
# indicators (list): 指标名称列表。
# x_test (numpy.array): 测试数据集。
# model (tf.keras.Model): 已训练的模型。

# 1. 绘制实际值与预测值对比
plot_actual_vs_predicted(df_final, ['xsl','shipin','dianzan','pinglun','fenxiang','shoucang','renshu'])

# 2. 评估指标计算
evaluate_metrics(df_final, ['xsl','shipin','dianzan','pinglun','fenxiang','shoucang','renshu'])

# 3. 敏感性分析和梯度分析（假设 x_test, model 已定义）
sensitivity_results = []
gradient_results = []
for feature_idx in range(x_test.shape[2]):
    # 敏感性分析
    change_in_output = sensitivity_analysis(model, x_test, feature_idx)
    sensitivity_results.append(change_in_output)
    plot_sensitivity(feature_idx, change_in_output)

    # 梯度分析
    sample_input = x_test[0:1, :, :]  # 单个样本
    gradients = gradient_analysis(model, sample_input, target_idx=0)
    gradient_results.append(gradients)
    plot_gradient_contributions(gradients)

# 4. 汇总敏感性分析与梯度分析结果
plot_summary_sensitivity(np.array(sensitivity_results))
'''plot_summary_gradients(np.array(gradient_results))'''



import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --------- 绘制实际值与预测值对比的折线图 ---------
def plot_actual_vs_predicted(df, indicators):
    """
    绘制实际值与预测值的折线图。
    
    参数：
    df (pd.DataFrame)：包含实际值与预测值的数据。
    indicators (list)：需要绘制的指标名称列表。
    """
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(indicators, 1):
        plt.subplot(len(indicators) // 2 + len(indicators) % 2, 2, i)  # 自动排列子图
        plt.plot(df[column], label=f'Actual {column}', linestyle='-', marker='o', markersize=3)
        plt.plot(df[f'{column}_pred'], label=f'Predicted {column}', linestyle='--')
        plt.title(f'{column} vs {column}_pred', fontsize=14)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------- 评估指标计算与输出 ---------
def evaluate_metrics(df, indicators):
    """
    计算并打印评估指标：RMSE、MSE、MAE。
    
    参数：
    df (pd.DataFrame)：包含实际值与预测值的数据。
    indicators (list)：需要评估的指标名称列表。
    """
    for column in indicators:
        rmse = mean_squared_error(df[column], df[f'{column}_pred'], squared=False)
        mse = mean_squared_error(df[column], df[f'{column}_pred'])
        mae = mean_absolute_error(df[column], df[f'{column}_pred'])
        print(f'{column} ----> RMSE: {rmse:.4f}; MSE: {mse:.4f}; MAE: {mae:.4f}')


# --------- 输出最后一次预测值 ---------
def print_last_predictions(df, indicators):
    """
    输出最后一次预测值。
    
    参数：
    df (pd.DataFrame)：包含实际值与预测值的数据。
    indicators (list)：需要输出的指标名称列表。
    """
    print("\n最后一次预测值：")
    for column in indicators:
        actual = df[column].iloc[-1]
        predicted = df[f'{column}_pred'].iloc[-1]
        print(f'{column} -> 实际值: {actual:.4f}, 预测值: {predicted:.4f}')


# --------- 示例代码调用 ---------
# 假设以下变量已经定义：
# df_final (DataFrame): 实际值与预测值数据。
# indicators (list): 指标名称列表。
# x_test (numpy.array): 测试数据集。
# model (tf.keras.Model): 已训练的模型。

# 1. 绘制实际值与预测值对比
plot_actual_vs_predicted(df_final, ['xsl','shipin','dianzan','pinglun','fenxiang','shoucang','renshu'])

# 2. 评估指标计算
evaluate_metrics(df_final, ['xsl','shipin','dianzan','pinglun','fenxiang','shoucang','renshu'])

# 3. 输出最后一次预测值
print_last_predictions(df_final, ['xsl','shipin','dianzan','pinglun','fenxiang','shoucang','renshu'])
