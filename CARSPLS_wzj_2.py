# coding: utf-8
# Time: 2025\04\28
# Author: Zhijing Wu
# FileName: carspls_wzj
# Software: PyCharm,Jupyter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import copy,os,random

def PC_Cross_Validation(X, y, pc, cv):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)
    RMSECV = []
    
    if X.shape[1]<pc:
        pc = X.shape[1]#PLSR的主成分数不能大于X的特征数量。
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex

def Cross_Validation(X, y, pc, cv):
    '''
     x :光谱矩阵 nxm
     y :浓度阵 （化学值）
     pc:最大主成分数
     cv:交叉验证数量
     return :
            RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean

#定义自适应重加权采样函数（Adaptive reweighted sampling）
def weightsampling_in(w):
    #%输入参数是根据每次循环波长点保留率去除了（归零）绝对值较小的模型参数后
    #%的遗留模型参数。注意：w是取了绝对值后的模型参数向量。
    #%Bootstrap sampling
    w = w/np.sum(w) #同文献公式（3），计算w向量每一个元素所占权重
    N1 = w.shape[0]
    chosen_number = random.choices(range(0,N1),weights = w,k=N1)#符合梁文章的Fig.2
    #权重越大，波长点位置被选择的概率越大
    a = np.sort(chosen_number)#从小到大排序
    Vsel = np.unique(a)
    return Vsel #返回的Vsel就是ARS选择的波长点位置数据

def CARS_Cloud(X, y, num=50, f=8, cv=10):
    #CARS: Competitive Adaptive Reeighted Sampling method for variable selection
    #X: The data matrix of size m*N;
    #y: The response vector of size m*1;
    #num: The number of Monte Carlo Sampling runs.(第一个循环次数)
    #f: The maximum of principle componnents of PLSR
    #cv: The number of cross-validations
    Mx = X.shape[0]#样品数量
    Nx = X.shape[1]#样品波长点数量（特征数量）
    f = min([Mx,Nx,f])#f不超过样品数，也不超过样品特征数
    
    ratio = 0.8#建模校正集样品占比
    r0=1#初始波长点保留率
    r1=2/Nx
    
    Vsel = np.arange(0,Nx)#初始筛选波长点位置数据，
    #第一次循环取所有波长点数据，Vsel每次循环会改变数值和大小。这是一个需要重点关注的参数。
    Q = int(np.floor(Mx*ratio));#建模校正集样品数量
    W = np.zeros((Nx,num))#用于保存每次循环的PLSR模型的参数（每列），这是一个非常重要的参数
    Ratio = []#空列表，用于保存每次循环波长点保留率。
    
    WaveNum =[]#列表，用于保存下面各次循环保存的波长点数量
    
    ## 定义文章中的指数下降函数两个参数：
    #b就是文章中的k参数（公式6）；a还是a（文章公式5）
    b = np.log(r0/r1)/(num-1)
    a = r0*np.exp(b)
    x = copy.deepcopy(X)
    #主循环：
    for iter in np.arange(1,num+1):#循环50次（默认）
        Ratio.append(a*np.exp(-1*b*iter))#u*np.exp(-1*k*i):各次循环的波长点的保留率,同书P81推导
        K = int(np.round(Ratio[iter-1]*Nx))#K：每次循环保留的波长点的数量，每次循环都会变
        WaveNum = np.hstack((WaveNum, K)).astype(int)#WaveNum转为np数组，用于保存每次循环保留的波长点      #的数量
        cal_index = np.random.choice    \
            (np.arange(Mx), size=int(Q), replace=False)#从0~Mx-1中随机取Q个数，
        #且不重复
        xcal = x[np.ix_(list(cal_index),list(Vsel.astype(int)))]#按cal_index随机取样品数的光谱，
        #按Vsel取波长点数
        ycal = y[cal_index]
        
        MM = xcal.shape[0]
        NN = xcal.shape[1]
        f = min([MM,NN,f])
        pls = PLSRegression(n_components=f)#创建PLSR模型，主成分数为f,默认为8
        pls.fit(xcal,ycal)
        coef = pls.coef_#获取模型参数（1，n）反映每个波长点对y变量的相关贡献程度
        coef = coef.T#（n,1），注意n的大小每次循环会改变。
        w = np.zeros((Nx,1))#创建一个Nx*1的列向量
        w[Vsel.astype(int)] = coef#w也是一个需要重点关注的参数。
        W[:,iter-1] = w.squeeze() #将每次循环PLSR模型参数保存到W的每一列中。小w用来对模型参数进行排序
        w = np.abs(w)#对每次循环建立的PLSR模型的参数取绝对值，w全部变成正数了。
        indexw = np.argsort(-w.squeeze())##按模型参数绝对值大小，从大到小排序，indexw返回原数据在coef的
        #位置序号
        #coeff = coef[indexw];#coeff是按绝对值从大到小排序后的向量。
        KK = int(np.round(a*np.exp(-1*b*(iter+1))*Nx))
        w[indexw[KK:]]=0#只保留"波长点保留数"大小的模型参数
        #绝对值较小的模型参数归零，归零的比例为ratio
        # %+++ Eliminate some variables with small coefficients. 
        Vsel=weightsampling_in(w)#Adaptive reweighted sampling的程序实现函数%+++ Reweighted Sampling from the pool of retained variables.                 
        #Vsel= np.unique(Vsel)#用于查找并移除数组中的重复元素，并返回经过排序（从小到大）的唯一元素列表。数据量会下降            
        print('The %dth variable sampling finished.\n'%iter)    #%+++ Screen output.
    
    #Cross-Validation to choose an optimal subset;
    RMSEP = []#列表，保存下面各次循环的REMSEp数值
    #Q2_max = []#保存决定系数
    #Rpc = []#保存相关系数
    for i in range(num):
        vsel = np.where(W[:,i]!=0)
        _, rindex = PC_Cross_Validation(X[:,vsel].squeeze(), y, 20, 10)
        RMSE_mean = Cross_Validation(X[:,vsel].squeeze(), y, rindex+1, cv)
        RMSEP.append(RMSE_mean)

    RMSEP_min = np.min(RMSEP)
    indexOPT = np.argsort(RMSEP)[0]
    vsel = np.where(W[:,indexOPT]!=0)#vsel为tuple类型数据
    
    #绘图：
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(211)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('被选择的波长数量', fontsize=fonts)
    plt.title('最佳迭代次数：' + str(indexOPT) + '次', fontsize=fonts)
    plt.plot(np.arange(num), WaveNum)

    plt.subplot(212)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(num), RMSEP)
    plt.show()
    
    return vsel[0].astype(int)

