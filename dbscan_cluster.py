'''
使用DBSCAN对不同季节提取典型特征
'''

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import pyplot as plt
from color import color_list_v2 as color_list
import os
import time
plt.figure(figsize=(10,5)) #设置保存尺寸  width height


def read_data(file_path:str)->np.array:
    '''
    解析csv文件，返回对应矩阵，这里前三列为地区、年份、月份，需要舍去
    
    Args:
    file_path:某一个地区的的csv文件路径
    '''
    with open(file_path,"r",encoding="utf-8") as f:
        month_risk_array=np.loadtxt(f,dtype=np.float32,skiprows=1,delimiter=",")
        month_risk_array=month_risk_array[:,3:] #舍弃前三列，这里使用切片，共享内存,取定得列依据实际作出修改
    return month_risk_array



def cluster(X,eps:float,min_samples:int,n_jobs:int=None,distance_perm:int=2)->tuple:
    '''
    dbscan 聚类
    
    Args:
    X:有一系列样本组成的矩阵
    min_samples:minPoints
    eps:领域距离
    n_jobs:并行计算，一般用不到
    distance_perm:距离的计算公式

    Returns:np.ndarray 去除异常点的剩余曲线的平均值
    '''
    cluster_model=DBSCAN(eps=eps,min_samples=min_samples,p=distance_perm,n_jobs=n_jobs)
    cluster_model.fit(X)
    cluster_label=cluster_model.labels_#获取聚类结果，这里返回一个1-D array,-1代表异常值，其他数代表聚类结果
    # print(cluster_label)
    cluster_label_mask=(cluster_label!=-1) #将聚类结果为-1的样本剔除，然后在剩余样本中计算平均值，作为典型模式
    positive_sample=X[cluster_label_mask]
    positive_sample=positive_sample.mean(axis=0) #沿着第一个维度进行平均值的计算
    #如果不需要平均值直接返回
    
    return positive_sample,cluster_label


def plot_dbscan_cluster(original_data,label,method:str="all",save_path=None):
    '''
    绘制dbscan的聚类结果图

    Args:
    original_data:原始数据
    chose_col:选择的数据列
    lable:聚类结果标签
    method:绘图方法，如果是all,则将所有曲线绘制在一张图中，并且用不同的颜色标记，如果是each ，则每一类分开绘图
    save_path:保存路径
    '''

    data=original_data
    n=data.shape[0]
    if method=="all":
        for index in range(n):
            plt.plot(data[index,],"r-o",linewidth=1,markersize=2,c=color_list[index],label="kind: %s"%(str(label[index])))
        plt.legend()

    elif method=="each":
        number=np.unique(label)
        size=len(number)
        print(number)
        rows=max(1,int(np.sqrt(size)))
        cols=size//rows
        while rows*cols<size: #make sure size<=rows*cols
            cols+=1
 
        for i in range(size):
            current_label=number[i]
            mask=(label==current_label)
            current_data=data[mask,]
            
            plt.subplot(rows,cols,i+1)
            plt.title("kind:%s"%(str(current_label)))
            n=len(current_data)
            for index in range(n):
                line=current_data[index]
                plt.plot(line,"r-o",linewidth=1,c=color_list[index],markersize=2)
            plt.xlim((1,32))
            plt.ylim((0,np.max(original_data)*1.2)) 
    if save_path:
        plt.savefig(save_path,dpi=600) #600dpi 保存
    plt.show()



def main(file_path=None,area=None):
    # if not file_path:
    #     file_path="sun/handled_data-002-of-200.csv"  #这个是一个地区得文件名


    if not os.path.exists("cluster_image/%s"%area):
        os.makedirs("cluster_image/%s"%area)

    # _,file_name=file_path.rsplit("/",1) #或取文件名字，作为图片得名字
    now=int(time.time())
    image_path=os.path.join("cluster_image/%s"%area,"%s.png"%now)
    #这里需要根据月份拆分数据，如果是每年的数据没有月份丢失，直接切片，否则就亚从文件读取判断
    month_risk_array=read_data(file_path)

    # print(month_risk_array)

    #将数据归一化 如果有需要可以加上，这里提供了两种，0-1归一化和标准化
    # scaler=MinMaxScaler()
    # standared_risk=scaler.fit_transform(month_risk_array.T)
    size,features=month_risk_array.shape


    #计算聚类参数
    eps=np.max(month_risk_array)*0.1*np.sqrt(features)
    min_points=size//5
    
    _,label=cluster(month_risk_array,eps=eps,min_samples=min_points)  #或取分类标签
    
    plot_dbscan_cluster(month_risk_array,label=label,method="each",save_path=image_path) #根据分类标签绘图

if __name__=="__main__":
    main("sun/handled_data-002-of-200.csv",2)

#如果想要对所有地区聚类，可以遍历地区得数据所在得目录，然后用for循环即可
#代码如下，取消注释运行即可
# import os
# data_dir="" #具体填写你的所有地区所在得目录
# files=(os.path.join(data_dir,item) for item in os.listdir(data_dir)) #这里就是遍历目录，然后返回文件名得列表，然后循环
# for index,file in enumerate(files,start=1):
#     try:
#         main(file,index)
#     except Exception:
#         pass
