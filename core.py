'''
使用GSP返回聚类结果
'''
import numpy as np
import random
from matplotlib import pyplot as plt
import time
import os
from color import color_list_v2 as color_list
plt.figure(figsize=(10,5))
class GSP:
    def __init__(self,cluster_nums,g_0,max_iter,alpha,paticals_nums,top_k,*args,**kwargs):
        '''
        Args:
        cluster_nums:聚类数目
        g_0:万有引力常数初始值
        max_iter:最大迭代次数
        self.alpha:衰减系数
        paticals_nums:粒子数目
        top_k:更新粒子时用到的粒子数

        '''

        self.cluster_nums=cluster_nums 
        self.g_0=g_0 
        self.max_iter=max_iter
        self.alpha=alpha
        self.paticals_nums=paticals_nums
        self.top_k=top_k
        
    def samples_bound(self,samples):
        '''
        获取样本边界，反击最大值向量和最小值向量
        '''
        max_bound=np.max(samples,axis=0)
        min_bound=np.min(samples,axis=0)
        return max_bound,min_bound
        
    def check_bound(self,max_bound,min_bound,particals):
        nums,s1,s2=particals.shape
        for n in range(nums):
            for i in range(s1):
                for j in range(s2):
                    if particals[n,i,j]>max_bound[j]:
                        particals[n,i,j]=max_bound[j]

                    elif particals[n,i,j]<min_bound[j]:
                        particals[n,i,j]=min_bound[j]  

    def calculate_eps(self,risk_data,k=0.1):
        '''
        计算引力搜索程序的eps参数

        Args:
        risk_data:风险矩阵
        k:常数项，默认0.1

        Return:
        eps
        '''
        max_risk=np.max(risk_data)
        L=risk_data.shape[1]
        eps=k*np.sqrt(L)*max_risk
        return eps

    def intialize_particals(self,features,A,C,seed=True):
        '''
        初始化粒子位置矩阵和速度矩阵

        Args:
        features:提取的特征矩阵
        A:粒子数目
        C:聚类的类别数
        seed:如果为True，设定随机种子，每次结果将一样 bool

        Returns:
        particals:粒子的位置矩阵
        velocity:速度矩阵
        '''

        s1,s2=features.shape
        particals=np.zeros(shape=(A,C,s2),dtype=np.float32)
        for i in range(A):
            if seed:
                np.random.seed(i) #随机种子,不需要请注释
            label=np.random.randint(1,C+1,s1) #随即指定类别
            for k in range(C):
                mask=(label==k+1)
                s=np.mean(features[mask,],axis=0) #计算聚类中心
                particals[i,k,:]=s
        velocity=np.zeros(shape=(A,C,s2),dtype=np.float32) #初始化速度为0
        return particals,velocity


    def intialize_particals_v2(self,features,A,C,seed=True):
        '''
        初始化粒子位置矩阵和速度矩阵，和文档中不同，这里随即选择C个点作为初始聚类中心，和K-means的初始化类似

        Args:
        features:提取的特征矩阵
        A:粒子数目
        C:聚类的类别数
        seed:同上

        Returns:
        particals:粒子的位置矩阵
        velocity:速度矩阵
        '''
        s1,s2=features.shape
        
        particals=np.zeros(shape=(A,C,s2),dtype=np.float32)
        # rows=features.shape[0]
        step=s1//(C+1)
        for i in range(A):
            if seed:
                np.random.seed(i)#随机种子，不需要清注释
            # random_center_index=np.random.randint(0,s1,size=(C,)) #随即生成C个介于0~s2之间的样本下标，作为初始聚类中心
            start_index=np.random.randint(0,C)
            random_center_index=[start_index + item*step for item in range(C)]
            center=features[random_center_index,]
            particals[i,]=center
        velocity=np.zeros(shape=(A,C,s2),dtype=np.float32) #初始化速度为0
        return particals,velocity


    def solve_label_by_center(self,partical,samples):
        '''
        根据聚类中心计算标签

        Args:
        partical:一个粒子，也就是聚类中心
        samples:特征矩阵

        Returns:
        label:标签值
        '''
        s1=partical.shape[0]
        s2=samples.shape[0]
        distance=np.zeros(shape=(s2,s1),dtype=np.float32)
        for index,center in enumerate(partical):
            d=np.square(samples-center)
            d=np.sum(d,axis=1)
            d=np.sqrt(d)
            distance[:,index]=d
        label=distance.argmin(axis=1)+1
        return label

    def solve_fitness(self,particals,samples)->list:
        '''
        计算粒子的适应度

        Args:
        particals:粒子
        samples:样本
        
        Returns:fitness 适应度
        '''
        fitness=[]
        for partical in particals:
            score=0
            label=self.solve_label_by_center(partical,samples)
            cluster_num=partical.shape[0]
            for k in range(cluster_num):
                mask=(label==k+1) #如果当前的聚类中心没有一个样本，是可能出现的
                center=partical[k]
                k_sample=samples[mask,]
                if k_sample.size==0:
                    print("current center has 0 samples!")
                    continue
                d=np.square(k_sample-center)
                d=np.sum(d,axis=1)
                d=np.mean(d,axis=0)
                score+=d

            fitness.append(score)
        return fitness


    def solve_quality(self,scores:list)->list:
        '''
        根据适应度计算质量
        '''
        best_score=min(scores)
        worst_score=max(scores)
        scale=(best_score-worst_score)
        n=len(scores)
    
        if scale==0:  #如果粒子的适应度差为0,那么直接所有粒子质量设为1
            return [1 for _ in range(n)]
    
        quality=[]
        for index in range(n):
            score=(scores[index]-worst_score)/scale
            quality.append(score)
        total_quality=sum(quality)
        if quality!=0:
            for index in range(n):
                quality[index]/=total_quality
        else:
            raise ValueError("the quality is all zero!solve failed!")

        return quality

    def solve_partical_distance(self,p1,p2):
        '''
        计算两个粒子的欧式距离
        '''
        d=np.square(p1-p2)
        d=np.sum(d)
        d=np.sqrt(d)
        return d

    def find_topk(self,fitness,K):
        '''
        找到排名top_k的粒子，并返回其索引,因为文档中所用kbest的粒子来计算合力
        '''
        top_k=[]
        fitness_copy=fitness.copy()
        fitness.sort()
        for k in range(K):
            rank=fitness_copy.index(fitness[k])
            top_k.append(rank)
        return top_k

    def solve_acceleration(self,particals,quality,G,top_k,eps:float=0.01,k=0.1):
        '''
        计算加速度

        Args:
        particals:粒子
        quality:粒子质量
        G:系数
        top_k:排名k
        eps:系数，保证分母>0
        k:至少有一个粒子质量为0，对于这个粒子的加速度的值全部赋常数k #如果你们需要修改可以自行改变这个策略

        
        Returns:acceleration 加速度矩阵
        '''
        s1,s2,s3=particals.shape
        acceleration=np.zeros(shape=(s1,s2,s3),dtype=np.float32)
        for i in range(s1):
            F=np.zeros(shape=(s2,s3),dtype=np.float32)
            for j in range(s1):
                if i!=j and j in top_k:
                    for c in range(s2):
                        random_number=random.random()
                        F[c,:]+=random_number*G*(quality[i]*quality[j])*(particals[i,c,:]-\
                            particals[j,c,:])/(self.solve_partical_distance(particals[i],particals[j])+eps)
            if quality[i]!=0:
                acceleration_i=F/quality[i] #如果质量>0，计算加速度
            else:
                acceleration_i=np.ones(shape=(s2,s3),dtype=np.float32)*k  #否则，给一个常数的加速度
            acceleration[i,:,:]=acceleration_i
        return acceleration


    def update_velocity_particals(self,acceleration,particals,velocity,k=0.1):
        '''
        更新速度和粒子位置
        
        Args:
        acceleration:加速度
        paticals:粒子
        velocity:速度矩阵
        k:更新的幅度，防止更新幅度过大，导致越jie
        
        Returns:
        更新后的粒子和速度
        '''
        s1,s2,_=acceleration.shape
        for i in range(s1):
            for j in range(s2):
                velocity[i,j,:]=random.random()*velocity[i,j,:]+acceleration[i,j,:]*k #防止加速度过大
                particals[i,j,:]=particals[i,j,:]+velocity[i,j,:]
            
        return particals,velocity

    def update_centers(self,centers,samples):
        '''
        更新聚类中心
        Args:
        centers:聚类中心
        samples:样本

        Returns:
        更新后的聚类中心 centers
        '''
        K,_=centers.shape
        s2=samples.shape[0]
        distance=np.zeros(shape=(s2,K),dtype=np.float32)
        for index,center in enumerate(centers):
            d=np.square(samples-center)
            d=np.sum(d,axis=1)
            d=np.sqrt(d)
            distance[:,index]=d
        label=distance.argmin(axis=1)+1
        
        for k in range(K):
            mask=(label==k+1)
            new_center=np.mean(samples[mask],axis=0)
            centers[k,]=new_center
        return centers


    def update_particals_by_label(self,particals,features):
        '''
        根据最近原则，更新聚类中心
        '''
        s1=particals.shape[0]
        for i in range(s1):
            partical=particals[i,...]
            partical=self.update_centers(partical,features)
            particals[i,...]=partical


    def CDI(self,features,centers,label)->float:
        '''
        计算CDI指标

        Args:
        features:特征矩阵  
        centers:最后的聚类中心
        label:标签

        Returns:
        cdi 值
        '''
        center_distance=0
        cluster_distance=0
        s1,_=centers.shape
        for i in range(s1):
            for j in range(i+1,s1):
                dc=np.square(centers[i]-centers[j])
                dc=np.sum(dc)
                dc=np.sqrt(dc)
                center_distance+=dc
        center_distance=center_distance/(s1*(s1-1)/2)

        for k in range(s1):
            mask=(label==k+1)
            k_sample=features[mask]
            center=centers[k]
            dk=np.square(k_sample-center)
            dk=np.sum(dk,axis=1)
            dk=np.mean(dk,axis=0)
            dk=np.sqrt(dk)
            cluster_distance+=dk
        return np.sqrt(cluster_distance)/center_distance

    def plot_cluster(self,original_data,samples,centers,scale=True):
        '''
        绘制聚类图,默认保存在image目录下
        original_data:原始数据
        samples:特征矩阵
        centers:聚类中心
        scale:子图比例是否一致
        '''

        label=self.solve_label_by_center(centers,samples)
        #original_data=(original_data-np.min(original_data,axis=1,keepdims=True))/(np.max(original_data,axis=1,keepdims=True)-np.min(original_data,axis=1,keepdims=True))
        C=centers.shape[0]
        rows=int(np.sqrt(C))
        cols=C//rows
        while rows*cols<C:
            cols+=1

        for i in range(C):
            plt.subplot(rows,cols,i+1)
            plt.title("the %d-th cluster"%(i+1))
            mask=(label==(i+1))
            k_original=original_data[mask,]
            print("the %d-th cluster has %d sample!"%(i+1,k_original.shape[0]))
            print("-"*50)
            plt.xlim((0,35))
            if scale:
                plt.ylim((0,np.max(original_data*1.2)))
            else:
                plt.ylim((0,np.max(k_original*1.2)))
                
            n=len(k_original)
            for index in range(n):
                data=k_original[index]
                plt.plot(data,"r-o",linewidth=1,markersize=2,c=color_list[index])
            # plt.plot(k_original.T,"r-o",linewidth=1,markersize=2)
        if not os.path.exists("image"):
            os.mkdir("image")
        image_name="image/%s.jpg"%(int(time.time()))
        plt.tight_layout()
        plt.savefig(image_name,dpi=800)
        plt.show()