from core import GSP
from params import TrainParam
import random
import numpy as np
from extract_features import read_features,read_original_data

#初始化训练参数对象
train_param=TrainParam()

def calculate_fitness_error(f1,f2):
    '''
    计算两个适应度的差异，这里用欧式距离度量
    '''
    error=0
    assert len(f1)==len(f2),"the length of f1 and f2 must be equal!please check your input!"
    n=len(f1)
    for i in range(n):
        error+=np.square(f1[i]-f2[i])
    error=np.sqrt(error)
    return error

#格式化日志输出
def fitness_log(fitness):
    n=len(fitness)
    str_format=["{:8.4f}" for _ in range(n)]
    str_format=" | ".join(str_format)
    log_string=str_format.format(*(fitness))
    return log_string


#定义主函数
def main():
    gsp=GSP(
        cluster_nums=train_param.CLUSTER_NUMS,
        g_0=train_param.G,
        max_iter=train_param.MAX_ITER,
        alpha=train_param.ALPHA,
        paticals_nums=train_param.PATICALS_NUMS,
        top_k=train_param.K
    )

    #这些初始化参数在迭代过程中最好不要在gsp对象上修改，所以用副本
    G=gsp.g_0
    iter=0
    k=gsp.top_k
    fitness_error=999
    fitness_esp=0.01
    
    previous_fitness=[random.randint(1,10) \
        for _ in range(gsp.paticals_nums)]
    log_head=["{:^8}" for _ in range(gsp.paticals_nums)]
    log_head=" | ".join(log_head)
    log_name=["f_%d"%(number+1) for number in range(gsp.paticals_nums)]
    log_head=log_head.format(*log_name)

    samples_data=read_features(features_dir="features")
    samples=samples_data[:,1:]
    eps=gsp.calculate_eps(samples)
    max_bound,min_bound=gsp.samples_bound(samples)

    original_data=read_original_data(original_dir="area_data")
    original_data=original_data[:,1:]
    
    particals,velocity=gsp.intialize_particals_v2(samples,A=gsp.paticals_nums,C=gsp.cluster_nums,seed=False)

    print(log_head)
    print("Score as follows:")
    while fitness_error>fitness_esp and iter<gsp.max_iter:
        #计算适应度
        fitness=gsp.solve_fitness(particals,samples)

        #计算每个粒子的质量
        quality=gsp.solve_quality(fitness)

        #计算排名前K的粒子
        top_k=gsp.find_topk(fitness,k)

        #计算加速度
        acceleration=gsp.solve_acceleration(particals,quality,G,top_k,eps)

        #更新速度和位置
        particals,velocity=gsp.update_velocity_particals(acceleration,particals,velocity)

        #根据最邻近原则更新聚类中心
        gsp.update_particals_by_label(particals,samples)
        
        #检查是否越界
        gsp.check_bound(max_bound,min_bound,particals)

        #更新万有引力系数
        G=G*np.exp(-gsp.alpha*iter/gsp.max_iter)
        
        iter=iter+1

        fitness_error=calculate_fitness_error(previous_fitness,fitness)
        #保留上一次的适应度，以便下次迭代计算
        previous_fitness=fitness
        
        log_string=fitness_log(fitness)
        print(log_string)
        print("-"*len(log_string))

    fitness=gsp.solve_fitness(particals,samples)

    score=fitness[0]
    best_solver=0
    #找出适应度最好的粒子，作为当前的聚类结果

    for index,s in enumerate(fitness):
        if s<score:
            best_solver=index
            score=s
    best_c=particals[best_solver]

    # print(best_c,score)
    label=gsp.solve_label_by_center(best_c,samples)
    
    #可能会出现nan，导致计算失败
    if np.unique(label).shape[0]<gsp.cluster_nums:
        print("the intialized result is not good which caused our solver away from the right center!")
        return 

    print("Cluster result as follows:")
    gsp.plot_cluster(original_data,samples,best_c,scale=False) #scale为False，表示每个子图得坐标值范围可以不一致,True则为一致
    cdi=gsp.CDI(samples,best_c,label)
    print("the value of CDI is %.2f"%cdi)

if __name__=="__main__":
    main()