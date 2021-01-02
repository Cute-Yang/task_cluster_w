import numpy as np
import os
import datetime
import json
from params import ExtractParam
'''
提取特征
'''

params=ExtractParam()
spring_interval=params.SPRING_INTERVAL
summer_interval=params.SUMMER_INTERVAL
autum_interval=params.AUTUM_INTERVAL
winter_interval=params.WINTER_INTERVAL


def write_csv():
    '''
    返回每次的保存的文件路径，并在检查点文件中记录最近的一次保存结果
    '''
    current_time=datetime.datetime.now()
    if not os.path.exists("features"):
        os.mkdir("features")
    features_file="features/%s_%s_%s_%s:%s:%s.csv"%(current_time.year,current_time.month,\
        current_time.day,current_time.hour,current_time.minute,current_time.second)

    original_data_file="area_data/%s_%s_%s_%s:%s:%s.csv"%(current_time.year,current_time.month,\
        current_time.day,current_time.hour,current_time.minute,current_time.second)

    with open("features/checkpoint.json","w") as f:
        file_info=json.dumps(
            {
                "latest_file":features_file
            },ensure_ascii=False
        )
        f.write(file_info)

    with open("area_data/checkpoint.json","w") as f:
        file_info=json.dumps(
            {
                "latest_file":original_data_file
            },ensure_ascii=False
        )
        f.write(file_info)


    return features_file,original_data_file

def extract_stage(file_path):
    '''
    提取三个阶段的全年相对平均值
    '''
    with open(file_path,"r",encoding="utf-8") as f:
        original_data=np.loadtxt(f,delimiter=",",skiprows=1,dtype=np.float32,usecols=list(range(28)))
        
    value=original_data[:,3:]
    year_mean_risk=np.mean(value)
    first_stage=value[:,:10]
    second_stage=value[:,10:20]
    third_stage=value[:,20:]

    first_stage_risk=np.mean(first_stage)/year_mean_risk
    second_stage_risk=np.mean(second_stage)/year_mean_risk
    third_stage_risk=np.mean(third_stage)/year_mean_risk

    return first_stage_risk,second_stage_risk,third_stage_risk


def extract_stddev(file_path):
    '''
    提取全年个阶段的stddev
    
    Args:
    file_path:一个地区一年的数据的csv文件
    '''
    with open(file_path,"r",encoding="utf-8") as f:
        original_data=np.loadtxt(f,delimiter=",",skiprows=1,dtype=np.float32,usecols=list(range(28)))
        
    value=original_data[:,3:]

    first_stage=value[:,:10]
    second_stage=value[:10:20]
    third_stage=value[:,20:]

    first_stage_stddev=np.std(first_stage,axis=1)
    first_stage_stddev=np.mean(first_stage_stddev)
    first_stage_stddev=first_stage_stddev/np.mean(first_stage)

    second_stage_stddev=np.std(second_stage,axis=1)
    second_stage_stddev=np.mean(second_stage_stddev)
    second_stage_stddev=second_stage_stddev/np.mean(second_stage)

    third_stage_stddev=np.std(third_stage,axis=1)
    third_stage_stddev=np.mean(third_stage_stddev)
    third_stage_stddev=third_stage_stddev/np.mean(third_stage)

    stddev=(first_stage_stddev+second_stage_stddev+third_stage_stddev)/3
    return stddev


def extract_season_corr(file_path):
    '''
    计算季节相关系数
    '''
    spring,summer,autum,winter=[],[],[],[]
    with open(file_path) as f:
        f.readline()
        for content in f:
            content=content.split(",")
            content[-1]=content[-1].strip("\n")
            month=int(content[2])
            data=content[3:]
            if 3<=month<=5:
                spring.append(data)
            elif 6<=month<=8:
                summer.append(data)
            elif 9<=month<=11:
                autum.append(data)
            else:
                winter.append(data)
        spring=np.array(spring,dtype=np.float32)
        summer=np.array(summer,dtype=np.float32)
        autum=np.array(autum,dtype=np.float32)
        winter=np.array(winter,dtype=np.float32)

        #here should use dbscan to throw some noise,then compute the mean value
        spring=np.mean(spring,axis=0)
        summer=np.mean(summer,axis=0)
        autum=np.mean(autum,axis=0)
        winter=np.mean(winter,axis=0)
        try:
            seasons=np.vstack([spring,summer,autum,winter])
        except:
            return np.nan

        correlation=np.corrcoef(seasons)
        mean_corr=np.mean(np.sum(correlation)-seasons.shape[0])/2
        return mean_corr

def extract_season_corr_v2(file_path:str,spring_interval=spring_interval,\
    summer_interval=summer_interval,autum_interval=autum_interval,\
        winter_interval=winter_interval,ignore_head=True):
    '''
    直接使用每天的值计算季节的相关系数，而不是利用平均值  

    Args:
    file_path:经过处理后的数据，每一条是一个月的数据
    spring_interval:春天的月份，默认是3，4，5月
    summer_interval:夏天的月份，默认是6,7,8月
    autum_interval:秋天的月份，默认是9,10,11月
    winter_interval:冬天的月份，默认是12,1,2月

    Return:
    season_corr:float 季节平均相关系数

    Raise:
    ValueError:if month is not valid,outside 1~12
    '''

    f=open(file_path,mode="r",encoding="utf-8")
    if ignore_head:
        f.readline()
    spring,summer,autum,winter=[],[],[],[]

    for _,content in enumerate(f):
        content=content.split(",")  #默认是csv文件，所以这里用，对行进行分割
        content[-1]=content[-1].strip("\n") #去除换行符
        month=int(content[2]) #提取具体的月份
        data=content[3:]

        if month in spring_interval:
            spring.append(data)

        elif month in summer_interval:
            summer.append(data)

        elif month in autum_interval:
            autum.append(data)

        elif month in winter_interval:
            winter.append(data)
        else:
            raise ValueError("month should be between 1-12,but your input is %d"%month)

    spring=[item for items in spring for item in items] #展平
    summer=[item for items in summer for item in items]
    autum=[item for items in autum for item in items]
    winter=[item for items in winter for item in items]

    #比较不同季节的天数，根据最某个季节的最小季节对齐数据 ,这里具体我不知道你们的做法，我就直接按照最少天数对齐
    spring_days=len(spring)
    summer_days=len(summer)
    autum_days=len(autum)
    winter_days=len(winter)
    min_days=min(spring_days,summer_days,autum_days,winter_days)
    
    spring=spring[:min_days]
    summer=summer[:min_days]
    autum=autum[:min_days]
    winter=winter[:min_days]

    seasons=np.array([spring,summer,autum,winter],dtype=np.float32)
    seasons_corr=np.corrcoef(seasons,rowvar=True) #rowvar 参数取True,代表每一行是一个变量，False，代表每一列是一个变量
    seasons_corr=(np.sum(seasons_corr)-seasons.shape[0])/2  
    #这里返回的是相关系数矩阵，为对称矩阵，对角线元素为1,所以减去对角线，再除2，得到平均相关系数，seasons_corr计算出来是类似与下面的矩阵
    # 1,0  0.2   0.4  0.5
    # 0.2  1.0   0.7  0.3 
    # 0.4  0.7   1.0  0.1
    # 0.5  0.3   0.1  1.0
    return seasons_corr



def extract_features(root_data="split_data",year="2015"):
    '''
    提取所有的特征,并且保存，特征存放在features目录，并且检查点文件记录最新的一次数据，允许你保存每次的结果
    Args:
    root_data:存放地区不同年份数据的根目录，会自己去遍历
    year:你要计算的年份

    '''
    length=len(os.listdir(root_data))
    data_path_list=(os.path.join(root_data,str(item),"%s.csv"%year) for item in range(1,1+length))
    features=[]
    original_data=[]
    for _,data_path in enumerate(data_path_list,start=1):
        f1,f2,f3=extract_stage(data_path)
        stddev=extract_stddev(data_path)
        season_corr=extract_season_corr(data_path)
        data=[f1,f2,f3,stddev,season_corr]
        features.append(data)
        area_data=extract_samples(data_path)
        original_data.append(area_data)

    features=np.array(features)
    mask=np.where(~np.isnan(features).any(axis=1)) 

    valid_area=mask[0].reshape(-1,1) #记录有效的数据的地区编号
    valid_area=valid_area+1
    
    original_data=np.array(original_data)
    original_data=original_data[~np.isnan(features).any(axis=1)] 
    original_data=np.hstack([valid_area,original_data])

    features=features[~np.isnan(features).any(axis=1)]
    samples=np.hstack([valid_area,features])
    samples=samples.tolist()

    original_data=original_data.tolist()

    features_file,original_data_file=write_csv()
    
    with open(features_file,"w") as f:
        for sample in samples:
            sample=map(lambda item:str(item),sample)
            data=",".join(sample)
            f.write(data)
            f.write("\n")

    with open(original_data_file,"w") as f:
        for area_data in original_data:
            area_data=map(lambda item: str(item),area_data)
            area_data=",".join(area_data)
            f.write(area_data)
            f.write("\n")

def read_features(features_dir="features")->np.ndarray:
    '''
    从存放特征的目录读取最新的一次数据，并且返回矩阵，第一列为地区的编号
    '''
    with open("%s/checkpoint.json"%features_dir,"r") as f:
        latest_checkpoint=json.load(f)
        
    latest_features=latest_checkpoint["latest_file"]

    with open(latest_features,"r") as f:
        return np.loadtxt(f,dtype=np.float32,delimiter=",",skiprows=0)


def extract_samples(file_path):
    '''
    存放原始数据，在area_data目录，可修改write_csv函数参数改变存放位置
    '''
    with open(file_path,"r",encoding="utf-8") as f:
        original_data=np.loadtxt(f,dtype=np.float32,delimiter=",",skiprows=1)
    original_data=original_data[:,3:]
    return np.mean(original_data,axis=0)

def read_original_data(original_dir="area_data")->np.ndarray:
    '''
    从存放原始数据的目录读取最新的一次数据，并且返回矩阵，第一列为地区的编号
    '''
    with open("%s/checkpoint.json"%original_dir,"r") as f:
        latest_checkpoint=json.load(f)
        
    latest_original=latest_checkpoint["latest_file"]

    with open(latest_original,"r") as f:
        return np.loadtxt(f,dtype=np.float32,delimiter=",",skiprows=0)


if __name__=="__main__":
    extract_features()