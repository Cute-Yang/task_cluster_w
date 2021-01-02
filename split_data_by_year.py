'''
将不同地区的数据按照年份拆开，每个年份存一份csv文件

因为特征提取针对的是全年的数据，所以我按照每年算的

这里2015年数据较全，只有5月份数据缺失较多

所以每个地区按照2015年的数据进行代码的测试

存放在split_data目录下
'''

import os

def split_data_by_year(data_path,ignore=True):
    '''
    Args:
    data_path:每个地区的数据的文件路径，包含了所有年份的数据，这个文件可以由preprocess_data.py程序生成
    ignore:boool 是否忽略第一行，就是列名称

    Return:
    None

    Raise:
    ValueError:if param is invalid
    '''
    dest_data_format="split_data/%s/%s.csv"
    f=open(data_path,"r",encoding="utf-8")
    if ignore:
        f.readline()

    content=f.readline()
    content=content.split(",")
    area=content[0]
    year=content[1]
    current_year=year
    current_area=area
    current_dest=dest_data_format%(current_area,current_year)
    par_dir,_=current_dest.rsplit("/",1)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    fw=open(current_dest,"a+")
    fw.write(",".join(content))

    for _,content in enumerate(f):
        content=content.split(",")
        year=content[1]
        if year!=current_year:
            fw.close()
            current_year=year
            current_dest=dest_data_format%(current_area,current_year)
            par_dir,_=current_dest.rsplit("/",1)
            if not os.path.exists(par_dir):
                os.makedirs(par_dir)
            fw=open(current_dest,"a+")
        fw.write(",".join(content))


if __name__=="__main__":
    root_data_path="handled_data" #这里是补全的数据，因为每个月的天数不一样，按照31天补齐，用前面n天数据的平均值作为这一天的value
    
    data_path_list=(os.path.join(root_data_path,item) for \
        item in os.listdir(root_data_path))
    for data_path in data_path_list:
        split_data_by_year(data_path)