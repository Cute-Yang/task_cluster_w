### requirement.txt是依赖的第三方库

### 安装依赖
+ Windows
pip install -r requirement.txt -i https://mirrors.aliyun.com/pypi/simple
如果pip找不到，切换到pip所在目录

+ Linux/Mac
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

注意:如果requirements.txt如果不再当前目录，你需要根这个文件的全路径 比如/home/fish/requirements.txt

运行data_preprocess.py预处理数据

split_each_year.py将数据按照年份拆分

extract_features.py 提取特征

core.py是求解过程

### 在当前目录下 python3 core.py即可 其余类似
---


### 文件说明
+ preprocess_data.py  数据预处理，这个最好根据你们的数据作出修改，不一定是通用的
+ dbscan_cluster.py  dbscan聚类

+ extract_features.py 特征提取

+ core.py GSP的代码实现

+ main.py 函数主入口
+ params.py  参数配置，统一放在这里了
+ split_data_by_year.py 将数据按照年份分割
+ color.py 一些颜色参数

