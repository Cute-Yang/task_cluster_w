'''
程序中用到的参数，可以统一在这里修改
'''


'''
文件预处理用到的参数
'''
# 原始文件路径
ORIGEINAL_FIEL_PATH = "data/data.csv"

# 目标文件路径
DEST_DATA_FORMAT = "data_handled/handled_data-%03d-of-200.csv"

# 是否忽略第一行
IGNORE_HEAD = True

# 忽略的行数，这两个参数只能指定一个
IGNORE_NUMS = None

# 是否闰年
LEAP_YEAR = False

# 均值填充的步长
STRIDE = 3

# 允许最大缺失数目
MAX_NA_NUMS = 3

# 缺失标志
MISSING_FLAG = "NA"

# 保存日志
SAVE_LOG = True

LOG_PATH = "logs"


class PreprocessParam:
    def __init__(self):
        self.ORIGEINAL_FIEL_PATH = ORIGEINAL_FIEL_PATH
        self.DEST_DATA_FORMAT = DEST_DATA_FORMAT
        self.IGNORE_HEAD = IGNORE_HEAD
        self.IGNORE_NUMS = IGNORE_NUMS
        self.LEAP_YEAR = LEAP_YEAR
        self.STRIDE = STRIDE
        self.MAX_NA_NUMS = MAX_NA_NUMS
        self.MISSING_FLAG = MISSING_FLAG
        self.SAVE_LOG = SAVE_LOG
        self.LOG_PATH = LOG_PATH


'''
特征提取时用到的一些Hyper-Paramter
'''
# 春天的月份 tuple
SPRING_INTERVAL = (3, 4, 5)

# 夏天的月份 tuple
SUMMER_INTERVAL = (6, 7, 8)

# 秋天的月份 tuple
AUTUM_INTERVAL = (9, 10, 11)

# 冬天的月份 tuple
WINTER_INTERVAL = (12, 1, 2)


class ExtractParam:
    def __init__(self):
        self.SPRING_INTERVAL = SPRING_INTERVAL
        self.SUMMER_INTERVAL = SUMMER_INTERVAL
        self.AUTUM_INTERVAL = AUTUM_INTERVAL
        self.WINTER_INTERVAL = WINTER_INTERVAL


'''
训练参数，在使用引力搜索算法的时候的相关Hyper-Parameter
'''

# 聚类数量 int
CLUSTER_NUMS = 5

# 万有引力常数值,float>0
G = 1.2

# 最大迭代次数，int
MAX_ITER = 200

# 衰减系数,float>0
ALPHA = 20

# 初始粒子数目,int
PATICALS_NUMS = 6

# top_k，在计算粒子加速度时使用排名前K的粒子计算合力,int
K = 5


class TrainParam:
    def __init__(self):
        self.CLUSTER_NUMS = CLUSTER_NUMS
        self.G = G
        self.MAX_ITER = MAX_ITER
        self.ALPHA = ALPHA
        self.PATICALS_NUMS = PATICALS_NUMS
        self.K = K
