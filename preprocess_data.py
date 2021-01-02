'''
将原始数据按照地区划分，每个地区生成一个csv文件，这里我们将每个月的数据进行了对齐，
按照31天的长度，每个样本是长度为31的向量

文件的数据列为  地区 年份 月份  值向量
存放在handled_data目录下
'''
import time
import os
from params import PreprocessParam
param = PreprocessParam()

ORIGEINAL_FIEL_PATH = param.ORIGEINAL_FIEL_PATH
DEST_DATA_FORMAT = param.DEST_DATA_FORMAT
IGNORE_HEAD = param.IGNORE_HEAD
IGNORE_NUMS = param.IGNORE_NUMS
LEAP_YEAR = param.LEAP_YEAR
STRIDE = param.STRIDE
MAX_NA_NUMS = param.MAX_NA_NUMS
MISSING_FLAG = param.MISSING_FLAG
SAVE_LOG = param.SAVE_LOG
LOG_PATH = param.LOG_PATH


# 如果可能的话，需要判断年份是否闰年，2月份需要+1
MONTH_MAP = {
    "1": 31,
    "2": 28,
    "3": 31,
    "4": 30,
    "5": 31,
    "6": 30,
    "7": 31,
    "8": 31,
    "9": 30,
    "10": 31,
    "11": 30,
    "12": 31
}

if SAVE_LOG:
    now = int(time.time())
    if not LOG_PATH:
        log = open("logs/log-%s.txt" % now, "a+")
    else:
        log = open(os.path.join(LOG_PATH,"%s.txt"%now), "a+")
else:
    log = None


def na_handle_policy(data: list, na_nums: int, stride: int, info, log_ptr=None):
    '''
    处理NA的月份数据

    Args:
    data:需要进行处理NA的数据行
    na_nums:当前数据NA的个数
    stride:步数，表示取前stride个数据的平均作为当前NA的填充值，如果不足stride个，那么就取前面所有的数据的平均值
    info:当其数据的元信息，包含地区、年份、月份的一个三元组
    log_ptr:一个日志的文件指针，如果不是None,那么就写入处理的日志信息，默认存在logs目录下

    Return:
    handle_status:bool代表是否处理成功，如果为True,说明当前数据填充成功，指的是NA较少或者从一
    开始没有NA(有些月份从1月份开始缺失数据,按照时间序列的特性，不能填充)

    Raise:
    ValueError:invalid param
    '''
    handle_status = True
    if data[0] == MISSING_FLAG:
        log_info = "Area:%s,Time:%s-%s,missing data from 1 month!\n" % info
        print(log_info)
        if log_ptr:
            log_ptr.write(log_info)
        handle_status = False

    if na_nums > MAX_NA_NUMS:
        log_info = "Area:%s,Time:%s-%s" % info + \
            " too many NA value,contains %d NA value\n" % na_nums
        print(log_info)
        if log_ptr:
            log_ptr.write(log_info)
        handle_status = False

    if not handle_status:
        return handle_status

    for index, element in enumerate(data, start=0):
        if element == MISSING_FLAG:
            if index >= stride:
                # 如果长度大于等于stride
                new_value = sum(data[index-1:index-1-stride:-1])/stride
            else:
                new_value = sum(data[:index])/index
            new_value = round(new_value, 2)
            data[index] = new_value
    return handle_status


def append_policy(data, stride, current_date=None):
    '''
    将数据对对齐到31天

    Args:
    data:需要对齐的数据
    stride:步长，根之前一样
    current_date:optional param
    '''
    repeat_number = 31-len(data)
    while repeat_number:  # 循环对齐
        new_value = sum(data[-1:-1-stride:-1])/stride
        new_value = round(new_value, 2)
        data.append(new_value)
        repeat_number -= 1


fr = open(ORIGEINAL_FIEL_PATH, "r")
if IGNORE_HEAD and IGNORE_NUMS:
    raise ValueError("param ignore_head==True,ignore_nums should be None,\
        ignore_nums >=1 ,ignore_head must be False or None!")

if IGNORE_HEAD:
    fr.readline()  # ignore first line

if IGNORE_NUMS:
    for _ in range(IGNORE_NUMS):
        fr.readline()

current_month = "1"
current_area = "1"
current_year = "2015"
kind_info = [current_area, current_year, current_month]
current_dest = DEST_DATA_FORMAT % int(current_area)
fw = open(current_dest, "a+")
fw.write("area,year,month,value\n")
row_data = []
real_days = 0
current_day_number = 0
change_area = False
record_flag = True

for index, content in enumerate(fr):
    area, date, value = content.split(",")
    value = value.strip("\n")
    if value != MISSING_FLAG:
        value = float(value)
    year, month, day_number = date.split("/")

    change_area = (current_area != area)
    if change_area:  # 如果区域改变了,那么就改变当前文件指针，并且关闭上一次的文件句柄
        fw.close()
        current_dest = DEST_DATA_FORMAT % int(area)
        fw = open(current_dest, "a+")
        fw.write("area,year,month,value\n")
        change_area = False
        current_area = area

    if current_month == month and not change_area:
        offset = int(day_number)-current_day_number
        for _ in range(offset-1):
            row_data.append(MISSING_FLAG)
            real_days += 1
        row_data.append(value)
        real_days += 1
        current_day_number = int(day_number)

    else:
        expected_days = MONTH_MAP[current_month]
        if real_days < expected_days:  # 这里判断是否丢失后面天数的数据
            print("Area:%s,Time:%s-%s,missing value from %d to %d\n"
                  % (current_area, current_year, current_month, real_days+1, expected_days))
            if expected_days-real_days >= 20:  # 如果丢失超过20天，那么就丢弃，这些参数根据自己调整
                record_flag = False
        na_handle_status = True  # 初始化status
        if MISSING_FLAG in row_data:
            na_nums = row_data.count(MISSING_FLAG)
            na_handle_status = na_handle_policy(
                row_data, na_nums,
                stride=STRIDE,
                info=(current_area, current_year, current_month),
                log_ptr=log
            )

            if na_handle_status:  # 处理成功
                print("handled data with  Area:%s Time:%s-%s successfully!\n" %
                      (current_area, current_year, current_month))
            else:  # 处理失败
                print("handled faild with Area:%s Time:%s-%s\n" %
                      (current_area, current_year, current_month))

        if real_days < 31 and na_handle_status:  # 如果处理成功，并且对齐数据到31,这里可能将后面缺失太多的数据一并补齐，所以还不能写入
            append_policy(row_data, stride=STRIDE, current_date=current_month)

        if na_handle_status and record_flag:  # 处理成功，并且缺失后面的数据太多
            row_data = map(lambda i: str(i), row_data)
            handled_content = ",".join(kind_info)+","+",".join(row_data)
            fw.write(handled_content)
            fw.write("\n")

        # 更新判断状态的变量值
        record_flag = True
        current_year, current_month = year, month
        current_day_number = 0
        # 两天的偏移量，如果是1，说明连续，如果不是1，那对中间缺失的进行填充，补齐offset-1个数据
        offset = int(day_number)-current_day_number

        if change_area:
            current_area = area

        real_days = 0
        row_data = []
        for _ in range(offset-1):
            row_data.append(MISSING_FLAG)
            real_days += 1
        current_day_number = int(day_number)
        row_data.append(value)
        real_days += 1
        kind_info[:] = (current_area, current_year, current_month)

# 关闭文件句柄
fw.close()
fr.close()
if SAVE_LOG:
    log.close()
