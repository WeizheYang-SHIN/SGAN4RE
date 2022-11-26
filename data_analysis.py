# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2022/4/13 10:42 
# @Author : wz.yang 
# @File : data_analysis.py
# @desc :

import matplotlib.pyplot as plt
import random

# x_data = ["20{}年".format(i) for i in range(16,21)]
# y_data = [random.randint(100,300) for i in range(6)]
#
# plt.rcParams["font.sans-serif"]=['SimHei']
# plt.rcParams["axes.unicode_minus"]=False
#
# for i in range(len(x_data)):
#     plt.bar(x_data[i],y_data[i])
#
# plt.title("销量分析")
# plt.xlabel("年份")
# plt.ylabel("销量")
#
# plt.show()


datadir = './data\8_features_data'
train_data_dir = './data\8_features_data/All_8_features_train.txt'
dev_data_dir = './data\8_features_data/All_8_features_dev.txt'
test_data_dir = './data\8_features_data/All_8_features_test.txt'

lines_train = open(train_data_dir,'r',encoding='utf-8').readlines()
lines_dev = open(dev_data_dir,'r',encoding='utf-8').readlines()
lines_test = open(test_data_dir,'r',encoding='utf-8').readlines()
lines = lines_train + lines_dev + lines_test
relation_type_num = dict()
po_num = 0
ne_num = 0
for line in lines:
    relation_type = line.split(' ')[-1].strip()
    if relation_type != 'Negative':
        po_num += 1
        if relation_type not in relation_type_num:
            relation_type_num[relation_type] = 1
        else:
            relation_type_num[relation_type] += 1
    else:
        ne_num += 1

x_data = ["Negative","Positive"]
y_data = [ne_num,po_num]

plt.rcParams["font.sans-serif"]=['Times New Roman']
plt.rcParams["axes.unicode_minus"]=False
# plt.xticks(fontproperties='Times New Roman',size=18)
# plt.yticks(fontproperties='Times New Roman',size=18)


for i in range(len(x_data)):
    plt.bar(x_data[i],y_data[i])

plt.text("Negative",ne_num,"%.0f"% ne_num,ha="center",va= "bottom")
plt.text("Positive",po_num,"%.0f"% po_num,ha="center",va= "bottom")

plt.title("Data analysis of ACE 2005 Chinese dataset\n - positive and negative number ratio in ACE 2005 Chinese dataset")
plt.xlabel("Relation category")
plt.ylabel("Quantity")

plt.savefig("ACEChineseAllRatio.eps")
plt.show()


x_data = [k for k,v in relation_type_num.items()]
y_data = [v for k,v in relation_type_num.items()]

plt.rcParams["font.sans-serif"]=['Times New Roman']
plt.rcParams["axes.unicode_minus"]=False

for i in range(len(x_data)):
    plt.bar(x_data[i],y_data[i])
for x,y in zip(x_data,y_data):
    plt.text(x,y,"%.0f"% y,ha="center",va= "bottom")


plt.title("Data analysis of ACE 2005 Chinese dataset\n - all positive number ratio in ACE 2005 Chinese dataset")
plt.xlabel("Relation category")
plt.ylabel("Quantity")

plt.savefig("ACEChinesePositiveRelation.eps")
plt.show()

