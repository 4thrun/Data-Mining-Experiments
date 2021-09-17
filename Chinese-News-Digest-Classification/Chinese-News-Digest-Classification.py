#!/usr/bin/env python
# coding: utf-8

# # Chinese\-News\-Digest\-Classification
# 
# 56821 pieces of Chinese news digest fetched from websites\.
# 
# This dataset can be divided into 10 categories: international, cultural, entertainment, sports, finance, automobile, education, technology, real estate, securities\.
# 
# The project implements CNN (Convolutional Neural Networks) to deal with the categories\.

# In[1]:


# -*- coding: utf-8 -*-
import os
from multiprocessing import cpu_count
import numpy as np
import shutil
# CPU version used
# python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
# PaddlePaddle 1.8.0
import paddle # Baidu 
import paddle.fluid as fluid


# path 
data_root_path = '/home/aistudio/work/'
# data_root_path = './'
model_save_path = '/home/aistudio/work/infer/'
# model_save_path = './infer/'


# ## 1\. Data Preparation
# 

# ### 1\.1 Data Set and Dict 
# 
# Now we only have one file (the raw data): **news\_classify\_data\.txt**, and we have to prepare the data ready for our CNN model\. So before building the network, 3 more files have to be created\. 
# 
# The function `create_dict` is designed to generate **dict\_txt\.txt**, corresponding characters and numbers one by one\. 
# 
# After `create_dict`, `create_data_list` will be able to generate a test set: **test_list.txt** and a training set: **train_list.txt**\.

# In[2]:


# create a dataset 
def create_data_list(data_root_path):
    # initialize and clear the existing content 
    with open(data_root_path + 'test_list.txt', 'w') as f:
        pass 
    with open(data_root_path + "train_list.txt", 'w') as f:
        pass 

    with open(os.path.join(data_root_path, 'dict_txt.txt'), 'r', encoding='UTF-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    with open(os.path.join(data_root_path, 'news_classify_data.txt'), 'r', encoding='UTF-8') as f_data:
        lines = f_data.readlines()

    i = 0 
    for line in lines:
        title = line.split("_!_")[-1].replace('\n', '') # the last one 
        l = line.split("_!_")[1] # the second one 
        labs = ""
        # a way to select test set and training set
        if i%10 == 0:
            with open(os.path.join(data_root_path, "test_list.txt"), 'a', encoding='UTF-8') as f_test:
                for s in title:
                    lab = str(dict_txt[s])   
                    labs = labs + lab + ','
                labs = labs[:-1] 
                labs = labs + '\t' + l + '\n'
                f_test.write(labs)
        else:
            with open(os.path.join(data_root_path, 'train_list.txt'), 'a', encoding='UTF-8') as f_train:
                for s in title:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + l + '\n'
                f_train.write(labs) 
        i += 1
    print('dataset created! ')
    

# create a dict 
def create_dict(data_path, dict_path):
    dict_set = set()
    # read the data 
    with open(data_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    # turn the data into tuple 
    for line in lines:
        title = line.split("_!_")[-1].replace('\n', '')
        for s in title:
            dict_set.add(s)
    # turn tuple into dict
    # one Chinese character corresponds to one number
    dict_list = list()
    i = 0 
    for s in dict_set:
        dict_list.append([s, i])
        i = i + 1
    # add unknown character 
    # replace all unknown characters with `<unk>`
    dict_txt = dict(dict_list)
    end_dict = {'<unk>': i}
    dict_txt.update(end_dict)
    # save 
    with open(dict_path, 'w', encoding='UTF-8') as f:
        f.write(str(dict_txt))
    print('dict created! ')


def get_dict_length(dict_path):
    with open(dict_path, 'r', encoding='UTF-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())


if __name__ == '__main__':
    data_path = os.path.join(data_root_path, 'news_classify_data.txt')
    dict_path = os.path.join(data_root_path, 'dict_txt.txt')

    # create dict 
    create_dict(data_path=data_path, dict_path=dict_path)
    # create dataset 
    create_data_list(data_root_path=data_root_path)


# ### 1\.2 Data Reader 
# 
# `paddle.reader.xmap_readers()`: use user\-defined **mapper** to map the samples returned by the **reader** to the output queue through multi\-threading\.
# 
# Now we have to create **2 readers**: `train_reader` and `test_reader`\.

# In[3]:


# pretreatment
def data_mapper(sample):
    data, label = sample 
    data = [int(data) for data in data.split(",")]
    return data, int(label)


# create `test_reader`
def test_reader(test_list_path):
    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                yield data, label 

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


# create `train_reader`
def train_reader(train_list_path):
    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # scramble data
            np.random.shuffle(lines)
            for line in lines:
                data, label = line.split('\t')
                yield data, label 

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


# ## 2\. CNN
# 
# Input word vector sequence to generate a **feature map**\. Then, use the **max pooling over time** for the feature map to obtain the feature of entire sentence corresponding to this convolution kernel\. Finally, combine the features obtained by all the convolution kernels to obtain a fixed\-length vector representation of the text\.
# 
# In actual application scenarios, we use multiple convolution kernels to process sentences and convolution kernels with the same window size are stacked to form a matrix\. In this way, calculations can be completed more efficiently. (Of course we can use convolution kernels of different window sizes to process sentences\.)

# ### 2\.1 Create and Configure the network 
#   
# - Define the network
# - Define cost function
# - Define optimizer
# 
# 

# In[4]:


# create CNN
def CNN_net(data, dict_dim, class_dim=10, emb_dim=128, hid_dim=128, hid_dim2=98):  
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    conv_3 = fluid.nets.sequence_conv_pool(
        input = emb,
        num_filters = hid_dim,
        filter_size = 3,
        act = "tanh", 
        pool_type = 'sqrt', 
    )
    conv_4 = fluid.nets.sequence_conv_pool(
        input = emb, 
        num_filters = hid_dim2, 
        filter_size = 4, 
        act = 'tanh', 
        pool_type = 'sqrt', 
    )
    output = fluid.layers.fc(
        input = [conv_3, conv_4], 
        size = class_dim, 
        act = 'softmax', 
    )
    return output


# In[5]:


# define input
# `lod_level` != 0: specify the input data as a sequence 
words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
# get the length of the data dict
dict_dim = get_dict_length(os.path.join(data_root_path, 'dict_txt.txt'))
# get CNN model 
# model = CNN_net(words, dict_dim, 15)
# get classifier 
model = CNN_net(words, dict_dim)
# cost function and accuracy 
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# prediction 
test_program = fluid.default_main_program().clone(for_test=True)

# define optimizer 
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)
opt = optimizer.minimize(avg_cost)

# create an executor 
# as is mentioned at the beginning, we use CPU version (slow)
# place = fluid.CUDAPlace()
place = fluid.CPUPlace()

exe = fluid.Executor(place)
# initialize params 
exe.run(fluid.default_startup_program())


# ### 2\.2 Training 

# In[6]:


# readers for training set and test set 
train_r = paddle.batch(reader=train_reader(os.path.join(data_root_path, 'train_list.txt')), batch_size=128)
test_r = paddle.batch(reader=test_reader(os.path.join(data_root_path, 'test_list.txt')), batch_size=128)
 
# define data mapper 
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

EPOCH_NUM = 10


# training  
for pass_id in range(EPOCH_NUM):
    # train 
    for batch_id, data in enumerate(train_r()):
        train_cost, train_acc = exe.run(
            program = fluid.default_main_program(), 
            feed = feeder.feed(data), 
            fetch_list = [avg_cost, acc], 
            )
        if batch_id%100 == 0:
            print("Pass: %d, Batch: %d, Cost: %0.5f, Acc: %0.5f" % (pass_id, batch_id, train_cost[0], train_acc[0]))
    # test 
    test_costs = list()
    test_accs = list()
    for batch_id, data in enumerate(test_r()):
        test_cost, test_acc = exe.run(
            program = test_program,
            feed = feeder.feed(data), 
            fetch_list = [avg_cost, acc],  
        )
        test_costs.append(test_cost[0])
        test_accs.append(test_acc[0]) 
    # average test cost and accuracy 
    test_cost = (sum(test_costs)) / len(test_costs)
    test_acc = (sum(test_accs)) / len(test_accs)
    print('Test: %d, Cost: %0.5f, Acc: %0.5f' % (pass_id, test_cost, test_acc))


# save the prediction model 
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
fluid.io.save_inference_model(model_save_path, feeded_var_names=[words.name], target_vars=[model], executor=exe)
print('Model saved! ')


# In[7]:


# infer with the model 
# create executor 
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# import prediction program, data lists, classifier from model 
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=model_save_path, executor=exe)


# load data 
def get_data(sentence):
    # read data dict 
    with open(os.path.join(data_root_path, 'dict_txt.txt'), 'r', encoding='UTF-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    f_data.close()
    dict_txt = dict(dict_txt)
    # turn string into list 
    keys = dict_txt.keys()
    data = list()
    for s in sentence:
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data 


data = list()
# get graph data 
data1 = get_data("在获得诺贝尔文学奖7年之后，莫言15日晚间在山西汾阳贾家庄如是说")
data2 = get_data('综合“今日美国”、《世界日报》等当地媒体报道，芝加哥河滨警察局表示，')
data.append(data1)
data.append(data2)

# how many words in each sentence 
base_shape = [[len(c) for c in data]]

# generate prediction data 
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# generate prediction 
result = exe.run(
    program = infer_program,
    feed = {feeded_var_names[0]: tensor_words},
    fetch_list = target_var, 
    )

# categories 
names = ['文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '国际', '证券']

# show label with the greatest confidence 
for i in range(len(data)):
    lab = np.argsort(result)[0][i][-1]
    print("Prediction: %d, Name: %s, Confidence: %f" % (lab, names[lab], result[0][i][lab]))

