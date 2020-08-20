# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:35:10 2020

@author: User
"""


# --coding:utf-8--
# 定义层
import sys
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time
import os

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import tensorflow as tf
 
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) 
 
target_size = (224, 224) 
 
# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
 
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]
 
# 初始計算各類別的準確率
class_name = {'again':0,'back_to_top':0,'backward':0,'continue':0,'end_play':0,'end_program':0,'end':0,'enter':0,'faster':0,'forward':0,'loudly':0,'next_page':0,'pause':0,'previous_page':0,'repeat':0,'say_again':0,'search_again':0,'search':0,'slower':0,'speed':0,'start_playing':0,'start':0,'whisper':0}
time_sum = 0

def count_acc(image_path, name): # 計算命中個數
    image_class = str(image_path).split('\\')[-1]
    image_class = str(image_class).split('.')[0]
    name = str(name).split('[\'')[-1].split('\']')[0]
    if name in image_class:
        class_name[name] = int(class_name[name])+1

def count_avg(class_nums = 23): # 算平均值
    ccou = 0
    for key, value in class_name.items() :
        ccou += int(value)
    print('命中個數: ', ccou)
    ccou = ccou / (5*class_nums)
    print('準確率: ', ccou)
    print('平均花費時間: ', time_sum/(5*class_nums))
 
#labels = ("daisy", "dandelion","roses","sunflowers","tulips")
#labels = ("again", "back_to_top", "backward", "continue", "end", "end_play", "end_program", "enter", "faster", "forward", "loudly", "next_page", "pause", "previous_page", "repeat", "say_again", "search", "search_again", "slower", "speed", "start", "start_playing", "whisper")
# 画图函数
def plot_preds(image, preds,labels):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  plt.barh([0, 1,2,3,4], preds, alpha=0.5)
  plt.yticks([0, 1,2,3,4], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

# 载入模型
start_time = time.time()
model = load_model('./model_in/checkpoint-46e-val_acc_0.96.hdf5')
end_time = time.time()
print('model load time: ', end_time - start_time)

dict_genres = {0: 'again', 1: 'back_to_top', 2: 'backward', 3: 'continue', 4: 'end', 5: 'end_play', 6: 'end_program', 7: 'enter', 8: 'faster', 9: 'forward', 10: 'loudly', 11: 'next_page', 12: 'pause', 13: 'previous_page', 14: 'repeat', 15: 'say_again', 16: 'search', 17: 'search_again', 18: 'slower', 19: 'speed', 20: 'start', 21: 'start_playing', 22: 'whisper' }
path = os.getcwd()
train_data_dir = path + '\\Public_test\\'
level = os.listdir(train_data_dir)
for i in range(len(level)):
    img_path = train_data_dir + level[i] # 取檔名
    start_time = time.time()
    # 本地图片进行预测
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((320, 240))
    preds = predict(model, img, target_size)
    preds = np.array(preds) # 個個類別的相識程度(機率)
    maxpre = np.argmax(preds) # 最高的類別編號
    print(img_path, ' Predicted:', dict_genres[int(maxpre)])
    #plot_preds(img, preds,labels)
    end_time = time.time()
    readfile_time = end_time - start_time # 計算測試時間
    time_sum += readfile_time # 計算總花費時間
    print('image test time: ', readfile_time)
    count_acc(img_path, dict_genres[int(maxpre)]) # 累加命中個數

count_avg() # 計算準確率
