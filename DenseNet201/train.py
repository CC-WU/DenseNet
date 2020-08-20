# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 03:21:45 2020

@author: User
"""


import os
import sys
import glob
import matplotlib.pyplot as plt
 
from keras import __version__
from keras.applications.densenet import DenseNet201,preprocess_input
 
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
 
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) 

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt
 
 
# 数据准备
IM_WIDTH, IM_HEIGHT = 224, 224 #densenet指定的图片尺寸
 
 
 
train_dir = 'in/train'  # 训练集数据路径
val_dir = 'in/val' # 验证集数据
nb_classes= 23
nb_epoch = 2
batch_size = 8
 
nb_train_samples = get_nb_files(train_dir)      # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)       #验证集样本个数
nb_epoch = int(nb_epoch)                # epoch数量
batch_size = int(batch_size)           
 
#　图片生成器
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
test_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
 
# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,class_mode='categorical')
 
validation_generator = test_datagen.flow_from_directory(
val_dir,
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=batch_size,class_mode='categorical')
 
# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model
 
 
#搭建模型
model = DenseNet201(include_top=False)
model = add_new_last_layer(model, nb_classes)
#model.load_weights('../model/checkpoint-02e-val_acc_0.82.hdf5')  第二次训练可以接着第一次训练得到的模型接着训练
model.compile(optimizer=SGD(lr=0.001, momentum=0.9,decay=0.0001,nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
 
 
#更好地保存模型 Save the model after every epoch.
output_model_file = './model_in/checkpoint-{epoch:02d}e-val_acc_{val_accuracy:.2f}.h5'
#keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpoint = ModelCheckpoint(output_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)
 
 
 
 
#开始训练
history_ft = model.fit_generator(
train_generator,
samples_per_epoch=nb_train_samples,
nb_epoch=nb_epoch,
callbacks=[checkpoint],
validation_data=validation_generator,
nb_val_samples=nb_val_samples)
 
def plot_training(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.plot(epochs, acc, 'r-')
  plt.plot(epochs, val_acc, 'b')
  plt.title('Training and validation accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'r-')
  plt.plot(epochs, val_loss, 'b-')
  plt.title('Training and validation loss')
  plt.show()
 
# 训练的acc_loss图
plot_training(history_ft)