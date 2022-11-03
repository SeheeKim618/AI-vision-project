import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# CIFAR-100 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,100)
y_test=tf.keras.utils.to_categorical(y_test,100)

#하이퍼 매개변수 설정
batch_siz = 64
n_epoch  = 1
k=5 # k-folds
activation_func = 'relu'
dropout = 0.5
weight_decay = 0.00001
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 5

def scheduler(epoch, lr):
    start = 0.01
    drop = 0.1 # 0.1
    epochs_drop = 20.0 #20
    lr = start * (drop ** np.floor((epoch)/epochs_drop))
    return lr

lr_schedule = LearningRateScheduler(scheduler, verbose=1)

# 하이퍼 매개변수에 따라 교차 검증을 수행하고 정확률을 반환하는 함수
def cross_validation(data_gen,activation_function,dropout_rate,weight_decay):
    accuracy=[]
    for train_index,val_index in KFold(k).split(x_train):
        xtrain,xval=x_train[train_index],x_train[val_index]
        ytrain,yval=y_train[train_index],y_train[val_index]

        # 신경망 모델 설계
        vgg = Vgg((32,32,3), activation_function, dropout_rate, weight_decay)
        cnn = vgg.vgg16()
        #cnn = vgg.vgg_mini()

        # 신경망을 학습하고 정확률 평가
        cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
        if data_gen:
            generator=ImageDataGenerator(rotation_range=3.0,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
            cnn.fit_generator(generator.flow(x_train,y_train,batch_size=batch_siz),epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)
        else:
            cnn.fit(xtrain,ytrain,batch_size=batch_siz,epochs=n_epoch, validation_data=(x_test,y_test),verbose=2)
        accuracy.append(cnn.evaluate(xval,yval,verbose=2)[1])
    return accuracy
    
# evaluate[1]-> top1, evaluate[2]-> top5
    
class Vgg:
    def __init__(self,input_shape,activation,dropout_rate,weight_decay):
        self.input_shape = input_shape
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
    def vgg16(self):
        vgg16 = Sequential()
        
        vgg16.add(Conv2D(64,(3,3),activation=self.activation,padding='same',input_shape=self.input_shape,kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.3))

        vgg16.add(Conv2D(64,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())

        vgg16.add(MaxPooling2D(pool_size=(2,2)))#,strides=2))
        
        vgg16.add(Conv2D(128,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.4))

        vgg16.add(Conv2D(128,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())

        vgg16.add(MaxPooling2D(pool_size=(2,2)))#,strides=2))

        vgg16.add(Conv2D(256, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.4))
        
        vgg16.add(Conv2D(256, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.4))

        vgg16.add(Conv2D(256, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())

        vgg16.add(MaxPooling2D(pool_size=(2,2)))#,strides=2))
        
        vgg16.add(Conv2D(512, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.4))

        vgg16.add(Conv2D(512, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.4))

        vgg16.add(Conv2D(512, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())

        vgg16.add(MaxPooling2D(pool_size=(2,2)))#,strides=2))

        vgg16.add(Conv2D(512, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.4))
        
        vgg16.add(Conv2D(512, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())
        vgg16.add(Dropout(0.4))

        vgg16.add(Conv2D(512, (3, 3),activation=self.activation, padding='same',kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())

        vgg16.add(MaxPooling2D(pool_size=(2,2)))#,strides=2))
        vgg16.add(Dropout(0.5))
        
        vgg16.add(Flatten())
        vgg16.add(Dense(512,activation=self.activation,kernel_regularizer=regularizers.l2(self.weight_decay)))
        vgg16.add(BatchNormalization())

        vgg16.add(Dropout(0.5))
        vgg16.add(Dense(100,activation='softmax',kernel_regularizer=regularizers.l2(self.weight_decay)))
        
        return vgg16
    
    

acc = cross_validation(True,activation_func,dropout,weight_decay)

print(acc)
        