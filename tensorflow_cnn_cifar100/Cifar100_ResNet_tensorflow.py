import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers, Input
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
n_epoch  = 100
k=5 # k-folds
activation_func = 'relu'
dropout = 0.5
weight_decay = 0.03

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
        resnet = ResNet((32,32,3), activation_function, dropout_rate, weight_decay)
        cnn = resnet.resnet18()
        #cnn = vgg.vgg_mini()

        # 신경망을 학습하고 정확률 평가
        cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
        if data_gen:
            generator=ImageDataGenerator(rotation_range=3.0,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
            cnn.fit_generator(generator.flow(x_train,y_train,batch_size=batch_siz),epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)
        else:
            cnn.fit(xtrain,ytrain,batch_size=batch_siz,epochs=n_epoch, validation_data=(x_test,y_test),verbose=2)
        accuracy.append(cnn.evaluate(xval,yval,verbose=0)[1])
    return accuracy
    
    
    
class ResNet:
	def __init__(self,input_shape,activation,dropout_rate,weight_decay):
		self.input_shape = input_shape
		self.activation = activation
		self.dropout_rate = dropout_rate
		self.weight_decay = weight_decay
		self.initializer = tf.keras.initializers.HeNormal()

	def resnet18(self):
		input_layer = Input(shape=self.input_shape, name='InputLayer')
		
		x = Conv2D(64,(3,3),strides=(2,2),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv1_1')(input_layer)
		x_residual = BatchNormalization(name='batch_norm1_1')(x)

		x = Conv2D(64,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv2_1_1')(x_residual)
		x = BatchNormalization(name='batch_norm2_1_1')(x)
		x = Dropout(0.4)(x)
		x = Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv2_1_2')(x)
		x = Dropout(0.4)(x)
		x = tf.add(x_residual, x, name='residual2_1')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm2_1_2')(x)

		x = Conv2D(64,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv2_2_1')(x_residual)
		x = BatchNormalization(name='batch_norm2_2_1')(x)
		x = Dropout(0.4)(x)
		x = Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv2_2_2')(x)
		x = Dropout(0.4)(x)
		x = tf.add(x_residual, x, name='residual2_2_2')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm2_2_2')(x)
		

		x = Conv2D(128,(3,3),strides=(2,2),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv3_1_1')(x_residual)
		x = BatchNormalization(name='batch_norm3_1_1')(x)
		x = Conv2D(128,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv3_1_2')(x)
		x = Dropout(0.4)(x)
		x_residual = Conv2D(128,(3,3),strides=(2,2),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='shortcut3_1')(x_residual)
		x = tf.add(x_residual, x, name='residual3_1')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm3_1_2')(x)

		x = Conv2D(128,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv3_2_1')(x)
		x = BatchNormalization(name='batch_norm3_2_1')(x)
		x = Dropout(0.4)(x)
		x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv3_2_2')(x)
		x = Dropout(0.4)(x)
		x = tf.add(x_residual, x, name='residual3_2')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm3_2_2')(x)

		x = Conv2D(256,(3,3),strides=(2,2),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv4_1_1')(x_residual)
		x = BatchNormalization(name='batch_norm4_1_1')(x)
		x = Conv2D(256,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv4_1_2')(x)
		x_residual = Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='shortcut4_1')(x_residual)
		x = tf.add(x_residual, x,name='residual4_1')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm4_1_2')(x)

		x = Conv2D(256,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv4_2_1')(x)
		x = BatchNormalization(name='batch_norm4_2_1')(x)
		x = Dropout(0.4)(x)
		x = Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv4_2_2')(x)
		x = Dropout(0.4)(x)
		x = tf.add(x_residual, x, name='residual4_2')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm4_2_2')(x)

		x = Conv2D(512,(3,3),strides=(2,2),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv5_1_1')(x_residual)
		x = BatchNormalization(name='batch_norm5_1_1')(x)
		x = Conv2D(512,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv5_1_2')(x)
		x_residual = Conv2D(512,(3,3),strides=(2,2),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='shortcut5_1')(x_residual)
		x = tf.add(x_residual, x,name='residual5_1')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm5_1_2')(x)

		x = Conv2D(512,(3,3),activation=self.activation,padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv5_2_1')(x)
		x = BatchNormalization(name='batch_norm5_2_1')(x)
		x = Dropout(0.4)(x)
		x = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(self.weight_decay),name='conv5_2_2')(x)
		x = Dropout(0.4)(x)
		x = tf.add(x_residual, x,name='residual5_2')
		#x = Activation('relu')(x)
		x_residual = BatchNormalization(name='batch_norm5_2_2')(x)

		x = Flatten(name='flatten')(x_residual)
		x = Dropout(0.5)(x)
		x = Dense(100,activation='softmax',kernel_regularizer=regularizers.l2(self.weight_decay),name='fc_layer')(x)

		return tf.keras.Model(inputs=input_layer, outputs=x, name='ResNet18')



    
    
    
acc = cross_validation(True,activation_func,dropout,weight_decay)

print(acc)
        