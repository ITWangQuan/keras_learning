import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
# 载入自己写的API
from load_data import load_data
# 定义超参数
train_num = 60000
test_num = 10000
batch_size = 128
num_classes = 10
epochs = 20
# load data
(X_train, y_train), (X_test, y_test) = load_data()
# 把原始输入的维度(train_num, 28, 28)转成(train_num, 784)
# 在进行数据类型的转换和数据归一化处理
X_train = X_train.reshape(train_num, 784).astype('float32')
X_test = X_test.reshape(test_num, 784).astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# label为0~9共10个类别，keras要求格式为binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Dense表示全连接
model = Sequential()    # 创建一个模型实例
# 创建输入层到第一个隐藏层的网络模型
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))  # 每次随机舍弃的神经元的百分比
# 创建第一个隐藏层到第二个隐藏层的网络模型
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # 随机舍弃的神经元的百分比
# 创建输出层的网络模型
model.add(Dense(num_classes, activation='softmax'))

model.summary()  # 打印出模型概况
# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
# batch_size 表示每次输入的样本数，即批量大小
# epochs，迭代的epoch数量，即整个训练集训练的epoch数
# verbose 是屏显模式，值为0表示不显屏，值为1，表示显示进度条
#  值为2表示每个epoch都显示一行数据
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

