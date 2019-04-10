
# coding: utf-8

# # 深度学习-3 实现卷积神经网络

# ## 前言

# 在了解了卷积网络之后，不难发现，基本上所有的卷积网络都是按照："卷积层->池化层->卷积层->池化层...->全连接层->输出层"这样的形式进行堆叠排列的。这样的层级结构在Keras中使用Sequential模型来实现极为方便。

# ## 在Keras中实现卷积网络

# 首先，卷积网络是神经网络的一种，因此卷积网络中需要有各种层与激活函数。这些层有全连接层(Dense Layer)、卷积层(Conv Layer)和池化层(Pooling Layer)，因此需要导入Keras的Layers包中的如下内容：

# In[2]:


from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# 第四小节代码
from tensorflow.keras.callbacks import TensorBoard
import time
model_name = "kaggle_cat_dog-cnn-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

# 在准备完成之后，通过Pickle加载上一讲中准备好的X(特征集)与y(标签集)数据。

# In[4]:


import pickle
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))


# 为了更好地训练神经网络，在训练之前进行数据预处理显得至关重要。将数据进行归一化(Normalization)或者白化(Whitening)处理之后，神经网络算法的效果可以得到明显的提升。
# 归一化通常有：减均值、大小缩放和标准化三个方法。面对图像数据来说(像素取值范围通常是0-255)，大小缩放是最好的方法，此时只要将数据除以255即可。

# In[6]:


X = X / 255


# 下面便是使用Keras的Sequential模型来实现卷积网络了。
# 让我们来定义一个最简单的卷积网络，首先约定命名方法。使用Cx来标记卷积层，使用Px来表示池化层，使用Fx来表示全连接层。其中，x表示层的下标。
# 这个最简单的卷积网络将会由如下的几个层构成(读者可以自由扩展，或按照诸如LeNet-5等经典的网络结构来编写)。
# 首先是C1卷积层，这个卷积层是一个二维卷积层（Conv2D），卷积核的数目是64，即输出的维度是64，卷积核的大小为3x3，当使用该层作为第一层时，应提供input_shape参数。例如```input_shape = (128,128,3)```代表128x128的彩色RGB图像```（data_format='channels_last'）```。
# 紧接着C1卷积层需要提供一个激活函数，这里采用ReLU作为激活函数。
# 然后就是P1池化层，池化层采用最常见的MaxPooling，采用2x2的池化大小，也就是下采样因子。
# C2，P2重复上述层。
# 最后再输出之前我们需要将上述几个层的输出作为全连接层的输入。由于卷积层和池化层的输出是2D的，因此需要将其压平，此时需要用到Flatten，而后使用sigmoid激活函数将结果输出(我们只需对猫和狗进行分类，如果分类的类别过多地话则可以用softmax作为激活函数)。
# 这样，整个网络的结构就被定义完了。在进行训练之前，需要提供损失函数和优化函数等。这里使用二元交叉熵作为损失函数，采用ADAM作为优化器，使用accuracy作为性能评估项。
# 上述代码如下：

# In[9]:


import tensorflow as tf
# import keras
model = tf.keras.models.Sequential()

model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #Conv Layer是2D， DenseLayer是1D的 所以需要将ConvLayer压平
# model.add(Dense(64))
# model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"]) # 可以使用categorical_crossentropy作为损失函数

# model.fit(X, y, batch_size =32, epochs=10, validation_split=0.1)
# 第四小节代码
model.fit(X, y, batch_size =32, epochs=10, validation_split=0.1, callbacks=[tensorboard])

# 通过```model.fit```方法，我们给模型提供了数据集，指明了训练的epoch和验证集划分的比例。

# 可以看到，通过10个epochs之后，卷积网络的预测准确率在验证集上达到了0.8以上。这个结果也是普通的MLP绝对无法做到的。
# 通过采用更加先进的网络结构，可以得到更高的准确率。

# ## 图像分类模型

# 以时间为顺序，有这些卷积网络被提出。
# LeNet-5，AlexNet，VGGNet，GoogLeNet，ResNet和DenseNet等。这些图像分类模型的不断发展，推进着图像分类准确率不断提升。
# 这些网络结构虽然越来越复杂，但都可以通过Keras或是Tensorflow来实现。读者可以选择特定的网络，以提高准确度作为目标，来实现相应的图像分类模型。

# In[ ]:




