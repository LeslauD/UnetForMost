#!/usr/bin/env python
# coding: utf-8




import os 
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave





from keras.models import *
from keras.optimizers import *
from keras import regularizers as reg
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback,ModelCheckpoint,LearningRateScheduler
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate,Dropout





def Unet():
    inputs = Input((512,512,1))
    
#-------------------------------------------------------------------------------------------------------------    
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    # 2D卷积，参数：滤波器个数，滤波器尺寸（3*3），激活函数，是否填充（√），kernel初始化方法
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2))(conv1)
    # 2D最大池化，pool_size: 整数，或者 2 个整数表示的元组， 沿（垂直，水平）方向缩小比例的因数。 （2，2）会把输入张量的两个维度都缩小一半。
    # 如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
    
#-------------------------------------------------------------------------------------------------------------
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2))(conv2)
    
#-------------------------------------------------------------------------------------------------------------
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2))(conv3)
    
#-------------------------------------------------------------------------------------------------------------
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2))(drop4)
    # dropout是让某些神经元以一定的概率不工作，参数就是概率值
    # 在每次训练的时候，让一半的特征检测器停过工作，这样可以提高网络的泛化能力
    
#-------------------------------------------------------------------------------------------------------------

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # 下采样结束
#-------------------------------------------------------------------------------------------------------------
    # 开始上采样

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)#原来是3
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    # UpSampling2D可以看作是Pooling的反向操作，就是采用Nearest Neighbor interpolation来进行放大，说白了就是复制行和列的数据来扩充feature map的大小
    # merge:融合层计算输入张量列表的和；它接受一个张量的列表， 所有的张量必须有相同的输入尺寸， 然后返回一个张量（和输入张量尺寸相同）
    
    # 参数列表：
    # layers：该参数为Keras张量的列表，或Keras层对象的列表。该列表的元素数目必须大于1
    # mode：合并模式，如果为字符串，则为下列值之一{“sum”，“mul”，“concat”，“ave”，“cos”，“dot”}
    # concat是将待合并层输出沿着最后一个维度进行拼接，因此要求待合并层输出只有最后一个维度不同。 
#-------------------------------------------------------------------------------------------------------------
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
#-------------------------------------------------------------------------------------------------------------
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
#-------------------------------------------------------------------------------------------------------------
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
#----------------------------------------------------------------------------------------------------------
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
#-------------------------------------------------------------------------------------------------------------
    
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # 编译model,优化器，损失函数，度量

    return model





def Fit_plot(history, batch_size, epoch):
    iters = range(1,epoch+1)
    plt.figure()
    # acc
    plt.plot(iters, history.history['accuracy'], 'r', label='train acc')
    # loss
    plt.plot(iters, history.history['loss'], 'g', label='train loss')
        # val_acc
    plt.plot(iters, history.history['val_accuracy'], 'b', label='val acc')
        # val_loss
    plt.plot(iters, history.history['val_loss'], 'k', label='val loss')
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylim(0,1.1)
    plt.ylabel('acc-loss')
    plt.legend(loc="best")
    plt.title('Metrics (batch_size : '+str(batch_size)+' epoch : '+str(epoch)+' )')       
    plt.savefig('./Visuliza_Metrics.png')
    plt.show()





#normalization
#3D in,3D out
#0~255 to 0~1
def norm(img):
    for i in range(len(img)):
        img[i] = img[i]/255
    return img


# In[26]:


#binarization
#3D in,3D out
#from 0~1 float to 0 or 1 int
def bina(img):
    for i in range(len(img)):
        img[i][img[i]>=0.5]=1
        img[i][img[i]<0.5]=0
    return img


# In[27]:


#change image Edge Lenth
#500*500 to 512*512
#512*512 to 500*500
#3D in,3D out
def CimgEdge(imgs, Edge=512):
    Imgs = []
    for img in imgs:
        Imgs.append(cv.resize(img,(Edge,Edge),cv.INTER_LINEAR))
    Imgs = np.array(Imgs)
    return Imgs


# In[28]:


#change image Dimention
def CimgDime(imgs, Dime=4):
    if Dime==4:
        return imgs[:,:,:,np.newaxis]
    elif Dime==3:
        return np.squeeze(imgs)


# In[29]:


#def data generator
def imgen():
    gen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='wrap',)
    return gen


# In[30]:


#Raw image preprocessing
def Raw_prep(imgs):
    imgs = norm(imgs.astype('float32'))
    imgs = CimgEdge(imgs)
    imgs = CimgDime(imgs)
    return imgs


# In[31]:


#Mask image preprocessing
def Mask_prep(imgs): 
    imgs = norm(imgs)
    imgs = bina(imgs)
    imgs = CimgEdge(imgs)
    imgs = CimgDime(imgs)
    return imgs


# In[32]:


Train_Img = imread('../input/mostdata/Raw100_Traint.tif')
Test_Img = imread('../input/mostdata/Raw50_Test.tif')
Val_Img = imread('../input/mostdata/Raw50_val.tif')
Train_Mask = imread('../input/mostdata/Soma100Lab_Train.tif')
Test_Mask = bina(norm(imread('../input/mostdata/Soma50Lab_Test.tif')))
Val_Mask = imread('../input/mostdata/Soma50Lab_val.tif')


# In[33]:


#Initialize data generator
batch_size = 4
seed = 2019
image_gen = imgen().flow(Raw_prep(Train_Img),shuffle=False, batch_size=batch_size, seed=seed)
mask_gen = imgen().flow(Mask_prep(Train_Mask),shuffle=False, batch_size=batch_size, seed=seed)
Generator = zip(image_gen, mask_gen)
valdata = (Raw_prep(Val_Img), Mask_prep(Val_Mask))


# In[34]:


model = Unet()

epoch=100
steps_per_epoch=75

model_checkpoint = ModelCheckpoint('./model.hdf5', monitor='loss',verbose=1, save_best_only=True)
print('Fitting model...')
print("start time : ",time.strftime('%c'))

history = model.fit_generator(Generator, validation_data=valdata, epochs=epoch, verbose=1, 
                              callbacks=[model_checkpoint],steps_per_epoch=steps_per_epoch, 
                              max_queue_size=batch_size, shuffle=True, initial_epoch=0)

print("finish time : ", time.strftime('%c'))
print('Fit finished!')


# In[35]:


def predict(imgs):
    print('predict test data')
    predict_img = CimgEdge(CimgDime(model.predict(Raw_prep(imgs), batch_size=1, verbose=1),3),500)
    return predict_img


# In[36]:


def dice_coef(y_true, y_pred,smooth=1):  
    #值越大越好（0,1）
    y_true_f = y_true.flatten()#将img_array一维化
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)#|X∩Y|
    dice=(2 * intersection + smooth) / (np.sum(y_true_f*y_true_f) + np.sum(y_pred_f*y_pred_f) + smooth)  #（2*|X∩Y|）/（|X|+|Y|）  2*重叠区域大小/总的大小
    return dice


# In[37]:


def pred_plot(raw, lable, pred, order=-1):
    if order==-1:
        order=int(len(raw)/2)
    raw_ = raw[order]
    pred_ = pred[order]
    lable_ = lable[order]
    dice_ = (lable_ ^ bina(pred_).astype('int32'))
    
    print('Dice of raw['+str(order)+']:',dice_coef(lable_,pred_))
    
    fig = plt.figure(figsize=(15,15),facecolor='green')
    raw = fig.add_subplot(221)
    lable = fig.add_subplot(222)
    pred = fig.add_subplot(223)
    dice = fig.add_subplot(224)
    
    raw.set_title('Raw Image')
    raw.axis('off')
    raw.imshow(raw_,cmap="gray")
    
    lable.imshow(lable_,cmap="gray")
    lable.axis('off')
    lable.set_title('Lable Image')
    
    pred.imshow(pred_,cmap="gray")
    pred.axis('off')
    pred.set_title('Predict Image')
    
    dice.imshow(dice_,cmap="gray")
    dice.axis('off')
    dice.set_title('XOR Iamge')


# In[38]:


Fit_plot(history, batch_size, epoch) 


# In[39]:


Pred = predict(Test_Img)


# In[40]:


print('Dice of test data:',dice_coef(Test_Mask, Pred))
pred_plot(Test_Img, Test_Mask, Pred, order=1)


# In[ ]:





# In[ ]:




