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
    
    conv10 = Conv2D(2, 1, activation = 'sigmoid')(conv9)
    
#-------------------------------------------------------------------------------------------------------------
    
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy',dice])
    # 编译model,优化器，损失函数，度量

    return model
