#normalization
#3D in,3D out
#0~255 to 0~1
def norm(img):
    for i in range(len(img)):
        img[i] = img[i]/255
    return img





#binarization
#3D in,3D out
#from 0~1 float to 0 or 1 int
def bina(img):
    for i in range(len(img)):
        img[i][img[i]>=0.5]=1
        img[i][img[i]<0.5]=0
    return img




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





#change image Dimention
def CimgDime(imgs, Dime=4):
    if Dime==4:
        return imgs[:,:,:,np.newaxis]
    elif Dime==3:
        return np.squeeze(imgs)


#Raw image preprocessing
def Raw_prep(imgs):
    imgs = norm(imgs.astype('float32'))
    imgs = CimgEdge(imgs)
    imgs = CimgDime(imgs)
    return imgs



#Mask image preprocessing
def Mask_prep(imgs): 
    imgs = norm(imgs)
    imgs = bina(imgs)
    imgs = CimgEdge(imgs)
    imgs = CimgDime(imgs)
    return imgs


def CFatMask(soma,vess):
    mask = np.concatenate((soma,vess), axis = 3)
    return mask
def CSplMask(mask):
    soma = mask[:,:,:,0]
    vess = mask[:,:,:,1]
    return soma,vess


#img array in   | img array out
#Erode (腐蚀) Dilate (膨胀)  Change
def EDC(img):
    img = img.astype('uint8')
    kernel = np.ones((5,5),np.uint8)
    imgE = cv.erode(img, kernel, iterations = 2)  #对原图进行两次腐蚀
    imgD = cv.dilate(img, kernel, iterations = 2) #对原图进行两次膨胀
    
    return imgE,imgD



def dice_coef(y_true, y_pred,smooth=500*500*0.01):  
    y_trueE,y_trueD = EDC(y_true)
    y_trueE_f = y_trueE.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_trueE_f * y_pred_f)
    y_trueED=y_trueE+1-y_trueD
    dice=(2 * intersection + smooth) / ( np.sum(y_trueE_f) + np.sum( y_pred_f*y_trueED.flatten() ) +smooth )
    return dice

def dice(y_true, y_pred,smooth=1):  
    #值越大越好（0,1）
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    y_true_f = y_true.flatten()#将img_array一维化
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)#|X∩Y|
    dice=(2 * intersection + smooth) / (np.sum(y_true_f*y_true_f) + np.sum(y_pred_f*y_pred_f) + smooth)  #（2*|X∩Y|）/（|X|+|Y|）  2*重叠区域大小/总的大小
    return dice


def diceloss(y_true, y_pred,smooth=1):
    return 1-dice(y_true, y_pred,smooth=1)





