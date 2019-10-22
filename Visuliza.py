#---------------------------Fit Plot--------------------------------------------------------------------------
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






def dice_plot(Test_Mask,pred,model):
        dice=[]
        for i in range(len(pred)):
            dice.append(dice_coef(Test_Mask[i],pred[i]))
        plt.plot(np.arange(len(dice)),dice,linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=3)
        plt.title(model+': '+'Figure of Dice_Coef')
        plt.xlabel('picture')
        plt.xticks(np.arange(0,52,2))
        plt.ylabel('dice_coef')
        plt.yticks(np.arange(0,1.2,0.1))
        meanDice=sum(dice)/len(dice)
        plt.text(15, 1.2, 'meanDice=%lf'%meanDice,fontsize=12)
        plt.grid()
        plt.savefig('../result/1D/vessel/Visuliza_DiceCoef_'+model+'.png')
        plt.show()





def pred_plot(raw, lable, pred, order=-1):
    if order==-1:
        order=int(len(raw)/2)
    raw_ = raw[order]
    pred_ = pred[order]
    lable_ = lable[order]
    dice_ = (lable_ ^ bina(pred_).astype('int32'))
    
    print('Dice of raw['+str(order)+']:',dice_coef(lable_,pred_))
    
    fig = plt.figure(figsize=(15,15),facecolor='red')
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



