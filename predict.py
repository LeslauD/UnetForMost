def predict(imgs):
    print('predict test data')
    predict_img = CimgEdge(CimgDime(model.predict(Raw_prep(imgs), batch_size=1, verbose=1),3),500)
    return predict_img


Pred = predict(Test_Img)



somaPred , vesselPred = CSplMask(Pred)



print('Dice of test data (Soma) :',dice_coef(Test_SMask, somaPred))
pred_plot(Test_Img, Test_SMask, somaPred, order=10)
dice_plot(Test_SMask, somaPred)



print('Dice of test data (Vessel) :',dice_coef(Test_VMask, vesselPred))
pred_plot(Test_Img, Test_VMask, vesselPred, order=10)
dice_plot(Test_VMask, vesselPred)
