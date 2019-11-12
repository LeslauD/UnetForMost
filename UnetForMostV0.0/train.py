Train_Img = imread('./MostData/Raw100_Train.tif')
Test_Img = imread('./MostData/Raw50_Test.tif')
Val_Img = imread('./MostData/Raw50_val.tif')
Train_SMask = imread('./MostData/Soma100Lab_Train.tif')
Test_SMask = bina(norm(imread('./MostData/Soma50Lab_Test.tif')))
Val_SMask = imread('./MostData/Soma50Lab_val.tif')
Train_VMask = imread('./MostData/Vessel100Lab_Train.tif')
Test_VMask = bina(norm(imread('./MostData/Vessel50Lab_Test.tif')))
Val_VMask = imread('./MostData/Vessel50Lab_val.tif')



Train_Mask = CFatMask(Mask_prep(Train_SMask),Mask_prep(Train_VMask))
Val_Mask = CFatMask(Mask_prep(Val_SMask),Mask_prep(Val_VMask))



#Initialize data generator
batch_size = 4
seed = 2019
image_gen = imgen().flow(Raw_prep(Train_Img),shuffle=False, batch_size=batch_size, seed=seed)
mask_gen = imgen().flow(Train_Mask,shuffle=False, batch_size=batch_size, seed=seed)
Generator = zip(image_gen, mask_gen)
valdata = (Raw_prep(Val_Img), Val_Mask)




model = Unet()

epoch=30
steps_per_epoch=75

model_checkpoint = ModelCheckpoint('./model.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
print('Fitting model...')
print("start time : ",time.strftime('%c'))

history = model.fit_generator(Generator, validation_data=valdata, epochs=epoch, verbose=1, 
                              callbacks=[model_checkpoint],steps_per_epoch=steps_per_epoch, 
                              class_weight='auto',max_queue_size=batch_size, shuffle=False, initial_epoch=0)

print("finish time : ", time.strftime('%c'))
print('Fit finished!')


Fit_plot(history, batch_size, epoch) 


