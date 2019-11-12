#def data generator
def imgen():
    gen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='wrap',)
    return gen