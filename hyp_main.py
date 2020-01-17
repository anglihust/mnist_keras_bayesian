import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense,Conv2D, BatchNormalization,Dropout, Input,normalization,Activation,add,MaxPooling2D, Flatten,GaussianNoise
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import os
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
import matplotlib.pyplot as plt

def data():
    train = pd.read_csv(r'L:\Users\Ang\Documents\data\kaggle\digital\data\train.csv')
    label = train['label']
    data =  train.drop(labels=['label'],axis=1)
    label = label.to_numpy()
    label = to_categorical(label,num_classes = 10)
    data =  data.to_numpy()
    data=data.reshape(-1,28,28)
    data=data.reshape(-1,28,28,1)
    data = data/255.0
    datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    X_train, X_val, Y_train, Y_val = train_test_split(data, label, test_size = 0.1)
    datagen.fit(X_train)
    return datagen, X_train, Y_train, X_val, Y_val

def test_data():
    test = pd.read_csv(r'L:\Users\Ang\Documents\data\kaggle\digital\data\test.csv')
    test_data= test.to_numpy()
    test_data=test_data.reshape(-1,28,28)
    test_data=test_data.reshape(-1,28,28,1)
    test_data = test_data/255.0
    return test_data


def model(datagen,X_train,Y_train,X_val,Y_val):
    num_layers1 = {{choice([48, 64, 96])}}
    num_layers2 = {{choice([96, 128, 192])}}
    num_layers3 = {{choice([192, 256, 512])}}
    lrate = {{choice([0.0001, 0.0004,0.0008])}}
    epochs = 60
    batch_size = 64

    inputs = Input((28,28,1))
    nois=GaussianNoise(0.2)(inputs)
    conv1 = Conv2D(num_layers1, (3, 3), activation=None, padding='same')(nois)
    conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(num_layers1, (3, 3), activation=None, padding='same')(conv1)
    conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(num_layers1, (3, 3), activation=None, padding='same')(conv2)
    conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3= MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3= Dropout({{uniform(0,0.5)}})(conv3)

    conv4 = Conv2D(num_layers2, (3, 3), activation=None, padding='same')(conv3)
    conv4 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv4)
    conv4 = Activation('relu')(conv4)

    conv5 = Conv2D(num_layers2, (3, 3), activation=None, padding='same')(conv4)
    conv5 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = Conv2D(num_layers2, (3, 3), activation=None, padding='same')(conv5)
    conv6 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6= MaxPooling2D(pool_size=(2, 2))(conv6)
    conv6= Dropout({{uniform(0,0.5)}})(conv6)


    conv7 = Conv2D(num_layers3, (3, 3), activation=None, padding='same')(conv6)
    conv7 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv7)
    conv7 = Activation('relu')(conv7)

    conv8 = Conv2D(num_layers3, (3, 3), activation=None, padding='same')(conv7)
    conv8 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = Conv2D(num_layers3, (3, 3), activation=None, padding='same')(conv8)
    conv9 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv9)
    conv9 = Activation('relu')(conv9)
    conv9= MaxPooling2D(pool_size=(2, 2))(conv9)
    conv9= Dropout({{uniform(0,0.5)}})(conv9)
    conv9=Flatten()(conv9)

    dout1= Dense(256,activation = 'relu')(conv9)
    dout1 = normalization.BatchNormalization(epsilon=2e-05, axis=-1, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(dout1)
    dout1= Dropout({{uniform(0,0.5)}})(dout1)
    dout2 =Dense(10,activation = 'softmax')(dout1)
    model = Model(inputs=inputs, outputs=dout2)

    optimizer=Adam(lr=lrate, beta_1=0.9, beta_2=0.95, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    save_path=os.getcwd()
    checkpointer = []
    #checkpointer.append(ModelCheckpoint(filepath=os.path.join(save_path,'best_model.hdf5'), verbose=1, save_best_only=True))
    checkpointer.append(ReduceLROnPlateau(monitor='val_acc', patience=8, verbose=1, factor=0.5, min_lr=0.00001))
    checkpointer.append(EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True))

    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 32, steps_per_epoch=X_train.shape[0] // batch_size
                              ,callbacks=checkpointer)
    score, acc = model.evaluate(X_val, Y_val, verbose=0)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}






def main():
    datagen, X_train, Y_train, X_test, Y_test = data()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))

    tdata=test_data()
    results = best_model.predict(tdata)

    results = np.argmax(results,axis = 1)

    results = pd.Series(results,name='label')
    submission = pd.concat([pd.Series(range(1,28001),name = 'ImageId'),results],axis = 1)

    submission.to_csv('cnn_mnist_submission.csv',index=False)

if __name__ == '__main__':
    main()
