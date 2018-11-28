import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path

import tensorflow as tf
import keras
from keras import backend as K

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Sequential, Model 
from keras import models
from keras import layers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import optimizers

from  keras.applications import VGG16

import timeit



time = 0

'''
mri_weights = ["T2_1"] #, "T2_2", "T2"]
positions = ["Axial"] #, "Axial", "Sagital"]
nTrains = [135+168] #, 135+164, 40+48]
nValidations = [34+42] #, 34+42, 8+12]'''

f = open("model_times.txt","a+")

nTrain = 135+168
nVal = 34+42

train_dir = "Data/train/"
validation_dir = "./Data/validate/"

val = 512
lVal = 7
epochsNum = 100
nClasses = 2
image_size = 224

position = "Axial"
mri_weight = "T2_1"
num_layers = [4, 8, 12, 16, 18]

for num_layer in num_layers:
    print("............." + position + " " + mri_weight + " " + str(num_layer) + ".............")

    vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    # Freeze the layers except the last 4 layers
    for layer in vgg16_conv.layers[:-num_layer]:
        layer.trainable = False
     
    # Check the trainable status of the individual layers
    for layer in vgg16_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()
     
    # Add the vgg16 convolutional base model
    model.add(vgg16_conv)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))
     
    # Show a summary of the model. Check the number of trainable parameters
    print(model.summary())

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
     
    # Change the batchsize according to your system RAM
    train_batchsize = 32
    val_batchsize = 32
     
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='categorical',
            shuffle = True)
     
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle = True)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-6),
                  metrics=['acc'])

    #save model
    model_json = model.to_json()
    with open("vgg16_model"+position+"_"+mri_weight+"_"+str(num_layer)+".json", "w") as json_file:
        json_file.write(model_json)

    model_save_name = "vgg16_"+position+"_"+mri_weight+"_"+str(num_layer)+".h5"

    checkpoint = ModelCheckpoint(model_save_name, 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False, 
        mode='auto', 
        period=1)

    early = EarlyStopping(monitor='val_acc', 
        min_delta=0.001, 
        patience=8, 
        verbose=1, 
        mode='auto')

    # Train the model
    start = timeit.default_timer()

    history = model.fit_generator(
          train_generator,
          steps_per_epoch=train_generator.samples/train_generator.batch_size ,
          epochs=50,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples/validation_generator.batch_size,
          verbose=1,
          callbacks = [checkpoint, early])

    end = timeit.default_timer()
    time = end - start

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
     
    epochs = range(len(acc))

    f.write("model " + position + " " + mri_weight + " " + str(num_layer) + str(time) + " " + str(val_acc[np.argmax(val_acc)]) + "\r\n")

    plt.figure()
    fig = plt.gcf()
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy ' + position + " " + mri_weight)
    plt.legend()
    fig.savefig(position + "_VGG_" + mri_weight + '_acc'+"_"+str(num_layer)+'.png', dpi=100)
     
    plt.figure()
    fig = plt.gcf()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss' + position + " " + mri_weight)
    plt.legend()
    fig.savefig(position + "_VGG_" + mri_weight + '_loss'+"_"+str(num_layer)+'.png', dpi=100)

    from sklearn.metrics import confusion_matrix

    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=True)

    # Get the filenames from the generator
    fnames = validation_generator.filenames
     
    # Get the ground truth from generator
    ground_truth = validation_generator.classes
     
    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices
     
    # Getting the mapping from class index to class label
    idx2label = dict((v,y) for y,v in label2index.items())
     
    # Get the predictions from the model using the generator
    predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)
     
    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

    print(confusion_matrix(ground_truth, predicted_classes))

    K.clear_session()
    tf.reset_default_graph()

f.close()

