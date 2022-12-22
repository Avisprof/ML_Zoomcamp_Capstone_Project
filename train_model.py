
import numpy as np
import pandas as pd
import datetime
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

import bentoml

KERAS_MODEL_FILE_NAME = 'kitchenware_image_classiffication.h5'
BENTO_MODEL_FILE_NAME = 'bento_kitchenware_image_classiffication'

IMAGE_FOLDER = "./images"
SEED = 41

# hyperparameters
LEARNING_RATE = 0.01
SIZE_INNER = 250
DROPOUT_RATE = 0.2
INPUT_SIZE = 299

def make_model(model_class, 
              learning_rate=0.01,  
              size_inner=None,
              dropout_rate=None):
    
    input_shape = (INPUT_SIZE, INPUT_SIZE, 3)
    
    base_model = model_class(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
          
    model = keras.Sequential()
    
    model.add(keras.Input(shape=input_shape))
    model.add(base_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    
    if size_inner is not None:
        model.add(keras.layers.Dense(size_inner, activation='relu'))
        
    if dropout_rate is not None:
        model.add(keras.layers.Dropout(dropout_rate))
        
    model.add(keras.layers.Dense(6))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model 

# read the train data
data = pd.read_csv('train.csv')
data.columns = data.columns.str.lower()
data['file_name'] = data['id'].map('{:04}.jpg'.format)

#split the data
df_train_full, _ = train_test_split(data, train_size=100, random_state=SEED)
df_train, df_valid = train_test_split(df_train_full, test_size=0.2, random_state=SEED)

# fit model
start_date = datetime.datetime.now()
print(start_date.strftime('%H:%M:%S'))

model = make_model(InceptionV3,
                       learning_rate=LEARNING_RATE, 
                       size_inner=SIZE_INNER, 
                       dropout_rate=DROPOUT_RATE)

       
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_dataframe(df_train, 
                                        directory=IMAGE_FOLDER, 
                                        x_col='file_name', 
                                        y_col='label', 
                                        target_size=(INPUT_SIZE, INPUT_SIZE),
                                        batch_size=32,
                                        shuffle=True)

classes = np.array(list(train_ds.class_indices.keys()))

val_ds = train_gen.flow_from_dataframe(df_valid, 
                                    directory=IMAGE_FOLDER, 
                                    x_col='file_name', 
                                    y_col='label', 
                                    target_size=(INPUT_SIZE, INPUT_SIZE),
                                    batch_size=32,
                                    shuffle=False)

history = model.fit(train_ds, epochs=3, validation_data=val_ds)

time_delta = (datetime.datetime.now() - start_date).seconds
print(f'[RESULT] {time_delta} seconds, ' 
            f'val_accuracy: {np.mean(history.history["val_accuracy"]):.3f}' 
            f' +- {np.std(history.history["val_accuracy"][-5:]):.2f}')
print()


# save model to the file
model.save(KERAS_MODEL_FILE_NAME)
print(f'Fitted model is saved to {KERAS_MODEL_FILE_NAME}')

# save to bentoml
bentoml.keras.save_model(BENTO_MODEL_FILE_NAME, model, custom_objects={'classes':classes})
print(f'Fitted model is saved to bentoml with name {BENTO_MODEL_FILE_NAME}')






















