import os
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

data_dir = os.path.join(os.getcwd(),'data')

with open(os.path.join(data_dir,'images.p'),'rb') as f:
    images = pickle.load(f)

with open(os.path.join(data_dir,'labels.p'),'rb') as f:
    labels = pickle.load(f)

images = images.astype('float32') / 255.0

images = images.reshape(
    images.shape[0],
    100,
    100,
    1
)

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

labels = to_categorical(labels)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(100,100,1)
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dense(
        labels.shape[1],
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    images,
    labels,
    epochs=10,
    batch_size=16
)

model.save("final_model.h5")

print("Training Complete")
print("final_model.h5 Created")