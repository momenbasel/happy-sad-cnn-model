import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import imghdr


image_exts = ['jpeg', 'jpg', 'bmp', 'png']
#
# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         try:
#             img = cv2.imread(image_path)
#             tip = imghdr.what(image_path)
#             if tip not in image_exts:
#                 print('image not in ext list {}'.format(image_path))
#                 os.remove(image_path)
#         except Exception as e:
#             print('issue with {}'.format(image_path))


data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

#images as numpy array
#print(batch[0])


scaled = batch[0] /255
data = data.map(lambda x, y: (x/255,y))

# scaled_iterator = data.as_numpy_iterator().next()
# print(scaled_iterator[0].min())

#split
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

# print(train_size+val_size+test_size)

#take_data
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

###train

# data_dir = 'data'
#
# model = Sequential()
#
# model.add(Conv2D(16, (3,3), 1 , activation='relu'))
#
# model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3,3), 1, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(16, (3,3), 1, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
# logdir='logs'
#
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
#



##test
# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()
#
# for batch in test.as_numpy_iterator():
#     X, y = batch
#     yhat = model.predict(X)
#     pre.update_state(y, yhat)
#     re.update_state(y, yhat)
#     acc.update_state(y, yhat)
#
#
#
# print(pre.result(), re.result(), acc.result())



##save
# model.save(os.path.join('models','imageclassifier.h5'))
# new_model = load_model('imageclassifier.h5')
# new_model.predict(np.expand_dims(resize/255, 0))



##real data test
new_model = load_model('models/imageclassifier.h5')
img = cv2.imread('sad.jpg')
resize = tf.image.resize(img, (256,256))
answer = new_model.predict(np.expand_dims(resize/255, 0))

if answer > 0.5:
    print('person is sad :(')
else:
    print('person is happy :)')


plt.imshow(img)
plt.show()