"""
In summary, this is our directory structure:
```
data/
    train/
        clock_4_23_59.png
        (...)
    validation/
        clock_10_21_5.png
        (...)
```
"""

from keras import backend as keras_backend
from representation import *
from keras.utils import np_utils
from keras.datasets import mnist
import models
import coremltools
import data
import representation

def Train(generator,_model,n):

	train,label = generator.generateClockFaces(n)

	batch_size = 32
	if n < batch_size:
		batch_size = n

	_model.fit(train,label,batch_size=batch_size,shuffle=True,epochs=100,verbose=1)

def Learn(_model, generator):
	iterations = 100000
	i = 0
	while True:
		n = int(5000)
		print(i)
		Train(generator,_model,n)
		i += n
		if i >= iterations:
			break


keras_backend.set_image_data_format('channels_last')
#keras_backend.set_image_dim_ordering('th')

train_dir = os.path.join('data', 'train')
(x_train, y_train) = get_images(train_dir)

validation_dir = os.path.join('data', 'validation')
(x_validation, y_validation) = get_images(validation_dir)

num_train_samples = len(x_train)
num_validation_samples = len(x_validation)

img_width = 480
img_height = 480

num_color_channels = 1  # 1 means greyscale

if keras_backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_width, img_height)
    x_validation = x_validation.reshape(x_validation.shape[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
    x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
    x_validation = x_validation.reshape(x_validation.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

print(y_train[0])

print('num_train_samples', num_train_samples)
print('num_validation_samples', num_validation_samples)

model = models.get_cnn_model(num_color_channels, img_width, img_height)
#model = models.get_ann_model(num_color_channels, img_width, img_height)

print(model.summary())

generator = data.ClockGenerator(models.IMG_SIZE,models.INCLUDE_SECONDS_HAND,0.2)
generator.shakeVariance = 16

#Use this for longer
#Learn(model, generator)
#Train(generator, model, 200)

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20
)

model.save('testModel.h5')
#model.save(models.MODEL_H5_NAME)


output_labels = []
for i in range(0,12):
	output_labels.append("hour%d" % i)
for i in range(0,60):
	output_labels.append("minute%d" % i)
for i in range(0,60):
	output_labels.append("second%d" % i)

coreml_model = coremltools.converters.keras.convert('testModel.h5',input_names=['image'],image_input_names='image', class_labels=output_labels, image_scale=1/255.0)
#coreml_model = coremltools.converters.keras.convert(models.MODEL_H5_NAME,input_names=['image'],image_input_names='image', class_labels=output_labels, image_scale=1/255.0)
coreml_model.save('timeTest.mlmodel')
#coreml_model.save('time.mlmodel')
print(coreml_model)
