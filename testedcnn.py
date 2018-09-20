# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tqdm
import os
# Initialising the Convolutional neural network
Image_classifier = Sequential()
# Convolution
Image_classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Pooling
Image_classifier.add(MaxPooling2D(pool_size = (2, 2)))
# second convolutional layer
Image_classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
Image_classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Flattening
Image_classifier.add(Flatten())
# Full connection
Image_classifier.add(Dense(units = 128, activation = 'relu'))
Image_classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
Image_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_img_gen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_img_gen = ImageDataGenerator(rescale = 1./255)
training_set = train_img_gen.flow_from_directory('C:/Users/pradn/Desktop/BE/examples/test/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_img_gen.flow_from_directory('C:/Users/pradn/Desktop/BE/examples/test/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
Image_classifier.fit_generator(training_set,
steps_per_epoch = 40,
epochs = 5,
validation_data = test_set,
validation_steps = 20)
# Predicting the output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image

labels=[]
img_name=[]
for img in os.listdir("C:/Users/pradn/Desktop/BE/examples/prediction"):
	img_name.append("C:/Users/pradn/Desktop/BE/examples/prediction/"+img)
	test_image = image.load_img("C:/Users/pradn/Desktop/BE/examples/prediction/"+img, target_size = (64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = Image_classifier.predict(test_image)
	training_set.class_indices
	#print(img)
	"""imagess=mpimg.imread("C:/Users/pradn/Desktop/BE/examples/prediction/"+img)
	plt.imshow(imagess)
	plt.show()
	if result[0][0] == 1:
		prediction = 'yoga'
	else:
		prediction = 'dance'
	print(prediction)
	"""
	if result[0][0] == 1:
		prediction = 'yoga'
	else:
		prediction = 'dance'
	labels.append(prediction)
#code for plotting results
fig=plt.figure()
for num,data in enumerate(img_name[:12]): 
    final_plot = fig.add_subplot(3,4,num+1)
    orig = mpimg.imread(data)
    final_plot.imshow(orig)
    plt.title(labels[num])
    final_plot.axes.get_xaxis().set_visible(False)
    final_plot.axes.get_yaxis().set_visible(False)
plt.show()