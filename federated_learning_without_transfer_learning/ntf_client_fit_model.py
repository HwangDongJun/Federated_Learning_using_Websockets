# Setup library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import PIL.Image as Image
from PIL import ImageFile
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import matplotlib.pylab as plt
import efficientnet.tfkeras as efn

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		print(e)


class transfer_learning_fit(object):
	def __init__(self, config, weights):
		self.weights = weights
		self.image_shape = (config['image_shape'], config['image_shape'])
		self.batch_size = config['batch_size']
		self.learning_rate = config['learning_rate']
		self.epochs = config['epochs']
		self.optimizer = config['optimizer']
		self.model_link = config['model']
		self.class_names = np.array(['book', 'laptop', 'phone', 'wash', 'water'])

		tf.random.set_seed(2020)

	def image_generator(self):
		image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
																	rotation_range=15,
																	horizontal_flip=True,
																	brightness_range=[0.7,1.0])
		image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
		return image_gen_train, image_gen_val

	def gen_train_val_data(self):
		gen_train, gen_val = self.image_generator()

		train_data_dir = os.path.abspath('INPUT YOUR TRANING DATA SET PATH')
		train_data_gen = gen_train.flow_from_directory(directory=str(train_data_dir),
											batch_size=self.batch_size,
											color_mode='rgb',
											shuffle=True,
											target_size=self.image_shape,
											classes=list(self.class_names))

		return train_data_gen

	def select_optimizer(self, opti, lr):
		if opti == 'adam':
			return tf.keras.optimizers.Adam(learning_rate=lr)

	def set_model(self, vector_layer):
		#efficient_net = efn.EfficientNetB0(
		#	weights=None,
		#	input_shape=self.image_shape+(3,),
		#	include_top=False,
		#	pooling='max'
		#)

		#model = tf.keras.Sequential([
		#	efficient_net,
		#	layers.Dense(5, activation='softmax')
		#])

		mobilenet_v2 = tf.keras.applications.MobileNetV2(
			weights=None,
			input_shape=self.image_shape+(3,),
			include_top=False,
			pooling='max'
		)

		model = tf.keras.Sequential([
			mobilenet_v2,
			layers.Dense(5, activation='softmax')
		])

		return model

	def build_model(self):
		feature_vector_url = self.model_link
		feature_vector_layer = hub.KerasLayer(feature_vector_url,
										input_shape=self.image_shape+(3,))
		
		feature_vector_layer.trainable = True

		made_model = self.set_model(feature_vector_layer)

		print(made_model.summary())

		made_model.compile(
			optimizer=self.select_optimizer(self.optimizer, self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['acc'])

		return made_model, feature_vector_layer

	def train_model_tosave(self, weight):
		callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

		if weight == list():
			local_model, feature_layer = self.build_model()
			gen_train_data = self.gen_train_val_data()
			local_model.fit_generator(gen_train_data, epochs=self.epochs, callbacks=[callback])
		else:
			local_model, feature_layer = self.build_model()
			gen_train_data = self.gen_train_val_data()
			local_model.set_weights(weight)
			local_model.fit_generator(gen_train_data, epochs=self.epochs, callbacks=[callback])
			
		return local_model.get_weights()

	def get_weight_finetune_model(self, expath, feature_layer, gtrain_data):
		reloaded_model = tf.keras.models.load_model(expath)
		
		feature_layer.trainable = True

		callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

		reloaded_model.compile(
			optimizer=self.select_optimizer(self.optimizer, self.learning_rate*0.1),
			loss='categorical_crossentropy',
			metrics=['accuracy'])
		reloaded_model.fit_generator(gtrain_data, epochs=self.epochs+(self.epochs*2),
						initial_epoch=self.epochs, callbacks=[callback])

		return reloaded_model.get_weights() # Dense layer weight는 제외하고 반환

	def manage_train(self):
		get_weights = list()
		training_weight = self.train_model_tosave(self.weights)
			
		return training_weight
