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

		tf.random.set_seed(99)

	def image_generator(self):
		image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
																	rotation_range=15,
																	horizontal_flip=True,
																	brightness_range=[0.7,1.0])
		image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
		return image_gen_train, image_gen_val

	def gen_train_val_data(self):
		gen_train, gen_val = self.image_generator()

		train_data_dir = os.path.abspath('CLIENT TRAIN DATASET PATH')
		train_data_gen = gen_train.flow_from_directory(directory=str(train_data_dir),
												batch_size=self.batch_size,
												color_mode='rgb',
												shuffle=True,
												target_size=self.image_shape,
												classes=list(self.class_names))
		val_data_dir = os.path.abspath('CLIENT TEST DATASET PATH')
		val_data_gen = gen_val.flow_from_directory(directory=str(val_data_dir),
											batch_size=self.batch_size,
											color_mode='rgb',
											shuffle=True,
											target_size=self.image_shape,
											classes=list(self.class_names))
		return train_data_gen, val_data_gen

	def select_optimizer(self, opti, lr):
		if opti == 'adam':
			return tf.keras.optimizers.Adam(learning_rate=lr)

	def set_model(self, vector_layer):
		model = tf.keras.Sequential([
			vector_layer,
			layers.Dense(5, activation='softmax')
		])
		return model

	def build_model(self):
		feature_vector_url = self.model_link
		feature_vector_layer = hub.KerasLayer(feature_vector_url,
										input_shape=self.image_shape+(3,))

		# train setting false
		feature_vector_layer.trainable = False

		made_model = self.set_model(feature_vector_layer)
		made_model.compile(
			optimizer=self.select_optimizer(self.optimizer, self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['acc'])

		return made_model, feature_vector_layer

	def train_model_tosave(self, weight=None):
		local_model, feature_layer = self.build_model()
		gen_train_data, gen_val_data = self.gen_train_val_data()
		local_model.fit_generator(gen_train_data, epochs=self.epochs,
					validation_data=gen_val_data)

		# model layer 0번째 set weight
		if weight != None:
			local_model.layers[1].set_weights(weight)

		# Export model
		export_path = 'EXPORT MODEL PATH' 
		local_model.save(export_path, save_format='tf')

		return feature_layer, export_path, gen_train_data, gen_val_data

	def get_weight_finetune_model(self, expath, feature_layer, gtrain_data, gval_data):
		reloaded_model = tf.keras.models.load_model(expath)
		
		feature_layer.trainable = True

		reloaded_model.compile(
			optimizer=self.select_optimizer(self.optimizer, self.learning_rate*0.1),
			loss='categorical_crossentropy',
			metrics=['accuracy'])
		reloaded_model.fit_generator(gtrain_data, epochs=self.epochs+(self.epochs*2),
						initial_epoch=self.epochs, validation_data=gval_data)

		return reloaded_model.get_weights() # Dense layer weight는 제외하고 반환

	def manage_train(self):
		get_weights = list()
		if self.weights != list():
			flayer, epath, gtrain, gval = self.train_model_tosave(self.weights)
			get_weights = self.get_weight_finetune_model(epath, flayer, gtrain, gval)
		else: # 처음 학습이면 일반 방식 그대로 사용
			flayer, epath, gtrain, gval = self.train_model_tosave()
			get_weights = self.get_weight_finetune_model(epath, flayer, gtrain, gval)

		return get_weights
