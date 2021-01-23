from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pylab as plt
import efficientnet.tfkeras as efn

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from dp_optimizer import DPAdamGaussianOptimizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# "" -> using CPU, "0" -> using first gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
		self.l2_norm_clip = config['l2_norm_clip']
		self.noise_multiplier = config['noise_multiplier']
		self.num_microbatches = config['num_microbatches']
		self.total_data_size = config['total_data_size']
		self.delta = config['delta']
	
	# compute epsilon for differential privacy but not used here code.
	def compute_epsilon(self, steps):
		if self.noise_multiplier == 0.0:
			return float('inf')
		orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
		sampling_probability = self.batch_size / self.total_data_size
		rdp = compute_rdp(q=sampling_probability, noise_multiplier=self.noise_multiplier, steps=steps, orders=orders)
		return get_privacy_spent(orders, rdp, target_delta=self.delta)[0]

	# Same as the description of the comput_epsilon function.
	class EpsilonPrintingCallback(tf.keras.callbacks.Callback):
		def __init__(self):
			self.eps_history = list()
			super().__init__()

		def on_epoch_end(self, epoch, logs=None):
			# this compute 6920, 32 -> change every time it changes.
			eps = transfer_learning_fit.compute_epsilon(self, steps=(epoch + 1) * (self.total_data_size // self.batch_size))
			self.eps_history.append(eps)
			print(', eps = {}'.format(eps))

	def image_generator(self):
		image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
																	rotation_range=15,
																	horizontal_flip=True,
																	brightness_range=[0.7,1.0])

		return image_gen_train

	def get_gen_train_data(self):
		gen_train = self.image_generator()

		train_data_dir = os.path.abspath('/home/dnlabblocksci/websocket/3/dataset/data_balanced/new_train/client1/')
		train_data_gen = gen_train.flow_from_directory(directory=str(train_data_dir),
											batch_size=self.batch_size,
											color_mode='rgb',
											shuffle=True,
											target_size=self.image_shape,
											classes=list(self.class_names))

		return train_data_gen

	def select_optimizer(self, opti, lr):
		if opti == 'dp': # When using differential privacy.
			optimizer = DPAdamGaussianOptimizer(
				l2_norm_clip=self.l2_norm_clip,
				noise_multiplier=self.noise_multiplier,
				num_microbatches=self.num_microbatches,
				learning_rate=lr)

			return optimizer
		elif opti == 'not_dp': # When not using differential privacy.
			print('check not dp')
			optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

			return optimizer

	def set_model(self, base_model):
		if self.model_link == 'mobilenetv2':
			in_image = layers.Input(shape=self.image_shape+(3,))
			net = base_model(in_image)
			net = layers.GlobalAveragePooling2D()(net)
			net = layers.Dense(units=5, activation='softmax')(net)
			model = tf.keras.models.Model(in_image, net)
		
			return model
		elif self.model_link == 'efficientnetb0':
			model = tf.keras.Sequential([
				base_model,
				layers.Dense(5, activation='softmax')
			])

			return model
		elif self.model_link == 'cnn':
			model = tf.keras.Sequential([
				layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.image_shape+(3,)),
				layers.MaxPool2D((2, 2)),
				layers.Conv2D(32, (3, 3), activation='relu'),
				layers.MaxPool2D((2, 2)),
				layers.Conv2D(32, (3, 3), activation='relu'),
				layers.Flatten(),
				layers.Dense(32, activation='relu'),
				layers.Dense(16, activation='relu'),
				layers.Dense(5, activation='softmax')
			])

			return model

	# There may be duplicates of code, but to help understanding.
	def build_model(self, weight=None):
		if self.model_link == "mobilenetv2":
			mobilenet_model = tf.keras.applications.MobileNetV2(input_shape=self.image_shape+(3,),
															include_top=False,
															weights='imagenet',
															classes=5)

			if weight == None:
				mobilenet_model.trainable = False
			else:
				mobilenet_model.trainable = True
	
			made_model = self.set_model(mobilenet_model)
			print(made_model.summary())
	
			loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)

			if self.optimizer == 'not_dp':
				print('check not dp loss')
				loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	
			made_model.compile(optimizer=self.select_optimizer(self.optimizer, self.learning_rate),
								loss=loss,
								metrics=['accuracy'])

			return made_model, mobilenet_model
		elif self.model_link == "efficientnetb0":
			efficientnet_model = efn.EfficientNetB0(
									weights='imagenet',
									input_shape=self.image_shape+(3,),
									include_top=False,
									pooling='max',
									classes=5)

			if weight == None:
				efficientnet_model.trainable = False
			else:
				efficientnet_model.trainable = True

			made_model = self.set_model(efficientnet_model)
			print(made_model.summary())

			loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)

			if self.optimizer == 'not_dp':
				print('check not dp loss')
				loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

			made_model.compile(optimizer=self.select_optimizer(self.optimizer, self.learning_rate),
								loss=loss,
								metrics=['accuracy'])

			return made_model, efficientnet_model
		elif self.model_link == "cnn":
			made_model = self.set_model(None)
			print(made_model.summary())
			
			loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)

			if self.optimizer == 'not_dp':
				print('check not dp loss')
				loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

			made_model.compile(optimizer=self.select_optimizer(self.optimizer, self.learning_rate),
								loss=loss,
								metrics=['accuracy'])

			return made_model

	def train_model_tosave(self, weight=None):
		earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
		#eps_callback = self.EpsilonPrintingCallback()

		if self.model_link == 'cnn':
			if weight == list():
				local_model = self.build_model()
				gen_train_data = self.get_gen_train_data()
				local_model.fit(gen_train_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=[earlystop_callback])
			else:
				local_model = self.build_model()
				gen_train_data = self.get_gen_train_data()
				local_model.set_weights(weight)
				local_model.fit(gen_train_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=[earlystop_callback])

			return local_model.get_weights()
		else:
			if weight == None:
				local_model, base_model = self.build_model()
				gen_train_data = self.get_gen_train_data()
			
				#fit_history = local_model.fit(gen_train_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=[eps_callback, earlystop_callback])
				#eps_history = eps_callback.eps_history
				local_model.fit(gen_train_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=[earlystop_callback])
			
				return local_model, base_model, gen_train_data#, eps_history
			else:
				local_model, base_model = self.build_model(list())
				gen_train_data = self.get_gen_train_data()
				local_model.set_weights(weight)
				#fit_history = local_model.fit(gen_train_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=[eps_callback, earlystop_callback])
				#eps_history = eps_callback.eps_history
				local_model.fit(gen_train_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=[earlystop_callback])

				return local_model.get_weights()#, eps_history

	def get_weight_finetune_model(self, bmodel, gtrain_data, lmodel):
		bmodel.trainable = True

		earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
		#eps_callback = self.EpsilonPrintingCallback()

		loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)

		if self.optimizer == 'not_dp':
			print('check not dp loss')
			loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

		lmodel.compile(
			optimizer=self.select_optimizer(self.optimizer, self.learning_rate*0.1),
			loss=loss,
			metrics=['accuracy'])

		#fit_history = lmodel.fit(gen_train_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=[eps_callback, earlystop_callback])
		#eps_history = eps_callback.eps_history
		lmodel.fit(gtrain_data, batch_size=self.batch_size, epochs=self.epochs+(self.epochs*2),
					initial_epoch=self.epochs, callbacks=[earlystop_callback])

		return lmodel.get_weights()#, eps_history

	def manage_train(self): # now not return eps_history... need for process..
		if self.model_link == 'cnn':
			get_weights = list()
			training_weight = self.train_model_tosave(self.weights)

			return training_weight
		else:
			get_weights = list()
			if self.weights != list():
				training_weight = self.train_model_tosave(self.weights)
	
				return training_weight
			else:
				#lo_model, ba_model, gtrain, ehistory = self.train_model_tosave()
				lo_model, ba_model, gtrain = self.train_model_tosave()
				get_weights = self.get_weight_finetune_model(ba_model, gtrain, lo_model)
	
				return get_weights

