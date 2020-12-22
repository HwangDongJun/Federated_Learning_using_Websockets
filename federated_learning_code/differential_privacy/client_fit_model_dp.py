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

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

from absl import logging
import collections

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		#tf.config.experimental.set_virtual_device_configuration(
			#gpus[0],
			#[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=768)])
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

		tf.random.set_seed(2020)

	# new optimizer
	def make_optimizer_class(self, cls):
		"""Constructs a DP optimizer class from an existing one."""
		parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
		child_code = cls.compute_gradients.__code__
		GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
		if child_code is not parent_code:
			logging.warning(
			'WARNING: Calling make_optimizer_class() on class %s that overrides'
			'method compute_gradients(). Check to ensure that '
			'make_optimizer_class() does not interfere with overridden version.',
			cls.__name__)

		class DPOptimizerClass(cls):
			"""Differentially private subclass of given class cls."""

			_GlobalState = collections.namedtuple(
				'_GlobalState', ['l2_norm_clip', 'stddev'])
	    
			def __init__(
				self,
				dp_sum_query,
				num_microbatches=None,
				unroll_microbatches=False,
				*args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
				**kwargs):
	      
				super(DPOptimizerClass, self).__init__(*args, **kwargs)
				self._dp_sum_query = dp_sum_query
				self._num_microbatches = num_microbatches
				self._global_state = self._dp_sum_query.initial_global_state()
				self._unroll_microbatches = unroll_microbatches

			def compute_gradients(self,
		                  loss,
		                  var_list,
		                  gate_gradients=GATE_OP,
		                  aggregation_method=None,
		                  colocate_gradients_with_ops=False,
		                  grad_loss=None,
		                  gradient_tape=None,
		                  curr_noise_mult=0,
		                  curr_norm_clip=1):

				self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip, 
		                                                   curr_norm_clip*curr_noise_mult)
				self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip, 
		                                                        curr_norm_clip*curr_noise_mult)
	      

				# TF is running in Eager mode, check we received a vanilla tape.
				if not gradient_tape:
					raise ValueError('When in Eager mode, a tape needs to be passed.')

				vector_loss = loss()
				if self._num_microbatches is None:
					self._num_microbatches = tf.shape(input=vector_loss)[0]
				sample_state = self._dp_sum_query.initial_sample_state(var_list)
				microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
				sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

				def process_microbatch(i, sample_state):
					"""Process one microbatch (record) with privacy helper."""
					microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
					grads = gradient_tape.gradient(microbatch_loss, var_list)
					sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
					return sample_state
	    
				for idx in range(self._num_microbatches):
					sample_state = process_microbatch(idx, sample_state)

				if curr_noise_mult > 0:
					grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
				else:
					grad_sums = sample_state

				def normalize(v):
					return v / tf.cast(self._num_microbatches, tf.float32)

				final_grads = tf.nest.map_structure(normalize, grad_sums)
				grads_and_vars = final_grads#list(zip(final_grads, var_list))
	    
				return grads_and_vars

		return DPOptimizerClass

	def make_gaussian_optimizer_class(self, cls):
		"""Constructs a DP optimizer with Gaussian averaging of updates."""

		class DPGaussianOptimizerClass(self.make_optimizer_class(cls)):
			"""DP subclass of given class cls using Gaussian averaging."""

			def __init__(
				self,
				l2_norm_clip,
				noise_multiplier,
				num_microbatches=None,
				ledger=None,
				unroll_microbatches=False,
				*args,  # pylint: disable=keyword-arg-before-vararg
				**kwargs):
				dp_sum_query = gaussian_query.GaussianSumQuery(
				l2_norm_clip, l2_norm_clip * noise_multiplier)

				if ledger:
					dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
		                                              ledger=ledger)

				super(DPGaussianOptimizerClass, self).__init__(
					dp_sum_query,
					num_microbatches,
					unroll_microbatches,
					*args,
					**kwargs)

			@property
			def ledger(self):
				return self._dp_sum_query.ledger

		return DPGaussianOptimizerClass
	# ========================================================================================

	def image_generator(self):
		image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
																	rotation_range=15,
																	horizontal_flip=True,
																	brightness_range=[0.7,1.0])
		image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
		return image_gen_train, image_gen_val

	def gen_train_val_data(self):
		gen_train, gen_val = self.image_generator()

		train_data_dir = os.path.abspath('/home/dnlabblocksci/websocket/3/dataset/data_balanced/new_train/client1/')
		train_data_gen = gen_train.flow_from_directory(directory=str(train_data_dir),
											batch_size=self.batch_size,
											color_mode='rgb',
											shuffle=True,
											target_size=self.image_shape,
											classes=list(self.class_names))
		return train_data_gen

	def select_optimizer(self, opti, lr):
		if opti == 'dp':
			print('dp_optimizer start')
			#return tf.keras.optimizers.Adam(learning_rate=lr)
			GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
			DPGradientDescentGaussianOptimizer_NEW = self.make_gaussian_optimizer_class(GradientDescentOptimizer)

			return DPGradientDescentGaussianOptimizer_NEW(
						l2_norm_clip=self.l2_norm_clip,
						noise_multiplier=self.noise_multiplier,
						num_microbatches=self.num_microbatches,
						learning_rate=lr)

	def set_model(self, vector_layer):
		#model = tf.keras.Sequential([
		#	vector_layer,
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

	def build_model(self, weight=None):
		feature_vector_url = self.model_link
		feature_vector_layer = hub.KerasLayer(feature_vector_url,
										input_shape=self.image_shape+(3,))
		
		if weight == None:
			# train setting false
			feature_vector_layer.trainable = False
		else: # 이전에 학습한 가중치 존재
			feature_vector_layer.trainable = True

		made_model = self.set_model(feature_vector_layer)
		print(made_model.summary())
		#made_model.compile(
			#optimizer=self.select_optimizer(self.optimizer, self.learning_rate),
			#loss='categorical_crossentropy',
			#metrics=['categorical_accuracy'])
			# metrics=['acc'])
		made_model.compile(
			optimizer=self.select_optimizer(self.optimizer, self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		return made_model, feature_vector_layer

	def train_model_tosave(self, weight=None):
		callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

		if weight == None:
			local_model, feature_layer = self.build_model()
			#gen_train_data, gen_val_data = self.gen_train_val_data(1)
			#local_model.fit_generator(gen_train_data, epochs=self.epochs,
					#validation_data=gen_val_data)
			gen_train_data = self.gen_train_val_data()
			local_model.fit_generator(gen_train_data, epochs=self.epochs, callbacks=[callback])

			# Export model
			export_path = './save_model/client1/' 
			local_model.save(export_path, save_format='tf')

			return feature_layer, export_path, gen_train_data
		else:
			local_model, feature_layer = self.build_model(list())
			# 여기서부터는 코드 중복이 많음 -> 데이터를 인위적으로 나눠야 하기에 중복 코드로 둠(조금씩 변경)
			#gen_train_data, gen_val_data = self.gen_train_val_data(2)
			#local_model.fit_generator(gen_train_data, epochs=self.epochs,
					#validation_data=gen_val_data)
			gen_train_data = self.gen_train_val_data()
			local_model.set_weights(weight)
			local_model.fit_generator(gen_train_data, epochs=self.epochs, callbacks=[callback])
			
			return local_model.get_weights()
			

	def get_weight_finetune_model(self, expath, feature_layer, gtrain_data):
		reloaded_model = tf.keras.models.load_model(expath)
		
		feature_layer.trainable = True

		callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

		#reloaded_model.compile(
			#optimizer=self.select_optimizer(self.optimizer, self.learning_rate*0.1),
			#loss='categorical_crossentropy',
			#metrics=['accuracy'])
		reloaded_model.compile(
			optimizer=self.select_optimizer(self.optimizer, self.learning_rate*0.1),
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		#reloaded_model.fit_generator(gtrain_data, epochs=self.epochs+(self.epochs*2),
						#initial_epoch=self.epochs, validation_data=gval_data)
		reloaded_model.fit_generator(gtrain_data, epochs=self.epochs+(self.epochs*2),
						initial_epoch=self.epochs, callbacks=[callback])

		return reloaded_model.get_weights() # Dense layer weight는 제외하고 반환

	def manage_train(self):
		get_weights = list()
		if self.weights != list():
			training_weight = self.train_model_tosave(self.weights)
			
			return training_weight
		else: # 처음 학습이면 일반 방식 그대로 사용
			flayer, epath, gtrain = self.train_model_tosave()
			get_weights = self.get_weight_finetune_model(epath, flayer, gtrain)

			return get_weights
