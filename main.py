""" Class for the main method (MIINT). """

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from sklearn.metrics import accuracy_score, roc_auc_score
import miint.data_processing as data_lib

class netRNN(K.layers.Layer):
	'''
	Class for a network RNN
	'''

	def __init__(self, T, D, units, alpha, dr_rate):
		self.T = T
		self.D = D
		self.units = units
		self.alpha = alpha
		self.dr_rate = dr_rate
		self.state_size = 1
		self.output_size = 1


		super(netRNN, self).__init__()

	def build(self, input_shape):
		"""Build function for the neural network."""

		self.kernel_0 = self.add_weight(shape=(self.D, self.units),
			regularizer=K.regularizers.l2(self.alpha),
			name='kernel_0')

		self.kernel_1 = self.add_weight(shape=(self.units + 2, self.units),
			regularizer=K.regularizers.l2(self.alpha),
			name='kernel_1')

		self.kernel_3 = self.add_weight(shape=(self.units, 1),
			regularizer=K.regularizers.l2(self.alpha),
			name='kernel_3')

		self.bias_0 = self.add_weight(shape=(1, 1), name='bias_0')

		self.bias_1 = self.add_weight(shape=(1, 1), name='bias_1')

		self.bias_3 = self.add_weight(shape=(1, 1), name='bias_3')

		self.built = True

	@tf.function
	def call(self, inputs, states):
		'''
		Output is y_(t+1)
		'''
		x_t, edges_t, active_mask_t = tf.nest.flatten(inputs)

		# get exposure using prev states
		previous_states = states[0]

		# get total exposure
		total_exposure_prev = K.backend.dot(edges_t, previous_states)
		h_t = K.backend.dot(x_t, self.kernel_0) + self.bias_0

		# concat input representation with previous state and exposure
		ha_t = tf.concat([h_t, total_exposure_prev, previous_states], axis=1)
		ha_t = K.backend.dot(ha_t, self.kernel_1) + self.bias_1
		ha_t = tf.nn.tanh(ha_t)
		ha_t = K.backend.dot(ha_t, self.kernel_3) + self.bias_3
		output = tf.math.sigmoid(ha_t)

		# mask out inactive
		output = tf.multiply(output, active_mask_t)
		return output, [output]


def gen_splits(batch_size, N):
	'''
	generates new splits for each epoch
	'''
	if batch_size is None:
		splits = range(N)
	else:
		mls = list(range(N))
		np.random.shuffle(mls)

		splits = [mls[i:i + batch_size] for i in range(0, len(mls), batch_size)]
		splits = [sorted(split) for split in splits]
		assert sum([len(splits[i]) for i in range(len(splits))]) == N
	return splits


def data_generator(data, sample_weights, inds=None, fast=False, random=False,
	prop=None):
	'''
	spits out the training data for a set of inds (batch indicators)
	'''
	x, active_mask, edges, y_obs, obs_mask = data

	if inds is not None:
		if fast:
			active_mask_fin, in_samp_mask, inds_fin = data_lib.get_active_mask_fast(
				active_mask, edges, inds)
		elif random:
			active_mask_fin, in_samp_mask, inds_fin = data_lib.get_active_mask_random(
				active_mask, edges, inds, prop)
		else:
			active_mask_fin, in_samp_mask, inds_fin = data_lib.get_active_mask(
				active_mask, edges, inds)
	else:
		active_mask_fin = active_mask.copy()
		in_samp_mask = np.ones_like(active_mask_fin)
		inds_fin = range(x.shape[0])

	sub_edges = edges[inds_fin, :, :][:, :, inds_fin].copy()

	in_dict = {
		'X': np.float32(x[inds_fin, :, :]),
		'active_mask': np.float32(active_mask_fin[inds_fin, :, :]),
		'edges': np.float32(sub_edges),
		'obs_mask': np.float32(obs_mask[inds_fin, :, :]),
		'in_sample_mask': np.float32(in_samp_mask[inds_fin, :, :]),
		'sw': np.float32(sample_weights[inds_fin, :, :])
	}
	out_dict = {'y': np.float32(y_obs[inds_fin, :, :])}
	return {'in': in_dict, 'out': out_dict}


def test_data_generator(data, inds, fast=False, random=False, prop=.5):
	x, active_mask, edges = data
	if inds is not None:
		if fast:
			active_mask_fin, in_samp_mask, inds_fin = data_lib.get_active_mask_fast(
				active_mask, edges, inds)
		elif random:
			active_mask_fin, in_samp_mask, inds_fin = data_lib.get_active_mask_random(
				active_mask, edges, inds, prop)
		else:
			active_mask_fin, in_samp_mask, inds_fin = data_lib.get_active_mask(
				active_mask, edges, inds)
	else:
		active_mask_fin = active_mask.copy()
		inds_fin = range(x.shape[0])

	sub_edges = edges[inds_fin, :, :][:, :, inds_fin].copy()

	data_dict = {
		'X': np.float32(x[inds_fin, :, :]),
		'active_mask': np.float32(active_mask_fin[inds_fin, :, :]),
		'in_samp_mask': np.float32(in_samp_mask[inds_fin, :, :]),
		'edges': np.float32(sub_edges)
	}
	in_sample_position_mask = [ind in inds for ind in inds_fin]
	in_sample_position_ind = np.where(in_sample_position_mask)[0]
	return data_dict, list(in_sample_position_ind)


def create_arch(T, D, units=64, alpha=0, dr_rate=.3):
	"""Creates the architecture of miint"""

	X = K.Input(shape=(T, D))
	active_mask = K.Input(shape=(T, 1))
	edges = K.Input(shape=(T, None))
	ycell = netRNN(T=T, D=D, units=units, alpha=alpha, dr_rate=dr_rate)

	yrnn = K.layers.RNN(ycell, return_sequences=True)
	Y = yrnn((X, edges, active_mask))

	return K.Model(inputs=[X, active_mask, edges], outputs=Y)



class MIINT():
	"""docstring for MIINT"""
	def __init__(self, T, D, thresh=.5, units=64, alpha=0, dr_rate=.3,
		tf_param=1.0, lr=0.001, log_perf=True, minimum_delta=1e-10, patience=5):

		super(MIINT, self).__init__()
		# model params
		self.T = T
		self.D = D
		self.units = units
		self.alpha = alpha
		self.dr_rate = dr_rate

		# early stopping params
		self.minimum_delta = minimum_delta
		self.patience = patience

		# value of 'best' loss value-
		self.best_val_loss = 1e2
		self.loc_patience = 0

		self.lr = lr
		self.tf_param = tf_param
		self.thresh = thresh
		self.log_perf = log_perf

		# init model
		self.model = create_arch(T=self.T, D=self.D, units=self.units,
			dr_rate=self.dr_rate, alpha=self.alpha)

		self.optimizer = K.optimizers.Adam(learning_rate=self.lr)

		self.history = {'tr_loss': [], 'val_loss': [],
			'tr_lloss': [], 'val_lloss': [], 'tr_auc': [], 'val_auc': []}

	def predict(self, X, active_mask, edges, **unused_params):
		"""Computes predictions."""
		return self.model([X, active_mask, edges]).numpy()

	def predict_batch(self, data, batch_size=100, fast=False):
		"""Computes predictions."""
		ind_list = list(range(data[0].shape[0]))
		prediction_splits = [ind_list[i:i + batch_size]
			for i in range(0, len(ind_list), batch_size)]
		predictions = []
		for split in prediction_splits:
			prediction_data, in_sample_index = test_data_generator(data, split,
				fast=fast)
			sample_and_neighbor_predictions = self.predict(**prediction_data)
			sample_predictions = sample_and_neighbor_predictions[in_sample_index]
			predictions.append(sample_predictions)

		predictions = np.concatenate(predictions, axis=0)
		return predictions

	def loss_fn(self, y_mix, yh, obs_mask, active_mask, sw, in_sample_mask=None,
		**unused_kwargs):
		'''
		Computes cross entropy for labelled/unlabelled
		'''

		if len(obs_mask.shape) == 3:
			obs_mask = np.squeeze(obs_mask)

		if len(active_mask.shape) == 3:
			active_mask = np.squeeze(active_mask)

		if len(sw.shape) == 3:
			sw = np.squeeze(sw)

		if (in_sample_mask is not None) and len(in_sample_mask.shape) == 3:
			in_sample_mask = np.squeeze(in_sample_mask)

		elif in_sample_mask is None:
			in_sample_mask = np.ones_like(active_mask)

		active_mask_in_sample = active_mask * in_sample_mask
		li = tf.keras.losses.binary_crossentropy(y_mix, yh)
		li = tf.multiply(li, active_mask_in_sample)

		# make sure that loss is 0 for inactive
		assert np.max(li.numpy() * (1 - active_mask_in_sample)) == 0


		obs_act_flag = np.where(((active_mask_in_sample == 1) & (obs_mask == 1)), 1,
			np.where(((active_mask_in_sample == 1) & (obs_mask == 0)), 0, np.nan))

		numerator = tf.reduce_sum(sw * tf.multiply(li, obs_mask))
		denominator = np.nansum(sw * obs_act_flag)
		ll = tf.math.divide_no_nan(numerator, denominator)

		numerator = tf.reduce_sum(sw * tf.multiply(li, 1 - obs_mask))
		denominator = np.nansum(sw * (1 - obs_act_flag))
		lu = tf.math.divide_no_nan(numerator, denominator)

		tl = ll + self.tf_param * lu
		return tl, ll, lu

	def train_step(self, data):
		'''
		runs 1 step in an epoch (computes grads and updates params)
		'''
		yhb = (self.predict(**data['in']) > self.thresh) * 1

		# compute the "mixed" label
		y_mix = data['in']['obs_mask'] * data['out']['y']
		y_mix = y_mix + (1 - data['in']['obs_mask']) * yhb

		assert np.min(y_mix) >= 0
		active_mask = data['in']['active_mask']

		with tf.GradientTape() as tape:
			# compute loss
			yh = self.model([data['in']['X'], data['in']['active_mask'],
				data['in']['edges']])

			assert np.max(yh.numpy()) <= 1

			step_loss, _, _ = self.loss_fn(y_mix, yh,
				data['in']['obs_mask'], active_mask, data['in']['sw'],
				data['in']['in_sample_mask'])
			step_loss += sum(self.model.losses)

		gradients = tape.gradient(step_loss, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))


	def update_logger(self, data, tr=True, gt=None):
		'''
		computes evaluation metrics and updates the logger
		'''
		tgrp = "tr" if tr else "val"
		# get the predicted values
		yh = self.predict(**data['in'])
		yhb = (yh > self.thresh) * 1
		# get mixed label
		y_mix = data['in']['obs_mask'] * data['out']['y']
		y_mix = y_mix + (1 - data['in']['obs_mask']) * yhb
		assert np.min(y_mix) >= 0

		# get the losses
		loss, lloss, uloss = self.loss_fn(y_mix, yh, **data['in'])
		self.history[f'{tgrp}_loss'].append(loss)

		obs_act_flag = np.where(
			((data['in']['active_mask'] == 1) & (data['in']['obs_mask'] == 1)), 1,
			np.where(
				((data['in']['active_mask'] == 1) & (data['in']['obs_mask'] == 0)), 0,
				np.nan))

		# get the auc
		temp_y_mix = y_mix.reshape(y_mix.size)
		temp_yh = yh.reshape(yh.size)
		temp_sw = data['in']['sw'] * data['in']['active_mask']
		temp_sw = temp_sw.reshape(temp_sw.size)
		temp_obs_mask = data['in']['obs_mask'].reshape(data['in']['obs_mask'].size)

		temp_y_mix = temp_y_mix[temp_obs_mask == 1]
		temp_yh = temp_yh[temp_obs_mask == 1]
		temp_sw = temp_sw[temp_obs_mask == 1]

		self.history[f'{tgrp}_auc'].append(roc_auc_score(temp_y_mix, temp_yh,
			sample_weight=temp_sw))
		self.history[f'{tgrp}_lloss'].append(lloss)



	def fit(self, data_or, val_data_or, wdata_or, wval_data_or, batch_size,
		epochs, fast, sample_weights=None, preprocess=True):
		'''
		Main fitting function
		'''
		print("--------PREPROCESSING-----------")
		if preprocess:
			data = data_lib.preprocess_data(data_or)
			val_data = data_lib.preprocess_data(val_data_or)

		if sample_weights is None:
			sw_tr = np.ones((data[0].shape[0], data[0].shape[1], 1))
			sw_val = np.ones((val_data[0].shape[0], val_data[0].shape[1], 1))
		else:
			sw_tr, sw_val = sample_weights

		print("--------EVAL DATA-----------")
		# get evaluation data (train and validation)

		tr_data_all = data_generator(data, sw_tr, inds=None, fast=fast,
				random=False)

		val_data_all = data_generator(val_data, sw_val, inds=None,
				fast=fast, random=False)

		print("--------TRAINING-----------")
		# train/eval loop
		for epoch in range(epochs):
			if (self.loc_patience >= self.patience):
				# early stop if there is no improvement for a number of epochs
				break

			# train
			print('Training...')
			splits = gen_splits(batch_size, data[0].shape[0])
			for sid, split in enumerate(splits):
				print(f"----TP Epoch {epoch}, Step {sid}/{len(splits)}------")
				# get batch data

				tr_data = data_generator(data, sw_tr, split, fast=fast)
				# update gradients
				self.train_step(tr_data)
			# evaluate
			print('Evaluating...')
			self.update_logger(val_data_all, tr=False)


			# Early stopping:

			if ((self.best_val_loss - self.history['val_loss'][-1]) < self.minimum_delta):
				self.loc_patience += 1

			else:
				# update 'best_val_loss' variable to lowest loss encountered so far-
				self.best_val_loss = self.history['val_loss'][-1]
				# reset 'loc_patience' variable-
				self.loc_patience = 0

			if wdata_or is not None:
				# get current predictions
				yh_tr = self.predict_batch((wdata[0], wdata[1], wdata[2]),
					batch_size=1000)

				yh_val = self.predict_batch((wval_data[0], wval_data[1], wval_data[2]),
					batch_size=1000)

				yh_mtr = self.predict_batch((data[0], data[1], data[2]),
					batch_size=1000)

				yh_mval = self.predict_batch((val_data[0], val_data[1], val_data[2]),
					batch_size=1000)

				sw_tr, sw_val = data_lib.get_weights(
					(wdata_or, wval_data_or), (yh_tr, yh_val),
					(data_or, val_data_or), (yh_mtr, yh_mval),
					units=self.units,
					alpha=self.alpha, dr_rate=self.dr_rate,
					batch_size=batch_size, epochs=epochs)

				val_data_all = data_generator(val_data, sw_val)
