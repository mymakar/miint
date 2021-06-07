"""Functions for data processing and weighting of MIINT. """

import itertools
import numpy as np
from sklearn.metrics import roc_auc_score

import base_models.dnn_models as base_models


def get_xe(data):
	x, act_mask, edges, yh = data
	yh = yh * act_mask
	assert np.min(yh) >= 0
	E = []
	for i in range(x.shape[1]):
		if i == 0:
			E.append(np.zeros((x.shape[0], 1)))
		else:
			E_curr = np.dot(edges[:, i, :], yh[:, i - 1, :])
			E.append(E_curr)
	E = np.stack(E, axis=1)
	xe = np.concatenate([x, E], axis=2)
	xe = np.float32(xe)
	return xe


def get_weights(wdata, yh, data, yhm, units, alpha, dr_rate, batch_size,
	epochs):
	tr_data, val_data = wdata
	yh_tr, yh_val = yh

	mtr_data, mval_data = data
	yh_mtr, yh_mval = yhm

	# get the xe for training/validation
	xe_tr = get_xe((tr_data[0], tr_data[1], tr_data[2], yh_tr))
	xe_val = get_xe((val_data[0], val_data[1], val_data[2], yh_val))

	# get the testing label for the train/validation data
	tr_y_obs = tr_data[-1]
	if isinstance(tr_y_obs, np.ndarray):
		tr_tested_label = np.where(tr_y_obs == -1, 0, 1)
	else:
		tr_tested_label = np.where(tr_y_obs[()] == -1, 0, 1)

	val_y_obs = val_data[-1]
	if isinstance(val_y_obs, np.ndarray):
		val_tested_label = np.where(val_y_obs == -1, 0, 1)
	else:
		val_tested_label = np.where(val_y_obs[()] == -1, 0, 1)

	wtr_data = (xe_tr, tr_data[1], tr_data[2], tr_tested_label)
	wval_data = (xe_val, val_data[1], val_data[2], val_tested_label)

	# get the xe for prediction (i.e., main data)
	xe_mtr = get_xe((mtr_data[0], mtr_data[1], mtr_data[2], yh_mtr))
	xe_mval = get_xe((mval_data[0], mval_data[1], mval_data[2], yh_mval))

	# get the testing label for the main data
	mtr_y_obs = mtr_data[-1]
	if isinstance(mtr_y_obs, np.ndarray):
		mtr_tested_label = np.where(mtr_y_obs == -1, 0, 1)
	else:
		mtr_tested_label = np.where(mtr_y_obs[()] == -1, 0, 1)

	mval_y_obs = mval_data[-1]
	if isinstance(mval_y_obs, np.ndarray):
		mval_tested_label = np.where(mval_y_obs == -1, 0, 1)
	else:
		mval_tested_label = np.where(mval_y_obs[()] == -1, 0, 1)

	mtr_data = (xe_mtr, mtr_data[1], mtr_data[2], mtr_tested_label)
	mval_data = (xe_mval, mval_data[1], mval_data[2], mval_tested_label)

	T = xe_tr.shape[1]
	D = xe_tr.shape[2]

	M = base_models.BM(T=T, D=D, units=units,
			exp_func=None, alpha=alpha, dr_rate=dr_rate,
			log_perf=False)

	M.fit(wtr_data, wval_data, batch_size, epochs,
		gt_data=None, sample_weights=None, plot_fname=None)

	mtr_data_pp = M.data_generator(base_models.preprocess_data(mtr_data, None),
		np.ones_like(mtr_data[-1]))
	mval_data_pp = M.data_generator(base_models.preprocess_data(mval_data, None),
		np.ones_like(mval_data[-1]))

	temp = M.predict(**mval_data_pp['in'])
	weights_auc = roc_auc_score(mval_data[-1].reshape(mval_data[-1].size),
		temp.reshape(temp.size))
	print(f'weight auc: {weights_auc:.3f}')

	sw_tr = 1 / M.predict(**mtr_data_pp['in'])
	sw_val = 1 / M.predict(**mval_data_pp['in'])
	return sw_tr, sw_val


def preprocess_data(data):
	'''
	Args:
		data: needs to have
			x: input chars over time
			act_mask: tensor of 1/0 active or not at time point t
			edges: tensor of 1/0 contact network over time
			y_obs: 1 if infected, 0 if not, -1 if not observed
	'''
	x, act_mask, edges, y_obs = data
	if isinstance(y_obs, np.ndarray):
		obs_mask = np.where(y_obs == -1, 0, 1)
	else:
		obs_mask = np.where(y_obs[()] == -1, 0, 1)

	# make sure when inactive, label is marked as unobserved
	if isinstance(act_mask, np.ndarray):
		act_unobs_mask = (1 - act_mask) * obs_mask
	else:
		act_unobs_mask = (1 - act_mask[()]) * obs_mask
	assert np.sum(act_unobs_mask) == np.max(act_unobs_mask)
	assert np.max(act_unobs_mask) == np.min(act_unobs_mask)
	assert np.min(act_unobs_mask) == 0

	return x, act_mask, edges, y_obs, obs_mask


def get_active_mask(act_mask_or, edges, inds):
	# set the active indiviudals (i.e., the inds which\
	#  will contribute to loss) to be act + nbrs + nbrs' nbrs, etc
	# act_curr = act_mask_or[:, -1, :].copy()
	# those who are not in the current inds are inactive
	# act_curr[[i for i in range(act_curr.shape[0]) if i not in inds], :] = 0

	act_mask = []
	in_samp_mask = []
	inds_all = [inds]

	for i in range(edges.shape[1] - 1, 0, -1):
		# nbrs who contribute to predictions at time i
		act_nbrs = np.max(edges[inds_all[0], i, :], axis =0)
		act_nbrs = np.argwhere(act_nbrs > 0).ravel()
		act_nbrs = sorted(list(set(act_nbrs).union(set(inds))))
		act_nbrs = sorted(list(set(act_nbrs).union(set(inds_all[0]))))

		inds_all.insert(0, act_nbrs)

		# get wether or not they are active at the time when they contribute

	for i in range(edges.shape[1]):
		act_curr = act_mask_or[:, i, :].copy()
		act_curr[[j for j in range(act_curr.shape[0]) if j not in inds_all[i]], :] = 0
		act_mask.append(act_curr)

		in_samp_curr = act_mask_or[:, i, :].copy()
		in_samp_curr[[j for j in range(in_samp_curr.shape[0]) if j not in inds], :] = 0
		in_samp_mask.append(in_samp_curr)

	act_mask = np.concatenate(act_mask, axis=1)
	act_mask = act_mask[:, :, np.newaxis]

	in_samp_mask = np.concatenate(in_samp_mask, axis=1)
	in_samp_mask = in_samp_mask[:, :, np.newaxis]

	inds_flat = sorted(list(set(itertools.chain.from_iterable(inds_all))))

	return act_mask, in_samp_mask, inds_flat

