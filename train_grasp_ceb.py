import math
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import numpy as np
import torch
import pandas as pd

import random

from CEB_utlities.join_utilits_CEB_job import *
from GRASP.join_handler_ceb import *
from utils import *
from torch.nn.utils import clip_grad_norm_

OPS = ['lt', 'eq', 'in', 'like']


random.seed(42)

is_cuda = torch.cuda.is_available()

IMDB_DIRECTORY = "./queries/ceb-imdb-full/"	# directory to save the IMDB dataset
PROCESSED_WORKLOAD_DIRECTORY = "./processed_workloads/imdb/" # directory to save processed workloads

def train_grasp(epoch=100, feature_dim=256, lcs_dim=500, bs=128, lr=5e-4, residual=0):
	start_time = time.time()
	num_q = 3000
	cdf_model_choice = 'arcdf'
	
	# Initialize tracking variables
	train_losses = []  # SLE loss
	train_qerrors = []  # Q-error during training
	train_under_ratios = []  # Ratio of underestimations
	train_over_ratios = []   # Ratio of overestimations
	seen_test_losses = []
	unseen_test_losses = []
	# For early stopping
	# best_seen_qerror = float('inf')
	# best_model_state = None
	# epochs_without_improvement = 0


	template_list =  ['1a', '2a', '2b', '2c', '3a', '3b', '4a', '5a', '6a', 
					'7a', '8a', '9a', '9b', '10a', '11a', '11b']


	directory_list = [IMDB_DIRECTORY + temp_name + "/" for temp_name in template_list]


	res_file = open("./grasp_{}_ratio_{}_lcssize_{}_lr_{}_bs_{}.txt".format(",".join(template_list), sub_templates_in_training_ratio, lcs_dim, lr, bs), 'a')


	(table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax) = get_table_info()

	template2queries, template2cards, template2pgcards, test_template2queries, test_template2cards, test_template2pgcards , colid2featlen_per_table = read_query_file_batched(col2minmax, num_q=num_q, directory_list=directory_list
																																	,saved_ditectory=PROCESSED_WORKLOAD_DIRECTORY)
	t2ops = get_table_to_ops(template2queries, test_template2queries)

	print(t2ops)
	for t in t2ops:
		print(t)
		print(t2ops[t])

	join_handler = JoinHandler(table_list=table_list, table_dim_list=table_dim_list, table_key_groups=table_key_groups,
							table_size_list=table_sizes, cdf_model_choice=cdf_model_choice, hidden_size=feature_dim, lcs_size=lcs_dim, colid2featlen_per_table=colid2featlen_per_table)

	join_handler.init_ce_predictors(colid2featlen_per_table)
	join_handler.init_unified_lcs_predictors()
	join_handler.models_to_double()
	if is_cuda:
		join_handler.load_models_to_gpu()
	optimizer = torch.optim.Adam(join_handler.get_parameters(), lr=lr)

	training_loader_per_parent_template = {}
	in_test_loader_per_parent_template = {}
	ood_test_loader_per_parent_template = {}
	pg_q_errors = {'seen':[], 'unseen':[]}
	
	# For tracking query IDs in test sets
	seen_query_ids = []
	unseen_query_ids = []

	num_qs = 0
	test_unseen_num_qs = 0
	test_seen_num_qs = 0
	num_seen_templates = 0

	### load seen join templates
	for template in template2queries:
		num_seen_templates += 1
		training_queries = template2queries[template]
		all_training_cards = template2cards[template]
		training_pgcards = template2pgcards[template]

		if len(all_training_cards) > 100:
			num_train_per_temp = len(template2queries[template]) - 100
			test_seen_num_qs += 100

			num_qs += num_train_per_temp
			# for training
			for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(training_queries[:num_train_per_temp]):
				parent_t_list = tuple(table_tuple)

				if parent_t_list not in training_loader_per_parent_template:
					training_loader_per_parent_template[parent_t_list] = {}
					training_loader_per_parent_template[parent_t_list]['qs'] = []
					training_loader_per_parent_template[parent_t_list]['q_contexts'] = []
					training_loader_per_parent_template[parent_t_list]['qreps'] = []
					training_loader_per_parent_template[parent_t_list]['table_masks'] = []
					training_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
					training_loader_per_parent_template[parent_t_list]['is_stb'] = []
					training_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
					training_loader_per_parent_template[parent_t_list]['key_groups'] = []
					training_loader_per_parent_template[parent_t_list]['cards'] = []
				
				training_loader_per_parent_template[parent_t_list]['qs'].append(q)
				training_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
				training_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
				training_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
				training_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
				if len(list(table_tuple)) > 1:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
				else:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
				training_loader_per_parent_template[parent_t_list]['parent_t_list'].append(table_tuple)
				training_loader_per_parent_template[parent_t_list]['key_groups'].append(key_groups_per_q)
				if residual:
					res_card = all_training_cards[qid] / training_pgcards[qid]
					training_loader_per_parent_template[parent_t_list]['cards'].append(res_card)
				else:
					training_loader_per_parent_template[parent_t_list]['cards'].append(all_training_cards[qid])

			# seen test queries
			for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(training_queries[-100:]):
				parent_t_list = tuple(parent_table_tuple)
				if parent_t_list not in in_test_loader_per_parent_template:
					in_test_loader_per_parent_template[parent_t_list] = {}
					in_test_loader_per_parent_template[parent_t_list]['qs'] = []
					in_test_loader_per_parent_template[parent_t_list]['q_contexts'] = []
					in_test_loader_per_parent_template[parent_t_list]['qreps'] = []
					in_test_loader_per_parent_template[parent_t_list]['table_masks'] = []
					in_test_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
					in_test_loader_per_parent_template[parent_t_list]['is_stb'] = []
					in_test_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
					in_test_loader_per_parent_template[parent_t_list]['key_groups'] = []
					in_test_loader_per_parent_template[parent_t_list]['cards'] = []
					in_test_loader_per_parent_template[parent_t_list]['pg_cards'] = []
					in_test_loader_per_parent_template[parent_t_list]['query_ids'] = []

				
				in_test_loader_per_parent_template[parent_t_list]['qs'].append(q)
				in_test_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
				in_test_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
				in_test_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
				in_test_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
				if len(list(table_tuple)) > 1:
					in_test_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
				else:
					in_test_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
				in_test_loader_per_parent_template[parent_t_list]['parent_t_list'].append(parent_table_tuple)
				in_test_loader_per_parent_template[parent_t_list]['key_groups'].append(parent_key_groups)
				if residual:
					res_card = all_training_cards[num_train_per_temp+qid] / training_pgcards[num_train_per_temp+qid]
					in_test_loader_per_parent_template[parent_t_list]['cards'].append(res_card)
				else: 
					in_test_loader_per_parent_template[parent_t_list]['cards'].append(all_training_cards[num_train_per_temp+qid])
				in_test_loader_per_parent_template[parent_t_list]['pg_cards'].append(training_pgcards[num_train_per_temp+qid])
				in_test_loader_per_parent_template[parent_t_list]['query_ids'].append((template, num_train_per_temp+qid))
			
				actual = all_training_cards[num_train_per_temp+qid]
				pg = training_pgcards[num_train_per_temp+qid]
				pg_q_errors['seen'].append(max(pg/actual, actual/pg))
				seen_query_ids.append((template, num_train_per_temp+qid))

		else: # use all for training
			num_qs += len(training_queries)
			for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(training_queries):
				
				parent_t_list = tuple(table_tuple)

				if parent_t_list not in training_loader_per_parent_template:
					training_loader_per_parent_template[parent_t_list] = {}
					training_loader_per_parent_template[parent_t_list]['qs'] = []
					training_loader_per_parent_template[parent_t_list]['q_contexts'] = []
					training_loader_per_parent_template[parent_t_list]['qreps'] = []
					training_loader_per_parent_template[parent_t_list]['table_masks'] = []
					training_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
					training_loader_per_parent_template[parent_t_list]['is_stb'] = []
					training_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
					training_loader_per_parent_template[parent_t_list]['key_groups'] = []
					training_loader_per_parent_template[parent_t_list]['cards'] = []
				
				training_loader_per_parent_template[parent_t_list]['qs'].append(q)
				training_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
				training_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
				training_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
				training_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
				if len(list(table_tuple)) > 1:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
				else:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
				training_loader_per_parent_template[parent_t_list]['parent_t_list'].append(parent_table_tuple)
				training_loader_per_parent_template[parent_t_list]['key_groups'].append(key_groups_per_q)
				if residual:
					res_card = all_training_cards[qid] / training_pgcards[qid]
					training_loader_per_parent_template[parent_t_list]['cards'].append(res_card)
				else:
					training_loader_per_parent_template[parent_t_list]['cards'].append(all_training_cards[qid])

	### load unseen queries
	for template in test_template2queries:
		if len(test_template2queries[template]) < 10:
			continue

		test_unseen_num_qs += 100 #len(test_template2queries[template]) #NOTE: i think this is wrong and should be 100

		test_queries = test_template2queries[template]
		test_cards = test_template2cards[template]

		test_queries = list(test_queries)[:100]
		test_cards = list(test_cards)[:100]
		#pg test cards
		pg_test_cards = list(test_template2pgcards[template])[:100]

		for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(test_queries):
			parent_t_list = tuple(parent_table_tuple)

			if parent_t_list not in ood_test_loader_per_parent_template:
				ood_test_loader_per_parent_template[parent_t_list] = {}
				ood_test_loader_per_parent_template[parent_t_list]['qs'] = []
				ood_test_loader_per_parent_template[parent_t_list]['q_contexts'] = []
				ood_test_loader_per_parent_template[parent_t_list]['qreps'] = []
				ood_test_loader_per_parent_template[parent_t_list]['table_masks'] = []
				ood_test_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
				ood_test_loader_per_parent_template[parent_t_list]['is_stb'] = []
				ood_test_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
				ood_test_loader_per_parent_template[parent_t_list]['key_groups'] = []
				ood_test_loader_per_parent_template[parent_t_list]['cards'] = []
				ood_test_loader_per_parent_template[parent_t_list]['pg_cards'] = []
				ood_test_loader_per_parent_template[parent_t_list]['query_ids'] = []
			
			ood_test_loader_per_parent_template[parent_t_list]['qs'].append(q)
			ood_test_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
			ood_test_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
			ood_test_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
			ood_test_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
			if len(list(table_tuple)) > 1:
				ood_test_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
			else:
				ood_test_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
			ood_test_loader_per_parent_template[parent_t_list]['parent_t_list'].append(parent_table_tuple)
			ood_test_loader_per_parent_template[parent_t_list]['key_groups'].append(parent_key_groups)
			if residual:
				res_card = test_cards[qid] / pg_test_cards[qid]
				ood_test_loader_per_parent_template[parent_t_list]['cards'].append(res_card)
			else:
				ood_test_loader_per_parent_template[parent_t_list]['cards'].append(test_cards[qid])
			ood_test_loader_per_parent_template[parent_t_list]['pg_cards'].append(pg_test_cards[qid])
			ood_test_loader_per_parent_template[parent_t_list]['query_ids'].append((template, qid))
			pg = pg_test_cards[qid]
			actual = test_cards[qid]
			pg_q_errors['unseen'].append(max(pg/actual, actual/pg))
			unseen_query_ids.append((template, qid))

	num_batches = math.ceil(num_qs / bs)

	print("total number of train queries: {}".format(num_qs))
	print("total number of seen join templates: {}".format(num_seen_templates))
	print("total number of seen test queries: {}".format(test_seen_num_qs))
	print("total number of unseen test queries: {}".format(test_unseen_num_qs))

	res_file.write("total number of queries: {} \n".format(num_qs))
	res_file.write("total number of seen join templates: {}".format(num_seen_templates))
	res_file.write("total number of seen test queries: {} \n".format(test_seen_num_qs))	
	res_file.write("total number of unseen test queries: {} \n".format(test_unseen_num_qs))

	#### start loading data for pytorch training

	data_loader_list = []
	data_loader_iter_list = []
	train_key_groups_list = []
	train_tables_list = []

	in_test_data_loader_list = []
	in_test_parent_alias_list = []
	in_test_training_keys_list = []
	in_test_tables_list = []

	ood_test_data_loader_list = []
	ood_test_parent_alias_list = []
	ood_test_training_keys_list = []
	ood_test_tables_list = []

	for t_list in training_loader_per_parent_template:
		tmp_object = training_loader_per_parent_template[t_list]
		table_data_loader, table_list =  join_handler.load_training_queries(tmp_object['qs'], 
																			tmp_object['q_contexts'],
																			tmp_object['qreps'], 
																			tmp_object['cards'], 
																			bs, t_list,  is_cuda=is_cuda)
		
		data_loader_iterator = iter(table_data_loader)

		data_loader_list.append(table_data_loader)
		data_loader_iter_list.append(data_loader_iterator)
		train_key_groups_list.append(tmp_object['key_groups'][0])
		train_tables_list.append(t_list)

	for parent_t_list in in_test_loader_per_parent_template:
		tmp_object = in_test_loader_per_parent_template[parent_t_list]

		table_data_loader, parent_alias_list, training_keys, training_tables = join_handler.load_training_queries_w_masks(tmp_object['qs'], 
																													tmp_object['q_contexts'],
																													tmp_object['qreps'], 
																													tmp_object['key_groups'][0], 
																													tmp_object['table_masks'],
																													tmp_object['table_group_masks'], 
																													tmp_object['is_stb'], 
																													tmp_object['cards'], 
																													3000, is_cuda=is_cuda, is_shuffle=False)


		in_test_data_loader_list.append(table_data_loader)
		in_test_parent_alias_list.append(parent_alias_list)
		in_test_training_keys_list.append(training_keys)
		in_test_tables_list.append(training_tables)

	for parent_t_list in ood_test_loader_per_parent_template:
		tmp_object = ood_test_loader_per_parent_template[parent_t_list]
		#NOTE: is batchsize 3000 ok here?
		table_data_loader, parent_alias_list, training_keys, training_tables = join_handler.load_training_queries_w_masks(tmp_object['qs'], 
																													tmp_object['q_contexts'],
																													tmp_object['qreps'], 
																													tmp_object['key_groups'][0], 
																													tmp_object['table_masks'],
																													tmp_object['table_group_masks'], 
																													tmp_object['is_stb'], 
																													tmp_object['cards'], 
																													3000, is_cuda=is_cuda, is_shuffle=False)


		ood_test_data_loader_list.append(table_data_loader)
		ood_test_parent_alias_list.append(parent_alias_list)
		ood_test_training_keys_list.append(training_keys)
		ood_test_tables_list.append(training_tables)

	############ finish data loading

	progress_bar = tqdm(range(epoch), desc="Training progress")
	for epoch_id in progress_bar:
		epoch_start_time = time.time()
		join_handler.start_train()

		# Reset epoch statistics
		accu_loss_total = 0.
		epoch_qerrors = []
		epoch_under_count = 0
		epoch_over_count = 0
		epoch_total_preds = 0
		
		current_data_loader_ids = list(range(len(data_loader_iter_list)))
		while len(current_data_loader_ids) > 0:
			d_loader_id = random.choice(current_data_loader_ids)

			table_data_loader = data_loader_iter_list[d_loader_id]
			table_list = train_tables_list[d_loader_id]
			key_groups = train_key_groups_list[d_loader_id]

			try:
				# Fetch the next batch
				batch = next(table_data_loader)
				keys_order, tables_order = join_handler.template_to_group_order(key_groups)
				est_cards = join_handler.batch_estimate_join_queries_from_loader(batch, table_list,
																				keys_order, tables_order, is_cuda=is_cuda)
				if not est_cards.requires_grad:
					continue
				batch_cards = batch[-1]
				batch_cards = batch_cards.to(torch.float64)
				optimizer.zero_grad()

				adj_est_cards = torch.where(est_cards > 0, est_cards, 1.)

				sle_loss = torch.square(torch.log(torch.squeeze(adj_est_cards)) - torch.log(batch_cards))
				se_loss = torch.square(torch.squeeze(est_cards) - batch_cards)

				ultimate_loss = torch.where(torch.squeeze(est_cards) > 0, sle_loss, se_loss)
				total_loss = torch.mean(ultimate_loss)
				accu_loss_total += total_loss.item()

				# Calculate Q-error for this batch
				est_np = torch.squeeze(est_cards).detach().cpu().numpy()
				true_np = batch_cards.detach().cpu().numpy()
				
				# Ensure arrays are properly shaped
				if np.isscalar(est_np):
					est_np = np.array([est_np])
					true_np = np.array([true_np])
					
				q_errors = np.maximum(est_np/true_np, true_np/est_np)
				batch_qerror = np.median(q_errors)
				
				# Update epoch statistics
				epoch_qerrors.extend(q_errors.flatten())
				epoch_under_count += int(np.sum(est_np < true_np))
				epoch_over_count += int(np.sum(est_np > true_np))
				epoch_total_preds += est_np.size
				
				with torch.autograd.set_detect_anomaly(True):
					total_loss.backward()
					clip_grad_norm_(join_handler.get_parameters(), 10.)
					optimizer.step()	
					
				# Track batch-level metrics for plotting
				if not math.isnan(batch_qerror):
					train_qerrors.append(batch_qerror)
					train_under_ratios.append(epoch_under_count/epoch_total_preds)
					train_over_ratios.append(epoch_over_count/epoch_total_preds)

			except StopIteration:
				# When the iterator is exhausted, we get here
				current_data_loader_ids.remove(d_loader_id)
				data_loader_iter_list[d_loader_id] = iter(data_loader_list[d_loader_id])

		epoch_time = time.time() - epoch_start_time
		avg_loss = accu_loss_total / num_batches
		train_losses.append(avg_loss)
		
		# Calculate final metrics for the epoch
		epoch_qerrors = np.array(epoch_qerrors)
		median_qerror = np.median(epoch_qerrors)
		under_ratio = epoch_under_count / epoch_total_preds if epoch_total_preds > 0 else 0
		over_ratio = epoch_over_count / epoch_total_preds if epoch_total_preds > 0 else 0
		
		# Update progress bar
		progress_bar.set_postfix({
			'loss': f'{avg_loss:.4f}',
			'med_q': f'{median_qerror:.4f}',
			'under/over': f'{under_ratio:.2f}/{over_ratio:.2f}',
			'time': f'{epoch_time:.2f}s'
		})
		# Evaluate on test sets
		if epoch_id > 1:
			join_handler.start_eval()
			with torch.no_grad():
				all_est_cards = []
				all_true_cards = []

				for table_data_loader, test_parent_alias, test_keys, test_tables in zip(in_test_data_loader_list, in_test_parent_alias_list, in_test_training_keys_list, in_test_tables_list):
					for batch in table_data_loader:
						est_cards = join_handler.batch_estimate_join_queries_from_loader_w_mask(batch, test_parent_alias, test_keys, test_tables,  is_cuda=is_cuda)
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.where(est_cards > 1, est_cards, 1.)
							all_est_cards.extend(est_cards.cpu().detach())
							all_true_cards.extend(batch_cards.cpu().detach())
				q_errors = get_join_qerror(all_est_cards, all_true_cards, "seen", res_file, epoch_id)
				pg_q_errors = get_join_qerror(all_est_cards, all_true_cards, "seen", res_file, epoch_id) #TODO: this is wrong for pg
				seen_median_qerror = np.median(q_errors)
				seen_test_losses.append(seen_median_qerror)
				

				### unseen join templates 
				all_est_cards = []
				all_true_cards = []
				for table_data_loader, test_parent_alias, test_keys, test_tables in zip(ood_test_data_loader_list, ood_test_parent_alias_list, ood_test_training_keys_list, ood_test_tables_list):
					for batch in table_data_loader:
						est_cards = join_handler.batch_estimate_join_queries_from_loader_w_mask(batch, test_parent_alias, test_keys, test_tables, is_cuda=is_cuda)
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.where(est_cards > 1, est_cards, 1.)
							all_est_cards.extend(est_cards.cpu().detach())
							all_true_cards.extend(batch_cards.cpu().detach())
				q_errors = get_join_qerror(all_est_cards, all_true_cards, "unseen", res_file, epoch_id) #this will print results and save to file.
				pg_q_errors = get_join_qerror(all_est_cards, all_true_cards, "unseen", res_file, epoch_id) #TODO: this is wrong for pg
				unseen_median_qerror = np.median(q_errors)
				unseen_test_losses.append(unseen_median_qerror)
				

			# Save checkpoint every 10 epochs (saves model + optimizer + metrics)
			if (epoch_id + 1) % 10 == 0:
				# safe fallback for filename parts that might be undefined
				
				checkpoint = {
					'optimizer_state_dict': optimizer.state_dict(),
					'train_losses': train_losses,
					'train_qerrors': train_qerrors,
					'train_under_ratios': train_under_ratios,
					'train_over_ratios': train_over_ratios,
					'seen_test_losses': seen_test_losses,
					'unseen_test_losses': unseen_test_losses
				}
				join_handler.save_models(epoch_id + 1, bs, lr, lcs_dim)
			# last epoch:
			if epoch_id == epoch - 1:
				# Collect all data from seen test set
				seen_pg_cards = []
				seen_true_cards = []
				seen_est_cards = []
				seen_query_ids_list = []
				
				for table_data_loader, test_parent_alias, test_keys, test_tables, parent_t_list in zip(in_test_data_loader_list, in_test_parent_alias_list, in_test_training_keys_list, in_test_tables_list, in_test_loader_per_parent_template):
					pg_cards_for_loader = in_test_loader_per_parent_template[parent_t_list]['pg_cards']
					query_ids_for_loader = in_test_loader_per_parent_template[parent_t_list]['query_ids']
					
					offset = 0
					for batch in table_data_loader:
						# batch is a tuple of per-table tensors, so len(batch) counts tensors,
						# not rows; take the row count from the cards tensor instead.
						batch_size = len(batch[-1])
						est_cards = join_handler.batch_estimate_join_queries_from_loader_w_mask(batch, test_parent_alias, test_keys, test_tables, is_cuda=is_cuda)
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.where(est_cards > 1, est_cards, 1.)
							seen_est_cards.extend(est_cards.cpu().detach().numpy())
							seen_true_cards.extend(batch_cards.cpu().detach().numpy())
							seen_pg_cards.extend(pg_cards_for_loader[offset:offset + batch_size])
							seen_query_ids_list.extend(query_ids_for_loader[offset:offset + batch_size])
						offset += batch_size
				
				# Collect all data from unseen test set
				unseen_pg_cards = []
				unseen_true_cards = []
				unseen_est_cards = []
				unseen_query_ids_list = []
				
				for table_data_loader, test_parent_alias, test_keys, test_tables, parent_t_list in zip(ood_test_data_loader_list, ood_test_parent_alias_list, ood_test_training_keys_list, ood_test_tables_list, ood_test_loader_per_parent_template):
					pg_cards_for_loader = ood_test_loader_per_parent_template[parent_t_list]['pg_cards']
					query_ids_for_loader = ood_test_loader_per_parent_template[parent_t_list]['query_ids']
					
					offset = 0
					for batch in table_data_loader:
						# batch is a tuple of per-table tensors, so len(batch) counts tensors,
						# not rows; take the row count from the cards tensor instead.
						batch_size = len(batch[-1])
						est_cards = join_handler.batch_estimate_join_queries_from_loader_w_mask(batch, test_parent_alias, test_keys, test_tables, is_cuda=is_cuda)
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.where(est_cards > 1, est_cards, 1.)
							unseen_est_cards.extend(est_cards.cpu().detach().numpy())
							unseen_true_cards.extend(batch_cards.cpu().detach().numpy())
							unseen_pg_cards.extend(pg_cards_for_loader[offset:offset + batch_size])
							unseen_query_ids_list.extend(query_ids_for_loader[offset:offset + batch_size])
						offset += batch_size
				
				# Create dataframe for seen templates
				seen_df = pd.DataFrame({
					'template': [qid[0] for qid in seen_query_ids_list],
					'query_id': [qid[1] for qid in seen_query_ids_list],
					'type': ['seen'] * len(seen_query_ids_list),
					'pg_estimate': seen_pg_cards,
					'true_cardinality': seen_true_cards,
					'model_estimate': seen_est_cards
				})
				
				# Create dataframe for unseen templates
				unseen_df = pd.DataFrame({
					'template': [qid[0] for qid in unseen_query_ids_list],
					'query_id': [qid[1] for qid in unseen_query_ids_list],
					'type': ['unseen'] * len(unseen_query_ids_list),
					'pg_estimate': unseen_pg_cards,
					'true_cardinality': unseen_true_cards,
					'model_estimate': unseen_est_cards
				})
				
				# Combine both dataframes
				estimations_df = pd.concat([seen_df, unseen_df], ignore_index=True)
				
				# Save to CSV
				estimations_df.to_csv('estimations_dataframe.csv', index=False)
				print("Estimations dataframe saved to 'estimations_dataframe.csv'")
				
				# Plot estimations for both seen and unseen
				plot_estimations(seen_pg_cards, seen_true_cards, seen_est_cards, title='Seen Templates')
				plot_estimations(unseen_pg_cards, unseen_true_cards, unseen_est_cards, title='Unseen Templates')



	# Log total training time and finish
	total_time = time.time() - start_time
	print(f"Total training time: {total_time:.2f}s")
	
	
	# Plot training and validation curves
	# plt.figure(figsize=(12, 8))
	
	# # Plot 1: Training Loss and Q-Error
	# plt.subplot(2, 1, 1)
	# plt.plot(train_losses, label='SLE Loss', alpha=0.7)
	# plt.plot(np.convolve(train_qerrors, np.ones(50)/50, mode='valid'), 
	# 		label='Training Q-Error (MA50)', alpha=0.7)
	# plt.title('Training Metrics over Time')
	# plt.xlabel('Updates')
	# plt.ylabel('Value')
	# plt.legend()
	# plt.grid(True)
	
	# # Plot 2: Test Q-Error and Estimation Bias
	# plt.subplot(2, 1, 2)
	# plt.plot(seen_test_losses, label='Seen Templates', color='blue', alpha=0.7)
	# plt.plot(unseen_test_losses, label='Unseen Templates', color='red', alpha=0.7)
	
	# # Add under/over estimation as filled areas
	# under_ratio = np.convolve(train_under_ratios, np.ones(50)/50, mode='valid')
	# over_ratio = np.convolve(train_over_ratios, np.ones(50)/50, mode='valid')
	# x = np.arange(len(under_ratio))
	# plt.fill_between(x, 0, under_ratio, alpha=0.2, color='blue', label='Underestimation')
	# plt.fill_between(x, 0, over_ratio, alpha=0.2, color='red', label='Overestimation')
	
	# plt.title('Test Q-Error and Estimation Bias')
	# plt.xlabel('Epoch')
	# plt.ylabel('Value')
	# plt.legend()
	# plt.grid(True)
	
	# plt.tight_layout()
	# plot_path = f"./training_curves_{','.join(template_list)}_ratio_{sub_templates_in_training_ratio}.png"
	# plt.savefig(plot_path)
	# print(f"Training curves saved to {plot_path}")
	
	# # Save model
	# join_handler.save_models(epoch_id + 1, bs, lr, lcs_dim)
	
	# res_file.write("PG Test workload:{}: Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{} \n".format(
	# 'seen', np.mean(pg_q_errors['seen']), gmean(pg_q_errors['seen']), np.median(pg_q_errors['seen']),
	# np.percentile(pg_q_errors['seen'],90), np.percentile(pg_q_errors['seen'],95), np.percentile(pg_q_errors['seen'],99), np.max(pg_q_errors['seen'])))		
	
	# res_file.write("PG Test workload:{}: Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{} \n".format(
	# 'unseen', np.mean(pg_q_errors['unseen']), gmean(pg_q_errors['unseen']), np.median(pg_q_errors['unseen']),
	# np.percentile(pg_q_errors['unseen'],90), np.percentile(pg_q_errors['unseen'],95), np.percentile(pg_q_errors['unseen'],99), np.max(pg_q_errors['unseen'])))		
	# res_file.flush()

	res_file.close()
	return pg_q_errors

def plot_estimations(pg, true_cards, est, title='unseen'):

	plt.figure(figsize=(10, 6))
	plt.suptitle(title)
	plt.subplot(1,2,1)
	plt.plot(pg, label='PG Estimations', color='blue', alpha=0.7)
	plt.plot(true_cards, label='True Cardinalities', color='orange', alpha=0.7)
	plt.plot(est, label='GRASP Estimations', color='green', alpha=0.7)
	plt.plot(true_cards/pg, label='True/PG Ratio', color='red', alpha=0.7)
	plt.xlabel('Query Index')
	plt.ylabel('Cardinality')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('estimations_plot.png')
	print("Estimations plot saved as 'estimations_plot.png'")	

	plt.subplot(1,2,2)
	plt.scatter(true_cards, pg, label='PG Estimations', color='blue', alpha=0.7)
	plt.scatter(true_cards, est, label='GRASP Estimations', color='green', alpha=0.7)
	plt.legend()
	plt.xlabel('True Cardinalities')
	plt.show()
	

	
def main():
	"""
	Main function to parse command-line arguments and initiate the training process.
	Arguments:
		--epochs (int): Number of epochs for training (default: 100).
		--feature_dim (int): Dimension of hidden layer of CE model (default: 256).
		--lcs_dim (int): Dimension of the latent space of LCS model (default: 500).
		--bs (int): Batch size for training (default: 128).
	The function checks if CUDA is available and then calls the train_grasp function
	with the parsed arguments.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", help="number of epochs (default: 100)", type=int, default=100)
	parser.add_argument("--feature_dim", help="dimension of hidden layer of CE model (default: 256)", type=int, default=256)
	parser.add_argument("--lcs_dim", help="Dimension of the latent space of LCS model (default: 500)", type=int, default=500)
	parser.add_argument("--bs", help="batch size (default: 128)", type=int, default=128)
	parser.add_argument("--residual", type=int, default=0)
	args = parser.parse_args()

	is_cuda = torch.cuda.is_available()

	pg_q_errors = train_grasp(args.epochs, args.feature_dim, args.lcs_dim, args.bs, residual=args.residual)


if __name__ == "__main__":
	main()
    #TODO: I need to set the random see, to be sure of the residual performance.