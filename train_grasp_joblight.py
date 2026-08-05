"""Trains GRASP on the JOB-light workload.

Differs from train_grasp_ceb in how the test set is obtained: CEB holds out 100 queries
per join template plus a sample of unseen templates, all carved from the same directory.
Here the test set is the fixed external JOB-light benchmark (70 queries), so every
training query is used for training and there is a single test workload to report.
"""

import argparse
import math
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from CEB_utlities.join_utilits_CEB_job import get_table_to_ops
from CEB_utlities.join_utilits_joblight import (
	get_joblight_table_info,
	read_joblight_query_files,
)
from GRASP.join_handler_ceb import JoinHandler
from utils import get_join_qerror

import random

random.seed(42)

is_cuda = torch.cuda.is_available()

JOBLIGHT_TRAIN_DIRECTORY = "C:/Ottawa/RA/Query/Cardinality Estimation/RobustMSCN/queries/joblight_train/joblight-train-all"
JOBLIGHT_TEST_DIRECTORY = "C:/Ottawa/RA/Query/Cardinality Estimation/RobustMSCN/queries/joblight/all_joblight"
PROCESSED_WORKLOAD_DIRECTORY = "processed_workload/joblight/"
MIN_MAX_FILE = "queries/column_min_max_vals_imdb.csv"

TEST_BATCH_SIZE = 3000


def _new_bucket(with_pg=False):
	keys = ['qs', 'q_contexts', 'qreps', 'table_masks', 'table_group_masks', 'is_stb',
	        'key_groups', 'cards']
	if with_pg:
		keys += ['pg_cards', 'query_ids']
	return {key: [] for key in keys}


def _bucket_queries(template2queries, template2cards, template2pgcards, residual,
                    group_by_parent, with_pg):
	"""Groups query records into per-join-template buckets the JoinHandler can batch.

	group_by_parent picks which table tuple keys the bucket: training batches a subplan
	as a standalone query over its own tables, while evaluation runs it inside its
	parent template so the table masks line up.
	"""
	buckets = {}
	for template in template2queries:
		cards = template2cards[template]
		pg_cards = template2pgcards[template]

		for qid, record in enumerate(template2queries[template]):
			(q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple,
			 table_tuple, table_mask, table_group_mask) = record

			t_list = tuple(parent_table_tuple if group_by_parent else table_tuple)
			if t_list not in buckets:
				buckets[t_list] = _new_bucket(with_pg=with_pg)
			bucket = buckets[t_list]

			bucket['qs'].append(q)
			bucket['q_contexts'].append(q_context)
			bucket['qreps'].append(q_reps)
			bucket['table_masks'].append(table_mask)
			bucket['table_group_masks'].append(table_group_mask)
			bucket['is_stb'].append(0 if len(list(table_tuple)) > 1 else 1)
			bucket['key_groups'].append(parent_key_groups if group_by_parent else key_groups_per_q)
			bucket['cards'].append(cards[qid] / pg_cards[qid] if residual else cards[qid])

			if with_pg:
				bucket['pg_cards'].append(pg_cards[qid])
				bucket['query_ids'].append((template, qid))

	return buckets


def _evaluate(join_handler, test_loaders, is_cuda):
	"""Runs every test loader and returns estimates aligned with their pg cards / ids."""
	all_est = []
	all_true = []
	all_pg = []
	all_ids = []

	with torch.no_grad():
		for entry in test_loaders:
			offset = 0
			for batch in entry['loader']:
				batch_size = len(batch[-1])
				est_cards = join_handler.batch_estimate_join_queries_from_loader_w_mask(
					batch, entry['parent_alias'], entry['keys'], entry['tables'], is_cuda=is_cuda)

				if est_cards is not None:
					est_cards = torch.where(est_cards > 1, est_cards, 1.)
					all_est.extend(est_cards.cpu().detach().numpy().flatten().tolist())
					all_true.extend(batch[-1].cpu().detach().numpy().flatten().tolist())
					all_pg.extend(entry['pg_cards'][offset:offset + batch_size])
					all_ids.extend(entry['query_ids'][offset:offset + batch_size])

				offset += batch_size

	return all_est, all_true, all_pg, all_ids


def train_grasp(epoch=100, feature_dim=256, lcs_dim=500, bs=128, lr=5e-4, residual=0,
                num_train_q=None, include_subplans=True,
                train_directory=JOBLIGHT_TRAIN_DIRECTORY,
                test_directory=JOBLIGHT_TEST_DIRECTORY,
                saved_directory=PROCESSED_WORKLOAD_DIRECTORY):
	start_time = time.time()
	cdf_model_choice = 'arcdf'

	res_file = open("./grasp_joblight_lcssize_{}_lr_{}_bs_{}.txt".format(lcs_dim, lr, bs), 'a')

	(table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	 table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax) = \
		get_joblight_table_info(MIN_MAX_FILE)

	(template2queries, template2cards, template2pgcards, test_template2queries,
	 test_template2cards, test_template2pgcards, colid2featlen_per_table) = \
		read_joblight_query_files(col2minmax=col2minmax,
		                          train_directory=train_directory,
		                          test_directory=test_directory,
		                          saved_directory=saved_directory,
		                          num_train_q=num_train_q,
		                          include_subplans=include_subplans)

	t2ops = get_table_to_ops(template2queries, test_template2queries)
	for t in t2ops:
		print(t, t2ops[t])

	join_handler = JoinHandler(table_list=table_list, table_dim_list=table_dim_list,
	                           table_key_groups=table_key_groups, table_size_list=table_sizes,
	                           cdf_model_choice=cdf_model_choice, hidden_size=feature_dim,
	                           lcs_size=lcs_dim, colid2featlen_per_table=colid2featlen_per_table)

	join_handler.init_ce_predictors(colid2featlen_per_table)
	join_handler.init_unified_lcs_predictors()
	join_handler.models_to_double()
	if is_cuda:
		join_handler.load_models_to_gpu()
	optimizer = torch.optim.Adam(join_handler.get_parameters(), lr=lr)

	training_buckets = _bucket_queries(template2queries, template2cards, template2pgcards,
	                                   residual, group_by_parent=False, with_pg=False)
	test_buckets = _bucket_queries(test_template2queries, test_template2cards,
	                               test_template2pgcards, residual, group_by_parent=True,
	                               with_pg=True)

	num_qs = sum(len(bucket['cards']) for bucket in training_buckets.values())
	num_test_qs = sum(len(bucket['cards']) for bucket in test_buckets.values())
	num_batches = max(1, math.ceil(num_qs / bs))

	print("total number of train queries: {}".format(num_qs))
	print("total number of train join templates: {}".format(len(training_buckets)))
	print("total number of test queries: {}".format(num_test_qs))
	print("total number of test join templates: {}".format(len(test_buckets)))

	res_file.write("total number of train queries: {} \n".format(num_qs))
	res_file.write("total number of train join templates: {} \n".format(len(training_buckets)))
	res_file.write("total number of test queries: {} \n".format(num_test_qs))

	#### start loading data for pytorch training

	data_loader_list = []
	data_loader_iter_list = []
	train_key_groups_list = []
	train_tables_list = []

	for t_list in training_buckets:
		bucket = training_buckets[t_list]
		table_data_loader, _ = join_handler.load_training_queries(bucket['qs'],
		                                                          bucket['q_contexts'],
		                                                          bucket['qreps'],
		                                                          bucket['cards'],
		                                                          bs, t_list, is_cuda=is_cuda)

		data_loader_list.append(table_data_loader)
		data_loader_iter_list.append(iter(table_data_loader))
		train_key_groups_list.append(bucket['key_groups'][0])
		train_tables_list.append(t_list)

	test_loaders = []
	for parent_t_list in test_buckets:
		bucket = test_buckets[parent_t_list]
		# is_shuffle=False keeps estimates aligned with pg_cards and query_ids.
		table_data_loader, parent_alias_list, keys, tables = \
			join_handler.load_training_queries_w_masks(bucket['qs'],
			                                           bucket['q_contexts'],
			                                           bucket['qreps'],
			                                           bucket['key_groups'][0],
			                                           bucket['table_masks'],
			                                           bucket['table_group_masks'],
			                                           bucket['is_stb'],
			                                           bucket['cards'],
			                                           TEST_BATCH_SIZE, is_cuda=is_cuda,
			                                           is_shuffle=False)

		test_loaders.append({'loader': table_data_loader,
		                     'parent_alias': parent_alias_list,
		                     'keys': keys,
		                     'tables': tables,
		                     'pg_cards': bucket['pg_cards'],
		                     'query_ids': bucket['query_ids']})

	############ finish data loading

	### postgres baseline on the same test workload
	pg_est = []
	pg_true = []
	for bucket in test_buckets.values():
		pg_est.extend(bucket['pg_cards'])
		pg_true.extend(bucket['cards'])
	get_join_qerror(list(pg_est), list(pg_true), "postgres-joblight", res_file, -1)

	progress_bar = tqdm(range(epoch), desc="Training progress")
	for epoch_id in progress_bar:
		epoch_start_time = time.time()
		join_handler.start_train()

		accu_loss_total = 0.
		epoch_qerrors = []
		epoch_under_count = 0
		epoch_over_count = 0
		epoch_total_preds = 0

		current_data_loader_ids = list(range(len(data_loader_iter_list)))
		while len(current_data_loader_ids) > 0:
			d_loader_id = random.choice(current_data_loader_ids)

			table_data_loader = data_loader_iter_list[d_loader_id]
			table_list_for_loader = train_tables_list[d_loader_id]
			key_groups = train_key_groups_list[d_loader_id]

			try:
				batch = next(table_data_loader)
				keys_order, tables_order = join_handler.template_to_group_order(key_groups)
				est_cards = join_handler.batch_estimate_join_queries_from_loader(
					batch, table_list_for_loader, keys_order, tables_order, is_cuda=is_cuda)
				if not est_cards.requires_grad:
					continue

				batch_cards = batch[-1].to(torch.float64)
				optimizer.zero_grad()

				adj_est_cards = torch.where(est_cards > 0, est_cards, 1.)

				sle_loss = torch.square(torch.log(torch.squeeze(adj_est_cards)) - torch.log(batch_cards))
				se_loss = torch.square(torch.squeeze(est_cards) - batch_cards)

				ultimate_loss = torch.where(torch.squeeze(est_cards) > 0, sle_loss, se_loss)
				total_loss = torch.mean(ultimate_loss)
				accu_loss_total += total_loss.item()

				est_np = torch.squeeze(est_cards).detach().cpu().numpy()
				true_np = batch_cards.detach().cpu().numpy()

				if np.isscalar(est_np) or est_np.ndim == 0:
					est_np = np.array([est_np])
					true_np = np.array([true_np])

				q_errors = np.maximum(est_np / true_np, true_np / est_np)

				epoch_qerrors.extend(q_errors.flatten())
				epoch_under_count += int(np.sum(est_np < true_np))
				epoch_over_count += int(np.sum(est_np > true_np))
				epoch_total_preds += est_np.size

				total_loss.backward()
				clip_grad_norm_(join_handler.get_parameters(), 10.)
				optimizer.step()

			except StopIteration:
				current_data_loader_ids.remove(d_loader_id)
				data_loader_iter_list[d_loader_id] = iter(data_loader_list[d_loader_id])

		epoch_time = time.time() - epoch_start_time
		avg_loss = accu_loss_total / num_batches
		median_qerror = np.median(np.array(epoch_qerrors)) if epoch_qerrors else float('nan')
		under_ratio = epoch_under_count / epoch_total_preds if epoch_total_preds > 0 else 0
		over_ratio = epoch_over_count / epoch_total_preds if epoch_total_preds > 0 else 0

		progress_bar.set_postfix({
			'loss': '{:.4f}'.format(avg_loss),
			'med_q': '{:.4f}'.format(median_qerror),
			'under/over': '{:.2f}/{:.2f}'.format(under_ratio, over_ratio),
			'time': '{:.2f}s'.format(epoch_time),
		})

		if epoch_id > 1:
			join_handler.start_eval()
			all_est, all_true, all_pg, all_ids = _evaluate(join_handler, test_loaders, is_cuda)
			get_join_qerror(list(all_est), list(all_true), "joblight", res_file, epoch_id)

			if (epoch_id + 1) % 10 == 0:
				join_handler.save_models(epoch_id + 1, bs, lr, lcs_dim)

			if epoch_id == epoch - 1:
				estimations_df = pd.DataFrame({
					'template': [str(qid[0]) for qid in all_ids],
					'query_id': [qid[1] for qid in all_ids],
					'pg_estimate': all_pg,
					'true_cardinality': all_true,
					'model_estimate': all_est,
				})
				estimations_df.to_csv('joblight_estimations_dataframe.csv', index=False)
				print("Estimations dataframe saved to 'joblight_estimations_dataframe.csv'")

	print("Total training time: {:.2f}s".format(time.time() - start_time))
	res_file.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", help="number of epochs (default: 100)", type=int, default=100)
	parser.add_argument("--feature_dim", help="dimension of hidden layer of CE model (default: 256)",
	                    type=int, default=256)
	parser.add_argument("--lcs_dim", help="Dimension of the latent space of LCS model (default: 500)",
	                    type=int, default=500)
	parser.add_argument("--bs", help="batch size (default: 128)", type=int, default=128)
	parser.add_argument("--lr", help="learning rate (default: 5e-4)", type=float, default=5e-4)
	parser.add_argument("--residual", type=int, default=0)
	parser.add_argument("--num_train_q", type=int, default=None,
	                    help="cap on training query files (default: all)")
	parser.add_argument("--no_subplans", action="store_true",
	                    help="train on full queries only, skipping their subplans")
	parser.add_argument("--train_directory", default=JOBLIGHT_TRAIN_DIRECTORY)
	parser.add_argument("--test_directory", default=JOBLIGHT_TEST_DIRECTORY)
	parser.add_argument("--saved_directory", default=PROCESSED_WORKLOAD_DIRECTORY)
	args = parser.parse_args()

	train_grasp(args.epochs, args.feature_dim, args.lcs_dim, args.bs, lr=args.lr,
	            residual=args.residual, num_train_q=args.num_train_q,
	            include_subplans=not args.no_subplans,
	            train_directory=args.train_directory,
	            test_directory=args.test_directory,
	            saved_directory=args.saved_directory)


if __name__ == "__main__":
	main()
