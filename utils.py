import json
import os
import time
import numpy as np
from scipy.stats import gmean
from tqdm import tqdm


def resolve_run_id(run_id=None):
	"""Returns a label unique to this run, so concurrent runs never share output paths.

	An explicit run_id wins. Otherwise the scheduler's job id is used (SLURM array tasks
	get 'job_task'), which makes results directly traceable back to the batch job. With no
	scheduler - a local run - falls back to a timestamp plus pid.
	"""
	if run_id:
		return str(run_id)

	array_job = os.environ.get('SLURM_ARRAY_JOB_ID')
	array_task = os.environ.get('SLURM_ARRAY_TASK_ID')
	if array_job and array_task:
		return "{}_{}".format(array_job, array_task)

	for env_key in ['SLURM_JOB_ID', 'SLURM_JOBID', 'PBS_JOBID', 'LSB_JOBID', 'JOB_ID']:
		job_id = os.environ.get(env_key)
		if job_id:
			return str(job_id)

	return "{}-{}".format(time.strftime('%Y%m%d-%H%M%S'), os.getpid())


def make_run_dir(workload, run_id, results_root='results'):
	"""Creates (and returns) results_root/workload/run_id for this run's artifacts."""
	run_dir = os.path.join(results_root, workload, run_id)
	os.makedirs(run_dir, exist_ok=True)
	return run_dir


def save_run_config(run_dir, config):
	"""Dumps the run's hyperparameters next to its outputs, so a run id can be decoded later."""
	with open(os.path.join(run_dir, 'run_config.json'), 'w') as config_file:
		json.dump(config, config_file, indent=2, default=str)


def plot_estimations(pg, true_cards, est, title='unseen', out_dir='.'):
	"""Plots true vs. postgres vs. model cardinalities and saves the figure under out_dir."""
	# Imported lazily and forced to a non-interactive backend: training also runs on
	# headless servers where the default backend has no display to attach to.
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	# The callers pass Python lists; arithmetic below needs arrays.
	pg = np.asarray(pg, dtype=np.float64)
	true_cards = np.asarray(true_cards, dtype=np.float64)
	est = np.asarray(est, dtype=np.float64)
	# PG can report 0 rows, which would make the ratio inf/nan.
	ratio = true_cards / np.maximum(pg, 1.)

	plt.figure(figsize=(12, 6))
	plt.suptitle(title)
	plt.subplot(1, 2, 1)
	plt.plot(pg, label='PG Estimations', color='blue', alpha=0.7)
	plt.plot(true_cards, label='True Cardinalities', color='orange', alpha=0.7)
	plt.plot(est, label='GRASP Estimations', color='green', alpha=0.7)
	plt.plot(ratio, label='True/PG Ratio', color='red', alpha=0.7)
	plt.yscale('log')
	plt.xlabel('Query Index')
	plt.ylabel('Cardinality')
	plt.legend()
	plt.grid(True)

	plt.subplot(1, 2, 2)
	plt.scatter(true_cards, pg, label='PG Estimations', color='blue', alpha=0.7)
	plt.scatter(true_cards, est, label='GRASP Estimations', color='green', alpha=0.7)
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('True Cardinalities')
	plt.ylabel('Estimated Cardinality')
	plt.grid(True)

	plt.tight_layout()
	plot_path = os.path.join(out_dir, "estimations_plot_{}.png".format(title.replace(' ', '_').lower()))
	plt.savefig(plot_path)
	plt.close()
	print("Estimations plot saved as '{}'".format(plot_path))
	return plot_path

def get_queries(q_file_name, num_samples=50000, test_size=1000):
	queries = []
	intervals = []
	sels = []

	q_file = open(q_file_name, 'r')
	lines = q_file.readlines()
	for line in lines:
		q_obj = json.loads(line)
		queries.append(q_obj['query'])
		tmp = q_obj['interval']
		interval = []
		for i in range(len(tmp[0])):
			interval.append(tmp[0][i])
			interval.append(tmp[1][i])
		intervals.append(interval)
		sels.append(float(q_obj['card']) / num_samples)

	candi_size = 50000

	training_intervals_tmp = intervals[:candi_size]
	training_sels_tmp = sels[:candi_size]
	training_queries_tmp = queries[:candi_size]

	training_queries = []
	training_intervals = []
	training_sels = []

	ood_test_queries = []
	ood_test_intervals = []
	ood_test_sels = []

	low_count = 0
	median_count = 0
	high_count = 0

	q_id = 0
	for q, interval, sel in zip(training_queries_tmp, training_intervals_tmp, training_sels_tmp):
		card = sel * num_samples
		if q_id < 300:
			low_count += 1
			training_queries.append(q)
			training_intervals.append(interval)
			training_sels.append(sel)
		elif q_id < 35000:
			median_count += 1
		else:
			high_count += 1
			ood_test_intervals.append(interval)
			ood_test_sels.append(sel)
			ood_test_queries.append(q)
		q_id += 1

	ood_test_intervals = ood_test_intervals[:test_size]
	ood_test_sels = ood_test_sels[:test_size]
	ood_test_queries = ood_test_queries[:test_size]

	training_size = len(training_queries)
	actual_training_size = int(training_size * 0.9)

	test_queries = training_queries[actual_training_size:training_size]
	test_intervals = training_intervals[actual_training_size:training_size]
	test_sels = training_sels[actual_training_size:training_size]

	training_queries = training_queries[:actual_training_size]
	training_intervals = training_intervals[:actual_training_size]
	training_sels = training_sels[:actual_training_size]

	return training_queries, training_intervals, training_sels, test_queries, test_intervals, test_sels, ood_test_queries, ood_test_intervals, ood_test_sels

def get_qerror(preds, targets, workload_type='ID-Distribution', num_data=50000):
	qerror = []
	preds = preds.cpu().data.numpy()
	targets = targets.cpu().data.numpy()
	rmse = 0.
	for i in range(len(targets)):
		if preds[i] <= 1. / num_data:
			preds[i] = 1. / num_data

		rmse += np.square(preds[i] - targets[i])

		if (preds[i] > targets[i]):
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])

	rmse = np.sqrt(rmse / len(qerror))

	print("Test workload:{}: RMSE:{}, Median Qerror: {}".format(workload_type, rmse, np.median(qerror)))


def get_join_qerror(preds, targets, workload_type='In-Distribution', res_file=None, epoch_id=None):
	qerror = []
	for i in range(len(targets)):
		if preds[i] <= 1.:
			preds[i] = 1.

		if (preds[i] > targets[i]):
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])

	# tqdm.write instead of print: a bare print during training scrambles the progress bar.
	tqdm.write("Test workload:{}: Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{}".format(
		workload_type, np.mean(qerror), gmean(qerror), np.median(qerror), np.percentile(qerror,90), np.percentile(qerror,95), np.percentile(qerror,99), np.max(qerror)))
	
	if res_file is not None:
		res_file.write("epoch_id:{}; Test workload:{}: Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{} \n".format(
		epoch_id, workload_type, np.mean(qerror), gmean(qerror), np.median(qerror), np.percentile(qerror,90), np.percentile(qerror,95), np.percentile(qerror,99), np.max(qerror)))
		res_file.flush()
	return qerror


def extract_sublist(original_list, indices):
	return [original_list[i] for i in indices]