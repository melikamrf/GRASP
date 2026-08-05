"""Workload loading for the JOB-light benchmark.

JOB-light needs its own schema config rather than the one in join_utilits_CEB_job:
its predicates sit on columns that config lists as join keys and never as filters
(mc.company_id, mi.info_type_id, mk.keyword_id, ci.person_id, ...), while the only
join it ever performs is the star on title.id. Reusing the CEB config would both fail
to find the predicate columns and build latent-count-space predictors for seven key
groups that this workload never joins on.

Unlike CEB, the test set here is a fixed external directory (the 70 JOB-light queries)
rather than a split carved out of the training queries.
"""

import csv
import json
import os
import pickle

from CEB_utlities.join_utilits_CEB_job import (
	default_serializer,
	find_connected_clusters,
	generate_key_group_mask,
	normalize_val,
)
from CEB_utlities.query_representation.query import (
	get_joins,
	get_postgres_cardinalities,
	get_predicates,
	get_tables,
	get_true_cardinalities,
	load_qrep,
	subplan_to_joins,
)

# JOB-light spells the movie_info_idx alias 'mi_idx' where CEB uses 'mii'.
ALIAS = {
	'title': 't',
	'cast_info': 'ci',
	'movie_companies': 'mc',
	'movie_info': 'mi',
	'movie_info_idx': 'mi_idx',
	'movie_keyword': 'mk',
}

REVERSE_ALIAS = {alias: table for table, alias in ALIAS.items()}

FILTER_COLS = {
	'title': ['t.kind_id', 't.production_year'],
	'cast_info': ['ci.person_id', 'ci.role_id'],
	'movie_companies': ['mc.company_id', 'mc.company_type_id'],
	'movie_info': ['mi.info_type_id'],
	'movie_info_idx': ['mi_idx.info_type_id'],
	'movie_keyword': ['mk.keyword_id'],
}

JOIN_MAP = {
	'title.id': 'movie_id',
	'cast_info.movie_id': 'movie_id',
	'movie_companies.movie_id': 'movie_id',
	'movie_info.movie_id': 'movie_id',
	'movie_info_idx.movie_id': 'movie_id',
	'movie_keyword.movie_id': 'movie_id',
}

TABLE_SIZES = {
	'title': 2528312,
	'cast_info': 36244344,
	'movie_companies': 2609129,
	'movie_info': 14835720,
	'movie_info_idx': 1380035,
	'movie_keyword': 4523930,
}


def get_joblight_table_info(file_path='./queries/column_min_max_vals_imdb.csv'):
	"""Mirrors get_table_info from join_utilits_CEB_job, for the JOB-light schema."""
	col2minmax = {}
	for line in open(file_path, 'r').readlines()[1:]:
		parts = line.strip().split(',')
		col2minmax[parts[0]] = [int(parts[1]), int(parts[2]) + 1]

	table_list = []
	table_dim_list = []
	table_like_dim_list = []
	table_text_cols = {}
	table_normal_cols = {}
	col_type = {}

	for table, cols in FILTER_COLS.items():
		table_list.append(table)
		table_dim_list.append(len(cols))
		table_like_dim_list.append(0)
		table_text_cols[table] = []
		table_normal_cols[table] = list(cols)
		for col in cols:
			col_type[col] = 'number'

	table_key_groups = {}
	table_join_keys = {}
	for col_name, key_group in JOIN_MAP.items():
		table = col_name.split('.')[0]
		table_key_groups.setdefault(key_group, [])
		if table not in table_key_groups[key_group]:
			table_key_groups[key_group].append(table)
		table_join_keys.setdefault(table, [])
		if key_group not in table_join_keys[table]:
			table_join_keys[table].append(key_group)

	return (table_list, table_dim_list, table_like_dim_list, TABLE_SIZES, table_key_groups,
	        table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax)


def _parse_predicates(pred_cols, pred_types, pred_vals, aliases, col2minmax,
                      colid2featlen_per_table):
	table2normal_predicates = {alias: [] for alias in aliases}
	table2text_predicates = {alias: [] for alias in aliases}
	table2qreps = {alias: [] for alias in aliases}

	# Iterate pred_cols rather than the predicate strings: a two-sided range reads as two
	# strings ('t.production_year > 2005', '... < 2010') but one entry in these arrays.
	for i in range(len(pred_cols)):
		for j in range(len(pred_cols[i])):
			col = pred_cols[i][j]
			op = pred_types[i][j]
			val = pred_vals[i][j]

			alias_t = col.split('.')[0]
			full_t = REVERSE_ALIAS[alias_t]
			standard_col_name = ALIAS[full_t] + '.' + col.split('.')[1]
			col_id = FILTER_COLS[full_t].index(standard_col_name)

			min_val, max_val = col2minmax[standard_col_name]
			featlens = colid2featlen_per_table[full_t]

			if op == 'lt':
				low = 0. if val[0] is None else normalize_val(val[0], min_val, max_val, col)
				high = 1. if val[1] is None else normalize_val(val[1], min_val, max_val, col)
				table2normal_predicates[alias_t].append([col_id, [[low, high]]])
				table2qreps[alias_t].append([col_id, op, [low, high]])
				featlens[col_id] = max(2, featlens.get(col_id, 0))
			elif op == 'eq':
				if isinstance(val, dict):
					val = int(val['literal'])
				low = normalize_val(val, min_val, max_val, col)
				high = normalize_val(val + 1, min_val, max_val, col)
				table2normal_predicates[alias_t].append([col_id, [[low, high]]])
				table2qreps[alias_t].append([col_id, op, low])
				featlens[col_id] = max(1, featlens.get(col_id, 0))
			else:
				raise ValueError("unsupported JOB-light predicate op '{}' on {}".format(op, col))

	return table2normal_predicates, table2text_predicates, table2qreps


def _parse_joins(joins):
	key_groups = {}
	for join in joins:
		parts = join.strip().split(' = ')
		alias_table1 = parts[0].split('.')[0]
		alias_table2 = parts[1].split('.')[0]
		col1 = parts[0].split('.')[1]
		key_group = JOIN_MAP[REVERSE_ALIAS[alias_table1] + '.' + col1]
		key_groups.setdefault(key_group, []).append([alias_table1, alias_table2])

	key_groups = {key: key_groups[key] for key in sorted(key_groups)}
	for key_group in key_groups:
		key_groups[key_group].sort()
		key_groups[key_group] = find_connected_clusters(key_groups[key_group])
	return key_groups


def _raw_predicate_tuple(pred_cols, pred_types, pred_vals, keep_aliases=None):
	raw = []
	for i in range(len(pred_cols)):
		for j in range(len(pred_cols[i])):
			alias_t, col = pred_cols[i][j].split('.')
			if keep_aliases is not None and alias_t not in keep_aliases:
				continue
			val = pred_vals[i][j]
			if isinstance(val, list):
				val = tuple(val)
			elif isinstance(val, dict):
				val = int(val['literal'])
			raw.append((alias_t, col, pred_types[i][j], val))
	raw.sort(key=str)
	return tuple(raw)


def _raw_join_tuple(joins):
	raw = []
	for join in joins:
		if ' = ' not in join:
			continue
		parts = join.strip().split(' = ')
		t1, c1 = parts[0].split('.')
		t2, c2 = parts[1].split('.')
		raw.append((t2, c2, t1, c1) if t1 > t2 else (t1, c1, t2, c2))
	raw.sort()
	return tuple(raw)


def _load_directory(directory, col2minmax, colid2featlen_per_table, max_files=None,
                    with_subplans=False, saved_predicates=None):
	"""Parses every qrep in a directory into GRASP query_info records.

	Returns parallel lists of (query_info, true_card, pg_card, raw_description).
	"""
	files = sorted(file for file in os.listdir(directory) if file.endswith('.pkl'))
	if max_files is not None:
		files = files[:max_files]

	queries = []
	cards = []
	pg_cards = []
	raws = []

	for file_id, file in enumerate(files):
		if file_id % 5000 == 0:
			print("  {} / {} files".format(file_id, len(files)))

		qrep = load_qrep(os.path.join(directory, file))
		tables, aliases = get_tables(qrep)
		joins = get_joins(qrep)
		_, pred_cols, pred_types, pred_vals = get_predicates(qrep)

		table2normal_predicates, table2text_predicates, table2qreps = _parse_predicates(
			pred_cols, pred_types, pred_vals, aliases, col2minmax,
			colid2featlen_per_table)
		key_groups = _parse_joins(joins)

		trues = get_true_cardinalities(qrep)
		ests = get_postgres_cardinalities(qrep)

		sorted_aliases = sorted(aliases)
		table_mask, table_group_mask = generate_key_group_mask(sorted_aliases, key_groups)

		card = 0.
		pg_card = 0
		for subplan in trues:
			if len(subplan) == len(tables):
				card = trues[subplan]
				pg_card = ests[subplan]

		queries.append([table2normal_predicates, table2text_predicates, table2qreps,
		                key_groups, key_groups, sorted_aliases, sorted_aliases,
		                table_mask, table_group_mask])
		cards.append(card)
		pg_cards.append(pg_card)
		raws.append((file, _raw_join_tuple(joins),
		             _raw_predicate_tuple(pred_cols, pred_types, pred_vals)))

		if not with_subplans:
			continue

		for subplan in trues:
			if len(subplan) == len(tables):
				continue

			sub_aliases = sorted(subplan)
			sub_key_groups = {}
			for key_group, group_t_list in key_groups.items():
				for ts in group_t_list:
					intersect_ts = [t for t in ts if t in subplan]
					if len(intersect_ts) >= 2:
						sub_key_groups.setdefault(key_group, []).append(intersect_ts)

			sub_normal = {alias: [] for alias in sorted_aliases}
			sub_text = {alias: [] for alias in sorted_aliases}
			sub_qreps = {alias: [] for alias in sorted_aliases}
			for t in subplan:
				sub_normal[t] = table2normal_predicates[t]
				sub_text[t] = table2text_predicates[t]
				sub_qreps[t] = table2qreps[t]

			# Subplans repeat heavily across the workload; keep one per distinct
			# (table set, predicate set) so common single-table filters don't dominate.
			signature = json.dumps({t: sub_normal[t] for t in subplan}, sort_keys=True,
			                       default=default_serializer)
			table_key = tuple(sub_aliases)
			seen = saved_predicates.setdefault(table_key, set())
			if signature in seen:
				continue
			seen.add(signature)

			sub_table_mask, sub_table_group_mask = generate_key_group_mask(
				list(subplan), key_groups)

			queries.append([sub_normal, sub_text, sub_qreps, key_groups, sub_key_groups,
			                sorted_aliases, sub_aliases, sub_table_mask,
			                sub_table_group_mask])
			cards.append(trues[subplan])
			pg_cards.append(ests[subplan])
			raws.append((file, _raw_join_tuple(subplan_to_joins(qrep, subplan)),
			             _raw_predicate_tuple(pred_cols, pred_types, pred_vals,
			                                  keep_aliases=set(subplan))))

	return queries, cards, pg_cards, raws


def _group_by_table_tuple(queries, cards, pg_cards, raws):
	template2queries = {}
	template2cards = {}
	template2pgcards = {}
	template2raw = {}

	for query_info, card, pg_card, raw in zip(queries, cards, pg_cards, raws):
		table_list = tuple(query_info[-3])
		template2queries.setdefault(table_list, []).append(query_info)
		template2cards.setdefault(table_list, []).append(card)
		template2pgcards.setdefault(table_list, []).append(pg_card)
		template2raw.setdefault(table_list, []).append(raw)

	return template2queries, template2cards, template2pgcards, template2raw


def _write_raw_csv(path, template2cards, template2pgcards, template2raw):
	with open(path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(['template', 'join_tables', 'predicates', 'pgcard', 'true_card',
		                 'model_card'])
		for template in template2raw:
			for raw, card, pg_card in zip(template2raw[template], template2cards[template],
			                              template2pgcards[template]):
				writer.writerow([raw[0], str(raw[1]), str(raw[2]), pg_card, card, 'N/A'])


_CACHE_FILES = ['template2queries', 'template2cards', 'template2pgcards',
                'test_template2queries', 'test_template2cards', 'test_template2pgcards',
                'colid2featlen_per_table']


def read_joblight_query_files(col2minmax, train_directory, test_directory,
                              saved_directory, num_train_q=None, include_subplans=True):
	"""Loads JOB-light training queries and the fixed JOB-light test queries.

	Returns the same 7-tuple shape as read_query_file_batched, so the GRASP training
	code can consume it unchanged. Unlike that function the test set is not sampled
	from the training queries - it is the external `test_directory` verbatim.
	"""
	workload_file_path = "{}joblight-{}-{}-".format(
		saved_directory, num_train_q if num_train_q else 'all',
		'sub' if include_subplans else 'nosub')

	if os.path.exists(workload_file_path + 'template2queries.pkl'):
		loaded = []
		for name in _CACHE_FILES:
			print("load {}".format(name))
			with open(workload_file_path + name + '.pkl', "rb") as pickle_file:
				loaded.append(pickle.load(pickle_file))
		return tuple(loaded)

	os.makedirs(saved_directory, exist_ok=True)

	colid2featlen_per_table = {table: {} for table in TABLE_SIZES}
	saved_predicates = {}

	print("processing training queries from {}".format(train_directory))
	train_data = _load_directory(train_directory, col2minmax, colid2featlen_per_table,
	                             max_files=num_train_q, with_subplans=include_subplans,
	                             saved_predicates=saved_predicates)

	print("processing test queries from {}".format(test_directory))
	test_data = _load_directory(test_directory, col2minmax, colid2featlen_per_table)

	template2queries, template2cards, template2pgcards, template2raw = \
		_group_by_table_tuple(*train_data)
	test_template2queries, test_template2cards, test_template2pgcards, test_template2raw = \
		_group_by_table_tuple(*test_data)

	print("training queries: {} across {} join templates".format(
		len(train_data[0]), len(template2queries)))
	print("test queries: {} across {} join templates".format(
		len(test_data[0]), len(test_template2queries)))

	_write_raw_csv(workload_file_path + 'train_raw_queries.csv', template2cards,
	               template2pgcards, template2raw)
	_write_raw_csv(workload_file_path + 'test_raw_queries.csv', test_template2cards,
	               test_template2pgcards, test_template2raw)

	result = (template2queries, template2cards, template2pgcards, test_template2queries,
	          test_template2cards, test_template2pgcards, colid2featlen_per_table)

	for name, obj in zip(_CACHE_FILES, result):
		print("writing {}".format(name))
		with open(workload_file_path + name + '.pkl', "wb") as pickle_file:
			pickle.dump(obj, pickle_file)

	return result
