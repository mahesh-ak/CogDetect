import csv
import os

from code.prepare.lexstat import set_schema, make_wordlist, calc_lexstat
from code.prepare.feature7 import create_pandas_frame
from code.prepare.params import load_params
from code.prepare.pmi import get_asjp_data, prepare_lang_pair
from code.prepare.utils import make_sample_id, is_asjp_data, explode_sample_id



def prepare(dataset_path, params_dir):
	"""
	Calculates the features and targets for the given raw dataset and returns a
	pandas DataFrame containing the "prepared" data ready for SVM consumption.
	
	This function is a wrapper around the _prepare function (that does most of
	the work). The create_pandas_frame function takes care of feature7.
	"""
	samples, targets = _prepare(dataset_path, params_dir)
	return create_pandas_frame(dataset_path, samples, targets)



def _prepare(dataset_path, params_dir):
	"""
	Returns the samples and targets found in the dataset.
	
	The samples are {sample_id: [feature1, feature2,..]} for features 1-6 and
	the LexStat features for all sample IDs in the dataset.
	
	The targets are {sample_id: target} for all sample IDs in the samples {}.
	"""
	samples = {}  # sample_id: [feature1, ..., feature7]
	targets = {}  # sample_id: target
	params = load_params(params_dir)
	data = load_data(dataset_path)
	data_asjp = get_asjp_data(data, params)

	lang_pairs = [(a, b) for a in data.keys() for b in data.keys() if a < b]
	
	# pmi features
	for lang1, lang2 in lang_pairs:
		samples.update(prepare_lang_pair(lang1, lang2, data_asjp, params))
	gloss_len = get_average_gloss_len(data_asjp)
	for key, sample in samples.items():
		sample.append(gloss_len[key.split('/')[0]])
	# lexstat features
	schema = 'asjp' if is_asjp_data(data) else 'ipa'
	with set_schema(schema):
		lingpy_wordlist = make_wordlist(data, dataset_path, schema)
		
		for lang1, lang2 in lang_pairs:
			scores = calc_lexstat(lang1, lang2, lingpy_wordlist)
			for key, score in scores.items():
				assert key in samples
				samples[key].extend(list(score))
	
	# targets
	try:
		targets = load_targets(dataset_path, samples.keys(), data.keys())
	except:
		print((
			'Targets could not be loaded. '
			'But samples are OK, do not worry.'
		))
		targets = {}
	
	return samples, targets



def load_data(dataset_path):
	"""
	Extracts the relevant data from the dataset given.
	Returns {lang: {gloss: [transcription,]}}.
	
	Asserts that there are no entries with unknown or no transcriptions.
	"""
	data = {}  # lang: {gloss: [transcription,]}
	
	with open(dataset_path) as f:
		reader = csv.reader(f, delimiter='\t')
		next(reader)
		for line in reader:
			if line[0] not in data:
				data[line[0]] = {}
			if line[3] not in data[line[0]]:
				data[line[0]][line[3]]  = []
			assert line[5] not in ('', 'XXX')
			data[line[0]][line[3]].append(line[5])
	
	return data



def load_targets(dataset_path, keys, langs):
	"""
	Returns {pair_id: True/False}.
	"""
	langs = set(langs)
	
	data = {}  # {gloss: {lang: [cognate_class,]}}
	targets = {}  # {pair_id: True/False}
	
	with open(dataset_path) as f:
		reader = csv.reader(f, delimiter='\t')
		next(reader)
		for line in reader:
			if line[3] not in data:
				data[line[3]] = {}
			if line[0] not in data[line[3]]:
				data[line[3]][line[0]] = []
			data[line[3]][line[0]].append(line[6])
	
	for key in keys:
		gloss, lang1, lang2, key1, key2 = explode_sample_id(key, langs)
		targets[key] = data[gloss][lang1][key1] == data[gloss][lang2][key2]
	
	assert len(targets) == len(keys)
	assert set(targets.keys()) == set(keys)
	
	return targets



def get_average_gloss_len(data):
	"""
	Returns {gloss: average_gloss_length}.
	Expects {lang: {gloss: [transcription,]}}.
	"""
	glosses = set([gloss for lang in data for gloss in data[lang].keys()])
	gloss_len = dict.fromkeys(glosses, (0, 0,))
	
	for lang in data.keys():
		for gloss in data[lang].keys():
			for trans in data[lang][gloss]:
				t = gloss_len[gloss]
				gloss_len[gloss] = (t[0]+len(trans), t[1]+1,)
	
	gloss_len = {key: t[0]/t[1] for key, t in gloss_len.items()}
	
	return gloss_len



def write_samples(samples, dataset_name, output_dir):
	"""
	Writes the samples in .tsv format.
	
	The entries are ordered by the sample ID in order to make differences
	easily git-diff-able.
	"""
	samples_dir = os.path.join(output_dir, 'samples')
	if not os.path.exists(samples_dir):
		os.mkdir(samples_dir)
	
	file_path = os.path.join(samples_dir, dataset_name +'.tsv')
	
	with open(file_path, 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow([
			'sample_id', 'feature1', 'feature2', 'feature3', 'feature4',
			'feature5', 'feature6',
			'lexstat_simAA', 'lexstat_simBB', 'lexstat_simAB'
		])
		for key in sorted(samples.keys()):
			writer.writerow([key] + samples[key])



def write_targets(targets, dataset_name, output_dir):
	"""
	Writes the targets in .tsv format.
	
	The entries are ordered by the sample ID in order to make differences
	easily git-diff-able.
	"""
	targets_dir = os.path.join(output_dir, 'targets')
	if not os.path.exists(targets_dir):
		os.mkdir(targets_dir)
	
	file_path = os.path.join(targets_dir, dataset_name +'.tsv')
	
	with open(file_path, 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(['sample_id', 'target'])
		for key in sorted(targets.keys()):
			writer.writerow([key, int(targets[key])])



def write(frame, dataset_name, output_dir):
	"""
	Writes the given pandas DataFrame into a file with the given name in the
	given directory.
	"""
	file_path = os.path.join(output_dir, dataset_name +'.csv')
	frame.to_csv(file_path, index=False, float_format='%.10f')
