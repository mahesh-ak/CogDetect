from collections import defaultdict

import csv
import os.path
import tempfile

from numpy import *

import pandas as pd



def create_pandas_frame(dataset_path, samples, targets):
	"""
	Returns a pandas DataFrame object containing the prepared data. This is a
	wrapper that prepares the given samples and targets for consumption by the
	_create_pandas_frame function and returns the output of the latter.
	"""
	temp_dir = tempfile.TemporaryDirectory()
	
	samples_path = os.path.join(temp_dir.name, 'samples.tsv')
	targets_path = os.path.join(temp_dir.name, 'targets.tsv')
	
	samples_frame = pd.DataFrame([
		[key] + samples[key] for key in sorted(samples.keys())
		], columns=[
		'sample_id', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
		'feature6', 'lexstat_simAA', 'lexstat_simBB', 'lexstat_simAB'])
	samples_frame.to_csv(samples_path, sep='\t', index=False)
	
	with open(targets_path, 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(['sample_id', 'target'])
		for key in sorted(targets.keys()):
			writer.writerow([key, int(targets[key])])
	
	frame = _create_pandas_frame(dataset_path, samples_path, targets_path)
	
	temp_dir.cleanup()
	
	return frame



def _create_pandas_frame(dataset_path, samples_path, targets_path):
	"""
	Creates and returns a pandas DataFrame object that includes the dataset's
	samples and targets. Also, the samples are augmented by calculating and
	adding the feature7 column.
	
	Note that the function requires paths as arguments instead of the data
	itself (which is why the temp dir is create in the calling add_feature7).
	This is for reasons that were once reasonable.
	"""
	fname = dataset_path.split('/')[-1]
	db = fname.split('.')[0]
	# read in wordlist
	wordlist = pd.read_table(dataset_path,encoding='utf-8',na_filter=False,dtype=object)
	# keep track of synonyms within the same language
	synDict = defaultdict(lambda: 0)
	synocc = []
	for l,g in wordlist[['language','global_id']].values:
		synDict[l,g] += 1
		synocc.append(unicode(synDict[l,g]))
	wordlist['synonym_number'] = synocc
	dDict = {'sample_id':unicode,
				'feature1':double,
				'feature2':double,
				'feature3':double,
				'feature4':double,
				'feature5':double,
				'feature6':double,
				'feature8':double}
	# read in feature matrix for word pairs
	vectors = pd.read_table(samples_path,
							encoding='utf-8',na_filter=False,dtype=dDict)
	# read in cognacy judgments
	labels = pd.read_table(targets_path,
							encoding='utf-8',na_filter=False,dtype={'sample_id':unicode,
																	'target':int})
	# colect metadata for wordpairs in vectors
	metaRaw = array([x.split('/') for x in vectors.sample_id.values])
	meta = pd.DataFrame(c_[metaRaw[:,0],
							[x.split(',') for x in metaRaw[:,1]],
							[x.split(',') for x in metaRaw[:,2]]],
						columns=['global_id','l1','l2','id1','id2'])
	meta['sample_id'] = vectors.sample_id
	meta1 = pd.merge(wordlist[['global_id','language','gloss','synonym_number',
								'transcription','cognate_class']],
						meta,
						left_on=['global_id','language','synonym_number'],
						right_on=['global_id','l1','id1'])[['sample_id',
															'global_id',
															'l1','l2',
															'transcription',
															'cognate_class',
															'id2']]
	meta2 = pd.merge(wordlist[['global_id','language','gloss','synonym_number',
								'transcription','cognate_class']],
						meta1,
						left_on=['global_id','language','synonym_number'],
						right_on=['global_id','l2','id2'])[['sample_id',
															'gloss',
															'l1','transcription_y',
															'cognate_class_y',
															'l2','transcription_x',
															'cognate_class_x']]
	meta2.columns = ['sample_id',u'gloss', 'l1', u'w1', u'cc1', 'l2',
						u'w2', u'cc2']
	meta2 = meta2.ix[pd.match(vectors.sample_id,meta2.sample_id)]
	concepts = meta2.gloss.unique()
	feature7 = pd.Series([abs(corrcoef(array(vectors[meta2.gloss==c][['feature2',
																		'feature4']].values,
												double).T)[0,1])
							for c in concepts],
							index=concepts,dtype=double)
	feature7[feature7.isnull()] = 0
	vectors['feature7'] = feature7.ix[meta2.gloss.values].values
	combined = pd.merge(pd.merge(meta2,vectors,on='sample_id'),
						labels,on='sample_id')
	combined = combined[combined.columns[1:]]
	combined['db'] = db
	
	return combined
