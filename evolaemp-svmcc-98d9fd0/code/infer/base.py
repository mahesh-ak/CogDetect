from collections import defaultdict
from multiprocessing import Pool

import os.path
import random

import igraph

from numpy import *

import numpy.random as nprandom
import pandas as pd

from sklearn.metrics import adjusted_rand_score
from sklearn import svm



"""
The names of the datasets used for training.
"""
TRAIN_SETS = ['bai', 'chinese_2004', 'japanese', 'ob_ugrian', 'an_train', 'iel_train']


"""
The names of the datasets used for testing. Note that central_asian is manually
split in two files because of file size limits.
"""
TEST_SETS = ['chinese_1964', 'tujia', 'huon', 'aa', 'an_test', 'bah', 'ie_test', 'pn', 'rom', 'st', 'ura']

SPLIT = 4

TRAIN_SETS += ["{0}_prop_50_{1}_train".format(t, SPLIT) for t in TEST_SETS]

TEST_SETS = ["{0}_prop_50_{1}_test".format(t, SPLIT) for t in TEST_SETS]
"""
The relevant subset of features; for feature selection, simply alter this list.
"""
FEATURES = ['feature1', 'feature4', 'feature6', 'feature7', 'feature8']



"""
Module-level variables, used within the workhorse functions.
"""
training = None
trainingVectors = None
test = None



def infer(vectors_dir, output_dir):
	"""
	Inits and orchestrates the cognate class inferring algorithm.
	"""
	global training
	global trainingVectors
	global test
	
	dDict = {'gloss':unicode,
		'l1':unicode, 'w1':unicode, 'cc1':unicode,
		'l2':unicode, 'w2':unicode, 'cc2':unicode,
		'feature1':double, 'feature2':double, 'feature3':double,
		'feature4':double, 'feature5':double,
		'lexstat_simAA':double, 'lexstat_simBB':double, 'lexstat_simAB':double,
		'feature7':double, 'target':int, 'db':unicode }
	
	# load the training data
	training = pd.DataFrame()
	
	for dataset_name in TRAIN_SETS:
		file_path = os.path.join(vectors_dir, '{}.csv'.format(dataset_name))
		training = training.append(pd.read_csv(file_path, encoding='utf-8', dtype=dDict))
	
	training['feature8'] = 1-((2*training.lexstat_simAB)/(training.lexstat_simAA+training.lexstat_simBB))
	
	nprandom.seed(1234)
	random.seed(1234)
	trainingVectors = training.ix[nprandom.permutation(training.index)].drop_duplicates(['db','gloss'])
	
	# cross-validation over training data
	pool = Pool()
	totalCC = pool.map(f,training.db.unique())
	pool.close()
	pool.terminate()
	
	for db,wl in zip(training.db.unique(),totalCC):
		file_path = os.path.join(output_dir, '{}.svmCC.csv'.format(db))
		wl['fullCC'] = [':'.join(x) for x in wl[['db','concept','cc']].values]
		wl[['db','concept','doculect','counterpart',
			'fullCC','inferredCC']].to_csv(file_path, encoding='utf-8', index=False)
	
	# load the test data
	test = pd.DataFrame()
	
	for dataset_name in TEST_SETS:
		file_path = os.path.join(vectors_dir, '{}.csv'.format(dataset_name))
		test = test.append(pd.read_csv(file_path, encoding='utf-8', dtype=dDict))
	
	test['feature8'] = 1-((2*test.lexstat_simAB)/(test.lexstat_simAA+test.lexstat_simBB))
	
	for db in test.db.unique():
		file_path = os.path.join(output_dir, '{}.svmCC.csv'.format(db))
		wl = testCluster(db)
		wl.to_csv(file_path, encoding='utf-8', index=False)



def f(x):
	return svmInfomapCluster(x)



def infomap_clustering(threshold, matrix, taxa=False, revert=False):
	"""
	Compute the Infomap clustering analysis of the data. Taken from LingPy's
	implementation of the algorithm.
	"""
	if not igraph:
		raise ValueError("The package igraph is needed to run this analysis.")
	if not taxa:
		taxa = list(range(1, len(matrix) + 1))

	G = igraph.Graph()
	vertex_weights = []
	for i in range(len(matrix)):
		G.add_vertex(i)
		vertex_weights += [0]

	# variable stores edge weights, if they are not there, the network is
	# already separated by the threshold
	for i,row in enumerate(matrix):
		for j,cell in enumerate(row):
			if i < j:
				if cell <= threshold:
					G.add_edge(i, j)
		
	comps = G.community_infomap(edge_weights=None,
			vertex_weights=None)
	D = {}
	for i,comp in enumerate(comps.subgraphs()):
		vertices = [v['name'] for v in comp.vs]
		for vertex in vertices:
			D[vertex] = i+1

	if revert:
		return D

	clr = defaultdict(list)
	for i,t in enumerate(taxa):
		clr[D[i]] += [t]
	return clr



def svmInfomapCluster(vdb,featureSubset=FEATURES,th=.34,C=.82,kernel='linear',gamma=1E-3):
	"""
	The first argument is the validation data base, the rest of the training
	databases are used for training.
	"""
	newWordList = pd.DataFrame()
	fitting = trainingVectors[trainingVectors.db!=vdb]
	validation = training[training.db==vdb].copy()
	X = fitting[featureSubset].values
	y = fitting.target.values
	svClf = svm.SVC(kernel=kernel,C=C,gamma=gamma,
					probability=True)
	svClf.fit(X,y)
	nprandom.seed(1234)
	random.seed(1234)
	svScores = svClf.predict_proba(validation[featureSubset].values)[:,1]
	validation['svScores'] = svScores
	scores = pd.DataFrame()
	wordlist = pd.DataFrame()
	concepts = validation.gloss.unique()
	taxa = unique(validation[['l1','l2']].values.flatten())
	dataWordlist = vstack([validation[['gloss','l1','w1','cc1']].values,
							validation[['gloss','l2','w2','cc2']].values])
	dataWordlist = pd.DataFrame(dataWordlist,columns=['concept','doculect',
														'counterpart','cc'])
	dataWordlist = dataWordlist.drop_duplicates()
	dataWordlist.index = ['_'.join(map(unicode,x))
							for x in
							dataWordlist[['concept','doculect','counterpart']].values]
	validation['id_1'] = [c+'_'+l+'_'+unicode(w)
						for (c,l,w) in validation[['gloss','l1','w1']].values]
	validation['id_2'] = [c+'_'+l+'_'+unicode(w)
						for (c,l,w) in validation[['gloss','l2','w2']].values]
	for c in concepts:
		dataC= validation[validation.gloss==c].copy()
		dataC['id_1'] = [x.replace(' ','').replace(',','') for x in dataC.id_1]
		dataC['id_2'] = [x.replace(' ','').replace(',','') for x in dataC.id_2]
		wlC = dataWordlist[dataWordlist.concept==c].copy()
		if len(wlC)>1:
			wlC.index = [x.replace(' ','').replace(',','') for x in wlC.index]
			svMtx = zeros((len(wlC.index),len(wlC.index)))
			svMtx[pd.match(dataC.id_1,wlC.index),
						pd.match(dataC.id_2,wlC.index)] = dataC.svScores.values
			svMtx[pd.match(dataC.id_2,wlC.index),
						pd.match(dataC.id_1,wlC.index)] = dataC.svScores.values
			svDistMtx = log(1-svMtx)
			tth = log(th)-svDistMtx.min()
			svDistMtx -= svDistMtx.min()
			fill_diagonal(svDistMtx,0)
			pDict = infomap_clustering(tth,svDistMtx)
			pArray = vstack([c_[pDict[k],[k]*len(pDict[k])] for k in pDict.keys()])
			partitionIM = pArray[argsort(pArray[:,0]),1]
		else:
			partitionIM = array([1])
		wlC['inferredCC'] = [vdb+':'+c+':'+str(x) for x in partitionIM]
		wlC['db'] = vdb
		newWordList = pd.concat([newWordList,wlC])
	newWordList.index = arange(len(newWordList))
	return newWordList



def testCluster(vdb,featureSubset=FEATURES,C=0.82,gamma=9e-04,kernel='linear',th=.34):
	"""
	Inference on test data.
	"""
	newWordList = pd.DataFrame()
	fitting = trainingVectors
	validation = test[test.db==vdb].copy()
	X = fitting[featureSubset].values
	y = fitting.target.values
	svClf = svm.SVC(kernel=kernel,C=C,gamma=gamma,
					probability=True)
	svClf.fit(X,y)
	svScores = svClf.predict_proba(validation[featureSubset].values)[:,1]
	validation['svScores'] = svScores
	scores = pd.DataFrame()
	wordlist = pd.DataFrame()
	concepts = validation.gloss.unique()
	taxa = unique(validation[['l1','l2']].values.flatten())
	dataWordlist = vstack([validation[['gloss','l1','w1','cc1']].values,
							validation[['gloss','l2','w2','cc2']].values])
	dataWordlist = pd.DataFrame(dataWordlist,columns=['concept','doculect',
														'counterpart','cc'])
	dataWordlist = dataWordlist.drop_duplicates()
	dataWordlist.index = ['_'.join(map(unicode,x))
							for x in
							dataWordlist[['concept','doculect','counterpart']].values]
	validation['id_1'] = [c+'_'+l+'_'+unicode(w)
						for (c,l,w) in validation[['gloss','l1','w1']].values]
	validation['id_2'] = [c+'_'+l+'_'+unicode(w)
						for (c,l,w) in validation[['gloss','l2','w2']].values]
	for c in concepts:
		dataC= validation[validation.gloss==c].copy()
		dataC['id_1'] = [x.replace(' ','').replace(',','') for x in dataC.id_1]
		dataC['id_2'] = [x.replace(' ','').replace(',','') for x in dataC.id_2]
		wlC = dataWordlist[dataWordlist.concept==c].copy()
		if len(wlC)>1:
			wlC.index = [x.replace(' ','').replace(',','') for x in wlC.index]
			svMtx = zeros((len(wlC.index),len(wlC.index)))
			svMtx[pd.match(dataC.id_1,wlC.index),
						pd.match(dataC.id_2,wlC.index)] = dataC.svScores.values
			svMtx[pd.match(dataC.id_2,wlC.index),
						pd.match(dataC.id_1,wlC.index)] = dataC.svScores.values
			svDistMtx = log(1-svMtx)
			tth = log(th)-svDistMtx.min()
			svDistMtx -= svDistMtx.min()
			fill_diagonal(svDistMtx,0)
			pDict = infomap_clustering(tth,svDistMtx)
			pArray = vstack([c_[pDict[k],[k]*len(pDict[k])] for k in pDict.keys()])
			partitionIM = pArray[argsort(pArray[:,0]),1]
		else:
			partitionIM = array([1])
		wlC['inferredCC'] = [vdb+':'+c+':'+str(x) for x in partitionIM]
		wlC['db'] = vdb
		newWordList = pd.concat([newWordList,wlC])
	newWordList.index = arange(len(newWordList))
	return newWordList
