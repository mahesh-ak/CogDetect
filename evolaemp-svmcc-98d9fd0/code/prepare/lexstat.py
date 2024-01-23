import contextlib
import csv
import random
random.seed(1234)

from lingpy.basic.wordlist import Wordlist
from lingpy.compare.lexstat import LexStat
from lingpy import log, rc
from lingpy.sequence.sound_classes import ipa2tokens, asjp2tokens

from code.prepare.utils import make_sample_id



@contextlib.contextmanager
def disable_info_logs():
	"""
	Provides context within which the lingpy logging level is set to higher
	than the default INFO level.
	"""
	log._logger = log.logging.getLogger('lingpy')
	log._logger.setLevel(log.logging.INFO+1)
	
	yield
	
	log._logger.setLevel(log.logging.INFO)



@contextlib.contextmanager
def set_schema(schema):
	"""
	Provides context within which the lingpy schema is set to one of (ASJP,
	IPA). The schema is reverted back to IPA afterwards.
	
	This is necessary because other modules expect the schema to be IPA.
	"""
	assert schema.lower() in ('asjp', 'ipa')
	
	with disable_info_logs():
		rc(schema=schema.lower())
	
	yield
	
	with disable_info_logs():
		rc(schema='ipa')



def make_wordlist(data, dataset_path, schema='ipa'):
	"""
	Expects {lang: {gloss: [ipa,]}}; returns a Wordlist instance.
	The last column of the header is needed for the sample ID.
	"""
	try:
		tokens = load_tokens(dataset_path, schema)
		assert len(tokens) == len(data)
	except AssertionError:
		raise ValueError('Could not find tokens in {}'.format(dataset_path))
	
	new_data = {}  # the data formatted as LexStat wants it
	new_data[0] = ['doculect', 'concept', 'ipa', 'index', 'tokens']  # header
	
	key = 1
	for lang in sorted(data.keys()):
		for gloss in sorted(data[lang].keys()):
			for index, ipa in enumerate(data[lang][gloss]):
				new_data[key] = [lang, gloss, ipa, index+1]
				new_data[key].append(tokens[lang][gloss][index])
				key += 1
	
	return Wordlist(new_data)



def load_tokens(dataset_path, schema):
	"""
	Returns {lang: {gloss: [tokens,]}} dict from the given dataset or raises
	AssertionError if there are no tokens.
	"""
	tokens = {}
	
	with open(dataset_path) as f:
		reader = csv.reader(f, delimiter='\t')
		header = next(reader)
		assert 'tokens' in header
		
		for line in reader:
			if line[0] not in tokens:
				tokens[line[0]] = {}
			if line[3] not in tokens[line[0]]:
				tokens[line[0]][line[3]]  = []
			tokens[line[0]][line[3]].append(line[7].split())
	
	return tokens



def filter_wordlist(wordlist, lang1, lang2):
	"""
	Expects and returns a Wordlist instance, with the returned one retaining
	only entries of the two languages given.
	"""
	new_data = {}  # the data formatted as LexStat wants it
	new_data[0] = ['doculect', 'concept', 'ipa', 'index', 'tokens']  # header
	
	key = 1
	for entry in wordlist._data.values():
		if entry[0] in (lang1, lang2):
			new_data[key] = entry
			key += 1
	
	return Wordlist(new_data)



def make_lexstat(wordlist, scorer_runs=1000):
	"""
	Expects a Wordlist instance; returns a LexStat instance.
	The optional argument is used to speed up unit testing.
	"""
	with disable_info_logs():
		lex = LexStat(wordlist)
		lex.get_scorer(runs=scorer_runs, preprocessing=False)
	
	return lex



def get_pairs(lang1, lang2, lex):
	"""
	Returns all the lang1-lang2 pairs of words with the same Concepticon ID.
	Returns [] of LexStat ID tuples.
	
	Note that LexStat.pairs cannot be used here because the latter flattens
	transcription duplicates.
	"""
	pairs = []
	
	for gloss1, indices1 in lex.get_dict(col=lang1).items():
		for gloss2, indices2 in lex.get_dict(col=lang2).items():
			if gloss1 == gloss2:
				pairs.extend([
					(i, j) for i in indices1 for j in indices2
				])
	
	return pairs



def calc_lexstat(lang1, lang2, wordlist):
	"""
	Expects two language names and a Wordlist instance.
	Returns {pair_id: (self-similarity1, self-similarity2, similarity)}.
	"""
	assert isinstance(wordlist, Wordlist)
	lex = make_lexstat(filter_wordlist(wordlist, lang1, lang2))
	scores = {}
	
	for p1, p2 in get_pairs(lang1, lang2, lex):
		line1, line2 = lex[p1], lex[p2]
		assert line1[1] == line2[1]
		
		sample_id = make_sample_id(line1[1], lang1, lang2, line1[3], line2[3])
		scores[sample_id] = (
			lex.align_pairs(p1, p1, pprint=False, distance=False)[2],
			lex.align_pairs(p2, p2, pprint=False, distance=False)[2],
			lex.align_pairs(p1, p2, pprint=False, distance=False)[2],
		)
	
	return scores
