import re

from lingpy.sequence.sound_classes import ipa2tokens, tokens2class



def make_sample_id(gloss_id, lang1, lang2, index1, index2):
	"""
	Sample IDs should uniquely identify a feature row.
	Sample sample ID: 98/English,German/1,1
	"""
	assert lang1 < lang2
	s  = str(gloss_id) + '/'
	s += lang1 +','+ lang2 + '/'
	s += str(index1) +','+ str(index2)
	return s



def explode_sample_id(sample_id, langs):
	"""
	Returns (gloss, lang1, lang2, index1, index2).
	Expects the set of all possible langs as second argument.
	
	Note: some datasets contain language names with chars such as: `/`, `,`.
	"""
	gloss = sample_id.split('/')[0]
	
	lang_part = sample_id.split('/', maxsplit=1)[1]
	lang_part = lang_part.rsplit('/', maxsplit=1)[0]
	
	for lang1 in langs:
		if lang_part.startswith(lang1+','):
			lang2 = lang_part[len(lang1)+1:]
			if lang2 in langs:
				break
	
	assert lang1 in langs
	assert lang2 in langs
	
	key1, key2 = sample_id.rsplit('/', maxsplit=1)[1].split(',')
	key1, key2 = int(key1) - 1, int(key2) - 1
	
	return gloss, lang1, lang2, key1, key2



def clean_asjp(word):
	"""
	Removes ASJP diacritics.
	"""
	word = re.sub(r",","-",word)
	word = re.sub(r"\%","",word)
	word = re.sub(r"\*","",word)
	word = re.sub(r"\"","",word)
	word = re.sub(r".~","",word)
	word = re.sub(r"(.)(.)(.)\$",r"\2",word)
	word = re.sub(r" ","-",word)
	return word



def ipa_to_asjp(w, params):
	"""
	Lingpy IPA-to-ASJP converter plus some cleanup.
	Expects the params {} to contain the key: sounds.
	
	This function is called on IPA datasets.
	"""
	w = w.replace('\"','').replace('-','').replace(' ','')
	wA = ''.join(tokens2class(ipa2tokens(w, merge_vowels=False), 'asjp'))
	wAA = clean_asjp(wA.replace('0','').replace('I','3').replace('H','N'))
	asjp = ''.join([x for x in wAA if x in params['sounds']])
	assert len(asjp) > 0
	return asjp



def asjp_to_asjp(w, params):
	"""
	Cleans up the ASJP string and filters it to include the chars specified in
	the sounds parameter.
	
	This function is called on ASJP datasets.
	"""
	w = w.replace('\"','').replace('-','').replace(' ','')
	wAA = clean_asjp(w.replace('0','').replace('I','3').replace('H','N'))
	asjp = ''.join([x for x in wAA if x in params['sounds']])
	assert len(asjp) > 0
	return asjp



def is_asjp_data(data):
	"""
	Expects {lang: {gloss: [transcription,]}}.
	Checks whether the translation strings are ASCII.
	"""
	return all([len(s.encode()) == len(s)
		for lang in data.values()
		for trans in lang.values()
		for s in trans
	])
