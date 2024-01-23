import os.path



def find_dataset(this_dir, dataset_name):
	"""
	Returns the absolute path to the dataset.
	Returns None if the dataset is not found in the given dir.
	"""
	this_dir = os.path.abspath(this_dir)
	
	for file_name in os.listdir(this_dir):
		file_path = os.path.join(this_dir, file_name)
		
		if os.path.isdir(file_path) and file_name != 'input':
			res = find_dataset(file_path, dataset_name)
			if res is not None:
				return res
		
		if os.path.isfile(file_path):
			if file_name == dataset_name + '.tsv':
				return file_path
	
	return None



def find_all_datasets(this_dir):
	"""
	Returns [] of the absolute paths of all the datasets found in the dir.
	"""
	paths = []
	
	for file_name in os.listdir(this_dir):
		file_path = os.path.join(this_dir, file_name)
		
		if os.path.isdir(file_path):
			paths.extend(find_all_datasets(file_path))
		
		if os.path.isfile(file_path):
			if file_name.endswith('.tsv'):
				paths.append(file_path)
	
	return list(sorted(paths))



def get_dataset_name(dataset_path):
	"""
	Returns the name of the dataset.
	"""
	assert os.path.exists(dataset_path)
	file_name = os.path.basename(dataset_path)
	if file_name.endswith('.tsv'):
		return file_name[:-4]
