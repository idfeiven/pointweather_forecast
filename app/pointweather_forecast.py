import yaml

def _load_config() -> list:
	"""Load configuration from YAML file.
	"""
	with open('etc/config_download.yaml', 'r') as file:
		config = yaml.safe_load(file)
	return config