import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _load_config():
	"""Load configuration from the YAML file located in the plot/etc folder.
	The function is robust to tabs in the file and accepts either .yml or .yaml.
	"""
	base_dir = os.path.dirname(__file__)
	candidates = [os.path.join(base_dir, 'etc', 'config_plot.yml'), os.path.join(base_dir, 'etc', 'config_plot.yaml')]
	cfg_path = None
	for c in candidates:
		if os.path.exists(c):
			cfg_path = c
			break
	if cfg_path is None:
		raise FileNotFoundError('config_plot.yml or config_plot.yaml not found in plot/etc')

	with open(cfg_path, 'r', encoding='utf-8') as fh:
		raw = fh.read()

	# Replace tabs with two spaces (YAML forbids tabs for indentation)
	if '\t' in raw:
		raw = raw.replace('\t', '  ')

	config = yaml.safe_load(raw)
	return config


def plot_forecast_data(fcast_data: pd.DataFrame, variable: str) -> None:
	"""Plot deterministic forecast data for a given variable.
	Args:
		fcast_data (pd.DataFrame): DataFrame returned by `get_det_forecast_data`.
		variable (str): Short variable name as used in column names (e.g. 't2m').
	Returns:
		None
	"""
	config = _load_config()
	dict_name_vars = config['dict_name_vars']
	dict_name_units = config['dict_name_units']

	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
	colors = plt.cm.tab20(range(len(fcast_data['model'].unique())))

	try:
		fcast_data = fcast_data[[f'{variable}', 'forecast_date', 'model', 'locality', 'latitude', 'longitude']]
		# fcast_data = fcast_data.dropna(subset=[f'{variable}'])
		# fcast_data = fcast_data.drop_duplicates(subset=[f'{variable}']) #caution with duplicates with some variables, the api may return repeated values
	except KeyError:
		print(f"Variable '{variable}' not found in forecast data. Available variables: {fcast_data.columns.tolist()}")
		return
	
	# Plot each model's data
	for idx, model in enumerate(fcast_data['model'].unique()):
		data_model = fcast_data[fcast_data['model'] == model]
		axs.plot(data_model['forecast_date'], data_model[f'{variable}'], label=model, color=colors[idx])

	# Set title and labels
	n_models = len(fcast_data['model'].unique())
	loc_str = fcast_data['locality'].iloc[0]
	variable_name = dict_name_vars.get(variable, variable)
	variable_unit = dict_name_units.get(variable, '')

	plt.suptitle(f'{loc_str} {variable_name} Forecast.\n{fcast_data["forecast_date"].min()}', fontweight='bold', fontsize=14)
	plt.title(f'Latitude: {fcast_data["latitude"].iloc[0]:.2f}°N, Longitude: {fcast_data["longitude"].iloc[0]:.2f}°E', fontsize=8, loc='left')
	plt.title(f'{n_models} Deterministic Models used', fontsize=8, loc='center')
	plt.title('Powered by Open-Meteo. Plot by Iván Domínguez Fuentes', fontsize=8, loc='right')

	axs.set_xlabel('Date (UTC)')
	axs.set_ylabel(f'{variable_name.capitalize()} {variable_unit}')
	axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
	axs.grid(True, alpha=0.3)

	# Set x-ticks to show dates by day
	axs.set_xticks(pd.date_range(start=fcast_data['forecast_date'].min(), end=fcast_data['forecast_date'].max(), freq='D'))
	axs.set_xticklabels(pd.date_range(start=fcast_data['forecast_date'].min(), end=fcast_data['forecast_date'].max(), freq='D').date, rotation=45)
	axs.set_xlim(fcast_data['forecast_date'].min(), fcast_data['forecast_date'].max())
	plt.tight_layout()
	return fig


def plot_ens_boxplot(ens_fcst_data: pd.DataFrame, variable: str, max_boxes: int = 50) -> None:
	"""
	Plot ensemble forecast data as boxplots for a given variable.
	Args:
		ens_fcst_data (pd.DataFrame): DataFrame returned by `get_ens_forecast_data`.
		variable (str): Short variable name as used in column names (e.g. 't2m').
		max_boxes (int, optional): Maximum number of boxplots to display. Defaults to 50.
	Returns:
		None
	"""
	config = _load_config()
	dict_name_vars = config['dict_name_vars']
	dict_name_units = config['dict_name_units']

	# Select ensemble member columns for the requested variable
	cols = ens_fcst_data.columns[ens_fcst_data.columns.str.contains(variable)]
	if cols.empty:
		print(f"No ensemble columns found for variable '{variable}'.")
		return

	data = ens_fcst_data.loc[:, cols].copy()
	data = data.dropna(axis=1)
	dates = ens_fcst_data['forecast_date'].reset_index(drop=True)
	n = len(data)
	if n == 0:
		print("No data rows available to plot.")
		return

	# If too many time steps, sample evenly to at most `max_boxes`
	if n > max_boxes:
		indices = np.linspace(0, n - 1, max_boxes, dtype=int)
	else:
		indices = np.arange(n)

	# Compute percentiles across ensemble members for each selected time step
	q10 = data.quantile(0.10, axis=1).reset_index(drop=True)
	q25 = data.quantile(0.25, axis=1).reset_index(drop=True)
	q50 = data.quantile(0.50, axis=1).reset_index(drop=True)
	q75 = data.quantile(0.75, axis=1).reset_index(drop=True)
	q90 = data.quantile(0.90, axis=1).reset_index(drop=True)
	minv = data.min(axis=1).reset_index(drop=True)
	maxv = data.max(axis=1).reset_index(drop=True)

	# Prepare dicts for matplotlib.bxp
	bxp_stats = []
	labels = []
	for i in indices:
		bxp_stats.append({
			'Q1': float(q25.iloc[i]),
			'Q2': float(q50.iloc[i]),
			'Q3': float(q75.iloc[i]),
			'whislo': float(q10.iloc[i]),
			'whishi': float(q90.iloc[i]),
			'med': float(q50.iloc[i]),
			'label': pd.to_datetime(dates.iloc[i]).strftime('%Y-%m-%d %H:%M')
		})
		labels.append(bxp_stats[-1]['label'])

	# Matplotlib expects keys: med, q1, q3, whislo, whishi — map accordingly
	for d in bxp_stats:
		d['med'] = d.pop('med')
		d['q1'] = d.pop('Q1')
		d['q3'] = d.pop('Q3')

	fig, ax = plt.subplots(figsize=(max(8, len(bxp_stats) * 0.25), 6))
	ax.bxp(bxp_stats, showmeans=False, showfliers=False)

	# Overlay ensemble mean as a line for the sampled indices
	ens_mean = data.mean(axis=1).reset_index(drop=True)
	ax.plot(range(1, len(indices) + 1), ens_mean.iloc[indices], color='black', linewidth=1.5, label='Ensemble Mean')

	# Set major ticks (dates) and minor ticks (hours)
	major_ticks = []
	major_labels = []
	dates_seen = set()
	
	for i, idx in enumerate(indices):
		date = pd.to_datetime(dates.iloc[idx])
		date_only = date.date()
		if date_only not in dates_seen:
			major_ticks.append(i + 1)
			major_labels.append(date_only.strftime('%Y-%m-%d'))
			dates_seen.add(date_only)
	
	ax.set_xticks(major_ticks)
	ax.set_xticklabels(major_labels, rotation=45, ha='center')
	ax.set_xticks(range(1, len(indices) + 1), minor=True)
	ax.tick_params(axis='x', which='minor', length=3, labelsize=8)
	ax.grid(True, which='both', axis='both', alpha=0.3)

	variable_name = dict_name_vars.get(variable, variable)
	variable_unit = dict_name_units.get(variable, '')
	n_models = data.columns.size
	lat = ens_fcst_data['latitude'].iloc[0]
	lon = ens_fcst_data['longitude'].iloc[0]

	plt.suptitle(f"{ens_fcst_data['locality'].iloc[0]} {variable_name} Ensemble distribution", fontsize=14, fontweight='bold')
	plt.title(f'Latitude: {lat:.2f}°N, Longitude: {lon:.2f}°E', fontsize=8, loc='left')
	plt.title(f'{n_models} Ensemble Model scenarios used', fontsize=8, loc='center')
	plt.title('Powered by Open-Meteo. Plot by Iván Domínguez Fuentes', fontsize=8, loc='right')
	plt.xlabel('Forecast Date (UTC)')
	plt.ylabel(variable_name.capitalize() + ' ' + variable_unit)
	ax.legend()
	plt.tight_layout()
	return fig

def plot_ens_forecast_data(ens_fcst_data: pd.DataFrame, variable: str, quantiles: bool = False) -> None:
	"""Plot ensemble forecast data for a given variable.
	Args:
		ens_fcst_data (pd.DataFrame): DataFrame returned by `get_ens_forecast_data`.
		variable (str): Short variable name as used in column names (e.g. 't2m').
		quantiles (bool, optional): Whether to plot quantile ranges. Defaults to False.
	Returns:
		None
	"""
	config = _load_config()
	dict_name_vars = config['dict_name_vars']
	dict_name_units = config['dict_name_units']

	lat = ens_fcst_data['latitude'].iloc[0]
	lon = ens_fcst_data['longitude'].iloc[0]
	loc_str = ens_fcst_data['locality'].iloc[0]
	data = ens_fcst_data.loc[:, ens_fcst_data.columns.str.contains(variable)].copy()
	data = data.dropna(axis=1)
	data['ens_mean'] = data.mean(axis=1)
	data['forecast_date'] = ens_fcst_data['forecast_date']

	if quantiles==True:
		data['ens_p10'] = data.drop(['forecast_date', 'ens_mean'], axis=1).quantile(0.1, axis=1)
		data['ens_p90'] = data.drop(['forecast_date', 'ens_mean'], axis=1).quantile(0.9, axis=1)
		data['ens_p25'] = data.drop(['forecast_date', 'ens_mean'], axis=1).quantile(0.25, axis=1)
		data['ens_p75'] = data.drop(['forecast_date', 'ens_mean'], axis=1).quantile(0.75, axis=1)
		data['ens_p40'] = data.drop(['forecast_date', 'ens_mean'], axis=1).quantile(0.4, axis=1)
		data['ens_p60'] = data.drop(['forecast_date', 'ens_mean'], axis=1).quantile(0.6, axis=1)
		data['ens_min'] = data.drop(['forecast_date', 'ens_mean'], axis=1).min(axis=1)
		data['ens_max'] = data.drop(['forecast_date', 'ens_mean'], axis=1).max(axis=1)

	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
	
	if quantiles==True:
		plt.fill_between(data['forecast_date'], data['ens_p10'], data['ens_p90'], color='gray', alpha=0.2, label='10-90 Percentile')
		plt.fill_between(data['forecast_date'], data['ens_p25'], data['ens_p75'], color='gray', alpha=0.4, label='25-75 Percentile')
		plt.fill_between(data['forecast_date'], data['ens_p40'], data['ens_p60'], color='gray', alpha=0.6, label='40-60 Percentile')
		plt.fill_between(data['forecast_date'], data['ens_min'], data['ens_max'], color='gray', alpha=0.1, label='Min-Max Range')
		plt.plot(data['forecast_date'], data['ens_mean'], color='black', linewidth=2, label='Ensemble Mean')
		
		axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
		axs.grid(True, which='both', alpha=0.3)
		n_models = data.drop(['forecast_date', 'ens_mean'], axis=1).columns.size
		variable_name = dict_name_vars.get(variable, variable)
		
		plt.suptitle(f'{loc_str} {variable_name} Forecast.\n{data["forecast_date"].min()}', fontweight='bold', fontsize=14)
		plt.title(f'Latitude: {lat:.2f}°N, Longitude: {lon:.2f}°E', fontsize=8, loc='left')
		plt.title(f'{n_models} Ensemble Model scenarios used', fontsize=8, loc='center')
		plt.title('Powered by Open-Meteo. Plot by Iván Domínguez Fuentes', fontsize=8, loc='right')
		plt.xlabel('Date (UTC)')
		plt.xlim(data['forecast_date'].min(), data['forecast_date'].max())
		plt.ylabel(f'{variable_name.capitalize()}')
		plt.xticks(pd.date_range(start=data['forecast_date'].min(), end=data['forecast_date'].max(), freq='D'), pd.date_range(start=data['forecast_date'].min(), end=data['forecast_date'].max(), freq='D').date)
		plt.xticks(rotation=45)
		plt.tight_layout()
		return fig
	else:
		data.drop(['ens_mean'], axis=1).set_index('forecast_date').plot(ax=axs, legend=False, grid=True)
		data[['forecast_date', 'ens_mean']].set_index('forecast_date').plot(ax=axs, color='black', linewidth=2, label='Ensemble Mean')

		axs.grid(True, which='both', alpha=0.3)
		n_models = data.drop(['forecast_date', 'ens_mean'], axis=1).columns.size
		variable_name = dict_name_vars.get(variable, variable)
		variable_unit = dict_name_units.get(variable, '')
		
		plt.suptitle(f'{loc_str} {variable_name} Forecast.\n{data["forecast_date"].min()}', fontweight='bold', fontsize=14)
		plt.title(f'Latitude: {lat:.2f}°N, Longitude: {lon:.2f}°E', fontsize=8, loc='left')
		plt.title(f'{n_models} Ensemble Model scenarios used', fontsize=8, loc='center')
		plt.title('Powered by Open-Meteo. Plot by Iván Domínguez Fuentes', fontsize=8, loc='right')
		plt.xlabel('Date (UTC)')
		plt.ylabel(f'{variable_name.capitalize()} {variable_unit}')
		plt.xticks(pd.date_range(start=data['forecast_date'].min(), end=data['forecast_date'].max(), freq='D'), pd.date_range(start=data['forecast_date'].min(), end=data['forecast_date'].max(), freq='D').date)
		plt.xticks(rotation=45)
		plt.tight_layout()
		return fig

def plot_cdf_on_date(ens_fcst_data: pd.DataFrame, variable: str, date: str):
	"""
	Plot the cumulative distribution function (CDF) of ensemble forecasts for a specific date.
	Args:
		ens_fcst_data (pd.DataFrame): DataFrame returned by `get_ens_forecast_data`.
		variable (str): Short variable name as used in column names (e.g. 't2m').
		date (str): Specific date in 'YYYY-MM-DD HH:MM:SS+00:00' format to plot the CDF for.
	Returns:
		None
	"""
	config = _load_config()
	dict_name_vars = config['dict_name_vars']
	dict_name_units = config['dict_name_units']

	data = ens_fcst_data.drop(['forecast_date', 'latitude', 'longitude', 'locality'], axis=1)
	data_var = data.loc[:, data.columns.str.contains(variable)]
	data_var_clean = data_var.dropna(axis=1)
	quantiles = data_var_clean.quantile(q=np.arange(0,1.05,0.05), axis=1).reset_index(drop=True)

	quantiles = quantiles.T
	quantiles = quantiles.copy()
	quantiles['forecast_date'] = ens_fcst_data['forecast_date']
	q=np.arange(0,1.05,0.05)
	col = ['P' + str(int(100*i)) for i in q]
	quantiles = quantiles.set_index('forecast_date')
	quantiles.columns = col

	lat = ens_fcst_data['latitude'].iloc[0]
	lon = ens_fcst_data['longitude'].iloc[0]
	locality = ens_fcst_data['locality'].iloc[0]
	n_models = data_var_clean.shape[1]

	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
	quantiles[quantiles.index == date].T.plot(ax=axs, grid=True, lw=3)
	plt.suptitle(f'{locality}\n{dict_name_vars.get(variable, "")} Ensemble Percentiles on {date} UTC', fontweight='bold', fontsize=14)
	plt.title(f'Latitude: {lat:.2f}°N, Longitude: {lon:.2f}°E', fontsize=8, loc='left')
	plt.title(f'{n_models} Ensemble Model scenarios used', fontsize=8, loc='center')
	plt.title('Powered by Open-Meteo. Plot by Iván Domínguez Fuentes', fontsize=8, loc='right')
	plt.xlabel('Percentiles')
	axs.set_xticks(range(len(quantiles.columns)))
	axs.set_xticklabels(quantiles.columns.to_list())
	plt.xlim(left=0, right=len(quantiles.columns)-1)
	plt.ylabel(f'{dict_name_vars.get(variable, "")} {dict_name_units.get(variable, "")}')
	plt.tight_layout()
	return fig
