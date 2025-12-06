import requests
import pandas as pd
import requests_cache
import openmeteo_requests
from retry_requests import retry
from openmeteo_sdk.Variable import Variable

# ------------------------FUNCTIONS------------------------

def _setup_cache_openmeteo(expire_after: int = 3600, retries: int = 5, backoff_factor: float = 0.2) -> openmeteo_requests.Client:
	"""Setup cached Open-Meteo API client with retry on error.
	Args:
		expire_after (int, optional): Cache expiration time in seconds. Defaults to 3600.
		retries (int, optional): Number of retries on request failure. Defaults to 5.
		backoff_factor (float, optional): Backoff factor for retries. Defaults to 0.2.
	Returns:
		openmeteo_requests.Client: Configured Open-Meteo API client.
	"""
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = expire_after)
	retry_session = retry(cache_session, retries = retries, backoff_factor = backoff_factor)
	openmeteo = openmeteo_requests.Client(session = retry_session)
	return openmeteo

def _get_coordinates_from_locality(locality: str) -> tuple[float, float, str] | None:
    """
    Given a certain locality name in string format, get
    latitude, longitude and the response location from nominatim openstreetmap API.
    """
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": locality,
        "format": "jsonv2",
        "limit": 1,
        # Opcional pero recomendable:
        # "email": "ivan.dominguez@wdna.com",  # cámbialo por el tuyo real
    }

    headers = {
        # IMPORTANTE: identificar tu app, no simular Chrome
        "User-Agent": "pointweather-app"
    }

    response = requests.get(base_url, params=params, headers=headers, timeout=10)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.reason}")
        print(response.text)
        return None

    data = response.json()
    if not data:
        print("Can't get location data. Check response from server.")
        return None

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    location_api = data[0]["display_name"]
    print("Location data retrieved")
    return lat, lon, location_api

def _get_params_req(lat: float,
					lon: float,
					models: list,
					variable: str = 'temperature_2m',
					fcst_days: int = 14,
					temp_res: str = 'native') -> dict:
	"""Get parameters for Open-Meteo API request.
	Args:
		lat (float): Latitude of the location.
		lon (float): Longitude of the location.
		models (list): List of model names to request.
		variable (str, optional): The name of the variable to request. Defaults to 'temperature_2m'.
		fcst_days (int, optional): The number of forecast days to request. Defaults to 14.
		temp_res (str, optional): The temporal resolution of the forecast. Defaults to 'native'.
	Returns:
		dict: Dictionary of parameters for Open-Meteo API request.
	"""
	return {
		"latitude": lat,
		"longitude": lon,
		"hourly": variable,
		"models": models,
		"forecast_days": fcst_days,
		"temporal_resolution": temp_res,
	}

def _get_df_response_models(responses: list, models: list) -> pd.DataFrame:
	"""Get a DataFrame mapping responses to models.

	Args:
		responses (list): List of responses from Open-Meteo API.
		models (list): List of model names to request.

	Returns:
		pd.DataFrame: DataFrame with columns 'response' and 'model'.
	"""
	df_response_models = pd.DataFrame([responses, models]).T
	df_response_models.rename(columns={0: 'response', 1: 'model'}, inplace=True)
	return df_response_models

def get_det_forecast_data(locality: str,
						  models: list,
						  variables: list,
						  url_det: str,
						  fcast_days: int = 14,
						  temp_res: str='native') -> pd.DataFrame:

	"""Get deterministic forecast data for a given locality and models.
	Args:
		locality (str): Name of the locality (e.g. 'Barcelona').
		models (list): List of model names to request.
		variables (list): List of weather variables to request.
		fcast_days (int, optional): Number of forecast days to request. Defaults to 14.
		temp_res (str, optional): Temporal resolution for hourly data. Defaults to 'native'.

	Returns:
		pd.DataFrame: DataFrame with columns 'response' and 'model'.
	"""

	openmeteo = _setup_cache_openmeteo()
	lat, lon, loc_str = _get_coordinates_from_locality(locality)
	params = _get_params_req(lat, lon, models, variables, fcast_days, temp_res)
	responses = openmeteo.weather_api(url_det, params=params)
	df_response_models = _get_df_response_models(responses, models)

	dfs_pred = []
	for response in df_response_models['response'].dropna():

		try:
			print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
			print(f"Elevation: {response.Elevation()} m asl")
			print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
			print(f"Model Nº: {response.Model()}")
			
			# Process hourly data. The order of variables needs to be the same as requested.
			hourly = response.Hourly()
			hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
			hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
			hourly_snowfall = hourly.Variables(2).ValuesAsNumpy()
			hourly_pressure_msl = hourly.Variables(3).ValuesAsNumpy()
			hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
			hourly_wind_gusts_10m = hourly.Variables(5).ValuesAsNumpy()
			
			hourly_data = {"forecast_date": pd.date_range(
				start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
				end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
				freq = pd.Timedelta(seconds = hourly.Interval()),
				inclusive = "left"
			)}
			
			hourly_data["t2m"] = hourly_temperature_2m
			hourly_data["pr"] = hourly_precipitation
			hourly_data["prs"] = hourly_snowfall
			hourly_data["mslp"] = hourly_pressure_msl
			hourly_data["ws10"] = hourly_wind_speed_10m
			hourly_data["wg10"] = hourly_wind_gusts_10m
			hourly_data["cape"] = hourly.Variables(6).ValuesAsNumpy()
			hourly_data["t850"] = hourly.Variables(7).ValuesAsNumpy()
			hourly_data["t500"] = hourly.Variables(8).ValuesAsNumpy()
			hourly_data["gh500"] = hourly.Variables(9).ValuesAsNumpy()

			model = df_response_models[df_response_models['response'] == response]['model'].values[0]
			hourly_data['model'] = model
			
			hourly_dataframe = pd.DataFrame(data = hourly_data)
			dfs_pred.append(hourly_dataframe)
			print("\nHourly data\n", hourly_dataframe)
		except Exception as e:
			print(f"An error occurred while processing the response: {e}")

	df_all_models = pd.concat(dfs_pred, ignore_index = True)
	df_all_models['locality'] = loc_str
	df_all_models['latitude'] = lat
	df_all_models['longitude'] = lon

	return df_all_models

def get_ens_forecast_data(locality: str,
						  models: list,
						  variables: list,
						  url_ens: str,
						  ens_models: list,
						  fcast_days: int = 14,
						  temp_res: str='hourly_6') -> pd.DataFrame:

	"""Get ensemble forecast data for a given locality and models.
	Args:
		locality (str): Name of the locality (e.g. 'Barcelona').
		models (list): List of ensemble model names to request.
		variables (list): List of weather variables to request.
		fcast_days (int, optional): Number of forecast days to request. Defaults to 14.
		temp_res (str, optional): Temporal resolution for hourly data. Defaults to 'hourly_6'.
	Returns:
		pd.DataFrame: DataFrame with ensemble forecast data.
	"""

	# Setup the Open-Meteo API client with cache and retry on error
	openmeteo = _setup_cache_openmeteo()
	lat, lon, loc_str = _get_coordinates_from_locality(locality)
	params = _get_params_req(lat, lon, models, variables, fcast_days, temp_res)
	responses = openmeteo.weather_api(url_ens, params=params)
	df_response_models = _get_df_response_models(responses, ens_models)

	# Process 1 location and 13 models
	dfs_pred = []
	for response in df_response_models['response'].dropna():
		print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
		print(f"Elevation: {response.Elevation()} m asl")
		print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
		print(f"Model Nº: {response.Model()}")
		
		# Process hourly data. The order of variables needs to be the same as requested.
		hourly = response.Hourly()
		hourly_variables = list(map(lambda i: hourly.Variables(i), range(0, hourly.VariablesLength())))
		hourly_temperature_850hPa = filter(lambda x: x.Variable() == Variable.temperature and x.PressureLevel() == 850, hourly_variables)
		hourly_precipitation = filter(lambda x: x.Variable() == Variable.precipitation, hourly_variables)
		hourly_geopotential_height_500hPa = filter(lambda x: x.Variable() == Variable.geopotential_height and x.PressureLevel() == 500, hourly_variables)
		hourly_temperature_2m = filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, hourly_variables)
		hourly_pressure_msl = filter(lambda x: x.Variable() == Variable.pressure_msl, hourly_variables)
		hourly_wind_speed_10m = filter(lambda x: x.Variable() == Variable.wind_speed and x.Altitude() == 10, hourly_variables)
		hourly_wind_gusts_10m = filter(lambda x: x.Variable() == Variable.wind_gusts and x.Altitude() == 10, hourly_variables)
		hourly_temperature_500hPa = filter(lambda x: x.Variable() == Variable.temperature and x.PressureLevel() == 500, hourly_variables)
		hourly_cape = filter(lambda x: x.Variable() == Variable.cape, hourly_variables)
		
		hourly_data = {"forecast_date": pd.date_range(
			start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
			end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
			freq = pd.Timedelta(seconds = hourly.Interval()),
			inclusive = "left"
		)}
		
		# Process all hourly members
		model = df_response_models[df_response_models['response'] == response]['model'].values[0]
		
		for variable in hourly_temperature_850hPa:
			member = variable.EnsembleMember()
			hourly_data[f"t850_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_precipitation:
			member = variable.EnsembleMember()
			hourly_data[f"pr_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_geopotential_height_500hPa:
			member = variable.EnsembleMember()
			hourly_data[f"gh500_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_temperature_2m:
			member = variable.EnsembleMember()
			hourly_data[f"t2m_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_pressure_msl:
			member = variable.EnsembleMember()
			hourly_data[f"mslp_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_wind_speed_10m:
			member = variable.EnsembleMember()
			hourly_data[f"ws10_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_wind_gusts_10m:
			member = variable.EnsembleMember()
			hourly_data[f"wg10_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_temperature_500hPa:
			member = variable.EnsembleMember()
			hourly_data[f"t500_m_{member}_{model}"] = variable.ValuesAsNumpy()
		for variable in hourly_cape:
			member = variable.EnsembleMember()
			hourly_data[f"cape_m_{member}_{model}"] = variable.ValuesAsNumpy()
		
		hourly_dataframe = pd.DataFrame(data = hourly_data)
		dfs_pred.append(hourly_dataframe)
		print("\nHourly data\n", hourly_dataframe)

	dfs_pred = pd.concat(dfs_pred, axis=1)
	dfs_pred = dfs_pred.loc[:, ~dfs_pred.columns.duplicated()]
	dfs_pred['forecast_date'] = dfs_pred['forecast_date']
	dfs_pred['locality'] = loc_str
	dfs_pred['latitude'] = lat
	dfs_pred['longitude'] = lon

	return dfs_pred



#------------------------MAIN PROGRAM------------------------

def download_point_forecast(locality: str,
						 det_models: list,
						 ens_models: list,
						 variables: list,
						 url_det: str,
						 url_ens: str) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Download point forecast data for a given locality.
	Args:
		locality (str): Name of the locality (e.g. 'Barcelona').
		det_models (list): List of deterministic model names to request.
		ens_models (list): List of ensemble model names to request.
		variables (list): List of weather variables to request.
		url_det (str): URL for deterministic forecast API.
		url_ens (str): URL for ensemble forecast API.
	Returns:
		tuple[pd.DataFrame, pd.DataFrame]: Tuple containing deterministic and ensemble forecast DataFrames.
	"""
	fcst_data = get_det_forecast_data(locality,
								   models=det_models,
								   variables=variables,
								   url_det=url_det,
								   fcast_days=14,
								   temp_res='hourly_6')
	ens_fcst_data = get_ens_forecast_data(locality,
									   models=ens_models,
									   variables=variables,
									   url_ens=url_ens,
									   fcast_days=14,
									   temp_res='hourly_6')
	return fcst_data, ens_fcst_data


