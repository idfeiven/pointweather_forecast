import requests
import pandas as pd
import urllib.parse
import matplotlib.pyplot as plt

#----------------------------------------ENSEMBLE MULTIMODEL METEOGRAM---------------------------------------
'''
Using several open api platforms, plot an ensemble multimodel meteogram for a specified locality
'''

#----------------------------------------------CONFIG-----------------------------------------------

locality = ['Palma'] #set location

variables_list = ['temperature_2m', 'precipitation']#['windgusts_10m', 'precipitation']#['temperature_850hPa', 'precipitation']  #choose atmospheric fields
forecast_days = 10 #set forecast time
ensemble_nwp_centers = ['gfs_seamless', 'gem_global', 'icon_seamless', 'ecmwf_ifs04']#, 'gem_global', 'icon_seamless'] #choose numerical weather prediction centers
oper_nwp_centers = ['gfs', 'gem', 'dwd-icon', 'ecmwf']
#---------------------------------------------FUNCTIONS----------------------------------------------

def get_coordinates_from_locality(locality):
    url = f'https://nominatim.openstreetmap.org/search.php?q={locality}&format=jsonv2'
    response = requests.get(url).json()
    lat = response[0]["lat"]
    lon = response[0]["lon"]
    location_api = response[0]["display_name"]
    return(lat, lon, location_api)

def get_str_variables(variables_list):
    str_variables = f''
    for variable in variables_list:
        str_variables = str_variables + ',' + variable
    str_variables = str_variables[1:]
    return(str_variables)

def get_url_ensemble_forecast(ensemble, str_variables, lat, lon):
    urls_api_ens = []
    urls_api_ens.append(f'https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&models={ensemble}&hourly={str_variables}&forecast_days={forecast_days}')
    return(urls_api_ens)

def get_url_oper_forecast(oper_nwp, str_variables, lat, lon):
    urls_api_oper = []
    urls_api_oper.append(f'https://api.open-meteo.com/v1/{oper_nwp}?latitude={lat}&longitude={lon}&hourly={str_variables}&forecast_days={forecast_days}')
    return(urls_api_oper)

def get_point_forecast_data(urls_api):
    # df_fc_data = pd.DataFrame()
    for url in urls_api:
        response = requests.get(url)
        if response.status_code == 200: #successful request
            response_json = response.json()
            df_fc_data = pd.json_normalize(response_json, sep = '_')
            # df_fc_data = pd.concat([df_fc_data, df])
        if response.status_code == 400:
            response_json = response.json()
            print(response_json['reason'])
    return(df_fc_data)

def get_parsed_df_fc_data(df_fc_data, variables_list):

    cols_vars = list()
    for variable in variables_list:
        df_fc_data = df_fc_data[df_fc_data.columns.drop(list(df_fc_data.filter(regex=f'hourly_units_{variable}_member')))]
        #converts nested list in an element into multiple rows for each list element
        #get cols to explode:
        cols_vars.append( list(df_fc_data.filter(regex=f'hourly_{variable}')) )
    
    cols_to_explode = list(set(cols_vars[0] + cols_vars[1]))
    
    df_fc_data = df_fc_data.explode(cols_to_explode + ['hourly_time'])

    df_fc_data = df_fc_data.drop(['generationtime_ms', 'utc_offset_seconds', 'timezone_abbreviation'], axis = 1)
    df_fc_data = df_fc_data.reset_index().drop('index', axis = 1)

    df_fc_data['hourly_time'] = pd.to_datetime(df_fc_data['hourly_time'])
    df_fc_data[cols_to_explode] = df_fc_data[cols_to_explode].apply(pd.to_numeric, axis = 1)
    
    time_ini = df_fc_data['hourly_time'].min().strftime('%Y%m%d%H')
    time_end = df_fc_data['hourly_time'].max().strftime('%Y%m%d%H')
    
    return(df_fc_data, time_ini, time_end, cols_vars)

def plot_ensemble_meteogram(ensemble, df_ens_fc_data_parsed, df_oper_fc_data_parsed, cols_vars_ens, cols_vars_oper, time_ini, time_end):
    
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (16,8))

    N = str ( int ( (len(df_ens_fc_data_parsed.columns) - 10)/len(variables_list) ) )

    plt.suptitle(f'{ensemble} Ensemble meteogram for {df_ens_fc_data_parsed.locality[0]} ({df_ens_fc_data_parsed.latitude[0].round(2)}, {df_ens_fc_data_parsed.longitude[0].round(2)}) \n \
                N = {N} members. {variables_list[0]} and {variables_list[1]} | {time_ini}', fontsize = 16)
    axs.grid()

    axs.set_xlim([df_ens_fc_data_parsed['hourly_time'].min(), df_ens_fc_data_parsed['hourly_time'].max()])
    axs.set_ylim( [df_oper_fc_data_parsed['hourly_temperature_2m'].min() - 5.0, df_oper_fc_data_parsed['hourly_temperature_2m'].max() + 5.0] )
    units_ylabel1 = df_ens_fc_data_parsed.hourly_units_temperature_2m[0]
    axs.set_ylabel(f'{variables_list[0]} ({units_ylabel1})')
    fig.text(x = 0.15, y = 0.9, s = 'Powered by open-meteo.com. Plotted by @idfeiven', ha = 'center')

    axs.plot( df_ens_fc_data_parsed['hourly_time'],  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0.5, axis = 1), c = 'black', lw = 3, label = 'Median')
    axs.plot( df_ens_fc_data_parsed['hourly_time'],  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0.25, axis = 1), c = 'red', lw = 1, label = 'Quantile 25%')
    axs.plot( df_ens_fc_data_parsed['hourly_time'],  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0.75, axis = 1), c = 'red', lw = 1, label = 'Quantile 75%')
    axs.plot( df_ens_fc_data_parsed['hourly_time'],  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0, axis = 1), c = 'blue', lw = 1, label = 'Minimum')
    axs.plot( df_ens_fc_data_parsed['hourly_time'],  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 1, axis = 1), c = 'darkred', lw = 1, label = 'Maximum')
    axs.plot( df_ens_fc_data_parsed['hourly_time'],  df_ens_fc_data_parsed[cols_vars_ens[0][0]], c = 'darkviolet', lw = 2, label = 'Control' )
    axs.plot( df_oper_fc_data_parsed['hourly_time'],  df_oper_fc_data_parsed[cols_vars_oper[0][0]], c = 'green', lw = 2, label = 'Oper' )
    
    axs.fill_between(df_ens_fc_data_parsed['hourly_time'], df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0.25, axis = 1)
                     ,  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0.75, axis = 1)
                     , color = 'red', alpha = 0.5, label = '50% Members')
    axs.fill_between(df_ens_fc_data_parsed['hourly_time'], df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0.75, axis = 1)
                     ,  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 1, axis = 1)
                     , color = 'red', alpha = 0.3, label = '25% Members')
    axs.fill_between(df_ens_fc_data_parsed['hourly_time'], df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0, axis = 1)
                     ,  df_ens_fc_data_parsed[cols_vars_ens[0]].quantile(q = 0.25, axis = 1)
                     , color = 'red', alpha = 0.3)
    axs.legend()

    axs2 = axs.twinx()
    axs2.set_ylim([0,40])
    units_ylabel2 = df_ens_fc_data_parsed.hourly_units_precipitation[0]
    axs2.set_ylabel(f'{variables_list[1]} ({units_ylabel2})')

    df_fc_6h_pcp = df_ens_fc_data_parsed[cols_vars_ens[1] + ['hourly_time']].set_index('hourly_time').resample('6H').sum().reset_index()
    df_fc_6h_pcp_oper = df_oper_fc_data_parsed[cols_vars_oper[1] + ['hourly_time']].set_index('hourly_time').resample('6H').sum().reset_index()

    axs2.plot( df_fc_6h_pcp['hourly_time'],  df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 0.5, axis = 1), c = 'blue', lw = 3, label = 'Median')
    axs2.plot( df_fc_6h_pcp['hourly_time'],  df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 0.25, axis = 1), c = 'deepskyblue', lw = 2, label = 'Quantile 25%')
    axs2.plot( df_fc_6h_pcp['hourly_time'],  df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 0.75, axis = 1), c = 'deepskyblue', lw = 2, label = 'Quantile 75%')
    axs2.plot( df_fc_6h_pcp['hourly_time'],  df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 1, axis = 1), c = 'dodgerblue', lw = 1, label = 'Maximum')
    axs2.plot( df_fc_6h_pcp['hourly_time'],  df_fc_6h_pcp[cols_vars_ens[1][0]], c = 'darkviolet', lw = 2, label = 'Control' )
    axs2.plot( df_fc_6h_pcp_oper['hourly_time'],  df_fc_6h_pcp_oper[cols_vars_oper[1][0]], c = 'green', lw = 2, label = 'Oper' )

    axs2.fill_between(df_fc_6h_pcp['hourly_time'], df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 0, axis = 1)
                     ,  df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 0.5, axis = 1)
                     , color = 'lightskyblue', alpha = 0.8, label = 'Lower 50% Members')
    axs2.fill_between(df_fc_6h_pcp['hourly_time'], df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 0.5, axis = 1)
                     ,  df_fc_6h_pcp[cols_vars_ens[1]].quantile(q = 1, axis = 1)
                     , color = 'dodgerblue', alpha = 0.1, label = 'Upper 50% Members')
    axs2.legend()

    axs2.legend()
    axs2.legend(loc = 'upper left')
    plt.tight_layout()

    plt.savefig(f'./{ensemble}ensemble_meteogram_{df_ens_fc_data_parsed.latitude[0]}_{df_ens_fc_data_parsed.longitude[0]}_{time_ini}_{time_end}.jpg', dpi = 150)

def plot_superensemble_meteogram(df_super_ensemble, cols_vars, time_ini, time_end):
    
    cols_var_sens = list()
    for variable in variables_list:
        cols_var_sens.append( list(df_super_ensemble.filter(regex=f'hourly_{variable}')) )

    df_super_ensemble_times = df_super_ensemble[['hourly_time']].T.drop_duplicates().T

    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (16,8))

    plt.suptitle(f'Superensemble meteogram for {df_super_ensemble.locality.values[0][0]} ({df_super_ensemble.latitude.values[0][0].round(2)}, {df_super_ensemble.longitude.values[0][0].round(2)}) \n \
                N = 143 members. {variables_list[0]} and {variables_list[1]} | {time_ini}', fontsize = 16)
    axs.grid()

    axs.set_xlim([df_super_ensemble_times['hourly_time'].min(), df_super_ensemble_times['hourly_time'].max()])
    axs.set_ylim( [df_super_ensemble[cols_var_sens[0]].quantile(q = 0, axis = 1).min() - 5.0, df_super_ensemble[cols_var_sens[0]].quantile(q = 1, axis = 1).max() + 5.0] )
    units_ylabel1 = df_super_ensemble.hourly_units_temperature_2m.values[0][0]
    axs.set_ylabel(f'{variables_list[0]} ({units_ylabel1})')
    fig.text(x = 0.15, y = 0.9, s = 'Powered by open-meteo.com. Plotted by @idfeiven', ha = 'center')

    axs.plot( df_super_ensemble_times['hourly_time'],  df_super_ensemble[cols_var_sens[0]].quantile(q = 0.5, axis = 1), c = 'black', lw = 3, label = 'Median')
    axs.plot( df_super_ensemble_times['hourly_time'],  df_super_ensemble[cols_var_sens[0]].quantile(q = 0.25, axis = 1), c = 'red', lw = 2, label = 'Quantile 25%')
    axs.plot( df_super_ensemble_times['hourly_time'],  df_super_ensemble[cols_var_sens[0]].quantile(q = 0.75, axis = 1), c = 'red', lw = 2, label = 'Quantile 75%')
    axs.plot( df_super_ensemble_times['hourly_time'],  df_super_ensemble[cols_var_sens[0]].quantile(q = 0, axis = 1), c = 'blue', lw = 1, label = 'Minimum')
    axs.plot( df_super_ensemble_times['hourly_time'],  df_super_ensemble[cols_var_sens[0]].quantile(q = 1, axis = 1), c = 'darkred', lw = 1, label = 'Maximum')

    axs.fill_between(df_super_ensemble_times['hourly_time'], df_super_ensemble[cols_var_sens[0]].quantile(q = 0.25, axis = 1)
                     ,  df_super_ensemble[cols_var_sens[0]].quantile(q = 0.75, axis = 1)
                     , color = 'red', alpha = 0.5, label = '50% Members')
    axs.fill_between(df_super_ensemble_times['hourly_time'], df_super_ensemble[cols_var_sens[0]].quantile(q = 0.75, axis = 1)
                     ,  df_super_ensemble[cols_var_sens[0]].quantile(q = 1, axis = 1)
                     , color = 'red', alpha = 0.3, label = '25% Members')
    axs.fill_between(df_super_ensemble_times['hourly_time'], df_super_ensemble[cols_var_sens[0]].quantile(q = 0, axis = 1)
                     ,  df_super_ensemble[cols_var_sens[0]].quantile(q = 0.25, axis = 1)
                     , color = 'red', alpha = 0.3)
    axs.legend(loc = 'upper right')

    axs2 = axs.twinx()
    axs2.set_ylim([0,40])
    units_ylabel2 = df_super_ensemble.hourly_units_precipitation.values[0][0]
    axs2.set_ylabel(f'{variables_list[1]} ({units_ylabel2})')

    df_fc_super_ens_pcp = pd.concat([ df_super_ensemble_times['hourly_time'], df_super_ensemble[cols_vars[1]] ], axis = 1).set_index('hourly_time').resample('6H').sum().reset_index()

    axs2.plot( df_fc_super_ens_pcp['hourly_time'],  df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 0.5, axis = 1), c = 'blue', lw = 3, label = 'Median')
    axs2.plot( df_fc_super_ens_pcp['hourly_time'],  df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 0.25, axis = 1), c = 'deepskyblue', lw = 2, label = 'Quantile 25%')
    axs2.plot( df_fc_super_ens_pcp['hourly_time'],  df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 0.75, axis = 1), c = 'deepskyblue', lw = 2, label = 'Quantile 75%')
    axs2.plot( df_fc_super_ens_pcp['hourly_time'],  df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 1, axis = 1), c = 'dodgerblue', lw = 1, label = 'Maximum')

    axs2.fill_between(df_fc_super_ens_pcp['hourly_time'], df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 0, axis = 1)
                     ,  df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 0.5, axis = 1)
                     , color = 'lightskyblue', alpha = 0.8, label = 'Lower 50% Members')
    axs2.fill_between(df_fc_super_ens_pcp['hourly_time'], df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 0.5, axis = 1)
                     ,  df_fc_super_ens_pcp[cols_vars[1]].quantile(q = 1, axis = 1)
                     , color = 'dodgerblue', alpha = 0.1, label = 'Upper 50% Members')
    
    axs2.legend(loc = 'upper left')
    plt.tight_layout()

    plt.savefig(f'./super_ensemble_meteogram_{df_super_ensemble.locality.values[0][0]}_{time_ini}_{time_end}.jpg', dpi = 150)

#----------------------------------------------MAIN PROGRAM-----------------------------------------------
print('Parsing variables...')
str_variables = get_str_variables(variables_list)

df_super_ensemble = pd.DataFrame()
for loc in locality:

    print(f'Getting latitude/longitude from {loc}...')
    lat, lon, location_api = get_coordinates_from_locality(loc)

    index_oper_nwp = -1
    for ensemble in ensemble_nwp_centers:

        print('Parsing urls to get data...')
        urls_api_ens = get_url_ensemble_forecast(ensemble, str_variables, lat, lon)

        index_oper_nwp = index_oper_nwp + 1
        urls_api_oper = get_url_oper_forecast(oper_nwp_centers[index_oper_nwp], str_variables, lat, lon)
        
        print(f'Getting forecast data from {ensemble}...')
        df_fc_ens_data = get_point_forecast_data(urls_api_ens)
        df_fc_oper_data = get_point_forecast_data(urls_api_oper)
        
        print('Parsing forecast data...')
        df_ens_fc_data_parsed, time_ini, time_end, cols_vars_ens = get_parsed_df_fc_data(df_fc_ens_data, variables_list)
        df_ens_fc_data_parsed['locality'] = loc
        df_ens_fc_data_parsed['prediction_center'] = ensemble
        first_cols = ['locality', 'latitude','longitude', 'hourly_time', 'prediction_center']
        last_cols = [col for col in df_ens_fc_data_parsed.columns if col not in first_cols]
        df_ens_fc_data_parsed = df_ens_fc_data_parsed[first_cols + last_cols]

        df_oper_fc_data_parsed, time_ini, time_end, cols_vars_oper = get_parsed_df_fc_data(df_fc_oper_data, variables_list)
        df_oper_fc_data_parsed['locality'] = loc
        df_oper_fc_data_parsed['prediction_center'] = oper_nwp_centers[index_oper_nwp]
        first_cols = ['locality', 'latitude','longitude', 'hourly_time', 'prediction_center']
        last_cols = [col for col in df_oper_fc_data_parsed.columns if col not in first_cols]
        df_oper_fc_data_parsed = df_oper_fc_data_parsed[first_cols + last_cols]

        print(f'Plotting meteogram for {loc}')
        plot_ensemble_meteogram(ensemble, df_ens_fc_data_parsed, df_oper_fc_data_parsed, cols_vars_ens, cols_vars_oper, time_ini, time_end)
        
        print(f'Saving {ensemble} forecast data...')
        # df_ens_fc_data_parsed.to_csv(f'{ensemble}_ensemble_{loc}_{time_ini}.csv')
        df_super_ensemble = pd.concat([df_super_ensemble, df_ens_fc_data_parsed], axis = 1)
        
print(f'Plotting superensemble meteogram for {loc}')
plot_superensemble_meteogram(df_super_ensemble, cols_vars_ens, time_ini, time_end)
# df_super_ensemble.to_csv(f'super_ensemble_{loc}_{time_ini}.csv')

print('End of program.')