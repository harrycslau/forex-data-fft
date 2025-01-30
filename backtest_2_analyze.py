# %%
# For data manipulation
import os
from dotenv import load_dotenv
from tvDatafeed import TvDatafeedLive, Interval
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from dtaidistance import dtw
import pmdarima as pm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
from time import sleep
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
import sqlite3
import json


def initialize_data_sources(target_symbol, target_timeframe):
    global forex_data, prices
    
    # Fetch historical data for the target symbol
    forex_data = pd.read_csv("rawdata/XAUUSD_M15_BID_20240902-20241123.csv")
    prices = forex_data['Close']

    # Convert the 'Local time' column to datetime and set it as index
    forex_data['Local time'] = forex_data['Local time'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S.%f GMT%z'))
    forex_data.set_index('Local time', inplace=True)

    # Calculate the True Range (TR) and ATR-96
    forex_data['TR'] = np.maximum(forex_data['High'] - forex_data['Low'], 
                        np.abs(forex_data['High'] - forex_data['Close'].shift(1)),
                        np.abs(forex_data['Low'] - forex_data['Close'].shift(1)))
    forex_data['ATR'] = forex_data['TR'].rolling(window=96).mean()




# %%
def poly_forecast(signal, forecast_elements, fit_window=None, degree=2):
    """
    Extend a time series using polynomial fitting on a portion of the end of the series.

    Parameters:
    - signal: numpy array of original time series data.
    - forecast_elements: int, number of future points to extend.
    - degree: int, degree of the polynomial to fit.
    - fit_window: int or None, number of data points from the end of the series to use for fitting.
                  If None, use the entire series.

    Returns:
    - extended_time_series: numpy array of the original and extended time series.
    """
    # Determine the portion of the time series to use for fitting
    if fit_window is None or fit_window > len(signal):
        fit_window = len(signal)
    
    # Select the last `fit_window` points for fitting
    time_points = np.arange(len(signal) - fit_window, len(signal))
    fit_series = signal[-fit_window:]
    
    # Fit a polynomial of the specified degree to the selected portion of the series
    coefficients = np.polyfit(time_points, fit_series, degree)
    poly_func = np.poly1d(coefficients)
    
    # Create extended time points
    extended_time_points = np.arange(len(signal), len(signal) + forecast_elements)
    
    # Use the polynomial to predict extended values
    extended_values = poly_func(extended_time_points)
    
    # Combine the original series with the extended values
    extended_time_series = np.concatenate((signal, extended_values))
    
    return extended_time_series

# %%
def fft_forecast(signal, forecast_elements, fit_window=None, target_harmonics=40, trend_factor=0, trend_strength=0):
    # Store the full length of the original signal
    original_length = len(signal)
    
    # Use only the last `window` elements if specified for FFT, otherwise use the entire signal
    if fit_window is not None and fit_window < original_length:
        signal_fft = signal[-fit_window:]
    else:
        signal_fft = signal
    
    # Extend the FFT portion of the signal with a linear trend if specified
    if trend_strength > 0:
        trend_extension = signal_fft[-1] + trend_factor * np.arange(1, trend_strength + 1)
        extended_signal = np.concatenate((signal_fft, trend_extension))
    else:
        extended_signal = signal_fft
    
    N = len(extended_signal)
    
    # Perform FFT on the extended signal
    fft_result = np.fft.fft(extended_signal)
    fft_freq = np.fft.fftfreq(N)
    
    # Retain non-negative frequencies
    positive_freq_indices = fft_freq >= 0
    fft_freq_positive = fft_freq[positive_freq_indices]
    fft_result_positive = fft_result[positive_freq_indices]
    
    # Limit the number of harmonics
    num_harmonics = min(target_harmonics, len(fft_freq_positive))
    frequencies = fft_freq_positive[:num_harmonics]
    amplitudes = fft_result_positive[:num_harmonics]
    
    # Create extended indices including future points
    extended_indices = np.arange(N + forecast_elements)
    
    # Reconstruct the signal using the selected harmonics
    reconstructed_signal = np.zeros(N + forecast_elements)
    for i in range(num_harmonics):
        frequency = frequencies[i]
        amplitude_complex = amplitudes[i]
        amplitude = np.abs(amplitude_complex)
        phase = np.angle(amplitude_complex)
        omega = 2 * np.pi * frequency
        
        if frequency == 0:
            # DC component
            reconstructed_signal += (amplitude / N) * np.ones_like(extended_indices)
        else:
            reconstructed_signal += (2 * amplitude / N) * np.cos(omega * extended_indices + phase)
    
    # Define the length of the blend window as half of the original signal length
    blend_window = original_length // 2 if fit_window is None else fit_window // 2
    
    # Initialize the blended signal with the correct length
    blended_signal = np.zeros(original_length + forecast_elements)
    
    # Copy the original signal into the beginning of blended_signal
    blended_signal[:original_length - blend_window] = signal[:original_length - blend_window]
    
    # Blend the end of the original signal with the reconstructed signal
    for i in range(0, blend_window):  # Iterate through the last `blend_window` elements of the original signal
        idx = original_length - blend_window + i
        alpha = i / blend_window  # Adjusted alpha calculation to range from 0 to 1
        blended_signal[idx] = (1 - alpha) * signal[idx] + alpha * reconstructed_signal[blend_window+i]
    
    # Append the forecasted portion from the reconstructed signal
    blended_signal[original_length:] = reconstructed_signal[fit_window:fit_window+forecast_elements]
    
    return blended_signal


# %%
def poly_forecast_validation(signal, forecast_elements, validation_elements_list, input_elements_list, degree=2):
    
    results = []

    for validation_elements in validation_elements_list:
        for input_elements in input_elements_list:
            
            signal_forecast = poly_forecast(signal[:-validation_elements], validation_elements+forecast_elements, input_elements, degree)

            # Convert forecasts to numpy arrays if they are pandas Series
            if isinstance(signal_forecast, pd.Series):
                signal_forecast = signal_forecast.to_numpy()
            if isinstance(signal, pd.Series):
                signal = signal.to_numpy()
            
            # Extract validation portions
            actual_validation = signal[-validation_elements:]
            forecast_validation = signal_forecast[-validation_elements - forecast_elements:-forecast_elements]

            # Calculate distances
            dtw_value = dtw.distance(actual_validation, forecast_validation)
            rmse_value = np.sqrt(mean_squared_error(actual_validation, forecast_validation))

            # Store the results
            results.append({
                'poly_validation_elements': validation_elements,
                'poly_input_elements': input_elements,
                'rmse_validation': rmse_value,
                'dtw_validation': dtw_value
            })

    # Convert results to DataFrame and sort
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='rmse_validation', ascending=True).head(20)
    return df_results


# %%
def fft_forecast_validation(signal, forecast_elements, validation_elements_list, input_elements_list, harmonics_list):

    results = []

    # Ensure signal is a numpy array
    if isinstance(signal, pd.Series):
        signal = signal.to_numpy()

    # Iterate over all combinations of validation_elements, input_elements, and harmonics
    for validation_elements in validation_elements_list:
        for input_elements in input_elements_list:
            for harmonics in harmonics_list:
                
                # Define the total number of elements to forecast (validation + forecast)
                total_forecast_length = validation_elements + forecast_elements

                # Prepare the training signal by excluding the last 'validation_elements' points
                training_signal = signal[:-validation_elements]

                # Perform FFT-based forecasting
                try:
                    signal_forecast = fft_forecast(
                        signal=training_signal,
                        forecast_elements=total_forecast_length,
                        fit_window=input_elements,
                        target_harmonics=harmonics,
                        trend_factor=0,        # Set to 0 or adjust as needed
                        trend_strength=0       # Set to 0 or adjust as needed
                    )
                except ValueError as e:
                    print(f"ValueError for validation_elements={validation_elements}, "
                          f"input_elements={input_elements}, harmonics={harmonics}: {e}")
                    continue  # Skip this combination if an error occurs

                # Extract validation portions
                actual_validation = signal[-validation_elements:]
                forecast_validation = signal_forecast[-validation_elements - forecast_elements:-forecast_elements]

                # Calculate RMSE and DTW distance
                rmse_value = np.sqrt(mean_squared_error(actual_validation, forecast_validation))
                dtw_value = dtw.distance(actual_validation, forecast_validation)

                # Store the results
                results.append({
                    'fft_validation_elements': validation_elements,
                    'fft_input_elements': input_elements,
                    'fft_harmonics': harmonics,
                    'rmse_validation': rmse_value,
                    'dtw_validation': dtw_value
                })

    # Convert the results list to a pandas DataFrame
    df_results = pd.DataFrame(results)

    # Sort the DataFrame by RMSE in ascending order and select the top 20 results
    df_results = df_results.sort_values(by='rmse_validation', ascending=True).head(40)
    df_results = df_results.sort_values(by='dtw_validation', ascending=True).head(20)

    return df_results


# %%
def find_optimal_clusters(data, max_clusters=4):
    """Find optimal number of clusters using silhouette score."""
    # data is expected to be a 2D array of shape (n_samples, n_timepoints)
    n_samples = data.shape[0]
    data_reshaped = data.reshape(n_samples, -1)  # Ensure data is 2D
    
    # Try different numbers of clusters
    silhouette_scores = []
    for k in range(2, min(max_clusters + 1, n_samples)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data_reshaped)
        score = silhouette_score(data_reshaped, cluster_labels)
        silhouette_scores.append(score)
    
    # Return optimal number of clusters
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_k


def process_task(task_id, forex_data, prices, target_symbol, target_timeframe):
    # %%
    # Apply Savitzky-Golay filter to get the trend
    # Note that in scipy Savitzky-Golay filter, the last elements are calculated by padding,
    # so we would remove them to ensure an accurate trend data

    yvalues_trend = []
    yvalues_seasonal = []

    yvalues = prices[:-task_id]
    yvalues_trend.append(savgol_filter(yvalues,19,1))
    yvalues_trend.append(savgol_filter(yvalues_trend[0],59,1))
    yvalues_trend.append(savgol_filter(yvalues_trend[1],179,1))
    yvalues_seasonal.append(yvalues-yvalues_trend[0])
    yvalues_seasonal.append(yvalues_trend[0]-yvalues_trend[1])
    yvalues_seasonal.append(yvalues_trend[1]-yvalues_trend[2])

    yvalues_trend_forecast    = [None] * 3
    yvalues_seasonal_forecast = [None] * 3
    results_trend             = [None] * 3
    results_seasonal          = [None] * 3


    # %%
    # Definition of Forecast Parameters
    forecast_elements = 40

    t1_validation_elements_list = np.arange(16, 81, 8)
    t1_input_elements_list      = np.arange(12, 33, 4)

    s1_validation_elements_list = np.arange(16, 25, 4)
    s1_harmonics_list           = np.arange(10, 61, 10)
    s1_input_elements_list      = np.arange(400, 1601, 10)

    s0_validation_elements_list = [8]
    s0_harmonics_list           = np.arange(20, 60, 10)
    s0_input_elements_list      = np.arange(200, 800, 4)


    # %% [markdown]
    # ## 3.1 Medium Trend Forecast

    # %%
    results_trend[1] = poly_forecast_validation(yvalues_trend[1], forecast_elements, t1_validation_elements_list, t1_input_elements_list, degree=2)
    results_trend[1]

    # %%
    t1_forecast_all = []

    for _, row in results_trend[1][:10].iterrows():
        validation_elements = int(row['poly_validation_elements'])
        input_elements = int(row['poly_input_elements'])
        
        signal_forecast = poly_forecast(yvalues_trend[1][:-validation_elements], validation_elements+forecast_elements, input_elements)
        t1_forecast_all.append(signal_forecast)

    t1_forecast_mean = np.mean(t1_forecast_all, axis=0)


    # %% [markdown]
    # ## 3.2 Medium Seasonal / Small Trend Forecast

    # %%
    results_seasonal[1] = fft_forecast_validation(yvalues_seasonal[1], forecast_elements, s1_validation_elements_list, s1_input_elements_list, s1_harmonics_list)
    results_seasonal[1] 

    # %%
    s1_forecast_all = []

    for _, row in results_seasonal[1][:20].iterrows():
        validation_elements = int(row['fft_validation_elements'])
        input_elements = int(row['fft_input_elements'])
        harmonics = int(row['fft_harmonics'])
        signal_forecast = fft_forecast(yvalues_seasonal[1][:-validation_elements], validation_elements+forecast_elements, input_elements, harmonics, 0, 0)
        s1_forecast_all.append(signal_forecast)

    s1_forecast_mean = np.mean(s1_forecast_all, axis=0)
    t0_forecast_mean = t1_forecast_mean + s1_forecast_mean


    # %% [markdown]
    # ## 3.3 Small Seasonal Forecast

    # %%
    results_seasonal[0] = fft_forecast_validation(yvalues_seasonal[0], forecast_elements, s0_validation_elements_list, s0_input_elements_list, s0_harmonics_list)
    results_seasonal[0] 

    # %%
    s0_forecast_all = []

    for _, row in results_seasonal[0][:20].iterrows():
        validation_elements = int(row['fft_validation_elements'])
        input_elements = int(row['fft_input_elements'])
        harmonics = int(row['fft_harmonics'])
        signal_forecast = fft_forecast(yvalues_seasonal[0][:-validation_elements], validation_elements+forecast_elements, input_elements, harmonics, 0, 0)
        s0_forecast_all.append(signal_forecast)

    s0_forecast_mean = np.mean(s0_forecast_all, axis=0)


    # %%
    # Clustering

    # Convert to numpy arrays for easier manipulation
    s0_forecast_all = np.array(s0_forecast_all)

    # Get the validation and first equal amount of forecast data for clustering
    validation_elements = s0_validation_elements_list[0]
    end_idx = -forecast_elements + validation_elements
    start_idx = -forecast_elements - validation_elements
    clustering_data = s0_forecast_all[:, start_idx:end_idx]
    n_samples = clustering_data.shape[0]
    data_reshaped = clustering_data.reshape(n_samples, -1)

    # Find optimal number of clusters
    n_clusters = find_optimal_clusters(clustering_data, 3)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_reshaped)

    # Group forecasts by cluster
    cluster_members = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_members[label].append(idx)

    # Calculate cluster means and intra-cluster distances
    cluster_means = {}
    intra_cluster_distances = {}
    for label, indices in cluster_members.items():
        cluster_data = s0_forecast_all[indices]
        cluster_means[label] = np.mean(cluster_data, axis=0)
        
        # Compute intra-cluster distance (mean pairwise distance within the cluster)
        distances = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                dist = np.linalg.norm(cluster_data[i] - cluster_data[j])
                distances.append(dist)
        if distances:
            intra_cluster_distance = np.mean(distances)
        else:
            intra_cluster_distance = 0  # Single member in cluster
        intra_cluster_distances[label] = intra_cluster_distance


    # %% [markdown]
    # ## 6 Plotting

    # %%
    # Plotting Parameters
    time_factor_preset = {  'M5': 5,
                            'M15': 1/4,
                            'H1': 1,
                            'H4': 4,
                            'D1': 1,
                            'W1': 1 }

    grid_spacing_preset = { 'M5': 20,
                            'M15': 2,
                            'H1': 6,
                            'H4': 24,
                            'D1': 5,
                            'W1': 5 }

    time_unit = { 'M5': 'Minute',
                'M15': 'Hour',
                'H1': 'Hour',
                'H4': 'Hour',
                'D1': 'Day', 
                'W1': 'Week' }

    time_factor = time_factor_preset[target_timeframe]
    grid_spacing = grid_spacing_preset[target_timeframe]

    visualization_elements_original = 40
    visualization_elements_forecast = 24

    # %%
    # Prepare common data slices and indices
    len_original_slice = visualization_elements_original
    forecast_slice_start_idx = -forecast_elements-visualization_elements_original
    forecast_slice_end_idx   = -forecast_elements+visualization_elements_forecast
    original_slice_idx_list = np.arange(-visualization_elements_original+1, 1)
    forecast_slice_idx_list = np.arange(-visualization_elements_original+1, visualization_elements_forecast+1)
    seasonal_validation_elements = s0_validation_elements_list[0]

    # Prepare the slices, which is originally in different lengths
    prices_slice = yvalues[-visualization_elements_original:]
    t0_forecast_mean_slice = t0_forecast_mean[forecast_slice_start_idx:forecast_slice_end_idx]
    t1_forecast_mean_slice = t1_forecast_mean[forecast_slice_start_idx:forecast_slice_end_idx]
    yvalues_seasonal_slice = yvalues_seasonal[0][-visualization_elements_original:]
    s0_forecast_all_slice = np.array(s0_forecast_all)[:, forecast_slice_start_idx:forecast_slice_end_idx]
    cluster_means_slice = {}
    for label, data in cluster_means.items():
        cluster_means_slice[label] = np.array(data)[forecast_slice_start_idx:forecast_slice_end_idx]

    # Get the last data time
    last_datetime = forex_data.index[-task_id-1]

    # %%
    # Plotting
    plotting = False
    if (plotting):
        plt.figure(figsize=(12, 9))
        colors = ['forestgreen', 'red', 'darkorange', 'dodgerblue']


        # Plot 1: Original Price with Trend Forecasts
        plt.subplot(3, 1, 1)
        plt.plot(original_slice_idx_list*time_factor, prices_slice, label="Original Price", color='blue')
        plt.plot(forecast_slice_idx_list*time_factor, t0_forecast_mean_slice, label='Trend Component', color='violet')
        plt.axvline(x=-seasonal_validation_elements*time_factor, color='orange', linestyle='--', label="Validation Start")
        plt.axvline(x=0, color='red', linestyle='--', label="Forecast Start")
        plt.axvline(x=seasonal_validation_elements*time_factor, color='violet', linestyle='--', label="70% Boundary")
        plt.legend(loc = "lower left")
        plt.title(rf"$\mathbf{{{target_symbol}\ {target_timeframe}}}$ - Original Prices with Trend Forecasts - {last_datetime}")
        plt.xlabel(time_unit[target_timeframe])
        plt.ylabel("Prices")
        # Set x-axis limits
        x_min = -math.ceil(visualization_elements_original*time_factor)
        x_max = math.ceil(visualization_elements_forecast*time_factor)
        plt.xlim(x_min, x_max)
        x_ticks = np.arange(x_min - (x_min % grid_spacing), x_max + grid_spacing, grid_spacing)
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.5)



        # Plot 2: Original Price with Clustered Forecasts
        plt.subplot(3, 1, 2)
        plt.plot(original_slice_idx_list*time_factor, prices_slice, label="Original Price", color='blue')
        plt.plot(forecast_slice_idx_list*time_factor, t0_forecast_mean_slice, label='Mean Trend Component', color='violet')
        plt.axvline(x=-seasonal_validation_elements*time_factor, color='orange', linestyle='--')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.axvline(x=seasonal_validation_elements*time_factor, color='violet', linestyle='--')
        for label, indices in cluster_members.items():
            plt.plot(
                forecast_slice_idx_list*time_factor,
                t0_forecast_mean_slice+cluster_means_slice[label],
                color=colors[label],
                linewidth=2,
                label=f'Cluster {label} (n={len(indices)}, intra-dist={intra_cluster_distances[label]:.2f})'
            )
        plt.legend(loc = "lower left")
        plt.title(f"Original Prices with Clustered Forecasts")
        plt.xlabel(time_unit[target_timeframe])
        plt.ylabel("Prices")
        # Set x-axis limits
        x_min = -math.ceil(visualization_elements_original*time_factor)
        x_max = math.ceil(visualization_elements_forecast*time_factor)
        plt.xlim(x_min, x_max)
        x_ticks = np.arange(x_min - (x_min % grid_spacing), x_max + grid_spacing, grid_spacing)
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.5)


        # Plot 3: Original Seasonal vs Clustered Seasonal Forecasts
        plt.subplot(3, 1, 3)
        plt.plot(original_slice_idx_list*time_factor, yvalues_seasonal_slice, label="Original Seasonal", color='blue')
        plt.axvline(x=-seasonal_validation_elements*time_factor, color='orange', linestyle='--')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.axvline(x=seasonal_validation_elements*time_factor, color='violet', linestyle='--')
        for row in s0_forecast_all_slice:
            plt.plot(forecast_slice_idx_list*time_factor, row, color='pink', alpha=0.15)
        for label, indices in cluster_members.items():
            plt.plot(
                forecast_slice_idx_list*time_factor,
                cluster_means_slice[label],
                color=colors[label],
                linewidth=2,
                label=f'Cluster {label} (n={len(indices)}, intra-dist={intra_cluster_distances[label]:.2f})'
            )
        plt.legend(loc = "lower left")
        plt.title(f"Original Seasonal vs Clustered Seasonal Forecasts")
        plt.xlabel(time_unit[target_timeframe])
        plt.ylabel("Prices")
        # Set x-axis limits
        x_min = -math.ceil(visualization_elements_original*time_factor)
        x_max = math.ceil(visualization_elements_forecast*time_factor)
        plt.xlim(x_min, x_max)
        x_ticks = np.arange(x_min - (x_min % grid_spacing), x_max + grid_spacing, grid_spacing)
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.5)


        plt.tight_layout()
        plt.savefig(f'backtest_results/{task_id:04}.png')
        plt.close()

    # %%
    # Prepare data to write

    # Calculate clustersize and clusterdist
    clustersize = [len(indices) for label, indices in cluster_members.items()]
    clusterdist = [intra_cluster_distances[label] for label in cluster_members.keys()]

    # Prepare cluster data
    cluster_data_dict = {
        f"cluster_{i}": cluster_means[i][-forecast_elements-1:-forecast_elements+s0_validation_elements_list[0]].tolist()
        for i in range(len(cluster_means))
    }

    # Prepare trend data
    trend_data = t0_forecast_mean[-forecast_elements-1:-forecast_elements+s0_validation_elements_list[0]].tolist()

    # Create results dictionary
    results = {
        "task_id": task_id,
        "time": last_datetime.isoformat(),
        "price": forex_data["Close"].iloc[-task_id-1],
        "atr96": forex_data["ATR"].iloc[-task_id-1],
        "clusternum": len(cluster_means),
        "clustersize": json.dumps(clustersize),
        "clusterdist": json.dumps(clusterdist),
        "clusterdata": json.dumps(cluster_data_dict),
        "trenddata": json.dumps(trend_data)
    }

    write_to_db(results, target_symbol, target_timeframe)


# %%
# Write results to SQLite
def write_to_db(results, target_symbol, target_timeframe):

    conn = sqlite3.connect(f"backtest_results/{target_symbol}_{target_timeframe}.db")
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            task_id INTEGER,
            time TEXT,
            price REAL,
            atr96 REAL,
            clusternum INTEGER,
            clustersize TEXT,
            clusterdist TEXT,
            clusterdata TEXT,
            trenddata TEXT
        )
    ''')

    # Insert data
    cursor.execute('INSERT INTO results (task_id, time, price, atr96, clusternum, clustersize, clusterdist, clusterdata, trenddata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (results["task_id"], results["time"], results["price"], results["atr96"], results["clusternum"], results["clustersize"], results["clusterdist"], results["clusterdata"], results["trenddata"]))
        
    conn.commit()
    conn.close()


# %%
# Main script
if __name__ == "__main__":

    # target information (IMPORTANT!)
    target_symbol = 'XAUUSD'
    target_timeframe = 'M15'

    initialize_data_sources(target_symbol, target_timeframe)
    task_ids = range(1, 3901)

#    for task_id in task_ids:
#        print(f'Processing task {task_id}...')
#        process_task(task_id)
 
    # Use partial to pass additional arguments to the worker function
    process_task_partial = partial(process_task, forex_data=forex_data, prices=prices, target_symbol=target_symbol, target_timeframe=target_timeframe)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_task_partial, task_ids), total=len(task_ids), desc="Processing Tasks"))

    print("Parallel processing complete.")

# %%
