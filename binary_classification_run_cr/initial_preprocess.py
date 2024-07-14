import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.signal import find_peaks, periodogram
from scipy.stats import entropy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("Initial_Preprocess_cr.log", mode='w'),
                        logging.StreamHandler()
                    ]
                    )
# Configure the random state
random_state = 1000

# Define DroneSignalsDataset class
class DroneSignalsDataset(Dataset):
    """
    Class for custom dataset of drone data comprised of
    x_iq (torch.tensor.float): signals iq data(n_samples x 2 x input_vec_length)
    x_spec (torch.tensor.float): signals spectrogram (n_samples x 2 x num_segments x num_segments)
    y (torch.tensor.long): targets (n_samples)
    snr (torch.tensor.int): SNRs per sample (n_samples)
    duty_cycle (torch.tensor.float): duty cycle length per sample (n_samples)
    Args:
        Dataset (torch tensor):
    """
    def __init__(self, x_iq, x_spec, y, snr, duty_cycle):
        self.x_iq = x_iq
        self.x_spec = x_spec
        self.y = y
        self.snr = snr
        self.duty_cycle = duty_cycle

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_iq[idx], self.x_spec[idx], self.y[idx], self.snr[idx], self.duty_cycle[idx]

def calculate_entropy(signal):
    histogram, _ = np.histogram(signal, bins=30, density=True)
    return entropy(histogram)

def extract_iq_features(iq_data, batch_name='', bar_position=None):
    iq_features_dfs = []
    channels = ['RE', 'IM']

    # Iterate with a progress bar
    for i in tqdm(range(iq_data.shape[0]), desc=f"Extracting features of {batch_name}", position=bar_position):
        channel_vectors = iq_data[i, :, :]
        channels_iq_features = []
        for channel_vector, channel in zip(channel_vectors, channels):
            channel_prefix = f'IQ_{channel}_'
            channel_iq_features = []
            channel_feature_names = []

            # Use vectorized operations for basic statistical features
            channel_iq_features.extend([
                np.mean(channel_vector), np.min(channel_vector), np.max(channel_vector),
                np.median(channel_vector), np.std(channel_vector), skew(channel_vector),
                kurtosis(channel_vector)
            ])
            channel_feature_names.extend([
                channel_prefix + 'Mean', channel_prefix + 'Min', channel_prefix + 'Max',
                channel_prefix + 'Median', channel_prefix + 'StdDev', channel_prefix + 'Skewness',
                channel_prefix + 'Kurtosis'
            ])

            # Peak count
            peaks = find_peaks(channel_vector)[0]
            channel_iq_features.append(len(peaks))
            channel_feature_names.append(channel_prefix + 'PeakCount')

            # RMS
            channel_iq_features.append(np.sqrt(np.mean(np.square(channel_vector))))
            channel_feature_names.append(channel_prefix + 'RMS')

            # Advanced Time-Domain Features
            autocorr = np.correlate(channel_vector, channel_vector, mode='full')[len(channel_vector) - 1]
            zero_crossings = np.where(np.diff(np.signbit(channel_vector)))[0].size
            channel_iq_features.extend([autocorr, zero_crossings])
            channel_feature_names.extend([channel_prefix + 'Autocorr', channel_prefix + 'ZeroCrossings'])

            # Entropy
            signal_entropy = calculate_entropy(channel_vector)
            channel_iq_features.append(signal_entropy)
            channel_feature_names.append(channel_prefix + 'Entropy')

            # Frequency Domain Features
            freq_domain = fft(channel_vector)
            freq_magnitude = np.abs(freq_domain)
            freq_variance = np.var(freq_magnitude)
            dominant_freq = np.argmax(freq_magnitude)
            spectral_entropy = -np.sum(
                (freq_magnitude / freq_magnitude.sum()) * np.log(freq_magnitude / freq_magnitude.sum() + 1e-10))
            channel_iq_features.extend([dominant_freq, freq_variance, spectral_entropy])
            channel_feature_names.extend(
                [channel_prefix + 'DominantFreq', channel_prefix + 'FreqVariance', channel_prefix + 'SpectralEntropy'])

            # Spectral Descriptors
            freqs, power_spec = periodogram(channel_vector, fs=2.0)
            spectral_centroid = np.sum(freqs * power_spec) / np.sum(power_spec)
            spectral_flatness = np.exp(np.mean(np.log(power_spec + 1e-10))) / np.mean(power_spec)
            freq_band_power = np.sum(power_spec[(freqs >= 1.0) & (freqs <= 10.0)])
            channel_iq_features.extend([spectral_centroid, spectral_flatness, freq_band_power])
            channel_feature_names.extend([channel_prefix + 'SpectralCentroid', channel_prefix + 'SpectralFlatness',
                                          channel_prefix + 'FreqBandPower'])

            channels_iq_features.append(pd.DataFrame([channel_iq_features], columns=channel_feature_names))
        iq_features_dfs.append(pd.concat(channels_iq_features, axis=1))

    result_df = pd.concat(iq_features_dfs, axis=0).reset_index(drop=True)
    return result_df

def extract_spectrogram_features(spec_data, batch_name='', bar_position=None):
    spec_features_dfs = []
    channels = ['RE', 'IM']

    # Iterate over each sample in spec_data with a progress bar
    for sample_index in tqdm(range(spec_data.shape[0]), desc=f"Extracting features of {batch_name}", position=bar_position):
        sample_features = []
        for channel_index, channel in enumerate(channels):
            channel_matrix = spec_data[sample_index, channel_index, :, :]
            channel_vector = channel_matrix.flatten()
            channel_prefix = f'Spec_{channel}_'
            channel_features = []
            channel_feature_names = []

            # Basic Statistical Features
            channel_features.extend([
                np.mean(channel_vector), np.min(channel_vector), np.max(channel_vector),
                np.median(channel_vector), np.std(channel_vector)
            ])
            channel_feature_names.extend([
                channel_prefix + 'Mean', channel_prefix + 'Min', channel_prefix + 'Max',
                channel_prefix + 'Median', channel_prefix + 'StdDev'
            ])

            # Skewness, Kurtosis, and Peak Count
            channel_features.extend([skew(channel_vector), kurtosis(channel_vector), len(find_peaks(channel_vector)[0])])
            channel_feature_names.extend([channel_prefix + 'Skewness', channel_prefix + 'Kurtosis', channel_prefix + 'PeakCount'])

            # RMS and Autocorrelation
            channel_features.append(np.sqrt(np.mean(np.square(channel_vector))))
            channel_feature_names.append(channel_prefix + 'RMS')
            autocorr = np.correlate(channel_vector, channel_vector, mode='full')[len(channel_vector)-1]
            channel_features.append(autocorr)
            channel_feature_names.append(channel_prefix + 'Autocorr')

            # Zero Crossings and Entropy
            zero_crossings = np.where(np.diff(np.signbit(channel_vector)))[0].size
            signal_entropy = calculate_entropy(channel_vector)
            channel_features.extend([zero_crossings, signal_entropy])
            channel_feature_names.extend([channel_prefix + 'ZeroCrossings', channel_prefix + 'Entropy'])

            # Frequency Domain Features
            freq_domain = fft(channel_vector)
            freq_magnitude = np.abs(freq_domain)
            dominant_freq = np.argmax(freq_magnitude)
            freq_variance = np.var(freq_magnitude)
            spectral_entropy = -np.sum((freq_magnitude / np.sum(freq_magnitude)) * np.log(freq_magnitude / np.sum(freq_magnitude) + 1e-10))
            channel_features.extend([dominant_freq, freq_variance, spectral_entropy])
            channel_feature_names.extend([channel_prefix + 'DominantFreq', channel_prefix + 'FreqVariance', channel_prefix + 'SpectralEntropy'])

            # Spectral Descriptors
            freqs, power_spec = periodogram(channel_vector, fs=2.0)
            spectral_centroid = np.sum(freqs * power_spec) / np.sum(power_spec)
            spectral_flatness = np.exp(np.mean(np.log(power_spec + 1e-10))) / np.mean(power_spec)
            freq_band_power = np.sum(power_spec[(freqs >= 0.25) & (freqs <= 0.75)])
            channel_features.extend([spectral_centroid, spectral_flatness, freq_band_power])
            channel_feature_names.extend([channel_prefix + 'SpectralCentroid', channel_prefix + 'SpectralFlatness', channel_prefix + 'FreqBandPower'])

            sample_features.append(pd.DataFrame([channel_features], columns=channel_feature_names))
        spec_features_dfs.append(pd.concat(sample_features, axis=1))

    result_df = pd.concat(spec_features_dfs, axis=0).reset_index(drop=True)
    return result_df

def data_engineering_process(dataset, batch_size = 100, write_ondisk=True, engineered_dataset_directory='./ds/'):
    dataset_size = dataset.y.size()[0]
    # with ProcessPoolExecutor() as executor:
    with ThreadPoolExecutor() as executor:
        # Launch parallel tasks for feature extraction
        iq_futures = [
            executor.submit(extract_iq_features, dataset.x_iq.numpy()[i: batch_size+i if batch_size+i<=dataset_size else dataset_size], f'iq_batch_[{i}_{batch_size+i if batch_size+i<=dataset_size else dataset_size})', p) for p, i in enumerate(range(0, dataset_size, batch_size), start=0)
        ]

        spec_futures = [
            executor.submit(extract_spectrogram_features, dataset.x_spec.numpy()[i: batch_size+i if batch_size+i<=dataset_size else dataset_size], f'spec_batch_[{i}_{batch_size+i if batch_size+i<=dataset_size else dataset_size})', p) for p, i in enumerate(range(0, dataset_size, batch_size), int(np.ceil(dataset_size/batch_size)))
        ]

        # Wait for all tasks to complete
        iq_results = [future.result() for future in iq_futures]
        spec_results = [future.result() for future in spec_futures]


    time.sleep(5)  # Sleep for 5 seconds

    # Assign results to their respective dataframes
    iq_features_df = pd.concat(iq_results, axis=0, ignore_index=True)
    spec_features_df = pd.concat(spec_results, axis=0, ignore_index=True)

    # Concatenate the extracted features from both IQ and Spectrogram
    engineered_dataset = pd.concat([iq_features_df, spec_features_df], axis=1)

    logging.info(f"Extracted IQ Features shape: {iq_features_df.shape}")
    logging.info(f"Extracted Spectrogram Features shape: {spec_features_df.shape}")

    # Additional feature processing and saving logic remains unchanged
    engineered_dataset['SNR'] = dataset.snr.numpy()
    engineered_dataset['Duty_Cycle'] = dataset.duty_cycle.numpy()

    class_mapping = {
        0: 'DJI', 1: 'FutabaT14', 2: 'FutabaT7', 3: 'Graupner', 4: 'Noise', 5: 'Taranis', 6: 'Turnigy'
    }
    engineered_dataset['Class'] = [class_mapping[label] for label in dataset.y.numpy()]

    if write_ondisk:
        if not os.path.exists(engineered_dataset_directory):
            os.makedirs(engineered_dataset_directory)
        engineered_dataset.to_csv(f'{engineered_dataset_directory}engineered_dataset.csv', index=False)
        logging.info(f"The engineered data are saved as CSV files to {engineered_dataset_directory}")

    logging.info(f"Data engineering process is completed")
    return engineered_dataset

def load_dataset(path):
    # load data
    dataset_dict = torch.load(path)
    logging.info(f"Dataset keys: {dataset_dict.keys()}")
    #
    x_iq = dataset_dict['x_iq']
    x_spec = dataset_dict['x_spec']
    y = dataset_dict['y']
    snr = dataset_dict['snr']
    duty_cycle = dataset_dict['duty_cycle']

    dataset = DroneSignalsDataset(x_iq, x_spec, y, snr, duty_cycle)
    return dataset

def get_random_sample_dataset(dataset, sample_ratio=0.7, write_ondisk=True, dataset_directory ='./ds/', dataset_name='test_dataset.pt', random_state=1000):
    generator = torch.Generator().manual_seed(random_state)
    # Splitting the dataset
    test_size = int(len(dataset) * sample_ratio)
    remain_size = len(dataset) - test_size
    test_set, _ = random_split(dataset, [test_size, remain_size], generator=generator)


    test_dataset = DroneSignalsDataset(x_iq=test_set.dataset.x_iq[test_set.indices,:],
                                       x_spec=test_set.dataset.x_spec[test_set.indices,:],
                                       y=test_set.dataset.y[test_set.indices],
                                       snr=test_set.dataset.snr[test_set.indices],
                                       duty_cycle=test_set.dataset.duty_cycle[test_set.indices]
                                       )
    if write_ondisk:
        # Create the directory if it does not exist
        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        # Save the dataset to disk
        dataset_path = os.path.join(dataset_directory, dataset_name)

        # Save the dataset to disk
        torch.save({
            'x_iq': test_dataset.x_iq,
            'x_spec': test_dataset.x_spec,
            'y': test_dataset.y,
            'snr': test_dataset.snr,
            'duty_cycle': test_dataset.duty_cycle
        }, dataset_path)

        logging.info(f"Test Dataset saved to '{dataset_path}' with size of {len(test_dataset.y)}")

    return test_dataset

def main():
    logging.info('Initial data preprocess is started.')
    dataset_directory = '../ds/'

    # load the original dataset
    logging.info('Loading the original dataset.')
    dataset = load_dataset(path=dataset_directory + 'dataset.pt')

    # # Load a sample dataset from the original data
    # logging.info('Loading a sample dataset.')
    # dataset = get_random_sample_dataset(dataset=dataset, sample_ratio=0.01, dataset_directory=dataset_directory, random_state=random_state)


    logging.info("Start data engineering process")
    engineered_dataset = data_engineering_process(dataset=dataset, batch_size = 100, write_ondisk=True, engineered_dataset_directory= dataset_directory)

    logging.info('Initial data preprocess is completed.')

if __name__ == '__main__':
    main()
