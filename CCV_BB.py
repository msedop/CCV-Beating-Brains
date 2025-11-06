import numpy as np
import pandas as pd

import multiprocessing
from multiprocessing import Pool, cpu_count
import time
import os

import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import mne
from mne.datasets import misc
from mne_icalabel import label_components
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from mne.viz import plot_topomap
from mne.channels import Layout
from mne import create_info
from mne.channels import find_ch_adjacency
from mne.channels import find_layout, make_eeg_layout
from mne.annotations import Annotations
from mne import events_from_annotations 

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

from scipy.signal import correlate
from scipy.signal import welch
from scipy.signal import find_peaks, peak_prominences
import scipy.stats as stats
from scipy.stats import iqr
from scipy.stats import zscore
import scipy.sparse

import networkx as nx
from collections import defaultdict

from datetime import datetime
from datetime import timedelta
from mne.time_frequency import tfr_array_morlet

from mne.datasets import ssvep
import asrpy
from asrpy import ASR

plt.close("all")

edf_path = r"C:\Users\msedo\Documents\CCV - Beating Brains\CCV EEG\20250611_095543_438.EEG_5c0895de-0e46-4267-9ad3-ed14a72538bb.edf"

# Load the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Print basic info
print(raw.info)

channels = raw.info['ch_names'][:20]


sfreq = raw.info['sfreq']

# Plot with time in minutes
#raw.plot(time_format='%M:%S')  # Minutes:Seconds format



# --- Plot (shows vertical lines at H:M:S points) ---
raw.plot(time_format='%H:%M:%S')

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')


# Compute and plot PSD
fig = raw.compute_psd(tmax=np.inf, fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)

# Change the title
fig.axes[0].set_title("PSD before filters")
# ============ Pre-processing ====================================

# Optional bandpass and notch filtering (re-apply if unsure of quality)
raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
raw.notch_filter(freqs=[50], fir_design='firwin')  # Notch for 50 Hz power-line noise

raw.plot(time_format='%M:%S')  # Minutes:Seconds format

# Compute and plot PSD
fig = raw.compute_psd(tmax=np.inf, fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)

# Change the title
fig.axes[0].set_title("PSD after filters")

raw.plot(time_format='%H:%M:%S')

# ================= Bad channel detection (Tukey) ================

# Calculate standard deviation for each channel
data = raw.get_data()
channel_std = np.std(data, axis=1)

def tukey_outliers(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return np.where((data < lower_bound) | (data > upper_bound))[0]

# Detect outlier channels
outlier_indices = tukey_outliers(channel_std)


# Define bad channels
bad_channels = [raw.ch_names[i] for i in outlier_indices if 'EOG' not in raw.ch_names[i]]

if bad_channels:
    bad_indices = [raw.ch_names.index(ch) for ch in bad_channels]
    bad_data, times = raw[bad_indices, :]

    # Plot Bad channels 
    plt.figure(figsize=(12, 6))
    for i, channel_data in enumerate(bad_data):
        plt.plot(times, channel_data * 1e6 + i * 100, label=bad_channels[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude + offset (µV)')
    plt.title('Bad EEG Channels (Stacked)')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()
else:
    print("No bad channels were detected.")



# Get the list of good channels
good_channels = [ch for ch in raw.ch_names if ch not in bad_channels]

# Get data for the good channels
good_indices = [raw.ch_names.index(ch) for ch in good_channels]
good_data, times = raw[good_indices, :]


# Plot good channels
plt.figure(figsize=(14, 6))
for i, ch_data in enumerate(good_data):
    plt.plot(times, ch_data * 1e6 + i * 100, label=good_channels[i])  # µV scaling and stacking

plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset (µV)')
plt.title('Good EEG Channels (Stacked)')
plt.legend(loc='upper right', fontsize='x-small', ncol=2)
plt.tight_layout()
plt.show()

all_channels = [ch for ch in raw.ch_names]
all_data, times = raw[: , :]


raw_g = raw.copy().pick(good_channels)

# Decimation to reduce computational complexity of ASR and ICA
raw_g.resample(256)

# =========================== Segmentation ==========================================

# Parameters
min_segment_duration = 15 * 60  # 15 minutes in seconds

# Start from third event
start_idx = 2
segments = []

i = start_idx
while i < len(onset_list):
    seg_start = onset_list[i]
    seg_end = seg_start
    j = i
    # Group consecutive events until segment is >= 15 min or run out of events
    while j < len(onset_list):
        seg_end = onset_list[j] + duration_list[j]
        total_duration = seg_end - seg_start
        if total_duration >= min_segment_duration:
            break
        j += 1
    # Store segment index range
    segments.append((i, j))
    i = j + 1  # Start next segment after current

n_events = len(onset_list)
max_time = raw_g.times[-1]

for seg_num, (seg_start_idx, seg_end_idx) in enumerate(segments, 1):
    seg_onset = onset_list[seg_start_idx]
    # If seg_end_idx is out of bounds, set seg_end to the end of the recording
    if seg_end_idx >= n_events:
        seg_end = max_time
    else:
        seg_end = onset_list[seg_end_idx] + duration_list[seg_end_idx]
        # Ensure seg_end does not exceed data
        seg_end = min(seg_end, max_time)
    raw_segment = raw_g.copy().crop(tmin=seg_onset, tmax=seg_end)
    print(f"Segment {seg_num}: Events {seg_start_idx + 1}-{min(seg_end_idx + 1, n_events)}, "
          f"Start={seg_onset:.1f}s, End={seg_end:.1f}s, Duration={(seg_end-seg_onset)/60:.2f} min")

 
# =========================== ASR ==========================================

# -------------------------- ASR Calibration --------------------------------
# Get the onsets and durations of the first two annotations
annotations = raw.annotations
onset1 = annotations.onset[0]
duration1 = annotations.duration[0]
onset2 = annotations.onset[1]
duration2 = annotations.duration[1]

# Calculate the combined time window: from the start of the first to the end of the second
start_time = onset1
end_time = onset2 + duration2

raw_segment = raw_g.copy().crop(tmin=start_time, tmax=end_time)

raw_segment.plot(title="First two annotations (continuous)", time_format='%H:%M:%S')

# Use the segment from the start of the first annotation to the end of the second
asr_calib_start = onset1
asr_calib_end = onset2 + duration2

# Crop this segment from the raw data
raw_calib = raw.copy().crop(tmin=asr_calib_start, tmax=asr_calib_end)

# Pick only EEG channels (if not already done)
picks = mne.pick_types(raw_calib.info, eeg=True)
raw_calib.pick(picks)

# Fit ASR using this calibration segment
asr = asrpy.ASR(sfreq=raw_calib.info['sfreq'], cutoff=5)  # Lower cuttoff values (5-10) are more restrictive. Higher values (20-50) are more permissive
asr.fit(raw_calib)


# -------------------- ASR computation and result storage ---------------------

asr_cleaned_segments = []
raw_segments = []

n_events = len(onset_list)
max_time = raw_g.times[-1]

for seg_num, (seg_start_idx, seg_end_idx) in enumerate(segments, 1):
    seg_onset = onset_list[seg_start_idx]
    if seg_end_idx >= n_events:
        seg_end = max_time
    else:
        seg_end = onset_list[seg_end_idx] + duration_list[seg_end_idx]
        seg_end = min(seg_end, max_time)
    # Extract segment
    segment = raw_g.copy().crop(tmin=seg_onset, tmax=seg_end)
    segment.pick(picks)
    # Store original segment for plotting
    raw_segments.append(segment.copy())
    # Apply ASR and store cleaned segment
    cleaned_segment = asr.transform(segment, mem_splits=100)
    asr_cleaned_segments.append(cleaned_segment)
    print(f"ASR applied to segment {seg_num} (events {seg_start_idx+1}-{min(seg_end_idx+1, n_events)})")

# --------------------- ASR cleaned signal Plotting ------------------------

channel_names = raw.info['ch_names']


for seg_num, (raw_segment, clean_segment) in enumerate(zip(raw_segments, asr_cleaned_segments), 1):
    # Get data and times
    raw_data, times = raw_segment[picks, :]
    clean_data, _ = clean_segment[picks, :]
    # Convert to µV
    raw_data *= 1e6
    clean_data *= 1e6

    # Plot overlay
    plot_sfreq = 50  # Hz
    downsample_factor = int(raw_g.info['sfreq'] // plot_sfreq)
    plot_duration = 30  # seconds
    
    n_plot = min(int(plot_duration * plot_sfreq), raw_data.shape[1] // downsample_factor)
    raw_data_ds = raw_data[:, :n_plot * downsample_factor:downsample_factor]
    clean_data_ds = clean_data[:, :n_plot * downsample_factor:downsample_factor]
    times_ds = times[:n_plot * downsample_factor:downsample_factor]
    
    plt.figure(figsize=(16, 8))
    for i in range(n_channels):
        plt.plot(times_ds, raw_data_ds[i] + i * offset, color='red', alpha=0.5, linewidth=0.8,
                 label='Raw' if i == 0 else "")
        plt.plot(times_ds, clean_data_ds[i] + i * offset, color='black', alpha=0.9, linewidth=0.8,
                 label='ASR Cleaned' if i == 0 else "")
    plt.yticks([i * offset for i in range(n_channels)], channel_names)
    plt.xlabel("Time (s)")
    plt.ylabel("EEG Channels (stacked)")
    plt.title(f"Segment {seg_num}: Raw (Red) vs ASR-Cleaned (Black)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
 
# ====================== Segmentation of ASR cleaned data =================   

from scipy.stats import zscore

def compute_channel_metrics(window_uV):
    # 1. Variance
    variance = np.var(window_uV, axis=1)
    # 2. Median gradient (median of absolute diff between consecutive samples)
    median_grad = np.median(np.abs(np.diff(window_uV, axis=1)), axis=1)
    # 3. Amplitude range (max - min)
    amp_range = np.ptp(window_uV, axis=1)
    # 4. Deviation from mean amplitude (mean(abs(signal - mean(signal))))
    mean_amp = np.mean(window_uV, axis=1, keepdims=True)
    dev_from_mean = np.mean(np.abs(window_uV - mean_amp), axis=1)
    return np.vstack([variance, median_grad, amp_range, dev_from_mean])

def find_clean_window_faster(raw_segment, window_sec=600, step_sec=60, z_thresh=3):
    data, times = raw_segment.get_data(return_times=True)
    sfreq = raw_segment.info['sfreq']
    n_samples_window = int(window_sec * sfreq)
    n_samples_step = int(step_sec * sfreq)
    n_samples = data.shape[1]
    data_uV = data * 1e6

    for start in range(0, n_samples - n_samples_window + 1, n_samples_step):
        window = data_uV[:, start:start + n_samples_window]
        metrics = compute_channel_metrics(window)  # shape (4, n_channels)
        metrics_z = zscore(metrics, axis=1)
        # Mark bad any channel with any metric zscore > z_thresh (abs)
        bad_channels = np.any(np.abs(metrics_z) > z_thresh, axis=0)
        if not np.any(bad_channels):
            tmin = times[start]
            tmax = times[start + n_samples_window - 1]
            return raw_segment.copy().crop(tmin=tmin, tmax=tmax)
    return None  # No clean window found



clean_segments = []
for seg_num, (seg_start_idx, seg_end_idx) in enumerate(segments, 1):
    seg_onset = onset_list[seg_start_idx]
    if seg_end_idx >= n_events:
        seg_end = max_time
    else:
        seg_end = onset_list[seg_end_idx] + duration_list[seg_end_idx]
        seg_end = min(seg_end, max_time)
    segment = raw_g.copy().crop(tmin=seg_onset, tmax=seg_end)
    segment.pick(picks)
    clean_10min = find_clean_window_faster(segment)
    if clean_10min is not None:
        print(f"Found HAPPE/FASTER-style clean 10-min window in segment {seg_num}")
        clean_segments.append(clean_10min)
    else:
        print(f"No clean window found in segment {seg_num}") 



for i, clean_segment in enumerate(clean_segments, 1):
    # Get channel names and data
    picks = mne.pick_types(clean_segment.info, eeg=True)
    channel_names = [clean_segment.ch_names[j] for j in picks]
    data, times = clean_segment[picks, :]
    data = data * 1e6  # convert to µV

    # Plot
    plt.figure(figsize=(16, 8))
    offset = 200  # µV vertical spacing
    n_channels = len(picks)
    for ch in range(n_channels):
        plt.plot(times, data[ch] + ch * offset, color='black', linewidth=0.8)
    plt.yticks([ch * offset for ch in range(n_channels)], channel_names)
    plt.xlabel("Time (s)")
    plt.ylabel("EEG Channels (stacked)")
    plt.title(f"Clean 10-min window, Segment {i}")
    plt.tight_layout()
    plt.show()

# ================================= ICA ====================================


def infer_region(ch_name):
    
    # Simplified region mapping
    region_map = {
        "AF": "Frontal", "Fp": "Frontal", "F": "Frontal", "FC": "Frontal",
        "T": "Temporal", "C": "Central", "CP": "Parietal", "P": "Parietal",
        "PO": "Occipital", "O": "Occipital"
    }    
    
    for prefix, region in region_map.items():
        if ch_name.startswith(prefix):
            return region
    return "Unknown"

def compute_frontal_correlation(component_ts, raw_data, sfreq, ch_names, start_time=10, duration=10):
    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)

    eog_indices = [i for i, ch in enumerate(ch_names) if ch in ['EOG1 (FP1)', 'EOG2 (FP2)']]
    if not eog_indices:
        return np.nan

    ica_segment = component_ts[start_sample:end_sample]
    raw_segment = raw_data[eog_indices, start_sample:end_sample]
    
    corrs = [np.corrcoef(ica_segment, raw_segment[i])[0, 1] for i in range(len(eog_indices))]
    return np.mean(np.abs(corrs))


def fast_mi_matrix(sources, eeg, n_bins=16):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    sources_disc = est.fit_transform(sources.T).T
    eeg_disc = est.fit_transform(eeg.T).T

    mi = np.zeros((sources.shape[0], eeg.shape[0]))
    for i in range(sources.shape[0]):
        for j in range(eeg.shape[0]):
            mi[i, j] = mutual_info_score(sources_disc[i], eeg_disc[j])
    return mi


def plot_clean_signals_with_labels(cleaned_data, times, channel_names, offset=100):
    """
    Plots cleaned EEG signals with channel names labeled on the left.

    Parameters:
        cleaned_data (np.ndarray): EEG data (n_channels, n_times), in µV
        times (np.ndarray): Time vector in seconds
        channel_names (list): List of channel names
        offset (int): Vertical offset between channel traces
    """
    plt.figure(figsize=(14, 8))
    n_channels = len(channel_names)

    for i in range(n_channels):
        y_offset = i * offset
        plt.plot(times, cleaned_data[i] + y_offset, color='black', linewidth=0.5)
        plt.text(times[0] - 1, y_offset, channel_names[i], va='center', ha='right', fontsize=8)

    plt.yticks(np.arange(n_channels) * offset, [])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + offset (µV)")
    plt.title("Cleaned EEG Signals")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()


def compute_ica(raw, sfreq):
    
    data = raw.copy().drop_channels(['EOG1 (FP1)', 'EOG2 (FP2)'])
    
    
    # Perform ICA
    ica = ICA(n_components=int(len(data.ch_names)-1), random_state=97, max_iter=800)
    ica.fit(data)
    
    # Apply ICLabel to classify the ICA components
    labels = label_components(data, ica, method='iclabel')
    ica.labels_ = labels['labels'] #if you want to attach the labels to the ica object
    scores = labels['y_pred_proba'] #class probabilities per component
    
    # Visualize the ICA components
    #ica.plot_components()
    
    # ---------------- ICA components info ------------------------

    # Define frequency bands
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 100)
    }
    
    # Create output folder
    output_dir = r'C:\Users\msedo\Documents\CCV Stefano'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get ICA sources as a NumPy array
    sources_array = ica.get_sources(data).get_data()
    
    #--------------------- MI data -------------------------------------------- 
    
    sources_sub = sources_array[:, ::10]
    eeg_sub = data.get_data()[:, ::10]
    mi_matrix = fast_mi_matrix(sources_sub, eeg_sub)

    #--------------------------------------------------------------------------        

    times = data.times
    components = ica.get_components()
    ch_names = data.ch_names
    
    # Analysis loop per component
    summary = []
    
    for i in range(ica.n_components_):
        ts = sources_array[i]
        
        # Peak-to-peak amplitude (µV)
        ptp = np.ptp(ts) * 1e6
        
        # Welch computation for frequency power
        power_freqs, power = welch(ts, fs=sfreq, nperseg=2048)
        
        # Dominant frequency
        dom_freq = power_freqs[np.argmax(power)]
    
        # Frequency band powers
        band_powers = {
            band: round(np.trapz(power[(power_freqs >= low) & (power_freqs <= high)],
                                 power_freqs[(power_freqs >= low) & (power_freqs <= high)]), 2)
            for band, (low, high) in bands.items()
        }
        
        max_band = max(band_powers, key=band_powers.get)
        
        # Top 3 channels contributing to the component
        weights = components[:, i]
        abs_weights = np.abs(weights)
        top_indices = abs_weights.argsort()[-3:][::-1]
        top_channels = [ch_names[idx] for idx in top_indices]
    
        # Dominant region
        max_ch = ch_names[np.argmax(abs_weights)]
        region = infer_region(max_ch)
        
        # Infer regions of the top 3 channels
        top_regions = [infer_region(ch) for ch in top_channels]
        
        # Count how many are in the same region as max_ch
        same_region_count = top_regions.count(region)
        
        
        eog_ch_names = ['EOG1 (FP1)', 'EOG2 (FP2)']
        raw_data_eog = raw_data_eog = raw.get_data(picks=eog_ch_names) * 1e6
        
        frontal_corr = compute_frontal_correlation(ts, raw_data_eog, sfreq, eog_ch_names, start_time=10, duration=10)
    
        # Classification confidence
        label = ica.labels_[i]
        conf = labels['y_pred_proba'][i][0] if isinstance(labels['y_pred_proba'][i], (list, np.ndarray)) else labels['y_pred_proba'][i]
        
        # Plotting component time series, topography and power spectrum (plots saved in separate files)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        axs[0].plot(times, ts)
        axs[0].set_title(f'Component {i} Time Series')
        plot_topomap(weights, data.info, axes=axs[1], show=False)
        axs[1].set_title(f'Component {i} Topography')
        axs[2].semilogy(power_freqs, power)
        axs[2].set_xlim(0, 100)
        axs[2].set_title('Power Spectrum')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ica_component_{i}_Stefano_CCV.png'), dpi=300)


        # ------------ MI values ----------------------------------------------
        
        mi_values = mi_matrix[i]
        mean_mi = np.mean(mi_values)
        max_mi = np.max(mi_values)
        top_mi_channel = ch_names[np.argmax(mi_values)]
        
        # -------------- ECG correlation components --------------------------



        # ---------- Summary --------------------------------------------------
        
        summary.append({
            "Component": i,
            "Predicted Label": label,
            "Confidence": round(conf, 3),
            "Peak-to-Peak (µV)": round(ptp, 2),
            "Dominant Frequency (Hz)": round(dom_freq, 2),
            "Delta Power": band_powers["Delta"],
            "Theta Power": band_powers["Theta"],
            "Alpha Power": band_powers["Alpha"],
            "Beta Power": band_powers["Beta"],
            "Gamma Power": band_powers["Gamma"],
            "Max Power Band": max_band,
            "Top Channels": ", ".join(top_channels),
            "Max Channel": max_ch,
            "Region": region,
            "Same Region": same_region_count,
            "Mean MI": round(mean_mi, 4),
            "Max MI": round(max_mi, 4),
            "Top MI Channel": top_mi_channel,
            "FC": frontal_corr
        })
        
   
    # Save summary
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    df.to_excel(os.path.join(output_dir, f'Stefano_CCV_{prueba}.xlsx'), index=True)
    

    #--------------------- ICA component Rejection ----------------------------------

    custom_reject = []
    for comp in summary:
        if (comp["Predicted Label"] == 'eye blink' or comp["Predicted Label"] == 'other') :
            if (
                (comp["Confidence"] > 0.5 or comp["FC"] > 0.3) and
                comp["Dominant Frequency (Hz)"] < 4 and
                comp["Max Power Band"] == "Delta" and
                (comp["Region"] == "Frontal" or comp["Top MI Channel"].startswith(("Fp", "AF", "F")) or comp["FC"] == np.max(df["FC"]))
            ):
                custom_reject.append(comp["Component"])

                
        elif comp["Predicted Label"] == 'muscle artifact':
            if (
                comp["Confidence"] > 0.5 and
                (comp["Max Power Band"] == "Beta" or comp["Max Power Band"] == "Gamma")
            ):
                custom_reject.append(comp["Component"])
                
                
        elif comp["Predicted Label"] == 'channel noise':
            custom_reject.append(comp["Component"])        


    # Apply ICA
    ica.exclude = custom_reject

    # Apply ICA to the EEG-only data
    cleaned_raw = ica.apply(raw.copy())
    
    print("Rejected components:", custom_reject)

    print("ica.exclude =", ica.exclude)


    print(f"Rejected components: {ica.exclude}")
    for i in ica.exclude:
        print(f"Component {i}: Label = {labels['labels'][i]}, Confidence = {labels['y_pred_proba'][i]}")

    
    # Before ICA cleaning
    raw.plot(title="Re-Referenced EEG pre ICA", n_channels=49)

    # After ICA cleaning
    cleaned_raw.plot(title="Cleaned EEG (ICA applied)", n_channels=49)
    
    
    # ------------------------ Manual Visualisation of blink component ----------------

    # Define EOG channels
    eog_channels = ['EOG1 (FP1)', 'EOG2 (FP2)']
    
    # Filter only the components in custom_reject with valid FC
    valid_fc_components = [
        comp for comp in summary
        if comp["Component"] in custom_reject and not np.isnan(comp["FC"])
    ]
    
    if valid_fc_components:
        # Select the component with highest FC from the rejected ones
        blink_component = max(valid_fc_components, key=lambda comp: comp["FC"])["Component"]
        max_fc_value = max(valid_fc_components, key=lambda comp: comp["FC"])["FC"]
        print(f"Selected rejected component with highest EOG correlation: {blink_component} (FC = {max_fc_value:.3f})")
    else:
        blink_component = None
        print("No rejected component had valid frontal correlation (FC).")
    

    scale_factor = 50
    
    # Time window
    start_time = 100  # seconds
    duration = 30    # seconds
    sfreq = data.info['sfreq']
    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)
    times = data.times[start_sample:end_sample]
    
    # ICA component time series
    ica_ts = sources_array[blink_component, start_sample:end_sample]
    
    # Get raw and cleaned EEG data
    raw_eeg_data = raw.get_data() * 1e6  # original in µV
    cleaned_eeg_data = cleaned_raw.get_data() * 1e6  # cleaned in µV
    
    # Plotting
    fig, axs = plt.subplots(len(eog_channels), 1, figsize=(12, 3 * len(eog_channels)), sharex=True)
    
    for i, ch in enumerate(eog_channels):
        idx = raw.ch_names.index(ch)
        raw_ts = raw_eeg_data[idx, start_sample:end_sample]
        clean_ts = cleaned_eeg_data[idx, start_sample:end_sample]
    
        axs[i].plot(times, raw_ts, label=f'{ch} (raw EEG)', color='black')
        axs[i].plot(times, ica_ts * scale_factor, label=f'ICA Component {blink_component} ×{scale_factor}', color='blue', alpha=0.7)
        axs[i].plot(times, clean_ts, label=f'{ch} (cleaned EEG)', color='green', linestyle='--')
        axs[i].set_ylabel("µV / scaled a.u.")
        axs[i].set_title(f"{ch} vs ICA Component {blink_component}")
        axs[i].legend()
        axs[i].grid(True)
    
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    
        
    # ------------------- Plot of clean signals overlapped with signals before ICA --------------

    # Get EEG data and times
    car_data, times = raw.get_data(return_times=True)
    cleaned_data = cleaned_raw.get_data()
    
    # Convert from Volts to microvolts
    car_data *= 1e6
    cleaned_data *= 1e6
    
    # Define vertical offset
    offset = 100
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    for i, ch_name in enumerate(raw.ch_names):
        y_offset = i * offset
        plt.plot(times, car_data[i] + y_offset, color='red', linewidth=0.5)
        plt.plot(times, cleaned_data[i] + y_offset, color='black', linewidth=0.5)
        # Add channel label (optional)
        plt.text(times[0] - 1, y_offset, ch_name, va='center', ha='right', fontsize=8)
    
    # Add labeled y-axis ticks at channel positions
    yticks = np.arange(len(raw.ch_names)) * offset
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude + offset (µV)')
    plt.title('EEG Before (Red) and After ICA (Black)')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
    
    chs = data.ch_names
    
    plot_clean_signals_with_labels(cleaned_data, times, chs)
    
    
    return summary, cleaned_raw


# asr_cleaned_segments: list of preprocessed, ASR-cleaned MNE Raw objects (one per segment)
# sfreq: sampling frequency (you can get it from each segment with segment.info['sfreq'])

all_ica_summaries = []
all_cleaned_ica = []

for seg_num, asr_segment in enumerate(asr_cleaned_segments, 1):
    print(f"\n=== Computing ICA for ASR-cleaned Segment {seg_num} ===")
    # Compute ICA on this segment
    summary, cleaned_raw = compute_ica(asr_segment, asr_segment.info['sfreq'])
    # Store results
    all_ica_summaries.append(summary)
    all_cleaned_ica.append(cleaned_raw)

 
# ====================== Segmentation of relevant phases =====================

raw_g.set_annotations(raw.annotations.copy())
cleaned_ASR_noeog = raw_g.copy().drop_channels(['EOG1 (FP1)', 'EOG2 (FP2)'])

segments = []

# Always sort annotations first
ann = raw.annotations.copy()
ann._sort()
raw.set_annotations(ann)

onsets = raw.annotations.onset
labels = raw.annotations.description

# Create segments from pairs of annotations
for i in range(len(onsets) - 1):
    t_start = onsets[i]
    t_end = onsets[i + 1]
    label = labels[i]

    # Create raw segment
    segment = cleaned_ASR_noeog.copy().crop(tmin=t_start, tmax=t_end, include_tmax=False)
    segment.info['description'] = label
    segments.append((label, segment))

# Add final segment from last annotation to end
final_label = labels[-1]
final_segment = cleaned_ASR_noeog.copy().crop(tmin=onsets[-1])
final_segment.info['description'] = final_label
segments.append((final_label, final_segment))

def plot_all_segments_full(segments):
    """
    Plot each full EEG segment in the list with annotation label as title.
    
    Parameters:
        segments (list): List of (label, raw_segment) tuples
    """
    for i, (label, segment) in enumerate(segments):
        duration = segment.times[-1]
        print(f"\n📊 Plotting segment {i + 1}/{len(segments)} — Label: '{label}' — Duration: {duration:.2f}s")

        # Ensure EEG channel type is recognized
        eeg_picks = mne.pick_types(segment.info, eeg=True)
        if len(eeg_picks) == 0:
            print(f"⚠️ No EEG channels found in segment '{label}'. Skipping.")
            continue

        segment.plot(
            title=f"Segment {i+1}: {label}",
            picks='eeg',
            scalings='auto',
            duration=duration,
            n_channels=len(segment.ch_names),
            block=True
        )

plot_all_segments_full(segments)


# ------------------------- Apply ICA to segments -----------------------------

ica_applied_segments = [] 

for i, (label, segment) in enumerate(segments):
    cleaned_segment = ica_model.apply(segment.copy())
    ica_applied_segments.append((label, cleaned_segment)) 

    # Get EEG data before and after ICA 
    raw_data = segment.get_data(picks="eeg") * 1e6  # in µV 
    cleaned_data = cleaned_segment.get_data(picks="eeg") * 1e6 
    times = segment.times 
    ch_names = segment.ch_names 
    offset = 100  # µV vertical spacing between channels 
     
    # Plot 
    plt.figure(figsize=(14, 6)) 
    for ch_idx in range(len(ch_names)): 
        y_offset = ch_idx * offset 
        plt.plot(times, raw_data[ch_idx] + y_offset, color='red', linewidth=0.5, label='Raw' if ch_idx == 0 else "") 
        plt.plot(times, cleaned_data[ch_idx] + y_offset, color='black', linewidth=0.5, label='ICA Cleaned' if ch_idx == 0 else "") 
        plt.text(times[0] - 1, y_offset, ch_names[ch_idx], va='center', ha='right', fontsize=7) 
     
    plt.title(f"Segment {i + 1}: {label} — Raw (Red) vs ICA Cleaned (Black)") 
    plt.xlabel("Time (s)") 
    plt.ylabel("Amplitude + offset (µV)") 
    plt.legend(loc="upper right") 
    plt.tight_layout() 
    plt.grid(True, axis='x') 
    plt.show()
    
# ------------------------ Refining selected segments -------------------------


def find_clean_segment_with_burst_check(data, sfreq, segment_duration_sec=600, step_sec=30,
                                        z_thresh=4.0, ptp_thresh_uV=150, burst_fraction_thresh=0.4,
                                        subwindow_sec=20, subwindow_burst_fraction=0.2, max_subwindows_with_burst=3):
    """
    Advanced clean segment selection using variance, ptp, and burst rejection over short subwindows.

    Parameters:
        data (np.ndarray): EEG data (n_channels, n_samples)
        sfreq (float): Sampling frequency
        segment_duration_sec (int): Duration of the segment to evaluate (in seconds)
        step_sec (int): Sliding step size between segments (in seconds)
        z_thresh (float): Z-score threshold for variance-based outlier rejection
        ptp_thresh_uV (float): Peak-to-peak amplitude threshold in µV
        burst_fraction_thresh (float): Maximum fraction of channels allowed to be bursty
        subwindow_sec (int): Duration of each subwindow inside the segment
        subwindow_burst_fraction (float): Channel burst threshold inside each subwindow
        max_subwindows_with_burst (int): Max number of subwindows allowed to contain bursts

    Returns:
        best_start_time (float): Start time of cleanest segment
        start_times (list): All tested segment start times
        quality_scores (list): Quality scores per segment (lower is better)
    """
    n_channels, n_samples = data.shape
    segment_samples = int(segment_duration_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    subwindow_samples = int(subwindow_sec * sfreq)

    cleanest_score = np.inf
    best_start_sample = 0
    quality_scores = []
    start_times = []

    for start in range(0, n_samples - segment_samples + 1, step_samples):
        end = start + segment_samples
        segment = data[:, start:end]
        var_per_channel = np.var(segment, axis=1)
        z_var = zscore(var_per_channel)

        ptp_per_channel = np.ptp(segment, axis=1) * 1e6
        burst_fraction = np.sum(ptp_per_channel > ptp_thresh_uV) / n_channels

        # Subwindow burst detection
        n_subwindows = segment_samples // subwindow_samples
        bursty_subwindows = 0

        for i in range(n_subwindows):
            sw_start = i * subwindow_samples
            sw_end = sw_start + subwindow_samples
            subsegment = segment[:, sw_start:sw_end]
            ptp_sw = np.ptp(subsegment, axis=1) * 1e6
            burst_fraction_sw = np.sum(ptp_sw > ptp_thresh_uV) / n_channels
            if burst_fraction_sw > subwindow_burst_fraction:
                bursty_subwindows += 1

        if (np.any(np.abs(z_var) > z_thresh) or
            burst_fraction > burst_fraction_thresh or
            bursty_subwindows > max_subwindows_with_burst):
            quality = np.inf
        else:
            quality = np.mean(np.abs(z_var)) + burst_fraction + 0.1 * bursty_subwindows

        quality_scores.append(quality)
        start_times.append(start / sfreq)

        if quality < cleanest_score:
            cleanest_score = quality
            best_start_sample = start

    best_start_time = best_start_sample / sfreq
    return best_start_time, start_times, quality_scores

def refine_segments(ica_applied_segments, sfreq, min_duration_sec=120): 
    
    refined_segments = [] 

    for label, segment in segments: 
        duration = segment.times[-1] 
        print(f"\n🔍 Processing: {label} — Duration: {duration:.2f} seconds") 
     
        if duration < min_duration_sec: 
            # Keep segment as-is 
            refined_segments.append((label, segment)) 
            print("🟢 Used full segment (shorter than 2 minutes).") 
        else: 
            # Extract data and apply cleanest segment search 
            data = segment.get_data() 
            best_start_time, _, _ = find_clean_segment_with_burst_check( 
                data, sfreq, segment_duration_sec=min_duration_sec 
            ) 
     
            best_segment = segment.copy().crop( 
                tmin=best_start_time, 
                tmax=best_start_time + min_duration_sec 
            ) 
            best_segment.info['description'] = label + " (2min best)" 
            refined_segments.append((label, best_segment)) 
            print(f"✅ Selected best 2-minute sub-segment starting at {best_start_time:.2f}s.") 
            
        cleaned_data = segment.get_data(picks="eeg") * 1e6  # in µV 
        times = segment.times 
        
        # Plot 
        plt.figure(figsize=(14, 6)) 
        for ch_idx in range(len(ch_names)): 
            y_offset = ch_idx * offset 
            plt.plot(times, cleaned_data[ch_idx] + y_offset, color='black', linewidth=0.5) 
            plt.text(times[0] - 1, y_offset, ch_names[ch_idx], va='center', ha='right', fontsize=7) 
         
        plt.title(f"Segment {i + 1}: {label} 2 min clean segment") 
        plt.xlabel("Time (s)") 
        plt.ylabel("Amplitude + offset (µV)") 
        plt.legend(loc="upper right") 
        plt.tight_layout() 
        plt.grid(True, axis='x') 
        plt.show()

     
    return refined_segments 

refined_segments = refine_segments(ica_applied_segments, sfreq=raw_clean_asr.info['sfreq'])

# ============== Interpolate Bad Channels After ICA ====================

interpolated_segments = []

for i, (label, segment) in enumerate(refined_segments):
        
    # Copy full segment with all channels
    interpolated_raw = raw.copy()
    interpolated_raw.resample(256)
    
    # Replace good channels with ICA-cleaned data
    ica_data = segment.get_data()
    for ch in segment.ch_names:
        interpolated_raw._data[interpolated_raw.ch_names.index(ch)] = ica_data[segment.ch_names.index(ch)]
        
    # Mark bads and interpolate
    interpolated_raw.info['bads'] = bad_channels
    interpolated_raw.interpolate_bads(reset_bads=True, mode='accurate')
    
    interpolated_segments.append((label, interpolated_raw)) 
    
    # Sanity check
    cleaned_data = segment.get_data()
    interp_data = interpolated_raw.copy().pick(good_channels).get_data()
    
    diff = np.abs(cleaned_data - interp_data)
    max_diff = np.max(diff)
    
    print(f"Max absolute difference between good channel signals: {max_diff:.3e} µV")
    
    if max_diff < 1e-10:
        print("✅ Good channels match exactly after interpolation.")
    else:
        print("⚠️ Good channels do NOT match — something altered them.")
        



# Get data and time vector
interp_data_all, times = interpolated_raw.get_data(return_times=True)
ch_names = interpolated_raw.ch_names

# Convert to microvolts
interp_data_all *= 1e6
offset = 100  # µV vertical spacing between channels

# Plot
plt.figure(figsize=(14, 8))
for i, ch_data in enumerate(interp_data_all):
    plt.plot(times, ch_data + i * offset, color='black', linewidth=0.5)
    plt.text(times[0] - 1, i * offset, ch_names[i], va='center', ha='right', fontsize=8)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude + offset (µV)")
plt.title("Full EEG After ICA and Interpolation (Stacked)")
plt.yticks(np.arange(len(ch_names)) * offset, ch_names)
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()


# ======= 2s Epoch generation and rejection of remaining artifacts ===========


# ==================== FREQUENCY ANALYSIS =================================

# ---------------- Time-Frequency Representation --------------------------

# ----------------- Grouping channels by region ---------------------------

# Define regions (exclude EOG region from analysis)
regions = {
    'Frontal':  ['F3', 'F4', 'F7', 'F8', 'Fz'],
    'Central':  ['C3', 'C4', 'Cz'],
    'Parietal': ['P3', 'P4', 'P7', 'P8', 'Pz'],
    'Temporal': ['T7', 'T8'],
    'Occipital': ['O1', 'O2', 'Oz']
    # 'EOG': ['EOG1 (FP1)', 'EOG2 (FP2)']  # EOG excluded
}

# Get channel indices for each region
region_picks = {
    region: mne.pick_channels(raw.info['ch_names'], ch_names) 
    for region, ch_names in regions.items()
}

# Compute the average signal for each region across its channels
region_signals = {}
for region, picks in region_picks.items():
    data, _ = cleaned_raw[picks]                  # shape: (len(picks), n_times)
    region_avg = np.mean(data, axis=0)    # average across channels
    region_signals[region] = region_avg

    

# ------------------ TFR Morlet wavelet --------------------------------



# Stack region signals into an array of shape (n_regions, n_times)
regions_list = list(region_signals.keys())  # ensure consistent order
data_matrix = np.stack([region_signals[r] for r in regions_list], axis=0)  # shape: (5, n_times)

# Define frequency range and wavelet settings
freqs = np.arange(1.0, 50.0, 1.0)  # frequencies from 1 Hz to 49 Hz
n_cycles = 6                      # 6 cycles for each frequency band

# tfr_array_morlet expects shape (n_epochs, n_channels, n_times)
epoch_data = data_matrix[np.newaxis, ...]  # add an "epoch" dimension -> shape: (1, 5, n_times)
power = tfr_array_morlet(epoch_data, sfreq=cleaned_raw_noeog.info['sfreq'], freqs=freqs,
                         n_cycles=n_cycles, output='power')  # shape: (1, 5, n_freqs, n_times)
power = power[0]  # drop the epoch dimension -> shape: (5, n_freqs, n_times)


# Convert power to log-scale (dB-like) for better visualization
log_power = np.log10(power + 1e-12)


# Get annotation times (seconds from start) and descriptions
event_times = cleaned_raw_noeog.annotations.onset            # e.g. array([10.0, 1440.0, 1860.0])
event_labels = list(cleaned_raw_noeog.annotations.description)


regions = regions_list  
n_regions = len(regions)
fig, axes = plt.subplots(n_regions, 1, figsize=(10, 10), sharex=True)

# Determine a common color scale across all regions
vmin = np.min(log_power)
vmax = np.max(log_power)

for i, region in enumerate(regions):
    ax = axes[i]
    # Select the log-power for this region (shape: n_freqs x n_times)
    region_tfr = log_power[i]  
    # Plot the TFR as an image
    img = ax.imshow(region_tfr, origin='lower', aspect='auto',
                    extent=[0, cleaned_raw_noeog.n_times/raw.info['sfreq'], freqs[0], freqs[-1]],
                    vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(region)
    # Mark annotation events with vertical lines
    for t, label in zip(event_times, event_labels):
        ax.axvline(x=t, color='red', linestyle='--', linewidth=1.5)  # vertical line at event time
        # Optionally, add a text label for the event above the line
        ax.text(t + 2, freqs[-1] - 1, label, rotation=90, verticalalignment='top',
                color='red', fontsize=8)




# ============================= CAR RE-REFERENCING ===========================================
'''
# Create a copy of the raw object
raw_car = raw_clean_segment.copy()

# Drop bad channels entirely
raw_car.pick_channels(good_channels)

# Apply average reference to the good channels only
raw_car.set_eeg_reference(ref_channels='average', projection=False)

# Get data and times for plotting
car_data, times = raw_car.get_data(return_times=True)


plt.figure(figsize=(14, 6))
for i, ch_data in enumerate(car_data):
    plt.plot(times, ch_data * 1e6 + i * 100, label=raw_car.ch_names[i])  # µV scaling + vertical offset

plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset (µV)')
plt.title('Good EEG Channels After Common Average Referencing (CAR)')
plt.legend(loc='upper right', fontsize='x-small', ncol=2)
plt.tight_layout()
plt.show()
'''



# ============== ICA evaluation through band power comparison =================


def compare_band_power_by_region(raw_before, raw_after, bands=None):
    """
    Compare average band power per brain region before vs after ICA,
    with percent change annotations, all in a single subplot figure.

    Parameters:
        raw_before (mne.io.Raw): EEG before ICA
        raw_after (mne.io.Raw): EEG after ICA
        bands (dict): Optional frequency bands
    """
    if bands is None:
        bands = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 80)
        }

    sfreq = raw_before.info['sfreq']
    ch_names = raw_before.ch_names
    data_b = raw_before.get_data(picks="eeg") * 1e6
    data_a = raw_after.get_data(picks="eeg") * 1e6

    region_channels = defaultdict(list)
    for i, ch in enumerate(ch_names):
        region = infer_region(ch)
        region_channels[region].append(i)

    band_names = list(bands.keys())
    n_bands = len(band_names)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5), sharey=False)

    if n_bands == 1:
        axes = [axes]

    for idx, (band_name, (low, high)) in enumerate(bands.items()):
        region_power_b = {}
        region_power_a = {}

        for region, indices in region_channels.items():
            pw_b, pw_a = [], []
            for i in indices:
                freqs_b, psd_b = welch(data_b[i], fs=sfreq, nperseg=4 * sfreq)
                freqs_a, psd_a = welch(data_a[i], fs=sfreq, nperseg=4 * sfreq)

                mask_b = (freqs_b >= low) & (freqs_b <= high)
                mask_a = (freqs_a >= low) & (freqs_a <= high)

                pw_b.append(np.trapz(psd_b[mask_b], freqs_b[mask_b]))
                pw_a.append(np.trapz(psd_a[mask_a], freqs_a[mask_a]))

            region_power_b[region] = np.mean(pw_b)
            region_power_a[region] = np.mean(pw_a)

        regions = sorted(region_power_b.keys())
        values_b = [region_power_b[r] for r in regions]
        values_a = [region_power_a[r] for r in regions]
        x = np.arange(len(regions))
        width = 0.35

        ax = axes[idx]
        bars_b = ax.bar(x - width / 2, values_b, width, label='Before ICA', color='gray')
        bars_a = ax.bar(x + width / 2, values_a, width, label='After ICA', color='green')

        # Annotate percent change above After bars
        for i, (b, a) in enumerate(zip(values_b, values_a)):
            if b != 0:
                change = 100 * (a - b) / b
                ax.text(x[i] + width / 2, a + max(values_b) * 0.05,
                        f"{change:+.1f}%", ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45)
        ax.set_title(f"{band_name}")
        ax.grid(True, axis='y')

        if idx == 0:
            ax.set_ylabel("Mean Band Power (µV²)")
    
    # Use constrained layout to auto-handle spacing
    fig.set_constrained_layout(True)
    
    # Add super title
    fig.suptitle("Band Power by Region (Before vs After ICA)", fontsize=16)
    
    # Add legend outside below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)

    plt.show()


compare_band_power_by_region(cleaned_raw_noeog, cleaned_raw_noeog)



# ======================= Frequency band analysis =========================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from collections import defaultdict

def plot_band_power_by_region(raw, bands=None):
    """
    Plot average band power per brain region from a single raw EEG object.

    Parameters:
        raw (mne.io.Raw): EEG data (e.g., after ICA or ASR)
        bands (dict): Optional dictionary of frequency bands
    """
    if bands is None:
        bands = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 80)
        }

    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    data = raw.get_data(picks="eeg") * 1e6  # Convert V → µV

    # Group channels by region
    region_channels = defaultdict(list)
    for i, ch in enumerate(ch_names):
        region = infer_region(ch)  # You must define this function
        region_channels[region].append(i)

    band_names = list(bands.keys())
    n_bands = len(band_names)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5), sharey=False)

    if n_bands == 1:
        axes = [axes]

    for idx, (band_name, (low, high)) in enumerate(bands.items()):
        region_power = {}

        for region, indices in region_channels.items():
            powers = []
            for i in indices:
                freqs, psd = welch(data[i], fs=sfreq, nperseg=4 * sfreq)
                mask = (freqs >= low) & (freqs <= high)
                power = np.trapz(psd[mask], freqs[mask])
                powers.append(power)
            region_power[region] = np.mean(powers)

        # Bar plot
        regions = sorted(region_power.keys())
        values = [region_power[r] for r in regions]
        x = np.arange(len(regions))

        ax = axes[idx]
        bars = ax.bar(x, values, width=0.5, color='skyblue')
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45)
        ax.set_title(f"{band_name}")
        ax.set_ylabel("Mean Band Power (µV²)")
        ax.grid(True, axis='y')

    fig.set_constrained_layout(True)
    fig.suptitle("Band Power by Region", fontsize=16)
    plt.show()

plot_band_power_by_region(cleaned_raw_noeog)