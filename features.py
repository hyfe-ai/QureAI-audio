# -*- coding: utf-8 -*-
"""
@author: George Kafentzis
Hyfe Inc.

Tools imported from PyAudio Analysis Toolkit: https://github.com/tyiannak/pyAudioAnalysis
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610
Giannakopoulos, Theodoros. "pyaudioanalysis: An open-source python library for audio signal analysis." 
PloS one 10.12 (2015): e0144610.
"""
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct


def fbf_feature_extraction(signal, fs, window, step, deltas=False):
    """
    This function implements the short-term windowing process.
    For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a np matrix.
    ARGUMENTS
        signal:         the input signal samples
        fs:             the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:           the short-term window step (in samples)
        deltas:         (opt) True/False if delta features are to be
                        computed
    RETURNS
        features (numpy.ndarray):        contains features
                                         (n_feats x numOfShortTermWindows)
        feature_names (python list):     contains feature names
                                         (n_feats x numOfShortTermWindows)
    """

    window = int(window)
    step = int(step)

    # signal normalization
    #signal = np.double(signal)
    signal = signal / (2.0 ** 15)

    signal = dc_normalize(signal)

    number_of_samples = len(signal)  # total number of samples
    current_position = 0
    count_fr = 0
    NFFT = 1024
    
    n_spectral_feats = 5
    n_prosodic_feats = 4
    n_mfcc_feats = 13
    n_fbank_feats = 40
    n_total_feats = n_spectral_feats + n_prosodic_feats + n_mfcc_feats + n_fbank_feats

    # define list of feature names
    feature_names = ["zcr", "energy", "energy_entropy", "intensity", 
                     "spcent", "spspread", "spentr", "spflux", "sprollof"]
    feature_names += ["mfcc_{0:d}".format(mfcc_i)
                      for mfcc_i in range(1, n_mfcc_feats + 1)]
    feature_names += ["fbanks_{0:d}".format(fbanks_i)
                      for fbanks_i in range(1, n_fbank_feats + 1)]

    feature_vector_prev = np.zeros((n_total_feats, 1))
    # add names for delta features:
    if deltas:
        feature_names_2 = feature_names + ["delta " + f for f in feature_names]
        feature_names = feature_names_2
        
    features = []
    # for each short-term window to end of signal
    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        # get current window
        x = signal[current_position:current_position + window]
        
        # get Hamming window
        N = len(x)
        w = np.hamming(N)
        # normalize window and apply
        w = w / np.sum(w)
        x = x * w
        
        # get fft magnitude
        fft_magnitude = abs(fft(x, NFFT))

        # normalize fft - not necessary for normalized window
        #fft_magnitude = fft_magnitude / len(fft_magnitude)
        fft_magnitude = fft_magnitude[0:int(NFFT/2)]

        # keep previous fft mag (used in spectral flux)
        if count_fr == 1:
            fft_magnitude_previous = fft_magnitude.copy()

        # update window position
        current_position = current_position + step

        feature_vector = np.zeros((n_total_feats, 1))
                
        # zero crossing rate
        feature_vector[0] = zero_crossing_rate(x)
        # short-term energy
        feature_vector[1] = energy(x)
        # short-term entropy of energy
        feature_vector[2] = energy_entropy(x)
        # intensity
        feature_vector[3] = intensity(x)
         # sp centroid/spread
        [feature_vector[4], feature_vector[5]] = spectral_centroid_spread(fft_magnitude, fs)
        # spectral entropy
        feature_vector[6] = spectral_entropy(fft_magnitude)
        # spectral flux
        feature_vector[7] = spectral_flux(fft_magnitude, fft_magnitude_previous)
        # spectral rolloff
        feature_vector[8] = spectral_rolloff(fft_magnitude, 0.90)
        # MFCCs
        mfcc_feats_end = n_total_feats - n_fbank_feats
        feature_vector[n_total_feats-n_mfcc_feats-n_fbank_feats:mfcc_feats_end, 0] = mfcc(x, fs, nb_coeff=13, nb_filt=40, nb_fft=NFFT).copy()
        # fbank energies
        fbank_feats_end = n_total_feats
        feature_vector[n_total_feats-n_fbank_feats:fbank_feats_end, 0] = filter_banks_coeff(x, fs, nb_filt=40, nb_fft=NFFT).copy()
        #mfccs = lr.feature.mfcc(y=x, sr=fs, n_mfcc=12)
        #feature_vector[n_total_feats-n_mfcc_feats:mfcc_feats_end, 0] = mfccs
        
        # if deltas == False:
        features.append(feature_vector)
        # else:
        #     # delta features
        #     if count_fr > 1:
        #         delta = feature_vector - feature_vector_prev
        #         feature_vector_2 = np.concatenate((feature_vector, delta))
        #     else:
        #         feature_vector_2 = np.concatenate((feature_vector,
        #                                             np.zeros(feature_vector.
        #                                                     shape)))
        #     feature_vector_prev = feature_vector
        #     features.append(feature_vector_2)
        
        fft_magnitude_previous = fft_magnitude.copy()

    features = np.concatenate(features, 1)

    
    return features, feature_names


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                     FEATURE EXTRACTION METHODS START                         """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def dc_normalize(sig_array):
    """Removes DC and normalizes to -1, 1 range"""
    sig_array_norm = sig_array.copy()
    sig_array_norm -= sig_array_norm.mean()
    sig_array_norm /= abs(sig_array_norm).max() + 1e-10
    return sig_array_norm


def zero_crossing_rate(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float32(count_zero) / np.float32(count - 1.0)


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float32(len(frame))


def energy_entropy(frame, n_short_blocks=10, eps = 10e-8):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy


def intensity(frame, eps = 10e-8):
    """Computes the intensity of a frame using normalized energy"""
    # Get energy
    en = energy(frame)
    # Threshold
    Pa = 2 * 10**(-5)
    # Energy over Pa^2
    inty = 10*np.log10(en/(Pa**2) + eps)
    
    return inty


def spectral_centroid_spread(fft_magnitude, sampling_rate, eps = 10e-8):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(fft_magnitude) + 1)) * \
          (sampling_rate / (2.0 * len(fft_magnitude)))

    Xt = fft_magnitude.copy()
    Xt_max = Xt.max()
    if Xt_max == 0:
        Xt = Xt / eps
    else:
        Xt = Xt / Xt_max

    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    centroid = (NUM / DEN)

    # Spread:
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)

    # Normalize:
    centroid = centroid / (sampling_rate / 2.0)
    spread = spread / (sampling_rate / 2.0)

    return centroid, spread


def spectral_entropy(signal, n_short_blocks=10, eps = 10e-8):
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy


def spectral_flux(fft_magnitude, previous_fft_magnitude, eps = 10e-8):
    """
    Computes the spectral flux feature of the current frame
    """
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum(
        (fft_magnitude / fft_sum - previous_fft_magnitude /
         previous_fft_sum) ** 2)

    return sp_flux


def spectral_rolloff(fft, c=0.90, eps = 10e-8):
    """Computes spectral roll-off"""

    # Total energy
    energy = np.sum(fft ** 2)

    # Roll off threshold
    threshold = c * energy

    # Compute cumulative energy
    cum_energy = np.cumsum(fft ** 2) + eps

    # Find the spectral roll off as the frequency position
    [roll_off, ] = np.nonzero(cum_energy > threshold)

    # Normalize
    if len(roll_off) > 0:
        roll_off = np.float32(roll_off[0]) / (float(len(fft)))
    else:
        roll_off = 0.0

    return roll_off


    '''
    Computes the Filter Bank coefficients
    '''
def filter_banks_coeff(signal, sample_rate, nb_filt=40, nb_fft=512):

    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(signal, nb_fft))

    # Power Spectrum
    pow_frames = ((1.0 / nb_fft) * (mag_frames ** 2))
    low_freq_mel = 0

    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nb_filt + 2)

    # Convert Mel to Hz
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((nb_fft + 1) * hz_points / sample_rate)

    # Calculate filter banks
    fbank = np.zeros((nb_filt, int(np.floor(nb_fft / 2 + 1))))
    for m in range(1, nb_filt + 1):

        # left
        f_m_minus = int(bin[m - 1])

        # center
        f_m = int(bin[m])

        # right
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)

    # Numerical Stability
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

    # dB
    filter_banks = 20 * np.log10(filter_banks)

    return filter_banks

'''
Computes the MFCCs
'''
def mfcc(signal, sample_rate, nb_coeff=12, nb_filt=40, nb_fft=512, return_fbank=False):

    # Apply filter bank on spectogram
    filter_banks = filter_banks_coeff(signal, sample_rate, nb_filt=nb_filt, nb_fft=nb_fft)

    # Compute MFCC coefficients
    mfcc = dct(filter_banks, type=2, axis=-1, norm='ortho')[1: (nb_coeff + 1)]

    # Return MFFCs and Filter banks coefficients
    if return_fbank is True:
        return np.concatenate((mfcc, filter_banks))
    else:
        return mfcc