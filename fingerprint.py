"""
fingerprint.py
---
Audio fingerprinting library with optimised pathway, for a scaleable solution
Utilizes, byte packing hashes (cache efficiency), vectorised binary search (O(log N) search)
compressed, scaled down integer types for storage (to reduce memory footprint and faster access).
"""

import os #Standard Python library for scanning directories and handling file paths.
import numpy as np
import librosa #Audio processing library used here for loading audio and computing Short-Time Fourier Transforms (STFT)
from scipy.ndimage import maximum_filter

SR = 22050 #Target audio frequency is set to 8,000 samples per second
SONG_FOLDER = "songs"
DB_OUTPUT_PATH = "song_database.npz" #using numpy native format to utitlize the superfast vector operations. 

#loads the audio through librosa
def load_audio(path, target_sr=8000):
    try:
        data, sr = librosa.load(path, sr=target_sr, mono=True)
        max_val = np.max(np.abs(data)) #Computes absolute peak amplitude across all samples.
        if max_val > 0:
            data = data / (max_val + 1e-9)
        return data, sr
    except Exception as e:
        raise RuntimeError(f"Error loading file {path}: {e}")

#gets the list of songs
def list_songs(folder):
    if not os.path.isdir(folder):
        return []
    return [(os.path.splitext(f)[0], os.path.join(folder, f)) 
            for f in sorted(os.listdir(folder)) if f.lower().endswith(".mp3")]

#spectrogram generation using librosa stft, as it is faster in cpu than scipy.signal.stft
def spectrogram(audio, sr, nperseg=1024, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    hop_length = nperseg - noverlap
    stft_matrix = librosa.stft(audio, n_fft=nperseg, hop_length=hop_length, window='hann')
    Sxx = np.abs(stft_matrix)
    f = librosa.fft_frequencies(sr=sr, n_fft=nperseg)
    t = librosa.frames_to_time(np.arange(Sxx.shape[1]), sr=sr, hop_length=hop_length)
    return f, t, Sxx

#filtering the peak values to get the fingerprint
def find_peaks_2d(Sxx, amp_min_db=-25, neighborhood=(20, 5)):
    Sxx_db = 20 * np.log10(Sxx + 1e-9)
    Sxx_db -= Sxx_db.max()
    local_max = maximum_filter(Sxx_db, size=neighborhood) == Sxx_db
    above_thresh = Sxx_db > amp_min_db
    freq_idx, time_idx = np.where(local_max & above_thresh)
    return np.column_stack((freq_idx, time_idx)).astype(np.int32) if len(freq_idx) > 0 else np.empty((0, 2), dtype=np.int32)

#optimising the hash size, by byte packing into int64 (a bit of overkill tbh)
def pack_hash(f1, f2, dt):
    return (np.uint64(f1) << 32) | (np.uint64(f2) << 16) | np.uint64(dt)

#hash generation function using peak values.
#can generate paired hashes (f1, f2, dt) or single-peak hashes (f1 only) when use_pairs=False.
def generate_hashes(peaks, fan_out=5, min_dt=1, max_dt=50, use_pairs=True):
    if len(peaks) == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int32)
    
    peaks = peaks[np.argsort(peaks[:, 1])]  
    hashes = []
    anchor_times = []
    
    if not use_pairs:
        #single peak hash: just store frequency bin as hash
        for f1, t1 in peaks:
            hashes.append(np.uint64(f1))
            anchor_times.append(t1)
        return np.array(hashes, dtype=np.uint64), np.array(anchor_times, dtype=np.int32)

    for i, (f1, t1) in enumerate(peaks):
        count = 0
        for f2, t2 in peaks[i + 1:]:
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            hashes.append(pack_hash(f1, f2, dt))
            anchor_times.append(t1)
            count += 1
            if count >= fan_out:
                break
                
    return np.array(hashes, dtype=np.uint64), np.array(anchor_times, dtype=np.int32)

#adds white noise to an audio signal at a target signal-to-noise ratio (in dB).
#useful for stress-testing fingerprint robustness against noisy recordings.
def add_noise(audio, snr_db=20):
    signal_power = np.mean(audio ** 2)
    #SNR(dB) = 10*log10(signal_power / noise_power), solve for noise_power
    noise_power = signal_power / (10 ** (snr_db / 10) + 1e-9)
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return audio + noise

#simulates a pitch shift by resampling (the classic "speeding up a tape" effect).
#shifts by `semitones` semitones — also changes tempo proportionally.
def pitch_shift_resample(audio, sr, semitones):
    if semitones == 0:
        return audio.copy()
    #ratio > 1 means higher pitch (fewer samples for same duration)
    ratio = 2 ** (semitones / 12)
    new_length = int(len(audio) / ratio)
    #np.interp maps the resampled indices back onto the original signal
    resampled = np.interp(
        np.linspace(0, len(audio) - 1, new_length),
        np.arange(len(audio)),
        audio,
    )
    return resampled

#builds the in-memory fingerprint database from a list of (name, audio, sr) tuples.
#returns sorted hash/song/anchor arrays plus metadata, ready for match_query.
def build_database(songs_list, nperseg=1024, fan_out=5, use_pairs=True):
    song_names = []
    collected_hashes = []
    collected_songs = []
    collected_anchors = []
    peaks_by_song = {}

    for idx, (name, audio, sr) in enumerate(songs_list):
        song_names.append(name)

        _, _, Sxx = spectrogram(audio, sr, nperseg=nperseg)
        peaks = find_peaks_2d(Sxx)
        peaks_by_song[f"peaks_{idx}"] = peaks

        hashes, anchors = generate_hashes(peaks, fan_out=fan_out, use_pairs=use_pairs)

        if len(hashes) > 0:
            collected_hashes.append(hashes)
            # Optimization: Cast to np.uint8 since library has <= 256 songs
            collected_songs.append(np.full(len(hashes), idx, dtype=np.uint8))
            collected_anchors.append(anchors.astype(np.int32))

    #concatenate per-song arrays into one flat database, then sort by hash
    #so match_query can binary-search via np.searchsorted in O(log N).
    if collected_hashes:
        db_hashes = np.concatenate(collected_hashes)
        db_songs = np.concatenate(collected_songs)
        db_anchors = np.concatenate(collected_anchors)

        sort_idx = np.argsort(db_hashes)
        db_hashes = db_hashes[sort_idx]
        db_songs = db_songs[sort_idx]
        db_anchors = db_anchors[sort_idx]

        # Optimization: Precompute per-song hash counts to skip run-time array scans
        song_hash_counts = np.bincount(db_songs, minlength=len(song_names)).astype(np.int32)
    else:
        db_hashes = np.empty(0, dtype=np.uint64)
        db_songs = np.empty(0, dtype=np.uint8)
        db_anchors = np.empty(0, dtype=np.int32)
        song_hash_counts = np.zeros(len(song_names), dtype=np.int32)

    return {
        "db_hashes": db_hashes,
        "db_songs": db_songs,
        "db_anchors": db_anchors,
        "song_names": np.array(song_names, dtype=object),
        "song_hash_counts": song_hash_counts,
        **peaks_by_song,
    }

#ya, this is the part I match query with database, the intial python loop was too slow, so switch to searchsort of numpy.
def match_query(audio, sr, db_hashes, db_songs=None, db_anchors=None, song_names=None, nperseg=1024, fan_out=5, use_pairs=True):
    #if database dict was passed as 3rd parameter, unpack it
    if isinstance(db_hashes, dict):
        database = db_hashes
        db_hashes = database["db_hashes"]
        db_songs = database["db_songs"]
        db_anchors = database["db_anchors"]
        song_names = database["song_names"]

    _, _, Sxx = spectrogram(audio, sr, nperseg=nperseg)
    q_peaks = find_peaks_2d(Sxx)
    q_hashes, q_times = generate_hashes(q_peaks, fan_out=fan_out, use_pairs=use_pairs)

    if len(q_hashes) == 0 or len(db_hashes) == 0:
        return None, {}, q_peaks, 0, 0

    q_sort_idx = np.argsort(q_hashes)
    q_hashes = q_hashes[q_sort_idx]
    q_times = q_times[q_sort_idx]

    # binary search bounds lookup via np.searchsorted to skip O(N) masks and have O(log(N)) lookup
    left_bounds = np.searchsorted(db_hashes, q_hashes, side="left")
    right_bounds = np.searchsorted(db_hashes, q_hashes, side="right")

    valid_mask = right_bounds > left_bounds
    if not np.any(valid_mask):
        return None, {}, q_peaks, 0, 0
    
    
    l_intervals = left_bounds[valid_mask]
    r_intervals = right_bounds[valid_mask]
    matched_q_times_base = q_times[valid_mask]

    segment_lengths = r_intervals - l_intervals
    total_matches = segment_lengths.sum()

    db_indices = np.repeat(l_intervals, segment_lengths) + (
        np.arange(total_matches) - np.repeat(np.cumsum(segment_lengths) - segment_lengths, segment_lengths)
    )

    matched_db_songs = db_songs[db_indices]
    matched_db_anchors = db_anchors[db_indices]
    matched_q_times = np.repeat(matched_q_times_base, segment_lengths)

    offsets = matched_db_anchors - matched_q_times
    combined_votes = (offsets.astype(np.int64) << 32) | matched_db_songs.astype(np.int64)
    unique_keys, counts = np.unique(combined_votes, return_counts=True)
    
    #build nested offset histogram: {song_name: {offset: count}} to match notebook visualization
    offset_hist = {}
    best_song_name = None
    best_score = 0
    best_offset = 0

    for key, count in zip(unique_keys, counts):
        s_idx = int(key & 0xFFFFFFFF)
        offset_val = int(key >> 32)
        s_name = song_names[s_idx]
        
        if s_name not in offset_hist:
            offset_hist[s_name] = {}
        offset_hist[s_name][offset_val] = int(count)

        if count > best_score:
            best_score = int(count)
            best_song_name = s_name
            best_offset = offset_val

    return best_song_name, offset_hist, q_peaks

#directly build the database from the songs, build once use multiple times,
#this custom database is more suitable than redis and sqlite for this particular purpose.
if __name__ == "__main__":
    print(f"Compiling structural signatures from folder: '{SONG_FOLDER}' at {SR}Hz...")
    song_files = list_songs(SONG_FOLDER)
    if not song_files:
        print(f"Error: Add .mp3 assets to '{SONG_FOLDER}' before building.")
        exit(1)

    #load every song and pair it with its name + sample rate for build_database
    songs_list = []
    for name, path in song_files:
        print(f" -> Processing structural constellations: {name}.mp3")
        audio, sr = load_audio(path, target_sr=SR)
        songs_list.append((name, audio, sr))

    save_payload = build_database(songs_list)
    np.savez(DB_OUTPUT_PATH, **save_payload)
    print(f"\nSaved optimized database with {len(save_payload['db_hashes'])} hashes to '{DB_OUTPUT_PATH}'")
