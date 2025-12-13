import os
os.environ['PYDUB_UTILS_QUIET'] = "1"

import streamlit as st
import asyncio
import pyaudio
import wave
import tempfile
import sys
import warnings
import base64
import numpy as np
import json
import threading
import time
from datetime import datetime, timedelta
from io import BytesIO
from pydub import AudioSegment
from shazamio import Shazam
import logging
from pathlib import Path
import csv
import librosa
import pickle
import hashlib
from scipy.signal import find_peaks
#--- AI Imports begin
import requests
import random
from collections import Counter
#---- AI imports End

sys.stdout = open(os.devnull, 'w')  # Suppress all stdout

# Enhanced warning and logging suppression
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress pydub/ffmpeg warnings
logging.getLogger('pydub.converter').setLevel(logging.ERROR)
logging.getLogger('pydub.utils').setLevel(logging.ERROR)

# Suppress streamlit warnings
logging.getLogger('streamlit').setLevel(logging.ERROR)
st.set_option('client.showErrorDetails', False)


# =============================================================================
# Offline Fingerprinting System (Moved to top to fix NameError)
# =============================================================================

def load_offline_database():
    """Load offline fingerprint database from file"""
    try:
        if os.path.exists(st.session_state.offline_db_file):
            with open(st.session_state.offline_db_file, 'rb') as f:
                st.session_state.offline_database = pickle.load(f)
            st.session_state.offline_db_loaded = True
        else:
            st.session_state.offline_db_loaded = False
    except Exception as e:
        st.error(f"Error loading offline database: {str(e)}")
        st.session_state.offline_database = {}
        st.session_state.offline_db_loaded = False

def save_offline_database():
    """Save offline fingerprint database to file"""
    try:
        with open(st.session_state.offline_db_file, 'wb') as f:
            pickle.dump(st.session_state.offline_database, f)
        return True
    except Exception as e:
        st.error(f"Error saving offline database: {str(e)}")
        return False

def update_offline_db_stats():
    """Update statistics about the offline database"""
    try:
        db = st.session_state.offline_database
        st.session_state.offline_db_stats['songs'] = len(db)
        
        artists = set()
        total_fingerprints = 0
        
        for song_id, song_data in db.items():
            artists.add(song_data['artist'])
            total_fingerprints += len(song_data['fingerprint'])
        
        st.session_state.offline_db_stats['artists'] = len(artists)
        st.session_state.offline_db_stats['fingerprints'] = total_fingerprints
    except:
        pass

def extract_audio_fingerprint(audio_file, sr=22050):
    """Extract audio fingerprint using spectral peaks"""
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr, duration=30)
        
        # Compute spectrogram
        stft = librosa.stft(y, hop_length=512, n_fft=2048)
        magnitude = np.abs(stft)
        
        # Find spectral peaks
        fingerprint = []
        for t in range(magnitude.shape[1]):
            spectrum = magnitude[:, t]
            peaks, _ = find_peaks(spectrum, height=np.mean(spectrum) * 2)
            
            # Take top peaks
            if len(peaks) > 0:
                peak_values = spectrum[peaks]
                top_peaks = peaks[np.argsort(peak_values)[-5:]]  # Top 5 peaks
                fingerprint.extend([(int(peak), int(t)) for peak in top_peaks])
        
        return fingerprint
    except Exception as e:
        st.error(f"Error extracting fingerprint: {str(e)}")
        return None

async def identify_song_for_offline_db(audio_file):
    """Identify song using Shazam API to get full metadata for offline DB"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            shazam = Shazam()
            
            if not os.path.exists(audio_file):
                return None
                
            file_size = os.path.getsize(audio_file)
            if file_size < 1000:
                return None
            
            for attempt in range(2):
                try:
                    with open(audio_file, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    
                    out = await shazam.recognize(audio_data)
                    
                    if out and 'track' in out:
                        return out
                    elif out:
                        return None
                        
                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(1)
                        continue
            
            return None
            
    except Exception:
        return None

def add_song_to_offline_db(audio_file, song_name, artist="Unknown"):
    """Add a song to the offline database with full metadata from Shazam"""
    fingerprint = extract_audio_fingerprint(audio_file)
    if not fingerprint:
        return False
    
    # Identify song using Shazam to get full metadata
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        shazam_result = loop.run_until_complete(identify_song_for_offline_db(audio_file))
    finally:
        loop.close()
    
    # Extract metadata
    album = "Unknown"
    released = "Unknown"
    genre = "Unknown"
    
    if shazam_result and 'track' in shazam_result:
        track = shazam_result['track']
        
        # Extract album and release info
        if 'sections' in track:
            for section in track['sections']:
                if section.get('type') == 'SONG':
                    metadata = section.get('metadata', [])
                    for meta in metadata:
                        if meta.get('title') == 'Album':
                            album = meta.get('text', 'Unknown')
                        elif meta.get('title') == 'Released':
                            released = meta.get('text', 'Unknown')
        
        # Extract genre
        if 'genres' in track:
            genre = track['genres'].get('primary', 'Unknown')
        
        # Use identified song name and artist if available
        song_name = track.get('title', song_name)
        artist = track.get('subtitle', artist)
    
    song_id = hashlib.md5(f"{song_name}_{artist}_{album}".encode()).hexdigest()
    
    st.session_state.offline_database[song_id] = {
        'name': song_name,
        'artist': artist,
        'album': album,
        'released': released,
        'genre': genre,
        'fingerprint': fingerprint
    }
    
    save_offline_database()
    update_offline_db_stats()
    return True

def identify_song_offline(audio_file, threshold=0.3):
    """Identify a song from the offline database"""
    query_fingerprint = extract_audio_fingerprint(audio_file)
    if not query_fingerprint:
        return None
    
    best_match = None
    best_score = 0
    
    for song_id, song_data in st.session_state.offline_database.items():
        score = compare_fingerprints(query_fingerprint, song_data['fingerprint'])
        if score > best_score and score > threshold:
            best_score = score
            best_match = {
                'name': song_data['name'],
                'artist': song_data['artist'],
                'album': song_data.get('album', 'Unknown'),
                'released': song_data.get('released', 'Unknown'),
                'genre': song_data.get('genre', 'Unknown'),
                'confidence': score
            }
    
    return best_match

def compare_fingerprints(fp1, fp2):
    """Compare two fingerprints and return similarity score"""
    if not fp1 or not fp2:
        return 0
    
    matches = 0
    total_points = min(len(fp1), len(fp2))
    
    if total_points == 0:
        return 0
    
    # Convert to sets for faster lookup
    fp1_set = set(fp1)
    fp2_set = set(fp2)
    
    # Count exact matches
    matches = len(fp1_set.intersection(fp2_set))
    
    # Calculate similarity score
    similarity = matches / max(len(fp1_set), len(fp2_set))
    return similarity

# Page configuration
st.set_page_config(
    page_title="RadioSport Song ID",
    page_icon="🧟🎵",
    layout="wide",
    menu_items={
        'Report a Bug': "https://github.com/rkarikari/RadioSport-chat",
        'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."
    }
)

# App title and description
st.title("🎵RadioSport Song ID")
st.markdown("**Identify songs using your microphone - Now with Auto Radio Monitoring & Offline ID!**")

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'identified_songs' not in st.session_state:
        st.session_state.identified_songs = []
    
    if 'audio_devices' not in st.session_state:
        st.session_state.audio_devices = []
    
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = None
    
    if 'selected_song' not in st.session_state:
        st.session_state.selected_song = None
    
    # Add initialization for current song result
    if 'current_song_result' not in st.session_state:
        st.session_state.current_song_result = None
    
    # Auto mode state variables
    if 'auto_mode_active' not in st.session_state:
        st.session_state.auto_mode_active = False
    
    if 'auto_mode_thread' not in st.session_state:
        st.session_state.auto_mode_thread = None
    
    if 'auto_mode_stats' not in st.session_state:
        st.session_state.auto_mode_stats = {
            'start_time': None,
            'total_songs': 0,
            'unique_songs': 0,
            'genres_detected': {},
            'artists_detected': {},
            'failed_detections': 0,
            'session_log': []
        }
    
    if 'auto_mode_settings' not in st.session_state:
        st.session_state.auto_mode_settings = {
            'monitoring_duration': 60,  # minutes
            'detection_threshold': 0.3,  # volume threshold for song change detection
            'silence_duration': 3,  # seconds of silence before considering song change
            'min_song_length': 30,  # minimum seconds between identifications
            'station_name': 'Radio Station',
            'output_format': 'CSV'
        }
    
    # Offline database initialization
    if 'offline_database' not in st.session_state:
        st.session_state.offline_database = {}
    
    # Initialize offline database flags
    if 'offline_db_loaded' not in st.session_state:
        st.session_state.offline_db_loaded = False
    
    if 'offline_db_file' not in st.session_state:
        st.session_state.offline_db_file = "radiosport_offline_db.pkl"
    
    # Load offline database if not already loaded
    if not st.session_state.offline_db_loaded:
        load_offline_database()
    
    if 'offline_db_stats' not in st.session_state:
        st.session_state.offline_db_stats = {
            'songs': 0,
            'artists': 0,
            'fingerprints': 0
        }
        update_offline_db_stats()

    # Offline identification settings
    if 'use_offline_id' not in st.session_state:
        st.session_state.use_offline_id = False
    
    # Initialize add_to_offline_db
    if 'add_to_offline_db' not in st.session_state:
        st.session_state.add_to_offline_db = False

initialize_session_state()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 12  # Fixed at 12 seconds for manual mode
AUTO_RECORD_SECONDS = 6  # 6 seconds for auto mode

def get_audio_devices():
    """Get available audio input devices"""
    try:
        audio = pyaudio.PyAudio()
        devices = []
        device_count = audio.get_device_count()
        
        for i in range(device_count):
            try:
                device_info = audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    devices.append({
                        'index': i,
                        'name': device_info.get('name', f'Device {i}'),
                        'channels': device_info.get('maxInputChannels'),
                        'sample_rate': device_info.get('defaultSampleRate'),
                        'host_api': device_info.get('hostApi')
                    })
            except:
                continue
        
        audio.terminate()
        return devices
    except Exception as e:
        st.error(f"Error accessing audio devices: {str(e)}")
        return []

def analyze_audio_quality(audio_data):
    """Analyze audio quality and provide feedback"""
    try:
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = audio_data
            
        max_amplitude = np.max(np.abs(audio_array))
        volume_level = max_amplitude / 32767.0
        
        if volume_level < 0.1:
            return "🔴 VERY LOW - Audio too quiet! Increase volume significantly."
        elif volume_level < 0.3:
            return "🟡 LOW - Audio quiet. Try increasing volume."
        elif volume_level < 0.7:
            return "🟢 GOOD - Audio level looks good!"
        elif volume_level < 0.9:
            return "🔵 HIGH - Good strong audio signal!"
        else:
            return "🟠 VERY HIGH - Audio might be clipping. Try reducing volume slightly."
            
    except Exception:
        return "Could not analyze audio quality"

def detect_song_change(current_audio, previous_audio, threshold=0.3):
    """Detect if a song change has occurred based on audio analysis"""
    try:
        if previous_audio is None:
            return True
        
        # Convert to numpy arrays
        current = np.frombuffer(current_audio, dtype=np.int16)
        previous = np.frombuffer(previous_audio, dtype=np.int16)
        
        # Ensure same length for comparison
        min_len = min(len(current), len(previous))
        current = current[:min_len]
        previous = previous[:min_len]
        
        # Calculate cross-correlation to detect similarity
        correlation = np.corrcoef(current, previous)[0, 1]
        
        # If correlation is low or NaN, assume song change
        if np.isnan(correlation) or correlation < threshold:
            return True
        
        return False
    except:
        return True  # Assume song change if analysis fails

async def identify_song(audio_file_path):
    """Identify song using Shazam API"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            shazam = Shazam()
            
            if not os.path.exists(audio_file_path):
                return None
                
            file_size = os.path.getsize(audio_file_path)
            if file_size < 1000:
                return None
            
            for attempt in range(2):
                try:
                    with open(audio_file_path, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    
                    out = await shazam.recognize(audio_data)
                    
                    if out and 'track' in out:
                        return out
                    elif out:
                        st.warning("Shazam responded but no song found in audio.")
                        
                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(1)
                        continue
            
            return None
            
    except Exception:
        return None

def calculate_optimal_gain(test_audio_data):
    """Calculate optimal gain based on audio level analysis"""
    try:
        if isinstance(test_audio_data, bytes):
            audio_array = np.frombuffer(test_audio_data, dtype=np.int16)
        else:
            audio_array = test_audio_data
            
        max_amplitude = np.max(np.abs(audio_array))
        volume_level = max_amplitude / 32767.0
        
        # Target volume level around 0.6 (60%)
        target_level = 0.6
        
        if volume_level > 0.01:  # Avoid division by very small numbers
            optimal_gain = target_level / volume_level
            # Clamp gain between reasonable bounds
            optimal_gain = max(0.1, min(5.0, optimal_gain))
            return optimal_gain
        else:
            return 2.0  # Default higher gain for very quiet audio
            
    except Exception:
        return 1.0  # Default gain if calculation fails

def record_audio_with_auto_gain(device_index=None, duration=RECORD_SECONDS, show_progress=True):
    """Record audio from microphone with automatic gain control"""
    try:
        audio = pyaudio.PyAudio()
        
        if device_index is None:
            device_index = audio.get_default_input_device_info()['index']
        
        try:
            device_info = audio.get_device_info_by_index(device_index)
            device_name = device_info.get('name', f'Device {device_index}')
        except Exception as e:
            if show_progress:
                st.error(f"Cannot access selected device: {str(e)}")
            audio.terminate()
            return None
        
        try:
            stream = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              input_device_index=device_index,
                              frames_per_buffer=CHUNK)
        except Exception as e:
            if show_progress:
                st.error(f"Cannot access microphone: {str(e)}")
            audio.terminate()
            return None
        
        # Phase 1: Record a short sample for gain calibration (2 seconds)
        if show_progress:
            st.info("🔧 Calibrating audio levels...")
        calibration_frames = []
        calibration_chunks = int(RATE / CHUNK * 2)  # 2 seconds for calibration
        
        for i in range(calibration_chunks):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                calibration_frames.append(data)
            except Exception:
                break
        
        # Calculate optimal gain from calibration
        if calibration_frames:
            calibration_audio = b''.join(calibration_frames)
            optimal_gain = calculate_optimal_gain(calibration_audio)
            if show_progress:
                st.info(f"🎚️ Auto-adjusted gain: {optimal_gain:.1f}x")
        else:
            optimal_gain = 1.0
            if show_progress:
                st.warning("⚠️ Could not calibrate gain, using default")
        
        # Phase 2: Record the actual audio with optimal gain
        frames = []
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            volume_indicator = st.empty()
        
        max_volume = 0
        for i in range(0, int(RATE / CHUNK * duration)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Apply automatic gain adjustment
                if optimal_gain != 1.0:
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    audio_chunk = (audio_chunk * optimal_gain).astype(np.int16)
                    data = audio_chunk.tobytes()
                
                frames.append(data)
                
                # Volume monitoring
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                volume = np.max(np.abs(audio_chunk)) / 32767.0
                max_volume = max(max_volume, volume)
                
                if show_progress:
                    progress = (i + 1) / (RATE / CHUNK * duration)
                    progress_bar.progress(progress)
                    remaining_time = duration - (i * CHUNK / RATE)
                    status_text.text(f"🎤 Recording... {remaining_time:.1f}s remaining")
                    
                    volume_bars = int(volume * 20)
                    volume_display = "🔊" + "█" * volume_bars + "░" * (20 - volume_bars)
                    
                    if volume < 0.1:
                        volume_color = "🔴"
                    elif volume < 0.3:
                        volume_color = "🟡"
                    elif volume < 0.7:
                        volume_color = "🟢"
                    elif volume < 0.9:
                        volume_color = "🔵"
                    else:
                        volume_color = "🟠"
                    
                    volume_indicator.text(f"{volume_color} Volume: {volume_display} ({volume:.1%})")
                
            except Exception:
                break
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save the recording
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        try:
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                if show_progress:
                    st.success(f"✅ Recording completed! (Gain: {optimal_gain:.1f}x, Max Volume: {max_volume:.1%})")
                return temp_file.name
            else:
                if show_progress:
                    st.error("Failed to save recording. Please try again.")
                return None
                
        except Exception as e:
            if show_progress:
                st.error(f"Error saving recording: {str(e)}")
            return None
        
    except Exception as e:
        if show_progress:
            st.error(f"Unexpected error: {str(e)}")
        return None

def save_auto_mode_log(session_data, format_type='CSV'):
    """Save auto mode session data to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        station_name = st.session_state.auto_mode_settings['station_name'].replace(' ', '_')
        
        if format_type == 'CSV':
            filename = f"RadioSport_AutoMode_{station_name}_{timestamp}.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'title', 'artist', 'album', 'genre', 'released', 'detection_confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write session metadata
                writer.writerow({
                    'timestamp': 'SESSION_START',
                    'title': session_data['start_time'],
                    'artist': f"Station: {station_name}",
                    'album': f"Total Songs: {session_data['total_songs']}",
                    'genre': f"Unique Songs: {session_data['unique_songs']}",
                    'released': f"Failed: {session_data['failed_detections']}",
                    'detection_confidence': ''
                })
                
                # Write songs
                for song in session_data['session_log']:
                    if song['status'] == 'identified':
                        track = song['data'].get('track', {})
                        
                        # Extract genre
                        genre = 'Unknown'
                        if 'genres' in track:
                            genre = track['genres'].get('primary', 'Unknown')
                        
                        # Extract album and release info
                        album = 'Unknown'
                        released = 'Unknown'
                        
                        if 'sections' in track:
                            for section in track['sections']:
                                if section.get('type') == 'SONG':
                                    metadata = section.get('metadata', [])
                                    for meta in metadata:
                                        if meta.get('title') == 'Album':
                                            album = meta.get('text', 'Unknown')
                                        elif meta.get('title') == 'Released':
                                            released = meta.get('text', 'Unknown')
                        
                        writer.writerow({
                            'timestamp': song['timestamp'],
                            'title': track.get('title', 'Unknown'),
                            'artist': track.get('subtitle', 'Unknown'),
                            'album': album,
                            'genre': genre,
                            'released': released,
                            'detection_confidence': 'High'
                        })
                    else:
                        writer.writerow({
                            'timestamp': song['timestamp'],
                            'title': 'DETECTION_FAILED',
                            'artist': song.get('error', 'Unknown error'),
                            'album': '',
                            'genre': '',
                            'released': '',
                            'detection_confidence': 'Failed'
                        })
        
        else:  # JSON format
            filename = f"RadioSport_AutoMode_{station_name}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(session_data, jsonfile, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    except Exception as e:
        st.error(f"Error saving log file: {str(e)}")
        return None

def display_song_info(song_data, compact=False):
    """Display identified song information in 2-column layout"""
    if not song_data or 'track' not in song_data:
        st.warning("🚫 No song identified.")
        return
    
    track = song_data['track']
    
    # Create 2-column layout
    if compact:
        # For compact display (history list items)
        col1, col2 = st.columns([1, 2])
    else:
        # For full display
        col1, col2 = st.columns([1, 2])
    
    with col1:
        # Album art column
        if 'images' in track and 'coverart' in track['images']:
            st.image(track['images']['coverart'], width=150 if compact else 200)
        else:
            # Placeholder if no album art
            st.markdown(f"""
            <div style="width: {'150px' if compact else '200px'}; height: {'150px' if compact else '200px'}; 
                        background-color: #f0f2f6; border-radius: 8px; 
                        display: flex; align-items: center; justify-content: center;">
                <span style="font-size: {'24px' if compact else '32px'};">🎵</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Song details column
        if not compact:
            st.subheader("🎵 Song Details")
        
        # Main song info
        st.markdown(f"**🎵 Title:** {track.get('title', 'Unknown')}")
        st.markdown(f"**🎤 Artist:** {track.get('subtitle', 'Unknown')}")
        
        # Additional metadata
        album = "Unknown"
        released = "Unknown"
        genre = "Unknown"
        
        # Extract from track metadata
        if 'album' in track:
            album = track['album']
        if 'released' in track:
            released = track['released']
        if 'genre' in track:
            genre = track['genre']
        
        # Fallback to sections if not directly available
        if album == "Unknown" or released == "Unknown":
            if 'sections' in track:
                for section in track['sections']:
                    if section.get('type') == 'SONG':
                        metadata = section.get('metadata', [])
                        for meta in metadata:
                            if meta.get('title') == 'Album':
                                album = meta.get('text', 'Unknown')
                            elif meta.get('title') == 'Released':
                                released = meta.get('text', 'Unknown')
        
        st.markdown(f"**💿 Album:** {album}")
        st.markdown(f"**📅 Released:** {released}")
        
        if 'genres' in track:
            genres = track['genres'].get('primary', 'Unknown')
            st.markdown(f"**🎭 Genre:** {genres}")
        elif genre != "Unknown":
            st.markdown(f"**🎭 Genre:** {genre}")

def display_offline_song_info(song_data):
    """Display offline identified song information"""
    if not song_data:
        st.warning("🚫 No song identified.")
        return
    
    # Create 2-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Placeholder for album art
        st.markdown(f"""
        <div style="width: 200px; height: 200px; 
                    background-color: #f0f2f6; border-radius: 8px; 
                    display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 32px;">🎵</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎵 Song Details (Offline ID)")
        st.markdown(f"**🎵 Title:** {song_data.get('name', 'Unknown')}")
        st.markdown(f"**🎤 Artist:** {song_data.get('artist', 'Unknown')}")
        st.markdown(f"**💿 Album:** {song_data.get('album', 'Unknown')}")
        st.markdown(f"**📅 Released:** {song_data.get('released', 'Unknown')}")
        st.markdown(f"**🎭 Genre:** {song_data.get('genre', 'Unknown')}")
        st.markdown(f"**🔐 Confidence:** {song_data.get('confidence', 0):.1%}")

def process_audio_file(audio_file):
    """Process audio file for song identification with enhanced cleanup"""
    with st.spinner("🔍 Identifying song..."):
        try:
            # Verify file exists and has content
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                st.error("Invalid audio file. Please try recording again.")
                return
            
            # Try offline identification first if enabled
            offline_result = None
            if st.session_state.use_offline_id:
                offline_result = identify_song_offline(audio_file)
            
            # If offline identification found a match
            if offline_result and offline_result['confidence'] > 0.4:
                st.session_state.current_song_result = {
                    'offline': True,
                    'track': {
                        'title': offline_result['name'],
                        'subtitle': offline_result['artist'],
                        'album': offline_result.get('album', 'Unknown'),
                        'released': offline_result.get('released', 'Unknown'),
                        'genre': offline_result.get('genre', 'Unknown'),
                        'images': {'coverart': ''},
                        'genres': {'primary': offline_result.get('genre', 'Offline ID')}
                    }
                }
                st.session_state.history_needs_save = True
                song_info = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'title': offline_result['name'],
                    'artist': offline_result['artist'],
                    'album': offline_result.get('album', 'Unknown'),
                    'released': offline_result.get('released', 'Unknown'),
                    'genre': offline_result.get('genre', 'Unknown'),
                    'data': st.session_state.current_song_result
                }
                st.session_state.identified_songs.append(song_info)
                st.rerun()
                return
            
            # Otherwise use Shazam
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(identify_song(audio_file))
            finally:
                loop.close()
            
            if result:
                # Store the result in session state for display in the right column
                st.session_state.current_song_result = result
                st.session_state.history_needs_save = True  # Trigger auto-save
                
                # Extract additional metadata
                album = "Unknown"
                released = "Unknown"
                if 'sections' in result['track']:
                    for section in result['track']['sections']:
                        if section.get('type') == 'SONG':
                            metadata = section.get('metadata', [])
                            for meta in metadata:
                                if meta.get('title') == 'Album':
                                    album = meta.get('text', 'Unknown')
                                elif meta.get('title') == 'Released':
                                    released = meta.get('text', 'Unknown')
                
                song_info = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'title': result.get('track', {}).get('title', 'Unknown'),
                    'artist': result.get('track', {}).get('subtitle', 'Unknown'),
                    'album': album,
                    'released': released,
                    'genre': result.get('track', {}).get('genres', {}).get('primary', 'Unknown'),
                    'data': result
                }
                st.session_state.identified_songs.append(song_info)
                
                # Add to offline DB if enabled
                if st.session_state.add_to_offline_db and 'track' in result:
                    title = result['track'].get('title', 'Unknown')
                    artist = result['track'].get('subtitle', 'Unknown')
                    if add_song_to_offline_db(audio_file, title, artist):
                        st.toast(f"✅ Added '{title}' to offline database with full details!")
                
                # Force a rerun to update the display
                st.rerun()
            else:
                # Clear any previous result and show error
                st.session_state.current_song_result = None
                st.error("No song identified. Try recording again with music playing clearly.")
                
        except Exception as e:
            # Clear any previous result and show error
            st.session_state.current_song_result = None
            st.error(f"Song identification failed: {str(e)}")
        
        finally:
            # Always attempt to clean up the temporary file
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            except Exception:
                pass  # Silently ignore cleanup errors

def history_tab():
    """Enhanced song history interface with auto-save/load functionality"""
    # Auto-save file name
    HISTORY_FILE = "radiosport_song_history.json"
    
    # Load history from file on initial run
    if 'history_loaded' not in st.session_state:
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    st.session_state.identified_songs = json.load(f)
                st.success(f"✅ Loaded {len(st.session_state.identified_songs)} songs from history")
            st.session_state.history_loaded = True
        except Exception as e:
            st.error(f"Error loading history: {str(e)}")
    
    # Function to save history to file
    def save_history():
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(st.session_state.identified_songs, f, default=str)
            return True
        except Exception as e:
            st.error(f"Error saving history: {str(e)}")
            return False
    
    # Auto-save history after any modification
    if 'history_needs_save' in st.session_state and st.session_state.history_needs_save:
        if save_history():
            st.session_state.history_needs_save = False
            # Show a temporary success message
            success_msg = st.empty()
            success_msg.success("💾 History auto-saved successfully!")
            time.sleep(2)
            success_msg.empty()
    
    # Refresh control for history tab
    if 'history_refresh' not in st.session_state:
        st.session_state.history_refresh = time.time()
    
    st.markdown("### 📚 Song History")
    
    if st.session_state.identified_songs:
        # Header with total count and action buttons
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.write(f"**Total songs identified:** {len(st.session_state.identified_songs)}")
        
        with col2:
            # Create three columns for the buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                # Export History button
                csv_data = "timestamp,title,artist,album,genre,released\n"
                for song in st.session_state.identified_songs:
                    # Use stored metadata if available
                    title = song.get('title', 'Unknown').replace('"', '""')
                    artist = song.get('artist', 'Unknown').replace('"', '""')
                    album = song.get('album', 'Unknown').replace('"', '""')
                    genre = song.get('genre', 'Unknown').replace('"', '""')
                    released = song.get('released', 'Unknown').replace('"', '""')
                    
                    csv_data += (f'"{song["timestamp"]}","{title}","{artist}",'
                                f'"{album}","{genre}","{released}"\n')
                
                st.download_button(
                    "📤 Export",
                    data=csv_data,
                    file_name=f"RadioSport_History_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download history as CSV file"
                )
            
            with btn_col2:
                # Clear History button
                if st.button("🗑️ Clear", help="Clear all history", key="history_clear_btn"):
                    st.session_state.identified_songs = []
                    st.session_state.selected_song = None
                    if 'selected_history_index' in st.session_state:
                        del st.session_state.selected_history_index
                    st.session_state.history_needs_save = True
                    st.rerun()
            
            with btn_col3:
                # Manual Save button (optional, can be removed)
                if st.button("💾 Save", help="Manually save history", key="history_save_btn"):
                    if save_history():
                        st.success("History saved successfully!")
        
        st.markdown("---")
        
        # Get songs in reverse chronological order (newest first)
        songs = list(reversed(st.session_state.identified_songs))
        
        if len(songs) > 0:
            # Display the latest song in full detail
            st.markdown("### 🆕 Latest Song")
            with st.container():
                st.markdown("*Most recently identified*")
                display_song_info(songs[0]['data'])
                st.markdown(f"🕒 **Identified:** {songs[0]['timestamp']}")
            
            st.markdown("---")
            
            # Display older songs if there are any
            if len(songs) > 1:
                st.markdown("### 📋 Previous Songs")
                
                # Initialize selected song index in session state if not exists
                if 'selected_history_index' not in st.session_state:
                    st.session_state.selected_history_index = None
                
                # Create main layout for history browsing
                main_col1, main_col2 = st.columns([1, 1])
                
                with main_col1:
                    st.markdown("**📜 Song List** *(Click to view details)*")
                    
                    # Display older songs in compact format (skip the first one as it's shown above)
                    for i, song in enumerate(songs[1:], 1):
                        is_selected = st.session_state.selected_history_index == i
                        
                        with st.container():
                            # Highlight selected song
                            if is_selected:
                                st.markdown("**🎵 SELECTED:**")
                                st.markdown(f"""
                                <div style="background-color: #e8f4f8; padding: 10px; border-radius: 8px; border-left: 4px solid #1f77b4;">
                                """, unsafe_allow_html=True)
                            
                            # Create clickable song item
                            song_col1, song_col2 = st.columns([3, 1])
                            
                            with song_col1:
                                # Truncate long titles for compact display
                                title = song['title']
                                artist = song['artist']
                                if len(title) > 30:
                                    title = title[:27] + "..."
                                if len(artist) > 25:
                                    artist = artist[:22] + "..."
                                
                                st.markdown(f"**{title}**")
                                st.markdown(f"*by {artist}*")
                                
                                # Show timestamp
                                time_parts = song['timestamp'].split()
                                date_str = time_parts[0] if len(time_parts) > 0 else ""
                                time_str = time_parts[1] if len(time_parts) > 1 else ""
                                st.markdown(f"🕒 {date_str} {time_str}")
                            
                            with song_col2:
                                button_label = "🎵" if is_selected else "👁️"
                                button_help = "Click to close" if is_selected else "Click to view"
                                
                                if st.button(button_label, key=f"select_song_{i}", help=button_help):
                                    if st.session_state.selected_history_index == i:
                                        # Deselect if already selected
                                        st.session_state.selected_history_index = None
                                        st.session_state.selected_song = None
                                    else:
                                        # Select this song
                                        st.session_state.selected_history_index = i
                                        st.session_state.selected_song = song
                                    st.rerun()
                            
                            if is_selected:
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Add spacing between items
                            st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
                
                with main_col2:
                    st.markdown("**🎵 Song Details**")
                    
                    # Display detailed info for selected song
                    if (st.session_state.selected_history_index is not None and 
                        st.session_state.selected_song):
                        
                        # Use compact=False for full details
                        display_song_info(st.session_state.selected_song['data'], compact=False)
                        
                        st.markdown("---")
                        st.markdown(f"🕒 **Identified:** {st.session_state.selected_song['timestamp']}")
                        
                        # Button to clear selection
                        if st.button("✖️ Close Details", key="close_details"):
                            st.session_state.selected_history_index = None
                            st.session_state.selected_song = None
                            st.rerun()
                    else:
                        # Help text when no song is selected
                        st.info("👈 Select a song from the list to view full details")
                        
                        st.markdown("**📋 History Features:**")
                        st.markdown("• 🆕 Latest song shown above in full detail")
                        st.markdown("• 👁️ Click any song to view complete information")
                        st.markdown("• 🎵 Selected songs are highlighted")
                        st.markdown("• ✖️ Close details to return to list view")
                        st.markdown("• 🗑️ Clear all history when needed")
    else:
        # Empty state
        st.info("📭 No songs identified yet. Use the Song ID tab to identify your first song!")
        
        st.markdown("**🚀 Get Started:**")
        st.markdown("1. 🎤 Go to the 'Song ID' tab")
        st.markdown("2. 🎵 Play music near your microphone") 
        st.markdown("3. 📱 Click 'Start Recording & Identify Song'")
        st.markdown("4. 📚 Return here to view your song history")
        
        st.markdown("**📋 History Features:**")
        st.markdown("• 🖼️ Album artwork and song details in organized layout")
        st.markdown("• 🆕 Latest songs displayed prominently")
        st.markdown("• 📜 Browsable list of all identified songs")
        st.markdown("• 🔗 Direct links to streaming platforms")
        st.markdown("• 🕒 Timestamp tracking for each identification")
        st.markdown("• 💾 History automatically saved between sessions")
    
    # AUTO-REFRESH FOR HISTORY TAB WHEN AUTO MODE IS ACTIVE
    if st.session_state.auto_mode_active:
        current_time = time.time()
        elapsed = current_time - st.session_state.history_refresh
        
        # Adaptive refresh rates:
        if elapsed < 5:  # Ultra-frequent when recent activity
            sleep_time = 1
        elif elapsed < 30:  # Frequent during countdown
            sleep_time = 2
        else:  # Standard refresh
            sleep_time = 5
        
        # Force refresh after sleep
        time.sleep(sleep_time)
        st.session_state.history_refresh = time.time()
        st.rerun()

def auto_mode_tab():
    """Auto mode interface - 3 column layout with Recent Songs List in right column"""
    # TIMESTAMP TRACKER FOR AUTO-REFRESH
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()

    st.markdown("### 📻 Auto Radio Station Monitoring")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### 🛠️ Auto Mode Settings")
        
        # Audio device selection
        if not st.session_state.audio_devices:
            st.session_state.audio_devices = get_audio_devices()
        
        if st.session_state.audio_devices:
            device_options = [f"🎤 {device['name']}" for device in st.session_state.audio_devices]
            selected_device_idx = st.selectbox(
                "Audio Source:", range(len(device_options)),
                format_func=lambda x: device_options[x], key="auto_device_selector"
            )
            st.session_state.selected_device = st.session_state.audio_devices[selected_device_idx]['index']
        else:
            st.error("No audio input devices found!")
            if st.button("🔄 Refresh Devices", key="auto_refresh"):
                st.session_state.audio_devices = get_audio_devices()
                st.rerun()
            return
        
        # Station name
        station_name = st.text_input(
            "📻 Station Name:", 
            value="Radio Station",
            help="Name for the radio station being monitored",
            key="station_name_input"
        )
        
        # Auto-identify interval
        auto_interval = st.slider(
            "🔄 Auto-Identify Every (seconds):",
            min_value=150, max_value=600,
            value=240,  # Default 4.0 minutes
            help="How often to automatically identify songs",
            key="auto_interval_slider"
        )
        
        # Show interval in minutes for clarity
        interval_minutes = auto_interval / 60
        if interval_minutes >= 1:
            st.caption(f"⏱️ Interval: {interval_minutes:.1f} minutes")
        
        # Offline identification setting
        st.session_state.use_offline_id = st.checkbox(
            "🔌 Use Offline Identification",
            value=st.session_state.use_offline_id,
            help="Use offline fingerprinting instead of Shazam (requires pre-trained database)",
            key="use_offline_id_auto"  # Unique key for auto mode
        )
        
        # Add to offline DB setting
        st.session_state.add_to_offline_db = st.checkbox(
            "📥 Add to Offline Database",
            value=st.session_state.add_to_offline_db,
            help="Add successfully identified songs to offline database with full details",
            key="add_to_offline_db_auto"  # Unique key for auto mode
        )
        
        # Initialize session state for auto mode
        if 'auto_mode_active' not in st.session_state:
            st.session_state.auto_mode_active = False
        if 'auto_mode_stats' not in st.session_state:
            st.session_state.auto_mode_stats = {
                'start_time': None,
                'songs_identified': 0,
                'last_song_key': None,
                'errors': 0,
                'last_check': None
            }
        if 'auto_mode_message' not in st.session_state:
            st.session_state.auto_mode_message = ""
        
        st.markdown("---")
        
        # Control buttons
        if not st.session_state.auto_mode_active:
            if st.button("🚀 Start Auto Monitoring", type="primary", key="start_auto"):
                st.session_state.auto_mode_active = True
                st.session_state.auto_mode_stats = {
                    'start_time': datetime.now(),
                    'songs_identified': 0,
                    'last_song_key': None,
                    'errors': 0,
                    'last_check': datetime.now(),
                    'interval': auto_interval,
                    'station_name': station_name
                }
                st.session_state.auto_mode_message = "🚀 Auto monitoring started!"
                st.success("Auto monitoring started!")
                st.rerun()
        else:
            if st.button("⏹️ Stop Auto Monitoring", type="secondary", key="stop_auto"):
                st.session_state.auto_mode_active = False
                st.session_state.auto_mode_message = "⏹️ Auto monitoring stopped"
                st.info("Auto monitoring stopped!")
                st.rerun()
        
        # Reset button
        if st.button("🔄 Reset Stats", key="reset_auto_stats"):
            st.session_state.auto_mode_stats = {
                'start_time': None,
                'songs_identified': 0,
                'last_song_key': None,
                'errors': 0,
                'last_check': None
            }
            st.success("Statistics reset!")
            st.rerun()
    
    with col2:
        st.markdown("#### 📊 Live Monitoring Status")
        
        if st.session_state.auto_mode_active:
            # Show current status message
            if st.session_state.auto_mode_message:
                st.info(st.session_state.auto_mode_message)
            
            # Calculate timing
            current_time = datetime.now()
            start_time = st.session_state.auto_mode_stats.get('start_time', current_time)
            last_check = st.session_state.auto_mode_stats.get('last_check', start_time)
            
            # Calculate elapsed time
            elapsed = current_time - start_time
            elapsed_str = f"{int(elapsed.total_seconds() // 60):02d}:{int(elapsed.total_seconds() % 60):02d}"
            
            # Calculate time until next check
            interval = st.session_state.auto_mode_stats.get('interval', 150)
            time_since_check = (current_time - last_check).total_seconds()
            time_to_next = max(0, interval - time_since_check)
            
            # Display timing info
            st.metric("⏰ Running Time", elapsed_str)
            st.metric("🎵 Songs Identified", st.session_state.auto_mode_stats.get('songs_identified', 0))
            
            if time_to_next > 0:
                next_minutes = int(time_to_next // 60)
                next_seconds = int(time_to_next % 60)
                if next_minutes > 0:
                    st.metric("⏳ Next Check", f"{next_minutes}m {next_seconds:02d}s")
                else:
                    st.metric("⏳ Next Check", f"{next_seconds}s")
            else:
                st.metric("⏳ Next Check", "NOW!")
            
            # Auto identification logic
            if time_to_next <= 0:
                st.session_state.auto_mode_message = "🎤 Recording and identifying..."
                
                # Create a placeholder for immediate feedback
                status_placeholder = st.empty()
                status_placeholder.info("🎤 Recording audio...")
                
                # Perform identification
                try:
                    # Record audio
                    audio_file = record_audio_with_auto_gain(
                        st.session_state.selected_device, 
                        duration=8,  # Shorter duration for auto mode
                        show_progress=False
                    )
                    
                    if audio_file and os.path.exists(audio_file):
                        status_placeholder.info("🔍 Identifying song...")
                        
                        # Try offline identification first if enabled
                        offline_result = None
                        if st.session_state.use_offline_id:
                            offline_result = identify_song_offline(audio_file)
                        
                        # If offline identification found a match
                        if offline_result and offline_result['confidence'] > 0.4:
                            # Create unique song key
                            song_key = f"{offline_result['name']}|{offline_result['artist']}"
                            
                            # Update current song result IMMEDIATELY for display in column 3
                            st.session_state.current_song_result = {
                                'offline': True,
                                'track': {
                                    'title': offline_result['name'],
                                    'subtitle': offline_result['artist'],
                                    'album': offline_result.get('album', 'Unknown'),
                                    'released': offline_result.get('released', 'Unknown'),
                                    'genre': offline_result.get('genre', 'Unknown'),
                                    'images': {'coverart': ''},
                                    'genres': {'primary': offline_result.get('genre', 'Offline ID')}
                                }
                            }
                            
                            # Check if this is a new song
                            if song_key != st.session_state.auto_mode_stats.get('last_song_key'):
                                # New song - add to history
                                song_info = {
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'title': offline_result['name'],
                                    'artist': offline_result['artist'],
                                    'album': offline_result.get('album', 'Unknown'),
                                    'released': offline_result.get('released', 'Unknown'),
                                    'genre': offline_result.get('genre', 'Unknown'),
                                    'data': st.session_state.current_song_result
                                }
                                st.session_state.identified_songs.append(song_info)
                                st.session_state.history_needs_save = True  # Trigger auto-save
                                
                                # Update stats
                                st.session_state.auto_mode_stats['songs_identified'] += 1
                                st.session_state.auto_mode_stats['last_song_key'] = song_key
                                
                                st.session_state.auto_mode_message = f"🆕 NEW: {offline_result['name']}"
                                status_placeholder.success(f"🆕 New song identified: {offline_result['name']}")
                            else:
                                st.session_state.auto_mode_message = f"🔄 SAME: {offline_result['name']}"
                                status_placeholder.info("🔄 Same song still playing")
                        else:
                            # Identify song using Shazam
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            try:
                                result = loop.run_until_complete(identify_song(audio_file))
                            finally:
                                loop.close()
                            
                            if result and 'track' in result:
                                track = result['track']
                                title = track.get('title', 'Unknown')
                                artist = track.get('subtitle', 'Unknown')
                                
                                # Create unique song key
                                song_key = f"{title.lower().strip()}|{artist.lower().strip()}"
                                
                                # Update current song result IMMEDIATELY for display in column 3
                                st.session_state.current_song_result = result
                                
                                # Check if this is a new song
                                if song_key != st.session_state.auto_mode_stats.get('last_song_key'):
                                    # New song - add to history
                                    # Extract additional metadata
                                    album = "Unknown"
                                    released = "Unknown"
                                    if 'sections' in track:
                                        for section in track['sections']:
                                            if section.get('type') == 'SONG':
                                                metadata = section.get('metadata', [])
                                                for meta in metadata:
                                                    if meta.get('title') == 'Album':
                                                        album = meta.get('text', 'Unknown')
                                                    elif meta.get('title') == 'Released':
                                                        released = meta.get('text', 'Unknown')
                                    
                                    song_info = {
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'title': title,
                                        'artist': artist,
                                        'album': album,
                                        'released': released,
                                        'genre': track.get('genres', {}).get('primary', 'Unknown'),
                                        'data': result
                                    }
                                    st.session_state.identified_songs.append(song_info)
                                    st.session_state.history_needs_save = True  # Trigger auto-save
                                    
                                    # Add to offline DB if enabled
                                    if st.session_state.add_to_offline_db:
                                        if add_song_to_offline_db(audio_file, title, artist):
                                            st.toast(f"✅ Added '{title}' to offline database with full details!")
                                    
                                    # Update stats
                                    st.session_state.auto_mode_stats['songs_identified'] += 1
                                    st.session_state.auto_mode_stats['last_song_key'] = song_key
                                    
                                    st.session_state.auto_mode_message = f"🆕 NEW: {title} by {artist}"
                                    status_placeholder.success(f"🆕 New song identified: {title}")
                                else:
                                    st.session_state.auto_mode_message = f"🔄 SAME: {title} by {artist}"
                                    status_placeholder.info("🔄 Same song still playing")
                            else:
                                # Clear current result if no song identified
                                st.session_state.current_song_result = None
                                st.session_state.auto_mode_message = "🚫 No song identified"
                                st.session_state.auto_mode_stats['errors'] = st.session_state.auto_mode_stats.get('errors', 0) + 1
                                status_placeholder.warning("🚫 Could not identify song")
                        
                        # Clean up audio file
                        try:
                            os.unlink(audio_file)
                        except:
                            pass
                    
                    else:
                        st.session_state.current_song_result = None
                        st.session_state.auto_mode_message = "❌ Recording failed"
                        st.session_state.auto_mode_stats['errors'] = st.session_state.auto_mode_stats.get('errors', 0) + 1
                        status_placeholder.error("❌ Recording failed")
                
                except Exception as e:
                    st.session_state.current_song_result = None
                    st.session_state.auto_mode_message = f"❌ Error: {str(e)}"
                    st.session_state.auto_mode_stats['errors'] = st.session_state.auto_mode_stats.get('errors', 0) + 1
                    status_placeholder.error(f"❌ Identification failed: {str(e)}")
                
                # Update last check time and force immediate rerun
                st.session_state.auto_mode_stats['last_check'] = datetime.now()
                
                # Clear the status placeholder after a moment
                time.sleep(2)
                status_placeholder.empty()
                
                # Force immediate page refresh to update UI
                st.rerun()
            
        else:
            st.info("⏸️ Auto monitoring is inactive")
            st.markdown("**🚀 How Auto Mode Works:**")
            st.markdown("• ⏰ Waits for specified interval")
            st.markdown("• 🎤 Records 8 seconds of audio")
            st.markdown("• 🔍 Identifies the song using Shazam")
            st.markdown("• 🆕 Adds NEW songs to history")
            st.markdown("• 🔄 Shows if same song is still playing")
        
        # Show error count if any
        if st.session_state.auto_mode_stats.get('errors', 0) > 0:
            st.warning(f"⚠️ Errors encountered: {st.session_state.auto_mode_stats['errors']}")
    
    with col3:
        st.markdown("### 🎵 Recent Songs")
        
        if st.session_state.identified_songs:
            # Get last 10 songs in reverse order (newest first)
            recent_songs = list(reversed(st.session_state.identified_songs[-5:]))
            
            st.markdown(f"**📋 Last {len(recent_songs)} Songs Identified**")
            
            for i, song in enumerate(recent_songs):
                with st.container():
                    # Extract time from timestamp (just show time, not date)
                    timestamp_parts = song['timestamp'].split(' ')
                    time_str = timestamp_parts[1] if len(timestamp_parts) > 1 else song['timestamp']
                    
                    # Truncate long titles/artists for compact display
                    title = song['title']
                    artist = song['artist']
                    if len(title) > 25:
                        title = title[:22] + "..."
                    if len(artist) > 20:
                        artist = artist[:17] + "..."
                    
                    # Different styling for the newest song
                    if i == 0:
                        st.markdown(f"""
                        <div style="background-color: #e8f4f8; padding: 8px; border-radius: 6px; border-left: 3px solid #1f77b4; margin-bottom: 8px;">
                            <div style="font-weight: bold; color: #1f77b4;">🆕 LATEST</div>
                            <div style="font-weight: bold; font-size: 14px;">{title}</div>
                            <div style="color: #666; font-size: 12px;">{artist}</div>
                            <div style="color: #888; font-size: 11px;">🕒 {time_str}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="padding: 6px 8px; border-bottom: 1px solid #eee; margin-bottom: 4px;">
                            <div style="font-weight: 500; font-size: 13px;">{title}</div>
                            <div style="color: #666; font-size: 11px;">{artist}</div>
                            <div style="color: #888; font-size: 10px;">🕒 {time_str}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("📭 No songs identified yet")
            
            if not st.session_state.auto_mode_active:
                st.markdown("**🚀 Start Auto Mode to begin identifying songs automatically!**")
                st.markdown("---")
                st.markdown("**📋 This list will show:**")
                st.markdown("• 🆕 Most recent song highlighted")
                st.markdown("• 🎵 Song title and artist")
                st.markdown("• 🕒 Time of identification")
                st.markdown("• 📊 Last 10 songs identified")
            else:
                st.markdown("**🎤 Auto mode is running...**")
                st.markdown("Songs will appear here as they're identified!")
    
    # AUTO-REFRESH LOGIC FOR ENTIRE TAB
    if st.session_state.auto_mode_active:
        # Calculate next refresh interval
        current_time = time.time()
        elapsed = current_time - st.session_state.last_refresh
        
        # Determine optimal sleep time based on activity level
        if elapsed < 5:  # Ultra-frequent when recent activity
            sleep_time = 1
        elif elapsed < 30:  # Frequent during countdown
            sleep_time = 2
        else:  # Standard refresh
            sleep_time = 5
        
        # Force refresh after sleep
        time.sleep(sleep_time)
        st.session_state.last_refresh = time.time()
        st.rerun()

#------- AI functions Begin
#------- AI functions Begin

def perform_chat_query(songs, question, ai_mode):
    """Perform AI chat query about the playlist"""
    try:
        # Prepare playlist data
        playlist_data = prepare_playlist_data(songs)
        
        # Create chat prompt
        prompt = create_chat_prompt(playlist_data, question)
        
        # Call appropriate AI service
        if ai_mode == "Local":
            result = call_local_ai_chat(prompt)
        else:
            result = call_cloud_ai_chat(prompt)
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

def create_chat_prompt(playlist_data, question):
    """Create prompt for chat query about playlist"""
    
    # Summarize playlist for context
    song_list = []
    for song in playlist_data['songs'][:15]:  # Limit to prevent token overflow
        song_str = f"'{song['title']}' by {song['artist']}"
        if song.get('genre'):
            song_str += f" ({song['genre']})"
        song_list.append(song_str)
    
    artist_counts = Counter(playlist_data['artists']).most_common(8)
    top_artists = [f"{artist} ({count}x)" for artist, count in artist_counts]
    
    prompt = f"""You are a music analyst AI. Answer the user's question about their playlist based on the data below.

PLAYLIST CONTEXT:
- Total Songs: {playlist_data['total_songs']}
- Top Artists: {', '.join(top_artists)}
- Recent Songs: {', '.join(song_list)}

USER QUESTION: {question}

Provide a helpful, specific answer based on their actual playlist data. Use emojis and be conversational but informative. If recommending music, explain why it fits their taste."""

    return prompt

def call_local_ai_chat(prompt):
    """Call local Ollama AI for chat query"""
    try:
        server = st.session_state.ai_settings['ollama_server']
        selected_model = st.session_state.ai_settings.get('selected_model')
        
        if not selected_model:
            return "Error: No model selected"
        
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "num_predict": 1000,
                "stop": ["Human:", "Assistant:", "User:"]
            }
        }
        
        response = requests.post(f"{server}/api/generate", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response generated')
        else:
            return f"Error: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Local AI Error: {str(e)}"

def call_cloud_ai_chat(prompt):
    """Call OpenRouter cloud AI for chat query"""
    try:
        keys = st.session_state.ai_settings['openrouter_keys']
        selected_model = st.session_state.ai_settings.get('selected_cloud_model')
        
        if not keys or not selected_model:
            return "Error: No API keys or model selected"
        
        api_key = random.choice(keys)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://radiosport-songid.streamlit.app",
            "X-Title": "RadioSport Song Analyzer"
        }
        
        payload = {
            "model": selected_model['id'],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.8
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Cloud AI Error: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Cloud AI Error: {str(e)}"

#------- AI functions Begin
#------- AI functions Begin
def ai_analysis_tab():
    """AI Analysis tab for song history with Local/Cloud AI support - FIXED VERSION"""
    
    # Initialize session state for AI settings
    if 'ai_settings' not in st.session_state:
        st.session_state.ai_settings = {
            'ai_mode': 'Cloud',
            'ollama_server': 'http://localhost:11434',
            'openrouter_keys': [],
            'analysis_cache': {},
            'available_models': [],
            'cloud_models': [],
            'selected_cloud_model': None
        }
    
    # Load OpenRouter API keys from secrets
    try:
        keys = []
        # Check if openrouter section exists in secrets
        if 'openrouter' in st.secrets:
            for i in range(1, 4):
                key = st.secrets.openrouter.get(f'api_key{i}', '')
                if key:
                    keys.append(key)
        st.session_state.ai_settings['openrouter_keys'] = keys
    except Exception as e:
        st.session_state.ai_settings['openrouter_keys'] = []
    
    st.markdown("### 🤖 AI Playlist Analysis")
    # All AI controls in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ AI Engine Settings")
        
        # AI Mode Selection with auto-refresh
        previous_mode = st.session_state.ai_settings.get('ai_mode', 'Cloud')
        ai_mode = st.radio(
            "AI Engine:",
            ["Local", "Cloud"],
            key="ai_mode_selector",
            help="Choose between local Ollama or cloud OpenRouter"
        )
        
        # Auto-refresh when AI mode changes
        mode_changed = ai_mode != previous_mode
        if mode_changed:
            st.session_state.ai_settings['ai_mode'] = ai_mode
            
            # Auto-load models when switching modes
            if ai_mode == "Local":
                ollama_server = st.session_state.ai_settings['ollama_server']
                with st.spinner("Auto-loading Ollama models..."):
                    result = test_ollama_connection_improved(ollama_server)
                    if result['status'] == 'success':
                        st.session_state.ai_settings['available_models'] = result['models']
                    
                    else:
                        st.session_state.ai_settings['available_models'] = []
                        if result['status'] == 'warning':
                            st.warning(result['message'])
                        else:
                            st.error(result['message'])
            
            elif ai_mode == "Cloud":
                with st.spinner("Auto-loading cloud models..."):
                    cloud_models = get_openrouter_models()
                    st.session_state.ai_settings['cloud_models'] = cloud_models
                    if cloud_models:
                      
                        # Auto-select first model if none selected
                        if not st.session_state.ai_settings.get('selected_cloud_model'):
                            st.session_state.ai_settings['selected_cloud_model'] = cloud_models[0]
                    else:
                        st.error("❌ Failed to load cloud models")
        
        if ai_mode == "Local":
            # Ollama server configuration
            ollama_server = st.text_input(
                "🖥️ Server:",
                value=st.session_state.ai_settings['ollama_server'],
                help="Ollama server address (default: http://localhost:11434)",
                key="ollama_server_input"
            )
            st.session_state.ai_settings['ollama_server'] = ollama_server
            
            # Test connection button only (no refresh)
            if st.button("🔍 Test & Get Models", key="test_ollama"):
                with st.spinner("Testing Ollama connection..."):
                    result = test_ollama_connection_improved(ollama_server)
                    
                    if result['status'] == 'success':
                       
                        st.session_state.ai_settings['available_models'] = result['models']
                        # Auto-select first model if none selected
                        if result['models'] and not st.session_state.ai_settings.get('selected_model'):
                            st.session_state.ai_settings['selected_model'] = result['models'][0]
                    elif result['status'] == 'warning':
                        st.warning(result['message'])
                        st.info(result['details'])
                        st.session_state.ai_settings['available_models'] = []
                    else:
                        st.error(result['message'])
                        st.error(result['details'])
                        st.session_state.ai_settings['available_models'] = []
            
            # Model selection dropdown
            available_models = st.session_state.ai_settings.get('available_models', [])
            if available_models:
                selected_model = st.selectbox(
                    "🤖 Select Model:",
                    available_models,
                    key="model_selector",
                    help="Choose which Ollama model to use for analysis"
                )
                st.session_state.ai_settings['selected_model'] = selected_model
                
            else:
                st.warning("⚠️ No models available. Test connection first or install models.")
                st.code("""# Install models with:
ollama pull llama3.2:3b
ollama pull mistral
ollama pull phi3:mini""")
                
                if 'selected_model' in st.session_state.ai_settings:
                    del st.session_state.ai_settings['selected_model']
        
        else:  # Cloud mode
            # OpenRouter configuration
            num_keys = len(st.session_state.ai_settings['openrouter_keys'])
            if num_keys > 0:
                
                
                # Auto-load cloud models when cloud mode is selected (if not changed above)
                if not mode_changed and not st.session_state.ai_settings.get('cloud_models'):
                    with st.spinner("Loading available models from OpenRouter..."):
                        cloud_models = get_openrouter_models()
                        st.session_state.ai_settings['cloud_models'] = cloud_models
                        
                        if cloud_models:
                           
                            # Auto-select first model if none selected
                            if not st.session_state.ai_settings.get('selected_cloud_model'):
                                st.session_state.ai_settings['selected_cloud_model'] = cloud_models[0]
                        else:
                            st.error("❌ Failed to load models")
                
                # Manual refresh button for cloud models
                if st.button("🔄 Refresh Models", key="refresh_cloud_models", help="Refresh cloud model list"):
                    with st.spinner("Refreshing models..."):
                        cloud_models = get_openrouter_models()
                        st.session_state.ai_settings['cloud_models'] = cloud_models
                        if cloud_models:
                            st.success(f"Refreshed: {len(cloud_models)} models")
                        else:
                            st.error("Failed to refresh models")
                
                # Cloud model selection
                cloud_models = st.session_state.ai_settings.get('cloud_models', [])
                if cloud_models:
                    # Create display names for models (all are free now)
                    model_options = []
                    for model in cloud_models:
                        display_name = f"{model['name']} - {model['context_length']:,} tokens"
                        model_options.append(display_name)
                    
                    selected_idx = st.selectbox(
                        "🤖 Select Model:",
                        range(len(model_options)),
                        format_func=lambda x: model_options[x],
                        key="cloud_model_selector",
                        help="Choose which OpenRouter model to use for analysis"
                    )
                    
                    if selected_idx is not None:
                        selected_cloud_model = cloud_models[selected_idx]
                        st.session_state.ai_settings['selected_cloud_model'] = selected_cloud_model
                        
                        # Show model info
                        model_info = selected_cloud_model
                        st.markdown(f"**Context:** {model_info['context_length']:,}")
                
                else:
                    st.warning("⚠️ Models are loading automatically. Please wait...")
                    st.info("If models don't load, try the refresh button.")
                    
            else:
                st.error("❌ No API keys found in secrets.toml")
                st.code(""".streamlit/secrets.toml:

[openrouter]
api_key1 = "your_key_1"
api_key2 = "your_key_2" 
api_key3 = "your_key_3"
                """)
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("### 🎯 Analysis Options")
        
        analysis_depth = st.selectbox(
            "Analysis Depth:",
            ["Quick Overview", "Detailed Analysis", "Deep Dive"],
            help="Choose analysis complexity",
            key="analysis_depth_select"
        )
        
        include_recommendations = st.checkbox(
            "Include Recommendations",
            value=True,
            help="Add music recommendations based on analysis",
            key="ai_include_recommendations"
        )
        
        # Check if we can analyze
        songs = st.session_state.get('identified_songs', [])
        can_analyze = (
            len(songs) > 0 and
            (
                (ai_mode == "Cloud" and 
                 len(st.session_state.ai_settings['openrouter_keys']) > 0 and
                 st.session_state.ai_settings.get('selected_cloud_model')) or
                (ai_mode == "Local" and st.session_state.ai_settings.get('selected_model'))
            )
        )
        
        # Show status
        if ai_mode == "Local":
            selected_model = st.session_state.ai_settings.get('selected_model')
            if selected_model:
                st.info(f"Ready with {selected_model}")
            else:
                st.warning("⚠️ Select a model to enable analysis")
        else:  # Cloud mode
            selected_cloud_model = st.session_state.ai_settings.get('selected_cloud_model')
            if selected_cloud_model:
                st.info(f"Ready with {selected_cloud_model['name']}")
            else:
                st.warning("⚠️ Load and select a model to enable analysis")
        
        # Analyze button
        analyze_button = st.button(
            "🚀 Analyze Playlist", 
            type="primary",
            disabled=not can_analyze,
            help="Analyze your playlist with AI" if can_analyze else 
                 "Select a model and ensure songs are loaded",
            key="analyze_playlist_btn"
        )

    # MAIN CONTENT AREA
    # Check if analysis was triggered
    if analyze_button:
        if not can_analyze:
            st.error("❌ Cannot analyze - please select a model and ensure songs are loaded.")
            return
        
        # Perform analysis
        with st.spinner("🤖 AI is analyzing your playlist..."):
            try:
                analysis_result = perform_ai_analysis(
                    st.session_state.identified_songs,
                    ai_mode,
                    analysis_depth,
                    include_recommendations
                )
                
                if analysis_result:
                    st.session_state.analysis_result = analysis_result
                    
                else:
                    st.error("❌ Analysis failed - please try again")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display results if available
    if 'analysis_result' in st.session_state:
        display_analysis_results(st.session_state.analysis_result)
    else:
        # Show getting started info
        st.markdown("""
        ### 🎵 Welcome to AI Playlist Analysis!
        
        **What this does:**
        - Analyzes your music taste and listening patterns
        - Provides personality insights based on your playlist
        - Gives personalized music recommendations
        - Offers interactive chat about your music
        
        **To get started:**
        1. Make sure you have songs loaded (check the debug info above)
        2. Configure your AI engine in the sidebar
        3. Select a model
        4. Click "Analyze Playlist"
        
        **Having issues?** Make sure you have songs loaded and a model selected in the sidebar.
        """)

# Replace the original perform_ai_analysis function call with clean version
# Replace the original perform_ai_analysis function call with the safe version
# In your analyze_button logic, use:
# analysis_result = perform_ai_analysis_safe(...) instead of perform_ai_analysis(...)

def get_openrouter_models():
    """Get list of available models from OpenRouter API"""
    try:
        # Use a random API key for model fetching
        keys = st.session_state.ai_settings.get('openrouter_keys', [])
        if not keys:
            return []
        
        api_key = random.choice(keys)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://radiosport-songid.streamlit.app",
            "X-Title": "RadioSport Song Analyzer"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('data', [])
            
            # Filter and format models - ONLY FREE MODELS
            filtered_models = []
            for model in models:
                model_info = {
                    'id': model.get('id', ''),
                    'name': model.get('name', model.get('id', '')),
                    'context_length': model.get('context_length', 0),
                    'pricing': model.get('pricing', {}),
                    'free': is_model_free(model.get('pricing', {}))
                }
                
                # Only include FREE models with context length info
                if model_info['context_length'] > 0 and model_info['free']:
                    filtered_models.append(model_info)
            
            # Sort by context length (highest first)
            filtered_models.sort(key=lambda x: -x['context_length'])
            
            return filtered_models
        else:
            return []
            
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return []

def is_model_free(pricing):
    """Check if a model is free based on pricing info"""
    if not pricing:
        return False
    
    prompt_price = pricing.get('prompt', '0')
    completion_price = pricing.get('completion', '0')
    
    try:
        return float(prompt_price) == 0 and float(completion_price) == 0
    except:
        return False

def call_cloud_ai(playlist_data, analysis_depth, include_recommendations):
    """Call OpenRouter cloud AI for analysis with selected model"""
    try:
        keys = st.session_state.ai_settings['openrouter_keys']
        if not keys:
            return "No API keys configured"
        
        selected_model = st.session_state.ai_settings.get('selected_cloud_model')
        if not selected_model:
            return "No model selected"
        
        # Load balance - randomly select a key
        api_key = random.choice(keys)
        
        # Create analysis prompt
        prompt = create_analysis_prompt(playlist_data, analysis_depth, include_recommendations)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://radiosport-songid.streamlit.app",
            "X-Title": "RadioSport Song Analyzer"
        }
        
        # Adjust max_tokens based on model context length
        max_tokens = min(2000, selected_model['context_length'] // 4)  # Use 1/4 of context for response
        
        payload = {
            "model": selected_model['id'],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90  # Increased timeout for larger models
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            try:
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                return f"Cloud AI Error: HTTP {response.status_code} - {error_detail}"
            except:
                return f"Cloud AI Error: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Cloud AI Error: {str(e)}"

def prepare_playlist_data(songs):
    """Prepare song data for AI analysis"""
    playlist_data = {
        'total_songs': len(songs),
        'songs': [],
        'artists': [],
        'timeframe': None
    }
    
    for song in songs:
        song_info = {
            'title': song.get('title', 'Unknown'),
            'artist': song.get('artist', 'Unknown'),
            'timestamp': song.get('timestamp', ''),
        }
        
        # Extract additional metadata if available
        if 'data' in song and 'track' in song['data']:
            track = song['data']['track']
            
            # Get genre
            if 'genres' in track:
                song_info['genre'] = track['genres'].get('primary', 'Unknown')
            
            # Get release info and album
            if 'sections' in track:
                for section in track['sections']:
                    if section.get('type') == 'SONG':
                        metadata = section.get('metadata', [])
                        for meta in metadata:
                            if meta.get('title') == 'Album':
                                song_info['album'] = meta.get('text', 'Unknown')
                            elif meta.get('title') == 'Released':
                                song_info['year'] = meta.get('text', 'Unknown')
        
        playlist_data['songs'].append(song_info)
        playlist_data['artists'].append(song_info['artist'])
    
    # Calculate timeframe
    timestamps = [s.get('timestamp') for s in playlist_data['songs'] if s.get('timestamp')]
    if timestamps:
        dates = sorted([t.split()[0] for t in timestamps])
        playlist_data['timeframe'] = {'start': dates[0], 'end': dates[-1]}
    
    return playlist_data

def call_local_ai(playlist_data, analysis_depth, include_recommendations):
    """Call local Ollama AI for analysis - always queries server for models"""
    try:
        server = st.session_state.ai_settings['ollama_server']
        selected_model = st.session_state.ai_settings.get('selected_model')
        
        if not selected_model:
            return "Error: No model selected. Please select a model first."
        
        # Verify the selected model is still available
        available_models = get_available_models(server)
        if not available_models:
            return "Error: Cannot connect to Ollama server or no models found"
        
        if selected_model not in available_models:
            return f"Error: Selected model '{selected_model}' is no longer available. Available models: {', '.join(available_models)}"
        
        # Create analysis prompt
        prompt = create_analysis_prompt(playlist_data, analysis_depth, include_recommendations)
        
        # Prepare payload with the user-selected model
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 2000,
                "stop": ["Human:", "Assistant:", "User:"]
            }
        }
        
        # Make the generation request
        response = requests.post(f"{server}/api/generate", json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response']
            else:
                return f"Error: Unexpected response format from Ollama"
        else:
            try:
                error_detail = response.json().get('error', 'Unknown error')
                return f"Error: HTTP {response.status_code} - {error_detail}"
            except:
                return f"Error: HTTP {response.status_code} - {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be too large or the server is overloaded."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama server. Make sure Ollama is running."
    except Exception as e:
        return f"Local AI Error: {str(e)}"

def create_analysis_prompt(playlist_data, analysis_depth, include_recommendations):
    """Create AI analysis prompt based on playlist data"""
    
    # Summarize playlist for prompt
    song_list = []
    for song in playlist_data['songs'][:20]:  # Limit to prevent token overflow
        song_str = f"'{song['title']}' by {song['artist']}"
        if song.get('genre'):
            song_str += f" ({song['genre']})"
        if song.get('year'):
            song_str += f" [{song['year']}]"
        song_list.append(song_str)
    
    artist_counts = Counter(playlist_data['artists']).most_common(5)
    top_artists = [f"{artist} ({count}x)" for artist, count in artist_counts]
    
    prompt = f"""Analyze this music playlist and provide insights about the listener's music taste and personality:

PLAYLIST DATA:
- Total Songs: {playlist_data['total_songs']}
- Top Artists: {', '.join(top_artists)}
- Timeframe: {playlist_data.get('timeframe', {}).get('start', 'Unknown')} to {playlist_data.get('timeframe', {}).get('end', 'Unknown')}

SONGS:
{chr(10).join(song_list)}

ANALYSIS REQUIREMENTS:
1. **Listener Profile**: What type of music lover is this person?
2. **Musical Preferences**: Dominant genres, eras, and styles
3. **Mood & Energy**: Overall playlist mood and energy levels
4. **Personality Insights**: What does this playlist say about their personality?
5. **Listening Patterns**: When and how might they listen to music?
6. **Musical Sophistication**: Are they casual listeners or music enthusiasts?
"""

    if analysis_depth == "Detailed Analysis":
        prompt += "\n7. **Seasonal/Temporal Patterns**: Any time-based listening trends"
        prompt += "\n8. **Cultural Influences**: Geographic or cultural music influences"
    
    elif analysis_depth == "Deep Dive":
        prompt += "\n7. **Emotional Resonance**: What emotions does this playlist evoke?"
        prompt += "\n8. **Social Context**: Group vs solo listening preferences"
        prompt += "\n9. **Musical Evolution**: How their taste might be evolving"
        prompt += "\n10. **Lifestyle Alignment**: How music fits their lifestyle"
    
    if include_recommendations:
        prompt += "\n\nRECOMMENDATIONS: Suggest 3-5 artists or songs they might enjoy based on this analysis."
    
    prompt += "\n\nProvide a comprehensive but concise analysis. Use emojis and clear formatting for readability."
    
    return prompt

def perform_ai_analysis(songs, ai_mode, analysis_depth, include_recommendations):
    """Perform AI analysis on playlist"""
    try:
        # Prepare data
        playlist_data = prepare_playlist_data(songs)
        
        # Call appropriate AI service
        if ai_mode == "Local":
            result = call_local_ai(playlist_data, analysis_depth, include_recommendations)
        else:
            result = call_cloud_ai(playlist_data, analysis_depth, include_recommendations)
        
        return {
            'analysis': result,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'mode': ai_mode,
            'depth': analysis_depth,
            'song_count': len(songs)
        }
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def display_analysis_results(analysis_result):
    """Display AI analysis results with formatting and chat query feature"""
    st.markdown("#### 🤖 AI Analysis Results")
    
    # Analysis metadata
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    
    with col_meta1:
        st.metric("🔧 AI Mode", analysis_result['mode'])
    with col_meta2:
        st.metric("📊 Depth", analysis_result['depth'])
    with col_meta3:
        st.metric("🎵 Songs", analysis_result['song_count'])
    
    st.markdown("---")
    
    # Main analysis content
    analysis_text = analysis_result['analysis']
    
    # Format the analysis text for better display
    if "Error:" in analysis_text:
        st.error(analysis_text)
    else:
        # Split by numbered sections if present
        sections = analysis_text.split('\n\n')
        
        for section in sections:
            if section.strip():
                # Check if it's a header (contains numbers or **bold**)
                if any(char in section for char in ['1.', '2.', '3.', '**', '#']):
                    st.markdown(section)
                else:
                    st.write(section)
    
    # Analysis timestamp
    st.caption(f"🕒 Generated: {analysis_result['timestamp']}")
    
    st.markdown("---")
    
    # Chat Query Section
    st.markdown("#### 💬 Ask AI About Your Playlist")
    st.markdown("*Ask specific questions about your music taste, get recommendations, or dive deeper into your listening habits.*")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize chat input value in session state if not exists
    if 'chat_input_value' not in st.session_state:
        st.session_state.chat_input_value = ""
    
    # Chat input
    chat_query = st.text_input(
        "Ask a question about your playlist:",
        placeholder="e.g., What mood does my playlist suggest? Recommend songs for working out?",
        key="chat_query_input",
        value=st.session_state.chat_input_value
    )
    
    # Chat buttons
    col_chat1, col_chat2, col_chat3 = st.columns([2, 1, 1])
    
    with col_chat1:
        ask_button = st.button("💬 Ask AI", key="ask_ai", type="primary")
    
    with col_chat2:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col_chat3:
        # Quick question buttons
        quick_questions = [
            "Recommend workout songs",
            "Analyze my mood patterns",
            "What genres dominate?",
            "Songs for relaxation",
            "Compare to popular trends"
        ]
        
        if st.button("⚡ Quick Questions", key="show_quick"):
            if 'show_quick_questions' not in st.session_state:
                st.session_state.show_quick_questions = True
            else:
                st.session_state.show_quick_questions = not st.session_state.show_quick_questions
    
    # Show quick question buttons
    if st.session_state.get('show_quick_questions', False):
        st.markdown("**Quick Questions:**")
        quick_cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            with quick_cols[i % 3]:
                if st.button(f"💡 {question}", key=f"quick_{i}"):
                    st.session_state.chat_input_value = question
                    st.rerun()
    
    # Process chat query
    if ask_button and chat_query.strip():
        # Check if AI is properly configured
        ai_mode = st.session_state.ai_settings['ai_mode']
        can_chat = (
            (ai_mode == "Cloud" and 
             len(st.session_state.ai_settings['openrouter_keys']) > 0 and
             st.session_state.ai_settings.get('selected_cloud_model')) or
            (ai_mode == "Local" and st.session_state.ai_settings.get('selected_model'))
        )
        
        if not can_chat:
            st.error("❌ AI not configured. Please select a model in the sidebar first.")
        else:
            with st.spinner("🤖 AI is thinking..."):
                chat_response = perform_chat_query(
                    st.session_state.identified_songs,
                    chat_query,
                    ai_mode
                )
                
                if chat_response:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': chat_query,
                        'answer': chat_response,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
            
            # Clear input by updating session state value
            st.session_state.chat_input_value = ""
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### 💭 Chat History")
        
        # Reverse order to show newest first
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"💬 {chat['question']} ({chat['timestamp']})", expanded=(i == 0)):
                if "Error:" in chat['answer']:
                    st.error(chat['answer'])
                else:
                    st.write(chat['answer'])
    
    st.markdown("---")
    
    # Actions
    col_act1, col_act2, col_act3 = st.columns(3)
    
    with col_act1:
        if st.button("🔄 Refresh Analysis", key="refresh_analysis"):
            if hasattr(st.session_state, 'analysis_result'):
                del st.session_state.analysis_result
            st.rerun()
    
    with col_act2:
        # Export analysis
        export_data = f"""RadioSport AI Playlist Analysis
Generated: {analysis_result['timestamp']}
Mode: {analysis_result['mode']} | Depth: {analysis_result['depth']} | Songs: {analysis_result['song_count']}

{analysis_text}
"""
        
        # Add chat history to export
        if st.session_state.chat_history:
            export_data += "\n\n" + "="*50 + "\nCHAT HISTORY\n" + "="*50 + "\n"
            for chat in st.session_state.chat_history:
                export_data += f"\nQ ({chat['timestamp']}): {chat['question']}\nA: {chat['answer']}\n{'-'*30}\n"
        
        st.download_button(
            "📤 Export Analysis",
            data=export_data,
            file_name=f"playlist_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col_act3:
        # Export chat only
        if st.session_state.chat_history:
            chat_export = "RadioSport AI Chat History\n" + "="*40 + "\n"
            for chat in st.session_state.chat_history:
                chat_export += f"\nQ ({chat['timestamp']}): {chat['question']}\nA: {chat['answer']}\n{'-'*30}\n"
            
            st.download_button(
                "💬 Export Chat",
                data=chat_export,
                file_name=f"playlist_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Export chat history only"
            )

def get_available_models(server_url):
    """Get list of available models from Ollama server"""
    try:
        response = requests.get(f"{server_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            return [model['name'] for model in models]
        else:
            return []
    except:
        return []

def test_ollama_connection_improved(server_url):
    """Improved Ollama connection testing with detailed feedback"""
    try:
        # Test basic connectivity
        response = requests.get(f"{server_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            
            if models:
                model_info = []
                for model in models[:5]:  # Show first 5 models
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    size_gb = size / (1024**3) if size > 0 else 0
                    modified = model.get('modified_at', '')[:10]  # Just date part
                    model_info.append(f"• {name} ({size_gb:.1f}GB, {modified})")
                
                return {
                    'status': 'success',
                    'message': f"✅ Connected successfully! Found {len(models)} model(s)",
                    'details': "\n".join(model_info),
                    'models': [m['name'] for m in models]
                }
            else:
                return {
                    'status': 'warning',
                    'message': "⚠️ Connected but no models found",
                    'details': "Install a model first:\n• ollama pull llama3.2:3b\n• ollama pull mistral",
                    'models': []
                }
        else:
            return {
                'status': 'error',
                'message': f"❌ Server responded with HTTP {response.status_code}",
                'details': response.text[:200] if response.text else "No error details",
                'models': []
            }
            
    except requests.exceptions.ConnectionError:
        return {
            'status': 'error',
            'message': "❌ Cannot connect to server",
            'details': "Make sure Ollama is running:\n• Start Ollama desktop app, or\n• Run 'ollama serve' in terminal",
            'models': []
        }
    except requests.exceptions.Timeout:
        return {
            'status': 'error',
            'message': "❌ Connection timeout",
            'details': "Server is not responding. Check if Ollama is running properly.",
            'models': []
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': "❌ Unexpected error",
            'details': str(e),
            'models': []
        }

#-------- AI functions End.
#-------- AI functions End.

# =============================================================================
# Offline Music ID Tab
# =============================================================================

def offline_music_tab():
    """Offline music identification interface"""
    st.markdown("### 🔌 Offline Music Identification")
    st.write("Identify songs without internet connection using audio fingerprinting")
    
    # Database stats
    db_stats = st.session_state.offline_db_stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Songs in Database", db_stats['songs'])
    with col2:
        st.metric("Unique Artists", db_stats['artists'])
    with col3:
        st.metric("Total Fingerprints", f"{db_stats['fingerprints']:,}")
    
    st.markdown("---")
    
    # Identification section
    st.markdown("#### 🔍 Identify Song")
    st.write("Upload an audio file to identify it from your offline database")
    
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        key="offline_identify",
        accept_multiple_files=True  # Enable multiple file uploads
    )
    
    # Batch processing button
    if uploaded_files and st.button("Identify All Songs", type="primary", key="batch_identify_btn"):
        # Create a placeholder for results
        results_placeholder = st.empty()
        results_placeholder.info("Processing files...")
        
        # Prepare container for results table
        results_container = st.container()
        
        # Initialize results list
        batch_results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing file {i+1} of {len(uploaded_files)}: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Analyze the file
                result = identify_song_offline(tmp_file_path)
                
                # Add to results
                batch_results.append({
                    'file': uploaded_file.name,
                    'result': result
                })
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results in a table
        with results_container:
            st.markdown("### 📊 Batch Identification Results")
            
            # Create table data
            table_data = []
            for res in batch_results:
                if res['result']:
                    table_data.append({
                        "File": res['file'],
                        "Status": "✅ Identified",
                        "Song": res['result']['name'],
                        "Artist": res['result']['artist'],
                        "Confidence": f"{res['result']['confidence']:.1%}",
                        "Details": "View"
                    })
                else:
                    table_data.append({
                        "File": res['file'],
                        "Status": "❌ Not Found",
                        "Song": "",
                        "Artist": "",
                        "Confidence": "",
                        "Details": ""
                    })
            
            # Show results in a table
            st.dataframe(table_data, use_container_width=True)
            
            # Add download button for results
            csv_data = "File,Status,Song,Artist,Confidence\n"
            for row in table_data:
                csv_data += f"{row['File']},{row['Status']},{row['Song']},{row['Artist']},{row['Confidence']}\n"
            
            st.download_button(
                "📥 Download Results as CSV",
                data=csv_data,
                file_name="batch_song_identification_results.csv",
                mime="text/csv"
            )
            
        results_placeholder.empty()
    
    # Single file identification
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.audio(uploaded_file)
            
            if st.button(f"Identify {uploaded_file.name}", key=f"identify_{uploaded_file.name}"):
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        result = identify_song_offline(tmp_file_path)
                        
                        if result:
                            st.success("🎉 Song identified offline!")
                            display_offline_song_info(result)
                        else:
                            st.warning("🤷 Song not found in offline database")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
    
    st.markdown("---")
    
    # Add song section
    st.markdown("#### ➕ Add Song to Database")
    st.write("Build your offline database by adding new songs with full details from Shazam")
    
    col1, col2 = st.columns(2)
    with col1:
        song_name = st.text_input("Song Name", key="offline_song_name")
        artist_name = st.text_input("Artist", key="offline_artist_name")
    with col2:
        audio_file = st.file_uploader(
            "Audio File",
            type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
            key="offline_add_file"
        )
    
    if audio_file:
        st.audio(audio_file)
    
    if st.button("Add to Database", key="offline_add_btn"):
        if not song_name:
            st.error("Please enter a song name")
        elif not audio_file:
            st.error("Please upload an audio file")
        else:
            with st.spinner("Processing audio and creating fingerprint..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    success = add_song_to_offline_db(
                        tmp_file_path, 
                        song_name, 
                        artist_name or "Unknown"
                    )
                    
                    if success:
                        st.success(f"✅ Added '{song_name}' to offline database with full details!")
                        st.rerun()
                    else:
                        st.error("Failed to process audio file")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
    
    # NEW SECTION: Batch add songs to offline database
    st.markdown("---")
    st.markdown("#### ➕➕ Batch Add Songs to Database")
    st.write("Upload multiple audio files to automatically identify and add to offline database")
    
    batch_files = st.file_uploader(
        "Choose audio files (batch)",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        key="offline_batch_add",
        accept_multiple_files=True
    )
    
    if batch_files and st.button("Add All to Database", key="batch_add_btn"):
        # Create a placeholder for results
        results_placeholder = st.empty()
        results_placeholder.info("Processing batch files...")
        
        # Prepare container for results
        results_container = st.container()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results tracking
        success_count = 0
        fail_count = 0
        batch_results = []
        
        # Process each file
        for i, uploaded_file in enumerate(batch_files):
            # Update progress
            progress = (i + 1) / len(batch_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {i+1}/{len(batch_files)}: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Identify song using Shazam
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    shazam_result = loop.run_until_complete(identify_song(tmp_file_path))
                finally:
                    loop.close()
                
                if shazam_result and 'track' in shazam_result:
                    track = shazam_result['track']
                    title = track.get('title', 'Unknown')
                    artist = track.get('subtitle', 'Unknown')
                    
                    # Add to offline DB
                    if add_song_to_offline_db(tmp_file_path, title, artist):
                        success_count += 1
                        batch_results.append({
                            'file': uploaded_file.name,
                            'status': '✅ Success',
                            'song': title,
                            'artist': artist
                        })
                    else:
                        fail_count += 1
                        batch_results.append({
                            'file': uploaded_file.name,
                            'status': '❌ Failed',
                            'song': '',
                            'artist': ''
                        })
                else:
                    fail_count += 1
                    batch_results.append({
                        'file': uploaded_file.name,
                        'status': '❌ Not identified',
                        'song': '',
                        'artist': ''
                    })
            except Exception as e:
                fail_count += 1
                batch_results.append({
                    'file': uploaded_file.name,
                    'status': f'❌ Error: {str(e)}',
                    'song': '',
                    'artist': ''
                })
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        results_placeholder.empty()
        
        # Display results
        with results_container:
            st.success(f"✅ Batch processing complete! Success: {success_count}, Failed: {fail_count}")
            
            # Show results in a table
            if batch_results:
                st.markdown("### 📊 Batch Processing Results")
                
                # Create table data
                table_data = []
                for res in batch_results:
                    table_data.append({
                        "File": res['file'],
                        "Status": res['status'],
                        "Song": res.get('song', ''),
                        "Artist": res.get('artist', '')
                    })
                
                st.dataframe(table_data, use_container_width=True)
                
                # Add download button for results
                csv_data = "File,Status,Song,Artist\n"
                for row in table_data:
                    csv_data += f"{row['File']},{row['Status']},{row['Song']},{row['Artist']}\n"
                
                st.download_button(
                    "📥 Download Results as CSV",
                    data=csv_data,
                    file_name="batch_song_addition_results.csv",
                    mime="text/csv"
                )
        
        # Update stats and rerun
        update_offline_db_stats()
        st.rerun()
    
    st.markdown("---")
    
    # Database browser
    st.markdown("#### 📚 Database Contents")
    
    if st.session_state.offline_database:
        # Convert database to DataFrame for display
        db_data = []
        for song_id, song_info in st.session_state.offline_database.items():
            db_data.append({
                'Song': song_info['name'],
                'Artist': song_info['artist'],
                'Album': song_info.get('album', 'Unknown'),
                'Released': song_info.get('released', 'Unknown'),
                'Genre': song_info.get('genre', 'Unknown'),
                'Fingerprint Points': len(song_info['fingerprint'])
            })
        
        # Sort by song name
        db_data_sorted = sorted(db_data, key=lambda x: x['Song'])
        
        # Display in a table
        st.dataframe(db_data_sorted, use_container_width=True)
        
        # Export option
        if st.button("💾 Export Database", key="export_offline_db"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                pickle.dump(st.session_state.offline_database, tmp_file)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as f:
                st.download_button(
                    "📥 Download Database",
                    f,
                    file_name="radiosport_offline_db.pkl",
                    mime="application/octet-stream"
                )
    else:
        st.info("No songs in offline database yet. Add some songs using the form above!")

def main_tab():
    """Main song identification interface - desktop only"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎤 Desktop Microphone Recording")
        
        # Audio device selection
        if not st.session_state.audio_devices:
            st.session_state.audio_devices = get_audio_devices()
        
        if st.session_state.audio_devices:
            device_options = [f"🎤 {device['name']}" for device in st.session_state.audio_devices]
            selected_device_idx = st.selectbox(
                "Audio Source:", range(len(device_options)),
                format_func=lambda x: device_options[x], key="device_selector"
            )
            st.session_state.selected_device = st.session_state.audio_devices[selected_device_idx]['index']
            selected_device_info = st.session_state.audio_devices[selected_device_idx]

        else:
            st.error("No audio input devices found!")
            if st.button("🔄 Refresh Devices", key="refresh_devices_btn"):
                st.session_state.audio_devices = get_audio_devices()
                st.rerun()
            return
        
        # Offline identification setting
        st.session_state.use_offline_id = st.checkbox(
            "🔌 Use Offline Identification",
            value=st.session_state.use_offline_id,
            help="Use offline fingerprinting instead of Shazam (requires pre-trained database)",
            key="use_offline_id_main"  # Unique key for main tab
        )
        
        # Add to offline DB setting
        st.session_state.add_to_offline_db = st.checkbox(
            "📥 Add to Offline Database",
            value=st.session_state.add_to_offline_db,
            help="Add successfully identified songs to offline database with full details",
            key="add_to_offline_db_main"  # Unique key for main tab
        )
        
        # Main recording button
        if st.button("🎤 Start Recording & Identify Song", type="primary", key="main_record_btn"):
            with st.spinner("Recording and analyzing..."):
                audio_file = record_audio_with_auto_gain(st.session_state.selected_device)
                if audio_file:
                    process_audio_file(audio_file)
        
        # Tips section
        st.markdown("---")
        st.markdown("**💡 Tips for better identification:**")
        st.markdown("- Play music clearly without background noise")
        st.markdown("- Hold microphone close to the audio source")
        st.markdown("- Avoid speaking during recording")
        st.markdown("- Use high quality audio sources when possible")
    
    with col2:
        st.markdown("### 🎵 Current Song Result")
        if hasattr(st.session_state, 'current_song_result') and st.session_state.current_song_result:
            if st.session_state.current_song_result.get('offline'):
                st.info("🔌 Identified using offline database")
                display_song_info(st.session_state.current_song_result)
            else:
                display_song_info(st.session_state.current_song_result)
        else:
            # Placeholder content
            st.info("🎯 Song identification results will appear here after recording.")
            st.markdown("**How it works:**")
            st.markdown("1. Click the recording button")
            st.markdown("2. Play music near your microphone")
            st.markdown("3. The app will analyze the audio")
            st.markdown("4. Results appear here and in history")
            
            # Show offline database stats if available
            if st.session_state.offline_db_stats['songs'] > 0:
                st.markdown("---")
                st.markdown("### 🔌 Offline Database Stats")
                st.metric("Songs", st.session_state.offline_db_stats['songs'])
                st.metric("Artists", st.session_state.offline_db_stats['artists'])
                st.metric("Fingerprints", f"{st.session_state.offline_db_stats['fingerprints']:,}")

def main():
    """Main application with enhanced tabbed interface including Auto Mode"""
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎤 Song ID", "📻 Auto Mode", "📚 History", "🔌 Offline ID", "🤖 AI Analysis", "📊 Stats"
    ])
    
    with tab1:
        main_tab()
    
    with tab2:
        auto_mode_tab()
    
    with tab3:
        history_tab()
        
    with tab4:
        offline_music_tab()
        
    with tab5:
        ai_analysis_tab()
        
    with tab6:
        st.markdown("### 📊 Application Statistics")
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Songs Identified", len(st.session_state.identified_songs))
        with col2:
            st.metric("Offline DB Songs", st.session_state.offline_db_stats['songs'])
        with col3:
            st.metric("Offline Fingerprints", f"{st.session_state.offline_db_stats['fingerprints']:,}")
        
        st.markdown("---")
        
        # Database info
        st.markdown("#### 🔌 Offline Database Info")
        if st.session_state.offline_database:
            artists = set()
            for song_id, song_data in st.session_state.offline_database.items():
                artists.add(song_data['artist'])
            
            st.write(f"**Unique Artists:** {len(artists)}")
            
            # Artist distribution
            artist_counts = {}
            for song_data in st.session_state.offline_database.values():
                artist = song_data['artist']
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
            
            # Top artists
            top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            st.write("**Top Artists:**")
            for artist, count in top_artists:
                st.write(f"- {artist}: {count} songs")
        else:
            st.info("No songs in offline database yet")
        
        st.markdown("---")
        
        # System info
        st.markdown("#### ⚙️ System Information")
        st.write(f"**Python Version:** {sys.version.split()[0]}")
        st.write(f"**Streamlit Version:** {st.__version__}")
        st.write(f"**Audio Devices:** {len(st.session_state.audio_devices)} available")

if __name__ == "__main__":
    main()