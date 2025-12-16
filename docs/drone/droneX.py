import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import random
import time
import json

# ============================================================================
# APP METADATA & VERSION TRACKING
# ============================================================================
APP_VERSION = "1.3.0"
APP_NAME = "RadioSport X-Repeater"
APP_DESCRIPTION = "Drone-Borne Repeater RF Coverage Analyzer"
DEVELOPER = "RNK"
COPYRIGHT = "Copyright © RNK, 2026 RadioSport. All rights reserved."
GITHUB_URL = "https://github.com/rkarikari/stem"

# ============================================================================
# INITIALIZE UI
# ============================================================================
def initialize_ui():
    st.set_page_config(
        page_title=f"{APP_NAME} - {APP_DESCRIPTION}",
        page_icon="📡",
        layout="wide",
        menu_items={
            'Report a Bug': GITHUB_URL,
            'About': f"{APP_NAME} v{APP_VERSION}\n\n{COPYRIGHT}"
        }
    )

initialize_ui()

# ============================================================================
# CUSTOM CSS FOR IMPROVED UI
# ============================================================================
st.markdown(f"""
<style>
    .app-title {{
        font-size: 28px;
        font-weight: 700;
        color: #1a73e8;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #1a73e8 0%, #6c8ef5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .app-subtitle {{
        font-size: 14px;
        color: #5f6368;
        margin: 0 0 10px 0;
        padding: 0;
    }}
    .version-badge {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 8px;
    }}
    .main-header {{
        font-size: 24px;
        font-weight: 600;
        color: #1f77b4;
        margin: 0;
        padding: 0;
    }}
    .sub-header {{
        font-size: 13px;
        color: #666;
        margin: 0 0 10px 0;
        padding: 0;
    }}
    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }}
    .stMetric {{
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }}
    div[data-testid="stExpander"] {{
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }}
    .sidebar-title {{
        font-size: 22px;
        font-weight: 700;
        color: #1a73e8;
        margin: 0 0 15px 0;
        padding: 0;
        text-align: center;
        background: linear-gradient(135deg, #1a73e8 0%, #6c8ef5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .footer {{
        text-align: center;
        color: #666;
        font-size: 12px;
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid #e0e0e0;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'drone_location' not in st.session_state:
    st.session_state.drone_location = None
if 'saved_configs' not in st.session_state:
    st.session_state.saved_configs = []
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False
if 'current_tip' not in st.session_state:
    st.session_state.current_tip = ""
if 'last_cleanup' not in st.session_state:
    st.session_state.last_cleanup = pd.Timestamp.now()
if 'app_start_time' not in st.session_state:
    st.session_state.app_start_time = pd.Timestamp.now()




# Load saved configs from file
if 'saved_configs' not in st.session_state:
    try:
        with open('saved_configs.json', 'r') as f:
            st.session_state.saved_configs = json.load(f)
    except:
        st.session_state.saved_configs = []

# ============================================================================
# RF CALCULATION FUNCTIONS (OPTIMIZED & CACHED)
# ============================================================================

# Clear cache on code changes - increment this number to force recalculation
CACHE_VERSION = 2  # Changed from 1 to force recalculation

@st.cache_data(ttl=1)
def calculate_fspl_db(distance_km, freq_mhz, _cache_version=CACHE_VERSION):
    """Calculate Free Space Path Loss in dB - pure physics, no artificial corrections"""
    if distance_km <= 0.001:
        return 0
    distance_m = distance_km * 1000
    # FSPL = 20×log₁₀(d) + 20×log₁₀(f) + 32.45
    # This formula already accounts for frequency-dependent path loss
    fspl = 20 * np.log10(distance_m) + 20 * np.log10(freq_mhz) + 32.45
    return fspl

@st.cache_data(ttl=1)
def calculate_two_ray_model(distance_km, freq_mhz, h_tx_m, h_rx_m):
    """
    Calculate path loss using Two-Ray Ground Reflection Model
    Physics-based model for flat earth with ground reflection
    """
    if distance_km <= 0.001:
        return 0
    
    distance_m = distance_km * 1000
    wavelength_m = 3e8 / (freq_mhz * 1e6)
    
    # Calculate critical distance (breakpoint)
    d_critical_m = (4 * np.pi * h_tx_m * h_rx_m) / wavelength_m
    
    if distance_m <= 0:
        return 0
    elif distance_m < d_critical_m:
        # Use free space before breakpoint
        return calculate_fspl_db(distance_km, freq_mhz)
    else:
        # Two-ray model: PL = 40×log₁₀(d) - 20×log₁₀(h_tx) - 20×log₁₀(h_rx)
        # This formula already includes frequency effects via wavelength
        base_loss = 40 * np.log10(distance_m) - 20 * np.log10(h_tx_m) - 20 * np.log10(h_rx_m)
        
        return base_loss

@st.cache_data(ttl=1)
def calculate_okumura_hata(distance_km, freq_mhz, h_tx_m, h_rx_m, environment='suburban'):
    """
    Calculate path loss using Okumura-Hata model for VHF/UHF
    Valid for: 150-1500 MHz, 1-20 km, h_tx: 30-200m, h_rx: 1-10m
    """
    # Check validity range
    if freq_mhz < 150 or freq_mhz > 1500:
        # Outside Okumura-Hata range, use free space with VHF correction
        if freq_mhz < 150:
            # For VHF frequencies, use free space with VHF ground wave advantage
            return calculate_fspl_db(distance_km, freq_mhz) - 8
        else:
            return calculate_fspl_db(distance_km, freq_mhz)
    
    if distance_km < 1 or distance_km > 20:
        # Use free space for distances outside 1-20 km
        return calculate_fspl_db(distance_km, freq_mhz)
    
    if h_tx_m < 30:
        # For drone altitudes < 30m, apply correction factor
        drone_correction = 20 * np.log10(30 / max(h_tx_m, 10))
        h_tx_m = 30
    else:
        drone_correction = 0
    
    # Convert distance to meters for calculations
    d_km = min(max(distance_km, 1), 20)
    
    # Calculate a(hr) - mobile station antenna height correction factor
    if freq_mhz <= 300:
        a_hr = (1.1 * np.log10(freq_mhz) - 0.7) * h_rx_m - (1.56 * np.log10(freq_mhz) - 0.8)
    else:
        a_hr = 8.29 * (np.log10(1.54 * h_rx_m))**2 - 1.1 if freq_mhz <= 200 else \
               3.2 * (np.log10(11.75 * h_rx_m))**2 - 4.97
    
    # Calculate basic path loss for urban area
    L_urban = (69.55 + 26.16 * np.log10(freq_mhz) - 
               13.82 * np.log10(h_tx_m) - a_hr + 
               (44.9 - 6.55 * np.log10(h_tx_m)) * np.log10(d_km))
    
    # Apply environment corrections
    if environment == 'urban':
        L = L_urban
    elif environment == 'suburban':
        # Suburban area correction
        L = L_urban - 2 * (np.log10(freq_mhz / 28))**2 - 5.4
    else:  # rural
        # Rural area correction
        L = L_urban - 4.78 * (np.log10(freq_mhz))**2 + 18.33 * np.log10(freq_mhz) - 40.94
    
    # Apply drone altitude correction if needed
    L += drone_correction
    
    return L

@st.cache_data(ttl=60)
def calculate_itu_p1546(distance_km, freq_mhz, h_tx_m, time_percent=50):
    """
    Calculate path loss using ITU-R P.1546 model
    Properly implements frequency interpolation per ITU-R P.1546 standard
    """
    if distance_km <= 0.001:
        return 0
    
    # Clamp to valid ranges
    d_km = min(max(distance_km, 1), 1000)
    h_eff = max(h_tx_m, 10)
    
    # Determine reference frequency for interpolation
    if freq_mhz <= 100:
        # Below 100 MHz: extrapolate from 100 MHz curve
        ref_freq = 100
        freq_correction = -20 * np.log10(freq_mhz / ref_freq)
    elif freq_mhz <= 600:
        # Between 100-600 MHz: interpolate between 100 and 600 MHz curves
        # Use logarithmic interpolation
        ref_freq = 600
        freq_correction = -9 * np.log10(freq_mhz / 100) / np.log10(6)
    elif freq_mhz <= 2000:
        # Between 600-2000 MHz: interpolate between 600 and 2000 MHz curves
        ref_freq = 600
        freq_correction = 9.3 * np.log10(freq_mhz / 600) / np.log10(3.33)
    else:
        # Above 2000 MHz: extrapolate from 2000 MHz curve
        ref_freq = 2000
        freq_correction = 20 * np.log10(freq_mhz / ref_freq)
    
    # Base field strength at reference frequency (600 MHz)
    # Using empirical model: E = E0 - 20×log(d) for land paths
    E0 = 106.9
    
    # Distance attenuation (20 dB per decade)
    distance_loss = 20 * np.log10(d_km)
    
    # Height gain correction (more benefit at higher altitudes)
    if h_eff <= 10:
        height_gain = 0
    elif h_eff <= 50:
        height_gain = 20 * np.log10(h_eff / 10)
    elif h_eff <= 300:
        height_gain = 20 + 10 * np.log10(h_eff / 50)
    else:
        height_gain = 20 + 10 * np.log10(300 / 50)
    
    # Time percentage correction (time variability)
    if time_percent >= 50:
        time_factor = 0
    elif time_percent >= 10:
        time_factor = 8  # 10-50%
    elif time_percent >= 1:
        time_factor = 15  # 1-10%
    else:
        time_factor = 25  # <1%
    
    # Calculate field strength in dBµV/m
    E = E0 - distance_loss + height_gain + freq_correction + time_factor
    
    # Convert field strength to path loss
    # PL = EIRP - E + 20×log(f) - 77.2
    # For 1kW ERP: EIRP_dBm = 60 dBm
    # E is in dBµV/m, need to convert to received power
    pl = 139.3 - E + 20 * np.log10(freq_mhz) - 77.2
    
    # Ensure path loss is not less than free space
    min_loss = calculate_fspl_db(distance_km, freq_mhz)
    return max(pl, min_loss)

@st.cache_data
def calculate_vhf_advantage(distance_km, freq_mhz):
    """
    [DEPRECATED in v1.2.0]
    VHF advantage now handled by propagation models directly.
    This function kept for API compatibility but returns 0.
    """
    return 0

@st.cache_data
def calculate_uhf_penalty(distance_km, freq_mhz, environment='suburban'):
    """
    [DEPRECATED in v1.2.0]
    UHF penalty now handled by propagation models directly.
    This function kept for API compatibility but returns 0.
    """
    return 0


@st.cache_data
def calculate_path_loss_db(distance_km, freq_mhz, n=2.0, altitude_m=0, 
                          propagation_model='simple', h_rx_m=2.0, environment='suburban',
                          time_percent=50):
    """
    Calculate path loss with optional time percentage for statistical models.
    
    v1.2.0 FIX: Removed custom VHF advantage and UHF penalty.
    Propagation models now handle frequency effects naturally without artificial adjustments.
    This produces physically realistic VHF/UHF range ratios (1.2-1.5:1) instead of 
    impossible ratios (10:1).
    """
    if distance_km <= 0.001:
        return 0
    
    # Altitude bonus (reduced path loss at higher altitudes)
    altitude_bonus = max(0, (altitude_m / 1000) * 3)
    
    # Get base path loss from selected model
    if propagation_model == 'fspl':
        base_loss = calculate_fspl_db(distance_km, freq_mhz)
    elif propagation_model == 'two_ray':
        base_loss = calculate_two_ray_model(distance_km, freq_mhz, altitude_m, h_rx_m)
    elif propagation_model == 'okumura_hata':
        if freq_mhz < 150:
            base_loss = calculate_itu_p1546(distance_km, freq_mhz, altitude_m, time_percent)
        else:
            base_loss = calculate_okumura_hata(distance_km, freq_mhz, altitude_m, h_rx_m, environment)
    elif propagation_model == 'itu_p1546':
        base_loss = calculate_itu_p1546(distance_km, freq_mhz, altitude_m, time_percent)
    else:
        if n == 2.0:
            base_loss = calculate_fspl_db(distance_km, freq_mhz)
        else:
            fspl_1km = calculate_fspl_db(1.0, freq_mhz)
            base_loss = fspl_1km + 10 * n * np.log10(distance_km)
    
    # Apply only altitude bonus - let models handle frequency effects naturally
    final_loss = base_loss - altitude_bonus
    
    # Ensure loss is at least free space loss
    min_loss = calculate_fspl_db(distance_km, freq_mhz)
    return max(final_loss, min_loss)

@st.cache_data
def calculate_fresnel_zone(distance_km, freq_mhz, zone_percent=60):
    """Calculate Fresnel zone radius at given percentage"""
    if distance_km <= 0:
        return 0
    wavelength = 3e8 / (freq_mhz * 1e6)
    distance_m = distance_km * 1000
    return np.sqrt(wavelength * distance_m / 4) * (zone_percent / 100)

@st.cache_data
def calculate_received_power(tx_power_w, tx_gain_dbi, rx_gain_dbi, distance_km, 
                            freq_mhz, n=2.0, additional_loss_db=0, swr=1.0, 
                            altitude_m=0, propagation_model='simple', h_rx_m=2.0, 
                            environment='suburban', time_percent=50,
                            polarization_mismatch=0, antenna_efficiency=0.9):
    """Calculate received power with all losses considered"""
    if distance_km <= 0.001:
        return 100
    
    swr_loss = 10 * np.log10(swr) if swr > 1.0 else 0
    tx_power_dbm = 10 * np.log10(tx_power_w * 1000)
    
    efficiency_loss = -10 * np.log10(antenna_efficiency) if antenna_efficiency < 1.0 else 0
    
    path_loss = calculate_path_loss_db(distance_km, freq_mhz, n, altitude_m, 
                                      propagation_model, h_rx_m, environment, time_percent)
    
    total_losses = (path_loss + additional_loss_db + swr_loss + 
                   polarization_mismatch + efficiency_loss)
    
    rx_power_dbm = (tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_losses)
    
    return rx_power_dbm

@st.cache_data
def calculate_sensitivity_from_nf(nf_db, bandwidth_khz, snr_required_db=12):
    """Calculate receiver sensitivity from noise figure and bandwidth"""
    bandwidth_hz = bandwidth_khz * 1000
    noise_floor_dbm = -174 + 10 * np.log10(bandwidth_hz)
    
    theoretical_sensitivity = noise_floor_dbm + nf_db + snr_required_db
    
    min_practical_sensitivity = -127
    
    return max(theoretical_sensitivity, min_practical_sensitivity)

@st.cache_data
def calculate_range(tx_power_w, tx_gain_dbi, rx_gain_dbi, rx_sensitivity_dbm,
                   freq_mhz, n=2.0, additional_loss_db=0, swr=1.0, fade_margin_db=0, 
                   altitude_m=0, propagation_model='simple', h_rx_m=2.0, 
                   environment='suburban', time_percent=50,
                   polarization_mismatch=0, antenna_efficiency=0.9):
    """Calculate maximum range with improved convergence"""
    if tx_power_w <= 0:
        return 0
    
    # Calculate radio horizon first
    radio_horizon_km = 4.12 * np.sqrt(altitude_m)
    
    # For very low altitudes, limit maximum search distance
    if radio_horizon_km < 1.0:
        radio_horizon_km = max(radio_horizon_km, 0.5)
    
    required_power = rx_sensitivity_dbm + fade_margin_db
    
    # Check if we can reach the horizon
    rx_power_at_horizon = calculate_received_power(
        tx_power_w, tx_gain_dbi, rx_gain_dbi, radio_horizon_km, freq_mhz, n,
        additional_loss_db, swr, altitude_m, propagation_model, h_rx_m, 
        environment, time_percent, polarization_mismatch, antenna_efficiency
    )
    
    if rx_power_at_horizon >= required_power:
        return radio_horizon_km
    
    # Binary search with improved tolerance and iteration limits
    min_dist, max_dist = 0.01, radio_horizon_km  # Start at 10m minimum
    tolerance = 0.001  # 1 meter precision
    max_iterations = 50  # Reduced iterations with better tolerance
    iteration = 0
    
    # Quick check at 1 km to avoid unnecessary iterations
    if max_dist >= 1.0:
        rx_power_1km = calculate_received_power(
            tx_power_w, tx_gain_dbi, rx_gain_dbi, 1.0, freq_mhz, n,
            additional_loss_db, swr, altitude_m, propagation_model, h_rx_m, 
            environment, time_percent, polarization_mismatch, antenna_efficiency
        )
        if rx_power_1km < required_power:
            # Range is less than 1 km, search between 0.01 and 1
            max_dist = 1.0
    
    while (max_dist - min_dist > tolerance) and (iteration < max_iterations):
        mid_dist = (min_dist + max_dist) / 2
        rx_power = calculate_received_power(
            tx_power_w, tx_gain_dbi, rx_gain_dbi, mid_dist, freq_mhz, n,
            additional_loss_db, swr, altitude_m, propagation_model, h_rx_m, 
            environment, time_percent, polarization_mismatch, antenna_efficiency
        )
        
        if rx_power >= required_power:
            min_dist = mid_dist
        else:
            max_dist = mid_dist
        
        iteration += 1
    
    # Return the conservative estimate (lower bound)
    return min_dist

@st.cache_data
def calculate_radio_horizon(altitude_m):
    """Calculate radio horizon distance in km"""
    return 4.12 * np.sqrt(altitude_m)

# ============================================================================
# UI HEADER
# ============================================================================
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown(f"<div class='app-title'>📡 {APP_NAME}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-subtitle'>Wouxun KG-UV9D Plus | 2m/70cm Cross-Band | Drone-Borne RF Coverage Simulation</div>", 
                unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='version-badge'>v{APP_VERSION}</div>", unsafe_allow_html=True)
with col3:
    pass

st.markdown("<hr style='margin:5px 0;'>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS (ENHANCED)
# ============================================================================
st.sidebar.markdown(f"<div class='sidebar-title'>⚙️ System Configuration</div>", unsafe_allow_html=True)

# App Info in Sidebar
with st.sidebar.expander("ℹ️ App Info", expanded=False):
    st.markdown(f"""
    **{APP_NAME}** v{APP_VERSION}
    
    **Description:** {APP_DESCRIPTION}
    
    **Developer:** {DEVELOPER}
    
    **Copyright:** {COPYRIGHT}
    
    **GitHub:** [Report Bug]({GITHUB_URL})
    
    **Session Started:** {st.session_state.app_start_time.strftime('%Y-%m-%d %H:%M:%S')}
    
    **Saved Configs:** {len(st.session_state.saved_configs)}
    """)

# Radio Configuration
with st.sidebar.expander("📻 Radio Parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        tx_power_2m = st.slider("2m TX (W)", 0.1, 5.0, 5.0, 0.05, key="tx2m", help="Max 5W per specs")
        swr_2m = st.slider("2m SWR", 1.0, 3.0, 1.0, 0.1, key="swr2m")
    with col2:
        tx_power_70cm = st.slider("70cm TX (W)", 0.1, 4.0, 4.0, 0.05, key="tx70cm", help="Max 4W per specs")
        swr_70cm = st.slider("70cm SWR", 1.0, 3.0, 1.3, 0.1, key="swr70cm")

# Antenna Configuration
with st.sidebar.expander("📶 Antenna System", expanded=True):
    antenna_gain = st.slider("Gain (dBi)", -3, 9, 0, 1, key="ant_gain")
    antenna_polarization = st.selectbox("Polarization", ["Vertical", "Horizontal", "Circular"], key="pol")
    antenna_pattern = st.selectbox("Pattern", ["Omnidirectional", "Directional"], key="pattern")
    
    if antenna_polarization == "Vertical":
        polarization_mismatch = 0
    elif antenna_polarization == "Horizontal":
        polarization_mismatch = 20
    else:
        polarization_mismatch = 3
    
    if antenna_pattern == "Omnidirectional":
        antenna_efficiency = 0.9
    else:
        antenna_efficiency = 0.85

# Receiver Performance
with st.sidebar.expander("📊 Receiver Specs", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        noise_figure = st.slider("NF (dB)", 3, 15, 8, 1, key="nf", 
                                help="Noise Figure: degrades SNR. Lower is better.")
        if_bandwidth_khz = st.slider("BW (kHz)", 6.0, 25.0, 12.5, 0.5, key="bw",
                                     help="IF Bandwidth: narrower improves sensitivity but may affect signal quality")
    with col2:
        desense_penalty = st.slider("Desense (dB)", 0, 25, 12, 1, key="desense",
                                   help="Desensitization from nearby transmitters. Use filters to reduce.")
        snr_required = st.slider("Req SNR (dB)", 6, 20, 12, 1, key="snr_req",
                                help="Signal-to-Noise Ratio required for reliable decoding")
    
    # Display calculated sensitivities with clear distinction
    st.markdown("---")
    st.markdown("**Calculated Sensitivities:**")
    
    # Calculate values
    noise_floor = -174 + 10 * np.log10(if_bandwidth_khz * 1000)
    theoretical_sens = noise_floor + noise_figure + snr_required
    effective_sens = max(theoretical_sens, -127) + desense_penalty
    
    col_sens1, col_sens2 = st.columns(2)
    with col_sens1:
        st.metric("Theoretical", f"{theoretical_sens:.1f} dBm",
                 help="Best-case sensitivity without desense or practical limits")
    with col_sens2:
        st.metric("Effective", f"{effective_sens:.1f} dBm",
                 help="Real-world sensitivity including desense penalty",
                 delta=f"{effective_sens - theoretical_sens:+.1f} dB")
    
    # Show breakdown
    with st.expander("🔍 Sensitivity Breakdown", expanded=False):
        st.text(f"Noise Floor:      {noise_floor:.1f} dBm")
        st.text(f"+ Noise Figure:   {noise_figure:+.1f} dB")
        st.text(f"+ Required SNR:   {snr_required:+.1f} dB")
        st.text(f"─────────────────────────────")
        st.text(f"= Theoretical:    {theoretical_sens:.1f} dBm")
        st.text(f"+ Desense:        {desense_penalty:+.1f} dB")
        st.text(f"─────────────────────────────")
        st.text(f"= Effective:      {effective_sens:.1f} dBm")
        st.text(f"")
        st.text(f"Formula: kTB + NF + SNR + Desense")
        st.text(f"where kTB = -174 + 10log₁₀(BW)")


# Drone Configuration
with st.sidebar.expander("🚁 Drone Platform", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        drone_altitude = st.slider("Alt (m AGL)", 10, 120, 100, 10, key="alt", help="FAA max 120m (400ft)")
        if drone_altitude > 120:
            st.warning("⚠️ Altitude exceeds FAA limit for drones")
        ground_rx_height = st.slider("RX Height (m)", 1, 10, 2, 1, key="rx_height")
    with col2:
        required_fade_margin = st.slider("Fade Mgn (dB)", 0, 25, 10, 1, key="fade")
        link_availability = st.slider("Availability (%)", 90.0, 99.99, 95.0, 0.01, key="avail")

# Environment & Propagation
with st.sidebar.expander("🌍 Environment & Model", expanded=True):
    propagation_model = st.selectbox(
        "Propagation Model",
        ["simple", "fspl", "two_ray", "okumura_hata", "itu_p1546"],
        format_func=lambda x: {
            "simple": "Simple (Path Loss Exp.)",
            "fspl": "Free Space (Ideal)",
            "two_ray": "Two-Ray Ground",
            "okumura_hata": "Okumura-Hata (Urban/Suburban)",
            "itu_p1546": "ITU-R P.1546 (VHF/UHF)"
        }[x],
        key="prop_model"
    )
    
    if propagation_model in ["simple", "itu_p1546"]:
        path_loss_exponent = st.slider("PL Exponent", 2.0, 4.5, 2.0, 0.1, key="pl_exp")
    else:
        path_loss_exponent = 2.0
    
    if propagation_model == "okumura_hata":
        environment = st.selectbox("Environment", ["suburban", "urban", "rural"], key="env")
    else:
        environment = "suburban"
    
    additional_loss = st.slider("Cable Loss (dB)", 0, 30, 0, 1, key="add_loss")
    atmospheric_loss = st.slider("Atmospheric (dB/km)", 0.0, 0.5, 0.1, 0.05, key="atm_loss")

# Advanced Settings
with st.sidebar.expander("🔬 Advanced Settings"):
    freq_2m = st.number_input("2m Freq (MHz)", 144.0, 148.0, 146.0, 0.1, key="freq_2m")
    freq_70cm = st.number_input("70cm Freq (MHz)", 420.0, 450.0, 446.0, 0.5, key="freq_70cm")
    multipath_fade = st.slider("Multipath Fade (dB)", 0, 15, 0, 1, key="multipath")

# ============================================================================
# CALCULATIONS WITH ALL FACTORS
# ============================================================================

nominal_sensitivity = calculate_sensitivity_from_nf(noise_figure, if_bandwidth_khz, snr_required)

effective_sensitivity = nominal_sensitivity + desense_penalty

total_additional_loss = (additional_loss + multipath_fade + 
                        polarization_mismatch + (10 * np.log10(1/antenna_efficiency) if antenna_efficiency < 1 else 0))

availability_margin_map = {
    90.0: 0,
    95.0: 3,
    99.0: 8,
    99.5: 12,
    99.9: 20,
    99.99: 30
}

closest_avail = min(availability_margin_map.keys(), key=lambda x: abs(x - link_availability))
additional_availability_margin = availability_margin_map[closest_avail]

total_fade_margin = required_fade_margin + additional_availability_margin

if propagation_model == 'itu_p1546':
    time_percent = 100 - link_availability
else:
    time_percent = 50

# Calculate ranges with FIXED function
range_2m = calculate_range(
    tx_power_2m, antenna_gain, antenna_gain, effective_sensitivity, 
    freq_2m, path_loss_exponent, total_additional_loss, swr_2m, 
    total_fade_margin, drone_altitude, propagation_model, 
    ground_rx_height, environment, time_percent,
    polarization_mismatch, antenna_efficiency
)

range_70cm = calculate_range(
    tx_power_70cm, antenna_gain, antenna_gain, effective_sensitivity,
    freq_70cm, path_loss_exponent, total_additional_loss, swr_70cm, 
    total_fade_margin, drone_altitude, propagation_model, 
    ground_rx_height, environment, time_percent,
    polarization_mismatch, antenna_efficiency
)

# Apply atmospheric loss correction (frequency-dependent)
# UHF suffers slightly more atmospheric absorption
atm_factor_2m = atmospheric_loss
atm_factor_70cm = atmospheric_loss * 1.15  # 15% more atmospheric loss at UHF

range_2m = range_2m / (1 + atm_factor_2m * range_2m / 100)
range_70cm = range_70cm / (1 + atm_factor_70cm * range_70cm / 100)

system_range = min(range_2m, range_70cm)

radio_horizon = calculate_radio_horizon(drone_altitude)

# Generate power vs range data
power_levels = np.arange(0.5, 5.1, 0.1)
ranges_2m = [min(calculate_range(p, antenna_gain, antenna_gain, effective_sensitivity,
                                 freq_2m, path_loss_exponent, total_additional_loss, swr_2m, 
                                 total_fade_margin, drone_altitude, propagation_model, 
                                 ground_rx_height, environment, time_percent,
                                 polarization_mismatch, antenna_efficiency), 100)
             for p in power_levels]

# ============================================================================
# MAIN CONTENT TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Coverage Map", 
    "📊 Link Budget", 
    "📈 Performance", 
    "🎯 Optimization",
    "📋 Summary",
    "📚 Help & About"
])

with tab1:
    col_map, col_info = st.columns([3, 1])
    
    with col_map:
        st.markdown("**Interactive Coverage Map** - Click to place drone")
        
        if st.session_state.drone_location is None:
            map_center = [5.6037, -0.1870]
        else:
            map_center = st.session_state.drone_location
        
        m = folium.Map(
            location=map_center, 
            zoom_start=12, 
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        if st.session_state.drone_location is not None:
            folium.Marker(
                st.session_state.drone_location,
                popup=f"""
                <b>{APP_NAME}</b><br>
                Altitude: {drone_altitude}m AGL<br>
                2m Range: {range_2m:.1f} km<br>
                70cm Range: {range_70cm:.1f} km<br>
                System: {system_range:.1f} km<br>
                Horizon: {radio_horizon:.1f} km<br>
                Availability: {link_availability:.1f}%
                """,
                tooltip="Drone Repeater Station",
                icon=folium.Icon(color='red', icon='broadcast-tower', prefix='fa')
            ).add_to(m)
            
            folium.Circle(
                st.session_state.drone_location,
                radius=range_2m * 1000,
                popup=f"2m Band: {range_2m:.1f} km",
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.15,
                weight=2,
                dashArray='5, 5'
            ).add_to(m)
            
            folium.Circle(
                st.session_state.drone_location,
                radius=range_70cm * 1000,
                popup=f"70cm Band: {range_70cm:.1f} km",
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.1,
                weight=2,
                dashArray='10, 5'
            ).add_to(m)
            
            folium.Circle(
                st.session_state.drone_location,
                radius=system_range * 1000,
                popup=f"System Range: {system_range:.1f} km",
                color='purple',
                fill=True,
                fillColor='purple',
                fillOpacity=0.2,
                weight=3
            ).add_to(m)
            
            folium.Circle(
                st.session_state.drone_location,
                radius=radio_horizon * 1000,
                popup=f"Radio Horizon: {radio_horizon:.1f} km",
                color='orange',
                fill=False,
                weight=1,
                dashArray='2, 5'
            ).add_to(m)
        
        m.add_child(folium.LatLngPopup())
        
        map_key = f"map_{drone_altitude}_{tx_power_2m:.2f}_{propagation_model}_{required_fade_margin}_{link_availability}"
        map_data = st_folium(m, width=800, height=600, key=map_key)
        
        if map_data and map_data.get("last_clicked"):
            st.session_state.drone_location = [
                map_data["last_clicked"]["lat"],
                map_data["last_clicked"]["lng"]
            ]
            st.rerun()
    
    with col_info:
        if st.session_state.drone_location:
            st.markdown("**📍 Position**")
            st.text(f"Lat: {st.session_state.drone_location[0]:.4f}°")
            st.text(f"Lon: {st.session_state.drone_location[1]:.4f}°")
            st.text(f"Alt: {drone_altitude} m AGL")
            
            st.markdown("**📡 Coverage**")
            st.metric("System", f"{system_range:.1f} km")
            st.metric("2m Band", f"{range_2m:.1f} km")
            st.metric("70cm Band", f"{range_70cm:.1f} km")
            st.metric("Horizon", f"{radio_horizon:.1f} km")
            st.metric("Availability", f"{link_availability:.1f}%", 
                     f"+{additional_availability_margin:.0f}dB margin")
            
            fresnel_2m = calculate_fresnel_zone(system_range, freq_2m, 60)
            fresnel_70cm = calculate_fresnel_zone(system_range, freq_70cm, 60)
            
            st.markdown("**🎯 Fresnel Zone**")
            st.text(f"2m (60%): {fresnel_2m:.1f} m")
            st.text(f"70cm (60%): {fresnel_70cm:.1f} m")
            
            if system_range >= radio_horizon * 0.9:
                st.info("ℹ️ Horizon Limited")
            elif range_2m <= range_70cm * 0.8:
                st.warning("⚠️ 2m Band Limited")
            else:
                st.warning("⚠️ 70cm Band Limited")
            
            st.markdown("**🔍 Key Factors**")
            st.text(f"NF: {noise_figure} dB")
            st.text(f"BW: {if_bandwidth_khz} kHz")
            st.text(f"Desense: {desense_penalty} dB")
            st.text(f"Pol Loss: {polarization_mismatch} dB")
            
            if st.button("🗑️ Clear Location"):
                st.session_state.drone_location = None
                st.rerun()

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📻 2m Band Link Budget ({freq_2m} MHz)**")
        
        # Calculate both sensitivities for display
        noise_floor = -174 + 10 * np.log10(if_bandwidth_khz * 1000)
        theoretical_sensitivity = noise_floor + noise_figure + snr_required
        practical_min_sensitivity = -127  # Real-world limit
        
        # Calculate 2m band link budget variables
        tx_power_dbm_2m = 10 * np.log10(tx_power_2m * 1000)
        path_loss_2m = calculate_path_loss_db(
            range_2m, freq_2m, path_loss_exponent, drone_altitude,
            propagation_model, ground_rx_height, environment, time_percent
        )
        swr_loss_2m = 10 * np.log10(swr_2m) if swr_2m > 1.0 else 0
        efficiency_loss_2m = -10 * np.log10(antenna_efficiency)
        
        rx_power_2m = calculate_received_power(
            tx_power_2m, antenna_gain, antenna_gain, range_2m, freq_2m,
            path_loss_exponent, total_additional_loss, swr_2m, drone_altitude,
            propagation_model, ground_rx_height, environment, time_percent,
            polarization_mismatch, antenna_efficiency
        )
        fade_margin_2m = rx_power_2m - effective_sensitivity
        
        link_budget_data = {
            'Parameter': [
                'TX Power', 'TX Ant Gain', 'EIRP', 'Path Loss', 
                'SWR Loss', 'Cable Loss', 'Multipath', 'Pol Mismatch',
                'Ant Efficiency', 'RX Ant Gain', 'RX Power',
                '─────────', 
                'Noise Floor (kTB)', 'Noise Figure', 'Required SNR',
                'Theoretical Sens', 'Practical Min Sens', 'Desense Penalty',
                'Effective Sens', 
                '─────────',
                'Fade Margin', 'Avail Margin', 'Total Margin'
            ],
            'Value (dBm/dB)': [
                f"{tx_power_dbm_2m:.1f}",
                f"+{antenna_gain:.1f}",
                f"{tx_power_dbm_2m + antenna_gain:.1f}",
                f"-{path_loss_2m:.1f}",
                f"-{swr_loss_2m:.1f}",
                f"-{additional_loss:.1f}",
                f"-{multipath_fade:.1f}",
                f"-{polarization_mismatch:.1f}",
                f"-{efficiency_loss_2m:.1f}",
                f"+{antenna_gain:.1f}",
                f"{rx_power_2m:.1f}",
                "─────────",
                f"{noise_floor:.1f}",
                f"+{noise_figure:.1f}",
                f"+{snr_required:.1f}",
                f"= {theoretical_sensitivity:.1f}",
                f"max({theoretical_sensitivity:.1f}, {practical_min_sensitivity:.1f})",
                f"+{desense_penalty:.1f}",
                f"= {effective_sensitivity:.1f}",
                "─────────",
                f"{fade_margin_2m:.1f}",
                f"+{additional_availability_margin:.1f}",
                f"{fade_margin_2m - additional_availability_margin:.1f}"
            ]
        }
        
        df_2m = pd.DataFrame(link_budget_data)
        st.dataframe(df_2m, hide_index=True, width="stretch", height=600)
        
        if fade_margin_2m < 6:
            st.error(f"❌ Critical: {fade_margin_2m:.1f} dB margin")
        elif fade_margin_2m < total_fade_margin:
            st.warning(f"⚠️ Below target: {fade_margin_2m:.1f} dB (need {total_fade_margin} dB)")
        else:
            st.success(f"✅ Good: {fade_margin_2m:.1f} dB margin")
    
    with col2:
        st.markdown(f"**📻 70cm Band Link Budget ({freq_70cm} MHz)**")
        
        tx_power_dbm_70cm = 10 * np.log10(tx_power_70cm * 1000)
        path_loss_70cm = calculate_path_loss_db(
            range_70cm, freq_70cm, path_loss_exponent, drone_altitude,
            propagation_model, ground_rx_height, environment, time_percent
        )
        swr_loss_70cm = 10 * np.log10(swr_70cm) if swr_70cm > 1.0 else 0
        efficiency_loss_70cm = -10 * np.log10(antenna_efficiency)
        
        rx_power_70cm = calculate_received_power(
            tx_power_70cm, antenna_gain, antenna_gain, range_70cm, freq_70cm,
            path_loss_exponent, total_additional_loss, swr_70cm, drone_altitude,
            propagation_model, ground_rx_height, environment, time_percent,
            polarization_mismatch, antenna_efficiency
        )
        fade_margin_70cm = rx_power_70cm - effective_sensitivity
        
        link_budget_data_70cm = {
            'Parameter': [
                'TX Power', 'TX Ant Gain', 'EIRP', 'Path Loss',
                'SWR Loss', 'Cable Loss', 'Multipath', 'Pol Mismatch',
                'Ant Efficiency', 'RX Ant Gain', 'RX Power',
                'RX Sensitivity', 'Fade Margin', 'Avail Margin', 'Total Margin'
            ],
            'Value (dBm/dB)': [
                f"{tx_power_dbm_70cm:.1f}",
                f"+{antenna_gain:.1f}",
                f"{tx_power_dbm_70cm + antenna_gain:.1f}",
                f"-{path_loss_70cm:.1f}",
                f"-{swr_loss_70cm:.1f}",
                f"-{additional_loss:.1f}",
                f"-{multipath_fade:.1f}",
                f"-{polarization_mismatch:.1f}",
                f"-{efficiency_loss_70cm:.1f}",
                f"+{antenna_gain:.1f}",
                f"{rx_power_70cm:.1f}",
                f"{effective_sensitivity:.1f}",
                f"{fade_margin_70cm:.1f}",
                f"+{additional_availability_margin:.1f}",
                f"{fade_margin_70cm - additional_availability_margin:.1f}"
            ]
        }
        
        df_70cm = pd.DataFrame(link_budget_data_70cm)
        st.dataframe(df_70cm, hide_index=True, width="stretch", height=500)
        
        if fade_margin_70cm > fade_margin_2m + 5:
            st.info("ℹ️ 70cm has better margin")
        elif fade_margin_70cm < fade_margin_2m - 5:
            st.warning("⚠️ 70cm has worse margin than 2m")
        else:
            st.success(f"✅ Balanced: {fade_margin_70cm:.1f} dB margin")
    
    st.markdown("**📉 Received Power vs Distance**")
    
    distances = np.linspace(0.1, min(max(range_2m, range_70cm) * 1.5, 100), 200)
    
    rx_powers_2m = [
        calculate_received_power(
            tx_power_2m, antenna_gain, antenna_gain, d, freq_2m,
            path_loss_exponent, total_additional_loss, swr_2m, drone_altitude,
            propagation_model, ground_rx_height, environment, time_percent,
            polarization_mismatch, antenna_efficiency
        ) for d in distances
    ]
    
    rx_powers_70cm = [
        calculate_received_power(
            tx_power_70cm, antenna_gain, antenna_gain, d, freq_70cm,
            path_loss_exponent, total_additional_loss, swr_70cm, drone_altitude,
            propagation_model, ground_rx_height, environment, time_percent,
            polarization_mismatch, antenna_efficiency
        ) for d in distances
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=distances, y=rx_powers_2m, mode='lines',
        name=f'2m ({freq_2m} MHz)', line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=distances, y=rx_powers_70cm, mode='lines',
        name=f'70cm ({freq_70cm} MHz)', line=dict(color='green', width=2)
    ))
    
    fig.add_hline(
        y=effective_sensitivity, line_dash="dash", 
        line_color="red", annotation_text="Effective Sensitivity",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=effective_sensitivity + total_fade_margin,
        line_dash="dot", line_color="orange",
        annotation_text=f"Target ({link_availability:.1f}% Avail)",
        annotation_position="right"
    )
    
    fig.add_vline(
        x=range_2m, line_dash="dash", line_color="blue",
        annotation_text=f"2m: {range_2m:.1f}km"
    )
    
    fig.add_vline(
        x=range_70cm, line_dash="dash", line_color="green",
        annotation_text=f"70cm: {range_70cm:.1f}km"
    )
    
    fig.update_layout(
        xaxis_title="Distance (km)",
        yaxis_title="RX Power (dBm)",
        hovermode='x unified',
        height=350,
        margin=dict(t=20, b=40, l=40, r=20),
        title=f"Propagation Model: {propagation_model.upper()} | Environment: {environment}"
    )
    st.plotly_chart(fig, width="stretch")

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Power vs Range**")
        
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=power_levels, y=ranges_2m, mode='lines+markers',
                                      name='2m', line=dict(color='blue', width=2),
                                      marker=dict(size=4)))
        fig_power.add_trace(go.Scatter(x=[tx_power_2m], y=[range_2m], mode='markers',
                                      name='Current', marker=dict(size=12, color='red', symbol='star')))
        
        if len(ranges_2m) > 0:
            derivatives = np.diff(ranges_2m) / np.diff(power_levels)
            threshold = 0.5 * max(derivatives)
            optimal_idx = np.where(derivatives > threshold)[0]
            if len(optimal_idx) > 0:
                optimal_power = power_levels[optimal_idx[-1]]
                fig_power.add_vrect(x0=optimal_power-0.2, x1=optimal_power+0.2, fillcolor="yellow", opacity=0.2,
                                  annotation_text=f"Optimal ~{optimal_power:.1f}W", annotation_position="top left")
        
        fig_power.update_layout(xaxis_title="TX Power (W)", yaxis_title="Range (km)",
                               hovermode='x unified', height=350, margin=dict(t=20, b=40, l=40, r=20),
                               title=f"Availability: {link_availability:.1f}%")
        st.plotly_chart(fig_power, width="stretch")
        
        power_efficiency = range_2m / tx_power_2m if tx_power_2m > 0 else 0
        st.metric("Power Efficiency", f"{power_efficiency:.1f} km/W")
    
    with col2:
        st.markdown("**Altitude Impact**")
        
        altitudes = np.arange(10, 121, 10)
        horizon_ranges = 4.12 * np.sqrt(altitudes)
        
        rf_ranges = []
        for alt in altitudes:
            r = calculate_range(
                tx_power_2m, antenna_gain, antenna_gain, effective_sensitivity,
                freq_2m, path_loss_exponent, total_additional_loss, swr_2m,
                total_fade_margin, alt, propagation_model,
                ground_rx_height, environment, time_percent,
                polarization_mismatch, antenna_efficiency
            )
            rf_ranges.append(min(r, 100))
        
        fig_alt = go.Figure()
        fig_alt.add_trace(go.Scatter(x=altitudes, y=horizon_ranges, mode='lines+markers',
                                    name='Radio Horizon', line=dict(color='orange', width=2, dash='dash'),
                                    marker=dict(size=4)))
        fig_alt.add_trace(go.Scatter(x=altitudes, y=rf_ranges, mode='lines+markers',
                                    name='RF Range', line=dict(color='purple', width=2),
                                    marker=dict(size=4)))
        
        current_horizon = 4.12 * np.sqrt(drone_altitude)
        fig_alt.add_trace(go.Scatter(x=[drone_altitude], y=[current_horizon], mode='markers',
                                    name='Current Horizon', marker=dict(size=12, color='red', symbol='star')))
        fig_alt.add_trace(go.Scatter(x=[drone_altitude], y=[range_2m], mode='markers',
                                    name='Current RF Range', marker=dict(size=12, color='blue', symbol='star')))
        
        fig_alt.update_layout(xaxis_title="Altitude (m)", yaxis_title="Range (km)",
                             hovermode='x unified', height=350, margin=dict(t=20, b=40, l=40, r=20))
        st.plotly_chart(fig_alt, width="stretch")
        
        altitude_efficiency = range_2m / drone_altitude if drone_altitude > 0 else 0
        st.metric("Altitude Efficiency", f"{altitude_efficiency:.2f} km/m")
    
    st.markdown("**🎯 Propagation Model Comparison**")
    
    models = ["fspl", "two_ray", "okumura_hata", "itu_p1546"]
    model_names = ["Free Space", "Two-Ray", "Okumura-Hata", "ITU-P.1546"]
    model_ranges = []
    
    for model in models:
        if model == "okumura_hata":
            env = environment
        else:
            env = "suburban"
        
        r = calculate_range(
            tx_power_2m, antenna_gain, antenna_gain, effective_sensitivity,
            freq_2m, 2.0, total_additional_loss, swr_2m,
            required_fade_margin, drone_altitude, model,
            ground_rx_height, env, 50,
            polarization_mismatch, antenna_efficiency
        )
        model_ranges.append(min(r, 100))
    
    fig_models = go.Figure()
    fig_models.add_trace(go.Bar(x=model_names, y=model_ranges,
                               marker_color=['blue', 'green', 'orange', 'red']))
    
    current_model_idx = models.index(propagation_model) if propagation_model in models else 0
    fig_models.add_hline(y=range_2m, line_dash="dash", line_color="purple",
                        annotation_text=f"Current: {range_2m:.1f} km")
    
    fig_models.update_layout(xaxis_title="Propagation Model", yaxis_title="Range (km)",
                           height=300, margin=dict(t=20, b=40, l=40, r=20),
                           title="Comparison of Different Propagation Models")
    st.plotly_chart(fig_models, width="stretch")

with tab4:
    st.markdown("**Optimization Recommendations**")
    
    recommendations = []
    scores = []
    
    vhf_uhf_ratio = range_2m / range_70cm if range_70cm > 0 else 10
    if vhf_uhf_ratio < 0.7:
        recommendations.append("• **Physics Alert**: VHF range should typically exceed UHF range. Current VHF/UHF ratio is unrealistic. Check model parameters.")
        scores.append(1)
    elif vhf_uhf_ratio < 0.9:
        recommendations.append("• **Range Ratio**: VHF range is lower than expected compared to UHF. Consider VHF propagation advantages.")
        scores.append(2)
    else:
        recommendations.append(f"• **Range Ratio**: Good (VHF: {range_2m:.1f}km, UHF: {range_70cm:.1f}km)")
        scores.append(3)
    
    if noise_figure > 10:
        recommendations.append(f"• **Noise Figure**: Current NF={noise_figure}dB is high. Lowering to 6dB could improve sensitivity by {noise_figure-6}dB")
        scores.append(1)
    elif noise_figure > 6:
        recommendations.append(f"• **Noise Figure**: Current NF={noise_figure}dB is moderate. Consider better LNA for improved sensitivity")
        scores.append(2)
    else:
        recommendations.append("• **Noise Figure**: Good")
        scores.append(3)
    
    if if_bandwidth_khz > 15:
        recommendations.append(f"• **Bandwidth**: {if_bandwidth_khz}kHz is wide. Narrow to 12.5kHz for better sensitivity")
        scores.append(1)
    elif if_bandwidth_khz < 8:
        recommendations.append(f"• **Bandwidth**: {if_bandwidth_khz}kHz is narrow. Consider impact on signal quality")
        scores.append(2)
    else:
        recommendations.append("• **Bandwidth**: Optimal")
        scores.append(3)
    
    if desense_penalty > 15:
        recommendations.append(f"• **Desense**: {desense_penalty}dB penalty is high. Add filters to reduce")
        scores.append(1)
    elif desense_penalty > 8:
        recommendations.append(f"• **Desense**: {desense_penalty}dB penalty is moderate")
        scores.append(2)
    else:
        recommendations.append("• **Desense**: Good")
        scores.append(3)
    
    if antenna_gain < 0:
        recommendations.append("• **Antenna**: Negative gain. Replace with at least 0dBi antenna")
        scores.append(1)
    elif antenna_gain < 3:
        recommendations.append(f"• **Antenna**: {antenna_gain}dBi is low. Consider 3-6dBi antenna")
        scores.append(2)
    else:
        recommendations.append(f"• **Antenna**: {antenna_gain}dBi is good")
        scores.append(3)
    
    if swr_2m > 2.0 or swr_70cm > 2.0:
        recommendations.append("• **SWR**: High (>2.0). Tune antennas for better efficiency")
        scores.append(1)
    elif swr_2m > 1.5 or swr_70cm > 1.5:
        recommendations.append("• **SWR**: Moderate (1.5-2.0). Could be improved")
        scores.append(2)
    else:
        recommendations.append("• **SWR**: Good (<1.5)")
        scores.append(3)
    
    if tx_power_2m < 1.0:
        recommendations.append(f"• **Power**: {tx_power_2m}W is low. Increase to 1.5-2.0W for optimal efficiency")
        scores.append(1)
    elif tx_power_2m > 3.0:
        recommendations.append(f"• **Power**: {tx_power_2m}W is high. Consider battery life vs range trade-off")
        scores.append(2)
    else:
        recommendations.append(f"• **Power**: {tx_power_2m}W is in optimal range")
        scores.append(3)
    
    if drone_altitude < 50:
        recommendations.append(f"• **Altitude**: {drone_altitude}m is low. Increase to 80-100m for better coverage")
        scores.append(1)
    elif drone_altitude < 80:
        recommendations.append(f"• **Altitude**: {drone_altitude}m is moderate")
        scores.append(2)
    else:
        recommendations.append(f"• **Altitude**: {drone_altitude}m is good")
        scores.append(3)
    
    if propagation_model == 'okumura_hata' and freq_2m < 150:
        recommendations.append("• **Propagation Model**: Okumura-Hata not designed for 2m band (<150 MHz). Using VHF-optimized corrections.")
        scores.append(2)
    
    if link_availability > 99.0 and fade_margin_2m < 15:
        recommendations.append(f"• **Availability**: {link_availability:.1f}% requires high reliability. Ensure fade margin >15dB")
        scores.append(1)
    elif link_availability > 95.0 and fade_margin_2m < 10:
        recommendations.append(f"• **Availability**: {link_availability:.1f}% requires moderate reliability")
        scores.append(2)
    else:
        recommendations.append(f"• **Availability**: {link_availability:.1f}% requirement is well supported")
        scores.append(3)
    
    for rec in recommendations:
        if "Good" in rec or "Optimal" in rec:
            st.success(rec)
        elif "moderate" in rec.lower() or "consider" in rec.lower():
            st.info(rec)
        else:
            st.warning(rec)
    
    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score >= 2.7:
            st.success(f"✅ **Overall Score: {avg_score:.1f}/3.0** - Well optimized!")
        elif avg_score >= 2.0:
            st.info(f"⚠️ **Overall Score: {avg_score:.1f}/3.0** - Room for improvement")
        else:
            st.error(f"❌ **Overall Score: {avg_score:.1f}/3.0** - Significant improvements needed")

with tab5:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Configuration Summary**")
        
        config_data = {
            'Category': ['Radio', 'Radio', 'Antenna', 'Antenna', 'Receiver', 'Receiver', 'Receiver', 'Drone', 'Drone', 'Environment', 'Propagation'],
            'Parameter': ['2m TX Power', '70cm TX Power', 'Gain', 'Polarization', 'Noise Figure', 'Bandwidth', 'Desense', 'Altitude', 'RX Height', 'Atmos Loss', 'Model'],
            'Value': [f"{tx_power_2m:.2f}W", f"{tx_power_70cm:.2f}W", f"{antenna_gain:+.0f}dBi", 
                     antenna_polarization, f"{noise_figure:.0f}dB", f"{if_bandwidth_khz:.1f}kHz", 
                     f"{desense_penalty:.0f}dB", f"{drone_altitude}m", f"{ground_rx_height}m",
                     f"{atmospheric_loss:.2f}dB/km", propagation_model.upper()]
        }
        
        df_config = pd.DataFrame(config_data)
        st.dataframe(df_config, hide_index=True, width="stretch", height=400)
    
    with col2a:
        st.metric("System Range", f"{system_range:.1f} km", 
                 f"{'2m' if range_2m < range_70cm else '70cm'} limited")
        st.metric("2m Range", f"{range_2m:.1f} km")
        st.metric("70cm Range", f"{range_70cm:.1f} km")
        st.metric("Radio Horizon", f"{radio_horizon:.1f} km")

    with col2b:
        st.metric("Fade Margin", f"{fade_margin_2m:.1f} dB", 
                 f"{'✓' if fade_margin_2m >= total_fade_margin else '✗'} target")
        st.metric("Availability", f"{link_availability:.1f}%", 
                 f"+{additional_availability_margin:.0f}dB")
        
        # Calculate both sensitivities
        noise_floor = -174 + 10 * np.log10(if_bandwidth_khz * 1000)
        theoretical_sensitivity = noise_floor + noise_figure + snr_required
        
        st.metric("Sensitivity (Theoretical)", f"{theoretical_sensitivity:.0f} dBm",
                 help="Best-case without desense")
        st.metric("Sensitivity (Effective)", f"{effective_sensitivity:.0f} dBm",
                 delta=f"+{desense_penalty:.0f} dB desense",
                 help="Real-world with desense penalty")
        
        efficiency = (system_range / radio_horizon) * 100
        st.metric("Horizon Util.", f"{min(efficiency, 100):.0f}%")
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**📊 Performance Analysis**")
        
        power_eff = range_2m / tx_power_2m if tx_power_2m > 0 else 0
        altitude_eff = range_2m / drone_altitude if drone_altitude > 0 else 0
        antenna_eff = antenna_gain * 10
        
        metrics_df = pd.DataFrame({
            'Metric': ['Power Efficiency', 'Altitude Efficiency', 'Antenna Score', 
                      'Receiver Quality', 'System Balance'],
            'Value': [f"{power_eff:.2f} km/W", f"{altitude_eff:.2f} km/m", 
                     f"{antenna_eff:.0f}/90", 
                     f"{'Good' if noise_figure < 8 else 'Fair' if noise_figure < 12 else 'Poor'}",
                     f"{'Balanced' if abs(range_2m - range_70cm) < 5 else 'Imbalanced'}"],
            'Score': [f"{min(power_eff * 20, 100):.0f}/100", 
                     f"{min(altitude_eff * 100, 100):.0f}/100",
                     f"{antenna_eff:.0f}/90",
                     f"{max(0, 100 - (noise_figure-6)*10):.0f}/100",
                     f"{max(0, 100 - abs(range_2m - range_70cm)*5):.0f}/100"]
        })
        
        st.dataframe(metrics_df, hide_index=True, width="stretch", height=200)
    
    with col4:
        st.markdown("**🎯 Quick Actions**")
        
        if st.button("📈 Maximize Range", width="stretch"):
            st.info("To maximize range: 1) Increase altitude, 2) Use directional antenna, 3) Reduce NF, 4) Add filtering")
        
        if st.button("⚡ Optimize Efficiency", width="stretch"):
            st.info("For efficiency: 1) Set power to 1.5-2.0W, 2) Tune antennas (SWR<1.5), 3) Use optimal bandwidth")
        
        if st.button("🛡️ Improve Reliability", width="stretch"):
            st.info("For reliability: 1) Increase fade margin, 2) Add diversity, 3) Use cavity filters, 4) Lower NF")
        
        if st.button("💰 Cost-Effective", width="stretch"):
            st.info("Cost-effective upgrades: 1) Better antenna, 2) LNA, 3) Filters, 4) Tune existing equipment")

with tab6:
    st.markdown(f"**Help & About {APP_NAME}**")
    
    col_about, col_features = st.columns(2)
    
    with col_about:
        with st.expander("📖 About This App", expanded=True):
            st.markdown(f"""
            ### {APP_NAME} v{APP_VERSION}
            
            **{APP_DESCRIPTION}**
            
            A comprehensive RF planning tool for drone-based cross-band repeater systems.
            
            **Key Features:**
            - Multi-band RF coverage analysis (2m/70cm)
            - Physics-based propagation models
            - Real-time interactive mapping
            - Link budget calculations
            - Optimization recommendations
            
            **Developer:** {DEVELOPER}
            
            **Copyright:** {COPYRIGHT}
            
            **Source Code:** [GitHub]({GITHUB_URL})
            
            **Session Started:** {st.session_state.app_start_time.strftime('%Y-%m-%d %H:%M:%S')}
            """)
    
    with col_features:
        with st.expander("🚀 Key Features", expanded=True):
            st.markdown("""
            ### Advanced Features
            
            **📡 RF Physics Engine:**
            - VHF/UHF physics-based corrections
            - Radio horizon awareness
            - Realistic range calculations
            
            **🗺️ Interactive Mapping:**
            - Click-to-place drone positioning
            - Multi-layer coverage visualization
            - KML export for Google Earth
            
            **📊 Comprehensive Analysis:**
            - Link budget calculations
            - Propagation model comparisons
            - Power vs range optimization
            
            **⚡ Optimization Tools:**
            - Automatic recommendations
            - Performance scoring
            - Quick action suggestions
            """)
    
    st.markdown("---")
    
    col_guide, col_tech = st.columns(2)
    
    with col_guide:
        with st.expander("📖 User Guide", expanded=False):
            st.markdown("""
            ### How to Use This Tool
            
            1. **Configure Parameters**: Use sidebar to set all system parameters
            2. **Set Location**: Click on map to place drone (optional)
            3. **Analyze Results**: Review different tabs for analysis
            4. **Optimize**: Use recommendations to improve system
            5. **Export**: Download reports and data for sharing
            
            ### Key Parameters Explained
            
            - **Noise Figure (NF)**: Lower is better. Affects receiver sensitivity
            - **Bandwidth (BW)**: Narrower = better sensitivity but may affect signal quality
            - **Desense**: Loss from nearby transmitters. Use filters to reduce
            - **SWR**: Should be <1.5 for good efficiency
            - **Availability**: Higher % = more reliable but shorter range
            """)
    
    with col_tech:
        with st.expander("⚙️ Technical Details", expanded=False):
            st.markdown("""
            ### Calculation Formulas
            
            **Receiver Sensitivity**:
            ```
            Noise Floor = -174 dBm/Hz + 10log10(BW)
            Theoretical Sensitivity = Noise Floor + NF + Required SNR
            Minimum Practical Sensitivity = -127 dBm (real-world limit)
            Effective Sensitivity = max(Theoretical, -127) + Desense
            ```
            
            **Radio Horizon**:
            ```
            d_horizon = 4.12 × √h
            Where: h = altitude in meters, d = horizon in km
            ```
            
            **Range Calculation Fix**:
            ```
            1. Calculate radio horizon for given altitude
            2. Check if received power at horizon >= required power
            3. If YES: return radio horizon (reached physical limit)
            4. If NO: binary search between 0.1 km and radio horizon
            ```
            
            **Expected VHF/UHF Range Ratios**:
            ```
            Typical VHF/UHF (2m/70cm) range ratio: 1.2-1.5:1
            Current ratio: {range_2m/range_70cm if range_70cm > 0 else 'N/A':.2f}:1

            Factors affecting ratio:
            - Free space path loss: ~9-10 dB difference
            - Ground reflections: VHF benefits more
            - Diffraction: VHF diffracts better around obstacles
            - Atmospheric absorption: UHF suffers slightly more

            If ratio exceeds 2:1, check:
            - Propagation model selection
            - Atmospheric loss settings
            - Environment type
            - Frequency-dependent antenna efficiency
            ```            
            
            ### Version History
            
            **v1.1.0** (Current):
            - Added RadioSport branding
            - Version tracking system
            - Enhanced UI/UX
            - App info panel
            
            **v1.0.0**:
            - Fixed range calculation bug
            - Radio horizon physical limit respect
            - VHF/UHF physics corrections
            
            ### License & Usage
            
            This tool is provided for educational and planning purposes.
            Always verify calculations with field testing.
            Commercial use requires permission.
            """)

# ============================================================================
# EXPORT OPTIONS
# ============================================================================
st.markdown("---")
st.markdown("### 📤 Export Results")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📄 Full Report", width="stretch"):
        report_text = f"""
# {APP_NAME} RF Coverage Analysis Report

## App Information
- **Application**: {APP_NAME} v{APP_VERSION}
- **Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Session Start**: {st.session_state.app_start_time.strftime('%Y-%m-%d %H:%M:%S')}

## Configuration Summary
- **Radio**: Wouxun KG-UV9D Plus
- **2m Band**: {freq_2m} MHz, {tx_power_2m:.1f}W
- **70cm Band**: {freq_70cm} MHz, {tx_power_70cm:.1f}W
- **Antenna**: {antenna_gain:+.0f}dBi {antenna_pattern}, {antenna_polarization}
- **Altitude**: {drone_altitude}m AGL
- **Environment**: {environment}
- **Propagation Model**: {propagation_model.upper()}
- **Availability**: {link_availability:.1f}%

## Performance Results
- **System Range**: {system_range:.1f} km
- **2m Range**: {range_2m:.1f} km
- **70cm Range**: {range_70cm:.1f} km
- **Radio Horizon**: {radio_horizon:.1f} km
- **Horizon Utilization**: {(range_2m/radio_horizon*100 if radio_horizon > 0 else 0):.0f}%
- **Fade Margin**: {fade_margin_2m:.1f} dB
- **Effective Sensitivity**: {effective_sensitivity:.0f} dBm

## Receiver Characteristics
- **Noise Figure**: {noise_figure} dB
- **Bandwidth**: {if_bandwidth_khz} kHz
- **Required SNR**: {snr_required} dB
- **Desense Penalty**: {desense_penalty} dB

## Sensitivity Calculation
- **Noise Floor (kTB)**: {-174 + 10*np.log10(if_bandwidth_khz*1000):.1f} dBm
- **Theoretical Sensitivity**: {noise_floor + noise_figure + snr_required:.1f} dBm (kTB + NF + SNR)
- **Practical Minimum**: -127 dBm (real-world limit)
- **Effective Sensitivity**: {effective_sensitivity:.0f} dBm (includes desense)
- **Sensitivity Formula**: max(kTB + NF + SNR, -127 dBm) + Desense

## VHF vs UHF Analysis
- **VHF/UHF Range Ratio**: {range_2m/range_70cm if range_70cm > 0 else 'N/A':.2f}
- **Expected Ratio**: 1.5-3.0 (VHF should have better range)
- **Status**: {'✓ Realistic' if range_2m >= range_70cm * 0.8 else '⚠️ Check parameters'}

## Link Budget (2m Band)
- TX Power: {10*np.log10(tx_power_2m*1000):.1f} dBm
- TX Antenna Gain: +{antenna_gain:.1f} dBi
- EIRP: {10*np.log10(tx_power_2m*1000) + antenna_gain:.1f} dBm
- Path Loss: {calculate_path_loss_db(system_range, freq_2m, path_loss_exponent, drone_altitude, propagation_model, ground_rx_height, environment, time_percent):.1f} dB
- Total Additional Losses: {total_additional_loss:.1f} dB
- RX Power: {calculate_received_power(tx_power_2m, antenna_gain, antenna_gain, system_range, freq_2m, path_loss_exponent, total_additional_loss, swr_2m, drone_altitude, propagation_model, ground_rx_height, environment, time_percent, polarization_mismatch, antenna_efficiency):.1f} dBm

## Link Budget (70cm Band)
- TX Power: {10*np.log10(tx_power_70cm*1000):.1f} dBm
- Path Loss: {calculate_path_loss_db(range_70cm, freq_70cm, path_loss_exponent, drone_altitude, propagation_model, ground_rx_height, environment, time_percent):.1f} dB
- Total Additional Losses: {total_additional_loss:.1f} dB

## Recommendations
"""
        
        vhf_uhf_ratio = range_2m / range_70cm if range_70cm > 0 else 10
        if vhf_uhf_ratio < 0.7:
            report_text += f"- **CRITICAL**: VHF range ({range_2m:.1f}km) is unrealistically low compared to UHF ({range_70cm:.1f}km). Verify propagation model and parameters.\n"
        elif vhf_uhf_ratio < 0.9:
            report_text += f"- **Warning**: VHF range should typically exceed UHF range. Current ratio: {vhf_uhf_ratio:.2f}\n"
        
        if propagation_model == 'okumura_hata' and freq_2m < 150:
            report_text += f"- **Note**: Okumura-Hata model not designed for 2m band. Using VHF-optimized corrections.\n"
        
        if noise_figure > 8:
            report_text += f"- Reduce noise figure from {noise_figure}dB to 6dB for better sensitivity\n"
        if desense_penalty > 10:
            report_text += f"- Add cavity filters to reduce desense penalty of {desense_penalty}dB\n"
        if antenna_gain < 3:
            report_text += f"- Upgrade to higher gain antenna (>3dBi)\n"
        if drone_altitude < 80:
            report_text += f"- Increase altitude to 80-100m for better coverage\n"
        if swr_2m > 1.5:
            report_text += f"- Tune antenna for lower SWR (current: {swr_2m:.1f})\n"
        
        report_text += f"""
## Physical Reality Check (v1.2.0)
- **Radio Horizon**: {radio_horizon:.1f} km (absolute limit for line-of-sight)
- **Horizon Utilization**: {(range_2m/radio_horizon*100 if radio_horizon > 0 else 0):.0f}%
- **Propagation Model**: {propagation_model.upper()} handles frequency effects naturally
- **VHF/UHF Behavior**: Ratio of {range_2m/range_70cm if range_70cm > 0 else 'N/A':.2f} (expected: 1.2-1.5)
- **Cross-Band Limitation**: System limited by weaker of 2 bands
- **Beyond Horizon**: Not considered (requires special propagation modes)

## Notes
- Report generated by: {APP_NAME} v{APP_VERSION}
- Developer: {DEVELOPER}
- Copyright: {COPYRIGHT}
- For planning purposes only - verify with field testing
- **Important**: With minimal settings and good link budget, range approaches radio horizon
- **Important**: VHF should have equal or better range than UHF in most scenarios
"""
        
        st.download_button(
            "📥 Download Markdown",
            data=report_text,
            file_name=f"{APP_NAME.replace(' ', '_')}_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="stretch"
        )

with col2:
    if st.button("📊 Data CSV", width="stretch"):
        csv_data = pd.DataFrame({
            'Parameter': ['System Range', '2m Range', '70cm Range', 'Radio Horizon',
                         'Horizon Utilization', '2m Fade Margin', '70cm Fade Margin', 
                         'Coverage Area', 'VHF/UHF Ratio', 'Propagation Model',
                         'TX Power 2m', 'TX Power 70cm', 'Altitude', 'Desense',
                         'Noise Figure', 'Bandwidth', 'Availability', 'Avail Margin',
                         'App Version', 'Generation Time'],
            'Value': [system_range, range_2m, range_70cm, radio_horizon,
                     (range_2m/radio_horizon*100 if radio_horizon > 0 else 0),
                     fade_margin_2m, fade_margin_70cm, np.pi * (system_range ** 2),
                     range_2m/range_70cm if range_70cm > 0 else 0,
                     propagation_model.upper(),
                     tx_power_2m, tx_power_70cm, drone_altitude, desense_penalty,
                     noise_figure, if_bandwidth_khz, link_availability, additional_availability_margin,
                     APP_VERSION, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Unit': ['km', 'km', 'km', 'km', '%', 'dB', 'dB', 'km²', 'ratio', 'model',
                    'W', 'W', 'm', 'dB', 'dB', 'kHz', '%', 'dB', 'version', 'timestamp']
        })
        
        st.download_button(
            "📥 Download CSV",
            data=csv_data.to_csv(index=False),
            file_name=f"{APP_NAME.replace(' ', '_')}_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch"
        )

with col3:
    if st.button("🗺️ KML Export", width="stretch"):
        if st.session_state.drone_location:
            kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{APP_NAME} Coverage</name>
    <description>RF Coverage Analysis - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</description>
    <Placemark>
      <name>{APP_NAME} Station</name>
      <description>
        Application: {APP_NAME} v{APP_VERSION}
        Altitude: {drone_altitude}m AGL
        2m Range: {range_2m:.1f} km
        70cm Range: {range_70cm:.1f} km
        System Range: {system_range:.1f} km
        Radio Horizon: {radio_horizon:.1f} km
        Availability: {link_availability:.1f}%
        Propagation: {propagation_model.upper()}
        VHF/UHF Ratio: {range_2m/range_70cm if range_70cm > 0 else 'N/A':.2f}
      </description>
      <Point>
        <coordinates>{st.session_state.drone_location[1]},{st.session_state.drone_location[0]},{drone_altitude}</coordinates>
      </Point>
    </Placemark>
    <Placemark>
      <name>System Coverage</name>
      <Style>
        <LineStyle>
          <color>ff0000ff</color>
          <width>2</width>
        </LineStyle>
        <PolyStyle>
          <color>330000ff</color>
        </PolyStyle>
      </Style>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
"""
            center_lat = st.session_state.drone_location[0]
            center_lon = st.session_state.drone_location[1]
            
            for angle in range(0, 361, 5):
                lat_offset = (system_range / 111.0) * np.cos(np.radians(angle))
                lon_offset = (system_range / (111.0 * np.cos(np.radians(center_lat)))) * np.sin(np.radians(angle))
                
                point_lat = center_lat + lat_offset
                point_lon = center_lon + lon_offset
                
                kml_content += f"              {point_lon},{point_lat},0\n"
            
            kml_content += """            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""
            
            st.download_button(
                "📥 Download KML",
                data=kml_content,
                file_name=f"{APP_NAME.replace(' ', '_')}_coverage_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.kml",
                mime="application/vnd.google-earth.kml+xml",
                width="stretch"
            )
        else:
            st.warning("Set drone location on map first")

with col4:
    if st.button("📋 Copy Config", width="stretch"):
        config_text = f"""{APP_NAME} v{APP_VERSION}
2m: {tx_power_2m}W @ {freq_2m}MHz | 70cm: {tx_power_70cm}W @ {freq_70cm}MHz
Alt: {drone_altitude}m | Gain: {antenna_gain}dBi | NF: {noise_figure}dB
Horizon: {radio_horizon:.1f}km | 2m Range: {range_2m:.1f}km | 70cm Range: {range_70cm:.1f}km
Horizon Utilization: {(range_2m/radio_horizon*100 if radio_horizon > 0 else 0):.0f}%
VHF/UHF Ratio: {range_2m/range_70cm if range_70cm > 0 else 'N/A':.2f}
Availability: {link_availability:.1f}% | Model: {propagation_model}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        st.code(config_text, language="text")
        st.info("Select and copy the text above")

with col5:

# ============================================================================
# SAVE CONFIG BUTTON (MOVED HERE AFTER VARIABLES ARE DEFINED)
# ============================================================================

    if st.button("💾 Save Config", width="stretch"):
        config = {
            'tx_2m': tx_power_2m,
            'tx_70cm': tx_power_70cm,
            'antenna_gain': antenna_gain,
            'altitude': drone_altitude,
            'nf': noise_figure,
            'bw': if_bandwidth_khz,
            'desense': desense_penalty,
            'fade_margin': required_fade_margin,
            'availability': link_availability,
            'propagation_model': propagation_model,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.session_state.saved_configs.append(config)
        
        # Save to JSON file for persistence
        try:
            with open('saved_configs.json', 'w') as f:
                json.dump(st.session_state.saved_configs, f, indent=2)
            st.success("✅ Configuration saved to file!")
        except Exception as e:
            st.warning(f"⚠️ Saved to session only: {str(e)}")

st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div class='footer'>
    <p><strong>{APP_NAME} v{APP_VERSION}</strong></p>
    <p>Wouxun KG-UV9D Plus | Multi-Model Propagation | VHF/UHF Physics Corrected</p>
    <p>✅ Fixed range calculation | ✅ Radio horizon respected | ✅ Realistic range ratios</p>
    <p>{COPYRIGHT} | <a href="{GITHUB_URL}" target="_blank">GitHub</a> | Always verify with field measurements</p>
</div>
""", unsafe_allow_html=True)

# Session cleanup
if (pd.Timestamp.now() - st.session_state.last_cleanup).total_seconds() > 300:
    if len(st.session_state.saved_configs) > 20:
        st.session_state.saved_configs = st.session_state.saved_configs[-10:]
    st.session_state.last_cleanup = pd.Timestamp.now()
    
