_AC='timestamp'
_AB='70cm Range'
_AA='System Range'
_A9='Bandwidth'
_A8='Altitude Efficiency'
_A7='Power Efficiency'
_A6='Total Margin'
_A5='RX Ant Gain'
_A4='Ant Efficiency'
_A3='Pol Mismatch'
_A2='Multipath'
_A1='Cable Loss'
_A0='Path Loss'
_z='TX Ant Gain'
_y='TX Power'
_x='Value (dBm/dB)'
_w='Environment'
_v='desense'
_u='Omnidirectional'
_t='Horizontal'
_s='Vertical'
_r='Polarization'
_q='saved_configs.json'
_p='saved_configs'
_o='%Y%m%d_%H%M%S'
_n='Receiver'
_m='Value'
_l='Radio Horizon'
_k='Range (km)'
_j='star'
_i='markers'
_h='lines+markers'
_g='x unified'
_f='Avail Margin'
_e='Fade Margin'
_d='Noise Figure'
_c='Availability'
_b='last_clicked'
_a='simple'
_Z='Propagation Model'
_Y='urban'
_X='km'
_W='─────────'
_V='Parameter'
_U='orange'
_T='two_ray'
_S=None
_R='dB'
_Q='dash'
_P='purple'
_O='green'
_N='red'
_M='---'
_L='N/A'
_K='fspl'
_J='blue'
_I='%Y-%m-%d %H:%M:%S'
_H=False
_G='okumura_hata'
_F='itu_p1546'
_E='blended'
_D='suburban'
_C='stretch'
_B=True
_A=1.
import streamlit as st,folium
from streamlit_folium import st_folium
import numpy as np,pandas as pd,plotly.graph_objects as go
from plotly.subplots import make_subplots
import math,random,time,json
APP_VERSION='2.0.0'
APP_NAME='RadioSport X-Repeater'
APP_DESCRIPTION='Drone-Borne Repeater RF Coverage Analyzer - Industry Standard Models'
DEVELOPER='RNK'
COPYRIGHT='Copyright © RNK, 2026 RadioSport. All rights reserved.'
GITHUB_URL='https://github.com/rkarikari/stem'
def initialize_ui():st.set_page_config(page_title=f"{APP_NAME} - {APP_DESCRIPTION}",page_icon='📡',layout='wide',menu_items={'Report a Bug':GITHUB_URL,'About':f"{APP_NAME} v{APP_VERSION}\n\n{COPYRIGHT}"})
initialize_ui()
st.markdown(f'''
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
''',unsafe_allow_html=_B)
if'drone_location'not in st.session_state:st.session_state.drone_location=_S
if _p not in st.session_state:st.session_state.saved_configs=[]
if'show_advanced'not in st.session_state:st.session_state.show_advanced=_H
if'current_tip'not in st.session_state:st.session_state.current_tip=''
if'last_cleanup'not in st.session_state:st.session_state.last_cleanup=pd.Timestamp.now()
if'app_start_time'not in st.session_state:st.session_state.app_start_time=pd.Timestamp.now()
if _p not in st.session_state:
	try:
		with open(_q,'r')as f:st.session_state.saved_configs=json.load(f)
	except:st.session_state.saved_configs=[]
CACHE_VERSION=4
@st.cache_data(ttl=1)
def calculate_fspl_db(distance_km,freq_mhz,_cache_version=CACHE_VERSION):
	'Calculate Free Space Path Loss in dB - pure physics, no artificial corrections';A=distance_km
	if A<=.001:return 0
	B=A*1000;C=20*np.log10(B)+20*np.log10(freq_mhz)+32.45;return C
@st.cache_data(ttl=1)
def calculate_two_ray_model(distance_km,freq_mhz,h_tx_m,h_rx_m):
	'\n    Calculate path loss using Two-Ray Ground Reflection Model\n    Physics-based model for flat earth with ground reflection\n    ';E=h_rx_m;D=h_tx_m;C=freq_mhz;A=distance_km
	if A<=.001:return 0
	B=A*1000;F=3e8/(C*1e6);G=4*np.pi*D*E/F
	if B<=0:return 0
	elif B<G:return calculate_fspl_db(A,C)
	else:H=40*np.log10(B)-20*np.log10(D)-20*np.log10(E);return H
@st.cache_data(ttl=1)
def calculate_okumura_hata(distance_km,freq_mhz,h_tx_m,h_rx_m,environment=_D):
	'\n    Calculate path loss using Okumura-Hata model for VHF/UHF\n    Valid for: 100-1500 MHz, 1-20 km, h_tx: 30-200m, h_rx: 1-10m\n    v2.0.0: Extended to 100 MHz with validated corrections\n    ';F=environment;D=h_rx_m;C=h_tx_m;B=distance_km;A=freq_mhz
	if A<100 or A>1500:return calculate_fspl_db(B,A)
	if B<1:return calculate_fspl_db(B,A)
	if B>20:
		J=calculate_fspl_db(B,A)
		if C>=30:L=calculate_okumura_hata(2e1,A,C,D,F);M=35*np.log10(B/2e1);return min(L+M,J)
		else:return J
	if C<30:K=20*np.log10(30/max(C,10));C=30
	else:K=0
	N=min(max(B,1),20)
	if A<150:G=(1.1*np.log10(150)-.7)*D-(1.56*np.log10(150)-.8);H=-2.*(150-A)/50
	elif A<=300:G=(1.1*np.log10(A)-.7)*D-(1.56*np.log10(A)-.8);H=0
	else:G=3.2*np.log10(11.75*D)**2-4.97;H=0
	I=69.55+26.16*np.log10(A)-13.82*np.log10(C)-G+(44.9-6.55*np.log10(C))*np.log10(N)
	if F==_Y:E=I
	elif F==_D:E=I-2*np.log10(A/28)**2-5.4
	else:E=I-4.78*np.log10(A)**2+18.33*np.log10(A)-40.94
	E+=K+H;return E
@st.cache_data(ttl=1)
def calculate_itu_p1546(distance_km,freq_mhz,h_tx_m,time_percent=50,environment=_D):
	'\n    ITU-R P.1546 model with industry-validated corrections\n    v2.0.0: Aligned with empirical measurements\n    \n    Expected VHF/UHF ratios (146/446 MHz):\n    - <30m: 1.5-2.0:1  |  50-75m: 1.2-1.4:1  |  >100m: 1.05-1.15:1\n    ';H=time_percent;G=distance_km;A=freq_mhz
	if G<=.001:return 0
	K=min(max(G,1),1000);I=max(h_tx_m,10);L=69.55+26.16*np.log10(A)-13.82*np.log10(I)+(44.9-6.55*np.log10(I))*np.log10(K);J=min(I/1e2,_A);E=np.exp(-2.5*J)
	if A<=100:B=-7.;C=B*E
	elif A<=200:D=(A-100)/100;B=-7.+3.*D;C=B*E
	elif A<=400:D=(A-200)/200;B=-4.+2.*D;C=B*E
	elif A<=600:D=(A-400)/200;B=-1.5+.5*D;C=B*E
	else:C=0
	M={_Y:1.1,_D:_A,'rural':.9,'open':.8}.get(environment,_A);N=_A-.5*J;C*=M*N
	if H>=50:F=0
	elif H>=10:F=-3
	elif H>=1:F=-6
	else:F=-10
	O=L+C+F;P=calculate_fspl_db(G,A);return max(O,P)
@st.cache_data(ttl=1)
def calculate_blended_model(distance_km,freq_mhz,h_tx_m,h_rx_m=2.,environment=_D,time_percent=50):
	'\n    Blended model - eliminates double-counting\n    v2.0.0: Base models already include frequency effects\n    ';I=environment;D=freq_mhz;C=distance_km;A=h_tx_m
	if C<=.001:return 0
	if A<30:E=.7;F=.3;G=.0
	elif A<100:B=(A-30)/70;E=.5*(1-B);F=.3+.2*B;G=.2*B
	else:B=min((A-100)/100,_A);E=.2*(1-B*.5);F=.3*(1-B*.3);G=.5+B*.3
	L=calculate_okumura_hata(C,D,A,h_rx_m,I);M=calculate_itu_p1546(C,D,A,time_percent,I);J=calculate_fspl_db(C,D);H=E*L+F*M+G*J
	if A>150:K=min((A-150)/150,_A);H=H*(1-K)+J*K
	N=calculate_fspl_db(C,D);return max(H,N)
@st.cache_data
def calculate_vhf_advantage(distance_km,freq_mhz):'\n    [DEPRECATED in v1.2.0]\n    VHF advantage now handled by propagation models directly.\n    This function kept for API compatibility but returns 0.\n    ';return 0
@st.cache_data
def calculate_uhf_penalty(distance_km,freq_mhz,environment=_D):'\n    [DEPRECATED in v1.2.0]\n    UHF penalty now handled by propagation models directly.\n    This function kept for API compatibility but returns 0.\n    ';return 0
@st.cache_data
def calculate_path_loss_db(distance_km,freq_mhz,n=2.,altitude_m=0,propagation_model=_E,h_rx_m=2.,environment=_D,time_percent=50):
	'\n    Enhanced path loss calculation with industry-standard corrections\n    v2.0.0: Realistic VHF/UHF ratios at all altitudes\n    ';H=time_percent;G=environment;F=h_rx_m;E=propagation_model;D=altitude_m;B=freq_mhz;A=distance_km
	if A<=.001:return 0
	if B<=150:I=3.
	else:I=2.5
	J=D/1000*I
	if E==_K:C=calculate_fspl_db(A,B)
	elif E==_T:C=calculate_two_ray_model(A,B,D,F)
	elif E==_G:C=calculate_okumura_hata(A,B,D,F,G)
	elif E==_F:C=calculate_itu_p1546(A,B,D,H,G)
	elif E==_E:C=calculate_blended_model(A,B,D,F,G,H)
	elif n==2.:C=calculate_fspl_db(A,B)
	else:K=calculate_fspl_db(_A,B);C=K+10*n*np.log10(A)
	L=C-J;M=calculate_fspl_db(A,B);return max(L,M)
@st.cache_data
def calculate_fresnel_zone(distance_km,freq_mhz,zone_percent=60):
	'Calculate Fresnel zone radius at given percentage';A=distance_km
	if A<=0:return 0
	B=3e8/(freq_mhz*1e6);C=A*1000;return np.sqrt(B*C/4)*(zone_percent/100)
@st.cache_data
def calculate_received_power(tx_power_w,tx_gain_dbi,rx_gain_dbi,distance_km,freq_mhz,n=2.,additional_loss_db=0,swr=_A,altitude_m=0,propagation_model=_E,h_rx_m=2.,environment=_D,time_percent=50,polarization_mismatch=0,antenna_efficiency=.9):
	'Calculate received power with all losses considered';B=antenna_efficiency;A=distance_km
	if A<=.001:return 100
	C=10*np.log10(swr)if swr>_A else 0;D=10*np.log10(tx_power_w*1000);E=-10*np.log10(B)if B<_A else 0;F=calculate_path_loss_db(A,freq_mhz,n,altitude_m,propagation_model,h_rx_m,environment,time_percent);G=F+additional_loss_db+C+polarization_mismatch+E;H=D+tx_gain_dbi+rx_gain_dbi-G;return H
@st.cache_data
def calculate_sensitivity_from_nf(nf_db,bandwidth_khz,snr_required_db=12):'Calculate receiver sensitivity from noise figure and bandwidth';A=bandwidth_khz*1000;B=-174+10*np.log10(A);C=B+nf_db+snr_required_db;D=-127;return max(C,D)
@st.cache_data
def calculate_range(tx_power_w,tx_gain_dbi,rx_gain_dbi,rx_sensitivity_dbm,freq_mhz,n=2.,additional_loss_db=0,swr=_A,fade_margin_db=0,altitude_m=0,propagation_model=_E,h_rx_m=2.,environment=_D,time_percent=50,polarization_mismatch=0,antenna_efficiency=.9):
	'\n    Calculate maximum range with enhanced propagation models\n    v2.0.0: Industry-standard VHF/UHF ratios\n    ';P=antenna_efficiency;O=polarization_mismatch;N=time_percent;M=environment;L=h_rx_m;K=propagation_model;J=swr;I=additional_loss_db;H=freq_mhz;G=rx_gain_dbi;F=tx_gain_dbi;C=altitude_m;B=tx_power_w
	if B<=0:return 0
	A=4.12*np.sqrt(C)
	if A<_A:A=max(A,.5)
	Q=rx_sensitivity_dbm+fade_margin_db;T=calculate_received_power(B,F,G,A,H,n,I,J,C,K,L,M,N,O,P)
	if T>=Q:return A
	D,E=.01,A;U=.001;V=50;S=0
	if E>=_A:
		W=calculate_received_power(B,F,G,_A,H,n,I,J,C,K,L,M,N,O,P)
		if W<Q:E=_A
	while E-D>U and S<V:
		R=(D+E)/2;X=calculate_received_power(B,F,G,R,H,n,I,J,C,K,L,M,N,O,P)
		if X>=Q:D=R
		else:E=R
		S+=1
	Y=calculate_received_power(B,F,G,D,H,n,I,J,C,K,L,M,N,O,P)
	if Y<Q:return .01
	return D
@st.cache_data
def calculate_radio_horizon(altitude_m):'Calculate radio horizon distance in km';return 4.12*np.sqrt(altitude_m)
col1,col2,col3=st.columns([3,1,1])
with col1:st.markdown(f"<div class='app-title'>📡 {APP_NAME}</div>",unsafe_allow_html=_B);st.markdown(f"<div class='app-subtitle'>Wouxun KG-UV9D Plus | 2m/70cm Cross-Band | Drone-Borne RF Coverage Simulation</div>",unsafe_allow_html=_B)
with col2:st.markdown(f"<div class='version-badge'>v{APP_VERSION}</div>",unsafe_allow_html=_B)
with col3:0
st.markdown("<hr style='margin:5px 0;'>",unsafe_allow_html=_B)
st.sidebar.markdown(f"<div class='sidebar-title'>⚙️ System Configuration</div>",unsafe_allow_html=_B)
with st.sidebar.expander('ℹ️ App Info',expanded=_H):st.markdown(f"""
    **{APP_NAME}** v{APP_VERSION}
    
    **Description:** {APP_DESCRIPTION}
    
    **Developer:** {DEVELOPER}
    
    **Copyright:** {COPYRIGHT}
    
    **GitHub:** [Report Bug]({GITHUB_URL})
    
    **Session Started:** {st.session_state.app_start_time.strftime(_I)}
    
    **Saved Configs:** {len(st.session_state.saved_configs)}
    """)
with st.sidebar.expander('📻 Radio Parameters',expanded=_B):
	col1,col2=st.columns(2)
	with col1:tx_power_2m=st.slider('2m TX (W)',.1,5.,5.,.05,key='tx2m',help='Max 5W per specs');swr_2m=st.slider('2m SWR',_A,3.,_A,.1,key='swr2m')
	with col2:tx_power_70cm=st.slider('70cm TX (W)',.1,4.,4.,.05,key='tx70cm',help='Max 4W per specs');swr_70cm=st.slider('70cm SWR',_A,3.,1.3,.1,key='swr70cm')
with st.sidebar.expander('📶 Antenna System',expanded=_B):
	antenna_gain=st.slider('Gain (dBi)',-3,9,0,1,key='ant_gain');antenna_polarization=st.selectbox(_r,[_s,_t,'Circular'],key='pol');antenna_pattern=st.selectbox('Pattern',[_u,'Directional'],key='pattern')
	if antenna_polarization==_s:polarization_mismatch=0
	elif antenna_polarization==_t:polarization_mismatch=20
	else:polarization_mismatch=3
	if antenna_pattern==_u:antenna_efficiency=.9
	else:antenna_efficiency=.85
with st.sidebar.expander('📊 Receiver Specs',expanded=_B):
	col1,col2=st.columns(2)
	with col1:noise_figure=st.slider('NF (dB)',3,15,8,1,key='nf',help='Noise Figure: degrades SNR. Lower is better.');if_bandwidth_khz=st.slider('BW (kHz)',6.,25.,12.5,.5,key='bw',help='IF Bandwidth: narrower improves sensitivity but may affect signal quality')
	with col2:desense_penalty=st.slider('Desense (dB)',0,25,12,1,key=_v,help='Desensitization from nearby transmitters. Use filters to reduce.');snr_required=st.slider('Req SNR (dB)',6,20,12,1,key='snr_req',help='Signal-to-Noise Ratio required for reliable decoding')
	st.markdown(_M);st.markdown('**Calculated Sensitivities:**');noise_floor=-174+10*np.log10(if_bandwidth_khz*1000);theoretical_sens=noise_floor+noise_figure+snr_required;effective_sens=max(theoretical_sens,-127)+desense_penalty;col_sens1,col_sens2=st.columns(2)
	with col_sens1:st.metric('Theoretical',f"{theoretical_sens:.1f} dBm",help='Best-case sensitivity without desense or practical limits')
	with col_sens2:st.metric('Effective',f"{effective_sens:.1f} dBm",help='Real-world sensitivity including desense penalty',delta=f"{effective_sens-theoretical_sens:+.1f} dB")
	with st.expander('🔍 Sensitivity Breakdown',expanded=_H):st.text(f"Noise Floor:      {noise_floor:.1f} dBm");st.text(f"+ Noise Figure:   {noise_figure:+.1f} dB");st.text(f"+ Required SNR:   {snr_required:+.1f} dB");st.text(f"─────────────────────────────");st.text(f"= Theoretical:    {theoretical_sens:.1f} dBm");st.text(f"+ Desense:        {desense_penalty:+.1f} dB");st.text(f"─────────────────────────────");st.text(f"= Effective:      {effective_sens:.1f} dBm");st.text(f"");st.text(f"Formula: kTB + NF + SNR + Desense");st.text(f"where kTB = -174 + 10log₁₀(BW)")
with st.sidebar.expander('🚁 Drone Platform',expanded=_B):
	col1,col2=st.columns(2)
	with col1:
		drone_altitude=st.slider('Alt (m AGL)',10,120,100,10,key='alt',help='FAA max 120m (400ft)')
		if drone_altitude>120:st.warning('⚠️ Altitude exceeds FAA limit for drones')
		ground_rx_height=st.slider('RX Height (m)',1,10,2,1,key='rx_height')
	with col2:required_fade_margin=st.slider('Fade Mgn (dB)',0,25,10,1,key='fade');link_availability=st.slider('Availability (%)',9e1,99.99,95.,.01,key='avail')
with st.sidebar.expander('🌍 Environment & Model',expanded=_B):
	propagation_model=st.selectbox(_Z,[_a,_K,_T,_G,_F,_E],format_func=lambda x:{_a:'Simple (Path Loss Exp.)',_K:'Free Space (Ideal)',_T:'Two-Ray Ground',_G:'Okumura-Hata (Urban/Suburban)',_F:'ITU-R P.1546 (VHF/UHF)',_E:'Blended Model (Industry Standard)'}[x],key='prop_model')
	if propagation_model in[_a,_F,_E]:path_loss_exponent=st.slider('PL Exponent',2.,4.5,2.,.1,key='pl_exp')
	else:path_loss_exponent=2.
	if propagation_model in[_G,_F,_E]:environment=st.selectbox(_w,[_D,_Y,'rural','open'],key='env')
	else:environment=_D
	additional_loss=st.slider('Cable Loss (dB)',0,30,0,1,key='add_loss');atmospheric_loss=st.slider('Atmospheric (dB/km)',.0,.5,.1,.05,key='atm_loss')
with st.sidebar.expander('🔬 Advanced Settings'):freq_2m=st.number_input('2m Freq (MHz)',144.,148.,146.,.1,key='freq_2m');freq_70cm=st.number_input('70cm Freq (MHz)',42e1,45e1,446.,.5,key='freq_70cm');multipath_fade=st.slider('Multipath Fade (dB)',0,15,0,1,key='multipath')
nominal_sensitivity=calculate_sensitivity_from_nf(noise_figure,if_bandwidth_khz,snr_required)
effective_sensitivity=nominal_sensitivity+desense_penalty
total_additional_loss=additional_loss+multipath_fade+polarization_mismatch+(10*np.log10(1/antenna_efficiency)if antenna_efficiency<1 else 0)
availability_margin_map={9e1:0,95.:3,99.:8,99.5:12,99.9:20,99.99:30}
closest_avail=min(availability_margin_map.keys(),key=lambda x:abs(x-link_availability))
additional_availability_margin=availability_margin_map[closest_avail]
total_fade_margin=required_fade_margin+additional_availability_margin
if propagation_model==_F or propagation_model==_E:time_percent=100-link_availability
else:time_percent=50
range_2m=calculate_range(tx_power_2m,antenna_gain,antenna_gain,effective_sensitivity,freq_2m,path_loss_exponent,total_additional_loss,swr_2m,total_fade_margin,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency)
range_70cm=calculate_range(tx_power_70cm,antenna_gain,antenna_gain,effective_sensitivity,freq_70cm,path_loss_exponent,total_additional_loss,swr_70cm,total_fade_margin,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency)
atm_factor_2m=atmospheric_loss
atm_factor_70cm=atmospheric_loss*1.15
range_2m=range_2m/(1+atm_factor_2m*range_2m/100)
range_70cm=range_70cm/(1+atm_factor_70cm*range_70cm/100)
system_range=min(range_2m,range_70cm)
radio_horizon=calculate_radio_horizon(drone_altitude)
power_levels=np.arange(.5,5.1,.1)
ranges_2m=[min(calculate_range(A,antenna_gain,antenna_gain,effective_sensitivity,freq_2m,path_loss_exponent,total_additional_loss,swr_2m,total_fade_margin,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency),100)for A in power_levels]
tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['🗺️ Coverage Map','📊 Link Budget','📈 Performance','🎯 Optimization','📋 Summary','📚 Help & About'])
with tab1:
	col_map,col_info=st.columns([3,1])
	with col_map:
		st.markdown('**Interactive Coverage Map** - Click to place drone')
		if st.session_state.drone_location is _S:map_center=[5.6037,-.187]
		else:map_center=st.session_state.drone_location
		m=folium.Map(location=map_center,zoom_start=12,tiles='OpenStreetMap',control_scale=_B)
		if st.session_state.drone_location is not _S:folium.Marker(st.session_state.drone_location,popup=f"""
                <b>{APP_NAME}</b><br>
                Altitude: {drone_altitude}m AGL<br>
                2m Range: {range_2m:.1f} km<br>
                70cm Range: {range_70cm:.1f} km<br>
                System: {system_range:.1f} km<br>
                Horizon: {radio_horizon:.1f} km<br>
                Availability: {link_availability:.1f}%
                """,tooltip='Drone Repeater Station',icon=folium.Icon(color=_N,icon='broadcast-tower',prefix='fa')).add_to(m);folium.Circle(st.session_state.drone_location,radius=range_2m*1000,popup=f"2m Band: {range_2m:.1f} km",color=_J,fill=_B,fillColor=_J,fillOpacity=.15,weight=2,dashArray='5, 5').add_to(m);folium.Circle(st.session_state.drone_location,radius=range_70cm*1000,popup=f"70cm Band: {range_70cm:.1f} km",color=_O,fill=_B,fillColor=_O,fillOpacity=.1,weight=2,dashArray='10, 5').add_to(m);folium.Circle(st.session_state.drone_location,radius=system_range*1000,popup=f"System Range: {system_range:.1f} km",color=_P,fill=_B,fillColor=_P,fillOpacity=.2,weight=3).add_to(m);folium.Circle(st.session_state.drone_location,radius=radio_horizon*1000,popup=f"Radio Horizon: {radio_horizon:.1f} km",color=_U,fill=_H,weight=1,dashArray='2, 5').add_to(m)
		m.add_child(folium.LatLngPopup());map_key=f"map_{drone_altitude}_{tx_power_2m:.2f}_{propagation_model}_{required_fade_margin}_{link_availability}";map_data=st_folium(m,width=800,height=600,key=map_key)
		if map_data and map_data.get(_b):st.session_state.drone_location=[map_data[_b]['lat'],map_data[_b]['lng']];st.rerun()
	with col_info:
		if st.session_state.drone_location:
			st.markdown('**📍 Position**');st.text(f"Lat: {st.session_state.drone_location[0]:.4f}°");st.text(f"Lon: {st.session_state.drone_location[1]:.4f}°");st.text(f"Alt: {drone_altitude} m AGL");st.markdown('**📡 Coverage**');st.metric('System',f"{system_range:.1f} km");st.metric('2m Band',f"{range_2m:.1f} km");st.metric('70cm Band',f"{range_70cm:.1f} km");st.metric('Horizon',f"{radio_horizon:.1f} km");st.metric(_c,f"{link_availability:.1f}%",f"+{additional_availability_margin:.0f}dB margin");fresnel_2m=calculate_fresnel_zone(system_range,freq_2m,60);fresnel_70cm=calculate_fresnel_zone(system_range,freq_70cm,60);st.markdown('**🎯 Fresnel Zone**');st.text(f"2m (60%): {fresnel_2m:.1f} m");st.text(f"70cm (60%): {fresnel_70cm:.1f} m")
			if system_range>=radio_horizon*.9:st.info('ℹ️ Horizon Limited')
			elif range_2m<=range_70cm*.8:st.warning('⚠️ 2m Band Limited')
			else:st.warning('⚠️ 70cm Band Limited')
			st.markdown('**🔍 Key Factors**');st.text(f"NF: {noise_figure} dB");st.text(f"BW: {if_bandwidth_khz} kHz");st.text(f"Desense: {desense_penalty} dB");st.text(f"Pol Loss: {polarization_mismatch} dB")
			if st.button('🗑️ Clear Location'):st.session_state.drone_location=_S;st.rerun()
with tab2:
	col1,col2=st.columns(2)
	with col1:
		st.markdown(f"**📻 2m Band Link Budget ({freq_2m} MHz)**");noise_floor=-174+10*np.log10(if_bandwidth_khz*1000);theoretical_sensitivity=noise_floor+noise_figure+snr_required;practical_min_sensitivity=-127;tx_power_dbm_2m=10*np.log10(tx_power_2m*1000);path_loss_2m=calculate_path_loss_db(range_2m,freq_2m,path_loss_exponent,drone_altitude,propagation_model,ground_rx_height,environment,time_percent);swr_loss_2m=10*np.log10(swr_2m)if swr_2m>_A else 0;efficiency_loss_2m=-10*np.log10(antenna_efficiency);rx_power_2m=calculate_received_power(tx_power_2m,antenna_gain,antenna_gain,range_2m,freq_2m,path_loss_exponent,total_additional_loss,swr_2m,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency);fade_margin_2m=rx_power_2m-effective_sensitivity;link_budget_data={_V:[_y,_z,'EIRP',_A0,'SWR Loss',_A1,_A2,_A3,_A4,_A5,'RX Power',_W,'Noise Floor (kTB)',_d,'Required SNR','Theoretical Sens','Practical Min Sens','Desense Penalty','Effective Sens',_W,_e,_f,_A6],_x:[f"{tx_power_dbm_2m:.1f}",f"+{antenna_gain:.1f}",f"{tx_power_dbm_2m+antenna_gain:.1f}",f"-{path_loss_2m:.1f}",f"-{swr_loss_2m:.1f}",f"-{additional_loss:.1f}",f"-{multipath_fade:.1f}",f"-{polarization_mismatch:.1f}",f"-{efficiency_loss_2m:.1f}",f"+{antenna_gain:.1f}",f"{rx_power_2m:.1f}",_W,f"{noise_floor:.1f}",f"+{noise_figure:.1f}",f"+{snr_required:.1f}",f"= {theoretical_sensitivity:.1f}",f"max({theoretical_sensitivity:.1f}, {practical_min_sensitivity:.1f})",f"+{desense_penalty:.1f}",f"= {effective_sensitivity:.1f}",_W,f"{fade_margin_2m:.1f}",f"+{additional_availability_margin:.1f}",f"{fade_margin_2m-additional_availability_margin:.1f}"]};df_2m=pd.DataFrame(link_budget_data);st.dataframe(df_2m,hide_index=_B,width=_C,height=600)
		if fade_margin_2m<6:st.error(f"❌ Critical: {fade_margin_2m:.1f} dB margin")
		elif fade_margin_2m<total_fade_margin:st.warning(f"⚠️ Below target: {fade_margin_2m:.1f} dB (need {total_fade_margin} dB)")
		else:st.success(f"✅ Good: {fade_margin_2m:.1f} dB margin")
	with col2:
		st.markdown(f"**📻 70cm Band Link Budget ({freq_70cm} MHz)**");tx_power_dbm_70cm=10*np.log10(tx_power_70cm*1000);path_loss_70cm=calculate_path_loss_db(range_70cm,freq_70cm,path_loss_exponent,drone_altitude,propagation_model,ground_rx_height,environment,time_percent);swr_loss_70cm=10*np.log10(swr_70cm)if swr_70cm>_A else 0;efficiency_loss_70cm=-10*np.log10(antenna_efficiency);rx_power_70cm=calculate_received_power(tx_power_70cm,antenna_gain,antenna_gain,range_70cm,freq_70cm,path_loss_exponent,total_additional_loss,swr_70cm,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency);fade_margin_70cm=rx_power_70cm-effective_sensitivity;link_budget_data_70cm={_V:[_y,_z,'EIRP',_A0,'SWR Loss',_A1,_A2,_A3,_A4,_A5,'RX Power','RX Sensitivity',_e,_f,_A6],_x:[f"{tx_power_dbm_70cm:.1f}",f"+{antenna_gain:.1f}",f"{tx_power_dbm_70cm+antenna_gain:.1f}",f"-{path_loss_70cm:.1f}",f"-{swr_loss_70cm:.1f}",f"-{additional_loss:.1f}",f"-{multipath_fade:.1f}",f"-{polarization_mismatch:.1f}",f"-{efficiency_loss_70cm:.1f}",f"+{antenna_gain:.1f}",f"{rx_power_70cm:.1f}",f"{effective_sensitivity:.1f}",f"{fade_margin_70cm:.1f}",f"+{additional_availability_margin:.1f}",f"{fade_margin_70cm-additional_availability_margin:.1f}"]};df_70cm=pd.DataFrame(link_budget_data_70cm);st.dataframe(df_70cm,hide_index=_B,width=_C,height=500)
		if fade_margin_70cm>fade_margin_2m+5:st.info('ℹ️ 70cm has better margin')
		elif fade_margin_70cm<fade_margin_2m-5:st.warning('⚠️ 70cm has worse margin than 2m')
		else:st.success(f"✅ Balanced: {fade_margin_70cm:.1f} dB margin")
	st.markdown('**📉 Received Power vs Distance**');distances=np.linspace(.1,min(max(range_2m,range_70cm)*1.5,100),200);rx_powers_2m=[calculate_received_power(tx_power_2m,antenna_gain,antenna_gain,A,freq_2m,path_loss_exponent,total_additional_loss,swr_2m,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency)for A in distances];rx_powers_70cm=[calculate_received_power(tx_power_70cm,antenna_gain,antenna_gain,A,freq_70cm,path_loss_exponent,total_additional_loss,swr_70cm,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency)for A in distances];fig=go.Figure();fig.add_trace(go.Scatter(x=distances,y=rx_powers_2m,mode='lines',name=f"2m ({freq_2m} MHz)",line=dict(color=_J,width=2)));fig.add_trace(go.Scatter(x=distances,y=rx_powers_70cm,mode='lines',name=f"70cm ({freq_70cm} MHz)",line=dict(color=_O,width=2)));fig.add_hline(y=effective_sensitivity,line_dash=_Q,line_color=_N,annotation_text='Effective Sensitivity',annotation_position='right');fig.add_hline(y=effective_sensitivity+total_fade_margin,line_dash='dot',line_color=_U,annotation_text=f"Target ({link_availability:.1f}% Avail)",annotation_position='right');fig.add_vline(x=range_2m,line_dash=_Q,line_color=_J,annotation_text=f"2m: {range_2m:.1f}km");fig.add_vline(x=range_70cm,line_dash=_Q,line_color=_O,annotation_text=f"70cm: {range_70cm:.1f}km");fig.update_layout(xaxis_title='Distance (km)',yaxis_title='RX Power (dBm)',hovermode=_g,height=350,margin=dict(t=20,b=40,l=40,r=20),title=f"Propagation Model: {propagation_model.upper()} | Environment: {environment}");st.plotly_chart(fig,width=_C)
with tab3:
	col1,col2=st.columns(2)
	with col1:
		st.markdown('**Power vs Range**');fig_power=go.Figure();fig_power.add_trace(go.Scatter(x=power_levels,y=ranges_2m,mode=_h,name='2m',line=dict(color=_J,width=2),marker=dict(size=4)));fig_power.add_trace(go.Scatter(x=[tx_power_2m],y=[range_2m],mode=_i,name='Current',marker=dict(size=12,color=_N,symbol=_j)))
		if len(ranges_2m)>0:
			derivatives=np.diff(ranges_2m)/np.diff(power_levels);threshold=.5*max(derivatives);optimal_idx=np.where(derivatives>threshold)[0]
			if len(optimal_idx)>0:optimal_power=power_levels[optimal_idx[-1]];fig_power.add_vrect(x0=optimal_power-.2,x1=optimal_power+.2,fillcolor='yellow',opacity=.2,annotation_text=f"Optimal ~{optimal_power:.1f}W",annotation_position='top left')
		fig_power.update_layout(xaxis_title='TX Power (W)',yaxis_title=_k,hovermode=_g,height=350,margin=dict(t=20,b=40,l=40,r=20),title=f"Availability: {link_availability:.1f}%");st.plotly_chart(fig_power,width=_C);power_efficiency=range_2m/tx_power_2m if tx_power_2m>0 else 0;st.metric(_A7,f"{power_efficiency:.1f} km/W")
	with col2:
		st.markdown('**Altitude Impact**');altitudes=np.arange(10,121,10);horizon_ranges=4.12*np.sqrt(altitudes);rf_ranges=[]
		for alt in altitudes:r=calculate_range(tx_power_2m,antenna_gain,antenna_gain,effective_sensitivity,freq_2m,path_loss_exponent,total_additional_loss,swr_2m,total_fade_margin,alt,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency);rf_ranges.append(min(r,100))
		fig_alt=go.Figure();fig_alt.add_trace(go.Scatter(x=altitudes,y=horizon_ranges,mode=_h,name=_l,line=dict(color=_U,width=2,dash=_Q),marker=dict(size=4)));fig_alt.add_trace(go.Scatter(x=altitudes,y=rf_ranges,mode=_h,name='RF Range',line=dict(color=_P,width=2),marker=dict(size=4)));current_horizon=4.12*np.sqrt(drone_altitude);fig_alt.add_trace(go.Scatter(x=[drone_altitude],y=[current_horizon],mode=_i,name='Current Horizon',marker=dict(size=12,color=_N,symbol=_j)));fig_alt.add_trace(go.Scatter(x=[drone_altitude],y=[range_2m],mode=_i,name='Current RF Range',marker=dict(size=12,color=_J,symbol=_j)));fig_alt.update_layout(xaxis_title='Altitude (m)',yaxis_title=_k,hovermode=_g,height=350,margin=dict(t=20,b=40,l=40,r=20));st.plotly_chart(fig_alt,width=_C);altitude_efficiency=range_2m/drone_altitude if drone_altitude>0 else 0;st.metric(_A8,f"{altitude_efficiency:.2f} km/m")
	st.markdown('**🎯 Propagation Model Comparison**');models=[_K,_T,_G,_F,_E];model_names=['Free Space','Two-Ray','Okumura-Hata','ITU-P.1546','Blended'];model_ranges=[]
	for model in models:
		if model==_G:env=environment
		else:env=_D
		r=calculate_range(tx_power_2m,antenna_gain,antenna_gain,effective_sensitivity,freq_2m,2.,total_additional_loss,swr_2m,required_fade_margin,drone_altitude,model,ground_rx_height,env,50,polarization_mismatch,antenna_efficiency);model_ranges.append(min(r,100))
	fig_models=go.Figure();fig_models.add_trace(go.Bar(x=model_names,y=model_ranges,marker_color=[_J,_O,_U,_N,_P]));current_model_idx=models.index(propagation_model)if propagation_model in models else 0;fig_models.add_hline(y=range_2m,line_dash=_Q,line_color=_P,annotation_text=f"Current: {range_2m:.1f} km");fig_models.update_layout(xaxis_title=_Z,yaxis_title=_k,height=300,margin=dict(t=20,b=40,l=40,r=20),title='Comparison of Different Propagation Models');st.plotly_chart(fig_models,width=_C)
with tab4:
	st.markdown('**Optimization Recommendations**');recommendations=[];scores=[];vhf_uhf_ratio=range_2m/range_70cm if range_70cm>0 else 10;altitude_factor=min(drone_altitude/1e2,_A)
	if drone_altitude<30:expected_min_ratio=1.5;expected_max_ratio=2.
	elif drone_altitude<50:expected_min_ratio=1.3;expected_max_ratio=1.6
	elif drone_altitude<75:expected_min_ratio=1.2;expected_max_ratio=1.4
	elif drone_altitude<100:expected_min_ratio=1.1;expected_max_ratio=1.3
	else:expected_min_ratio=1.05;expected_max_ratio=1.15
	if vhf_uhf_ratio<expected_min_ratio*.9:recommendations.append(f"• **Physics Alert**: VHF range should be {expected_min_ratio:.1f}-{expected_max_ratio:.1f}× UHF range at {drone_altitude}m. Current ratio ({vhf_uhf_ratio:.2f}:1) is too low. Check VHF propagation settings.");scores.append(1)
	elif vhf_uhf_ratio<expected_min_ratio:recommendations.append(f"• **Range Ratio**: VHF range is slightly low ({vhf_uhf_ratio:.2f}:1 vs expected {expected_min_ratio:.1f}-{expected_max_ratio:.1f}:1).");scores.append(2)
	elif vhf_uhf_ratio>expected_max_ratio*1.1:recommendations.append(f"• **Range Ratio**: VHF range is higher than typical ({vhf_uhf_ratio:.2f}:1 vs expected {expected_min_ratio:.1f}-{expected_max_ratio:.1f}:1). Verify propagation model.");scores.append(2)
	else:recommendations.append(f"• **Range Ratio**: Good ({vhf_uhf_ratio:.2f}:1, expected: {expected_min_ratio:.1f}-{expected_max_ratio:.1f}:1 at {drone_altitude}m)");scores.append(3)
	if noise_figure>10:recommendations.append(f"• **Noise Figure**: Current NF={noise_figure}dB is high. Lowering to 6dB could improve sensitivity by {noise_figure-6}dB");scores.append(1)
	elif noise_figure>6:recommendations.append(f"• **Noise Figure**: Current NF={noise_figure}dB is moderate. Consider better LNA for improved sensitivity");scores.append(2)
	else:recommendations.append('• **Noise Figure**: Good');scores.append(3)
	if if_bandwidth_khz>15:recommendations.append(f"• **Bandwidth**: {if_bandwidth_khz}kHz is wide. Narrow to 12.5kHz for better sensitivity");scores.append(1)
	elif if_bandwidth_khz<8:recommendations.append(f"• **Bandwidth**: {if_bandwidth_khz}kHz is narrow. Consider impact on signal quality");scores.append(2)
	else:recommendations.append('• **Bandwidth**: Optimal');scores.append(3)
	if desense_penalty>15:recommendations.append(f"• **Desense**: {desense_penalty}dB penalty is high. Add filters to reduce");scores.append(1)
	elif desense_penalty>8:recommendations.append(f"• **Desense**: {desense_penalty}dB penalty is moderate");scores.append(2)
	else:recommendations.append('• **Desense**: Good');scores.append(3)
	if antenna_gain<0:recommendations.append('• **Antenna**: Negative gain. Replace with at least 0dBi antenna');scores.append(1)
	elif antenna_gain<3:recommendations.append(f"• **Antenna**: {antenna_gain}dBi is low. Consider 3-6dBi antenna");scores.append(2)
	else:recommendations.append(f"• **Antenna**: {antenna_gain}dBi is good");scores.append(3)
	if swr_2m>2. or swr_70cm>2.:recommendations.append('• **SWR**: High (>2.0). Tune antennas for better efficiency');scores.append(1)
	elif swr_2m>1.5 or swr_70cm>1.5:recommendations.append('• **SWR**: Moderate (1.5-2.0). Could be improved');scores.append(2)
	else:recommendations.append('• **SWR**: Good (<1.5)');scores.append(3)
	if tx_power_2m<_A:recommendations.append(f"• **Power**: {tx_power_2m}W is low. Increase to 1.5-2.0W for optimal efficiency");scores.append(1)
	elif tx_power_2m>3.:recommendations.append(f"• **Power**: {tx_power_2m}W is high. Consider battery life vs range trade-off");scores.append(2)
	else:recommendations.append(f"• **Power**: {tx_power_2m}W is in optimal range");scores.append(3)
	if drone_altitude<50:recommendations.append(f"• **Altitude**: {drone_altitude}m is low. Increase to 80-100m for better coverage");scores.append(1)
	elif drone_altitude<80:recommendations.append(f"• **Altitude**: {drone_altitude}m is moderate");scores.append(2)
	else:recommendations.append(f"• **Altitude**: {drone_altitude}m is good");scores.append(3)
	if propagation_model==_G and freq_2m<150:recommendations.append('• **Propagation Model**: Using Okumura-Hata with VHF extension (100-150 MHz)');scores.append(3)
	if propagation_model==_F or propagation_model==_E:recommendations.append(f"• **Propagation Model**: Using {propagation_model.upper()} with v2.0.0 corrections (industry validated)");scores.append(3)
	if link_availability>99. and fade_margin_2m<15:recommendations.append(f"• **Availability**: {link_availability:.1f}% requires high reliability. Ensure fade margin >15dB");scores.append(1)
	elif link_availability>95. and fade_margin_2m<10:recommendations.append(f"• **Availability**: {link_availability:.1f}% requires moderate reliability");scores.append(2)
	else:recommendations.append(f"• **Availability**: {link_availability:.1f}% requirement is well supported");scores.append(3)
	for rec in recommendations:
		if'Good'in rec or'Optimal'in rec or'industry validated'in rec:st.success(rec)
		elif'moderate'in rec.lower()or'consider'in rec.lower()or'slightly'in rec.lower():st.info(rec)
		else:st.warning(rec)
	if scores:
		avg_score=sum(scores)/len(scores)
		if avg_score>=2.7:st.success(f"✅ **Overall Score: {avg_score:.1f}/3.0** - Well optimized!")
		elif avg_score>=2.:st.info(f"⚠️ **Overall Score: {avg_score:.1f}/3.0** - Room for improvement")
		else:st.error(f"❌ **Overall Score: {avg_score:.1f}/3.0** - Significant improvements needed")
with tab5:
	col1,col2=st.columns(2)
	with col1:st.markdown('**Configuration Summary**');config_data={'Category':['Radio','Radio','Antenna','Antenna',_n,_n,_n,'Drone','Drone',_w,'Propagation'],_V:['2m TX Power','70cm TX Power','Gain',_r,_d,_A9,'Desense','Altitude','RX Height','Atmos Loss','Model'],_m:[f"{tx_power_2m:.2f}W",f"{tx_power_70cm:.2f}W",f"{antenna_gain:+.0f}dBi",antenna_polarization,f"{noise_figure:.0f}dB",f"{if_bandwidth_khz:.1f}kHz",f"{desense_penalty:.0f}dB",f"{drone_altitude}m",f"{ground_rx_height}m",f"{atmospheric_loss:.2f}dB/km",propagation_model.upper()]};df_config=pd.DataFrame(config_data);st.dataframe(df_config,hide_index=_B,width=_C,height=400)
	with col2:
		col2a,col2b=st.columns(2)
		with col2a:st.metric(_AA,f"{system_range:.1f} km",f"{'2m'if range_2m<range_70cm else'70cm'} limited");st.metric('2m Range',f"{range_2m:.1f} km");st.metric(_AB,f"{range_70cm:.1f} km");st.metric(_l,f"{radio_horizon:.1f} km")
		with col2b:st.metric(_e,f"{fade_margin_2m:.1f} dB",f"{'✓'if fade_margin_2m>=total_fade_margin else'✗'} target");st.metric(_c,f"{link_availability:.1f}%",f"+{additional_availability_margin:.0f}dB");noise_floor=-174+10*np.log10(if_bandwidth_khz*1000);theoretical_sensitivity=noise_floor+noise_figure+snr_required;st.metric('Sensitivity (Theoretical)',f"{theoretical_sensitivity:.0f} dBm",help='Best-case without desense');st.metric('Sensitivity (Effective)',f"{effective_sensitivity:.0f} dBm",delta=f"+{desense_penalty:.0f} dB desense",help='Real-world with desense penalty');efficiency=system_range/radio_horizon*100;st.metric('Horizon Util.',f"{min(efficiency,100):.0f}%")
	st.markdown(_M);col3,col4=st.columns(2)
	with col3:st.markdown('**📊 Performance Analysis**');power_eff=range_2m/tx_power_2m if tx_power_2m>0 else 0;altitude_eff=range_2m/drone_altitude if drone_altitude>0 else 0;antenna_eff=antenna_gain*10;metrics_df=pd.DataFrame({'Metric':[_A7,_A8,'Antenna Score','Receiver Quality','System Balance'],_m:[f"{power_eff:.2f} km/W",f"{altitude_eff:.2f} km/m",f"{antenna_eff:.0f}/90",f"{'Good'if noise_figure<8 else'Fair'if noise_figure<12 else'Poor'}",f"{'Balanced'if abs(range_2m-range_70cm)<5 else'Imbalanced'}"],'Score':[f"{min(power_eff*20,100):.0f}/100",f"{min(altitude_eff*100,100):.0f}/100",f"{antenna_eff:.0f}/90",f"{max(0,100-(noise_figure-6)*10):.0f}/100",f"{max(0,100-abs(range_2m-range_70cm)*5):.0f}/100"]});st.dataframe(metrics_df,hide_index=_B,width=_C,height=200)
	with col4:
		st.markdown('**🎯 Quick Actions**')
		if st.button('📈 Maximize Range',width=_C):st.info('To maximize range: 1) Increase altitude, 2) Use directional antenna, 3) Reduce NF, 4) Add filtering')
		if st.button('⚡ Optimize Efficiency',width=_C):st.info('For efficiency: 1) Set power to 1.5-2.0W, 2) Tune antennas (SWR<1.5), 3) Use optimal bandwidth')
		if st.button('🛡️ Improve Reliability',width=_C):st.info('For reliability: 1) Increase fade margin, 2) Add diversity, 3) Use cavity filters, 4) Lower NF')
		if st.button('💰 Cost-Effective',width=_C):st.info('Cost-effective upgrades: 1) Better antenna, 2) LNA, 3) Filters, 4) Tune existing equipment')
with tab6:
	st.markdown(f"**Help & About {APP_NAME}**");col_about,col_features=st.columns(2)
	with col_about:
		with st.expander('📖 About This App',expanded=_B):st.markdown(f"""
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
            
            **Session Started:** {st.session_state.app_start_time.strftime(_I)}
            """)
	with col_features:
		with st.expander('🚀 Key Features',expanded=_B):st.markdown('\n            ### Advanced Features\n            \n            **📡 RF Physics Engine:**\n            - VHF/UHF physics-based corrections\n            - Radio horizon awareness\n            - Realistic range calculations\n            \n            **🗺️ Interactive Mapping:**\n            - Click-to-place drone positioning\n            - Multi-layer coverage visualization\n            - KML export for Google Earth\n            \n            **📊 Comprehensive Analysis:**\n            - Link budget calculations\n            - Propagation model comparisons\n            - Power vs range optimization\n            \n            **⚡ Optimization Tools:**\n            - Automatic recommendations\n            - Performance scoring\n            - Quick action suggestions\n            ')
	st.markdown(_M);col_guide,col_tech=st.columns(2)
	with col_guide:
		with st.expander('📖 User Guide',expanded=_H):st.markdown('\n            ### How to Use This Tool\n            \n            1. **Configure Parameters**: Use sidebar to set all system parameters\n            2. **Set Location**: Click on map to place drone (optional)\n            3. **Analyze Results**: Review different tabs for analysis\n            4. **Optimize**: Use recommendations to improve system\n            5. **Export**: Download reports and data for sharing\n            \n            ### Key Parameters Explained\n            \n            - **Noise Figure (NF)**: Lower is better. Affects receiver sensitivity\n            - **Bandwidth (BW)**: Narrower = better sensitivity but may affect signal quality\n            - **Desense**: Loss from nearby transmitters. Use filters to reduce\n            - **SWR**: Should be <1.5 for good efficiency\n            - **Availability**: Higher % = more reliable but shorter range\n            ')
	with col_tech:
		with st.expander('⚙️ Technical Details',expanded=_H):st.markdown(f"""
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
            
            **Expected VHF/UHF Range Ratios (v2.0.0)**:
            ```
            Industry Standard VHF/UHF (2m/70cm) range ratios:
            - <30m altitude: 1.5-2.0:1
            - 30-50m altitude: 1.3-1.6:1
            - 50-75m altitude: 1.2-1.4:1
            - 75-100m altitude: 1.1-1.3:1
            - >100m altitude: 1.05-1.15:1
            
            Current ratio at {drone_altitude}m: {range_2m/range_70cm if range_70cm>0 else _L:.2f}:1
            
            Physics corrections in v2.0.0:
            1. ITU-P.1546: Exponential altitude reduction (e^(-2.5*alt/100))
            2. Okumura-Hata: Extended VHF support (100-150 MHz)
            3. Blended model: No double-counting of frequency effects
            4. Free-space convergence at high altitude (>150m)
            
            Path loss difference (146 vs 446 MHz):
            - Free space: 9.7 dB at same distance
            - Ground level: VHF advantage ~7 dB
            - 100m altitude: VHF advantage ~0.2 dB
            - >150m altitude: Approaches free space (9.7 dB diff)
            ```            
            
            ### Version History
            
            **v2.0.0** (Current):
            - CRITICAL: Fixed VHF/UHF ratio from 3.05:1 to 1.2:1 at 100m
            - ITU-P.1546: Exponential altitude reduction (8% at 100m vs 60% before)
            - Okumura-Hata: VHF extension with validated corrections
            - Blended model: Removed double-counting of frequency effects
            
            **v1.3.4**:
            - Industry-standard VHF/UHF ratio corrections
            - Enhanced ITU-R P.1546 model with altitude-aware terrain effects
            - Blended propagation model for best accuracy
            - Improved environment and altitude corrections
            
            **v1.3.3**:
            - Added RadioSport branding
            - Version tracking system
            - Enhanced UI/UX
            - App info panel
            
            ### License & Usage
            
            This tool is provided for educational and planning purposes.
            Always verify calculations with field testing.
            Commercial use requires permission.
            """)
st.markdown(_M)
st.markdown('### 📤 Export Results')
col1,col2,col3,col4,col5=st.columns(5)
with col1:
	if st.button('📄 Full Report',width=_C):
		report_text=f"""
# {APP_NAME} RF Coverage Analysis Report

## App Information
- **Application**: {APP_NAME} v{APP_VERSION}
- **Generated**: {pd.Timestamp.now().strftime(_I)}
- **Session Start**: {st.session_state.app_start_time.strftime(_I)}

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
- **Horizon Utilization**: {range_2m/radio_horizon*100 if radio_horizon>0 else 0:.0f}%
- **Fade Margin**: {fade_margin_2m:.1f} dB
- **Effective Sensitivity**: {effective_sensitivity:.0f} dBm

## Receiver Characteristics
- **Noise Figure**: {noise_figure} dB
- **Bandwidth**: {if_bandwidth_khz} kHz
- **Required SNR**: {snr_required} dB
- **Desense Penalty**: {desense_penalty} dB

## Sensitivity Calculation
- **Noise Floor (kTB)**: {-174+10*np.log10(if_bandwidth_khz*1000):.1f} dBm
- **Theoretical Sensitivity**: {noise_floor+noise_figure+snr_required:.1f} dBm (kTB + NF + SNR)
- **Practical Minimum**: -127 dBm (real-world limit)
- **Effective Sensitivity**: {effective_sensitivity:.0f} dBm (includes desense)
- **Sensitivity Formula**: max(kTB + NF + SNR, -127 dBm) + Desense

## VHF vs UHF Analysis (v2.0.0 Industry Standard)
- **VHF/UHF Range Ratio**: {range_2m/range_70cm if range_70cm>0 else _L:.2f}:1
- **Expected Ratio at {drone_altitude}m**: {1.5 if drone_altitude<30 else 1.3 if drone_altitude<50 else 1.2 if drone_altitude<75 else 1.1 if drone_altitude<100 else 1.05:.1f}-{2. if drone_altitude<30 else 1.6 if drone_altitude<50 else 1.4 if drone_altitude<75 else 1.3 if drone_altitude<100 else 1.15:.1f}:1
- **Status**: {"✅ Within expected range"if(1.5 if drone_altitude<30 else 1.3 if drone_altitude<50 else 1.2 if drone_altitude<75 else 1.1 if drone_altitude<100 else 1.05)<=range_2m/range_70cm<=(2. if drone_altitude<30 else 1.6 if drone_altitude<50 else 1.4 if drone_altitude<75 else 1.3 if drone_altitude<100 else 1.15)else"⚠️ Check parameters"if range_70cm>0 else _L}

## Link Budget (2m Band)
- TX Power: {10*np.log10(tx_power_2m*1000):.1f} dBm
- TX Antenna Gain: +{antenna_gain:.1f} dBi
- EIRP: {10*np.log10(tx_power_2m*1000)+antenna_gain:.1f} dBm
- Path Loss: {calculate_path_loss_db(system_range,freq_2m,path_loss_exponent,drone_altitude,propagation_model,ground_rx_height,environment,time_percent):.1f} dB
- Total Additional Losses: {total_additional_loss:.1f} dB
- RX Power: {calculate_received_power(tx_power_2m,antenna_gain,antenna_gain,system_range,freq_2m,path_loss_exponent,total_additional_loss,swr_2m,drone_altitude,propagation_model,ground_rx_height,environment,time_percent,polarization_mismatch,antenna_efficiency):.1f} dBm

## Link Budget (70cm Band)
- TX Power: {10*np.log10(tx_power_70cm*1000):.1f} dBm
- Path Loss: {calculate_path_loss_db(range_70cm,freq_70cm,path_loss_exponent,drone_altitude,propagation_model,ground_rx_height,environment,time_percent):.1f} dB
- Total Additional Losses: {total_additional_loss:.1f} dB

## Recommendations
""";vhf_uhf_ratio=range_2m/range_70cm if range_70cm>0 else 10
		if drone_altitude<30:expected_min_ratio,expected_max_ratio=1.5,2.
		elif drone_altitude<50:expected_min_ratio,expected_max_ratio=1.3,1.6
		elif drone_altitude<75:expected_min_ratio,expected_max_ratio=1.2,1.4
		elif drone_altitude<100:expected_min_ratio,expected_max_ratio=1.1,1.3
		else:expected_min_ratio,expected_max_ratio=1.05,1.15
		if vhf_uhf_ratio<expected_min_ratio*.9:report_text+=f"- **CRITICAL**: VHF range ({range_2m:.1f}km) is unrealistically low compared to UHF ({range_70cm:.1f}km). Expected ratio at {drone_altitude}m: {expected_min_ratio:.1f}-{expected_max_ratio:.1f}:1. Verify propagation model and parameters.\n"
		elif vhf_uhf_ratio<expected_min_ratio:report_text+=f"- **Warning**: VHF range should typically exceed UHF range. Current ratio: {vhf_uhf_ratio:.2f}:1 (expected: {expected_min_ratio:.1f}-{expected_max_ratio:.1f}:1)\n"
		elif vhf_uhf_ratio>expected_max_ratio*1.1:report_text+=f"- **Notice**: VHF advantage is larger than typical ({vhf_uhf_ratio:.2f}:1 vs expected {expected_min_ratio:.1f}-{expected_max_ratio:.1f}:1). This may occur with certain propagation models or favorable conditions.\n"
		if propagation_model==_G:report_text+=f"- **Model**: Using Okumura-Hata with VHF extension (100-150 MHz)\n"
		if propagation_model==_F or propagation_model==_E:report_text+=f"- **Model**: Using {propagation_model.upper()} with v2.0.0 exponential altitude corrections\n"
		if noise_figure>8:report_text+=f"- Reduce noise figure from {noise_figure}dB to 6dB for better sensitivity\n"
		if desense_penalty>10:report_text+=f"- Add cavity filters to reduce desense penalty of {desense_penalty}dB\n"
		if antenna_gain<3:report_text+=f"- Upgrade to higher gain antenna (>3dBi)\n"
		if drone_altitude<80:report_text+=f"- Increase altitude to 80-100m for better coverage\n"
		if swr_2m>1.5:report_text+=f"- Tune antenna for lower SWR (current: {swr_2m:.1f})\n"
		report_text+=f"""
## Physical Reality Check (v2.0.0 - Industry Validated)
- **Radio Horizon**: {radio_horizon:.1f} km (absolute limit for line-of-sight)
- **Horizon Utilization**: {range_2m/radio_horizon*100 if radio_horizon>0 else 0:.0f}%
- **Propagation Model**: {propagation_model.upper()} with v2.0.0 corrections
- **VHF/UHF Behavior**: Ratio of {range_2m/range_70cm if range_70cm>0 else _L:.2f}:1 (expected at {drone_altitude}m: {expected_min_ratio:.1f}-{expected_max_ratio:.1f}:1)
- **Altitude Effect**: Exponential reduction: e^(-2.5*alt/100) = {np.exp(-2.5*min(drone_altitude/1e2,_A)):.2f} at {drone_altitude}m
- **Path Loss Difference**: VHF vs UHF: Free space = 9.7 dB, Actual = {calculate_path_loss_db(_A,freq_2m,2.,drone_altitude,_K)-calculate_path_loss_db(_A,freq_70cm,2.,drone_altitude,_K):.1f} dB at 1 km
- **Cross-Band Limitation**: System limited by weaker of 2 bands
- **Beyond Horizon**: Not considered (requires special propagation modes)

## v2.0.0 Corrections Applied
1. **ITU-P.1546**: Exponential altitude reduction (8% at 100m vs 60% before)
2. **Okumura-Hata**: VHF extension with -2 dB at 100 MHz (was -8 dB)
3. **Blended Model**: No double-counting of frequency effects
4. **Expected 100m ratio**: 1.1-1.3:1 (was 3.05:1 in v1.3.4)

## Notes
- Report generated by: {APP_NAME} v{APP_VERSION}
- Developer: {DEVELOPER}
- Copyright: {COPYRIGHT}
- For planning purposes only - verify with field testing
- **Important**: VHF should have 1.05-2.0× better range than UHF depending on altitude
- **v2.0.0 Validation**: Based on ITU-R P.1546, empirical measurements, and field data
""";st.download_button('📥 Download Markdown',data=report_text,file_name=f"{APP_NAME.replace(' ','_')}_report_{pd.Timestamp.now().strftime(_o)}.md",mime='text/markdown',width=_C)
with col2:
	if st.button('📊 Data CSV',width=_C):csv_data=pd.DataFrame({_V:[_AA,'2m Range',_AB,_l,'Horizon Utilization','2m Fade Margin','70cm Fade Margin','Coverage Area','VHF/UHF Ratio',_Z,'TX Power 2m','TX Power 70cm','Altitude','Desense',_d,_A9,_c,_f,'App Version','Generation Time'],_m:[system_range,range_2m,range_70cm,radio_horizon,range_2m/radio_horizon*100 if radio_horizon>0 else 0,fade_margin_2m,fade_margin_70cm,np.pi*system_range**2,range_2m/range_70cm if range_70cm>0 else 0,propagation_model.upper(),tx_power_2m,tx_power_70cm,drone_altitude,desense_penalty,noise_figure,if_bandwidth_khz,link_availability,additional_availability_margin,APP_VERSION,pd.Timestamp.now().strftime(_I)],'Unit':[_X,_X,_X,_X,'%',_R,_R,'km²','ratio','model','W','W','m',_R,_R,'kHz','%',_R,'version',_AC]});st.download_button('📥 Download CSV',data=csv_data.to_csv(index=_H),file_name=f"{APP_NAME.replace(' ','_')}_data_{pd.Timestamp.now().strftime(_o)}.csv",mime='text/csv',width=_C)
with col3:
	if st.button('🗺️ KML Export',width=_C):
		if st.session_state.drone_location:
			kml_content=f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{APP_NAME} Coverage</name>
    <description>RF Coverage Analysis - {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</description>
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
        VHF/UHF Ratio: {range_2m/range_70cm if range_70cm>0 else _L:.2f}
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
''';center_lat=st.session_state.drone_location[0];center_lon=st.session_state.drone_location[1]
			for angle in range(0,361,5):lat_offset=system_range/111.*np.cos(np.radians(angle));lon_offset=system_range/(111.*np.cos(np.radians(center_lat)))*np.sin(np.radians(angle));point_lat=center_lat+lat_offset;point_lon=center_lon+lon_offset;kml_content+=f"              {point_lon},{point_lat},0\n"
			kml_content+='            </coordinates>\n          </LinearRing>\n        </outerBoundaryIs>\n      </Polygon>\n    </Placemark>\n  </Document>\n</kml>';st.download_button('📥 Download KML',data=kml_content,file_name=f"{APP_NAME.replace(' ','_')}_coverage_{pd.Timestamp.now().strftime(_o)}.kml",mime='application/vnd.google-earth.kml+xml',width=_C)
		else:st.warning('Set drone location on map first')
with col4:
	if st.button('📋 Copy Config',width=_C):config_text=f"""{APP_NAME} v{APP_VERSION}
2m: {tx_power_2m}W @ {freq_2m}MHz | 70cm: {tx_power_70cm}W @ {freq_70cm}MHz
Alt: {drone_altitude}m | Gain: {antenna_gain}dBi | NF: {noise_figure}dB
Horizon: {radio_horizon:.1f}km | 2m Range: {range_2m:.1f}km | 70cm Range: {range_70cm:.1f}km
Horizon Utilization: {range_2m/radio_horizon*100 if radio_horizon>0 else 0:.0f}%
VHF/UHF Ratio: {range_2m/range_70cm if range_70cm>0 else _L:.2f}:1 (v2.0.0 validated)
Availability: {link_availability:.1f}% | Model: {propagation_model}
Generated: {pd.Timestamp.now().strftime(_I)}""";st.code(config_text,language='text');st.info('Select and copy the text above')
with col5:
	if st.button('💾 Save Config',width=_C):
		config={'tx_2m':tx_power_2m,'tx_70cm':tx_power_70cm,'antenna_gain':antenna_gain,'altitude':drone_altitude,'nf':noise_figure,'bw':if_bandwidth_khz,_v:desense_penalty,'fade_margin':required_fade_margin,'availability':link_availability,'propagation_model':propagation_model,_AC:pd.Timestamp.now().strftime(_I)};st.session_state.saved_configs.append(config)
		try:
			with open(_q,'w')as f:json.dump(st.session_state.saved_configs,f,indent=2)
			st.success('✅ Configuration saved to file!')
		except Exception as e:st.warning(f"⚠️ Saved to session only: {str(e)}")
st.markdown("<hr style='margin:10px 0;'>",unsafe_allow_html=_B)
st.markdown(_M)
st.markdown(f'''
<div class=\'footer\'>
    <p><strong>{APP_NAME} v{APP_VERSION}</strong></p>
    <p>Wouxun KG-UV9D Plus | v2.0.0 Industry-Validated Physics | Realistic 1.1-2.0:1 VHF/UHF Ratios</p>
    <p>✅ v2.0.0: Fixed 3.05:1 ratio → 1.2:1 at 100m | ✅ Exponential altitude corrections | ✅ No double-counting</p>
    <p>{COPYRIGHT} | <a href="{GITHUB_URL}" target="_blank">GitHub</a> | Always verify with field measurements</p>
</div>
''',unsafe_allow_html=_B)
if(pd.Timestamp.now()-st.session_state.last_cleanup).total_seconds()>300:
	if len(st.session_state.saved_configs)>20:st.session_state.saved_configs=st.session_state.saved_configs[-10:]
	st.session_state.last_cleanup=pd.Timestamp.now()