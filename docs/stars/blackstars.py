'\nGhana Black Stars AI Lineup Builder - Streamlit Version with OpenRouter AI Integration\nAuthor: RNK RadioSport\nVersion: 2.1.0 - Complete Features & Full Player Database\n'
_AV='🔑 OpenRouter API Key'
_AU='Ibrahim Osman'
_AT='Leicester'
_AS='AJ Auxerre'
_AR='Hearts of Oak'
_AQ='St. Gallen'
_AP='squad_mode'
_AO='world_cup_squad'
_AN='ai_analysis'
_AM='Formation Power Rankings'
_AL='text/plain'
_AK='primary'
_AJ='%B %d, %Y at %H:%M'
_AI='message'
_AH='choices'
_AG='worldcup'
_AF='application/json'
_AE='ATTACK'
_AD='3-5-2 (Offensive)'
_AC='Medeama SC'
_AB='api_usage'
_AA='selected_formation'
_A9='failed'
_A8='DEFENSIVE MID'
_A7='ATTACKING MID'
_A6='STRIKER'
_A5='3-4-3 (Ultra Attack)'
_A4='full'
_A3='%Y%m%d_%H%M%S'
_A2='score'
_A1=False
_A0='STRIKERS'
_z='CAM'
_y='RM'
_x='LM'
_w=None
_v='user'
_u='total_caps'
_t='RWB'
_s='DM'
_r='='
_q='-'
_p='avg_age'
_o='LWB'
_n='role'
_m='chemistry'
_l='RST'
_k='LST'
_j='MIDFIELD'
_i='avg_rating'
_h='RCM'
_g='LCM'
_f='DEF'
_e='defense'
_d='attack'
_c='openrouter'
_b='content'
_a='GOALKEEPER'
_Z='DEFENSE'
_Y='LB'
_X='ATT'
_W='VERSATILE'
_V='RCB'
_U='LCB'
_T='---'
_S='RB'
_R='MID'
_Q='GK'
_P='LW'
_O='CM'
_N='CB'
_M='ST'
_L='RW'
_K=True
_J='versatility'
_I='form'
_H='caps'
_G='name'
_F='age'
_E='label'
_D='club'
_C='rating'
_B='fullName'
_A='positions'
import streamlit as st,json,requests
from datetime import datetime
import time
from io import BytesIO
from reportlab.lib.pagesizes import A4,letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.pdfgen import canvas
import base64
st.set_page_config(page_title='Tactical Analysis',page_icon='🏆',layout='wide',initial_sidebar_state='expanded',menu_items={'Report a Bug':'https://github.com/rkarikari/stem','About':'Copyright © RNK, 2026 RadioSport. All rights reserved.\n\nVersion 2.1.0\n\nFeatures:\n• Complete Player Database  \n• AI Analysis  \n• Formation Rankings  \n• Interactive Chat\n'})
st.markdown('\n<style>\n    .main {\n        background: linear-gradient(135deg, #1a472a 0%, #0d2818 100%);\n    }\n    .stButton>button {\n        width: 100%;\n        background: linear-gradient(135deg, #d4af37 0%, #fcd116 100%);\n        color: #1a472a;\n        font-weight: bold;\n        border: none;\n        padding: 0.5rem 1rem;\n        border-radius: 0.5rem;\n        transition: all 0.3s;\n    }\n    .stButton>button:hover {\n        transform: translateY(-2px);\n        box-shadow: 0 5px 15px rgba(212, 175, 55, 0.5);\n    }\n    .player-card {\n        background: rgba(255, 255, 255, 0.1);\n        backdrop-filter: blur(20px);\n        border-radius: 12px;\n        padding: 1rem;\n        border: 1px solid rgba(255, 255, 255, 0.2);\n        margin: 0.5rem 0;\n        transition: all 0.3s;\n    }\n    .player-card:hover {\n        transform: translateX(5px);\n        border-color: #d4af37;\n    }\n    .ai-analysis-box {\n        background: linear-gradient(135deg, rgba(26, 71, 42, 0.95), rgba(13, 40, 24, 0.95));\n        border: 3px solid #fcd116;\n        border-radius: 15px;\n        padding: 2rem;\n        margin: 1rem 0;\n        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);\n    }\n    h1, h2, h3 {\n        color: #fcd116;\n    }\n    .stSelectbox label, .stTextInput label {\n        color: #d4af37 !important;\n        font-weight: bold;\n    }\n    .pitch-container {\n        background: linear-gradient(180deg, #2d5016 0%, #1a472a 50%, #2d5016 100%);\n        padding: 2rem 1rem;\n        border-radius: 15px;\n        border: 3px solid #fcd116;\n        min-height: 600px;\n    }\n    .formation-line {\n        display: flex;\n        justify-content: center;\n        align-items: center;\n        gap: 1rem;\n        margin: 2rem 0;\n        flex-wrap: wrap;\n    }\n    .position-slot {\n        display: flex;\n        flex-direction: column;\n        align-items: center;\n        min-width: 120px;\n    }\n    .line-label {\n        text-align: center;\n        color: #fcd116;\n        font-weight: bold;\n        font-size: 0.9rem;\n        margin: 0.5rem 0;\n        text-transform: uppercase;\n        letter-spacing: 1px;\n    }\n</style>\n',unsafe_allow_html=_K)
OPENROUTER_API_URL='https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODELS_URL='https://openrouter.ai/api/v1/models'
APP_VERSION='2.0.0'
APP_NAME='Ghana Black Stars AI Lineup Builder'
RELEASE_DATE='2025'
FEATURES=['Complete 64-Player Database','12 Tactical Formations','AI-Powered Analysis',_AM,'Interactive Tactical Chat','World Cup Squad Builder','PDF/Text/JSON Export']
if'lineup'not in st.session_state:st.session_state.lineup={}
if _AN not in st.session_state:st.session_state.ai_analysis=_w
if'api_keys'not in st.session_state:st.session_state.api_keys={_c:''}
if _AB not in st.session_state:st.session_state.api_usage={}
if'openrouter_models'not in st.session_state:st.session_state.openrouter_models=[]
if'selected_model'not in st.session_state:st.session_state.selected_model='mistralai/mistral-7b-instruct'
if _AO not in st.session_state:st.session_state.world_cup_squad=[]
if _AP not in st.session_state:st.session_state.squad_mode=_A4
if'chat_history'not in st.session_state:st.session_state.chat_history=[]
if'current_tab'not in st.session_state:st.session_state.current_tab='Lineup Builder'
PLAYERS={_Q:[{_G:'Ati-Zigi',_B:'Lawrence Ati-Zigi',_D:_AQ,_C:78,_F:28,_H:18,_I:8,_J:1},{_G:'Asare',_B:'Benjamin Asare',_D:_AR,_C:76,_F:25,_H:8,_I:9,_J:1},{_G:'Nurudeen',_B:'Abdul Manaf Nurudeen',_D:'KAS Eupen',_C:74,_F:26,_H:6,_I:7,_J:1},{_G:'Anang',_B:'Joseph Anang',_D:"St Patrick's",_C:74,_F:24,_H:3,_I:7,_J:1},{_G:'Wollacott',_B:'Joseph Wollacott',_D:'Hibernian',_C:73,_F:29,_H:9,_I:7,_J:1}],_f:[{_G:'Salisu',_B:'Mohammed Salisu',_D:'Monaco',_C:82,_F:26,_H:22,_I:8,_J:7,_A:[_N,_Y]},{_G:'Djiku',_B:'Alexander Djiku',_D:'Spartak Moscow',_C:81,_F:31,_H:37,_I:8,_J:7,_A:[_N,_S]},{_G:'Lamptey',_B:'Tariq Lamptey',_D:'Brighton',_C:80,_F:24,_H:15,_I:8,_J:6,_A:[_S,_L]},{_G:'Aidoo',_B:'Joseph Aidoo',_D:'Celta Vigo',_C:79,_F:29,_H:45,_I:8,_J:6,_A:[_N]},{_G:'Seidu',_B:'Alidu Seidu',_D:'Stade Rennes',_C:78,_F:25,_H:22,_I:8,_J:7,_A:[_S,_N]},{_G:'Kohn',_B:'Derrick Kohn',_D:'Union Berlin',_C:77,_F:26,_H:10,_I:8,_J:6,_A:[_Y,_o]},{_G:'Mensah',_B:'Gideon Mensah',_D:_AS,_C:76,_F:27,_H:18,_I:7,_J:5,_A:[_Y]},{_G:'Ambrosius',_B:'Stephan Ambrosius',_D:_AQ,_C:77,_F:26,_H:10,_I:7,_J:6,_A:[_N]},{_G:'Mumin',_B:'Abdul Mumin',_D:'Rayo Vallecano',_C:76,_F:27,_H:15,_I:7,_J:6,_A:[_N]},{_G:'Adjei',_B:'Nathaniel Adjei',_D:'FC Lorient',_C:75,_F:25,_H:5,_I:7,_J:5,_A:[_N]},{_G:'Opoku J.',_B:'Jerome Opoku',_D:'Istanbul Basaksehir',_C:75,_F:26,_H:11,_I:7,_J:6,_A:[_N]},{_G:'Annan',_B:'Ebenezer Annan',_D:'AS Saint-Etienne',_C:74,_F:24,_H:7,_I:7,_J:5,_A:[_N]},{_G:'Simpson',_B:'Razak Simpson',_D:'Nations FC',_C:73,_F:24,_H:6,_I:8,_J:6,_A:[_S,_N]},{_G:'Adjetey',_B:'Jonas Adjetey',_D:'FC Basel',_C:74,_F:25,_H:7,_I:7,_J:6,_A:[_N]},{_G:'Afful',_B:'Isaac Afful',_D:'Samartex',_C:71,_F:23,_H:3,_I:7,_J:5,_A:[_S]},{_G:'Schindler',_B:'Kingsley Schindler',_D:'Hannover 96',_C:73,_F:32,_H:9,_I:6,_J:6,_A:[_S,_L]}],_R:[{_G:'Partey',_B:'Thomas Partey',_D:'Arsenal',_C:84,_F:32,_H:58,_I:8,_J:7,_A:[_s,_O]},{_G:'Kudus',_B:'Mohammed Kudus',_D:'West Ham',_C:84,_F:25,_H:32,_I:9,_J:9,_A:['AM',_L,_M,_O]},{_G:'Ashimeru',_B:'Majeed Ashimeru',_D:'Anderlecht',_C:77,_F:28,_H:24,_I:7,_J:6,_A:[_O,_s]},{_G:'Samed',_B:'Salis Abdul Samed',_D:'RC Lens',_C:78,_F:25,_H:17,_I:8,_J:6,_A:[_s,_O]},{_G:'E. Owusu',_B:'Elisha Owusu',_D:_AS,_C:76,_F:28,_H:20,_I:7,_J:6,_A:[_s,_O]},{_G:'Sulemana I.',_B:'Ibrahim Sulemana',_D:'Atalanta',_C:76,_F:22,_H:8,_I:7,_J:6,_A:[_O]},{_G:'Diomande',_B:'Mohamed Diomande',_D:'Rangers',_C:74,_F:23,_H:5,_I:8,_J:7,_A:[_O,_s]},{_G:'Sibo',_B:'Kwasi Sibo',_D:_AR,_C:73,_F:23,_H:6,_I:8,_J:6,_A:[_O,_s]},{_G:'Francis',_B:'Abu Francis',_D:'Cercle Brugge',_C:75,_F:23,_H:10,_I:7,_J:6,_A:[_O,'AM']},{_G:'Addo E.',_B:'Edmund Addo',_D:'Sheriff Tiraspol',_C:73,_F:25,_H:12,_I:7,_J:5,_A:[_s]},{_G:'Mamudu',_B:'Kamaradini Mamudu',_D:_AC,_C:72,_F:23,_H:5,_I:7,_J:6,_A:[_O]},{_G:'Baidoo',_B:'Michael Baidoo',_D:'Elfsborg',_C:73,_F:26,_H:5,_I:7,_J:6,_A:[_O,'AM']},{_G:'Antwi',_B:'Emmanuel Antwi',_D:'FK Pribram',_C:71,_F:24,_H:4,_I:7,_J:6,_A:[_O]},{_G:'P. Owusu',_B:'Prince Owusu',_D:_AC,_C:70,_F:21,_H:3,_I:8,_J:6,_A:[_O]}],_W:[{_G:'Yirenkyi',_B:'Caleb Yirenkyi',_D:'Nordsjaelland',_C:73,_F:20,_H:4,_I:8,_J:8,_A:[_O,'AM',_S,_L]}],_X:[{_G:'I. Williams',_B:'Inaki Williams',_D:'Athletic Bilbao',_C:82,_F:31,_H:24,_I:9,_J:7,_A:[_M,_L]},{_G:'Semenyo',_B:'Antoine Semenyo',_D:'Bournemouth',_C:81,_F:26,_H:18,_I:9,_J:7,_A:[_M,_L]},{_G:'J. Ayew',_B:'Jordan Ayew',_D:_AT,_C:79,_F:34,_H:118,_I:8,_J:8,_A:[_M,_L,_P]},{_G:'Fatawu',_B:'Abdul Fatawu Issahaku',_D:_AT,_C:79,_F:21,_H:23,_I:8,_J:7,_A:[_L,_P]},{_G:'Thomas-Asante',_B:'Brandon Thomas-Asante',_D:'Coventry City',_C:78,_F:27,_H:9,_I:9,_J:6,_A:[_M,_L]},{_G:'Sulemana K.',_B:'Kamaldeen Sulemana',_D:'Southampton',_C:78,_F:23,_H:18,_I:7,_J:8,_A:[_P,_L,_M]},{_G:'Bukari',_B:'Osman Bukari',_D:'Austin FC',_C:77,_F:26,_H:16,_I:8,_J:7,_A:[_L,_P]},{_G:'Nuamah',_B:'Ernest Nuamah',_D:'Lyon',_C:77,_F:22,_H:12,_I:8,_J:7,_A:[_L,_P]},{_G:'Paintsil',_B:'Joseph Paintsil',_D:'LA Galaxy',_C:77,_F:28,_H:12,_I:8,_J:7,_A:[_L,_P]},{_G:'Bonsu Baah',_B:'Christopher Bonsu Baah',_D:'KRC Genk',_C:75,_F:23,_H:8,_I:8,_J:7,_A:[_L,_P]},{_G:_AU,_B:_AU,_D:'Feyenoord',_C:75,_F:20,_H:7,_I:8,_J:7,_A:[_L,_P,'AM']},{_G:'Adu Kwabena',_B:'Prince Kwabena Adu',_D:'Viktoria Plzen',_C:75,_F:22,_H:2,_I:8,_J:7,_A:[_M,_L,_P]},{_G:'Afriyie',_B:'Jerry Afriyie',_D:'RAAL La Louviere',_C:73,_F:20,_H:3,_I:8,_J:6,_A:[_M]},{_G:'Afena-Gyan',_B:'Felix Afena-Gyan',_D:'Cremonese',_C:73,_F:21,_H:5,_I:7,_J:6,_A:[_M]},{_G:'Opoku',_B:'Kwame Opoku',_D:'Asante Kotoko',_C:72,_F:23,_H:2,_I:7,_J:5,_A:[_M]},{_G:'Fuseini',_B:'Mohammed Fuseini',_D:'Union SG',_C:71,_F:22,_H:1,_I:7,_J:6,_A:[_P,_M]},{_G:'Nkrumah',_B:'Kelvin Nkrumah',_D:_AC,_C:70,_F:21,_H:1,_I:8,_J:6,_A:[_P,_L]},{_G:'Owusu P.O.',_B:'Prince Osei Owusu',_D:'CF Montreal',_C:75,_F:28,_H:1,_I:8,_J:6,_A:[_M]}]}
FORMATIONS={_A5:[{_A:[_P,_M,_L],_E:_AE},{_A:[_x,_g,_h,_y],_E:_j},{_A:[_U,_N,_V],_E:_Z},{_A:[_Q],_E:_a}],'4-3-3 (Attacking)':[{_A:[_P,_M,_L],_E:_AE},{_A:[_g,_O,_h],_E:_j},{_A:[_Y,_U,_V,_S],_E:_Z},{_A:[_Q],_E:_a}],'4-2-3-1 (Balanced)':[{_A:[_M],_E:_A6},{_A:[_P,_z,_L],_E:_A7},{_A:['LDM','RDM'],_E:_A8},{_A:[_Y,_U,_V,_S],_E:_Z},{_A:[_Q],_E:_a}],'4-4-2 (Classic)':[{_A:[_k,_l],_E:_A0},{_A:[_x,_g,_h,_y],_E:_j},{_A:[_Y,_U,_V,_S],_E:_Z},{_A:[_Q],_E:_a}],'3-5-2 (Defensive)':[{_A:[_k,_l],_E:_A0},{_A:[_o,_g,_O,_h,_t],_E:_j},{_A:[_U,_N,_V],_E:_Z},{_A:[_Q],_E:_a}],_AD:[{_A:[_k,_l],_E:_A0},{_A:[_o,_z,_O,'CAM2',_t],_E:'ATTACKING MIDFIELD'},{_A:[_U,_N,_V],_E:_Z},{_A:[_Q],_E:_a}],'4-1-4-1 (Possession)':[{_A:[_M],_E:_A6},{_A:[_x,_g,_h,_y],_E:_j},{_A:['CDM'],_E:_A8},{_A:[_Y,_U,_V,_S],_E:_Z},{_A:[_Q],_E:_a}],'5-3-2 (Counter)':[{_A:[_k,_l],_E:_A0},{_A:[_g,_O,_h],_E:_j},{_A:[_o,_U,_N,_V,_t],_E:_Z},{_A:[_Q],_E:_a}],'4-4-2 Diamond':[{_A:[_k,_l],_E:_A0},{_A:[_z],_E:_A7},{_A:[_g,_h],_E:'CENTRAL MID'},{_A:['CDM'],_E:_A8},{_A:[_Y,_U,_V,_S],_E:_Z},{_A:[_Q],_E:_a}],'3-4-1-2':[{_A:[_k,_l],_E:_A0},{_A:[_z],_E:_A7},{_A:[_x,_g,_h,_y],_E:_j},{_A:[_U,_N,_V],_E:_Z},{_A:[_Q],_E:_a}],'5-4-1':[{_A:[_M],_E:_A6},{_A:[_x,_g,_h,_y],_E:_j},{_A:[_o,_U,_N,_V,_t],_E:_Z},{_A:[_Q],_E:_a}],'4-3-3 (Defensive)':[{_A:[_P,_M,_L],_E:_AE},{_A:[_g,'CDM',_h],_E:_j},{_A:[_Y,_U,_V,_S],_E:_Z},{_A:[_Q],_E:_a}],'4-2-3-1 (Narrow)':[{_A:[_M],_E:_A6},{_A:['LCAM',_z,'RCAM'],_E:_A7},{_A:['LDM','RDM'],_E:_A8},{_A:[_Y,_U,_V,_S],_E:_Z},{_A:[_Q],_E:_a}]}
@st.cache_data(ttl=3600)
def get_free_openrouter_models():
	'Dynamically load OpenRouter free models'
	try:
		C=requests.get(OPENROUTER_MODELS_URL,headers={'User-Agent':'Ghana-BlackStars-AI/2.0','Accept':_AF},timeout=30);C.raise_for_status();E=C.json();A=[]
		for B in E.get('data',[]):
			D=B.get('pricing',{});F=float(D.get('prompt','0'));G=float(D.get('completion','0'))
			if F==0 and G==0:
				H=B.get('id','')
				if H:A.append(B)
		if not A:return[]
		A.sort(key=lambda x:x.get('id',''));return A
	except Exception as I:st.error(f"Error loading OpenRouter models: {str(I)}");return[]
def get_load_balanced_api_key(provider=_c):
	'Get load-balanced API key';H='count';G='idx';A=provider;C=[]
	try:
		if A.lower()==_c:
			try:
				D=st.secrets[_c]['api_key']
				if D and D.strip():C.append(D.strip())
			except:pass
			for I in range(1,11):
				try:
					D=st.secrets[_c][f"api_key{I}"]
					if D and D.strip():C.append(D.strip())
				except:pass
	except Exception:
		if A in st.session_state.api_keys and st.session_state.api_keys[A]:return st.session_state.api_keys[A]
		return''
	if not C:
		if A in st.session_state.api_keys and st.session_state.api_keys[A]:return st.session_state.api_keys[A]
		return''
	if len(C)==1:return C[0]
	if _AB not in st.session_state:st.session_state.api_usage={}
	if A not in st.session_state.api_usage:st.session_state.api_usage[A]={G:0,_A9:set(),H:{}}
	B=st.session_state.api_usage[A];E=[A for A in C if A not in B[_A9]]
	if not E:B[_A9].clear();E=C
	F=E[B[G]%len(E)];B[G]=(B[G]+1)%len(E);B[H][F]=B[H].get(F,0)+1;B['last_used']=time.time();B['selected_key']=F;return F
def mark_key_failed(provider,key):
	'Mark key as failed';A=provider
	if _AB in st.session_state and A in st.session_state.api_usage:st.session_state.api_usage[A][_A9].add(key)
def call_openrouter_api(messages,model,api_key):'Call OpenRouter API';B={'Authorization':f"Bearer {api_key}",'Content-Type':_AF,'HTTP-Referer':'https://github.com/rkarikari/stem/blackstars.html','X-Title':'Ghana Black Stars'};C={'model':model,'messages':messages,'stream':_A1,'max_tokens':2048,'temperature':.7};A=requests.post(OPENROUTER_API_URL,headers=B,json=C,timeout=60);A.raise_for_status();return A
def get_all_players():
	'Get all players from database';A=[]
	for B in PLAYERS.values():A.extend(B)
	return A
def get_players_for_position(position,formation=_w):
	'Get suitable players for a position - Enhanced mapping with formation-specific rules';D=formation;E=D==_AD if D else st.session_state.get(_AA)==_AD;H={_Q:[_Q],_N:[_f,_W],_U:[_f],_V:[_f],_Y:[_f],_S:[_f,_W],_o:[_R,_X,_W]if E else[_f],_t:[_R,_X,_W]if E else[_f,_W],'CDM':[_R],'LDM':[_R],'RDM':[_R],_O:[_R,_W],_g:[_R,_W],_h:[_R,_W],_z:[_R,_W],'CAM2':[_R,_W],'LCAM':[_R],'RCAM':[_R],_x:[_R,_X,_W],_y:[_R,_X,_W],_P:[_X],_L:[_X,_W],_M:[_X],_k:[_X],_l:[_X]};I=H.get(position,[_R]);A=[]
	for F in I:
		if F in PLAYERS:A.extend(PLAYERS[F])
	if st.session_state.squad_mode==_AG:A=[A for A in A if A[_B]in st.session_state.world_cup_squad]
	G=set();B=[]
	for C in A:
		if C[_B]not in G:G.add(C[_B]);B.append(C)
	B.sort(key=lambda x:x[_C],reverse=_K);return B
def calculate_stats(lineup):
	'Calculate team statistics';D=lineup
	if not D:return{_i:0,_p:0,_u:0,_m:0,_d:0,_e:0}
	A=list(D.values());E=round(sum(A[_C]for A in A)/len(A),1);F=round(sum(A[_F]for A in A)/len(A),1);G=sum(A[_H]for A in A);H=sum(A[_I]for A in A)/len(A);I=sum(A.get(_J,5)for A in A)/len(A);J=min(100,round(H*8+I*4+(20 if len(A)>=11 else 0)));B=[B for C in[_X]for B in PLAYERS[C]if B in A];C=[B for C in[_f]for B in PLAYERS[C]if B in A];K=round(sum(A[_C]for A in B)/len(B))if B else 0;L=round(sum(A[_C]for A in C)/len(C))if C else 0;return{_i:E,_p:F,_u:G,_m:J,_d:K,_e:L}
def auto_select_best_xi(formation):
	'Auto-select best XI for formation';B=formation;C={};G=FORMATIONS[B];D=0;E=set()
	for(H,I)in enumerate(G):
		for(J,F)in enumerate(I[_A]):
			K=f"{F}_{H}_{J}_{D}";D+=1;L=get_players_for_position(F,B)
			for A in L:
				if A[_B]not in E:C[K]=A;E.add(A[_B]);break
	return C
def calculate_formation_rankings():
	'Calculate power rankings for all formations';D=[]
	for E in FORMATIONS.keys():
		F=FORMATIONS[E];J=0;G=0;A=0;B=0;H=0
		for M in F:
			for I in M[_A]:
				G+=1;K=get_players_for_position(I,E)
				if K:
					C=K[0];J+=C[_C]
					if I in[_M,_k,_l,_P,_L]:A+=C[_C]
					elif I in[_N,_U,_V,_Y,_S,_o,_t]:B+=C[_C]
					else:H+=C[_C]
		L=J/G if G>0 else 0;N=min(A,B,H)/10 if all([A,B,H])else 0;O=L+N;D.append({_G:E,_A2:round(O,1),_i:round(L,1),_d:round(A/max(1,len([A for B in F for A in B[_A]if A in[_M,_k,_l,_P,_L]])),0),_e:round(B/max(1,len([A for B in F for A in B[_A]if A in[_N,_U,_V,_Y,_S,_o,_t]])),0)})
	D.sort(key=lambda x:x[_A2],reverse=_K);return D
def create_ultimate_team(api_key,model):
	'Create the ultimate World Cup winning team with AI optimization';I=calculate_formation_rankings();A=I[0][_G];D=auto_select_best_xi(A);B=calculate_stats(D);E=f"Formation: {A}\n\nPlayers Selected:\n";J=FORMATIONS[A];F=0
	for(K,G)in enumerate(J):
		E+=f"\n{G[_E]}:\n"
		for(L,H)in enumerate(G[_A]):
			M=f"{H}_{K}_{L}_{F}";F+=1;C=D.get(M)
			if C:E+=f"  - {H}: {C[_B]} ({C[_D]}) - Rating: {C[_C]}, Age: {C[_F]}, Form: {C[_I]}/10\n"
	E+=f"""

Team Statistics:
- Average Rating: {B[_i]}
- Average Age: {B[_p]} years
- Total Caps: {B[_u]}
- Chemistry: {B[_m]}%
- Attack Power: {B[_d]}
- Defense Power: {B[_e]}
""";N=f"""You are building Ghana's ultimate World Cup winning team. Analyze this lineup:

{E}

Provide a comprehensive analysis with:

1. **Formation Analysis**: Why this formation is optimal for Ghana's playing style and player strengths.

2. **Tactical Strengths**: 3-4 key advantages this lineup provides.

3. **Tactical Adjustments**: Specific in-game tactical tweaks to maximize performance.

4. **Substitution Strategy**: 
   - **If Winning (Protecting Lead)**: Which 3 substitutions to make and when
   - **If Drawing (Need Goal)**: Which 3 attacking substitutions to make
   - **If Losing (Desperate)**: Ultra-attacking formation and substitution changes
   - Include specific player names from the full squad

5. **World Cup Optimization**: 
   - Tournament rotation strategy to manage fitness
   - How to adapt tactics against different opponent styles (possession-based, counter-attacking, physical)
   - Set-piece strategies (corners, free-kicks)
   - Penalty shootout preparation

6. **Fine-Tuning Recommendations**: 
   - 2-3 specific training focuses
   - Player partnerships to develop
   - Alternative formation options for different game scenarios

Be specific with player names, tactical instructions, and match situations."""
	try:O=[{_n:_v,_b:N}];P=call_openrouter_api(O,model,api_key);Q=P.json();R=Q[_AH][0][_AI][_b];st.session_state.selected_formation=A;return A,D,R
	except Exception as S:st.session_state.selected_formation=A;return A,D,f"Error generating analysis: {str(S)}"
def create_pdf_export(formation,lineup,stats,ai_analysis=_w,rankings=_w):
	'Create comprehensive PDF export of lineup and analysis';s='#f9f9f9';r='ROWBACKGROUNDS';q='Normal';p='#fcd116';g=rankings;Z=ai_analysis;Y=formation;X='GRID';W='FONTNAME';V='CENTER';U='TEXTCOLOR';T='#d4af37';P=lineup;O='TOPPADDING';N='BOTTOMPADDING';M='ALIGN';I='Helvetica-Bold';E=stats;D='FONTSIZE';C='BACKGROUND';a=BytesIO();t=SimpleDocTemplate(a,pagesize=A4,rightMargin=30,leftMargin=30,topMargin=30,bottomMargin=30);A=[];J=getSampleStyleSheet();u=ParagraphStyle('CustomTitle',parent=J['Heading1'],fontSize=24,textColor=colors.HexColor(T),spaceAfter=30,alignment=TA_CENTER,fontName=I);F=ParagraphStyle('CustomHeading',parent=J['Heading2'],fontSize=16,textColor=colors.HexColor(p),spaceAfter=12,fontName=I);v=ParagraphStyle('SubHeading',parent=J['Heading3'],fontSize=12,textColor=colors.HexColor(T),spaceAfter=8,fontName=I);b=J[q];A.append(Paragraph('GHANA BLACK STARS',u));A.append(Paragraph(f"Tactical Lineup Report - {Y}",F));A.append(Paragraph(f"Generated: {datetime.now().strftime(_AJ)}",b));A.append(Spacer(1,.3*inch));A.append(Paragraph('Team Statistics',F));w=[['Metric','Value'],['Average Rating',f"{E[_i]}"],['Average Age',f"{E[_p]} years"],['Total Caps',f"{E[_u]}"],['Chemistry',f"{E[_m]}%"],['Attack Power',f"{E[_d]}"],['Defense Power',f"{E[_e]}"]];h=Table(w,colWidths=[3*inch,2*inch]);h.setStyle(TableStyle([(C,(0,0),(-1,0),colors.HexColor(T)),(U,(0,0),(-1,0),colors.black),(M,(0,0),(-1,-1),V),(W,(0,0),(-1,0),I),(D,(0,0),(-1,0),12),(N,(0,0),(-1,0),12),(O,(0,0),(-1,0),8),(C,(0,1),(-1,-1),colors.HexColor('#f5f5f5')),(X,(0,0),(-1,-1),1,colors.black),(D,(0,1),(-1,-1),11),(O,(0,1),(-1,-1),6),(N,(0,1),(-1,-1),6)]));A.append(h);A.append(Spacer(1,.4*inch));A.append(Paragraph(f"Starting XI Formation: {Y}",F));A.append(Spacer(1,.2*inch));i=FORMATIONS[Y];K=0
	for(c,Q)in enumerate(i):
		A.append(Paragraph(f"{Q[_E]}",v));R=[]
		for(d,G)in enumerate(Q[_A]):
			e=f"{G}_{c}_{d}_{K}";K+=1;B=P.get(e)
			if B:R.append([G,B[_B],B[_D],str(B[_C]),f"{B[_F]}y",f"{B[_H]} caps",f"Form: {B[_I]}/10"])
			else:R.append([G,'Not Selected',_q,_q,_q,_q,_q])
		if R:j=Table(R,colWidths=[.7*inch,1.6*inch,1.4*inch,.6*inch,.5*inch,.7*inch,.8*inch]);j.setStyle(TableStyle([(C,(0,0),(-1,-1),colors.white),(U,(0,0),(-1,-1),colors.black),(M,(0,0),(0,-1),V),(M,(1,0),(-1,-1),'LEFT'),(W,(0,0),(-1,-1),'Helvetica'),(D,(0,0),(-1,-1),9),(X,(0,0),(-1,-1),.5,colors.grey),('VALIGN',(0,0),(-1,-1),'MIDDLE'),(O,(0,0),(-1,-1),5),(N,(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]));A.append(j);A.append(Spacer(1,.2*inch))
	if P:
		A.append(PageBreak());A.append(Paragraph('Complete Squad Details',F));A.append(Spacer(1,.2*inch));x=list(P.values());x.sort(key=lambda x:x[_C],reverse=_K);f=[['#','Player Name','Club','Position','Rating','Age','Caps','Form']];K=0
		for(c,Q)in enumerate(i):
			for(d,G)in enumerate(Q[_A]):
				e=f"{G}_{c}_{d}_{K}";K+=1;B=P.get(e)
				if B:f.append([str(len(f)),B[_B],B[_D],G,str(B[_C]),str(B[_F]),str(B[_H]),f"{B[_I]}/10"])
		k=Table(f,colWidths=[.4*inch,1.6*inch,1.3*inch,.8*inch,.6*inch,.5*inch,.5*inch,.6*inch]);k.setStyle(TableStyle([(C,(0,0),(-1,0),colors.HexColor('#1a472a')),(U,(0,0),(-1,0),colors.HexColor(p)),(M,(0,0),(-1,-1),V),(W,(0,0),(-1,0),I),(D,(0,0),(-1,0),10),(C,(0,1),(-1,-1),colors.white),(X,(0,0),(-1,-1),.5,colors.grey),(D,(0,1),(-1,-1),9),(O,(0,0),(-1,-1),5),(N,(0,0),(-1,-1),5),(r,(0,1),(-1,-1),[colors.white,colors.HexColor(s)])]));A.append(k);A.append(Spacer(1,.3*inch))
	if g:
		A.append(PageBreak());A.append(Paragraph(_AM,F));A.append(Spacer(1,.2*inch));l=[['Rank','Formation','Score','Avg Rating','Attack','Defense']]
		for(S,L)in enumerate(g[:10]):y='1st'if S==0 else'2nd'if S==1 else'3rd'if S==2 else f"{S+1}th";l.append([y,L[_G],str(L[_A2]),str(L[_i]),str(L[_d]),str(L[_e])])
		m=Table(l,colWidths=[.7*inch,1.8*inch,.8*inch,1*inch,.8*inch,.8*inch]);m.setStyle(TableStyle([(C,(0,0),(-1,0),colors.HexColor(T)),(U,(0,0),(-1,0),colors.black),(M,(0,0),(-1,-1),V),(W,(0,0),(-1,0),I),(D,(0,0),(-1,0),10),(C,(0,1),(-1,-1),colors.white),(X,(0,0),(-1,-1),1,colors.grey),(D,(0,1),(-1,-1),9),(O,(0,0),(-1,-1),6),(N,(0,0),(-1,-1),6),(r,(0,1),(-1,-1),[colors.white,colors.HexColor(s)])]));A.append(m);A.append(Spacer(1,.3*inch))
	if Z and Z.strip():
		A.append(PageBreak());A.append(Paragraph('AI Tactical Analysis',F));A.append(Spacer(1,.2*inch));z=Z.split('\n')
		for n in z:
			if n.strip():
				H=n.strip().replace('**','<b>').replace('**','</b>')
				if H and(H[0].isdigit()or H.startswith(_q)or H.startswith('•')):A0=ParagraphStyle('ListItem',parent=b,leftIndent=20,spaceAfter=8);A.append(Paragraph(H,A0))
				else:A.append(Paragraph(H,b))
				A.append(Spacer(1,.1*inch))
	A.append(Spacer(1,.5*inch));o=ParagraphStyle('Footer',parent=J[q],fontSize=9,textColor=colors.grey,alignment=TA_CENTER);A.append(Paragraph('Ghana Black Stars - Tactical Analysis',o));A.append(Paragraph('Generated by RNK RadioSport Analysis System',o));t.build(A);a.seek(0);return a
def create_simple_text_export(formation,lineup,stats,ai_analysis=_w):
	'Create simple text export as fallback';F=ai_analysis;E=formation;B=stats;A=f"""
GHANA BLACK STARS - TACTICAL LINEUP REPORT
{_r*70}

Formation: {E}
Generated: {datetime.now().strftime(_AJ)}

TEAM STATISTICS
{_r*70}
Average Rating: {B[_i]}
Average Age: {B[_p]} years
Total Caps: {B[_u]}
Chemistry: {B[_m]}%
Attack Power: {B[_d]}
Defense Power: {B[_e]}

STARTING XI
{_r*70}
""";I=FORMATIONS[E];G=0
	for(J,H)in enumerate(I):
		A+=f"\n{H[_E]}\n{_q*70}\n"
		for(K,D)in enumerate(H[_A]):
			L=f"{D}_{J}_{K}_{G}";G+=1;C=lineup.get(L)
			if C:A+=f"{D:8} | {C[_B]:25} | {C[_D]:20} | {C[_C]} | {C[_F]}y | {C[_H]} caps\n"
			else:A+=f"{D:8} | Not Selected\n"
	if F:A+=f"""

AI TACTICAL ANALYSIS
{_r*70}
{F}
"""
	A+=f"\n\n{_r*70}\n";A+='Ghana Black Stars - Tactical Analysis\n';A+='Generated by RNK RadioSport Analysis System\n';return A
def get_formation_context():
	'Get context about current formation and lineup for chat'
	if not st.session_state.lineup:return'No lineup selected yet.'
	B=calculate_stats(st.session_state.lineup);E=st.session_state.get(_AA,_A5);C=f"""Current Ghana Black Stars Setup:
Formation: {E}
Players in lineup: {len(st.session_state.lineup)}/11

Team Statistics:
- Average Rating: {B[_i]}
- Average Age: {B[_p]} years
- Total Caps: {B[_u]}
- Chemistry: {B[_m]}%
- Attack Power: {B[_d]}
- Defense Power: {B[_e]}

Selected Players:
""";H=FORMATIONS[E];F=0
	for(I,G)in enumerate(H):
		C+=f"\n{G[_E]}:\n"
		for(J,D)in enumerate(G[_A]):
			K=f"{D}_{I}_{J}_{F}";F+=1;A=st.session_state.lineup.get(K)
			if A:C+=f"  - {D}: {A[_B]} ({A[_D]}) - Rating: {A[_C]}, Age: {A[_F]}, Caps: {A[_H]}, Form: {A[_I]}/10\n"
			else:C+=f"  - {D}: Empty\n"
	return C
def chat_with_ai(user_message,api_key,model):
	'Chat with AI about formations and tactics';B=user_message;D=get_formation_context();A=[{_n:'system',_b:'You are an expert football tactical analyst specializing in the Ghana Black Stars national team. \nYou provide insightful analysis on formations, player selections, tactical strategies, and match preparation. \nYou consider player chemistry, formation strengths/weaknesses, opposition tactics, and World Cup readiness.\nBe specific, analytical, and provide actionable recommendations. Reference specific players when relevant.'},{_n:_v,_b:f"Here's the current Ghana Black Stars lineup context:\n\n{D}"}]
	for E in st.session_state.chat_history[-6:]:A.append(E)
	A.append({_n:_v,_b:B})
	try:F=call_openrouter_api(A,model,api_key);G=F.json();C=G[_AH][0][_AI][_b];st.session_state.chat_history.append({_n:_v,_b:B});st.session_state.chat_history.append({_n:'assistant',_b:C});return C
	except Exception as H:return f"Error: {str(H)}"
def main():
	st.markdown("\n    <div style='text-align: center; padding: 2rem 0;'>\n        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>BLACKSTARS</h1>\n        <p style='font-size: 1.2rem; color: #fcd116;'>• Tactical Analysis •</p>\n    </div>\n    ",unsafe_allow_html=_K);A,B,C=st.tabs(['⚽ Lineup Builder','💬 Chat Analysis','📊 Formation Rankings'])
	with A:render_lineup_builder_tab()
	with B:render_chat_analysis_tab()
	with C:render_formation_rankings_tab()
	st.markdown(_T);st.markdown(f"""
    <div style='text-align: center; color: #d4af37; font-size: 0.85rem; padding: 1rem 0;'>
        <p><strong>RNK RadioSport</strong> • {APP_NAME} v{APP_VERSION}</p>
        <p style='font-size: 0.75rem; color: #aaa;'>Build winning formations</p>
    </div>
    """,unsafe_allow_html=_K)
def render_chat_analysis_tab():
	'Render the interactive chat analysis tab';L='🤔 AI is thinking...';K='⚠️ Please create a lineup first in the Lineup Builder tab';st.markdown('## 💬 Interactive Tactical Chat');st.markdown('Ask questions about formations, player selections, tactics, and get AI-powered insights!')
	with st.sidebar:
		st.markdown('### 💬 Chat Settings');A=get_load_balanced_api_key(_c)
		if not A:A=st.text_input(_AV,type='password',value=st.session_state.api_keys.get(_c,''),help='Required for chat feature',key='chat_api_key');st.session_state.api_keys[_c]=A
		if A:0
		else:st.warning('⚠️ Add API key to chat')
		st.markdown(_T);st.markdown('### 🎯 Quick Questions');M=['What are the strengths of this formation?','Which players should I change?','How does this lineup compare to top teams?','What tactics should we use?','Who are the key players in this formation?',"What's our attacking strategy?","How's our defensive setup?",'Suggest improvements for midfield control']
		with st.expander('💡 Suggested Questions',expanded=_A1):
			for C in M:
				if st.button(C,key=f"quick_{hash(C)}",use_container_width=_K):
					if not A:st.error('⚠️ Please add your API key first')
					elif not st.session_state.lineup:st.warning(K)
					else:
						with st.spinner(L):D=st.session_state.selected_model;N=chat_with_ai(C,A,D)
						st.rerun()
		st.markdown(_T)
		if st.button('🔄 Clear Chat History',use_container_width=_K):st.session_state.chat_history=[];st.success('Chat cleared!');st.rerun()
	E,F=st.columns([2,1])
	with E:
		st.markdown('### 💭 Conversation')
		if not st.session_state.chat_history:st.info('👋 Start a conversation! Ask me anything about your Ghana Black Stars lineup, formations, tactics, or player selections.')
		else:
			for(S,G)in enumerate(st.session_state.chat_history):
				if G[_n]==_v:st.markdown(f"\n                    <div style='background: rgba(212, 175, 55, 0.2); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #d4af37;'>\n                        <strong>You:</strong><br>{G[_b]}\n                    </div>\n                    ",unsafe_allow_html=_K)
				else:st.markdown(f"\n                    <div style='background: rgba(0, 107, 63, 0.2); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #006b3f;'>\n                        <strong>AI Analyst:</strong><br>{G[_b]}\n                    </div>\n                    ",unsafe_allow_html=_K)
	with F:
		st.markdown('### 📋 Current Context')
		if st.session_state.lineup:B=calculate_stats(st.session_state.lineup);O=st.session_state.get(_AA,_A5);st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px;'>
                <strong>Formation:</strong> {O}<br>
                <strong>Players:</strong> {len(st.session_state.lineup)}/11<br>
                <strong>Rating:</strong> {B[_i]}<br>
                <strong>Chemistry:</strong> {B[_m]}%<br>
                <strong>Attack:</strong> {B[_d]}<br>
                <strong>Defense:</strong> {B[_e]}
            </div>
            """,unsafe_allow_html=_K)
		else:st.info('No lineup selected. Go to Lineup Builder tab to create one!')
	st.markdown(_T);H=st.text_input('💬 Ask about tactics, formations, players...',placeholder="e.g., 'What are the weaknesses of this formation?' or 'Should I play more defensively?'",key='chat_text_input');E,F=st.columns([3,1])
	with E:P=st.button('📤 Send Message',type=_AK,use_container_width=_K)
	with F:Q=st.button('💾 Export Chat',use_container_width=_K)
	if P and H:
		if not A:st.error('⚠️ Please add your API key in the sidebar')
		elif not st.session_state.lineup:st.warning(K)
		else:
			with st.spinner(L):D=st.session_state.selected_model;N=chat_with_ai(H,A,D);st.rerun()
	if Q and st.session_state.chat_history:
		I=f"""GHANA BLACK STARS - TACTICAL CHAT ANALYSIS
Generated: {datetime.now().strftime(_AJ)}
{_r*70}

{get_formation_context()}

{_r*70}
CONVERSATION HISTORY
{_r*70}

"""
		for J in st.session_state.chat_history:R='YOU'if J[_n]==_v else'AI ANALYST';I+=f"""
{R}:
{J[_b]}

{_q*70}
"""
		st.download_button(label='⬇️ Download Chat Transcript',data=I,file_name=f"blackstars_chat_{datetime.now().strftime(_A3)}.txt",mime=_AL,use_container_width=_K)
def render_formation_rankings_tab():
	'Render formation rankings tab';st.markdown('## ⚡ Formation Power Rankings');st.markdown('AI-powered analysis based on player strengths and tactical effectiveness');C=calculate_formation_rankings();st.markdown('### 🏆 Top 3 Formations');E=st.columns(3);F=['🥇','🥈','🥉']
	for(B,(G,D))in enumerate(zip(E,F)):
		if B<len(C):
			A=C[B]
			with G:st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(252, 209, 22, 0.3)); 
                            padding: 1.5rem; border-radius: 15px; text-align: center; border: 2px solid #d4af37;'>
                    <div style='font-size: 3rem;'>{D}</div>
                    <div style='font-size: 1.3rem; font-weight: bold; color: #fcd116; margin: 0.5rem 0;'>{A[_G]}</div>
                    <div style='font-size: 2rem; font-weight: bold; color: #d4af37;'>{A[_A2]}</div>
                    <div style='font-size: 0.9rem; color: #ccc; margin-top: 0.5rem;'>
                        ATT: {A[_d]} | DEF: {A[_e]}
                    </div>
                </div>
                """,unsafe_allow_html=_K)
	st.markdown(_T);st.markdown('### 📊 Complete Rankings')
	for(B,A)in enumerate(C):
		D='🥇'if B==0 else'🥈'if B==1 else'🥉'if B==2 else f"{B+1}.";H,I,J,K,L=st.columns([1,3,1,1,1])
		with H:st.markdown(f"### {D}")
		with I:st.markdown(f"**{A[_G]}**")
		with J:st.metric('Score',A[_A2])
		with K:st.metric(_X,A[_d])
		with L:st.metric(_f,A[_e])
		if B<len(C)-1:st.markdown(_T)
def render_lineup_builder_tab():
	'Render the main lineup builder tab';i='collapsed'
	if _AA not in st.session_state:st.session_state.selected_formation=_A5
	with st.sidebar:
		st.markdown('### ⚙️ Configuration')
		if not st.session_state.openrouter_models:
			with st.spinner('Loading AI models...'):st.session_state.openrouter_models=get_free_openrouter_models()
		S=st.session_state.openrouter_models
		if not S:
			st.error('Could not load models.')
			if st.button('🔄 Refresh Models'):st.cache_data.clear();st.session_state.openrouter_models=[];st.rerun()
		else:
			X=[f"{A.get(_G,A['id'])}"for A in S];Y=[A['id']for A in S];D,E=st.columns([6,1])
			with D:
				try:Z=Y.index(st.session_state.selected_model)
				except ValueError:Z=0
				j=st.selectbox('🤖 AI Model (Free)',range(len(X)),format_func=lambda x:X[x],index=Z);st.session_state.selected_model=Y[j]
			with E:
				if st.button('🔄'):st.cache_data.clear();st.session_state.openrouter_models=[];st.rerun()
		st.markdown(_T);C=get_load_balanced_api_key(_c)
		if not C:C=st.text_input(_AV,type='password',value=st.session_state.api_keys.get(_c,''),help='Get free key at https://openrouter.ai/keys');st.session_state.api_keys[_c]=C
		if not C:st.warning('⚠️ Add API key for AI features')
		st.markdown(_T);st.markdown('### 📊 Formation Strategy');B=st.selectbox('Choose Formation',list(FORMATIONS.keys()),index=list(FORMATIONS.keys()).index(st.session_state.selected_formation)if st.session_state.selected_formation in FORMATIONS.keys()else 0,label_visibility=i)
		if B!=st.session_state.selected_formation:st.session_state.selected_formation=B
		st.markdown(_T);st.markdown('### 🏆 Squad Mode');a=st.selectbox('Squad Selection',[_A4,_AG],format_func=lambda x:f"Full Player Pool (64)"if x==_A4 else f"World Cup Squad ({len(st.session_state.world_cup_squad)}/26)",index=0 if st.session_state.squad_mode==_A4 else 1)
		if a!=st.session_state.squad_mode:st.session_state.squad_mode=a;st.rerun()
		if st.button('⚽ Build WC Squad',use_container_width=_K):st.session_state.show_wc_builder=_K
		st.markdown(_T);st.markdown('### 🎯 Quick Actions')
		if st.button('🏆 Build Ultimate Team',use_container_width=_K,type=_AK):
			if not C:st.error('⚠️ API key required for AI optimization')
			else:
				with st.spinner('🤖 AI creating World Cup winning team...'):b,k,T=create_ultimate_team(C,st.session_state.selected_model);st.session_state.selected_formation=b;st.session_state.lineup=k;st.session_state.ai_analysis=T
				st.success(f"✅ Ultimate team created with {b}!");st.rerun()
		if st.button('🔄 Clear Lineup',use_container_width=_K):st.session_state.lineup={};st.rerun()
		if st.button('🎲 Auto-Fill Best XI',use_container_width=_K):st.session_state.lineup=auto_select_best_xi(B);st.rerun()
		if st.button('🎲 Randomize',use_container_width=_K):
			c={};N=FORMATIONS[B];J=0;d=set()
			for(O,K)in enumerate(N):
				for(P,e)in enumerate(K[_A]):
					G=f"{e}_{O}_{P}_{J}";J+=1;U=get_players_for_position(e,B);f=[A for A in U if A[_B]not in d]
					if f:import random as l;A=l.choice(f[:10]);c[G]=A;d.add(A[_B])
			st.session_state.lineup=c;st.rerun()
		st.markdown(_T);st.markdown('### 📄 Export Options');H=len(st.session_state.lineup)==0
		if st.button('📄 Export PDF Report',use_container_width=_K,disabled=H):
			if not H:
				try:I=calculate_stats(st.session_state.lineup);m=calculate_formation_rankings();n=create_pdf_export(B,st.session_state.lineup,I,st.session_state.ai_analysis,m);st.download_button(label='⬇️ Download PDF Report',data=n,file_name=f"blackstars_lineup_{datetime.now().strftime(_A3)}.pdf",mime='application/pdf',use_container_width=_K);st.success('✅ PDF Report generated!')
				except Exception as L:st.error(f"PDF generation failed: {str(L)}");I=calculate_stats(st.session_state.lineup);V=create_simple_text_export(B,st.session_state.lineup,I,st.session_state.ai_analysis);st.download_button(label='⬇️ Download Text Report (Fallback)',data=V,file_name=f"blackstars_lineup_{datetime.now().strftime(_A3)}.txt",mime=_AL,use_container_width=_K)
		if st.button('📝 Export Text Report',use_container_width=_K,disabled=H):
			if not H:I=calculate_stats(st.session_state.lineup);V=create_simple_text_export(B,st.session_state.lineup,I,st.session_state.ai_analysis);st.download_button(label='⬇️ Download Text Report',data=V,file_name=f"blackstars_lineup_{datetime.now().strftime(_A3)}.txt",mime=_AL,use_container_width=_K);st.success('✅ Text Report ready!')
		if st.button('📊 Export JSON Data',use_container_width=_K,disabled=H):
			if not H:I=calculate_stats(st.session_state.lineup);o={'formation':B,'generated':datetime.now().isoformat(),'statistics':I,'lineup':{A:{'position':A.split('_')[0],'player':B}for(A,B)in st.session_state.lineup.items()},_AN:st.session_state.ai_analysis,_AP:st.session_state.squad_mode,_AO:st.session_state.world_cup_squad};p=json.dumps(o,indent=2);st.download_button(label='⬇️ Download JSON Data',data=p,file_name=f"blackstars_lineup_{datetime.now().strftime(_A3)}.json",mime=_AF,use_container_width=_K);st.success('✅ JSON Data ready!')
		if H:st.info('ℹ️ Add players to lineup to enable export')
		st.markdown(_T);F=calculate_stats(st.session_state.lineup);st.markdown('### 📊 Team Statistics');D,E=st.columns(2)
		with D:st.metric('⭐ Avg Rating',f"{F[_i]}");st.metric('👥 Avg Age',f"{F[_p]}");st.metric('🎖️ Total Caps',f"{F[_u]}")
		with E:st.metric('⚡ Chemistry',f"{F[_m]}%");st.metric('⚔️ Attack',f"{F[_d]}");st.metric('🛡️ Defense',f"{F[_e]}")
		st.markdown(_T);st.markdown('### 🧠 AI Analysis');q=len(st.session_state.lineup)>=11 and C
		if st.button('🤖 Analyze Lineup',type=_AK,disabled=not q,use_container_width=_K):
			with st.spinner('🧠 AI analyzing tactical setup...'):
				r=list(st.session_state.lineup.values());s=f"""Analyze this Ghana Black Stars lineup for 2026 World Cup qualifiers:

Formation: {B}
Team Statistics: Rating {F[_i]}, Age {F[_p]}, Chemistry {F[_m]}%

Players:
{chr(10).join([f"- {A[_B]} ({A[_D]}) - Position: {list(st.session_state.lineup.keys())[list(st.session_state.lineup.values()).index(A)].split('_')[0]}, Rating: {A[_C]}, Form: {A[_I]}/10"for A in r])}

Provide a detailed tactical analysis:
1. **Formation Strengths** (3-4 key points)
2. **Potential Weaknesses** (2-3 concerns)
3. **Key Player Partnerships** (describe 2-3 combinations)
4. **Tactical Recommendations** (specific adjustments)
5. **Match Readiness Score** (1-10 with justification)

Be specific, tactical, and actionable."""
				try:t=[{_n:_v,_b:s}];u=call_openrouter_api(t,st.session_state.selected_model,C);v=u.json();T=v[_AH][0][_AI][_b];st.session_state.ai_analysis=T;st.success('✅ Analysis complete!')
				except requests.RequestException as L:
					Q=f"Error: {str(L)}"
					if'401'in str(L):Q='⚠️ Invalid API Key. Please check your OpenRouter API key.';mark_key_failed(_c,C)
					elif'429'in str(L):Q='⚠️ Rate limit exceeded. Please try again in a moment.'
					st.error(Q);st.session_state.ai_analysis=Q
		if len(st.session_state.lineup)<11:st.info(f"ℹ️ Add {11-len(st.session_state.lineup)} more players for full analysis")
		elif not C:st.info('ℹ️ Add API key to enable AI analysis')
	if st.session_state.get('show_wc_builder',_A1):
		st.markdown(_T);st.markdown('## 🏆 Build Your 2026 World Cup Squad');st.markdown(f"**Select exactly 26 players** | Currently selected: {len(st.session_state.world_cup_squad)}/26");D,E,W=st.columns(3)
		with D:
			if st.button('✅ Save & Use Squad',use_container_width=_K):
				if len(st.session_state.world_cup_squad)==26:st.session_state.squad_mode=_AG;st.session_state.show_wc_builder=_A1;st.success('World Cup squad saved!');st.rerun()
				else:st.error(f"Please select exactly 26 players. Currently: {len(st.session_state.world_cup_squad)}")
		with E:
			if st.button('⚡ Auto-Select Best 26',use_container_width=_K):g=get_all_players();g.sort(key=lambda x:x[_C],reverse=_K);st.session_state.world_cup_squad=[A[_B]for A in g[:26]];st.rerun()
		with W:
			if st.button('❌ Cancel',use_container_width=_K):st.session_state.show_wc_builder=_A1;st.rerun()
		st.markdown(_T);w=[('Goalkeepers',_Q),('Defenders',_f),('Midfielders',_R),('Versatile',_W),('Attackers',_X)]
		for(x,h)in w:
			if h not in PLAYERS:continue
			st.markdown(f"### {x}")
			for A in PLAYERS[h]:
				y=A[_B]in st.session_state.world_cup_squad;D,E,W=st.columns([1,4,1])
				with D:
					if st.checkbox(f"Select {A[_B]}",value=y,key=f"wc_{A[_B]}",label_visibility=i):
						if A[_B]not in st.session_state.world_cup_squad:
							if len(st.session_state.world_cup_squad)<26:st.session_state.world_cup_squad.append(A[_B]);st.rerun()
							else:st.warning('Maximum 26 players!')
					elif A[_B]in st.session_state.world_cup_squad:st.session_state.world_cup_squad.remove(A[_B]);st.rerun()
				with E:st.markdown(f"**{A[_B]}** • {A[_D]} • {A[_F]}y • {A[_H]} caps")
				with W:st.markdown(f"**{A[_C]}**")
		st.markdown(_T);return
	st.markdown('### 📋 Team Lineup')
	if st.session_state.lineup:
		N=FORMATIONS[B];J=0
		for(O,K)in enumerate(N):
			st.markdown(f"<div class='line-label'>{K[_E]}</div>",unsafe_allow_html=_K);z=len(K[_A]);A0=st.columns(z)
			for(P,M)in enumerate(K[_A]):
				G=f"{M}_{O}_{P}_{J}";J+=1
				with A0[P]:
					A=st.session_state.lineup.get(G)
					if A:
						st.markdown(f"""
                        <div class='player-card' style='background: linear-gradient(135deg, #ce1126, #8b0000); margin-bottom: 0.8rem;'>
                            <div style='text-align: center;'>
                                <div style='background: linear-gradient(135deg, #d4af37, #fcd116); color: #000; padding: 0.5rem; border-radius: 8px; font-weight: bold; font-size: 1.3rem; margin-bottom: 0.5rem;'>
                                    {A[_C]}
                                </div>
                                <div style='font-weight: bold; color: #fcd116; font-size: 1rem; margin-bottom: 0.3rem;'>{M}</div>
                                <div style='font-weight: bold; color: #fff; font-size: 0.9rem; margin-bottom: 0.3rem;'>{A[_G]}</div>
                                <div style='font-size: 0.75rem; color: #d4af37;'>{A[_D]}</div>
                                <div style='font-size: 0.7rem; color: #ccc; margin-top: 0.3rem;'>{A[_F]}y • {A[_H]} caps • Form {A[_I]}/10</div>
                            </div>
                        </div>
                        """,unsafe_allow_html=_K)
						if st.button(f"🔄 Change",key=f"change_{G}",use_container_width=_K):del st.session_state.lineup[G];st.rerun()
					else:
						st.markdown(f"""
                        <div class='player-card' style='background: linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(252, 209, 22, 0.3)); border: 2px dashed #fcd116; margin-bottom: 0.8rem;'>
                            <div style='text-align: center; padding: 1rem;'>
                                <div style='font-size: 1.2rem; font-weight: bold; color: #fcd116;'>{M}</div>
                                <div style='font-size: 0.8rem; color: #aaa; margin-top: 0.3rem;'>Empty</div>
                            </div>
                        </div>
                        """,unsafe_allow_html=_K)
						with st.expander(f"➕ Select {M}",expanded=_A1):
							U=get_players_for_position(M,B)
							for(A1,R)in enumerate(U[:15]):
								A2=f"select_{G}_{A1}";D,E=st.columns([3,1])
								with D:
									if st.button(f"{R[_B]} • {R[_D]}",key=A2,use_container_width=_K):st.session_state.lineup[G]=R;st.rerun()
								with E:st.markdown(f"<div style='text-align: center; background: #d4af37; color: #000; padding: 0.3rem; border-radius: 5px; font-weight: bold;'>{R[_C]}</div>",unsafe_allow_html=_K)
			if O<len(N)-1:st.markdown("<div style='margin: 1.5rem 0;'></div>",unsafe_allow_html=_K)
	else:st.info('🎯 Start building your lineup using the Auto-Fill button in the sidebar')
	if st.session_state.ai_analysis:
		st.markdown(_T);st.markdown('## 🧠 AI Tactical Analysis Report');st.markdown(f"""
        <div class='ai-analysis-box'>
            <div style='font-size: 1.15rem; line-height: 2; color: #ffffff; font-weight: 400;'>
                {st.session_state.ai_analysis.replace(chr(10),"<br>")}
            </div>
        </div>
        """,unsafe_allow_html=_K)
		if st.button('🔄 Clear Analysis',use_container_width=_K):st.session_state.ai_analysis=_w;st.rerun()
if __name__=='__main__':main()