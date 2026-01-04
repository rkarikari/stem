"""
Ghana Black Stars AI Lineup Builder - Streamlit Version with OpenRouter AI Integration
Author: RNK RadioSport
Version: 2.0.0 - Complete Features & Full Player Database
"""

import streamlit as st
import json
import requests
from datetime import datetime
import time

# Page Configuration
st.set_page_config(
    page_title="Black Stars Analysis",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a472a 0%, #0d2818 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #d4af37 0%, #fcd116 100%);
        color: #1a472a;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(212, 175, 55, 0.5);
    }
    .player-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    .player-card:hover {
        transform: translateX(5px);
        border-color: #d4af37;
    }
    .ai-analysis-box {
        background: linear-gradient(135deg, rgba(26, 71, 42, 0.95), rgba(13, 40, 24, 0.95));
        border: 3px solid #fcd116;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    h1, h2, h3 {
        color: #fcd116;
    }
    .stSelectbox label, .stTextInput label {
        color: #d4af37 !important;
        font-weight: bold;
    }
    .pitch-container {
        background: linear-gradient(180deg, #2d5016 0%, #1a472a 50%, #2d5016 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        border: 3px solid #fcd116;
        min-height: 600px;
    }
    .formation-line {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    .position-slot {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 120px;
    }
    .line-label {
        text-align: center;
        color: #fcd116;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODELS_URL = 'https://openrouter.ai/api/v1/models'

# Initialize Session State
if 'lineup' not in st.session_state:
    st.session_state.lineup = {}
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'openrouter': ''}
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {}
if 'openrouter_models' not in st.session_state:
    st.session_state.openrouter_models = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'mistralai/mistral-7b-instruct'
if 'world_cup_squad' not in st.session_state:
    st.session_state.world_cup_squad = []
if 'squad_mode' not in st.session_state:
    st.session_state.squad_mode = 'full'

# Complete Player Database - All 64 Players
PLAYERS = {
    'GK': [
        {'name': 'Ati-Zigi', 'fullName': 'Lawrence Ati-Zigi', 'club': 'St. Gallen', 'rating': 78, 'age': 28, 'caps': 18, 'form': 8, 'versatility': 1},
        {'name': 'Asare', 'fullName': 'Benjamin Asare', 'club': 'Hearts of Oak', 'rating': 76, 'age': 25, 'caps': 8, 'form': 9, 'versatility': 1},
        {'name': 'Nurudeen', 'fullName': 'Abdul Manaf Nurudeen', 'club': 'KAS Eupen', 'rating': 74, 'age': 26, 'caps': 6, 'form': 7, 'versatility': 1},
        {'name': 'Anang', 'fullName': 'Joseph Anang', 'club': "St Patrick's", 'rating': 74, 'age': 24, 'caps': 3, 'form': 7, 'versatility': 1},
        {'name': 'Wollacott', 'fullName': 'Joseph Wollacott', 'club': 'Hibernian', 'rating': 73, 'age': 29, 'caps': 9, 'form': 7, 'versatility': 1}
    ],
    'DEF': [
        {'name': 'Salisu', 'fullName': 'Mohammed Salisu', 'club': 'Monaco', 'rating': 82, 'age': 26, 'caps': 22, 'form': 8, 'versatility': 7, 'positions': ['CB', 'LB']},
        {'name': 'Djiku', 'fullName': 'Alexander Djiku', 'club': 'Spartak Moscow', 'rating': 81, 'age': 31, 'caps': 37, 'form': 8, 'versatility': 7, 'positions': ['CB', 'RB']},
        {'name': 'Lamptey', 'fullName': 'Tariq Lamptey', 'club': 'Brighton', 'rating': 80, 'age': 24, 'caps': 15, 'form': 8, 'versatility': 6, 'positions': ['RB', 'RW']},
        {'name': 'Seidu', 'fullName': 'Alidu Seidu', 'club': 'Stade Rennes', 'rating': 78, 'age': 25, 'caps': 22, 'form': 8, 'versatility': 7, 'positions': ['RB', 'CB']},
        {'name': 'Kohn', 'fullName': 'Derrick Kohn', 'club': 'Union Berlin', 'rating': 77, 'age': 26, 'caps': 10, 'form': 8, 'versatility': 6, 'positions': ['LB', 'LWB']},
        {'name': 'Mensah', 'fullName': 'Gideon Mensah', 'club': 'AJ Auxerre', 'rating': 76, 'age': 27, 'caps': 18, 'form': 7, 'versatility': 5, 'positions': ['LB']},
        {'name': 'Ambrosius', 'fullName': 'Stephan Ambrosius', 'club': 'St. Gallen', 'rating': 77, 'age': 26, 'caps': 10, 'form': 7, 'versatility': 6, 'positions': ['CB']},
        {'name': 'Mumin', 'fullName': 'Abdul Mumin', 'club': 'Rayo Vallecano', 'rating': 76, 'age': 27, 'caps': 15, 'form': 7, 'versatility': 6, 'positions': ['CB']},
        {'name': 'Adjei', 'fullName': 'Nathaniel Adjei', 'club': 'FC Lorient', 'rating': 75, 'age': 25, 'caps': 5, 'form': 7, 'versatility': 5, 'positions': ['CB']},
        {'name': 'Opoku J.', 'fullName': 'Jerome Opoku', 'club': 'Istanbul Basaksehir', 'rating': 75, 'age': 26, 'caps': 11, 'form': 7, 'versatility': 6, 'positions': ['CB']},
        {'name': 'Annan', 'fullName': 'Ebenezer Annan', 'club': 'AS Saint-Etienne', 'rating': 74, 'age': 24, 'caps': 7, 'form': 7, 'versatility': 5, 'positions': ['CB']},
        {'name': 'Simpson', 'fullName': 'Razak Simpson', 'club': 'Nations FC', 'rating': 73, 'age': 24, 'caps': 6, 'form': 8, 'versatility': 6, 'positions': ['RB', 'CB']},
        {'name': 'Adjetey', 'fullName': 'Jonas Adjetey', 'club': 'FC Basel', 'rating': 74, 'age': 25, 'caps': 7, 'form': 7, 'versatility': 6, 'positions': ['CB']},
        {'name': 'Afful', 'fullName': 'Isaac Afful', 'club': 'Samartex', 'rating': 71, 'age': 23, 'caps': 3, 'form': 7, 'versatility': 5, 'positions': ['RB']},
        {'name': 'Schindler', 'fullName': 'Kingsley Schindler', 'club': 'Hannover 96', 'rating': 73, 'age': 32, 'caps': 9, 'form': 6, 'versatility': 6, 'positions': ['RB', 'RW']}
    ],
    'MID': [
        {'name': 'Partey', 'fullName': 'Thomas Partey', 'club': 'Arsenal', 'rating': 84, 'age': 32, 'caps': 58, 'form': 8, 'versatility': 7, 'positions': ['DM', 'CM']},
        {'name': 'Kudus', 'fullName': 'Mohammed Kudus', 'club': 'West Ham', 'rating': 84, 'age': 25, 'caps': 32, 'form': 9, 'versatility': 9, 'positions': ['AM', 'RW', 'ST', 'CM']},
        {'name': 'Ashimeru', 'fullName': 'Majeed Ashimeru', 'club': 'Anderlecht', 'rating': 77, 'age': 28, 'caps': 24, 'form': 7, 'versatility': 6, 'positions': ['CM', 'DM']},
        {'name': 'Samed', 'fullName': 'Salis Abdul Samed', 'club': 'RC Lens', 'rating': 78, 'age': 25, 'caps': 17, 'form': 8, 'versatility': 6, 'positions': ['DM', 'CM']},
        {'name': 'E. Owusu', 'fullName': 'Elisha Owusu', 'club': 'AJ Auxerre', 'rating': 76, 'age': 28, 'caps': 20, 'form': 7, 'versatility': 6, 'positions': ['DM', 'CM']},
        {'name': 'Sulemana I.', 'fullName': 'Ibrahim Sulemana', 'club': 'Atalanta', 'rating': 76, 'age': 22, 'caps': 8, 'form': 7, 'versatility': 6, 'positions': ['CM']},
        {'name': 'Diomande', 'fullName': 'Mohamed Diomande', 'club': 'Rangers', 'rating': 74, 'age': 23, 'caps': 5, 'form': 8, 'versatility': 7, 'positions': ['CM', 'DM']},
        {'name': 'Sibo', 'fullName': 'Kwasi Sibo', 'club': 'Hearts of Oak', 'rating': 73, 'age': 23, 'caps': 6, 'form': 8, 'versatility': 6, 'positions': ['CM', 'DM']},
        {'name': 'Francis', 'fullName': 'Abu Francis', 'club': 'Cercle Brugge', 'rating': 75, 'age': 23, 'caps': 10, 'form': 7, 'versatility': 6, 'positions': ['CM', 'AM']},
        {'name': 'Addo E.', 'fullName': 'Edmund Addo', 'club': 'Sheriff Tiraspol', 'rating': 73, 'age': 25, 'caps': 12, 'form': 7, 'versatility': 5, 'positions': ['DM']},
        {'name': 'Mamudu', 'fullName': 'Kamaradini Mamudu', 'club': 'Medeama SC', 'rating': 72, 'age': 23, 'caps': 5, 'form': 7, 'versatility': 6, 'positions': ['CM']},
        {'name': 'Baidoo', 'fullName': 'Michael Baidoo', 'club': 'Elfsborg', 'rating': 73, 'age': 26, 'caps': 5, 'form': 7, 'versatility': 6, 'positions': ['CM', 'AM']},
        {'name': 'Antwi', 'fullName': 'Emmanuel Antwi', 'club': 'FK Pribram', 'rating': 71, 'age': 24, 'caps': 4, 'form': 7, 'versatility': 6, 'positions': ['CM']},
        {'name': 'P. Owusu', 'fullName': 'Prince Owusu', 'club': 'Medeama SC', 'rating': 70, 'age': 21, 'caps': 3, 'form': 8, 'versatility': 6, 'positions': ['CM']}
    ],
    'VERSATILE': [
        {'name': 'Yirenkyi', 'fullName': 'Caleb Yirenkyi', 'club': 'Nordsjaelland', 'rating': 73, 'age': 20, 'caps': 4, 'form': 8, 'versatility': 8, 'positions': ['CM', 'AM', 'RB', 'RW']}
    ],
    'ATT': [
        {'name': 'I. Williams', 'fullName': 'Inaki Williams', 'club': 'Athletic Bilbao', 'rating': 82, 'age': 31, 'caps': 24, 'form': 9, 'versatility': 7, 'positions': ['ST', 'RW']},
        {'name': 'Semenyo', 'fullName': 'Antoine Semenyo', 'club': 'Bournemouth', 'rating': 81, 'age': 26, 'caps': 18, 'form': 9, 'versatility': 7, 'positions': ['ST', 'RW']},
        {'name': 'J. Ayew', 'fullName': 'Jordan Ayew', 'club': 'Leicester', 'rating': 79, 'age': 34, 'caps': 118, 'form': 8, 'versatility': 8, 'positions': ['ST', 'RW', 'LW']},
        {'name': 'Fatawu', 'fullName': 'Abdul Fatawu Issahaku', 'club': 'Leicester', 'rating': 79, 'age': 21, 'caps': 23, 'form': 8, 'versatility': 7, 'positions': ['RW', 'LW']},
        {'name': 'Thomas-Asante', 'fullName': 'Brandon Thomas-Asante', 'club': 'Coventry City', 'rating': 78, 'age': 27, 'caps': 9, 'form': 9, 'versatility': 6, 'positions': ['ST', 'RW']},
        {'name': 'Sulemana K.', 'fullName': 'Kamaldeen Sulemana', 'club': 'Southampton', 'rating': 78, 'age': 23, 'caps': 18, 'form': 7, 'versatility': 8, 'positions': ['LW', 'RW', 'ST']},
        {'name': 'Nuamah', 'fullName': 'Ernest Nuamah', 'club': 'Lyon', 'rating': 77, 'age': 22, 'caps': 12, 'form': 8, 'versatility': 7, 'positions': ['RW', 'LW']},
        {'name': 'Paintsil', 'fullName': 'Joseph Paintsil', 'club': 'LA Galaxy', 'rating': 77, 'age': 28, 'caps': 12, 'form': 8, 'versatility': 7, 'positions': ['RW', 'LW']},
        {'name': 'Bonsu Baah', 'fullName': 'Christopher Bonsu Baah', 'club': 'KRC Genk', 'rating': 75, 'age': 23, 'caps': 8, 'form': 8, 'versatility': 7, 'positions': ['RW', 'LW']},
        {'name': 'Ibrahim Osman', 'fullName': 'Ibrahim Osman', 'club': 'Feyenoord', 'rating': 75, 'age': 20, 'caps': 7, 'form': 8, 'versatility': 7, 'positions': ['RW', 'LW', 'AM']},
        {'name': 'Adu Kwabena', 'fullName': 'Prince Kwabena Adu', 'club': 'Viktoria Plzen', 'rating': 75, 'age': 22, 'caps': 2, 'form': 8, 'versatility': 7, 'positions': ['ST', 'RW', 'LW']},
        {'name': 'Afriyie', 'fullName': 'Jerry Afriyie', 'club': 'RAAL La Louviere', 'rating': 73, 'age': 20, 'caps': 3, 'form': 8, 'versatility': 6, 'positions': ['ST']},
        {'name': 'Afena-Gyan', 'fullName': 'Felix Afena-Gyan', 'club': 'Cremonese', 'rating': 73, 'age': 21, 'caps': 5, 'form': 7, 'versatility': 6, 'positions': ['ST']},
        {'name': 'Opoku', 'fullName': 'Kwame Opoku', 'club': 'Asante Kotoko', 'rating': 72, 'age': 23, 'caps': 2, 'form': 7, 'versatility': 5, 'positions': ['ST']},
        {'name': 'Fuseini', 'fullName': 'Mohammed Fuseini', 'club': 'Union SG', 'rating': 71, 'age': 22, 'caps': 1, 'form': 7, 'versatility': 6, 'positions': ['LW', 'ST']},
        {'name': 'Nkrumah', 'fullName': 'Kelvin Nkrumah', 'club': 'Medeama SC', 'rating': 70, 'age': 21, 'caps': 1, 'form': 8, 'versatility': 6, 'positions': ['LW', 'RW']},
        {'name': 'Owusu P.O.', 'fullName': 'Prince Osei Owusu', 'club': 'CF Montreal', 'rating': 75, 'age': 28, 'caps': 1, 'form': 8, 'versatility': 6, 'positions': ['ST']}
    ]
}

# Complete Formation Database - All 12 Formations from HTML
FORMATIONS = {
    '4-3-3 (Attacking)': [
        {'positions': ['LW', 'ST', 'RW'], 'label': 'ATTACK'},
        {'positions': ['LCM', 'CM', 'RCM'], 'label': 'MIDFIELD'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-2-3-1 (Balanced)': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LW', 'CAM', 'RW'], 'label': 'ATTACKING MID'},
        {'positions': ['LDM', 'RDM'], 'label': 'DEFENSIVE MID'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-4-2 (Classic)': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '3-5-2 (Defensive)': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['LWB', 'LCM', 'CM', 'RCM', 'RWB'], 'label': 'MIDFIELD'},
        {'positions': ['LCB', 'CB', 'RCB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-1-4-1 (Possession)': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['CDM'], 'label': 'DEFENSIVE MID'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '3-4-3 (Ultra Attack)': [
        {'positions': ['LW', 'ST', 'RW'], 'label': 'ATTACK'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LCB', 'CB', 'RCB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '5-3-2 (Counter)': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['LCM', 'CM', 'RCM'], 'label': 'MIDFIELD'},
        {'positions': ['LWB', 'LCB', 'CB', 'RCB', 'RWB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-4-2 Diamond': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['CAM'], 'label': 'ATTACKING MID'},
        {'positions': ['LCM', 'RCM'], 'label': 'CENTRAL MID'},
        {'positions': ['CDM'], 'label': 'DEFENSIVE MID'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '3-4-1-2': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['CAM'], 'label': 'ATTACKING MID'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LCB', 'CB', 'RCB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '5-4-1': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LWB', 'LCB', 'CB', 'RCB', 'RWB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-3-3 (Defensive)': [
        {'positions': ['LW', 'ST', 'RW'], 'label': 'ATTACK'},
        {'positions': ['LCM', 'CDM', 'RCM'], 'label': 'MIDFIELD'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-2-3-1 (Narrow)': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LCAM', 'CAM', 'RCAM'], 'label': 'ATTACKING MID'},
        {'positions': ['LDM', 'RDM'], 'label': 'DEFENSIVE MID'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ]
}

@st.cache_data(ttl=3600)
def get_free_openrouter_models():
    """Dynamically load OpenRouter free models"""
    try:
        response = requests.get(
            OPENROUTER_MODELS_URL,
            headers={
                'User-Agent': 'Ghana-BlackStars-AI/2.0',
                'Accept': 'application/json'
            },
            timeout=30
        )
        response.raise_for_status()
        models_data = response.json()
        
        free_models = []
        for model in models_data.get('data', []):
            pricing = model.get('pricing', {})
            prompt_price = float(pricing.get('prompt', '0'))
            completion_price = float(pricing.get('completion', '0'))
            
            if prompt_price == 0 and completion_price == 0:
                model_id = model.get('id', '')
                if model_id:
                    free_models.append(model)
        
        if not free_models:
            return []
        
        free_models.sort(key=lambda x: x.get('id', ''))
        return free_models
    
    except Exception as e:
        st.error(f"Error loading OpenRouter models: {str(e)}")
        return []

def get_load_balanced_api_key(provider='openrouter'):
    """Get load-balanced API key"""
    keys = []
    
    try:
        if provider.lower() == 'openrouter':
            try:
                key = st.secrets['openrouter']['api_key']
                if key and key.strip():
                    keys.append(key.strip())
            except:
                pass
            
            for i in range(1, 11):
                try:
                    key = st.secrets['openrouter'][f'api_key{i}']
                    if key and key.strip():
                        keys.append(key.strip())
                except:
                    pass
    
    except Exception:
        if provider in st.session_state.api_keys and st.session_state.api_keys[provider]:
            return st.session_state.api_keys[provider]
        return ''
    
    if not keys:
        if provider in st.session_state.api_keys and st.session_state.api_keys[provider]:
            return st.session_state.api_keys[provider]
        return ''
    
    if len(keys) == 1:
        return keys[0]
    
    # Load balancing
    if 'api_usage' not in st.session_state:
        st.session_state.api_usage = {}
    
    if provider not in st.session_state.api_usage:
        st.session_state.api_usage[provider] = {
            'idx': 0,
            'failed': set(),
            'count': {}
        }
    
    usage = st.session_state.api_usage[provider]
    valid_keys = [k for k in keys if k not in usage['failed']]
    
    if not valid_keys:
        usage['failed'].clear()
        valid_keys = keys
    
    selected = valid_keys[usage['idx'] % len(valid_keys)]
    usage['idx'] = (usage['idx'] + 1) % len(valid_keys)
    usage['count'][selected] = usage['count'].get(selected, 0) + 1
    usage['last_used'] = time.time()
    usage['selected_key'] = selected
    
    return selected

def mark_key_failed(provider, key):
    """Mark key as failed"""
    if 'api_usage' in st.session_state and provider in st.session_state.api_usage:
        st.session_state.api_usage[provider]['failed'].add(key)

def call_openrouter_api(messages, model, api_key):
    """Call OpenRouter API"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/rkarikari/stem/blackstars.html',
        'X-Title': 'Ghana Black Stars'
    }
    
    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'max_tokens': 2048,
        'temperature': 0.7
    }
    
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response

def get_all_players():
    """Get all players from database"""
    all_players = []
    for category in PLAYERS.values():
        all_players.extend(category)
    return all_players

def get_players_for_position(position):
    """Get suitable players for a position - Enhanced mapping"""
    position_map = {
        'GK': ['GK'],
        'CB': ['DEF', 'VERSATILE'], 'LCB': ['DEF'], 'RCB': ['DEF'],
        'LB': ['DEF'], 'RB': ['DEF', 'VERSATILE'],
        'LWB': ['DEF'], 'RWB': ['DEF', 'VERSATILE'],
        'CDM': ['MID'], 'LDM': ['MID'], 'RDM': ['MID'],
        'CM': ['MID', 'VERSATILE'], 'LCM': ['MID', 'VERSATILE'], 'RCM': ['MID', 'VERSATILE'],
        'CAM': ['MID', 'VERSATILE'], 'LCAM': ['MID'], 'RCAM': ['MID'],
        'LM': ['MID', 'ATT', 'VERSATILE'], 'RM': ['MID', 'ATT', 'VERSATILE'],
        'LW': ['ATT'], 'RW': ['ATT', 'VERSATILE'],
        'ST': ['ATT'], 'LST': ['ATT'], 'RST': ['ATT']
    }
    
    categories = position_map.get(position, ['MID'])
    all_players = []
    
    for cat in categories:
        if cat in PLAYERS:
            all_players.extend(PLAYERS[cat])
    
    # Filter by world cup squad if in WC mode
    if st.session_state.squad_mode == 'worldcup':
        all_players = [p for p in all_players if p['fullName'] in st.session_state.world_cup_squad]
    
    # Remove duplicates
    seen = set()
    unique_players = []
    for player in all_players:
        if player['fullName'] not in seen:
            seen.add(player['fullName'])
            unique_players.append(player)
    
    unique_players.sort(key=lambda x: x['rating'], reverse=True)
    return unique_players

def calculate_stats(lineup):
    """Calculate team statistics"""
    if not lineup:
        return {'avg_rating': 0, 'avg_age': 0, 'total_caps': 0, 'chemistry': 0, 'attack': 0, 'defense': 0}
    
    players = list(lineup.values())
    avg_rating = round(sum(p['rating'] for p in players) / len(players), 1)
    avg_age = round(sum(p['age'] for p in players) / len(players), 1)
    total_caps = sum(p['caps'] for p in players)
    
    # Chemistry calculation
    avg_form = sum(p['form'] for p in players) / len(players)
    avg_versatility = sum(p.get('versatility', 5) for p in players) / len(players)
    chemistry = min(100, round((avg_form * 8) + (avg_versatility * 4) + (20 if len(players) >= 11 else 0)))
    
    # Attack and Defense scores
    attackers = [p for cat in ['ATT'] for p in PLAYERS[cat] if p in players]
    defenders = [p for cat in ['DEF'] for p in PLAYERS[cat] if p in players]
    
    attack = round(sum(p['rating'] for p in attackers) / len(attackers)) if attackers else 0
    defense = round(sum(p['rating'] for p in defenders) / len(defenders)) if defenders else 0
    
    return {
        'avg_rating': avg_rating,
        'avg_age': avg_age,
        'total_caps': total_caps,
        'chemistry': chemistry,
        'attack': attack,
        'defense': defense
    }

def auto_select_best_xi(formation):
    """Auto-select best XI for formation"""
    lineup = {}
    formation_data = FORMATIONS[formation]
    slot_counter = 0
    used_players = set()
    
    for line_idx, line in enumerate(formation_data):
        for pos_idx, pos in enumerate(line['positions']):
            slot_id = f"{pos}_{line_idx}_{pos_idx}_{slot_counter}"
            slot_counter += 1
            available = get_players_for_position(pos)
            
            for player in available:
                if player['fullName'] not in used_players:
                    lineup[slot_id] = player
                    used_players.add(player['fullName'])
                    break
    
    return lineup

def calculate_formation_rankings():
    """Calculate power rankings for all formations"""
    rankings = []
    
    for formation_name in FORMATIONS.keys():
        formation = FORMATIONS[formation_name]
        total_power = 0
        position_count = 0
        attack_power = 0
        defense_power = 0
        mid_power = 0
        
        for line in formation:
            for pos in line['positions']:
                position_count += 1
                available = get_players_for_position(pos)
                
                if available:
                    top_player = available[0]
                    total_power += top_player['rating']
                    
                    if pos in ['ST', 'LST', 'RST', 'LW', 'RW']:
                        attack_power += top_player['rating']
                    elif pos in ['CB', 'LCB', 'RCB', 'LB', 'RB', 'LWB', 'RWB']:
                        defense_power += top_player['rating']
                    else:
                        mid_power += top_player['rating']
        
        avg_rating = total_power / position_count if position_count > 0 else 0
        balance = min(attack_power, defense_power, mid_power) / 10 if all([attack_power, defense_power, mid_power]) else 0
        
        score = avg_rating + balance
        
        rankings.append({
            'name': formation_name,
            'score': round(score, 1),
            'avg_rating': round(avg_rating, 1),
            'attack': round(attack_power / max(1, len([p for l in formation for p in l['positions'] if p in ['ST', 'LST', 'RST', 'LW', 'RW']])), 0),
            'defense': round(defense_power / max(1, len([p for l in formation for p in l['positions'] if p in ['CB', 'LCB', 'RCB', 'LB', 'RB', 'LWB', 'RWB']])), 0)
        })
    
    rankings.sort(key=lambda x: x['score'], reverse=True)
    return rankings

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🇬🇭 BLACK STARS </h1>
        <p style='font-size: 1.2rem; color: #fcd116;'>2026 World Cup • AI-Powered Tactical Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Load models
        if not st.session_state.openrouter_models:
            with st.spinner('Loading AI models...'):
                st.session_state.openrouter_models = get_free_openrouter_models()
        
        available_models = st.session_state.openrouter_models
        
        if not available_models:
            st.error('Could not load models.')
            if st.button('🔄 Refresh Models'):
                st.cache_data.clear()
                st.session_state.openrouter_models = []
                st.rerun()
        else:
            model_display = [f"{model.get('name', model['id'])}" for model in available_models]
            model_ids = [model['id'] for model in available_models]
            
            col1, col2 = st.columns([6, 1])
            
            with col1:
                try:
                    default_idx = model_ids.index(st.session_state.selected_model)
                except ValueError:
                    default_idx = 0
                
                selected_index = st.selectbox(
                    '🤖 AI Model (Free)',
                    range(len(model_display)),
                    format_func=lambda x: model_display[x],
                    index=default_idx
                )
                
                st.session_state.selected_model = model_ids[selected_index]
            
            with col2:
                if st.button('🔄'):
                    st.cache_data.clear()
                    st.session_state.openrouter_models = []
                    st.rerun()
        
        st.markdown("---")
        
        # API Key
        api_key = get_load_balanced_api_key('openrouter')
        
        if not api_key:
            api_key = st.text_input(
                "🔑 OpenRouter API Key",
                type="password",
                value=st.session_state.api_keys.get('openrouter', ''),
                help="Get free key at https://openrouter.ai/keys"
            )
            st.session_state.api_keys['openrouter'] = api_key
        
        if not api_key:
            st.warning("⚠️ Add API key for AI features")
        
        st.markdown("---")
        
        # Formation Selection
        st.markdown("### 📊 Formation Strategy")
        formation = st.selectbox(
            "Choose Formation", 
            list(FORMATIONS.keys()), 
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Squad Mode Selection
        st.markdown("### 🏆 Squad Mode")
        squad_mode = st.selectbox(
            "Squad Selection",
            ['full', 'worldcup'],
            format_func=lambda x: f"Full Player Pool (64)" if x == 'full' else f"World Cup Squad ({len(st.session_state.world_cup_squad)}/26)",
            index=0 if st.session_state.squad_mode == 'full' else 1
        )
        
        if squad_mode != st.session_state.squad_mode:
            st.session_state.squad_mode = squad_mode
            st.rerun()
        
        if st.button("⚽ Build WC Squad", use_container_width=True):
            st.session_state.show_wc_builder = True
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔄 Clear Lineup", use_container_width=True):
            st.session_state.lineup = {}
            st.rerun()
        
        if st.button("🎲 Auto-Fill Best XI", use_container_width=True):
            st.session_state.lineup = auto_select_best_xi(formation)
            st.rerun()
        
        if st.button("🎲 Randomize", use_container_width=True):
            lineup = {}
            formation_data = FORMATIONS[formation]
            slot_counter = 0
            used_players = set()
            
            for line_idx, line in enumerate(formation_data):
                for pos_idx, pos in enumerate(line['positions']):
                    slot_id = f"{pos}_{line_idx}_{pos_idx}_{slot_counter}"
                    slot_counter += 1
                    available = get_players_for_position(pos)
                    filtered = [p for p in available if p['fullName'] not in used_players]
                    
                    if filtered:
                        import random
                        player = random.choice(filtered[:10])
                        lineup[slot_id] = player
                        used_players.add(player['fullName'])
            
            st.session_state.lineup = lineup
            st.rerun()
        
        st.markdown("---")
        
        # Team Stats
        stats = calculate_stats(st.session_state.lineup)
        st.markdown("### 📊 Team Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⭐ Avg Rating", f"{stats['avg_rating']}")
            st.metric("👥 Avg Age", f"{stats['avg_age']}")
            st.metric("🎖️ Total Caps", f"{stats['total_caps']}")
        with col2:
            st.metric("⚡ Chemistry", f"{stats['chemistry']}%")
            st.metric("⚔️ Attack", f"{stats['attack']}")
            st.metric("🛡️ Defense", f"{stats['defense']}")
        
        st.markdown("---")
        
        # AI Analysis
        st.markdown("### 🧠 AI Analysis")
        
        can_analyze = len(st.session_state.lineup) >= 11 and api_key
        
        if st.button("🤖 Analyze Lineup", type="primary", disabled=not can_analyze, use_container_width=True):
            with st.spinner("🧠 AI analyzing tactical setup..."):
                players_list = list(st.session_state.lineup.values())
                
                prompt = f"""Analyze this Ghana Black Stars lineup for 2026 World Cup qualifiers:

Formation: {formation}
Team Statistics: Rating {stats['avg_rating']}, Age {stats['avg_age']}, Chemistry {stats['chemistry']}%

Players:
{chr(10).join([f"- {p['fullName']} ({p['club']}) - Position: {list(st.session_state.lineup.keys())[list(st.session_state.lineup.values()).index(p)].split('_')[0]}, Rating: {p['rating']}, Form: {p['form']}/10" for p in players_list])}

Provide a detailed tactical analysis:
1. **Formation Strengths** (3-4 key points)
2. **Potential Weaknesses** (2-3 concerns)
3. **Key Player Partnerships** (describe 2-3 combinations)
4. **Tactical Recommendations** (specific adjustments)
5. **Match Readiness Score** (1-10 with justification)

Be specific, tactical, and actionable."""

                try:
                    messages = [{"role": "user", "content": prompt}]
                    response = call_openrouter_api(messages, st.session_state.selected_model, api_key)
                    data = response.json()
                    analysis = data['choices'][0]['message']['content']
                    st.session_state.ai_analysis = analysis
                    st.success("✅ Analysis complete!")
                
                except requests.RequestException as e:
                    error_msg = f"Error: {str(e)}"
                    if '401' in str(e):
                        error_msg = '⚠️ Invalid API Key. Please check your OpenRouter API key.'
                        mark_key_failed('openrouter', api_key)
                    elif '429' in str(e):
                        error_msg = '⚠️ Rate limit exceeded. Please try again in a moment.'
                    st.error(error_msg)
                    st.session_state.ai_analysis = error_msg
        
        if len(st.session_state.lineup) < 11:
            st.info(f"ℹ️ Add {11 - len(st.session_state.lineup)} more players for full analysis")
        elif not api_key:
            st.info("ℹ️ Add API key to enable AI analysis")
    
    # World Cup Squad Builder Modal
    if st.session_state.get('show_wc_builder', False):
        st.markdown("---")
        st.markdown("## 🏆 Build Your 2026 World Cup Squad")
        st.markdown(f"**Select exactly 26 players** | Currently selected: {len(st.session_state.world_cup_squad)}/26")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Save & Use Squad", use_container_width=True):
                if len(st.session_state.world_cup_squad) == 26:
                    st.session_state.squad_mode = 'worldcup'
                    st.session_state.show_wc_builder = False
                    st.success("World Cup squad saved!")
                    st.rerun()
                else:
                    st.error(f"Please select exactly 26 players. Currently: {len(st.session_state.world_cup_squad)}")
        
        with col2:
            if st.button("⚡ Auto-Select Best 26", use_container_width=True):
                all_players = get_all_players()
                all_players.sort(key=lambda x: x['rating'], reverse=True)
                st.session_state.world_cup_squad = [p['fullName'] for p in all_players[:26]]
                st.rerun()
        
        with col3:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_wc_builder = False
                st.rerun()
        
        st.markdown("---")
        
        # Player selection by position
        categories = [
            ('Goalkeepers', 'GK'),
            ('Defenders', 'DEF'),
            ('Midfielders', 'MID'),
            ('Versatile', 'VERSATILE'),
            ('Attackers', 'ATT')
        ]
        
        for cat_name, cat_key in categories:
            if cat_key not in PLAYERS:
                continue
                
            st.markdown(f"### {cat_name}")
            
            for player in PLAYERS[cat_key]:
                is_selected = player['fullName'] in st.session_state.world_cup_squad
                
                col1, col2, col3 = st.columns([1, 4, 1])
                
                with col1:
                    if st.checkbox(
                        "",
                        value=is_selected,
                        key=f"wc_{player['fullName']}",
                        label_visibility="collapsed"
                    ):
                        if player['fullName'] not in st.session_state.world_cup_squad:
                            if len(st.session_state.world_cup_squad) < 26:
                                st.session_state.world_cup_squad.append(player['fullName'])
                                st.rerun()
                            else:
                                st.warning("Maximum 26 players!")
                    else:
                        if player['fullName'] in st.session_state.world_cup_squad:
                            st.session_state.world_cup_squad.remove(player['fullName'])
                            st.rerun()
                
                with col2:
                    st.markdown(f"**{player['fullName']}** • {player['club']} • {player['age']}y • {player['caps']} caps")
                
                with col3:
                    st.markdown(f"**{player['rating']}**")
        
        st.markdown("---")
        return
    
    # Main Content - Lineup Display
    st.markdown("### 📋 Team Lineup")
    
    if st.session_state.lineup:
        formation_data = FORMATIONS[formation]
        slot_counter = 0
        
        for line_idx, line in enumerate(formation_data):
            st.markdown(f"<div class='line-label'>{line['label']}</div>", unsafe_allow_html=True)
            
            num_positions = len(line['positions'])
            cols = st.columns(num_positions)
            
            for pos_idx, position in enumerate(line['positions']):
                slot_id = f"{position}_{line_idx}_{pos_idx}_{slot_counter}"
                slot_counter += 1
                
                with cols[pos_idx]:
                    player = st.session_state.lineup.get(slot_id)
                    
                    if player:
                        st.markdown(f"""
                        <div class='player-card' style='background: linear-gradient(135deg, #ce1126, #8b0000); margin-bottom: 0.8rem;'>
                            <div style='text-align: center;'>
                                <div style='background: linear-gradient(135deg, #d4af37, #fcd116); color: #000; padding: 0.5rem; border-radius: 8px; font-weight: bold; font-size: 1.3rem; margin-bottom: 0.5rem;'>
                                    {player['rating']}
                                </div>
                                <div style='font-weight: bold; color: #fcd116; font-size: 1rem; margin-bottom: 0.3rem;'>{position}</div>
                                <div style='font-weight: bold; color: #fff; font-size: 0.9rem; margin-bottom: 0.3rem;'>{player['name']}</div>
                                <div style='font-size: 0.75rem; color: #d4af37;'>{player['club']}</div>
                                <div style='font-size: 0.7rem; color: #ccc; margin-top: 0.3rem;'>{player['age']}y • {player['caps']} caps • Form {player['form']}/10</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"🔄 Change", key=f"change_{slot_id}", use_container_width=True):
                            del st.session_state.lineup[slot_id]
                            st.rerun()
                    else:
                        st.markdown(f"""
                        <div class='player-card' style='background: linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(252, 209, 22, 0.3)); border: 2px dashed #fcd116; margin-bottom: 0.8rem;'>
                            <div style='text-align: center; padding: 1rem;'>
                                <div style='font-size: 1.2rem; font-weight: bold; color: #fcd116;'>{position}</div>
                                <div style='font-size: 0.8rem; color: #aaa; margin-top: 0.3rem;'>Empty</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander(f"➕ Select {position}", expanded=False):
                            available = get_players_for_position(position)
                            
                            for player_idx, player_option in enumerate(available[:15]):
                                btn_key = f"select_{slot_id}_{player_idx}"
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    if st.button(
                                        f"{player_option['fullName']} • {player_option['club']}", 
                                        key=btn_key,
                                        use_container_width=True
                                    ):
                                        st.session_state.lineup[slot_id] = player_option
                                        st.rerun()
                                
                                with col2:
                                    st.markdown(f"<div style='text-align: center; background: #d4af37; color: #000; padding: 0.3rem; border-radius: 5px; font-weight: bold;'>{player_option['rating']}</div>", unsafe_allow_html=True)
            
            if line_idx < len(formation_data) - 1:
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    else:
        st.info("🎯 Start building your lineup using the Auto-Fill button in the sidebar")
    
    # Formation Rankings
    st.markdown("---")
    st.markdown("## ⚡ Formation Rankings")
    st.markdown("AI-powered analysis based on player strengths")
    
    rankings = calculate_formation_rankings()
    
    for idx, rank in enumerate(rankings[:8]):
        medal = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{idx + 1}."
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{medal} {rank['name']}**")
        with col2:
            st.markdown(f"Score: **{rank['score']}**")
        with col3:
            st.markdown(f"ATT: {rank['attack']}")
        with col4:
            st.markdown(f"DEF: {rank['defense']}")
    
    # AI Analysis Display
    if st.session_state.ai_analysis:
        st.markdown("---")
        st.markdown("## 🧠 AI Tactical Analysis Report")
        
        st.markdown(f"""
        <div class='ai-analysis-box'>
            <div style='font-size: 1.15rem; line-height: 2; color: #ffffff; font-weight: 400;'>
                {st.session_state.ai_analysis.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Clear Analysis", use_container_width=True):
            st.session_state.ai_analysis = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #d4af37; font-size: 0.85rem; padding: 1rem 0;'>
        <p>🇬🇭 <strong>BlackStars Analysis</strong> • RNK RadioSport</p>
        <p style='font-size: 0.75rem; color: #aaa;'>Build winning formations for the 2026 World Cup journey</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
