"""
Ghana Black Stars AI Lineup Builder - Streamlit Version with OpenRouter AI Integration
Author: RNK RadioSport
Version: 1.3.0 - Complete Formations & Fixed Display
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

# Player Database
PLAYERS = {
    'GK': [
        {'name': 'Ati-Zigi', 'fullName': 'Lawrence Ati-Zigi', 'club': 'St. Gallen', 'rating': 78, 'age': 28, 'caps': 18, 'form': 8},
        {'name': 'Asare', 'fullName': 'Benjamin Asare', 'club': 'Hearts of Oak', 'rating': 76, 'age': 25, 'caps': 8, 'form': 9},
        {'name': 'Nurudeen', 'fullName': 'Abdul Manaf Nurudeen', 'club': 'KAS Eupen', 'rating': 74, 'age': 26, 'caps': 6, 'form': 7},
    ],
    'DEF': [
        {'name': 'Salisu', 'fullName': 'Mohammed Salisu', 'club': 'Monaco', 'rating': 82, 'age': 26, 'caps': 22, 'form': 8},
        {'name': 'Djiku', 'fullName': 'Alexander Djiku', 'club': 'Spartak Moscow', 'rating': 81, 'age': 31, 'caps': 37, 'form': 8},
        {'name': 'Lamptey', 'fullName': 'Tariq Lamptey', 'club': 'Brighton', 'rating': 80, 'age': 24, 'caps': 15, 'form': 8},
        {'name': 'Seidu', 'fullName': 'Alidu Seidu', 'club': 'Stade Rennes', 'rating': 78, 'age': 25, 'caps': 22, 'form': 8},
        {'name': 'Kohn', 'fullName': 'Derrick Kohn', 'club': 'Union Berlin', 'rating': 77, 'age': 26, 'caps': 10, 'form': 8},
        {'name': 'Mensah', 'fullName': 'Gideon Mensah', 'club': 'AJ Auxerre', 'rating': 76, 'age': 27, 'caps': 18, 'form': 7},
        {'name': 'Ambrosius', 'fullName': 'Stephan Ambrosius', 'club': 'St. Gallen', 'rating': 77, 'age': 26, 'caps': 10, 'form': 7},
        {'name': 'Mumin', 'fullName': 'Abdul Mumin', 'club': 'Rayo Vallecano', 'rating': 76, 'age': 27, 'caps': 15, 'form': 7},
    ],
    'MID': [
        {'name': 'Partey', 'fullName': 'Thomas Partey', 'club': 'Arsenal', 'rating': 84, 'age': 32, 'caps': 58, 'form': 8},
        {'name': 'Kudus', 'fullName': 'Mohammed Kudus', 'club': 'West Ham', 'rating': 84, 'age': 25, 'caps': 32, 'form': 9},
        {'name': 'Ashimeru', 'fullName': 'Majeed Ashimeru', 'club': 'Anderlecht', 'rating': 77, 'age': 28, 'caps': 24, 'form': 7},
        {'name': 'Samed', 'fullName': 'Salis Abdul Samed', 'club': 'RC Lens', 'rating': 78, 'age': 25, 'caps': 17, 'form': 8},
        {'name': 'E. Owusu', 'fullName': 'Elisha Owusu', 'club': 'AJ Auxerre', 'rating': 76, 'age': 28, 'caps': 20, 'form': 7},
        {'name': 'Sulemana I.', 'fullName': 'Ibrahim Sulemana', 'club': 'Atalanta', 'rating': 76, 'age': 22, 'caps': 8, 'form': 7},
    ],
    'ATT': [
        {'name': 'I. Williams', 'fullName': 'Inaki Williams', 'club': 'Athletic Bilbao', 'rating': 82, 'age': 31, 'caps': 24, 'form': 9},
        {'name': 'Semenyo', 'fullName': 'Antoine Semenyo', 'club': 'Bournemouth', 'rating': 81, 'age': 26, 'caps': 18, 'form': 9},
        {'name': 'J. Ayew', 'fullName': 'Jordan Ayew', 'club': 'Leicester', 'rating': 79, 'age': 34, 'caps': 118, 'form': 8},
        {'name': 'Fatawu', 'fullName': 'Abdul Fatawu Issahaku', 'club': 'Leicester', 'rating': 79, 'age': 21, 'caps': 23, 'form': 8},
        {'name': 'Sulemana K.', 'fullName': 'Kamaldeen Sulemana', 'club': 'Southampton', 'rating': 78, 'age': 23, 'caps': 18, 'form': 7},
        {'name': 'Nuamah', 'fullName': 'Ernest Nuamah', 'club': 'Lyon', 'rating': 77, 'age': 22, 'caps': 12, 'form': 8},
        {'name': 'Paintsil', 'fullName': 'Joseph Paintsil', 'club': 'LA Galaxy', 'rating': 77, 'age': 28, 'caps': 12, 'form': 8},
    ],
}

# Comprehensive Formation Database - Attack at top, GK at bottom
FORMATIONS = {
    '4-3-3 (Attack)': [
        {'positions': ['LW', 'ST', 'RW'], 'label': 'ATTACK'},
        {'positions': ['LCM', 'CM', 'RCM'], 'label': 'MIDFIELD'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-3-3 (Defensive)': [
        {'positions': ['LW', 'ST', 'RW'], 'label': 'ATTACK'},
        {'positions': ['LCM', 'CDM', 'RCM'], 'label': 'MIDFIELD'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-2-3-1 (Wide)': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LW', 'CAM', 'RW'], 'label': 'ATTACKING MID'},
        {'positions': ['LDM', 'RDM'], 'label': 'DEFENSIVE MID'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-2-3-1 (Narrow)': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LCAM', 'CAM', 'RCAM'], 'label': 'ATTACKING MID'},
        {'positions': ['LDM', 'RDM'], 'label': 'DEFENSIVE MID'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-4-2': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-4-2 (Holding)': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['LM', 'LCDM', 'RCDM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '4-1-4-1': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['CDM'], 'label': 'DEFENSIVE MID'},
        {'positions': ['LB', 'LCB', 'RCB', 'RB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '3-5-2': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['LWB', 'LCM', 'CM', 'RCM', 'RWB'], 'label': 'MIDFIELD'},
        {'positions': ['LCB', 'CB', 'RCB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '3-4-3': [
        {'positions': ['LW', 'ST', 'RW'], 'label': 'ATTACK'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LCB', 'CB', 'RCB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '3-4-1-2': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['CAM'], 'label': 'ATTACKING MID'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LCB', 'CB', 'RCB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '5-3-2': [
        {'positions': ['LST', 'RST'], 'label': 'STRIKERS'},
        {'positions': ['LCM', 'CM', 'RCM'], 'label': 'MIDFIELD'},
        {'positions': ['LWB', 'LCB', 'CB', 'RCB', 'RWB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
    '5-4-1': [
        {'positions': ['ST'], 'label': 'STRIKER'},
        {'positions': ['LM', 'LCM', 'RCM', 'RM'], 'label': 'MIDFIELD'},
        {'positions': ['LWB', 'LCB', 'CB', 'RCB', 'RWB'], 'label': 'DEFENSE'},
        {'positions': ['GK'], 'label': 'GOALKEEPER'}
    ],
}

@st.cache_data(ttl=3600)
def get_free_openrouter_models():
    """Dynamically load OpenRouter free models"""
    try:
        response = requests.get(
            OPENROUTER_MODELS_URL,
            headers={
                'User-Agent': 'Ghana-BlackStars-AI/1.0',
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

def get_players_for_position(position):
    """Get suitable players for a position - Enhanced mapping"""
    # Comprehensive position mapping
    position_map = {
        # Goalkeepers
        'GK': ['GK'],
        
        # Defenders
        'CB': ['DEF'], 'LCB': ['DEF'], 'RCB': ['DEF'],
        'LB': ['DEF'], 'RB': ['DEF'],
        'LWB': ['DEF'], 'RWB': ['DEF'],
        
        # Midfielders
        'CDM': ['MID'], 'LDM': ['MID'], 'RDM': ['MID'], 'LCDM': ['MID'], 'RCDM': ['MID'],
        'CM': ['MID'], 'LCM': ['MID'], 'RCM': ['MID'],
        'CAM': ['MID'], 'LCAM': ['MID'], 'RCAM': ['MID'],
        'LM': ['MID', 'ATT'], 'RM': ['MID', 'ATT'],
        
        # Attackers
        'LW': ['ATT'], 'RW': ['ATT'],
        'ST': ['ATT'], 'LST': ['ATT'], 'RST': ['ATT'],
    }
    
    categories = position_map.get(position, ['MID'])
    all_players = []
    
    for cat in categories:
        if cat in PLAYERS:
            all_players.extend(PLAYERS[cat])
    
    # Remove duplicates while preserving order
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
        return {'avg_rating': 0, 'avg_age': 0, 'total_caps': 0, 'chemistry': 0}
    
    players = list(lineup.values())
    avg_rating = round(sum(p['rating'] for p in players) / len(players), 1)
    avg_age = round(sum(p['age'] for p in players) / len(players), 1)
    total_caps = sum(p['caps'] for p in players)
    chemistry = min(100, round((sum(p['form'] for p in players) / len(players)) * 10))
    
    return {'avg_rating': avg_rating, 'avg_age': avg_age, 'total_caps': total_caps, 'chemistry': chemistry}

def render_formation_pitch(formation):
    """Render the formation pitch with proper layout"""
    formation_data = FORMATIONS[formation]
    slot_counter = 0
    
    st.markdown("<div class='pitch-container'>", unsafe_allow_html=True)
    
    for line_idx, line in enumerate(formation_data):
        # Line label
        st.markdown(f"<div class='line-label'>{line['label']}</div>", unsafe_allow_html=True)
        
        # Calculate proper spacing
        num_positions = len(line['positions'])
        cols = st.columns([1] + [2]*num_positions + [1])
        
        for pos_idx, position in enumerate(line['positions']):
            slot_id = f"{position}_{line_idx}_{pos_idx}_{slot_counter}"
            slot_counter += 1
            
            with cols[pos_idx + 1]:
                player = st.session_state.lineup.get(slot_id)
                
                if player:
                    st.markdown(f"""
                    <div class='player-card' style='background: linear-gradient(135deg, #ce1126, #8b0000); text-align: center;'>
                        <div style='font-size: 1.8rem; font-weight: bold; color: #fcd116;'>{player['rating']}</div>
                        <div style='font-size: 0.95rem; font-weight: bold; color: #fff; margin: 0.3rem 0;'>{player['name']}</div>
                        <div style='font-size: 0.75rem; color: #d4af37;'>{player['club']}</div>
                        <div style='font-size: 0.7rem; color: #aaa; margin-top: 0.2rem;'>{position}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"🔄 Change", key=f"change_{slot_id}", use_container_width=True):
                        del st.session_state.lineup[slot_id]
                        st.rerun()
                else:
                    st.markdown(f"""
                    <div class='player-card' style='background: linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(252, 209, 22, 0.3)); text-align: center; border: 2px dashed #fcd116;'>
                        <div style='font-size: 1.3rem; font-weight: bold; color: #fcd116; padding: 1.5rem 0.5rem;'>{position}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"➕ Select {position}", expanded=False):
                        available = get_players_for_position(position)
                        
                        for player_idx, player in enumerate(available[:10]):
                            btn_key = f"select_{slot_id}_{player_idx}"
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                if st.button(
                                    f"{player['fullName']} • {player['club']}", 
                                    key=btn_key,
                                    use_container_width=True
                                ):
                                    st.session_state.lineup[slot_id] = player
                                    st.rerun()
                            
                            with col2:
                                st.markdown(f"<div style='text-align: center; background: #d4af37; color: #000; padding: 0.3rem; border-radius: 5px; font-weight: bold;'>{player['rating']}</div>", unsafe_allow_html=True)
        
        # Add spacing between lines
        if line_idx < len(formation_data) - 1:
            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

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
            # Model selection
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
        
        if api_key:
            pass
        else:
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
        
        # Quick Actions
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔄 Clear Lineup", use_container_width=True):
            st.session_state.lineup = {}
            st.rerun()
        
        if st.button("🎲 Auto-Fill Best XI", use_container_width=True):
            lineup = {}
            formation_data = FORMATIONS[formation]
            slot_counter = 0
            
            for line_idx, line in enumerate(formation_data):
                for pos_idx, pos in enumerate(line['positions']):
                    slot_id = f"{pos}_{line_idx}_{pos_idx}_{slot_counter}"
                    slot_counter += 1
                    available = get_players_for_position(pos)
                    
                    for player in available:
                        if player['fullName'] not in [p['fullName'] for p in lineup.values()]:
                            lineup[slot_id] = player
                            break
            
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
        with col2:
            st.metric("🎖️ Total Caps", f"{stats['total_caps']}")
            st.metric("⚡ Chemistry", f"{stats['chemistry']}%")
        
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
    
    # Main Content Area
    st.markdown("### 📋 Team Lineup by Formation")
    
    if st.session_state.lineup:
        formation_data = FORMATIONS[formation]
        slot_counter = 0
        
        for line_idx, line in enumerate(formation_data):
            # Line label
            st.markdown(f"<div class='line-label'>{line['label']}</div>", unsafe_allow_html=True)
            
            # Create columns based on number of positions in this line
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
                            
                            for player_idx, player_option in enumerate(available[:10]):
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
            
            # Add spacing between formation lines
            if line_idx < len(formation_data) - 1:
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    else:
        st.info("🎯 Start building your lineup using the Auto-Fill button in the sidebar")
        st.markdown("""
        <div style='background: rgba(212, 175, 55, 0.1); padding: 1.5rem; border-radius: 10px; margin-top: 1rem;'>
            <h4 style='color: #fcd116; margin-bottom: 0.5rem;'>Quick Start:</h4>
            <ul style='color: #ccc; font-size: 0.9rem; line-height: 1.8;'>
                <li>Click "Auto-Fill Best XI" for instant lineup</li>
                <li>Or add players manually position by position</li>
                <li>Mix experience with youth for balance</li>
                <li>Complete 11 players for full AI analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Analysis Display (Full Width)
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
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Clear Analysis", use_container_width=True):
                st.session_state.ai_analysis = None
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #d4af37; font-size: 0.85rem; padding: 1rem 0;'>
        <p>🇬🇭 <strong> BlackStars Analysis </strong> • RNK RadioSport</p>
        <p style='font-size: 0.75rem; color: #aaa;'>Build winning formations for the 2026 World Cup journey</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()