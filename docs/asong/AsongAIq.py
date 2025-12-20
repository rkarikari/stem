_AK='selected_model'
_AJ='Content-Type'
_AI='api_usage'
_AH='selected_online_model'
_AG='auto_run_plots'
_AF='Radio Station'
_AE='station_name'
_AD='total_songs'
_AC='auto_mode_stats'
_AB='auto_mode_active'
_AA='current_song_result'
_A9='https://github.com/rkarikari/RadioSport-chat'
_A8='RadioSport AI'
_A7='http://localhost:11434'
_A6='https://openrouter.ai/api/v1/models'
_A5='streamlit'
_A4='original_prompt'
_A3='reason'
_A2='Deep Dive'
_A1='Detailed Analysis'
_A0='🔍 Identifying song...'
_z='subtitle'
_y='✅ Recording completed!'
_x='sample_rate'
_w='Released'
_v='metadata'
_u='SONG'
_t='model'
_s='application/json'
_r='messages'
_q='last_reset'
_p='bash'
_o='...'
_n='%Y-%m-%d %H:%M:%S'
_m='.wav'
_l='id'
_k='message'
_j='failed'
_i='\n'
_h='Quick Overview'
_g='api_provider'
_f='recommendations'
_e='primary'
_d='utf-8'
_c='name'
_b='document_cache_misses'
_a='model_cache_misses'
_Z='start_time'
_Y='code'
_X='ollama_host'
_W='document_cache_hits'
_V='ignore'
_U='user'
_T='Cloud'
_S='model_cache_hits'
_R='type'
_Q='Local'
_P='sections'
_O='---'
_N='openrouter'
_M='timestamp'
_L='text'
_K='role'
_J='genres'
_I='artist'
_H='track'
_G='data'
_F='title'
_E='content'
_D='Unknown'
_C=False
_B=None
_A=True
import os
os.environ['PYDUB_UTILS_QUIET']='1'
import warnings
warnings.filterwarnings(_V,category=DeprecationWarning)
warnings.filterwarnings(_V,category=SyntaxWarning)
warnings.filterwarnings(_V,category=RuntimeWarning)
import sys,streamlit as st,asyncio,wave,tempfile,base64,numpy as np,json,threading,time,io
from datetime import datetime,timedelta
from io import BytesIO
try:from pydub import AudioSegment;PYDUB_AVAILABLE=_A
except(ImportError,ModuleNotFoundError):PYDUB_AVAILABLE=_C;AudioSegment=_B
from shazamio import Shazam
import logging
from pathlib import Path
import csv,requests,subprocess,re,hashlib
from functools import lru_cache
import contextlib,traceback,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,pandas as pd,seaborn as sns,matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
import PyPDF2
PYAUDIO_AVAILABLE=_C
BROWSER_AUDIO_AVAILABLE=_A
try:import pyaudio;PYAUDIO_AVAILABLE=_A
except ImportError:PYAUDIO_AVAILABLE=_C
warnings.simplefilter(_V)
os.environ['PYTHONWARNINGS']=_V
logging.getLogger('pydub.converter').setLevel(logging.ERROR)
logging.getLogger('pydub.utils').setLevel(logging.ERROR)
logging.getLogger(_A5).setLevel(logging.ERROR)
try:st.set_option('client.showErrorDetails',_C)
except Exception:pass
CHUNK=1024
FORMAT=pyaudio.paInt16 if PYAUDIO_AVAILABLE else _B
CHANNELS=1
RATE=44100
RECORD_SECONDS=12
AUTO_RECORD_SECONDS=6
def rms_amplitude(audio_data):
	'Calculate RMS amplitude using numpy'
	if isinstance(audio_data,bytes):audio_array=np.frombuffer(audio_data,dtype=np.int16)
	else:audio_array=audio_data
	rms=np.sqrt(np.mean(audio_array.astype(np.float64)**2));return int(rms)
def max_amplitude(audio_data):
	'Calculate max amplitude using numpy'
	if isinstance(audio_data,bytes):audio_array=np.frombuffer(audio_data,dtype=np.int16)
	else:audio_array=audio_data
	return int(np.max(np.abs(audio_array)))
def apply_gain(audio_data,gain_factor):
	'Apply gain to audio using numpy'
	if isinstance(audio_data,bytes):audio_array=np.frombuffer(audio_data,dtype=np.int16)
	else:audio_array=audio_data
	gained_audio=np.clip(audio_array*gain_factor,-32768,32767);return gained_audio.astype(np.int16).tobytes()
def is_running_in_cloud():
	'Detect if app is running in cloud environment (Streamlit Cloud, Heroku, etc.)'
	try:
		import socket;hostname=socket.gethostname()
		if _A5 in hostname.lower():return _A
		cloud_indicators=['STREAMLIT_SHARING_MODE','STREAMLIT_SERVER_HEADLESS','DYNO','KUBERNETES_SERVICE_HOST','AWS_EXECUTION_ENV','GOOGLE_CLOUD_PROJECT']
		for indicator in cloud_indicators:
			if indicator in os.environ:return _A
		try:response=requests.get('http://localhost:11434/api/tags',timeout=1);return _C
		except(requests.ConnectionError,requests.Timeout):return _A
	except Exception:return _A
IS_CLOUD_ENVIRONMENT=is_running_in_cloud()
APP_VERSION='5.0.1'
OPENROUTER_API_URL='https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODELS_URL=_A6
DEFAULT_OLLAMA_HOST=_A7
OLLAMA_HOST=DEFAULT_OLLAMA_HOST
DEFAULT_MODEL='qwen3:4b'
CACHE_TTL=300
MAX_CACHE_ENTRIES=100
MAX_REASONING_LINES=5
REASONING_UPDATE_INTERVAL=.2
CODE_EXTENSIONS={'.py':'python','.js':'javascript','.ts':'typescript','.cpp':'cpp','.c':'c','.java':'java','.html':'html','.css':'css','.json':'json','.md':'markdown','.sh':_p,'.sql':'sql','.bat':'batch'}
IMAGE_EXTENSIONS='.png','.jpg','.jpeg'
st.set_page_config(page_title=_A8,page_icon='🧟🎵',layout='wide',menu_items={'Report a Bug':_A9,'About':'Copyright © RNK, 2025 RadioSport. All rights reserved.'})
def initialize_session_state():
	'Initialize all session state variables'
	if'identified_songs'not in st.session_state:st.session_state.identified_songs=[]
	if'audio_devices'not in st.session_state:st.session_state.audio_devices=[]
	if'selected_device'not in st.session_state:st.session_state.selected_device=_B
	if'selected_song'not in st.session_state:st.session_state.selected_song=_B
	if _AA not in st.session_state:st.session_state.current_song_result=_B
	if _AB not in st.session_state:st.session_state.auto_mode_active=_C
	if'auto_mode_thread'not in st.session_state:st.session_state.auto_mode_thread=_B
	if _AC not in st.session_state:st.session_state.auto_mode_stats={_Z:_B,_AD:0,'unique_songs':0,'genres_detected':{},'artists_detected':{},'failed_detections':0,'session_log':[]}
	if'auto_mode_settings'not in st.session_state:st.session_state.auto_mode_settings={'monitoring_duration':60,'detection_threshold':.3,'silence_duration':3,'min_song_length':30,_AE:_AF,'output_format':'CSV'}
	if'cache_stats'not in st.session_state:st.session_state.cache_stats={_S:0,_a:0,_W:0,_b:0,_q:datetime.now()}
	if'file_uploader_key'not in st.session_state:st.session_state.file_uploader_key='uploader_0'
	if'reasoning_window'not in st.session_state:st.session_state.reasoning_window=_B
	if _r not in st.session_state:st.session_state.messages=[]
	if _AG not in st.session_state:st.session_state.auto_run_plots=_A
	if'ollama_models'not in st.session_state:st.session_state.ollama_models=[]
	if'doc_cache'not in st.session_state:st.session_state.doc_cache={}
	if'last_reasoning_update'not in st.session_state:st.session_state.last_reasoning_update=time.time()
	if'base64_cache'not in st.session_state:st.session_state.base64_cache={}
	if'thinking_content'not in st.session_state:st.session_state.thinking_content=''
	if'in_thinking_block'not in st.session_state:st.session_state.in_thinking_block=_C
	if'reasoning_window_id'not in st.session_state:st.session_state.reasoning_window_id=f"reasoning_{time.time()}"
	if _g not in st.session_state:
		if IS_CLOUD_ENVIRONMENT:st.session_state.api_provider=_T;st.session_state.environment='cloud'
		else:st.session_state.api_provider=_Q;st.session_state.environment='local'
	if'api_keys'not in st.session_state:st.session_state.api_keys={_N:''}
	if _AH not in st.session_state:st.session_state.selected_online_model=0
	if'analysis_depth'not in st.session_state:st.session_state.analysis_depth=_h
	if'analysis_results'not in st.session_state:st.session_state.analysis_results={}
	if'quick_questions'not in st.session_state:st.session_state.quick_questions=["What's my most common genre?",'Who is my top artist?','What time period are most songs from?','Do I have a diverse taste?']
st.markdown('\n<style>\n.sidebar-title {\n    font-size: 1.5rem;\n    font-weight: bold;\n    color: #1f77b4;\n    margin-bottom: 0.5rem;\n    text-align: center;\n}\n.version-text {\n    font-size: 0.85rem;\n    color: #666;\n    text-align: center;\n    margin-bottom: 1rem;\n}\n.cache-stats {\n    font-size: 0.85rem;\n}\n.cache-stats table {\n    width: 100%;\n    border-collapse: collapse;\n}\n.cache-stats th, .cache-stats td {\n    padding: 4px 6px;\n    text-align: left;\n    border-bottom: 1px solid #eee;\n}\n.cache-stats th {\n    background-color: #f0f0f0;\n}\n.cache-hit {\n    color: #2ecc71;\n    font-weight: bold;\n}\n.cache-miss {\n    color: #e74c3c;\n    font-weight: bold;\n}\n.reasoning-window {\n    background-color: #f8f9fa;\n    border: 1px solid #dee2e6;\n    border-radius: 5px;\n    padding: 10px;\n    margin-top: 10px;\n    max-height: 200px;\n    overflow-y: auto;\n    font-family: monospace;\n    font-size: 0.9rem;\n}\n.reasoning-header {\n    font-weight: bold;\n    margin-bottom: 5px;\n    color: #1f77b4;\n}\n.analysis-section {\n    background-color: #f8f9fa;\n    border-left: 4px solid #1f77b4;\n    padding: 10px;\n    margin-bottom: 15px;\n    border-radius: 0 5px 5px 0;\n}\n.analysis-title {\n    font-weight: bold;\n    color: #1f77b4;\n    margin-bottom: 5px;\n}\n.recommendation-card {\n    background-color: #e8f4f8;\n    border-radius: 8px;\n    padding: 10px;\n    margin-bottom: 10px;\n}\n.recommendation-title {\n    font-weight: bold;\n    font-size: 1.1rem;\n}\n.recommendation-artist {\n    color: #666;\n}\n.recommendation-reason {\n    margin-top: 5px;\n    font-size: 0.9rem;\n}\n</style>\n',unsafe_allow_html=_A)
st.markdown('\n<script>\nwindow.MathJax = {\n    tex: {\n        inlineMath: [[\'$\', \'$\'], [\'\\(\', \'\\)\']]\n    },\n    svg: {\n        fontCache: \'global\'\n    },\n    startup: {\n        ready: function() {\n            MathJax.startup.defaultReady();\n            MathJax.startup.promise.then(function() {\n                window.dispatchEvent(new Event(\'mathjax-loaded\'));\n            });\n        }\n    }\n};\n</script>\n<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>\n',unsafe_allow_html=_A)
st.title('🎵RadioSport Song ID & AI Analysis')
def get_ollama_host():'Get the current Ollama host from session state or default';return st.session_state.get(_X,DEFAULT_OLLAMA_HOST)
def update_all_ollama_host_references(new_host):
	'Update OLLAMA_HOST in all imported modules';A='OLLAMA_HOST';import sys;globals()[A]=new_host;modules_updated=[]
	for(module_name,module)in sys.modules.items():
		if hasattr(module,A):old_value=getattr(module,A);setattr(module,A,new_host);modules_updated.append(f"{module_name}: {old_value} -> {new_host}")
		if hasattr(module,_X):old_value=getattr(module,_X);setattr(module,_X,new_host);modules_updated.append(f"{module_name}.ollama_host: {old_value} -> {new_host}")
	for(module_name,module)in sys.modules.items():
		for attr_name in['OLLAMA_BASE_URL','ollama_base_url','base_url','host_url']:
			if hasattr(module,attr_name):setattr(module,attr_name,new_host)
@st.cache_data(ttl=CACHE_TTL,show_spinner=_C)
def get_ollama_models_cached():
	try:
		api_url=f"{get_ollama_host()}/api/tags";response=requests.get(api_url,timeout=30);response.raise_for_status();data=response.json();models=[]
		for model_info in data.get('models',[]):
			model_name=model_info.get(_c,'')
			if model_name:models.append(model_name)
		if not models:return[]
		embedding_prefixes='nomic-embed-text','all-minilm','mxbai-embed','snowflake-arctic-embed','bge-','gte-','e5-';filtered_models=[]
		for model in models:
			model=model.strip()
			if model and not any(model.lower().startswith(prefix.lower())for prefix in embedding_prefixes):filtered_models.append(model)
		return filtered_models
	except requests.ConnectionError:st.error(f"Server Unavailable: {OLLAMA_HOST}.");return[]
	except requests.Timeout:st.error(f"Server Timeout: {OLLAMA_HOST}");return[]
	except requests.HTTPError as e:st.error(f"HTTP error: {e.response.status_code}");return[]
	except requests.RequestException as e:st.error(f"Server Error: {OLLAMA_HOST}: {str(e)}");return[]
	except json.JSONDecodeError:st.error('Invalid JSON response');return[]
	except Exception as e:st.error(f"Models error: {str(e)}");return[]
def get_ollama_models():
	start_time=time.time();result=get_ollama_models_cached();end_time=time.time()
	if end_time-start_time<.1 and result:st.session_state.cache_stats[_S]+=1
	else:st.session_state.cache_stats[_a]+=1
	return result
@lru_cache(maxsize=128)
def is_vision_model(model):
	'Enhanced vision model detection with comprehensive patterns'
	if not model:return _C
	model_lower=model.lower();vision_patterns=['qwen2\\.5?vl:','gemma3[^:]*:','llava:','moondream:','bakllava:','minicpm-v:','yi-vl:','internvl:','cogvlm:','vision','visual','multimodal','-v:','vl-'];return any(re.search(pattern,model_lower)for pattern in vision_patterns)
@lru_cache(maxsize=128)
def is_qwen3_model(model):
	'Enhanced Qwen3 model detection'
	if not model:return _C
	return bool(re.search('qwen3:',model.lower()))
@lru_cache(maxsize=256)
def get_file_type(file_name):
	ext=os.path.splitext(file_name)[1].lower()
	if ext in CODE_EXTENSIONS:return _Y,CODE_EXTENSIONS[ext]
	elif ext=='.txt':return _L,_B
	elif ext=='.pdf':return'pdf',_B
	elif ext in IMAGE_EXTENSIONS:return'image',_B
	return _B,_B
@st.cache_data(ttl=CACHE_TTL,max_entries=MAX_CACHE_ENTRIES,show_spinner=_C)
def split_message_cached(content):
	CODE_BLOCK_PATTERN=re.compile('```(\\w+)?\\n(.*?)```',re.DOTALL);parts=[];last_end=0
	for match in CODE_BLOCK_PATTERN.finditer(content):
		start,end=match.span()
		if last_end<start:
			text_content=content[last_end:start]
			if text_content.strip():parts.append({_R:_L,_E:text_content})
		language=match.group(1)or _L;code=match.group(2);parts.append({_R:_Y,'language':language,_Y:code});last_end=end
	if last_end<len(content):
		remaining_content=content[last_end:]
		if remaining_content.strip():parts.append({_R:_L,_E:remaining_content})
	return parts
def split_message(content):return split_message_cached(content)
def process_single_file(file_data):
	file_content,file_type,file_name=file_data
	try:
		if file_type in(_L,_Y):
			try:return file_content.decode(_d),_B
			except UnicodeDecodeError:return file_content.decode(_d,errors='replace'),_B
		elif file_type=='pdf':
			pdf_reader=PyPDF2.PdfReader(BytesIO(file_content));content_parts=[]
			for(page_num,page)in enumerate(pdf_reader.pages):
				try:
					text=page.extract_text()
					if text and text.strip():content_parts.append(text)
				except Exception as e:return _B,f"error: Failed to read PDF page {page_num+1}: {e}"
			return _i.join(content_parts),_B
		elif file_type=='image':return file_content,_B
		return _B,'unsupported'
	except Exception as e:return _B,f"error: {str(e)}"
def process_document(file_content,file_type,file_name):
	content_hash=hashlib.md5(file_content).hexdigest();cache_key=f"{content_hash}_{file_type}_{file_name}"
	if cache_key in st.session_state.doc_cache:st.session_state.cache_stats[_W]+=1;return st.session_state.doc_cache[cache_key]
	result=process_single_file((file_content,file_type,file_name))
	if len(st.session_state.doc_cache)>=MAX_CACHE_ENTRIES:oldest_key=next(iter(st.session_state.doc_cache));del st.session_state.doc_cache[oldest_key]
	st.session_state.doc_cache[cache_key]=result;st.session_state.cache_stats[_b]+=1;return result
def get_base64_image(file_content):
	content_hash=hashlib.md5(file_content).hexdigest()
	if content_hash in st.session_state.base64_cache:return st.session_state.base64_cache[content_hash]
	base64_str=base64.b64encode(file_content).decode(_d);st.session_state.base64_cache[content_hash]=base64_str
	if len(st.session_state.base64_cache)>=MAX_CACHE_ENTRIES:oldest_key=next(iter(st.session_state.base64_cache));del st.session_state.base64_cache[oldest_key]
	return base64_str
def get_load_balanced_api_key(provider):
	'Get load-balanced API key with backward compatibility.';C='general';B='count';A='idx';keys=[]
	try:
		if provider.lower()==_N:
			try:
				key=st.secrets[_N]['api_key']
				if key and key.strip():keys.append(key.strip())
			except:pass
			for i in range(1,11):
				try:
					key=st.secrets[_N][f"api_key{i}"]
					if key and key.strip():keys.append(key.strip())
				except:pass
			try:
				key_list=st.secrets[_N]['api_keylist']
				if isinstance(key_list,(list,tuple)):keys.extend([k.strip()for k in key_list if k and k.strip()])
				elif isinstance(key_list,str)and key_list.strip():keys.append(key_list.strip())
			except:pass
		base_key=f"{provider.upper()}_API_KEY"
		try:
			key=st.secrets[C][base_key]
			if key and key.strip():keys.append(key.strip())
		except:pass
		for i in range(1,11):
			try:
				key=st.secrets[C][f"{base_key}_{i}"]
				if key and key.strip():keys.append(key.strip())
			except:pass
	except Exception:return''
	if not keys:return''
	if len(keys)==1:return keys[0]
	if _AI not in st.session_state:st.session_state.api_usage={}
	if provider not in st.session_state.api_usage:st.session_state.api_usage[provider]={A:0,_j:set(),B:{}}
	usage=st.session_state.api_usage[provider];valid_keys=[k for k in keys if k not in usage[_j]]
	if not valid_keys:usage[_j].clear();valid_keys=keys
	selected=valid_keys[usage[A]%len(valid_keys)];usage[A]=(usage[A]+1)%len(valid_keys);usage[B][selected]=usage[B].get(selected,0)+1;usage['last_used']=time.time();usage['selected_key']=selected;return selected
def mark_key_failed(provider,key):
	'Mark key as failed for load balancing.'
	if _AI in st.session_state and provider in st.session_state.api_usage:st.session_state.api_usage[provider][_j].add(key)
def call_openrouter_api(messages,model,api_key):
	headers={'Authorization':f"Bearer {api_key}",_AJ:_s,'HTTP-Referer':_A9,'X-Title':_A8};max_tokens=2048;payload={_t:model,_r:messages,'stream':_A,'max_tokens':max_tokens,'temperature':.7}
	try:
		response=requests.post(OPENROUTER_API_URL,headers=headers,json=payload,stream=_A,timeout=60)
		if response.status_code!=200:
			try:error_data=response.json();error_msg=error_data.get('error',{}).get(_k,response.text)
			except:error_msg=response.text
			raise Exception(f"OpenRouter API Error {response.status_code}: {error_msg}")
		return response
	except requests.exceptions.RequestException as e:raise Exception(f"Network error: {str(e)}")
	except Exception as e:raise Exception(f"Unexpected error: {str(e)}")
@st.cache_data(ttl=3600)
def get_free_openrouter_models():
	'Dynamically load OpenRouter free models'
	try:
		response=requests.get(_A6,headers={'User-Agent':'RadioSport-AI/1.0','Accept':_s},timeout=30);response.raise_for_status();models_data=response.json();free_models=[]
		for model in models_data.get(_G,[]):
			pricing=model.get('pricing',{});prompt_price=float(pricing.get('prompt','0'));completion_price=float(pricing.get('completion','0'))
			if prompt_price==0 and completion_price==0:
				model_id=model.get(_l,'')
				if model_id:free_models.append(model)
		if not free_models:st.error('No free models found in OpenRouter API response.');return[]
		free_models.sort(key=lambda x:x.get(_l,''));return free_models
	except Exception as e:st.error(f"Error loading OpenRouter models: {str(e)}");return[]
def get_song_history_context():
	'Convert song history into a text context for AI analysis';A='-----------------\n'
	if not st.session_state.identified_songs:return'No song history available.'
	context='Song History:\n';context+=A
	for(idx,song)in enumerate(st.session_state.identified_songs,1):
		context+=f"{idx}. {song[_F]} by {song[_I]}\n";context+=f"   - Timestamp: {song[_M]}\n"
		if _G in song and _H in song[_G]:
			track=song[_G][_H]
			if _J in track:genre=track[_J].get(_e,_D);context+=f"   - Genre: {genre}\n"
			if _P in track:
				for section in track[_P]:
					if section.get(_R)==_u:
						metadata=section.get(_v,[])
						for meta in metadata:
							if meta.get(_F)=='Album':context+=f"   - Album: {meta.get(_L,_D)}\n"
							elif meta.get(_F)==_w:release_year=meta.get(_L,_D);context+=f"   - Released: {release_year}\n"
		context+=_i
	context+=A;context+=f"Total songs: {len(st.session_state.identified_songs)}\n";genres={}
	for song in st.session_state.identified_songs:
		if _G in song and _H in song[_G]:
			track=song[_G][_H]
			if _J in track:genre=track[_J].get(_e,_D);genres[genre]=genres.get(genre,0)+1
	if genres:
		context+='\nGenre Distribution:\n'
		for(genre,count)in sorted(genres.items(),key=lambda x:x[1],reverse=_A):context+=f"  - {genre}: {count} songs\n"
	artists={}
	for song in st.session_state.identified_songs:artists[song[_I]]=artists.get(song[_I],0)+1
	if artists:
		context+='\nTop Artists:\n'
		for(artist,count)in sorted(artists.items(),key=lambda x:x[1],reverse=_A)[:5]:context+=f"  - {artist}: {count} songs\n"
	return context
def prepare_song_history_data():
	'Prepare structured song history data for analysis';C='songs';B='time_periods';A='artists'
	if not st.session_state.identified_songs:return
	data={C:[],_J:{},A:{},B:{},_AD:len(st.session_state.identified_songs)}
	for song in st.session_state.identified_songs:
		song_data={_F:song[_F],_I:song[_I],_M:song[_M]};genre=_D
		if _G in song and _H in song[_G]:
			track=song[_G][_H]
			if _J in track:genre=track[_J].get(_e,_D)
		song_data['genre']=genre;data[C].append(song_data);data[_J][genre]=data[_J].get(genre,0)+1;data[A][song[_I]]=data[A].get(song[_I],0)+1;release_year=_D
		if _G in song and _H in song[_G]:
			track=song[_G][_H]
			if _P in track:
				for section in track[_P]:
					if section.get(_R)==_u:
						metadata=section.get(_v,[])
						for meta in metadata:
							if meta.get(_F)==_w:
								release_year=meta.get(_L,_D);year_match=re.search('\\d{4}',release_year)
								if year_match:release_year=year_match.group(0)
		song_data['release_year']=release_year
		if release_year!=_D:data[B][release_year]=data[B].get(release_year,0)+1
	data[_J]=dict(sorted(data[_J].items(),key=lambda x:x[1],reverse=_A));data[A]=dict(sorted(data[A].items(),key=lambda x:x[1],reverse=_A));data[B]=dict(sorted(data[B].items(),key=lambda x:x[0]));return data
def get_audio_devices():
	'Get available audio input devices';C='maxInputChannels';B='channels';A='index';browser_device=[{A:0,_c:'Browser Microphone (Default)',B:1,_x:RATE}]
	if not PYAUDIO_AVAILABLE:return browser_device
	try:
		audio=pyaudio.PyAudio();devices=[];device_count=audio.get_device_count()
		for i in range(device_count):
			try:
				device_info=audio.get_device_info_by_index(i)
				if device_info.get(C)>0:devices.append({A:i,_c:device_info.get(_c,f"Device {i}"),B:device_info.get(C),_x:device_info.get('defaultSampleRate'),'host_api':device_info.get('hostApi')})
			except:continue
		audio.terminate();return browser_device+devices if devices else browser_device
	except Exception:return browser_device
def record_audio_browser_simple(show_progress=_A):
	'Simple browser recording with st-audiorec'
	try:from st_audiorec import st_audiorec as audio_recorder_func
	except ImportError:st.error('⚠️ Package not installed: pip install st-audiorec');return
	if show_progress:st.info("🎤 Click 'Start recording' button below...")
	try:wav_audio_data=audio_recorder_func()
	except Exception as e:st.error(f"Recording error: {str(e)}");return
	if wav_audio_data is not _B:
		try:
			temp_file=tempfile.NamedTemporaryFile(delete=_C,suffix=_m);temp_file.write(wav_audio_data);temp_file.close()
			if show_progress:st.success(_y)
			return temp_file.name
		except Exception as e:st.error(f"File save error: {str(e)}");return
def record_audio_browser(duration=RECORD_SECONDS,show_progress=_A,key='browser_rec'):
	'Record audio from browser microphone'
	if mic_recorder is _B:st.error('⚠️ Please install: pip install streamlit-mic-recorder');return
	try:
		if show_progress:st.info(f"🎤 Click the button to record {duration} seconds of audio...")
		audio_data=mic_recorder(start_prompt='⏺️ Start Recording',stop_prompt='⏹️ Stop Recording',just_once=_A,use_container_width=_A,key=key)
		if audio_data:
			audio_bytes=audio_data['bytes'];sample_rate=audio_data[_x];temp_file=tempfile.NamedTemporaryFile(delete=_C,suffix=_m);temp_file.write(audio_bytes);temp_file.close()
			if show_progress:st.success(f"✅ Recording completed! (Sample rate: {sample_rate} Hz)")
			return temp_file.name
		else:
			if show_progress:st.warning('⏸️ No audio recorded yet. Click the button to start.')
			return
	except Exception as e:
		if show_progress:st.error(f"Recording error: {str(e)}")
		return
def record_audio_browser_advanced(duration=RECORD_SECONDS,show_progress=_A):
	'Advanced browser recording with audio-recorder-streamlit'
	if audio_recorder is _B:st.error('⚠️ Please install: pip install audio-recorder-streamlit');return
	if show_progress:st.info(f"🎤 Recording {duration} seconds...")
	audio_bytes=audio_recorder(energy_threshold=(-1.,1.),pause_threshold=duration,text='',recording_color='#e8b62c',neutral_color='#6aa36f',icon_name='microphone',icon_size='3x')
	if audio_bytes:
		temp_file=tempfile.NamedTemporaryFile(delete=_C,suffix=_m);temp_file.write(audio_bytes);temp_file.close()
		if show_progress:st.success(_y)
		return temp_file.name
def analyze_audio_quality(audio_data):
	'Analyze audio quality and provide feedback'
	try:
		if isinstance(audio_data,bytes):audio_array=np.frombuffer(audio_data,dtype=np.int16)
		else:audio_array=audio_data
		max_amplitude=np.max(np.abs(audio_array));volume_level=max_amplitude/32767.
		if volume_level<.1:return'🔴 VERY LOW - Audio too quiet! Increase volume significantly.'
		elif volume_level<.3:return'🟡 LOW - Audio quiet. Try increasing volume.'
		elif volume_level<.7:return'🟢 GOOD - Audio level looks good!'
		elif volume_level<.9:return'🔵 HIGH - Good strong audio signal!'
		else:return'🟠 VERY HIGH - Audio might be clipping. Try reducing volume slightly.'
	except Exception:return'Could not analyze audio quality'
def calculate_optimal_gain(test_audio_data):
	'Calculate optimal gain based on audio level analysis'
	try:
		if isinstance(test_audio_data,bytes):audio_array=np.frombuffer(test_audio_data,dtype=np.int16)
		else:audio_array=test_audio_data
		max_amplitude=np.max(np.abs(audio_array));volume_level=max_amplitude/32767.;target_level=.6
		if volume_level>.01:optimal_gain=target_level/volume_level;optimal_gain=max(.1,min(5.,optimal_gain));return optimal_gain
		else:return 2.
	except Exception:return 1.
async def identify_song(audio_file_path):
	'Identify song using Shazam API'
	try:
		with warnings.catch_warnings():
			warnings.simplefilter(_V);shazam=Shazam()
			if not os.path.exists(audio_file_path):return
			file_size=os.path.getsize(audio_file_path)
			if file_size<1000:return
			for attempt in range(2):
				try:
					with open(audio_file_path,'rb')as audio_file:audio_data=audio_file.read()
					out=await shazam.recognize(audio_data)
					if out and _H in out:return out
					elif out:st.warning('Shazam responded but no song found in audio.')
				except Exception:
					if attempt==0:await asyncio.sleep(1);continue
			return
	except Exception:return
def display_song_info(song_data,compact=_C):
	'Display identified song information in 2-column layout';D='200px';C='150px';B='coverart';A='images'
	if not song_data or _H not in song_data:st.warning('🚫 No song identified.');return
	track=song_data[_H]
	if compact:col1,col2=st.columns([1,2])
	else:col1,col2=st.columns([1,2])
	with col1:
		if A in track and B in track[A]:st.image(track[A][B],width=150 if compact else 200)
		else:st.markdown(f'''
            <div style="width: {C if compact else D}; height: {C if compact else D}; 
                        background-color: #f0f2f6; border-radius: 8px; 
                        display: flex; align-items: center; justify-content: center;">
                <span style="font-size: {"24px"if compact else"32px"};">🎵</span>
            </div>
            ''',unsafe_allow_html=_A)
	with col2:
		if not compact:st.subheader('🎵 Song Details')
		st.markdown(f"**🎵 Title:** {track.get(_F,_D)}");st.markdown(f"**🎤 Artist:** {track.get(_z,_D)}");album=_D;released=_D
		if _P in track:
			for section in track[_P]:
				if section.get(_R)==_u:
					metadata=section.get(_v,[])
					for meta in metadata:
						if meta.get(_F)=='Album':album=meta.get(_L,_D)
						elif meta.get(_F)==_w:released=meta.get(_L,_D)
		st.markdown(f"**💿 Album:** {album}");st.markdown(f"**📅 Released:** {released}")
		if _J in track:genres=track[_J].get(_e,_D);st.markdown(f"**🎭 Genre:** {genres}")
def process_audio_file(audio_file):
	'Process audio file for song identification with enhanced cleanup'
	with st.spinner(_A0):
		try:
			if not os.path.exists(audio_file)or os.path.getsize(audio_file)==0:st.error('Invalid audio file. Please try recording again.');return
			loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
			try:result=loop.run_until_complete(identify_song(audio_file))
			finally:loop.close()
			if result:st.session_state.current_song_result=result;song_info={_M:datetime.now().strftime(_n),_F:result.get(_H,{}).get(_F,_D),_I:result.get(_H,{}).get(_z,_D),_G:result};st.session_state.identified_songs.append(song_info);st.rerun()
			else:st.session_state.current_song_result=_B;st.error('No song identified. Try recording again with music playing clearly.')
		except Exception as e:st.session_state.current_song_result=_B;st.error('Song identification failed. Please try again.')
		finally:
			try:
				if os.path.exists(audio_file):os.unlink(audio_file)
			except Exception:pass
def history_tab():
	'Enhanced song history interface with 2-column layout throughout';B='**📋 History Features:**';A='selected_history_index';st.markdown('### 📚 Song History')
	if st.session_state.identified_songs:
		col1,col2=st.columns([3,1])
		with col1:st.write(f"**Total songs identified:** {len(st.session_state.identified_songs)}")
		with col2:
			if st.button('🗑️ Clear History'):
				st.session_state.identified_songs=[];st.session_state.selected_song=_B
				if A in st.session_state:del st.session_state.selected_history_index
				st.rerun()
		st.markdown(_O);songs=list(reversed(st.session_state.identified_songs))
		if len(songs)>0:
			st.markdown('### 🆕 Latest Song')
			with st.container():st.markdown('*Most recently identified*');display_song_info(songs[0][_G]);st.markdown(f"🕒 **Identified:** {songs[0][_M]}")
			st.markdown(_O)
			if len(songs)>1:
				st.markdown('### 📋 Previous Songs')
				if A not in st.session_state:st.session_state.selected_history_index=_B
				main_col1,main_col2=st.columns([1,1])
				with main_col1:
					st.markdown('**📜 Song List** *(Click to view details)*')
					for(i,song)in enumerate(songs[1:],1):
						is_selected=st.session_state.selected_history_index==i
						with st.container():
							if is_selected:st.markdown('**🎵 SELECTED:**');st.markdown(f'\n                                <div style="background-color: #e8f4f8; padding: 10px; border-radius: 8px; border-left: 4px solid #1f77b4;">\n                                ',unsafe_allow_html=_A)
							song_col1,song_col2=st.columns([3,1])
							with song_col1:
								title=song[_F];artist=song[_I]
								if len(title)>30:title=title[:27]+_o
								if len(artist)>25:artist=artist[:22]+_o
								st.markdown(f"**{title}**");st.markdown(f"*by {artist}*");time_parts=song[_M].split();date_str=time_parts[0]if len(time_parts)>0 else'';time_str=time_parts[1]if len(time_parts)>1 else'';st.markdown(f"🕒 {date_str} {time_str}")
							with song_col2:
								button_label='🎵'if is_selected else'👁️';button_help='Click to close'if is_selected else'Click to view'
								if st.button(button_label,key=f"select_song_{i}",help=button_help):
									if st.session_state.selected_history_index==i:st.session_state.selected_history_index=_B;st.session_state.selected_song=_B
									else:st.session_state.selected_history_index=i;st.session_state.selected_song=song
									st.rerun()
							if is_selected:st.markdown('</div>',unsafe_allow_html=_A)
							st.markdown('<div style="height: 15px;"></div>',unsafe_allow_html=_A)
				with main_col2:
					st.markdown('**🎵 Song Details**')
					if st.session_state.selected_history_index is not _B and st.session_state.selected_song:
						display_song_info(st.session_state.selected_song[_G],compact=_C);st.markdown(_O);st.markdown(f"🕒 **Identified:** {st.session_state.selected_song[_M]}")
						if st.button('✖️ Close Details',key='close_details'):st.session_state.selected_history_index=_B;st.session_state.selected_song=_B;st.rerun()
					else:st.info('👈 Select a song from the list to view full details');st.markdown(B);st.markdown('• 🆕 Latest song shown above in full detail');st.markdown('• 👁️ Click any song to view complete information');st.markdown('• 🎵 Selected songs are highlighted');st.markdown('• ✖️ Close details to return to list view');st.markdown('• 🗑️ Clear all history when needed')
	else:st.info('📭 No songs identified yet. Use the Song ID tab to identify your first song!');st.markdown('**🚀 Get Started:**');st.markdown("1. 🎤 Go to the 'Song ID' tab");st.markdown('2. 🎵 Play music near your microphone');st.markdown("3. 📱 Click 'Start Recording & Identify Song'");st.markdown('4. 📚 Return here to view your song history');st.markdown(B);st.markdown('• 🖼️ Album artwork and song details in organized layout');st.markdown('• 🆕 Latest songs displayed prominently');st.markdown('• 📜 Browsable list of all identified songs');st.markdown('• 🔗 Direct links to streaming platforms');st.markdown('• 🕒 Timestamp tracking for each identification')
def auto_mode_tab():
	'Auto mode interface - browser-based';G='❌ Recording failed';F='interval';E='⏳ Next Check';D='last_check';C='last_song_key';B='songs_identified';A='errors'
	if'auto_rec_counter'not in st.session_state:st.session_state.auto_rec_counter=0
	if'last_refresh'not in st.session_state:st.session_state.last_refresh=time.time()
	st.markdown('### 📻 Auto Radio Station Monitoring');col1,col2,col3=st.columns([1,1,1])
	with col1:
		st.markdown('#### 🛠️ Auto Mode Settings');st.info('🌐 Using browser microphone (works anywhere)');station_name=st.text_input('📻 Station Name:',value=_AF,help='Name for the radio station being monitored');auto_interval=st.slider('🔄 Auto-Identify Every (seconds):',min_value=150,max_value=600,value=240,help='How often to automatically identify songs');interval_minutes=auto_interval/60
		if interval_minutes>=1:st.caption(f"⏱️ Interval: {interval_minutes:.1f} minutes")
		if _AB not in st.session_state:st.session_state.auto_mode_active=_C
		if _AC not in st.session_state:st.session_state.auto_mode_stats={_Z:_B,B:0,C:_B,A:0,D:_B}
		if'auto_mode_message'not in st.session_state:st.session_state.auto_mode_message=''
		st.markdown(_O)
		if not st.session_state.auto_mode_active:
			if st.button('🚀 Start Auto Monitoring',type=_e,key='start_auto'):st.session_state.auto_mode_active=_A;st.session_state.auto_mode_stats={_Z:datetime.now(),B:0,C:_B,A:0,D:datetime.now(),F:auto_interval,_AE:station_name};st.session_state.auto_mode_message='🚀 Auto monitoring started!';st.success('Auto monitoring started!');st.rerun()
		elif st.button('⏹️ Stop Auto Monitoring',type='secondary',key='stop_auto'):st.session_state.auto_mode_active=_C;st.session_state.auto_mode_message='⏹️ Auto monitoring stopped';st.info('Auto monitoring stopped!');st.rerun()
		if st.button('🔄 Reset Stats',key='reset_auto_stats'):st.session_state.auto_mode_stats={_Z:_B,B:0,C:_B,A:0,D:_B};st.success('Statistics reset!');st.rerun()
	with col2:
		st.markdown('#### 📊 Live Monitoring Status')
		if st.session_state.auto_mode_active:
			if st.session_state.auto_mode_message:st.info(st.session_state.auto_mode_message)
			current_time=datetime.now();start_time=st.session_state.auto_mode_stats.get(_Z,current_time);last_check=st.session_state.auto_mode_stats.get(D,start_time);elapsed=current_time-start_time;elapsed_str=f"{int(elapsed.total_seconds()//60):02d}:{int(elapsed.total_seconds()%60):02d}";interval=st.session_state.auto_mode_stats.get(F,150);time_since_check=(current_time-last_check).total_seconds();time_to_next=max(0,interval-time_since_check);st.metric('⏰ Running Time',elapsed_str);st.metric('🎵 Songs Identified',st.session_state.auto_mode_stats.get(B,0))
			if time_to_next>0:
				next_minutes=int(time_to_next//60);next_seconds=int(time_to_next%60)
				if next_minutes>0:st.metric(E,f"{next_minutes}m {next_seconds:02d}s")
				else:st.metric(E,f"{next_seconds}s")
			else:st.metric(E,'NOW!')
			if time_to_next<=0 and st.session_state.auto_mode_active:
				st.session_state.auto_mode_message='🎤 Recording and identifying...'
				try:
					st.session_state.auto_rec_counter+=1
					with st.container():st.info('🎤 Auto-recording in progress...');audio_file=record_audio_browser(duration=AUTO_RECORD_SECONDS,show_progress=_C,key=f"auto_rec_{st.session_state.auto_rec_counter}")
					if audio_file and os.path.exists(audio_file):
						status_placeholder.info(_A0);loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
						try:result=loop.run_until_complete(identify_song(audio_file))
						finally:loop.close()
						if result and _H in result:
							track=result[_H];title=track.get(_F,_D);artist=track.get(_z,_D);song_key=f"{title.lower().strip()}|{artist.lower().strip()}";st.session_state.current_song_result=result
							if song_key!=st.session_state.auto_mode_stats.get(C):song_info={_M:datetime.now().strftime(_n),_F:title,_I:artist,_G:result};st.session_state.identified_songs.append(song_info);st.session_state.auto_mode_stats[B]+=1;st.session_state.auto_mode_stats[C]=song_key;st.session_state.auto_mode_message=f"🆕 NEW: {title} by {artist}";status_placeholder.success(f"🆕 New song identified: {title}")
							else:st.session_state.auto_mode_message=f"🔄 SAME: {title} by {artist}";status_placeholder.info('🔄 Same song still playing')
						else:st.session_state.current_song_result=_B;st.session_state.auto_mode_message='🚫 No song identified';st.session_state.auto_mode_stats[A]=st.session_state.auto_mode_stats.get(A,0)+1;status_placeholder.warning('🚫 Could not identify song')
						try:os.unlink(audio_file)
						except:pass
					else:st.session_state.current_song_result=_B;st.session_state.auto_mode_message=G;st.session_state.auto_mode_stats[A]=st.session_state.auto_mode_stats.get(A,0)+1;status_placeholder.error(G)
				except Exception as e:st.session_state.current_song_result=_B;st.session_state.auto_mode_message=f"❌ Error: {str(e)}";st.session_state.auto_mode_stats[A]=st.session_state.auto_mode_stats.get(A,0)+1;status_placeholder.error(f"❌ Identification failed: {str(e)}")
				st.session_state.auto_mode_stats[D]=datetime.now();time.sleep(2);status_placeholder.empty();st.rerun()
		else:st.info('⏸️ Auto monitoring is inactive');st.markdown('**🚀 How Auto Mode Works:**');st.markdown('• ⏰ Waits for specified interval');st.markdown('• 🎤 Records 8 seconds of audio');st.markdown('• 🔍 Identifies the song using Shazam');st.markdown('• 🆕 Adds NEW songs to history');st.markdown('• 🔄 Shows if same song is still playing')
		if st.session_state.auto_mode_stats.get(A,0)>0:st.warning(f"⚠️ Errors encountered: {st.session_state.auto_mode_stats[A]}")
	with col3:
		st.markdown('### 🎵 Recent Songs')
		if st.session_state.identified_songs:
			recent_songs=list(reversed(st.session_state.identified_songs[-5:]));st.markdown(f"**📋 Last {len(recent_songs)} Songs Identified**")
			for(i,song)in enumerate(recent_songs):
				with st.container():
					timestamp_parts=song[_M].split(' ');time_str=timestamp_parts[1]if len(timestamp_parts)>1 else song[_M];title=song[_F];artist=song[_I]
					if len(title)>25:title=title[:22]+_o
					if len(artist)>20:artist=artist[:17]+_o
					if i==0:st.markdown(f'''
                        <div style="background-color: #e8f4f8; padding: 8px; border-radius: 6px; border-left: 3px solid #1f77b4; margin-bottom: 8px;">
                            <div style="font-weight: bold; color: #1f77b4;">🆕 LATEST</div>
                            <div style="font-weight: bold; font-size: 14px;">{title}</div>
                            <div style="color: #666; font-size: 12px;">{artist}</div>
                            <div style="color: #888; font-size: 11px;">🕒 {time_str}</div>
                        </div>
                        ''',unsafe_allow_html=_A)
					else:st.markdown(f'''
                        <div style="padding: 6px 8px; border-bottom: 1px solid #eee; margin-bottom: 4px;">
                            <div style="font-weight: 500; font-size: 13px;">{title}</div>
                            <div style="color: #666; font-size: 11px;">{artist}</div>
                            <div style="color: #888; font-size: 10px;">🕒 {time_str}</div>
                        </div>
                        ''',unsafe_allow_html=_A)
		else:
			st.info('📭 No songs identified yet')
			if not st.session_state.auto_mode_active:st.markdown('**🚀 Start Auto Mode to begin identifying songs automatically!**');st.markdown(_O);st.markdown('**📋 This list will show:**');st.markdown('• 🆕 Most recent song highlighted');st.markdown('• 🎵 Song title and artist');st.markdown('• 🕒 Time of identification');st.markdown('• 📊 Last 10 songs identified')
			else:st.markdown('**🎤 Auto mode is running...**');st.markdown("Songs will appear here as they're identified!")
	if st.session_state.auto_mode_active:
		current_time=time.time();elapsed=current_time-st.session_state.last_refresh
		if elapsed<5:sleep_time=1
		elif elapsed<30:sleep_time=2
		else:sleep_time=5
		time.sleep(sleep_time);st.session_state.last_refresh=time.time();st.rerun()
def main_tab():
	'Main song identification interface - Browser recording only'
	if'recording_start_time'not in st.session_state:st.session_state.recording_start_time=_B
	if'is_recording'not in st.session_state:st.session_state.is_recording=_C
	if'auto_stop_triggered'not in st.session_state:st.session_state.auto_stop_triggered=_C
	col1,col2=st.columns([1,1])
	with col1:
		st.markdown('### 🎤 Browser Microphone Recording');st.info('🌐 Works on any device - Desktop, Tablet, or Mobile')
		try:
			from st_audiorec import st_audiorec;st.markdown('#### 🎵 Record Audio');st.caption('Recording will automatically stop after 12 seconds')
			if st.session_state.is_recording and st.session_state.recording_start_time:
				elapsed=time.time()-st.session_state.recording_start_time;remaining=max(0,12-int(elapsed))
				if remaining>0:st.markdown(f"### ⏱️ Recording: {remaining} seconds remaining");progress=elapsed/12;st.progress(min(progress,1.));time.sleep(.5);st.rerun()
				elif not st.session_state.auto_stop_triggered:st.session_state.auto_stop_triggered=_A;st.info('⏹️ 12 seconds reached - Processing recording...')
			audio_bytes=st_audiorec()
			if audio_bytes is _B and not st.session_state.is_recording:0
			elif audio_bytes is not _B and not st.session_state.is_recording:st.session_state.is_recording=_A;st.session_state.recording_start_time=time.time();st.session_state.auto_stop_triggered=_C;st.rerun()
			if audio_bytes is not _B:
				elapsed_time=0
				if st.session_state.recording_start_time:elapsed_time=time.time()-st.session_state.recording_start_time
				if elapsed_time>=12 or st.session_state.auto_stop_triggered:
					audio_hash=hashlib.md5(audio_bytes).hexdigest()
					if'last_processed_audio'not in st.session_state:st.session_state.last_processed_audio=_B
					if st.session_state.last_processed_audio!=audio_hash:
						st.session_state.last_processed_audio=audio_hash;st.session_state.is_recording=_C;st.session_state.recording_start_time=_B;st.session_state.auto_stop_triggered=_C
						try:
							temp_file=tempfile.NamedTemporaryFile(delete=_C,suffix=_m);temp_file.write(audio_bytes);temp_file.close();st.success(_y)
							with st.spinner(_A0):process_audio_file(temp_file.name)
						except Exception as e:st.error(f"Error processing audio: {str(e)}");st.session_state.is_recording=_C;st.session_state.recording_start_time=_B
		except ImportError as e:st.error('❌ Browser audio recorder not available!');st.markdown('### 📦 Installation Required');st.markdown('The `st-audiorec` package is needed for browser recording.');st.code('pip install st-audiorec',language=_p);st.markdown('**Then restart the Streamlit app:**');st.code('streamlit run AsongAIq.py',language=_p);st.markdown(_O);st.markdown('**Package Info:**');st.markdown('- 🌐 Works in any browser');st.markdown('- 📱 Mobile-friendly');st.markdown("- 🔒 Secure (uses browser's Media API)")
		except Exception as e:st.error(f"Recording error: {str(e)}");st.exception(e)
		st.markdown(_O);st.markdown('**💡 Tips for Best Results:**');st.markdown('• 📊 Play music clearly near your device');st.markdown('• 🤫 Minimize background noise');st.markdown('• 🎵 Record during chorus or distinctive parts');st.markdown('• ⏱️ **Auto-stops at 12 seconds** - No manual stop needed!');st.markdown('• 🔒 Works on HTTPS or localhost only')
	with col2:
		st.markdown('### 🎵 Current Song Result')
		if hasattr(st.session_state,_AA)and st.session_state.current_song_result:display_song_info(st.session_state.current_song_result)
		else:st.info('🎯 Song identification results will appear here after recording.');st.markdown('**How it works:**');st.markdown("1. 🎤 Click 'Start recording'");st.markdown('2. ⏺️ Record 10-15 seconds of music');st.markdown("3. ⏹️ Click 'Stop recording'");st.markdown('4. 🔍 AI identifies the song');st.markdown('5. 📋 Results appear here!')
def generate_analysis_prompt(depth):
	'Generate analysis prompt based on selected depth'
	if depth==_h:return'Provide a quick overview of my song history. Highlight key trends and top artists.'
	elif depth==_A1:return'Provide a detailed analysis of my song history. Include trends in genres, time periods, and artist diversity. Also analyze listener personality and musical preferences.'
	elif depth==_A2:return'Perform a deep dive analysis of my song history. Explore emotional resonance, social context, lifestyle alignment, and musical sophistication. Include recommendations for similar artists.'
	return'Analyze my song history.'
def generate_ai_analysis_response(response_text):
	'Process AI response for structured display';recommendations=[];rec_match=re.search('## Recommendations:(.*?)(?=##|$)',response_text,re.DOTALL|re.IGNORECASE)
	if rec_match:
		rec_text=rec_match.group(1).strip();rec_items=re.split('\\d+\\.',rec_text)
		for item in rec_items:
			if not item.strip():continue
			title_match=re.search('"(.*?)"',item);artist_match=re.search('by\\s+(.*?)(?:\\n|$)',item);reason_match=re.search('Reason:(.*?)$',item,re.DOTALL)
			if title_match and artist_match:rec={_F:title_match.group(1),_I:artist_match.group(1).strip(),_A3:reason_match.group(1).strip()if reason_match else'No reason provided'};recommendations.append(rec)
	sections={};section_titles=re.findall('## (.*?)\\n',response_text)
	for title in section_titles:
		section_match=re.search('## '+re.escape(title)+'\\n(.*?)(?=##|$)',response_text,re.DOTALL)
		if section_match:sections[title.strip()]=section_match.group(1).strip()
	return{_f:recommendations,_P:sections,'full_response':response_text}
def export_analysis():
	'Export analysis results as text file'
	if not st.session_state.analysis_results:return
	output=io.StringIO();output.write('RadioSport AI Analysis Report\n');output.write('='*40+'\n\n');output.write(f"Generated: {datetime.now().strftime(_n)}\n");output.write(f"Analysis Depth: {st.session_state.analysis_depth}\n\n")
	for(title,content)in st.session_state.analysis_results.get(_P,{}).items():output.write(f"{title.upper()}\n");output.write('-'*len(title)+_i);output.write(f"{content}\n\n")
	if st.session_state.analysis_results.get(_f):
		output.write('RECOMMENDATIONS\n');output.write('='*40+_i)
		for(i,rec)in enumerate(st.session_state.analysis_results[_f],1):output.write(f'{i}. "{rec[_F]}" by {rec[_I]}\n');output.write(f"   Reason: {rec[_A3]}\n\n")
	return output.getvalue()
def export_chat_history():
	'Export chat history as text file';A='USER'
	if not st.session_state.messages:return
	output=io.StringIO();output.write('RadioSport AI Chat History\n');output.write('='*40+'\n\n');output.write(f"Generated: {datetime.now().strftime(_n)}\n\n")
	for msg in st.session_state.messages:role=A if msg[_K]==_U else'AI';content=msg.get(_A4,msg[_E])if role==A else msg[_E];output.write(f"{role}:\n");output.write(f"{content}\n\n");output.write('-'*40+'\n\n')
	return output.getvalue()
def ai_analysis_tab():
	'AI analysis tab for song history insights with all new features';C='text/plain';B='choices';A='assistant';st.markdown('### 🧠 AI Song History Analysis')
	with st.expander('📋 Song History Context',expanded=_C):history_context=get_song_history_context();st.text(history_context)
	st.markdown('#### 🔍 Analysis Depth');analysis_col1,analysis_col2,analysis_col3=st.columns(3)
	with analysis_col1:
		if st.button('🚀 Quick Overview',use_container_width=_A):st.session_state.analysis_depth=_h;st.session_state.auto_prompt=generate_analysis_prompt(_h);st.rerun()
	with analysis_col2:
		if st.button('🔎 Detailed Analysis',use_container_width=_A):st.session_state.analysis_depth=_A1;st.session_state.auto_prompt=generate_analysis_prompt(_A1);st.rerun()
	with analysis_col3:
		if st.button('🌊 Deep Dive',use_container_width=_A):st.session_state.analysis_depth=_A2;st.session_state.auto_prompt=generate_analysis_prompt(_A2);st.rerun()
	st.caption(f"Selected: **{st.session_state.analysis_depth}**")
	if st.session_state.analysis_results:
		st.markdown('#### 📊 Analysis Results');st.markdown(f"*Depth: {st.session_state.analysis_depth}*")
		for(title,content)in st.session_state.analysis_results.get(_P,{}).items():
			with st.expander(f"📌 {title}",expanded=_A):st.markdown(content)
		if st.session_state.analysis_results.get(_f):
			st.markdown('#### 🎵 Recommendations');st.info('Based on your song history, you might enjoy these songs:')
			for rec in st.session_state.analysis_results[_f]:
				with st.container():st.markdown(f'''
                    <div class="recommendation-card">
                        <div class="recommendation-title">"{rec[_F]}"</div>
                        <div class="recommendation-artist">by {rec[_I]}</div>
                        <div class="recommendation-reason">💡 {rec[_A3]}</div>
                    </div>
                    ''',unsafe_allow_html=_A)
	st.markdown('#### 💬 Quick Questions');q_cols=st.columns(4)
	for(i,question)in enumerate(st.session_state.quick_questions):
		with q_cols[i%4]:
			if st.button(question,key=f"q_{i}"):st.session_state.auto_prompt=question;st.rerun()
	st.markdown('#### 💬 AI Chat')
	for msg in st.session_state.messages:
		if msg[_K]==_U:
			with st.chat_message(_U):prompt=msg.get(_A4,msg[_E]);st.markdown(prompt,unsafe_allow_html=_A)
		elif msg[_K]==A:
			with st.chat_message(A):
				parts=split_message(msg[_E])
				for part in parts:
					if part[_R]==_L:st.markdown(part[_E],unsafe_allow_html=_A)
					elif part[_R]==_Y:
						with st.expander('View Code',expanded=_C):st.code(part[_Y],language=part['language'])
	prompt=st.chat_input('Ask about your song history...',key='ai_input')
	if'auto_prompt'in st.session_state:prompt=st.session_state.auto_prompt;del st.session_state.auto_prompt
	selected_model=st.session_state.get(_AK,DEFAULT_MODEL)
	if prompt and selected_model:
		history_context=get_song_history_context();full_prompt=f"{history_context}\n\nUser Question: {prompt}";message={_K:_U,_E:full_prompt,_A4:prompt,_t:selected_model};st.session_state.messages.append(message)
		with st.chat_message(_U):st.markdown(prompt,unsafe_allow_html=_A)
		messages=[];messages.append({_K:'system',_E:"You are an AI music analyst. Analyze the user's song history and provide insights."})
		for msg in st.session_state.messages[-6:]:
			if msg[_K]==_U:messages.append({_K:_U,_E:msg[_E]})
			elif msg[_K]==A:messages.append({_K:A,_E:msg[_E]})
		with st.chat_message(A):
			response_placeholder=st.empty();accumulated_response=''
			try:
				if st.session_state.api_provider==_Q:
					api_payload={_t:selected_model,_r:messages,'stream':_A};response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_AJ:_s},stream=_A,timeout=300);response.raise_for_status()
					for line in response.iter_lines():
						if line:
							try:
								data=json.loads(line.decode(_d))
								if _k in data and _E in data[_k]:content_chunk=data[_k][_E];accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
							except json.JSONDecodeError:continue
				elif st.session_state.api_provider==_T:
					api_key=st.session_state.api_keys[_N]
					if not api_key:st.error('OpenRouter API key is missing!');return
					try:response=call_openrouter_api(messages,selected_model,api_key)
					except Exception as e:st.error(f"API Error: {str(e)}");st.session_state.messages.append({_K:A,_E:f"Error: {str(e)}"});response_placeholder.markdown(f"**API Error**: {str(e)}",unsafe_allow_html=_A);return
					try:
						for chunk in response.iter_lines():
							if chunk:
								if chunk==b'':continue
								if chunk.startswith(b'data:'):
									try:
										data=json.loads(chunk.decode(_d)[5:])
										if B in data and len(data[B])>0:delta=data[B][0].get('delta',{});content_chunk=delta.get(_E,'');accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
									except json.JSONDecodeError:pass
					except Exception as e:st.error(f"Error processing stream: {str(e)}")
				response_msg={_K:A,_E:accumulated_response,_M:datetime.now().isoformat()};st.session_state.messages.append(response_msg)
				if'analyze'in prompt.lower()or'overview'in prompt.lower()or'dive'in prompt.lower():st.session_state.analysis_results=generate_ai_analysis_response(accumulated_response);st.rerun()
				if not accumulated_response.strip():st.error('The model did not generate any response. Please try again.')
			except requests.RequestException as e:
				error_message=f"Error communicating with API: {e}"
				if'401'in str(e):error_message+='\n\n⚠️ Invalid API Key - Please check your credentials'
				elif'429'in str(e):error_message+='\n\n⚠️ Rate limit exceeded - Try again later'
				st.error(error_message);st.session_state.messages.append({_K:A,_E:f"Error: {e}"});response_placeholder.markdown(error_message,unsafe_allow_html=_A)
	st.markdown(_O);export_col1,export_col2=st.columns(2)
	with export_col1:
		if st.button('💾 Export Analysis Report',use_container_width=_A):
			report=export_analysis()
			if report:st.download_button(label='Download Analysis',data=report,file_name='radio_sport_analysis.txt',mime=C)
			else:st.warning('No analysis to export')
	with export_col2:
		if st.button('📥 Export Chat History',use_container_width=_A):
			chat_history=export_chat_history()
			if chat_history:st.download_button(label='Download Chat',data=chat_history,file_name='radio_sport_chat.txt',mime=C)
			else:st.warning('No chat history to export')
def main():
	'Main application with enhanced tabbed interface';E='cache-miss';D='cache-hit';C='context_length';B='openrouter_models';A='AI provider:';initialize_session_state()
	with st.sidebar:
		st.markdown('<div class="sidebar-title">RadioSport AI 🧟🎵</div>',unsafe_allow_html=_A);st.markdown(f'<div class="version-text">Version {APP_VERSION}</div>',unsafe_allow_html=_A)
		if IS_CLOUD_ENVIRONMENT:st.subheader('☁️Cloud🌥️')
		else:st.subheader(_Q)
		st.markdown(_O);st.subheader('🎵 Song ID Settings');col1,col2=st.columns([2,1])
		with col1:
			if st.button('🔄 Audio Devices',use_container_width=_A):st.session_state.audio_devices=get_audio_devices();st.rerun()
		st.markdown(_O);st.subheader('🧠 AI Engine Settings')
		if IS_CLOUD_ENVIRONMENT:api_provider=_T;st.session_state.api_provider=_T;st.radio(A,[_T],index=0,key=_g,disabled=_A,help='Local AI not available in cloud environment')
		else:provider_options=[_Q,_T];current_provider=st.session_state.get(_g,_Q);current_index=0 if current_provider==_Q else 1;api_provider=st.radio(A,provider_options,index=current_index,key=_g)
		if api_provider==_Q:
			col1,col2=st.columns([2,1])
			with col1:
				if st.button('Refresh Models',use_container_width=_A):get_ollama_models_cached.clear();st.session_state.ollama_models=get_ollama_models();st.rerun()
			with col2:auto_refresh=st.checkbox('Auto',help='Auto-refresh models')
			if not st.session_state.ollama_models or auto_refresh:st.session_state.ollama_models=get_ollama_models()
			else:st.session_state.cache_stats[_S]+=1
			ollama_models=st.session_state.ollama_models
			if ollama_models:
				default_index=0
				if DEFAULT_MODEL in ollama_models:default_index=ollama_models.index(DEFAULT_MODEL)
				selected_model=st.selectbox('Select a model:',ollama_models,index=default_index,help=f"Available models: {len(ollama_models)}",key=_AK)
			else:st.warning(f"Models Unavailable: {st.session_state.get(_X,OLLAMA_HOST)}.");selected_model=_B
		if api_provider==_Q and not IS_CLOUD_ENVIRONMENT:
			with st.expander('🔧 Local Settings',expanded=_C):
				if _X not in st.session_state:st.session_state.ollama_host=DEFAULT_OLLAMA_HOST
				new_host=st.text_input('Ollama Host URL:',value=st.session_state.ollama_host,help='Default: http://localhost:11434\nRemote: http://192.168.x.x:11434',placeholder=_A7,key='ollama_host_input')
				if new_host!=st.session_state.ollama_host:
					st.session_state.ollama_host=new_host;update_all_ollama_host_references(new_host);get_ollama_models_cached.clear();st.session_state.ollama_models=[]
					try:new_models=get_ollama_models_cached();st.session_state.ollama_models=new_models;st.success(f"✅ Connected to {new_host} - Found {len(new_models)} models")
					except Exception as e:st.error(f"❌ Failed to connect to {new_host}: {str(e)}");st.session_state.ollama_models=[]
					st.rerun()
				if st.session_state.ollama_host:st.info(f"Current host: {st.session_state.ollama_host}")
		elif api_provider==_T:
			try:
				api_key=get_load_balanced_api_key(_N)
				if api_key:st.session_state.api_keys[_N]=api_key
				else:raise KeyError('No OPENROUTER API keys found')
			except:st.session_state.api_keys[_N]=st.text_input('OpenRouter API Key (free tier)',type='password',value=st.session_state.api_keys.get(_N,''),help='Get free API key at https://openrouter.ai')
			if B not in st.session_state:
				with st.spinner('Loading available models...'):st.session_state.openrouter_models=get_free_openrouter_models()
			available_models=st.session_state.openrouter_models
			if not available_models:
				st.error('Could not load models. Please refresh the page to try again.')
				if st.button('🔄 Refresh Models'):del st.session_state.openrouter_models;st.rerun()
				st.stop()
			model_display=[f"{model.get(_c,model[_l])}"for model in available_models];col1,col2=st.columns([6,1])
			with col1:
				selected_index=st.selectbox('Models (Free)',range(len(model_display)),format_func=lambda x:model_display[x],index=0,help='Showing only free models from OpenRouter. List updates when refreshed.',key=_AH)
				try:selected_index=int(selected_index)
				except(TypeError,ValueError):selected_index=0
				if model_display and 0<=selected_index<len(model_display):
					selected_model_info=available_models[selected_index];selected_model_id=selected_model_info[_l];st.session_state.selected_model=selected_model_id
					if C in selected_model_info:context_length=selected_model_info[C];st.caption(f"Context: {context_length:,} tokens")
				else:st.error('⚠️ Invalid model selection');st.session_state.selected_model=_B
			with col2:
				if st.button('🔄'):
					if B in st.session_state:del st.session_state.openrouter_models
					st.rerun()
		with st.expander('⚙️ AI Controls',expanded=_C):
			col1,col2=st.columns(2)
			with col1:st.session_state.enable_reasoning=st.toggle('Reasoning',value=st.session_state.get('enable_reasoning',_C),help='Enable reasoning for qwen3* models');st.session_state.auto_run_plots=st.toggle('AutoPlot',value=st.session_state.get(_AG,_A),help='Automatically execute Python code that generates plots')
			with col2:st.session_state.teacher_mode=st.toggle('Teacher',value=st.session_state.get('teacher_mode',_C),help='Activate enhanced teaching mode for detailed explanations')
		if st.button('Clear AI Chat',use_container_width=_A):st.session_state.messages=[];st.rerun()
		with st.expander('📊 Cache Statistics'):
			stats=st.session_state.cache_stats;total_model_requests=stats[_S]+stats[_a];total_doc_requests=stats[_W]+stats[_b];model_hit_rate=stats[_S]/max(total_model_requests,1)*100;doc_hit_rate=stats[_W]/max(total_doc_requests,1)*100;uptime=datetime.now()-stats[_q];uptime_str=f"{uptime.days}d {uptime.seconds//3600}h {uptime.seconds%3600//60}m";stats_html=f'''
            <div class="cache-stats">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Uptime</td><td>{uptime_str}</td></tr>
                    <tr><td colspan="2"><strong>Model Cache</strong></td></tr>
                    <tr><td>Hit Rate</td><td class="{D if model_hit_rate>50 else E}">{model_hit_rate:.1f}%</td></tr>
                    <tr><td>Hits</td><td class="cache-hit">{stats[_S]}</td></tr>
                    <tr><td>Misses</td><td class="cache-miss">{stats[_a]}</td></tr>
                    <tr><td colspan="2"><strong>Document Cache</strong></td></tr>
                    <tr><td>Hit Rate</td><td class="{D if doc_hit_rate>50 else E}">{doc_hit_rate:.1f}%</td></tr>
                    <tr><td>Hits</td><td class="cache-hit">{stats[_W]}</td></tr>
                    <tr><td>Misses</td><td class="cache-miss">{stats[_b]}</td></tr>
                    <tr><td colspan="2"><strong>Cache Info</strong></td></tr>
                    <tr><td>Doc Cache Size</td><td>{len(st.session_state.doc_cache)}</td></tr>
                    <tr><td>Base64 Cache Size</td><td>{len(st.session_state.base64_cache)}</td></tr>
                </table>
            </div>
            ''';st.markdown(stats_html,unsafe_allow_html=_A)
			if st.button('Reset Stats'):st.session_state.cache_stats={_S:0,_a:0,_W:0,_b:0,_q:datetime.now()};st.rerun()
	tab1,tab2,tab3,tab4=st.tabs(['🎤 Song ID','📻 Auto Mode','📚 History','🧠 AI Analysis'])
	with tab1:main_tab()
	with tab2:auto_mode_tab()
	with tab3:history_tab()
	with tab4:ai_analysis_tab()
if __name__=='__main__':main()