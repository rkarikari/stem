_w='stream'
_v='Content-Type'
_u='selected_key'
_t='last_used'
_s='FuncAnimation'
_r='unsupported'
_q='api_provider'
_p='auto_run_plots'
_o='cache_stats'
_n='https://github.com/rkarikari/RadioSport-chat'
_m='RadioSport AI'
_l='http://localhost:11434'
_k='https://openrouter.ai/api/v1/models'
_j='api_usage'
_i='application/json'
_h='context'
_g='image/gif'
_f='original_prompt'
_e='messages'
_d='id'
_c='text/plain'
_b='files'
_a='pdf'
_Z='last_reset'
_Y='python'
_X='utf-8'
_W='language'
_V='failed'
_U='model'
_T='image'
_S='name'
_R='ollama_host'
_Q='document_cache_misses'
_P='count'
_O='user'
_N='document_cache_hits'
_M='model_cache_misses'
_L='Cloud'
_K='model_cache_hits'
_J='text'
_I='Local'
_H='type'
_G='role'
_F='code'
_E='openrouter'
_D='content'
_C=None
_B=False
_A=True
import streamlit as st,requests,subprocess,json,base64
from io import BytesIO,StringIO
import PyPDF2,os,re
from datetime import datetime
import hashlib
from functools import lru_cache
import time,random,contextlib,traceback,sys,tempfile,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,numpy as np,pandas as pd,seaborn as sns,matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
APP_VERSION='3.9.2'
OPENROUTER_API_URL='https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODELS_URL=_k
IS_CLOUD_DEPLOYMENT=os.environ.get('STREAMLIT_SHARING_MODE')or os.environ.get('DYNO')or not os.path.exists('/usr/local/bin/ollama')
DEFAULT_OLLAMA_HOST=_l
OLLAMA_HOST=DEFAULT_OLLAMA_HOST
DEFAULT_MODEL='qwen3:4b'
CACHE_TTL=300
MAX_CACHE_ENTRIES=100
MAX_REASONING_LINES=5
REASONING_UPDATE_INTERVAL=.2
CODE_EXTENSIONS={'.py':_Y,'.js':'javascript','.ts':'typescript','.cpp':'cpp','.c':'c','.java':'java','.html':'html','.css':'css','.json':'json','.md':'markdown','.sh':'bash','.sql':'sql','.bat':'batch'}
IMAGE_EXTENSIONS='.png','.jpg','.jpeg'
def initialize_ui():
	B='deepseek';A='api_keys';st.set_page_config(page_title=_m,page_icon='🧟',layout='centered',menu_items={'Report a Bug':_n,'About':'Copyright © RNK, 2025 RadioSport. All rights reserved.'});session_defaults={_o:{_K:0,_M:0,_N:0,_Q:0,_Z:datetime.now()},'file_uploader_key':'uploader_0','reasoning_window':_C,_e:[],_p:_A,'ollama_models':[],'doc_cache':{},'last_reasoning_update':time.time(),'base64_cache':{},'thinking_content':'','in_thinking_block':_B,'reasoning_window_id':f"reasoning_{time.time()}",_q:_L if IS_CLOUD_DEPLOYMENT else _I,A:{_E:''},'selected_online_model':'mistralai/mistral-7b-instruct'}
	for(key,value)in session_defaults.items():
		if key not in st.session_state:st.session_state[key]=value
	if A in st.session_state and B in st.session_state.api_keys:st.session_state.api_keys[_E]=st.session_state.api_keys[B];del st.session_state.api_keys[B]
	if A in st.session_state and _E not in st.session_state.api_keys:st.session_state.api_keys[_E]=''
	st.markdown("\n    <style>\n    .sidebar-title {\n        font-size: 24px !important;\n        font-weight: bold;\n        margin-bottom: 0px !important;\n    }\n    .version-text {\n        font-size: 12px !important;\n        margin-top: 0px !important;\n        color: #666666;\n    }\n    .code-block {\n        background-color: #f6f8fa;\n        padding: 10px;\n        border-radius: 5px;\n        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;\n        white-space: pre-wrap;\n        font-size: 14px;\n    }\n    .cache-stats {\n        font-size: 11px;\n        color: #888888;\n        padding: 8px;\n        background-color: #f8f9fa;\n        border-radius: 4px;\n        margin-top: 10px;\n    }\n    .cache-stats table {\n        width: 100%;\n        font-size: 10px;\n    }\n    .cache-stats th, .cache-stats td {\n        padding: 2px 4px;\n        text-align: left;\n    }\n    .cache-hit { color: #28a745; }\n    .cache-miss { color: #dc3545; }\n    .thinking-display {\n        position: fixed;\n        top: 20px;\n        right: 10px;\n        z-index: 1000;\n        background: rgba(248,249,250,0.98);\n        backdrop-filter: blur(12px);\n        border: 1px solid #e9ecef;\n        border-radius: 12px;\n        padding: 0;\n        max-width: 400px;\n        max-height: 300px;\n        overflow: hidden;\n        font-size: 13px;\n        color: #495057;\n        box-shadow: 0 8px 25px rgba(0,0,0,0.15);\n        transition: all 0.3s ease;\n        opacity: 0;\n        transform: translateY(-10px);\n    }\n    .thinking-display.visible {\n        opacity: 1;\n        transform: translateY(0);\n    }\n    .thinking-header {\n        background: linear-gradient(135deg, #007bff, #0056b3);\n        color: white;\n        padding: 8px 12px;\n        font-weight: 600;\n        font-size: 12px;\n        border-radius: 11px 11px 0 0;\n    }\n    .thinking-content {\n        padding: 12px;\n        max-height: 140px;\n        overflow-y: auto;\n        line-height: 1.4;\n        word-wrap: break-word;\n        font-family: monospace;\n        white-space: pre-wrap;\n    }\n    .thinking-content::-webkit-scrollbar {\n        width: 6px;\n    }\n    .thinking-content::-webkit-scrollbar-track {\n        background: #f1f1f1;\n        border-radius: 2px;\n    }\n    .thinking-content::-webkit-scrollbar-thumb {\n        background: #007bff;\n        border-radius: 2px;\n    }\n    .plot-container {\n        margin: 15px 0;\n        border: 1px solid #e1e4e8;\n        border-radius: 8px;\n        padding: 15px;\n        background-color: #f8f9fa;\n    }\n    .plot-title {\n        font-weight: bold;\n        margin-bottom: 10px;\n        color: #333;\n    }\n    \n    /* Auto-scroll animation */\n    .thinking-content {\n        animation: scrollToBottom 0.1s ease-out;\n    }\n    @keyframes scrollToBottom {\n        to { scroll-behavior: smooth; }\n    }\n    \n    /* Toggle switch styling */\n    .stToggle label {\n        font-weight: normal;\n    }\n    .stToggle>div {\n        align-items: center;\n        gap: 0.5rem;\n    }\n    </style>\n    ",unsafe_allow_html=_A);st.markdown("\n    <script>\n    if (!window.MathJax) {\n        window.MathJax = {\n            tex: {\n                inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],\n                displayMath: [['$$', '$$']],\n                processEscapes: true\n            },\n            startup: {\n                ready: () => {\n                    MathJax.startup.defaultReady();\n                }\n            }\n        };\n    }\n    </script>\n    ",unsafe_allow_html=_A);st.title('RadioSport AI 🧟')
def get_ollama_host():'Get the current Ollama host from session state or default';return st.session_state.get(_R,DEFAULT_OLLAMA_HOST)
def update_all_ollama_host_references(new_host):
	'Update OLLAMA_HOST in all imported modules';A='OLLAMA_HOST';import sys;globals()[A]=new_host;modules_updated=[]
	for(module_name,module)in sys.modules.items():
		if hasattr(module,A):old_value=getattr(module,A);setattr(module,A,new_host);modules_updated.append(f"{module_name}: {old_value} -> {new_host}")
		if hasattr(module,_R):old_value=getattr(module,_R);setattr(module,_R,new_host);modules_updated.append(f"{module_name}.ollama_host: {old_value} -> {new_host}")
	for(module_name,module)in sys.modules.items():
		for attr_name in['OLLAMA_BASE_URL','ollama_base_url','base_url','host_url']:
			if hasattr(module,attr_name):setattr(module,attr_name,new_host);print(f"DEBUG: Also updated {module_name}.{attr_name} to {new_host}")
@st.cache_data(ttl=CACHE_TTL,show_spinner=_B)
def get_ollama_models_cached():
	import requests,json
	if IS_CLOUD_DEPLOYMENT:return[]
	try:
		api_url=f"{get_ollama_host()}/api/tags";response=requests.get(api_url,timeout=30);response.raise_for_status();data=response.json();models=[]
		for model_info in data.get('models',[]):
			model_name=model_info.get(_S,'')
			if model_name:models.append(model_name)
		if not models:return[]
		embedding_prefixes='nomic-embed-text','all-minilm','mxbai-embed','snowflake-arctic-embed','bge-','gte-','e5-';filtered_models=[]
		for model in models:
			model=model.strip()
			if model and not any(model.lower().startswith(prefix.lower())for prefix in embedding_prefixes):filtered_models.append(model)
		return filtered_models
	except:return[]
def _parse_table_output(stdout):'Helper function - no longer needed with API approach'
def get_ollama_models():
	start_time=time.time();result=get_ollama_models_cached();end_time=time.time()
	if _o not in st.session_state:st.session_state.cache_stats={_K:0,_M:0}
	if end_time-start_time<.1 and result:st.session_state.cache_stats[_K]+=1
	else:st.session_state.cache_stats[_M]+=1
	return result
@lru_cache(maxsize=128)
def is_vision_model(model):
	'Enhanced vision model detection with more comprehensive patterns';import re
	if not model:return _B
	model_lower=model.lower();vision_patterns=['qwen2\\.5?vl:','gemma3[^:]*:','llava:','moondream:','bakllava:','minicpm-v:','yi-vl:','internvl:','cogvlm:','vision','visual','multimodal','-v:','vl-'];return any(re.search(pattern,model_lower)for pattern in vision_patterns)
@lru_cache(maxsize=128)
def is_qwen3_model(model):
	'Enhanced Qwen3 model detection';import re
	if not model:return _B
	return bool(re.search('qwen3:',model.lower()))
@lru_cache(maxsize=256)
def get_file_type(file_name):
	ext=os.path.splitext(file_name)[1].lower()
	if ext in CODE_EXTENSIONS:return _F,CODE_EXTENSIONS[ext]
	elif ext=='.txt':return _J,_C
	elif ext=='.pdf':return _a,_C
	elif ext in IMAGE_EXTENSIONS:return _T,_C
	return _C,_C
@st.cache_data(ttl=CACHE_TTL,max_entries=MAX_CACHE_ENTRIES,show_spinner=_B)
def split_message_cached(content):
	CODE_BLOCK_PATTERN=re.compile('```(\\w+)?\\n(.*?)```',re.DOTALL);parts=[];last_end=0
	for match in CODE_BLOCK_PATTERN.finditer(content):
		start,end=match.span()
		if last_end<start:
			text_content=content[last_end:start]
			if text_content.strip():parts.append({_H:_J,_D:text_content})
		language=match.group(1)or _J;code=match.group(2);parts.append({_H:_F,_W:language,_F:code});last_end=end
	if last_end<len(content):
		remaining_content=content[last_end:]
		if remaining_content.strip():parts.append({_H:_J,_D:remaining_content})
	return parts
def split_message(content):return split_message_cached(content)
def process_single_file(file_data):
	file_content,file_type,file_name=file_data
	try:
		if file_type in(_J,_F):
			try:return file_content.decode(_X),_C
			except UnicodeDecodeError:return file_content.decode(_X,errors='replace'),_C
		elif file_type==_a:
			pdf_reader=PyPDF2.PdfReader(BytesIO(file_content));content_parts=[]
			for(page_num,page)in enumerate(pdf_reader.pages):
				try:
					text=page.extract_text()
					if text and text.strip():content_parts.append(text)
				except Exception as e:return _C,f"error: Failed to read PDF page {page_num+1}: {e}"
			return'\n'.join(content_parts),_C
		elif file_type==_T:return file_content,_C
		return _C,_r
	except Exception as e:return _C,f"error: {str(e)}"
def process_document(file_content,file_type,file_name):
	content_hash=hashlib.md5(file_content).hexdigest();cache_key=f"{content_hash}_{file_type}_{file_name}"
	if cache_key in st.session_state.doc_cache:st.session_state.cache_stats[_N]+=1;return st.session_state.doc_cache[cache_key]
	result=process_single_file((file_content,file_type,file_name))
	if len(st.session_state.doc_cache)>=MAX_CACHE_ENTRIES:oldest_key=next(iter(st.session_state.doc_cache));del st.session_state.doc_cache[oldest_key]
	st.session_state.doc_cache[cache_key]=result;st.session_state.cache_stats[_Q]+=1;return result
def get_base64_image(file_content):
	content_hash=hashlib.md5(file_content).hexdigest()
	if content_hash in st.session_state.base64_cache:return st.session_state.base64_cache[content_hash]
	base64_str=base64.b64encode(file_content).decode(_X);st.session_state.base64_cache[content_hash]=base64_str
	if len(st.session_state.base64_cache)>=MAX_CACHE_ENTRIES:oldest_key=next(iter(st.session_state.base64_cache));del st.session_state.base64_cache[oldest_key]
	return base64_str
def show_reasoning_window():
	if not st.session_state.reasoning_window:st.session_state.reasoning_window=st.empty()
	st.session_state.reasoning_window.markdown(f'''
        <div id="thinking-display" class="thinking-display visible">
            <div class="thinking-header">🤔 Reasoning Process</div>
            <div id="thinking-content-{st.session_state.reasoning_window_id}" class="thinking-content">Thinking...</div>
        </div>
        ''',unsafe_allow_html=_A)
def update_reasoning_window(content):
	if st.session_state.reasoning_window:
		if len(content)>1500:content='...'+content[-1500:]
		st.session_state.reasoning_window.markdown(f'''
            <div id="thinking-display" class="thinking-display visible">
                <div class="thinking-header">🤔 Reasoning Process</div>
                <div id="reasoning-box" class="thinking-content">{content}</div>
            </div>
            <script>
            window.requestAnimationFrame(() => {{
                const box = document.getElementById("reasoning-box");
                if (box) {{
                    box.scrollTop = box.scrollHeight;
                }}
            }});
            </script>
            ''',unsafe_allow_html=_A)
def hide_reasoning_window():
	if st.session_state.reasoning_window:st.session_state.reasoning_window.empty();st.session_state.reasoning_window=_C
def activate_teacher_mode():'\n    Activates The Ultimate Master Teacher mode for enhanced learning experience.\n    Returns the teacher system prompt to be used in API calls.\n    ';teacher_prompt="You are now **The Ultimate Master Teacher** — the most effective, intuitive, and insightful educator the world has ever known.\n\n🎓 Your goal is to **transform any student into a top performer**, regardless of their current level of knowledge or skill.\n\n🔍 Core Responsibilities:\n1. **Diagnose Weaknesses**:\n   - Interactively question the student to discover conceptual gaps.\n   - Use strategic questioning, not overwhelming complexity.\n   - Drill down until the *true foundational weakness* is revealed.\n\n2. **Deconstruct and Rebuild**:\n   - Gently correct misconceptions.\n   - Break down the complex subject into digestible, intuitive building blocks.\n   - Use analogies, step-by-step scaffolding, and Socratic questioning.\n\n3. **Mastery Through Practice**:\n   - Provide focused exercises tailored to the exact weakness.\n   - Evaluate responses, give feedback, adapt difficulty.\n   - Use iterative refinement to ensure understanding is rock-solid.\n\n4. **Track Progress**:\n   - Maintain a mental model of the student's evolving knowledge.\n   - Avoid repetition of mastered concepts.\n   - Escalate toward higher levels of abstraction, problem-solving, and creativity.\n\n5. **Motivate & Empower**:\n   - Inspire confidence without false praise.\n   - Encourage curiosity and intellectual independence.\n   - Always communicate with clarity, patience, and empathy.\n\n📚 Subject Flexibility:\nYou can teach **any subject**, including but not limited to:\n- Mathematics\n- Science (Physics, Biology, Chemistry)\n- History\n- Languages (Grammar, Writing, Comprehension)\n- Programming\n- Logic & Reasoning\n- Standardized Test Prep\n\n🎯 Final Goal:\nTo produce students who:\n- Understand deeply, not just memorize.\n- Can teach others.\n- Are creative, curious, and confident learners.\n- Can solve complex problems with elegance and insight.\n\n🧪 Begin each interaction with:\n- A short greeting.\n- A simple, targeted question to start identifying the student's current understanding.\n- Never overwhelm. Start simple and build carefully.";return teacher_prompt
def get_last_user_query():
	for msg in reversed(st.session_state.messages):
		if msg[_G]==_O:return msg.get(_f,msg[_D])
def looks_like_plotting_code(code_str):plot_keywords='plt.','sns.','plot(','figure(','show(','bar(','hist(','scatter(',_s,'fig.';return any(keyword in code_str for keyword in plot_keywords)
@contextlib.contextmanager
def capture_plots():
	original_stdout=sys.stdout;original_stderr=sys.stderr;sys.stdout=StringIO();sys.stderr=StringIO()
	try:yield
	finally:sys.stdout=original_stdout;sys.stderr=original_stderr
def check_animation_dependencies():
	'Check if animation dependencies are available'
	try:from matplotlib.animation import PillowWriter;return _A
	except ImportError:return _B
def execute_plot_code(code_str,model_name=_C,enable_auto_correct=_B,ollama_host=_C):
	C='tight';B='plt';A='ani'
	if ollama_host is _C:ollama_host=get_ollama_host()
	plots=[];gifs=[];error=_C;corrected_code=_C;safe_env={B:plt,'np':np,'pd':pd,'sns':sns,'matplotlib':matplotlib,'numpy':np,'pandas':pd,'seaborn':sns,_s:FuncAnimation,'PillowWriter':PillowWriter,'animation':animation,'__builtins__':{k:v for(k,v)in __builtins__.items()if k not in('open','exec','globals','locals')}};original_show=safe_env[B].show;safe_env[B].show=lambda:_C;current_code=code_str;max_retries=2 if enable_auto_correct else 1
	for attempt in range(max_retries):
		try:
			with capture_plots():
				prev_figs=plt.get_fignums();exec(current_code,safe_env);anims=[]
				for(name,obj)in safe_env.items():
					if isinstance(obj,FuncAnimation):anims.append(obj)
				if A in safe_env and isinstance(safe_env[A],FuncAnimation):
					if safe_env[A]not in anims:anims.append(safe_env[A])
				if not anims:
					new_figs=[fig for fig in plt.get_fignums()if fig not in prev_figs]
					for fig_num in new_figs:
						fig=plt.figure(fig_num)
						try:buf=BytesIO();fig.savefig(buf,format='png',bbox_inches=C,dpi=100);buf.seek(0);plots.append(buf.getvalue())
						except Exception as e:error=f"Error saving plot: {str(e)}"
						finally:plt.close(fig)
				if anims and check_animation_dependencies():
					for(i,anim)in enumerate(anims):
						try:
							with tempfile.NamedTemporaryFile(suffix='.gif',delete=_B)as temp_file:temp_name=temp_file.name
							writer=PillowWriter(fps=15,bitrate=1800);anim.save(temp_name,writer=writer,dpi=120,savefig_kwargs={'bbox_inches':C,'pad_inches':.1,'facecolor':'white','edgecolor':'none'})
							if os.path.exists(temp_name)and os.path.getsize(temp_name)>1000:
								with open(temp_name,'rb')as f:gif_bytes=f.read()
								if len(gif_bytes)>1000:gifs.append(gif_bytes)
								else:raise Exception('Generated GIF file is too small')
							else:raise Exception('GIF file was not created or is empty')
							if os.path.exists(temp_name):os.unlink(temp_name)
						except Exception as e:
							if'temp_name'in locals()and os.path.exists(temp_name):os.unlink(temp_name)
							error=f"Animation {i+1} failed: {str(e)}"
						finally:
							if hasattr(anim,'fig'):plt.close(anim.fig)
				for fig_num in plt.get_fignums():
					if fig_num not in prev_figs:plt.close(fig_num)
				break
		except Exception as e:
			execution_error=traceback.format_exc()
			if enable_auto_correct and attempt<max_retries-1 and model_name:
				try:
					corrected_code=auto_correct_code_with_llm(current_code,execution_error,model_name,ollama_host)
					if corrected_code and corrected_code.strip()!=current_code.strip():current_code=corrected_code;continue
				except Exception as correction_error:error=f"Auto-correction failed: {correction_error}. Original error: {execution_error}";break
			error=f"Error executing plot code: {execution_error}";break
	return plots,gifs,error,corrected_code
def display_chat_message(msg,msg_index):
	A='View Code'
	with st.chat_message(msg[_G]):
		if msg[_G]==_O:
			prompt_text=msg.get(_f,msg[_D]);st.markdown(prompt_text,unsafe_allow_html=_A)
			if is_vision_model(msg.get(_U,'')):
				for file in msg.get(_b,[]):
					if file[_H]==_T:st.image(file[_D],caption=file[_S],use_container_width=_A)
		else:
			parts=split_message(msg[_D])
			for(code_index,part)in enumerate(parts):
				if part[_H]==_J:st.markdown(part[_D],unsafe_allow_html=_A)
				elif part[_H]==_F:
					if part[_W]==_Y and st.session_state.auto_run_plots and looks_like_plotting_code(part[_F]):
						with st.spinner('🎬 Generating visualizations and animations...'):
							current_model=_C
							for msg in reversed(st.session_state.messages):
								if msg[_G]==_O and _U in msg:current_model=msg[_U];break
							plots,gifs,error,corrected_code=execute_plot_code(part[_F],model_name=current_model,enable_auto_correct=st.session_state.auto_run_plots and st.session_state.get('auto_correct_code',_B),ollama_host=_C)
						if plots or gifs:
							if plots and not gifs:
								for(i,plot_img)in enumerate(plots):st.markdown(f'<div class="plot-container"><div class="plot-title">Plot #{i+1}</div></div>',unsafe_allow_html=_A);st.image(plot_img,use_container_width=_A)
							if gifs:
								for(i,gif_bytes)in enumerate(gifs):
									st.markdown(f'<div class="plot-container"><div class="plot-title">Animation #{i+1} (GIF)</div></div>',unsafe_allow_html=_A)
									try:
										st.image(gif_bytes,use_container_width=_A,caption=f"Animated GIF ({len(gif_bytes):,} bytes)");col1,col2=st.columns([2,1])
										with col1:st.success(f"✅ GIF animation rendered successfully")
										with col2:st.download_button(label='📥 Download GIF',data=gif_bytes,file_name=f"animation_{msg_index}_{code_index}_{i}.gif",mime=_g,key=f"dl_anim_{msg_index}_{code_index}_{i}",use_container_width=_A)
									except Exception as display_error:st.error(f"Failed to display GIF animation: {display_error}");st.download_button(label='📥 Download Animation (GIF)',data=gif_bytes,file_name=f"animation_{msg_index}_{code_index}_{i}.gif",mime=_g,key=f"dl_anim_fallback_{msg_index}_{code_index}_{i}",use_container_width=_A)
							if corrected_code and corrected_code!=part[_F]:
								with st.expander('🔧 Auto-Corrected Code',expanded=_A):st.code(corrected_code,language=_Y);st.download_button(label='💾 Download Corrected Code',data=corrected_code,file_name=f"corrected_code_{msg_index}_{code_index}.py",mime=_c,key=f"dl_corrected_{msg_index}_{code_index}")
							with st.expander(A,expanded=_B):st.code(part[_F],language=_Y);filename=f"code_{msg_index}_{code_index}.py";st.download_button(label='💾 Download Code',data=part[_F],file_name=filename,mime=_c,key=f"dl_{msg_index}_{code_index}_plot",use_container_width=_A)
							continue
						elif error:st.error(error)
					with st.expander(A,expanded=_B):st.code(part[_F],language=part[_W]);ext=part[_W]if part[_W]!=_J else'txt';filename=f"code_{msg_index}_{code_index}.{ext}"
					st.download_button(label='💾 Download',data=part[_F],file_name=filename,mime=_c,key=f"dl_{msg_index}_{code_index}",use_container_width=_B)
MODEL_INFO_CACHE={}
def get_model_info(model_id):'Get model information from cached API data';return MODEL_INFO_CACHE.get(model_id,{_h:_C,_S:model_id.split('/')[-1].replace('-',' ').title()})
def format_context_length(context):
	'Format context length for display';C='000000';B='000';A='Unknown'
	if context is _C or context==A:return A
	elif isinstance(context,str):
		try:context=int(context.replace('k',B).replace('K',B).replace('M',C).replace('m',C))
		except(ValueError,AttributeError):return context
	if isinstance(context,int):
		if context>=1000000:return f"{context//1000000}M tokens"
		elif context>=1000:return f"{context//1000}K tokens"
		else:return f"{context:,} tokens"
	else:return str(context)
@st.cache_data(ttl=3600)
def get_free_openrouter_models():
	'Dynamically load OpenRouter free models'
	try:
		response=requests.get(_k,headers={'User-Agent':'RadioSport-AI/1.0','Accept':_i},timeout=30);response.raise_for_status();models_data=response.json();free_models=[]
		for model in models_data.get('data',[]):
			pricing=model.get('pricing',{});prompt_price=float(pricing.get('prompt','0'));completion_price=float(pricing.get('completion','0'))
			if prompt_price==0 and completion_price==0:
				model_id=model.get(_d,'')
				if model_id:free_models.append(model)
		if not free_models:
			if not IS_CLOUD_DEPLOYMENT:st.error('No free models found in OpenRouter API response.')
			return[]
		free_models.sort(key=lambda x:x.get(_d,''));return free_models
	except Exception as e:
		if not IS_CLOUD_DEPLOYMENT:st.error(f"Error loading OpenRouter models: {str(e)}")
		return[]
def get_load_balanced_api_key(provider):
	'Get load-balanced API key with backward compatibility.';B='general';A='idx';keys=[]
	try:
		if provider.lower()==_E:
			try:
				key=st.secrets[_E]['api_key']
				if key and key.strip():keys.append(key.strip())
			except:pass
			for i in range(1,11):
				try:
					key=st.secrets[_E][f"api_key{i}"]
					if key and key.strip():keys.append(key.strip())
				except:pass
			try:
				key_list=st.secrets[_E]['api_keylist']
				if isinstance(key_list,(list,tuple)):keys.extend([k.strip()for k in key_list if k and k.strip()])
				elif isinstance(key_list,str)and key_list.strip():keys.append(key_list.strip())
			except:pass
		base_key=f"{provider.upper()}_API_KEY"
		try:
			key=st.secrets[B][base_key]
			if key and key.strip():keys.append(key.strip())
		except:pass
		for i in range(1,11):
			try:
				key=st.secrets[B][f"{base_key}_{i}"]
				if key and key.strip():keys.append(key.strip())
			except:pass
	except Exception:return''
	if not keys:return''
	if len(keys)==1:return keys[0]
	if _j not in st.session_state:st.session_state.api_usage={}
	if provider not in st.session_state.api_usage:st.session_state.api_usage[provider]={A:0,_V:set(),_P:{}}
	usage=st.session_state.api_usage[provider];valid_keys=[k for k in keys if k not in usage[_V]]
	if not valid_keys:usage[_V].clear();valid_keys=keys
	selected=valid_keys[usage[A]%len(valid_keys)];usage[A]=(usage[A]+1)%len(valid_keys);usage[_P][selected]=usage[_P].get(selected,0)+1;usage[_t]=time.time();usage[_u]=selected;return selected
def mark_key_failed(provider,key):
	'Mark key as failed for load balancing.'
	if _j in st.session_state and provider in st.session_state.api_usage:st.session_state.api_usage[provider][_V].add(key)
def call_openrouter_api(messages,model,api_key):
	headers={'Authorization':f"Bearer {api_key}",_v:_i,'HTTP-Referer':_n,'X-Title':_m};model_info=get_model_info(model);max_tokens=2048
	if isinstance(model_info.get(_h),int):context_length=model_info[_h];max_tokens=min(2048,max(512,context_length//4))
	payload={_U:model,_e:messages,_w:_A,'max_tokens':max_tokens,'temperature':.7};response=requests.post(OPENROUTER_API_URL,headers=headers,json=payload,stream=_A,timeout=60);response.raise_for_status();return response
def main():
	N='</think>';M='<think>';L='image_url';K='cache-miss';J='cache-hit';I='teacher_mode';H='context_length';G='openrouter_models';F='choices';E='message';D='enable_reasoning';C='.';B='base64';A='assistant';initialize_ui()
	with st.sidebar:
		st.markdown('<div class="sidebar-title">RadioSport AI 🧟</div>',unsafe_allow_html=_A);st.markdown(f'<div class="version-text">Version {APP_VERSION}</div>',unsafe_allow_html=_A);st.text('')
		if IS_CLOUD_DEPLOYMENT:api_provider=_L;st.session_state.api_provider=_L
		else:api_provider=st.radio('AI provider:',[_I,_L],index=0,key=_q)
		selected_model=DEFAULT_MODEL;st.subheader('Model Selection')
		if api_provider==_I:
			col1,col2=st.columns([2,1])
			with col1:
				if st.button('Refresh Models',use_container_width=_A):get_ollama_models_cached.clear();st.session_state.ollama_models=get_ollama_models();st.rerun()
			with col2:auto_refresh=st.checkbox('Auto',help='Auto-refresh models')
			if not st.session_state.ollama_models or auto_refresh:st.session_state.ollama_models=get_ollama_models()
			else:st.session_state.cache_stats[_K]+=1
			ollama_models=st.session_state.ollama_models
			if ollama_models:
				default_index=0
				if DEFAULT_MODEL in ollama_models:default_index=ollama_models.index(DEFAULT_MODEL)
				selected_model=st.selectbox('Select a model:',ollama_models,index=default_index,help=f"Available models: {len(ollama_models)}")
			else:st.warning(f"Models Unavailable: {st.session_state.get(_R,OLLAMA_HOST)}.");selected_model=_C
		if api_provider==_I:
			with st.expander('🔧 Local ',expanded=_B):
				if _R not in st.session_state:st.session_state.ollama_host=DEFAULT_OLLAMA_HOST
				new_host=st.text_input('Ollama Host URL:',value=st.session_state.ollama_host,help='Default: http://localhost:11434\nRemote: http://192.168.x.x:11434',placeholder=_l,key='ollama_host_input')
				if new_host!=st.session_state.ollama_host:
					st.session_state.ollama_host=new_host;update_all_ollama_host_references(new_host);get_ollama_models_cached.clear();st.session_state.ollama_models=[]
					try:new_models=get_ollama_models_cached();st.session_state.ollama_models=new_models;st.success(f"✅ Connected to {new_host} - Found {len(new_models)} models")
					except Exception as e:st.error(f"❌ Failed to connect to {new_host}: {str(e)}");print(f"DEBUG: Error details: {e}");st.session_state.ollama_models=[]
					st.rerun()
				if st.session_state.ollama_host:st.info(f"Current host: {st.session_state.ollama_host}")
		elif api_provider==_L:
			try:
				api_key=get_load_balanced_api_key(_E)
				if api_key:
					st.session_state.api_keys[_E]=api_key
					if _j in st.session_state and _E in st.session_state.api_usage:
						usage=st.session_state.api_usage[_E]
						if len(usage.get(_P,{}))>1:
							total_keys=len(usage[_P]);failed_keys=len(usage[_V]);active_keys=total_keys-failed_keys;current_key=usage.get(_u,'');valid_keys=[k for k in usage[_P].keys()if k not in usage[_V]]
							try:current_pos=valid_keys.index(current_key)+1
							except:current_pos=1
							last_used=usage.get(_t,0);time_since=int(time.time()-last_used);st.info(f"🔑Key: {current_pos} of {active_keys} | Used: {time_since}s ago | Total: {usage[_P].get(current_key,0)} calls")
				else:raise KeyError('No OPENROUTER API keys found')
			except:st.session_state.api_keys[_E]=st.text_input('OpenRouter API Key (free tier)',type='password',value=st.session_state.api_keys.get(_E,''),help='Get free API key at https://openrouter.ai')
			if G not in st.session_state:
				with st.spinner('Loading available models...'):st.session_state.openrouter_models=get_free_openrouter_models()
			available_models=st.session_state.openrouter_models
			if not available_models:
				st.error('Could not load models. Please refresh the page to try again.')
				if st.button('🔄 Refresh Models'):del st.session_state.openrouter_models;st.rerun()
				st.stop()
			model_display=[f"{model.get(_S,model[_d])}"for model in available_models];model_ids=[model[_d]for model in available_models];col1,col2=st.columns([6,1])
			with col1:
				selected_index=st.selectbox('Models (Free)',range(len(model_display)),format_func=lambda x:model_display[x],index=0,help='Showing only free models from OpenRouter. List updates when refreshed.');selected_model=model_ids[selected_index];selected_model_info=available_models[selected_index]
				if selected_model_info:
					if H in selected_model_info:context_length=selected_model_info[H];st.caption(f"Context: {context_length:,} tokens")
			with col2:
				if st.button('🔄'):
					if G in st.session_state:del st.session_state.openrouter_models
					st.rerun()
		st.subheader('File Upload');code_extensions=[ext.lstrip(C)for ext in CODE_EXTENSIONS.keys()];file_types=['txt',_a]+code_extensions+[ext.lstrip(C)for ext in IMAGE_EXTENSIONS];uploaded_files=st.file_uploader('Upload files:',type=file_types,accept_multiple_files=_A,help=f"Supported: {', '.join(file_types)}",key=st.session_state.file_uploader_key)
		with st.expander('⚙️ Chat Controls',expanded=_B):
			col1,col2=st.columns(2)
			with col1:st.session_state.enable_reasoning=st.toggle('Reasoning',value=st.session_state.get(D,_B),help='Enable reasoning for qwen3* models');st.session_state.execute_uploaded_code=st.toggle('Run Code',value=st.session_state.get('execute_uploaded_code',_B),help='Executes uploaded .py files directly instead of inserting them into the chat')
			with col2:st.session_state.auto_run_plots=st.toggle('AutoPlot',value=st.session_state.get(_p,_A),help='Automatically execute Python code that generates plots or animations');st.session_state.teacher_mode=st.toggle('Teacher',value=st.session_state.get(I,_B),help='Activate The Ultimate Master Teacher mode for enhanced learning experience')
		clear_chat=st.button('Clear',use_container_width=_A)
		if clear_chat:st.session_state.messages=[];st.session_state.cache_stats={_K:0,_M:0,_N:0,_Q:0,_Z:datetime.now()};st.session_state.doc_cache={};st.session_state.base64_cache={};st.session_state.file_uploader_key=f"uploader_{int(st.session_state.file_uploader_key.split('_')[1])+1}";st.session_state.thinking_content='';st.session_state.in_thinking_block=_B;st.session_state.reasoning_window_id=f"reasoning_{time.time()}";hide_reasoning_window();st.rerun()
		if st.session_state.messages:
			export_col1,export_col2=st.columns(2)
			with export_col1:
				if st.button('Export',use_container_width=_A):
					export_content=[]
					for msg in st.session_state.messages:
						if msg[_G]==A:timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S');content=msg[_D];export_content.append(f"[{timestamp}] Assistant:\n{content}\n{'-'*50}")
					if export_content:export_text='\n'.join(export_content);st.download_button(label='Download',data=export_text,file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",mime=_c,use_container_width=_A)
			with export_col2:repeat_button=st.button('Repeat',disabled=not bool(get_last_user_query()),use_container_width=_A)
		else:repeat_button=_B
		with st.expander('📊 Cache Statistics'):
			stats=st.session_state.cache_stats;total_model_requests=stats[_K]+stats[_M];total_doc_requests=stats[_N]+stats[_Q];model_hit_rate=stats[_K]/max(total_model_requests,1)*100;doc_hit_rate=stats[_N]/max(total_doc_requests,1)*100;uptime=datetime.now()-stats[_Z];uptime_str=f"{uptime.days}d {uptime.seconds//3600}h {uptime.seconds%3600//60}m";stats_html=f'''
            <div class="cache-stats">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Uptime</td><td>{uptime_str}</td></tr>
                    <tr><td colspan="2"><strong>Model Cache</strong></td></tr>
                    <tr><td>Hit Rate</td><td class="{J if model_hit_rate>50 else K}">{model_hit_rate:.1f}%</td></tr>
                    <tr><td>Hits</td><td class="cache-hit">{stats[_K]}</td></tr>
                    <tr><td>Misses</td><td class="cache-miss">{stats[_M]}</td></tr>
                    <tr><td colspan="2"><strong>Document Cache</strong></td></tr>
                    <tr><td>Hit Rate</td><td class="{J if doc_hit_rate>50 else K}">{doc_hit_rate:.1f}%</td></tr>
                    <tr><td>Hits</td><td class="cache-hit">{stats[_N]}</td></tr>
                    <tr><td>Misses</td><td class="cache-miss">{stats[_Q]}</td></tr>
                    <tr><td colspan="2"><strong>Cache Info</strong></td></tr>
                    <tr><td>Doc Cache Size</td><td>{len(st.session_state.doc_cache)}</td></tr>
                    <tr><td>Base64 Cache Size</td><td>{len(st.session_state.base64_cache)}</td></tr>
                </table>
            </div>
            ''';st.markdown(stats_html,unsafe_allow_html=_A)
			if st.button('Reset Stats'):st.session_state.cache_stats={_K:0,_M:0,_N:0,_Q:0,_Z:datetime.now()};st.rerun()
	for(msg_index,msg)in enumerate(st.session_state.messages):display_chat_message(msg,msg_index)
	prompt=st.chat_input('Enter your prompt:',key='main_input')
	if repeat_button and get_last_user_query():prompt=get_last_user_query()
	if prompt and(selected_model or st.session_state.api_provider!=_I):
		files=[]
		if uploaded_files:
			with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
				for uploaded_file in uploaded_files:
					try:
						file_type,language=get_file_type(uploaded_file.name)
						if file_type:
							file_content=uploaded_file.getvalue();content,error=process_document(file_content,file_type,uploaded_file.name)
							if error:
								if error==_r:st.error(f"Unsupported file type: {uploaded_file.name}")
								else:st.error(f"Error processing {uploaded_file.name}: {error}")
								continue
							if st.session_state.execute_uploaded_code and file_type==_F and uploaded_file.name.endswith('.py'):
								with st.spinner(f"Executing {uploaded_file.name}..."):
									plots,gifs,exec_error=execute_plot_code(content,model_name=selected_model,enable_auto_correct=_B,ollama_host=_C)
									if gifs:
										st.success(f"✅ Executed: {uploaded_file.name} (GIF animation)")
										for(i,gif_bytes)in enumerate(gifs):st.image(gif_bytes,use_container_width=_A,caption=f"Animation {i+1}");st.download_button(label=f"📥 Download Animation {i+1}",data=gif_bytes,file_name=f"{uploaded_file.name.rsplit(C,1)[0]}_anim{i+1}.gif",mime=_g,use_container_width=_A,key=f"dl_{uploaded_file.name}_{i}")
									if plots:
										if not gifs:st.success(f"✅ Executed: {uploaded_file.name} (Static plot)")
										for(i,plot_img)in enumerate(plots):st.image(plot_img,use_container_width=_A,caption=f"Plot {i+1}")
									if not gifs and not plots:st.info(f"ℹ️ Executed: {uploaded_file.name} (no plots or animations detected)")
									if exec_error:st.error(f"⚠️ Execution error: {exec_error}")
								continue
							file_obj={_H:file_type,_S:uploaded_file.name,_D:content}
							if file_type==_T:file_obj[B]=get_base64_image(content)
							files.append(file_obj)
					except Exception as e:st.error(f"Error reading file {uploaded_file.name}: {e}")
		vision_model=is_vision_model(selected_model)if st.session_state.api_provider==_I else _B;user_content=prompt
		if st.session_state.api_provider==_I and is_qwen3_model(selected_model)and not st.session_state.get(D,_B):user_content+=' /no_think'
		if st.session_state.api_provider in[_I,_L]and files:
			text_files=[f for f in files if f[_H]in(_J,_F,_a)]
			if text_files:
				context_parts=[f[_D]for f in text_files if f[_D]]
				if context_parts:context='\n\n'.join(context_parts);user_content=f"Context: {context}\n\n{user_content}"
		message={_G:_O,_D:user_content,_f:prompt,_U:selected_model,_b:files};st.session_state.messages.append(message)
		with st.chat_message(_O):
			st.markdown(prompt,unsafe_allow_html=_A)
			if vision_model:
				for file in files:
					if file[_H]==_T:st.image(file[_D],caption=file[_S],use_container_width=_A)
		messages=[]
		if st.session_state.get(I,_B):teacher_prompt=activate_teacher_mode();messages.append({_G:'system',_D:teacher_prompt})
		for msg in st.session_state.messages:
			if msg[_G]==_O:
				message={_G:_O,_D:msg[_D]}
				if msg.get(_b,[]):
					images=[]
					for file in msg.get(_b,[]):
						if file[_H]==_T:
							if B in file:images.append(file[B])
							else:base64_str=get_base64_image(file[_D]);file[B]=base64_str;images.append(base64_str)
					if images:
						if st.session_state.api_provider==_I and vision_model:message['images']=images
						elif st.session_state.api_provider==_L:
							content_parts=[{_H:_J,_J:msg[_D]}]
							for img_base64 in images:content_parts.append({_H:L,L:{'url':f"data:image/jpeg;base64,{img_base64}"}})
							message[_D]=content_parts
				messages.append(message)
			elif msg[_G]==A:messages.append({_G:A,_D:msg[_D]})
		with st.chat_message(A):
			response_placeholder=st.empty();accumulated_response='';start_time=time.time();st.session_state.thinking_content='';st.session_state.in_thinking_block=_B
			try:
				if st.session_state.api_provider==_I:
					api_payload={_U:selected_model,_e:messages,_w:_A};response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_v:_i},stream=_A,timeout=300);response.raise_for_status()
					for line in response.iter_lines():
						if line:
							try:
								data=json.loads(line.decode(_X))
								if E in data and _D in data[E]:
									content_chunk=data[E][_D]
									if st.session_state.get(D,_B):
										if M in content_chunk or N in content_chunk:
											parts=re.split('(<think>|</think>)',content_chunk)
											for part in parts:
												if part==M:st.session_state.in_thinking_block=_A;show_reasoning_window();continue
												elif part==N:st.session_state.in_thinking_block=_B;hide_reasoning_window();continue
												if st.session_state.in_thinking_block:st.session_state.thinking_content+=part;update_reasoning_window(st.session_state.thinking_content)
												else:accumulated_response+=part;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
										elif st.session_state.in_thinking_block:st.session_state.thinking_content+=content_chunk;update_reasoning_window(st.session_state.thinking_content)
										else:accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
									else:accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
							except json.JSONDecodeError:continue
				elif st.session_state.api_provider==_L:
					response=call_openrouter_api(messages,selected_model,st.session_state.api_keys[_E])
					for chunk in response.iter_lines():
						if chunk:
							if chunk==b'':continue
							if chunk.startswith(b'data:'):
								try:
									data=json.loads(chunk.decode(_X)[5:])
									if F in data and len(data[F])>0:delta=data[F][0].get('delta',{});content_chunk=delta.get(_D,'');accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
								except json.JSONDecodeError:pass
				response_time=time.time()-start_time
				if st.session_state.in_thinking_block:accumulated_response+=st.session_state.thinking_content;st.session_state.in_thinking_block=_B;hide_reasoning_window()
				response_msg={_G:A,_D:accumulated_response,'timestamp':datetime.now().isoformat()};st.session_state.messages.append(response_msg)
				if not accumulated_response.strip():fallback_message='The model did not generate any response. Please try again.';response_placeholder.markdown(fallback_message,unsafe_allow_html=_A)
				if vision_model and uploaded_files:current_key=st.session_state.file_uploader_key;key_number=int(current_key.split('_')[1])+1;st.session_state.file_uploader_key=f"uploader_{key_number}"
				hide_reasoning_window();st.rerun()
			except requests.RequestException as e:
				error_message=f"Error communicating with API: {e}"
				if'401'in str(e):error_message+='\n\n⚠️ Invalid API Key - Please check your credentials'
				elif'429'in str(e):error_message+='\n\n⚠️ Rate limit exceeded - Try again later'
				st.error(error_message);st.session_state.messages.append({_G:A,_D:f"Error: {e}"});response_placeholder.markdown(error_message,unsafe_allow_html=_A);hide_reasoning_window()
	elif prompt and not selected_model:st.error('Please select a model before submitting a prompt.')
	if len(st.session_state.messages)>150:
		with st.sidebar:st.info('💡 Consider clearing chat for optimal performance')
if __name__=='__main__':main()