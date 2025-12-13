import streamlit as st
import requests
import subprocess
import json
import base64
from io import BytesIO, StringIO
import PyPDF2
import os
import re
from datetime import datetime
import hashlib
from functools import lru_cache
import time
import random
import contextlib
import traceback
import sys
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation

# Version tracking
APP_VERSION = "3.9.2"  # Added online API support

# API Settings
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Check if running in cloud environment (Streamlit Cloud, Heroku, etc.)
IS_CLOUD_DEPLOYMENT = os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('DYNO') or not os.path.exists('/usr/local/bin/ollama')

# Ollama settings (only used if local deployment detected)
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_HOST = DEFAULT_OLLAMA_HOST
DEFAULT_MODEL = "qwen3:4b"

# Performance settings
CACHE_TTL = 300
MAX_CACHE_ENTRIES = 100
MAX_REASONING_LINES = 5  # Increased to show more context
REASONING_UPDATE_INTERVAL = 0.2  # More frequent updates

# Code file extensions and their corresponding languages
CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".java": "java",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".md": "markdown",
    ".sh": "bash",
    ".sql": "sql",
    ".bat": "batch",
}

# Image file extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

def initialize_ui():
    st.set_page_config(
        page_title="RadioSport AI",
        page_icon="🧟",
        layout="centered",
        menu_items={
            'Report a Bug': "https://github.com/rkarikari/RadioSport-chat",
            'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."
        }
    )

    # Initialize session state
    session_defaults = {
        "cache_stats": {
            "model_cache_hits": 0,
            "model_cache_misses": 0,
            "document_cache_hits": 0,
            "document_cache_misses": 0,
            "last_reset": datetime.now()
        },
        "file_uploader_key": "uploader_0",
        "reasoning_window": None,
        "messages": [],
        "auto_run_plots": True,
        "ollama_models": [],
        "doc_cache": {},
        "last_reasoning_update": time.time(),
        "base64_cache": {},
        "thinking_content": "",
        "in_thinking_block": False,
        "reasoning_window_id": f"reasoning_{time.time()}",
        "api_provider": "Cloud" if IS_CLOUD_DEPLOYMENT else "Local",
        "api_keys": {
            "openrouter": ""
        },
        "selected_online_model": "mistralai/mistral-7b-instruct"
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Add this migration step to handle existing sessions
    if "api_keys" in st.session_state and "deepseek" in st.session_state.api_keys:
        st.session_state.api_keys["openrouter"] = st.session_state.api_keys["deepseek"]
        del st.session_state.api_keys["deepseek"]
    
    # Ensure openrouter key exists
    if "api_keys" in st.session_state and "openrouter" not in st.session_state.api_keys:
        st.session_state.api_keys["openrouter"] = ""

    # CSS with reasoning window styles
    st.markdown("""
    <style>
    .sidebar-title {
        font-size: 24px !important;
        font-weight: bold;
        margin-bottom: 0px !important;
    }
    .version-text {
        font-size: 12px !important;
        margin-top: 0px !important;
        color: #666666;
    }
    .code-block {
        background-color: #f6f8fa;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        white-space: pre-wrap;
        font-size: 14px;
    }
    .cache-stats {
        font-size: 11px;
        color: #888888;
        padding: 8px;
        background-color: #f8f9fa;
        border-radius: 4px;
        margin-top: 10px;
    }
    .cache-stats table {
        width: 100%;
        font-size: 10px;
    }
    .cache-stats th, .cache-stats td {
        padding: 2px 4px;
        text-align: left;
    }
    .cache-hit { color: #28a745; }
    .cache-miss { color: #dc3545; }
    .thinking-display {
        position: fixed;
        top: 20px;
        right: 10px;
        z-index: 1000;
        background: rgba(248,249,250,0.98);
        backdrop-filter: blur(12px);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 0;
        max-width: 400px;
        max-height: 300px;
        overflow: hidden;
        font-size: 13px;
        color: #495057;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        opacity: 0;
        transform: translateY(-10px);
    }
    .thinking-display.visible {
        opacity: 1;
        transform: translateY(0);
    }
    .thinking-header {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        padding: 8px 12px;
        font-weight: 600;
        font-size: 12px;
        border-radius: 11px 11px 0 0;
    }
    .thinking-content {
        padding: 12px;
        max-height: 140px;
        overflow-y: auto;
        line-height: 1.4;
        word-wrap: break-word;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .thinking-content::-webkit-scrollbar {
        width: 6px;
    }
    .thinking-content::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 2px;
    }
    .thinking-content::-webkit-scrollbar-thumb {
        background: #007bff;
        border-radius: 2px;
    }
    .plot-container {
        margin: 15px 0;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 15px;
        background-color: #f8f9fa;
    }
    .plot-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    
    /* Auto-scroll animation */
    .thinking-content {
        animation: scrollToBottom 0.1s ease-out;
    }
    @keyframes scrollToBottom {
        to { scroll-behavior: smooth; }
    }
    
    /* Toggle switch styling */
    .stToggle label {
        font-weight: normal;
    }
    .stToggle>div {
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # MathJax loading
    st.markdown("""
    <script>
    if (!window.MathJax) {
        window.MathJax = {
            tex: {
                inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
                displayMath: [['$$', '$$']],
                processEscapes: true
            },
            startup: {
                ready: () => {
                    MathJax.startup.defaultReady();
                }
            }
        };
    }
    </script>
    """, unsafe_allow_html=True)

    # Main window title
    st.title("RadioSport AI 🧟")

def get_ollama_host():
    """Get the current Ollama host from session state or default"""
    return st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST)

def update_all_ollama_host_references(new_host):
    """Update OLLAMA_HOST in all imported modules"""
    import sys
    
    # Update in current module
    globals()['OLLAMA_HOST'] = new_host
    
    
    # Update in all modules that have imported OLLAMA_HOST
    modules_updated = []
    for module_name, module in sys.modules.items():
        if hasattr(module, 'OLLAMA_HOST'):
            old_value = getattr(module, 'OLLAMA_HOST')
            setattr(module, 'OLLAMA_HOST', new_host)
            modules_updated.append(f"{module_name}: {old_value} -> {new_host}")
            
        # Also check for lowercase version
        if hasattr(module, 'ollama_host'):
            old_value = getattr(module, 'ollama_host')
            setattr(module, 'ollama_host', new_host)
            modules_updated.append(f"{module_name}.ollama_host: {old_value} -> {new_host}")
    
    
    # Also update any common variable names that might be used
    for module_name, module in sys.modules.items():
        for attr_name in ['OLLAMA_BASE_URL', 'ollama_base_url', 'base_url', 'host_url']:
            if hasattr(module, attr_name):
                setattr(module, attr_name, new_host)
                print(f"DEBUG: Also updated {module_name}.{attr_name} to {new_host}")


# Model fetching with caching
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ollama_models_cached():
    import requests
    import json
    
    # Return empty list silently if cloud deployment
    if IS_CLOUD_DEPLOYMENT:
        return []
    
    try:
        # Use the OLLAMA_HOST for API calls
        api_url = f"{get_ollama_host()}/api/tags"
        
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        models = []
        
        for model_info in data.get("models", []):
            model_name = model_info.get("name", "")
            if model_name:
                models.append(model_name)
        
        if not models:
            return []
        
        # Enhanced embedding model filtering
        embedding_prefixes = (
            "nomic-embed-text", "all-minilm", "mxbai-embed", 
            "snowflake-arctic-embed", "bge-", "gte-", "e5-"
        )
        
        # Filter out embedding models and empty entries
        filtered_models = []
        for model in models:
            model = model.strip()
            if model and not any(model.lower().startswith(prefix.lower()) for prefix in embedding_prefixes):
                filtered_models.append(model)
        
        return filtered_models
        
    except:
        # Silent fail for cloud deployments
        return []

def _parse_table_output(stdout):
    """Helper function - no longer needed with API approach"""
    # This function is kept for compatibility but not used
    pass

def get_ollama_models():
    start_time = time.time()
    result = get_ollama_models_cached()
    end_time = time.time()
    
    # Initialize cache stats if not present
    if "cache_stats" not in st.session_state:
        st.session_state.cache_stats = {
            "model_cache_hits": 0,
            "model_cache_misses": 0
        }
    
    if end_time - start_time < 0.1 and result:
        st.session_state.cache_stats["model_cache_hits"] += 1
    else:
        st.session_state.cache_stats["model_cache_misses"] += 1
    
    return result


# Model type checking with improved pattern matching
@lru_cache(maxsize=128)
def is_vision_model(model):
    """Enhanced vision model detection with more comprehensive patterns"""
    import re
    
    # Handle None or empty model names
    if not model:
        return False
        
    model_lower = model.lower()
    
    # Comprehensive vision model patterns
    vision_patterns = [
        r"qwen2\.5?vl:",
        r"gemma3[^:]*:",
        r"llava:",
        r"moondream:",
        r"bakllava:",
        r"minicpm-v:",
        r"yi-vl:",
        r"internvl:",
        r"cogvlm:",
        r"vision",
        r"visual",
        r"multimodal",
        r"-v:",  # Common vision model suffix
        r"vl-"   # Vision-language prefix
    ]
    
    return any(re.search(pattern, model_lower) for pattern in vision_patterns)


@lru_cache(maxsize=128)
def is_qwen3_model(model):
    """Enhanced Qwen3 model detection"""
    import re
    
    # Handle None or empty model names
    if not model:
        return False
        
    return bool(re.search(r"qwen3:", model.lower()))




# File type detection
@lru_cache(maxsize=256)
def get_file_type(file_name):
    ext = os.path.splitext(file_name)[1].lower()
    if ext in CODE_EXTENSIONS:
        return "code", CODE_EXTENSIONS[ext]
    elif ext == ".txt":
        return "text", None
    elif ext == ".pdf":
        return "pdf", None
    elif ext in IMAGE_EXTENSIONS:
        return "image", None
    return None, None

# Message splitting with caching
@st.cache_data(ttl=CACHE_TTL, max_entries=MAX_CACHE_ENTRIES, show_spinner=False)
def split_message_cached(content):
    CODE_BLOCK_PATTERN = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
    
    parts = []
    last_end = 0
    
    for match in CODE_BLOCK_PATTERN.finditer(content):
        start, end = match.span()
        if last_end < start:
            text_content = content[last_end:start]
            if text_content.strip():
                parts.append({"type": "text", "content": text_content})
        
        language = match.group(1) or "text"
        code = match.group(2)
        parts.append({"type": "code", "language": language, "code": code})
        last_end = end
    
    if last_end < len(content):
        remaining_content = content[last_end:]
        if remaining_content.strip():
            parts.append({"type": "text", "content": remaining_content})
    
    return parts

def split_message(content):
    return split_message_cached(content)

# Document processing
def process_single_file(file_data):
    file_content, file_type, file_name = file_data
    try:
        if file_type in ("text", "code"):
            try:
                return file_content.decode("utf-8"), None
            except UnicodeDecodeError:
                return file_content.decode("utf-8", errors="replace"), None
            
        elif file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            content_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        content_parts.append(text)
                except Exception as e:
                    return None, f"error: Failed to read PDF page {page_num + 1}: {e}"
            
            return "\n".join(content_parts), None
            
        elif file_type == "image":
            return file_content, None
            
        return None, "unsupported"
        
    except Exception as e:
        return None, f"error: {str(e)}"

def process_document(file_content, file_type, file_name):
    content_hash = hashlib.md5(file_content).hexdigest()
    cache_key = f"{content_hash}_{file_type}_{file_name}"
    
    # Use dedicated doc_cache in session state
    if cache_key in st.session_state.doc_cache:
        st.session_state.cache_stats["document_cache_hits"] += 1
        return st.session_state.doc_cache[cache_key]
    
    result = process_single_file((file_content, file_type, file_name))
    
    # Apply LRU eviction policy
    if len(st.session_state.doc_cache) >= MAX_CACHE_ENTRIES:
        oldest_key = next(iter(st.session_state.doc_cache))
        del st.session_state.doc_cache[oldest_key]
    
    st.session_state.doc_cache[cache_key] = result
    st.session_state.cache_stats["document_cache_misses"] += 1
    
    return result

# Base64 encoding with caching
def get_base64_image(file_content):
    content_hash = hashlib.md5(file_content).hexdigest()
    
    # Check cache
    if content_hash in st.session_state.base64_cache:
        return st.session_state.base64_cache[content_hash]
    
    # Compute and cache
    base64_str = base64.b64encode(file_content).decode("utf-8")
    st.session_state.base64_cache[content_hash] = base64_str
    
    # Apply LRU eviction policy
    if len(st.session_state.base64_cache) >= MAX_CACHE_ENTRIES:
        oldest_key = next(iter(st.session_state.base64_cache))
        del st.session_state.base64_cache[oldest_key]
    
    return base64_str

# Reasoning window functions
def show_reasoning_window():
    if not st.session_state.reasoning_window:
        st.session_state.reasoning_window = st.empty()
    
    st.session_state.reasoning_window.markdown(
        f'''
        <div id="thinking-display" class="thinking-display visible">
            <div class="thinking-header">🤔 Reasoning Process</div>
            <div id="thinking-content-{st.session_state.reasoning_window_id}" class="thinking-content">Thinking...</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

def update_reasoning_window(content):
    if st.session_state.reasoning_window:
        # Trim long content to prevent excessive DOM growth
        if len(content) > 1500:
            content = "..." + content[-1500:]

        st.session_state.reasoning_window.markdown(
            f'''
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
            ''',
            unsafe_allow_html=True
        )


def hide_reasoning_window():
    if st.session_state.reasoning_window:
        st.session_state.reasoning_window.empty()
        st.session_state.reasoning_window = None


def activate_teacher_mode():
    """
    Activates The Ultimate Master Teacher mode for enhanced learning experience.
    Returns the teacher system prompt to be used in API calls.
    """
    teacher_prompt = """You are now **The Ultimate Master Teacher** — the most effective, intuitive, and insightful educator the world has ever known.

🎓 Your goal is to **transform any student into a top performer**, regardless of their current level of knowledge or skill.

🔍 Core Responsibilities:
1. **Diagnose Weaknesses**:
   - Interactively question the student to discover conceptual gaps.
   - Use strategic questioning, not overwhelming complexity.
   - Drill down until the *true foundational weakness* is revealed.

2. **Deconstruct and Rebuild**:
   - Gently correct misconceptions.
   - Break down the complex subject into digestible, intuitive building blocks.
   - Use analogies, step-by-step scaffolding, and Socratic questioning.

3. **Mastery Through Practice**:
   - Provide focused exercises tailored to the exact weakness.
   - Evaluate responses, give feedback, adapt difficulty.
   - Use iterative refinement to ensure understanding is rock-solid.

4. **Track Progress**:
   - Maintain a mental model of the student's evolving knowledge.
   - Avoid repetition of mastered concepts.
   - Escalate toward higher levels of abstraction, problem-solving, and creativity.

5. **Motivate & Empower**:
   - Inspire confidence without false praise.
   - Encourage curiosity and intellectual independence.
   - Always communicate with clarity, patience, and empathy.

📚 Subject Flexibility:
You can teach **any subject**, including but not limited to:
- Mathematics
- Science (Physics, Biology, Chemistry)
- History
- Languages (Grammar, Writing, Comprehension)
- Programming
- Logic & Reasoning
- Standardized Test Prep

🎯 Final Goal:
To produce students who:
- Understand deeply, not just memorize.
- Can teach others.
- Are creative, curious, and confident learners.
- Can solve complex problems with elegance and insight.

🧪 Begin each interaction with:
- A short greeting.
- A simple, targeted question to start identifying the student's current understanding.
- Never overwhelm. Start simple and build carefully."""
    
    return teacher_prompt

# Last user query
def get_last_user_query():
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            return msg.get("original_prompt", msg["content"])
    return None

# Plot detection
def looks_like_plotting_code(code_str):
    plot_keywords = ("plt.", "sns.", "plot(", "figure(", "show(", "bar(", "hist(", "scatter(", "FuncAnimation", "fig.")
    return any(keyword in code_str for keyword in plot_keywords)

@contextlib.contextmanager
def capture_plots():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

# Animation dependencies check
def check_animation_dependencies():
    """Check if animation dependencies are available"""
    try:
        from matplotlib.animation import PillowWriter
        return True
    except ImportError:
        return False

# Optimized plot execution
# Updated execute_plot_code function signature and implementation
def execute_plot_code(code_str, model_name=None, enable_auto_correct=False, ollama_host=None):
    
    if ollama_host is None:
        ollama_host = get_ollama_host()
    
    plots = []
    gifs = []  # Store GIF data as bytes
    error = None
    corrected_code = None
    
    # Create a safe environment for execution with animation support
    safe_env = {
        'plt': plt,
        'np': np,
        'pd': pd,
        'sns': sns,
        'matplotlib': matplotlib,
        'numpy': np,
        'pandas': pd,
        'seaborn': sns,
        'FuncAnimation': FuncAnimation,
        'PillowWriter': PillowWriter,
        'animation': animation,
        '__builtins__': {
            k: v for k, v in __builtins__.items() 
            if k not in ('open', 'exec', 'globals', 'locals')
        }
    }
    
    # Override plt.show() to prevent blocking
    original_show = safe_env['plt'].show
    safe_env['plt'].show = lambda: None
    
    current_code = code_str
    max_retries = 2 if enable_auto_correct else 1
    
    for attempt in range(max_retries):
        try:
            with capture_plots():
                prev_figs = plt.get_fignums()
                
                # Execute the code
                exec(current_code, safe_env)
                
                # Capture animations from the environment first
                anims = []
                for name, obj in safe_env.items():
                    if isinstance(obj, FuncAnimation):
                        anims.append(obj)
                
                # Also check for animations stored in variables like 'ani'
                if 'ani' in safe_env and isinstance(safe_env['ani'], FuncAnimation):
                    if safe_env['ani'] not in anims:
                        anims.append(safe_env['ani'])
                
                # Only capture static plots if no animations were created
                if not anims:
                    # Capture new figures (static plots)
                    new_figs = [fig for fig in plt.get_fignums() if fig not in prev_figs]
                    
                    for fig_num in new_figs:
                        fig = plt.figure(fig_num)
                        try:
                            buf = BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                            buf.seek(0)
                            plots.append(buf.getvalue())
                        except Exception as e:
                            error = f"Error saving plot: {str(e)}"
                        finally:
                            plt.close(fig)
                
                # Process animations as GIFs
                if anims and check_animation_dependencies():
                    for i, anim in enumerate(anims):
                        try:
                            # Create a temporary file for the GIF
                            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
                                temp_name = temp_file.name
                            
                            # Use PillowWriter for GIF with optimized settings
                            writer = PillowWriter(
                                fps=15,
                                bitrate=1800
                            )
                            
                            # Save animation as GIF
                            anim.save(
                                temp_name,
                                writer=writer,
                                dpi=120,
                                savefig_kwargs={
                                    'bbox_inches': 'tight', 
                                    'pad_inches': 0.1,
                                    'facecolor': 'white',
                                    'edgecolor': 'none'
                                }
                            )
                            
                            # Verify and read GIF
                            if os.path.exists(temp_name) and os.path.getsize(temp_name) > 1000:
                                with open(temp_name, 'rb') as f:
                                    gif_bytes = f.read()
                                
                                if len(gif_bytes) > 1000:
                                    gifs.append(gif_bytes)
                                else:
                                    raise Exception("Generated GIF file is too small")
                            else:
                                raise Exception("GIF file was not created or is empty")
                            
                            # Clean up temporary file
                            if os.path.exists(temp_name):
                                os.unlink(temp_name)
                                
                        except Exception as e:
                            # Clean up failed GIF attempt
                            if 'temp_name' in locals() and os.path.exists(temp_name):
                                os.unlink(temp_name)
                            
                            error = f"Animation {i+1} failed: {str(e)}"
                        
                        finally:
                            # Always close the figure associated with the animation
                            if hasattr(anim, 'fig'):
                                plt.close(anim.fig)
                
                # Close any remaining figures
                for fig_num in plt.get_fignums():
                    if fig_num not in prev_figs:
                        plt.close(fig_num)
                
                # If we got here, execution was successful
                break
                
        except Exception as e:
            execution_error = traceback.format_exc()  # Capture full traceback
            
            # Try auto-correction if enabled and we have retries left
            if enable_auto_correct and attempt < max_retries - 1 and model_name:
                try:
                    corrected_code = auto_correct_code_with_llm(current_code, execution_error, model_name, ollama_host)
                    if corrected_code and corrected_code.strip() != current_code.strip():
                        current_code = corrected_code
                        continue  # Try again with corrected code
                except Exception as correction_error:
                    error = f"Auto-correction failed: {correction_error}. Original error: {execution_error}"
                    break
            
            error = f"Error executing plot code: {execution_error}"
            break
    
    return plots, gifs, error, corrected_code

# Updated display_chat_message function call (around line 670)
def display_chat_message(msg, msg_index):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            prompt_text = msg.get("original_prompt", msg["content"])
            st.markdown(prompt_text, unsafe_allow_html=True)
            
            if is_vision_model(msg.get("model", "")):
                for file in msg.get("files", []):
                    if file["type"] == "image":
                        st.image(
                            file["content"],
                            caption=file["name"],
                            use_container_width=True
                        )
        else:
            parts = split_message(msg["content"])
            for code_index, part in enumerate(parts):
                if part["type"] == "text":
                    st.markdown(part["content"], unsafe_allow_html=True)
                elif part["type"] == "code":
                    if (part["language"] == "python" and 
                        st.session_state.auto_run_plots and 
                        looks_like_plotting_code(part["code"])):
                        
                        # Use dual indication: spinner + reasoning window (option 3)
                        with st.spinner("🎬 Generating visualizations and animations..."):
                            # Get the current model from session state
                            current_model = None
                            for msg in reversed(st.session_state.messages):
                                if msg["role"] == "user" and "model" in msg:
                                    current_model = msg["model"]
                                    break
                            
                            # Execute with auto-correction if enabled
                            plots, gifs, error, corrected_code = execute_plot_code(
                                part["code"], 
                                model_name=current_model,
                                enable_auto_correct=(
                                    st.session_state.auto_run_plots and 
                                    st.session_state.get("auto_correct_code", False)
                                ),
                                ollama_host=None  # Add this parameter
                            )
                            
                        if plots or gifs:
                            # Only display static plots if no animations were created
                            if plots and not gifs:
                                for i, plot_img in enumerate(plots):
                                    st.markdown(
                                        f'<div class="plot-container">'
                                        f'<div class="plot-title">Plot #{i+1}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                    st.image(plot_img, use_container_width=True)
                            
                            # Display GIF animations (prioritized over static plots)
                            if gifs:
                                for i, gif_bytes in enumerate(gifs):
                                    st.markdown(
                                        f'<div class="plot-container">'
                                        f'<div class="plot-title">Animation #{i+1} (GIF)</div>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                    
                                    try:
                                        # Display GIF with proper sizing
                                        st.image(
                                            gif_bytes, 
                                            use_container_width=True,
                                            caption=f"Animated GIF ({len(gif_bytes):,} bytes)"
                                        )
                                        
                                        # Success indicator and download option in columns
                                        col1, col2 = st.columns([2, 1])
                                        with col1:
                                            st.success(f"✅ GIF animation rendered successfully")
                                        with col2:
                                            st.download_button(
                                                label="📥 Download GIF",
                                                data=gif_bytes,
                                                file_name=f"animation_{msg_index}_{code_index}_{i}.gif",
                                                mime="image/gif",
                                                key=f"dl_anim_{msg_index}_{code_index}_{i}",
                                                use_container_width=True
                                            )
                                    except Exception as display_error:
                                        st.error(f"Failed to display GIF animation: {display_error}")
                                        # Still offer download as fallback
                                        st.download_button(
                                            label="📥 Download Animation (GIF)",
                                            data=gif_bytes,
                                            file_name=f"animation_{msg_index}_{code_index}_{i}.gif",
                                            mime="image/gif",
                                            key=f"dl_anim_fallback_{msg_index}_{code_index}_{i}",
                                            use_container_width=True
                                        )
                            
                            # Show corrected code if auto-correction was used
                            if corrected_code and corrected_code != part["code"]:
                                with st.expander("🔧 Auto-Corrected Code", expanded=True):
                                    st.code(corrected_code, language="python")
                                    st.download_button(
                                        label="💾 Download Corrected Code",
                                        data=corrected_code,
                                        file_name=f"corrected_code_{msg_index}_{code_index}.py",
                                        mime="text/plain",
                                        key=f"dl_corrected_{msg_index}_{code_index}"
                                    )
                            
                            with st.expander("View Code", expanded=False):
                                st.code(part["code"], language="python")
                                
                                filename = f"code_{msg_index}_{code_index}.py"
                                st.download_button(
                                    label="💾 Download Code",
                                    data=part["code"],
                                    file_name=filename,
                                    mime="text/plain",
                                    key=f"dl_{msg_index}_{code_index}_plot",
                                    use_container_width=True
                                )
                            continue
                        elif error:
                            st.error(error)
                    
                    # Show code block regardless of success/failure
                    with st.expander("View Code", expanded=False):
                        st.code(part["code"], language=part["language"])
                        ext = part['language'] if part['language'] != 'text' else 'txt'
                        filename = f"code_{msg_index}_{code_index}.{ext}"
                    
                    st.download_button(
                        label="💾 Download",
                        data=part["code"],
                        file_name=filename,
                        mime="text/plain",
                        key=f"dl_{msg_index}_{code_index}",
                        use_container_width=False
                    )



# Cache for model information from API
MODEL_INFO_CACHE = {}

def get_model_info(model_id):
    """Get model information from cached API data"""
    return MODEL_INFO_CACHE.get(model_id, {
        "context": None,  # Changed from "Unknown" to None for better handling
        "name": model_id.split("/")[-1].replace("-", " ").title()
    })

def format_context_length(context):
    """Format context length for display"""
    if context is None or context == "Unknown":
        return "Unknown"
    elif isinstance(context, str):
        # Try to convert string to int first
        try:
            context = int(context.replace('k', '000').replace('K', '000').replace('M', '000000').replace('m', '000000'))
        except (ValueError, AttributeError):
            return context  # Return as-is if can't convert
    
    # Now format the integer
    if isinstance(context, int):
        if context >= 1000000:
            return f"{context//1000000}M tokens"
        elif context >= 1000:
            return f"{context//1000}K tokens"
        else:
            return f"{context:,} tokens"
    else:
        return str(context)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_free_openrouter_models():
    """Dynamically load OpenRouter free models"""
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                'User-Agent': 'RadioSport-AI/1.0',
                'Accept': 'application/json'
            },
            timeout=30
        )
        response.raise_for_status()
        
        models_data = response.json()
        free_models = []
        
        # Filter for free models - KEEP FULL MODEL OBJECTS
        for model in models_data.get('data', []):
            pricing = model.get('pricing', {})
            prompt_price = float(pricing.get('prompt', '0'))
            completion_price = float(pricing.get('completion', '0'))
            
            if prompt_price == 0 and completion_price == 0:
                model_id = model.get('id', '')
                if model_id:
                    # Return the full model object, not just the ID
                    free_models.append(model)
        
        if not free_models:
            # Silent fail in cloud mode, only show error in local mode
            if not IS_CLOUD_DEPLOYMENT:
                st.error("No free models found in OpenRouter API response.")
            return []
        
        # Sort alphabetically by model ID
        free_models.sort(key=lambda x: x.get('id', ''))
        return free_models
        
    except Exception as e:
        # Silent fail in cloud mode
        if not IS_CLOUD_DEPLOYMENT:
            st.error(f"Error loading OpenRouter models: {str(e)}")
        return []

def get_load_balanced_api_key(provider: str) -> str:
    """Get load-balanced API key with backward compatibility."""
    keys = []
    
    # Collect keys from multiple sources (backward compatible)
    try:
        # Legacy format: [openrouter] api_key = ""
        if provider.lower() == "openrouter":
            try:
                key = st.secrets["openrouter"]["api_key"]
                if key and key.strip(): keys.append(key.strip())
            except: pass
            
            # Section format: [openrouter] api_key1 = "", api_key2 = ""
            for i in range(1, 11):
                try:
                    key = st.secrets["openrouter"][f"api_key{i}"]
                    if key and key.strip(): keys.append(key.strip())
                except: pass
            
            # Section list format: [openrouter] api_keylist = "", "", ""
            try:
                key_list = st.secrets["openrouter"]["api_keylist"]
                if isinstance(key_list, (list, tuple)):
                    keys.extend([k.strip() for k in key_list if k and k.strip()])
                elif isinstance(key_list, str) and key_list.strip():
                    keys.append(key_list.strip())
            except: pass
        
        # Current format: [general] OPENROUTER_API_KEY = ""
        base_key = f"{provider.upper()}_API_KEY"
        try:
            key = st.secrets["general"][base_key]
            if key and key.strip(): keys.append(key.strip())
        except: pass
        
        # Multiple keys: [general] OPENROUTER_API_KEY_1 = ""
        for i in range(1, 11):
            try:
                key = st.secrets["general"][f"{base_key}_{i}"]
                if key and key.strip(): keys.append(key.strip())
            except: pass
                
    except Exception:
        return ""
    
    if not keys: return ""
    if len(keys) == 1: return keys[0]
    
    # Initialize session tracking
    if "api_usage" not in st.session_state:
        st.session_state.api_usage = {}
    if provider not in st.session_state.api_usage:
        st.session_state.api_usage[provider] = {"idx": 0, "failed": set(), "count": {}}
    
    usage = st.session_state.api_usage[provider]
    valid_keys = [k for k in keys if k not in usage["failed"]]
    if not valid_keys:
        usage["failed"].clear()  # Reset if all failed
        valid_keys = keys
    
    # Round robin selection
    selected = valid_keys[usage["idx"] % len(valid_keys)]
    usage["idx"] = (usage["idx"] + 1) % len(valid_keys)
    usage["count"][selected] = usage["count"].get(selected, 0) + 1
    usage["last_used"] = time.time()
    usage["selected_key"] = selected
    
    return selected

def mark_key_failed(provider: str, key: str):
    """Mark key as failed for load balancing."""
    if "api_usage" in st.session_state and provider in st.session_state.api_usage:
        st.session_state.api_usage[provider]["failed"].add(key)

# Enhanced API call function with context awareness
def call_openrouter_api(messages, model, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rkarikari/RadioSport-chat",
        "X-Title": "RadioSport AI"
    }
    
    # Get model context length for optimal token usage
    model_info = get_model_info(model)
    max_tokens = 2048  # Default
    
    # Adjust max_tokens based on context length
    if isinstance(model_info.get('context'), int):
        context_length = model_info['context']
        # Reserve some tokens for the response (about 25% of context or 2048, whichever is smaller)
        max_tokens = min(2048, max(512, context_length // 4))
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60
    )
    response.raise_for_status()
    return response

def main():
    initialize_ui()

# Sidebar controls
    with st.sidebar:
        st.markdown('<div class="sidebar-title">RadioSport AI 🧟</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="version-text">Version {APP_VERSION}</div>', unsafe_allow_html=True)
        
        st.text("")
        
        # Only show provider selection if not cloud deployment
        if IS_CLOUD_DEPLOYMENT:
            api_provider = "Cloud"
            st.session_state.api_provider = "Cloud"
        else:
            api_provider = st.radio(
                "AI provider:",
                ["Local", "Cloud"],
                index=0,
                key="api_provider"
            )
        
        selected_model = DEFAULT_MODEL
        
        st.subheader("Model Selection")
        if api_provider == "Local":
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("Refresh Models", use_container_width=True):
                    get_ollama_models_cached.clear()
                    st.session_state.ollama_models = get_ollama_models()
                    st.rerun()
            
            with col2:
                auto_refresh = st.checkbox("Auto", help="Auto-refresh models")
            
            if not st.session_state.ollama_models or auto_refresh:
                st.session_state.ollama_models = get_ollama_models()
            else:
                st.session_state.cache_stats["model_cache_hits"] += 1
            
            ollama_models = st.session_state.ollama_models
            
            if ollama_models:
                default_index = 0
                if DEFAULT_MODEL in ollama_models:
                    default_index = ollama_models.index(DEFAULT_MODEL)
                selected_model = st.selectbox(
                    "Select a model:",
                    ollama_models,
                    index=default_index,
                    help=f"Available models: {len(ollama_models)}"
                )
            else:
                st.warning(f"Models Unavailable: {st.session_state.get('ollama_host', OLLAMA_HOST)}.")
                selected_model = None
        
        # Ollama Host Configuration (collapsed by default)
        if api_provider == "Local":
            with st.expander("🔧 Local ", expanded=False):
                # Initialize OLLAMA_HOST in session state if not present
                if "ollama_host" not in st.session_state:
                    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST
                
                new_host = st.text_input(
                    "Ollama Host URL:",
                    value=st.session_state.ollama_host,
                    help="Default: http://localhost:11434\nRemote: http://192.168.x.x:11434",
                    placeholder="http://localhost:11434",
                    key="ollama_host_input"
                )
                
                if new_host != st.session_state.ollama_host:
                    st.session_state.ollama_host = new_host
                    
                    # Update OLLAMA_HOST everywhere it's been imported
                    update_all_ollama_host_references(new_host)
                    
                    # Clear model cache when host changes
                    get_ollama_models_cached.clear()
                    st.session_state.ollama_models = []
                    
                    # Force refresh of models from new host
                    try:
                        
                        # Fetch models from the new host immediately
                        new_models = get_ollama_models_cached()
                        st.session_state.ollama_models = new_models
                        st.success(f"✅ Connected to {new_host} - Found {len(new_models)} models")
                    except Exception as e:
                        st.error(f"❌ Failed to connect to {new_host}: {str(e)}")
                        print(f"DEBUG: Error details: {e}")
                        st.session_state.ollama_models = []
                    
                    st.rerun()
                
                # Show current host status
                if st.session_state.ollama_host:
                    st.info(f"Current host: {st.session_state.ollama_host}")
#-------------------------------                
        # Updated API provider section
        elif api_provider == "Cloud":
            try:
                api_key = get_load_balanced_api_key("openrouter")
                if api_key:
                    st.session_state.api_keys["openrouter"] = api_key
                    
                    # Show detailed key info if multiple keys
                    if "api_usage" in st.session_state and "openrouter" in st.session_state.api_usage:
                        usage = st.session_state.api_usage["openrouter"]
                        if len(usage.get("count", {})) > 1:
                            total_keys = len(usage["count"])
                            failed_keys = len(usage["failed"])
                            active_keys = total_keys - failed_keys
                            current_key = usage.get("selected_key", "")
                            
                            # Find current key position
                            valid_keys = [k for k in usage["count"].keys() if k not in usage["failed"]]
                            try:
                                current_pos = valid_keys.index(current_key) + 1
                            except:
                                current_pos = 1
                            
                            # Calculate time since last use (estimate time left)
                            last_used = usage.get("last_used", 0)
                            time_since = int(time.time() - last_used)
                            
                            # Show key info
                            st.info(f"🔑Key: {current_pos} of {active_keys} | "
                                   f"Used: {time_since}s ago | "
                                   f"Total: {usage['count'].get(current_key, 0)} calls")
                else:
                    raise KeyError("No OPENROUTER API keys found")
            except:
                st.session_state.api_keys["openrouter"] = st.text_input(
                    "OpenRouter API Key (free tier)",
                    type="password",
                    value=st.session_state.api_keys.get("openrouter", ""),
                    help="Get free API key at https://openrouter.ai"
                )
            
            # Cache models in session state to avoid repeated API calls
            if 'openrouter_models' not in st.session_state:
                with st.spinner("Loading available models..."):
                    st.session_state.openrouter_models = get_free_openrouter_models()
            
            available_models = st.session_state.openrouter_models
            
            if not available_models:
                st.error("Could not load models. Please refresh the page to try again.")
                if st.button("🔄 Refresh Models"):
                    del st.session_state.openrouter_models
                    st.rerun()
                st.stop()
            
            # Create display names and model IDs
            model_display = [f"{model.get('name', model['id'])}" for model in available_models]
            model_ids = [model['id'] for model in available_models]
#-->
            col1, col2 = st.columns([6, 1])
            with col1:            
                selected_index = st.selectbox(
                    "Models (Free)",
                    range(len(model_display)),
                    format_func=lambda x: model_display[x],
                    index=0,
                    help="Showing only free models from OpenRouter. List updates when refreshed."
                )
                
                selected_model = model_ids[selected_index]
                selected_model_info = available_models[selected_index]
                
                # Display context length information
                if selected_model_info:
                   #st.markdown("---")
                    
                    # Display context length if available
                    if 'context_length' in selected_model_info:
                        context_length = selected_model_info['context_length']
                        st.caption(f"Context: {context_length:,} tokens")
            with col2:                 
                # Add refresh button
                if st.button("🔄"):
                    if 'openrouter_models' in st.session_state:
                        del st.session_state.openrouter_models
                    st.rerun()
                        
#-----------------------------        
        # File upload only for local and OpenRouter
        st.subheader("File Upload")
        # Always include all file types regardless of model
        code_extensions = [ext.lstrip('.') for ext in CODE_EXTENSIONS.keys()]
        file_types = ["txt", "pdf"] + code_extensions + [ext.lstrip('.') for ext in IMAGE_EXTENSIONS]
        

        uploaded_files = st.file_uploader(
            "Upload files:",
            type=file_types,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(file_types)}",
            key=st.session_state.file_uploader_key
        )

        
        # Chat Controls in expander with toggle switches
        with st.expander("⚙️ Chat Controls", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.enable_reasoning = st.toggle(
                    "Reasoning",
                    value=st.session_state.get("enable_reasoning", False),
                    help="Enable reasoning for qwen3* models"
                )
                
                st.session_state.execute_uploaded_code = st.toggle(
                    "Run Code",
                    value=st.session_state.get("execute_uploaded_code", False),
                    help="Executes uploaded .py files directly instead of inserting them into the chat"
                )
                
            with col2:
                st.session_state.auto_run_plots = st.toggle(
                    "AutoPlot",
                    value=st.session_state.get("auto_run_plots", True),
                    help="Automatically execute Python code that generates plots or animations"
                )
                
                st.session_state.teacher_mode = st.toggle(
                    "Teacher",
                    value=st.session_state.get("teacher_mode", False),
                    help="Activate The Ultimate Master Teacher mode for enhanced learning experience"
                )

        # Clear button
        clear_chat = st.button("Clear", use_container_width=True)
        
        if clear_chat:
            st.session_state.messages = []
            st.session_state.cache_stats = {
                "model_cache_hits": 0,
                "model_cache_misses": 0,
                "document_cache_hits": 0,
                "document_cache_misses": 0,
                "last_reset": datetime.now()
            }
            st.session_state.doc_cache = {}
            st.session_state.base64_cache = {}
            st.session_state.file_uploader_key = f"uploader_{int(st.session_state.file_uploader_key.split('_')[1]) + 1}"
            st.session_state.thinking_content = ""
            st.session_state.in_thinking_block = False
            st.session_state.reasoning_window_id = f"reasoning_{time.time()}"  # Reset ID
            hide_reasoning_window()
            st.rerun()
        
        if st.session_state.messages:
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("Export", use_container_width=True):
                    export_content = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "assistant":
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            content = msg["content"]
                            export_content.append(f"[{timestamp}] Assistant:\n{content}\n{'-'*50}")
                    
                    if export_content:
                        export_text = "\n".join(export_content)
                        st.download_button(
                            label="Download",
                            data=export_text,
                            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            
            with export_col2:
                repeat_button = st.button(
                    "Repeat",
                    disabled=not bool(get_last_user_query()),
                    use_container_width=True
                )
        else:
            repeat_button = False
        
        with st.expander("📊 Cache Statistics"):
            stats = st.session_state.cache_stats
            total_model_requests = stats["model_cache_hits"] + stats["model_cache_misses"]
            total_doc_requests = stats["document_cache_hits"] + stats["document_cache_misses"]
            
            model_hit_rate = (stats["model_cache_hits"] / max(total_model_requests, 1)) * 100
            doc_hit_rate = (stats["document_cache_hits"] / max(total_doc_requests, 1)) * 100
            
            uptime = datetime.now() - stats["last_reset"]
            uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
            
            stats_html = f"""
            <div class="cache-stats">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Uptime</td><td>{uptime_str}</td></tr>
                    <tr><td colspan="2"><strong>Model Cache</strong></td></tr>
                    <tr><td>Hit Rate</td><td class="{'cache-hit' if model_hit_rate > 50 else 'cache-miss'}">{model_hit_rate:.1f}%</td></tr>
                    <tr><td>Hits</td><td class="cache-hit">{stats["model_cache_hits"]}</td></tr>
                    <tr><td>Misses</td><td class="cache-miss">{stats["model_cache_misses"]}</td></tr>
                    <tr><td colspan="2"><strong>Document Cache</strong></td></tr>
                    <tr><td>Hit Rate</td><td class="{'cache-hit' if doc_hit_rate > 50 else 'cache-miss'}">{doc_hit_rate:.1f}%</td></tr>
                    <tr><td>Hits</td><td class="cache-hit">{stats["document_cache_hits"]}</td></tr>
                    <tr><td>Misses</td><td class="cache-miss">{stats["document_cache_misses"]}</td></tr>
                    <tr><td colspan="2"><strong>Cache Info</strong></td></tr>
                    <tr><td>Doc Cache Size</td><td>{len(st.session_state.doc_cache)}</td></tr>
                    <tr><td>Base64 Cache Size</td><td>{len(st.session_state.base64_cache)}</td></tr>
                </table>
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)
            
            if st.button("Reset Stats"):
                st.session_state.cache_stats = {
                    "model_cache_hits": 0,
                    "model_cache_misses": 0,
                    "document_cache_hits": 0,
                    "document_cache_misses": 0,
                    "last_reset": datetime.now()
                }
                st.rerun()

    # Chat history display
    for msg_index, msg in enumerate(st.session_state.messages):
        display_chat_message(msg, msg_index)

    # Query input
    prompt = st.chat_input("Enter your prompt:", key="main_input")

    # Handle repeat button
    if repeat_button and get_last_user_query():
        prompt = get_last_user_query()

    # Main processing logic
    if prompt and (selected_model or st.session_state.api_provider != "Local"):
        # File processing
        files = []
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                for uploaded_file in uploaded_files:
                    try:
                        file_type, language = get_file_type(uploaded_file.name)
                        if file_type:
                            file_content = uploaded_file.getvalue()
                            content, error = process_document(file_content, file_type, uploaded_file.name)

                            if error:
                                if error == "unsupported":
                                    st.error(f"Unsupported file type: {uploaded_file.name}")
                                else:
                                    st.error(f"Error processing {uploaded_file.name}: {error}")
                                continue

                            # ✅ EXECUTE .py files if toggle is enabled
                            if (
                                st.session_state.execute_uploaded_code and
                                file_type == "code" and
                                uploaded_file.name.endswith(".py")
                            ):
                                with st.spinner(f"Executing {uploaded_file.name}..."):
                                    plots, gifs, exec_error = execute_plot_code(
                                        content,
                                        model_name=selected_model,
                                        enable_auto_correct=False,  # Removed auto-correction
                                        ollama_host=None  # Changed to None, will use get_ollama_host()
                                    )

                                    # ✅ Display GIFs first if present
                                    if gifs:
                                        st.success(f"✅ Executed: {uploaded_file.name} (GIF animation)")
                                        for i, gif_bytes in enumerate(gifs):
                                            st.image(gif_bytes, use_container_width=True, caption=f"Animation {i+1}")
                                            st.download_button(
                                                label=f"📥 Download Animation {i+1}",
                                                data=gif_bytes,
                                                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_anim{i+1}.gif",
                                                mime="image/gif",
                                                use_container_width=True,
                                                key=f"dl_{uploaded_file.name}_{i}"
                                            )

                                    # ✅ Display static plots (only if no GIFs or additionally)
                                    if plots:
                                        if not gifs:
                                            st.success(f"✅ Executed: {uploaded_file.name} (Static plot)")
                                        for i, plot_img in enumerate(plots):
                                            st.image(plot_img, use_container_width=True, caption=f"Plot {i+1}")

                                    if not gifs and not plots:
                                        st.info(f"ℹ️ Executed: {uploaded_file.name} (no plots or animations detected)")

                                    if exec_error:
                                        st.error(f"⚠️ Execution error: {exec_error}")
                                continue  # Do not attach this file to the prompt

                            # 🔗 For non-.py or skipped files, include for prompt
                            file_obj = {
                                "type": file_type,
                                "name": uploaded_file.name,
                                "content": content
                            }

                            if file_type == "image":
                                file_obj["base64"] = get_base64_image(content)

                            files.append(file_obj)

                    except Exception as e:
                        st.error(f"Error reading file {uploaded_file.name}: {e}")

        
        vision_model = is_vision_model(selected_model) if st.session_state.api_provider == "Local" else False
        
        # Content construction
        user_content = prompt
        if (st.session_state.api_provider == "Local" and 
            is_qwen3_model(selected_model) and 
            not st.session_state.get("enable_reasoning", False)):
            user_content += " /no_think"

        # ✅ Include files for BOTH local and OpenRouter models
        if (st.session_state.api_provider in ["Local", "Cloud"]) and files:
            text_files = [f for f in files if f["type"] in ("text", "code", "pdf")]
            if text_files:
                context_parts = [f["content"] for f in text_files if f["content"]]
                if context_parts:
                    context = "\n\n".join(context_parts)
                    user_content = f"Context: {context}\n\n{user_content}"
        
        # Store message
        message = {
            "role": "user",
            "content": user_content,
            "original_prompt": prompt,
            "model": selected_model,
            "files": files
        }
        st.session_state.messages.append(message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=True)
            if vision_model:
                for file in files:
                    if file["type"] == "image":
                        st.image(file["content"], caption=file["name"], use_container_width=True)
        
        # API request preparation
        messages = []

        # Add teacher mode system message if enabled
        if st.session_state.get("teacher_mode", False):
            teacher_prompt = activate_teacher_mode()
            messages.append({"role": "system", "content": teacher_prompt})

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                message = {"role": "user", "content": msg["content"]}
                
                # Handle images for BOTH local vision models AND OpenRouter models
                if msg.get("files", []):
                    images = []
                    for file in msg.get("files", []):
                        if file["type"] == "image":
                            if "base64" in file:
                                images.append(file["base64"])
                            else:
                                # Handle legacy messages without base64
                                base64_str = get_base64_image(file["content"])
                                file["base64"] = base64_str
                                images.append(base64_str)
                    
                    if images:
                        if st.session_state.api_provider == "Local" and vision_model:
                            # Ollama format
                            message["images"] = images
                        elif st.session_state.api_provider == "Cloud":
                            # OpenRouter format - convert to proper message format with image_url
                            content_parts = [{"type": "text", "text": msg["content"]}]
                            for img_base64 in images:
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                })
                            message["content"] = content_parts
                
                messages.append(message)
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Stream response with reasoning window
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            accumulated_response = ""
            start_time = time.time()
            st.session_state.thinking_content = ""
            st.session_state.in_thinking_block = False
            
            try:
                if st.session_state.api_provider == "Local":
                    api_payload = {
                        "model": selected_model,
                        "messages": messages,
                        "stream": True
                    }
                    
                    response = requests.post(
                        f"{get_ollama_host()}/api/chat",
                        json=api_payload,
                        headers={"Content-Type": "application/json"},
                        stream=True,
                        timeout=300
                    )
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode("utf-8"))
                                if "message" in data and "content" in data["message"]:
                                    content_chunk = data["message"]["content"]
                                    if st.session_state.get("enable_reasoning", False):
                                        # Process thinking tags
                                        if "<think>" in content_chunk or "</think>" in content_chunk:
                                            # Split content by thinking tags
                                            parts = re.split(r'(<think>|</think>)', content_chunk)
                                            for part in parts:
                                                if part == "<think>":
                                                    st.session_state.in_thinking_block = True
                                                    show_reasoning_window()
                                                    continue
                                                elif part == "</think>":
                                                    st.session_state.in_thinking_block = False
                                                    hide_reasoning_window()
                                                    continue
                                                
                                                if st.session_state.in_thinking_block:
                                                    st.session_state.thinking_content += part
                                                    update_reasoning_window(st.session_state.thinking_content)
                                                else:
                                                    accumulated_response += part
                                                    response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                        else:
                                            if st.session_state.in_thinking_block:
                                                st.session_state.thinking_content += content_chunk
                                                update_reasoning_window(st.session_state.thinking_content)
                                            else:
                                                accumulated_response += content_chunk
                                                response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                    else:
                                        accumulated_response += content_chunk
                                        response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                
                            except json.JSONDecodeError:
                                continue
                
                elif st.session_state.api_provider == "Cloud":
                    response = call_openrouter_api(
                        messages,
                        selected_model,
                        st.session_state.api_keys["openrouter"]
                    )
                    
                    for chunk in response.iter_lines():
                        if chunk:
                            # Skip keep-alive new lines
                            if chunk == b'':
                                continue
                            
                            # Handle event stream format
                            if chunk.startswith(b'data:'):
                                try:
                                    data = json.loads(chunk.decode("utf-8")[5:])
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        content_chunk = delta.get("content", "")
                                        accumulated_response += content_chunk
                                        response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                except json.JSONDecodeError:
                                    pass
                
                # After stream completes
                response_time = time.time() - start_time
                
                if st.session_state.in_thinking_block:
                    # If we ended in a thinking block, add content to main response
                    accumulated_response += st.session_state.thinking_content
                    st.session_state.in_thinking_block = False
                    hide_reasoning_window()
                
                response_msg = {
                    "role": "assistant",
                    "content": accumulated_response,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(response_msg)
                
                if not accumulated_response.strip():
                    fallback_message = "The model did not generate any response. Please try again."
                    response_placeholder.markdown(fallback_message, unsafe_allow_html=True)
                
                if vision_model and uploaded_files:
                    current_key = st.session_state.file_uploader_key
                    key_number = int(current_key.split('_')[1]) + 1
                    st.session_state.file_uploader_key = f"uploader_{key_number}"
                
                hide_reasoning_window()
                st.rerun()
                
            except requests.RequestException as e:
                error_message = f"Error communicating with API: {e}"
                if "401" in str(e):
                    error_message += "\n\n⚠️ Invalid API Key - Please check your credentials"
                elif "429" in str(e):
                    error_message += "\n\n⚠️ Rate limit exceeded - Try again later"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                response_placeholder.markdown(error_message, unsafe_allow_html=True)
                hide_reasoning_window()

    elif prompt and not selected_model:
        st.error("Please select a model before submitting a prompt.")

    # Memory cleanup reminder
    if len(st.session_state.messages) > 150:
        with st.sidebar:
            st.info("💡 Consider clearing chat for optimal performance")

if __name__ == "__main__":
    main()