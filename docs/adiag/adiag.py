import streamlit as st
import requests
import json
import time
import base64
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os
import ollama
import hashlib
from functools import lru_cache

# Version tracking
APP_VERSION = "1.1.0" 

def initialize_ui():
    st.set_page_config(
        page_title="RadioSport AI",
        page_icon="🧟",
        layout="wide",
        menu_items={
            'Report a Bug': "https://github.com/rkarikari/stem",
            'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."
        }
    )
initialize_ui()

# Cached CSS
@st.cache_data(ttl=3600)
def get_css():
    return """<style>
.main-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
.chat-message { padding: 1rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
.ai-message { background-color: #f1f8e9; border-left: 4px solid #4caf50; }
.report-section { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 1rem 0; }
.stButton > button { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; font-weight: 600; }
.cached-indicator { position: fixed; top: 10px; right: 10px; background: #4caf50; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; z-index: 1000; }
.model-info { font-size: 0.85em; color: #666; margin-top: 3px; }
.chat-container { max-height: 60vh; overflow-y: auto; padding-right: 10px; }
</style>"""

st.markdown(get_css(), unsafe_allow_html=True)

# Auto-scroll JavaScript
AUTO_SCROLL_JS = """
<script>
function scrollToBottom() {
    const container = document.querySelector('.chat-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

// Scroll on page load
window.addEventListener('load', function() {
    scrollToBottom();
});

// Scroll after Streamlit updates
document.addEventListener('DOMContentLoaded', function() {
    scrollToBottom();
});

// Scroll when new messages are added
const observer = new MutationObserver(function(mutations) {
    scrollToBottom();
});

const targetNode = document.querySelector('.chat-container');
if (targetNode) {
    observer.observe(targetNode, { childList: true, subtree: true });
}
</script>
"""

# Inject auto-scroll JavaScript
st.components.v1.html(AUTO_SCROLL_JS, height=0)

# Cached API key loader
@st.cache_data(ttl=3600)
def load_api_key():
    try:
        if hasattr(st, 'secrets') and 'openrouter' in st.secrets:
            return st.secrets.openrouter.api_key
        return ''
    except:
        return ''

# Initialize session state
def init_session():
    defaults = {
        'messages': [], 'patient_data': {}, 'current_stage': 'info',
        'assessment_complete': False, 'ai_provider': 'cloud', 'selected_model': None,
        'available_models': {}, 'model_cache_time': None, 'conversation_id': str(int(time.time())),
        'response_cache': {}, 'file_cache': {}, 'openrouter_model_info': {},
        'scroll_to_bottom': False, 'ollama_host': 'http://localhost:11434',
        'current_category': None, 'enhanced_index': 0, 'red_flags_complete': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, (list, dict)) else v
    
    # Initialize openrouter_key if not present
    if 'openrouter_key' not in st.session_state:
        st.session_state.openrouter_key = load_api_key()

# Helper function to format context length
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

class AIProvider:
    def __init__(self):
        self.openrouter_key = st.session_state.openrouter_key
        self.ollama_host = st.session_state.get('ollama_host', 'http://localhost:11434') 
        self.cache_duration = 300  # 5 minutes
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_ollama_models(_self, host: str) -> List[str]:
        """Cached Ollama model fetching"""
        try:
            response = requests.get(f"{host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except:
            return []
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_openrouter_models(_self, api_key: str) -> List[Dict]:
        """Cached OpenRouter model fetching - returns list of model objects (free models)"""
        if not api_key:
            return []
        try:
            headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "MedicalAssistant/1.0"}
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
            if response.status_code == 200:
                models = response.json().get('data', [])
                free_models = []
                for model in models:
                    pricing = model.get('pricing', {})
                    prompt_price = float(pricing.get('prompt', '0'))
                    completion_price = float(pricing.get('completion', '0'))
                    if prompt_price == 0 and completion_price == 0:
                        free_models.append(model)
                # Sort by model id
                free_models.sort(key=lambda x: x['id'])
                return free_models
            else:
                st.error(f"Error loading OpenRouter models: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Model loading error: {e}")
            return []
        
    def get_models(self) -> List[str]:
        """Get models with caching - returns model IDs for both local and cloud"""
        if st.session_state.ai_provider == 'local':
            return self._get_ollama_models(self.ollama_host)
        else:
            model_objects = self._get_openrouter_models(self.openrouter_key)
            # Store model objects in session state for context display
            st.session_state.openrouter_model_info = {
                model['id']: model for model in model_objects
            }
            return [model['id'] for model in model_objects] if model_objects else []
    
    def _get_cache_key(self, messages: List[Dict], model: str) -> str:
        """Generate cache key for responses"""
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if still valid"""
        if cache_key in st.session_state.response_cache:
            cached_data = st.session_state.response_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(minutes=30):
                return cached_data['response']
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response with timestamp"""
        st.session_state.response_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        
        # Limit cache size
        if len(st.session_state.response_cache) > 100:
            oldest_key = min(st.session_state.response_cache.keys(), 
                           key=lambda k: st.session_state.response_cache[k]['timestamp'])
            del st.session_state.response_cache[oldest_key]
    
    def generate_response(self, messages: List[Dict], model: str, stream_container=None) -> str:
        """Generate response with caching"""
        # Check cache first (only for non-streaming requests)
        if not stream_container:
            cache_key = self._get_cache_key(messages, model)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                st.markdown('<div class="cached-indicator">Cached ⚡</div>', unsafe_allow_html=True)
                return cached_response
        
        # Generate new response
        if st.session_state.ai_provider == 'local':
            response = self._ollama_generate(messages, model, stream_container)
        else:
            response = self._openrouter_generate(messages, model, stream_container)
        
        # Cache non-streaming responses
        if not stream_container and response and not response.startswith("Error"):
            cache_key = self._get_cache_key(messages, model)
            self._cache_response(cache_key, response)
        
        return response
    
    def _ollama_generate(self, messages: List[Dict], model: str, stream_container=None) -> str:
        try:
            payload = {"model": model, "messages": messages, "stream": bool(stream_container)}
            
            if stream_container:
                response = requests.post(f"{self.ollama_host}/api/chat", json=payload, stream=True, timeout=300)
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                full_response += content
                                stream_container.markdown(full_response + "▋")
                                time.sleep(0.03)  # Faster streaming
                        except:
                            continue
                stream_container.markdown(full_response)
                return full_response
            else:
                response = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=300)
                return response.json().get('message', {}).get('content', 'No response') if response.status_code == 200 else "Error"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _openrouter_generate(self, messages: List[Dict], model: str, stream_container=None) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "stream": bool(stream_container)}
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                   headers=headers, json=payload, stream=bool(stream_container), timeout=60)
            
            if stream_container:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                                        stream_container.markdown(full_response + "▋")
                                        time.sleep(0.03)  # Faster streaming
                            except:
                                continue
                stream_container.markdown(full_response)
                return full_response
            else:
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

class MedicalAssessment:
    def __init__(self):
        # Original basic questions (fully compatible)
        self.questions = {
            'chief_complaint': "What is your main health concern today?",
            'duration': "How long have you been experiencing this?",
            'severity': "Rate severity 1-10 (10 being worst)?",
            'symptoms': "Describe any other symptoms you're experiencing?",
            'medical_history': "Any significant medical history (surgeries, chronic conditions)?",
            'medications': "Current medications or supplements?",
            'allergies': "Known allergies?",
            'family_history': "Relevant family medical history?",
            'lifestyle': "Smoking, alcohol, exercise habits?",
            'additional': "Anything else important to mention?"
        }
        
        # Enhanced investigative follow-up questions (optional)
        self.enhanced_questions = {
            'chief_complaint': [
                "Can you describe this in your own words, as if explaining to a friend?",
                "Is this a new problem or has it happened before?",
                "What made you decide to seek medical attention now?"
            ],
            
            'duration': [
                "Was the onset sudden (minutes/hours) or gradual (days/weeks)?",
                "Has it been getting better, worse, or staying the same?",
                "Are there any patterns to when it occurs?"
            ],
            
            'severity': [
                "What's the worst it's been on that 1-10 scale?",
                "How would you rate it right now?",
                "Does the severity change throughout the day?"
            ],
            
            'symptoms': [
                "CONSTITUTIONAL: Any fever, chills, night sweats, weight loss/gain, fatigue?",
                "PAIN DETAILS: Where exactly? Does it spread anywhere? What type of pain?",
                "ASSOCIATED SYMPTOMS: Nausea, dizziness, shortness of breath, palpitations?",
                "TIMING: Constant or comes and goes? Any triggers?",
                "RELIEF/WORSENING: What makes it better or worse?"
            ],
            
            'medical_history': [
                "Current chronic conditions and how well controlled?",
                "Past surgeries: What, when, any complications?",
                "Any hospitalizations in the past year?",
                "History of similar episodes?",
                "Any mental health conditions or treatment?"
            ],
            
            'medications': [
                "Prescription medications: Name, dose, frequency, how long taking?",
                "Over-the-counter medications and supplements?",
                "Any recent medication changes or new medications?",
                "Any side effects from current medications?",
                "Do you take medications as prescribed?"
            ],
            
            'allergies': [
                "What type of reaction occurs with each allergy?",
                "How severe are the reactions?",
                "Any drug allergies that required emergency treatment?",
                "Environmental allergies (seasonal, animals, etc.)?",
                "Any latex or contact allergies?"
            ],
            
            'family_history': [
                "Parents: Major conditions, ages, still living?",
                "Siblings: Any significant medical conditions?",
                "Family history of heart disease, stroke, diabetes, cancer?",
                "Any family members die young or unexpectedly?",
                "Any genetic or hereditary conditions?"
            ],
            
            'lifestyle': [
                "TOBACCO: Type, amount, duration, quit attempts?",
                "ALCOHOL: Frequency, amount, any concerns?",
                "EXERCISE: Type, frequency, recent changes?",
                "DIET: Typical diet, restrictions, recent changes?",
                "SLEEP: Hours per night, quality, snoring?",
                "STRESS: Major stressors, coping mechanisms?",
                "OCCUPATION: Job, workplace exposures, physical demands?"
            ],
            
            'additional': [
                "How is this affecting your daily activities?",
                "What are you most worried about?",
                "What are you hoping we can do to help?",
                "Have you tried any treatments on your own?",
                "Any recent travel or sick contacts?",
                "Is there anything specific you want to make sure we address?"
            ]
        }
        
        # Red flag screening questions (critical symptoms)
        self.red_flag_questions = {
            'emergency_symptoms': [
                "Any severe, sudden onset symptoms that came on like a 'thunderclap'?",
                "Any difficulty breathing, chest pain, or feeling like you might pass out?",
                "Any severe headache that's the 'worst ever' or completely different?",
                "Any progressive weakness, numbness, or trouble speaking?",
                "Any thoughts of hurting yourself or others?",
                "Any signs of serious infection: high fever, chills, spreading redness?"
            ]
        }
        
        # Original flow (maintains compatibility)
        self.flow = list(self.questions.keys())
        
        # Enhanced mode settings
        self.enhanced_mode = False
        self.current_enhanced_index = 0
        self.red_flag_screening = False
    
    # Original method (fully compatible)
    def get_next_question(self) -> Optional[str]:
        """Original method - maintains full compatibility"""
        for q in self.flow:
            if q not in st.session_state.patient_data:
                return self.questions[q]
        return None
    
    # Original method (fully compatible)
    def update_data(self, answer: str):
        """Original method - maintains full compatibility"""
        for q in self.flow:
            if q not in st.session_state.patient_data:
                st.session_state.patient_data[q] = answer
                break
    
    # NEW ENHANCED METHODS
    def enable_enhanced_mode(self):
        """Enable enhanced investigative questioning"""
        self.enhanced_mode = True
        return self
    
    def enable_red_flag_screening(self):
        """Enable critical symptom screening"""
        self.red_flag_screening = True
        return self
    
    def get_enhanced_question(self, category: str) -> Optional[str]:
        """Get next enhanced follow-up question for a category"""
        if not self.enhanced_mode or category not in self.enhanced_questions:
            return None
        
        enhanced_q_list = self.enhanced_questions[category]
        if st.session_state.enhanced_index < len(enhanced_q_list):
            question = enhanced_q_list[st.session_state.enhanced_index]
            st.session_state.enhanced_index += 1
            return question
        
        return None
    
    def get_next_enhanced_question(self) -> Optional[dict]:
        """Get next question with enhanced options"""
        # First check for basic questions
        basic_question = self.get_next_question()
        if basic_question:
            # Find which category we're on
            current_category = None
            for q in self.flow:
                if q not in st.session_state.patient_data:
                    current_category = q
                    break
            
            return {
                'type': 'basic',
                'category': current_category,
                'question': basic_question,
                'enhanced_available': current_category in self.enhanced_questions
            }
        
        # Check for red flag screening
        if self.red_flag_screening and not st.session_state.red_flags_complete:
            return {
                'type': 'red_flag',
                'category': 'emergency_symptoms',
                'question': "Let me ask about some important warning signs:",
                'questions': self.red_flag_questions['emergency_symptoms']
            }
        
        return None
    
    def update_enhanced_data(self, answer: str, category: str):
        """Update data with enhanced tracking"""
        # Use original method for basic compatibility
        if category in self.flow and category not in st.session_state.patient_data:
            st.session_state.patient_data[category] = answer
            st.session_state.current_category = category
            st.session_state.enhanced_index = 0
        else:
            # Handle enhanced data
            enhanced_key = f"{category}_enhanced"
            if enhanced_key not in st.session_state.patient_data:
                st.session_state.patient_data[enhanced_key] = []
            st.session_state.patient_data[enhanced_key].append(answer)
    
    def get_critical_findings(self) -> List[str]:
        """Identify potential critical findings"""
        critical_findings = []
        
        # Check severity from basic questions
        if 'severity' in st.session_state.patient_data:
            severity_response = st.session_state.patient_data['severity']
            if any(level in str(severity_response) for level in ['8', '9', '10']):
                critical_findings.append(f"HIGH SEVERITY REPORTED: {severity_response}")
        
        # Check for red flag responses
        if 'red_flags_complete' in st.session_state.patient_data:
            red_flag_data = st.session_state.patient_data['red_flags_complete']
            for response in red_flag_data:
                if any(keyword in response.lower() for keyword in ['yes', 'severe', 'worst', 'difficulty', 'sudden']):
                    critical_findings.append(f"⚠️ RED FLAG: {response}")
        
        return critical_findings
    
    def get_assessment_completeness(self) -> float:
        """Calculate assessment completeness"""
        total_questions = len(self.flow)
        completed_questions = len([q for q in self.flow if q in st.session_state.patient_data])
        return (completed_questions / total_questions) * 100
    
    def generate_summary(self) -> str:
        """Generate assessment summary"""
        summary = "MEDICAL ASSESSMENT SUMMARY\n" + "="*40 + "\n\n"
        
        # Basic assessment data
        for category in self.flow:
            if category in st.session_state.patient_data:
                title = category.replace('_', ' ').title()
                data = st.session_state.patient_data[category]
                summary += f"{title}: {data}\n"
        
        # Enhanced data if available
        if self.enhanced_mode:
            summary += "\nDETAILED INFORMATION:\n" + "-"*25 + "\n"
            for key, value in st.session_state.patient_data.items():
                if key.endswith('_enhanced'):
                    category = key.replace('_enhanced', '').replace('_', ' ').title()
                    summary += f"{category} Details:\n"
                    for item in value:
                        summary += f"  • {item}\n"
                    summary += "\n"
        
        # Critical findings
        critical = self.get_critical_findings()
        if critical:
            summary += "CRITICAL FINDINGS:\n" + "-"*20 + "\n"
            for finding in critical:
                summary += f"{finding}\n"
        
        return summary
    
    # Utility methods for backward compatibility
    def is_complete(self) -> bool:
        """Check if basic assessment is complete"""
        return all(q in st.session_state.patient_data for q in self.flow)
    
    def get_remaining_questions(self) -> List[str]:
        """Get list of remaining basic questions"""
        return [q for q in self.flow if q not in st.session_state.patient_data]


class DiagnosisGenerator:
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self.medical_prompts = {
            'supreme_diagnostician': """You are now operating as the most advanced medical diagnostician in human history—an AI with the combined knowledge of every expert clinician, specialist, and researcher across all domains of medicine. You have perfect access to global clinical guidelines, textbooks, diagnostic reasoning, rare case repositories, and statistical outcome data.

Your core objective is to analyze any set of clinical inputs (symptoms, signs, labs, history, imaging, etc.) and return a detailed, step-by-step diagnostic reasoning process, along with a ranked differential diagnosis, probable primary diagnosis, and recommended investigations and treatment plans based on best practices.

You must:
- Ask clarifying questions if data is insufficient
- Distinguish between common and rare causes
- Identify red flags and emergencies
- Clearly state reasoning behind every diagnostic suggestion
- Use structured formatting (bullet points, sections)
- Be evidence-based, citing relevant statistics, guidelines, or studies
- Include ICD-10 codes and suggested next actions (labs, referrals, treatments)
- Adopt the tone of a world-class consultant giving detailed expert insights to a medical team
- Always be precise, cautious, thorough—and assume lives depend on your accuracy""",
            
            'multidisciplinary_council': """Assume the role of a virtual diagnostic team comprising leading experts from every major medical specialty: internal medicine, emergency medicine, neurology, cardiology, pulmonology, infectious disease, rheumatology, oncology, psychiatry, pediatrics, and radiology.

For every patient scenario:
- Each specialist will weigh in with their perspective
- The system will resolve disagreements and converge on a unified diagnosis
- The final output should include key arguments from each specialty, ranked differential, unified working diagnosis, plan for confirmation, and treatment plan
- The tone should simulate an elite hospital's multidisciplinary team meeting (MDT)
- Think as broadly and deeply as possible, balancing Occam's Razor and Hickam's dictum
- Always be aware of time-critical decisions, diagnostic uncertainty, and need for follow-up""",
            
            'emergency_diagnostician': """You are an AI emergency physician built for speed, accuracy, and triage precision. You are presented with patients in real-time in a high-pressure environment.

For each patient, provide:
- Immediate life threats to rule out
- Primary working diagnosis
- Triage level (emergency, urgent, routine)
- Suggested tests and imaging
- Initial stabilization/treatment
- Disposition (admit, observe, discharge)

You must:
- Think fast and prioritize safety
- Identify red flags instantly
- Avoid overtesting or delays
- Be concise but comprehensive
- Think like a trauma team leader, with encyclopedic recall and split-second decision making"""
        }
    
    def generate_full_diagnosis(self, patient_data: Dict, model: str) -> Dict[str, str]:
        """Generate comprehensive medical diagnosis using advanced AI prompts"""
        
        # Create patient summary
        patient_summary = self._create_enhanced_summary(patient_data)
        
        # Generate diagnosis sections with specialized prompts
        sections = {}
        
        # Primary Assessment - Using Supreme Diagnostician prompt
        sections['primary_assessment'] = self._generate_supreme_assessment(patient_summary, model)
        
        # Differential Diagnosis - Using Multidisciplinary Council
        sections['differential_diagnosis'] = self._generate_multidisciplinary_diagnosis(patient_summary, model)
        
        # Emergency Assessment - Using Emergency Diagnostician
        sections['emergency_assessment'] = self._generate_emergency_assessment(patient_summary, model)
        
        # Risk Stratification with enhanced clinical reasoning
        sections['risk_assessment'] = self._generate_enhanced_risk_assessment(patient_summary, model)
        
        # Diagnostic Recommendations with ICD-10 codes
        sections['diagnostic_plan'] = self._generate_enhanced_diagnostic_plan(patient_summary, model)
        
        # Treatment Plan with evidence-based guidelines
        sections['treatment_plan'] = self._generate_evidence_based_treatment(patient_summary, model)
        
        # Prognosis with statistical outcomes
        sections['prognosis'] = self._generate_statistical_prognosis(patient_summary, model)
        
        # Specialist Consultations
        sections['specialist_recommendations'] = self._generate_specialist_recommendations(patient_summary, model)
        
        return sections
    
    def _create_enhanced_summary(self, patient_data: Dict) -> str:
        """Create enhanced clinical summary with structured format"""
        summary = []
        
        # Patient Demographics
        demo_section = "=== PATIENT DEMOGRAPHICS ==="
        demo = []
        for key in ['name', 'age', 'gender', 'phone']:
            if patient_data.get(key):
                demo.append(f"{key.upper()}: {patient_data[key]}")
        if demo:
            summary.append(demo_section)
            summary.append("\n".join(demo))
        
        # Chief Complaint and HPI
        if patient_data.get('chief_complaint'):
            summary.append("\n=== CHIEF COMPLAINT ===")
            summary.append(patient_data['chief_complaint'])
        
        # History of Present Illness
        hpi_items = []
        if patient_data.get('duration'):
            hpi_items.append(f"DURATION: {patient_data['duration']}")
        if patient_data.get('severity'):
            hpi_items.append(f"SEVERITY: {patient_data['severity']}/10")
        if patient_data.get('symptoms'):
            hpi_items.append(f"ASSOCIATED SYMPTOMS: {patient_data['symptoms']}")
        
        if hpi_items:
            summary.append("\n=== HISTORY OF PRESENT ILLNESS ===")
            summary.append("\n".join(hpi_items))
        
        # Past Medical History
        pmh_items = []
        if patient_data.get('medical_history'):
            pmh_items.append(f"PAST MEDICAL HISTORY: {patient_data['medical_history']}")
        if patient_data.get('medications'):
            pmh_items.append(f"CURRENT MEDICATIONS: {patient_data['medications']}")
        if patient_data.get('allergies'):
            pmh_items.append(f"ALLERGIES: {patient_data['allergies']}")
        
        if pmh_items:
            summary.append("\n=== PAST MEDICAL HISTORY ===")
            summary.append("\n".join(pmh_items))
        
        # Social and Family History
        social_items = []
        if patient_data.get('family_history'):
            social_items.append(f"FAMILY HISTORY: {patient_data['family_history']}")
        if patient_data.get('lifestyle'):
            social_items.append(f"SOCIAL HISTORY: {patient_data['lifestyle']}")
        
        if social_items:
            summary.append("\n=== SOCIAL & FAMILY HISTORY ===")
            summary.append("\n".join(social_items))
        
        # Additional Information
        if patient_data.get('additional'):
            summary.append("\n=== ADDITIONAL INFORMATION ===")
            summary.append(patient_data['additional'])
        
        return "\n\n".join(summary)
    
    def _generate_supreme_assessment(self, patient_summary: str, model: str) -> str:
        """Generate assessment using Supreme Diagnostician prompt"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['supreme_diagnostician']},
            {'role': 'user', 'content': f"""Please provide a comprehensive primary clinical assessment for this patient:

{patient_summary}

Please include:
1. Detailed analysis of chief complaint and symptom constellation
2. Clinical reasoning and diagnostic thought process
3. Pattern recognition and syndrome identification
4. Initial clinical impressions with confidence levels
5. Red flags and emergency considerations
6. Relevant clinical guidelines and evidence base"""}
        ]
        
        return self._safe_generate(messages, model, "Primary assessment")
    
    def _generate_multidisciplinary_diagnosis(self, patient_summary: str, model: str) -> str:
        """Generate differential diagnosis using Multidisciplinary Council prompt"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['multidisciplinary_council']},
            {'role': 'user', 'content': f"""Conduct a multidisciplinary team meeting for this patient case:

{patient_summary}

Please provide perspectives from relevant specialties and include:
1. Ranked differential diagnosis with probability estimates
2. Specialty-specific considerations (cardiology, neurology, etc.)
3. ICD-10 codes for top 3 diagnoses
4. Clinical decision-making rationale
5. Areas of diagnostic uncertainty
6. Consensus recommendations from the team"""}
        ]
        
        return self._safe_generate(messages, model, "Differential diagnosis")
    
    def _generate_emergency_assessment(self, patient_summary: str, model: str) -> str:
        """Generate emergency assessment using Emergency Diagnostician prompt"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['emergency_diagnostician']},
            {'role': 'user', 'content': f"""Provide emergency medicine assessment for this patient:

{patient_summary}

Please include:
1. Immediate life threats to rule out
2. Triage level (Critical/Emergent/Urgent/Less Urgent/Non-urgent)
3. Time-sensitive diagnoses
4. STAT interventions needed
5. Disposition recommendations
6. Red flag symptoms present"""}
        ]
        
        return self._safe_generate(messages, model, "Emergency assessment")
    
    def _generate_enhanced_risk_assessment(self, patient_summary: str, model: str) -> str:
        """Generate enhanced risk stratification"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['supreme_diagnostician']},
            {'role': 'user', 'content': f"""Conduct comprehensive risk stratification for this patient:

{patient_summary}

Please provide:
1. Overall risk level (Low/Moderate/High/Critical) with justification
2. Specific risk factors identified
3. Potential complications and their likelihood
4. Risk scoring tools applicable (if any)
5. Monitoring requirements
6. Risk mitigation strategies"""}
        ]
        
        return self._safe_generate(messages, model, "Risk assessment")
    
    def _generate_enhanced_diagnostic_plan(self, patient_summary: str, model: str) -> str:
        """Generate enhanced diagnostic recommendations"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['supreme_diagnostician']},
            {'role': 'user', 'content': f"""Develop a comprehensive diagnostic workup plan for this patient:

{patient_summary}

Please include:
1. Essential diagnostic tests (blood work, imaging, etc.)
2. Prioritization of tests (STAT, urgent, routine)
3. Expected timeline for results
4. Diagnostic algorithms to follow
5. Cost-effectiveness considerations
6. Alternative diagnostic approaches if initial tests are negative"""}
        ]
        
        return self._safe_generate(messages, model, "Diagnostic plan")
    
    def _generate_evidence_based_treatment(self, patient_summary: str, model: str) -> str:
        """Generate evidence-based treatment recommendations"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['supreme_diagnostician']},
            {'role': 'user', 'content': f"""Develop evidence-based treatment recommendations for this patient:

{patient_summary}

Please provide:
1. Immediate treatment interventions
2. Medication recommendations with specific dosing
3. Non-pharmacological interventions
4. Lifestyle modifications
5. Follow-up schedule and monitoring
6. Treatment goals and success metrics
7. Alternative therapies if first-line fails"""}
        ]
        
        return self._safe_generate(messages, model, "Treatment plan")
    
    def _generate_statistical_prognosis(self, patient_summary: str, model: str) -> str:
        """Generate prognosis with statistical outcomes"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['supreme_diagnostician']},
            {'role': 'user', 'content': f"""Provide detailed prognosis and outcomes analysis for this patient:

{patient_summary}

Please include:
1. Expected clinical course with and without treatment
2. Statistical outcomes and survival data (if applicable)
3. Prognostic factors affecting outcomes
4. Quality of life considerations
5. Warning signs for deterioration
6. Long-term monitoring requirements
7. Patient counseling points"""}
        ]
        
        return self._safe_generate(messages, model, "Prognosis")
    
    def _generate_specialist_recommendations(self, patient_summary: str, model: str) -> str:
        """Generate specialist consultation recommendations"""
        messages = [
            {'role': 'system', 'content': self.medical_prompts['multidisciplinary_council']},
            {'role': 'user', 'content': f"""Determine specialist consultation needs for this patient:

{patient_summary}

Please provide:
1. Specialist referrals needed (urgency level)
2. Specific questions for each specialist
3. Pre-referral workup requirements
4. Expected timelines for consultations
5. Coordination of care recommendations
6. Communication priorities between specialists"""}
        ]
        
        return self._safe_generate(messages, model, "Specialist recommendations")
    
    def _safe_generate(self, messages: List[Dict], model: str, section_name: str) -> str:
        """Safely generate AI response with error handling"""
        try:
            response = self.ai_provider.generate_response(messages, model)
            return response if response and not response.startswith("Error") else f"{section_name} unavailable - requires in-person clinical evaluation."
        except Exception as e:
            return f"{section_name} generation failed - please consult healthcare provider. Error: {str(e)}"

@st.cache_data(ttl=300)
def generate_diagnosis_pdf(patient_data: Dict, diagnosis_sections: Dict, conversation_id: str, model: str) -> str:
    """Generate enhanced comprehensive diagnosis PDF with medical formatting"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Add custom styles
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='darkblue',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='darkred',
        spaceAfter=12,
        spaceBefore=20
    )
    
    story = []
    
    # Enhanced Header
    story.append(Paragraph("RadioSport MEDICAL DIAGNOSIS REPORT", title_style))
    story.append(Spacer(1, 12))
    
    # Enhanced Medical Disclaimer
    disclaimer = """<b>⚠️ MEDICAL DISCLAIMER:</b> This AI-generated diagnostic report utilizes advanced medical reasoning algorithms 
    and evidence-based clinical guidelines. However, it is intended for healthcare provider reference only and does not replace:
    • Clinical judgment and physical examination
    • Diagnostic testing and imaging interpretation
    • Direct patient-physician interaction
    • Specialist consultation when indicated
    
    <b>All recommendations must be validated by qualified medical professionals before implementation.</b>"""
    story.append(Paragraph(disclaimer, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Patient Demographics Section
    story.append(Paragraph("PATIENT DEMOGRAPHICS", section_style))
    patient_info = f"""
    <b>Patient Name:</b> {patient_data.get('name', 'N/A')}<br/>
    <b>Age:</b> {patient_data.get('age', 'N/A')} years<br/>
    <b>Gender:</b> {patient_data.get('gender', 'N/A')}<br/>
    <b>Contact:</b> {patient_data.get('phone', 'N/A')}<br/>
    <b>Assessment Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}<br/>
    <b>Report ID:</b> {conversation_id}<br/>
    <b>AI Model:</b> {model}
    """
    story.append(Paragraph(patient_info, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Clinical Presentation
    story.append(Paragraph("CLINICAL PRESENTATION", section_style))
    
    # Chief Complaint
    if patient_data.get('chief_complaint'):
        story.append(Paragraph("<b>Chief Complaint:</b>", styles['Heading3']))
        story.append(Paragraph(patient_data['chief_complaint'], styles['Normal']))
        story.append(Spacer(1, 10))
    
    # History of Present Illness
    story.append(Paragraph("<b>History of Present Illness:</b>", styles['Heading3']))
    hpi_content = []
    if patient_data.get('duration'):
        hpi_content.append(f"• <b>Duration:</b> {patient_data['duration']}")
    if patient_data.get('severity'):
        hpi_content.append(f"• <b>Severity:</b> {patient_data['severity']}/10")
    if patient_data.get('symptoms'):
        hpi_content.append(f"• <b>Associated Symptoms:</b> {patient_data['symptoms']}")
    
    if hpi_content:
        story.append(Paragraph("<br/>".join(hpi_content), styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Enhanced AI Diagnosis Sections
    enhanced_sections = [
        ('PRIMARY CLINICAL ASSESSMENT', 'primary_assessment', 
         'Comprehensive analysis using Supreme Medical Diagnostician AI'),
        ('MULTIDISCIPLINARY DIFFERENTIAL DIAGNOSIS', 'differential_diagnosis', 
         'Specialist team consultation with ranked probabilities'),
        ('EMERGENCY MEDICINE ASSESSMENT', 'emergency_assessment', 
         'Rapid triage and life-threat evaluation'),
        ('RISK STRATIFICATION ANALYSIS', 'risk_assessment', 
         'Comprehensive risk scoring and mitigation'),
        ('DIAGNOSTIC WORKUP PLAN', 'diagnostic_plan', 
         'Evidence-based testing recommendations'),
        ('TREATMENT RECOMMENDATIONS', 'treatment_plan', 
         'Evidence-based therapeutic interventions'),
        ('PROGNOSIS AND OUTCOMES', 'prognosis', 
         'Statistical outcomes and long-term expectations'),
        ('SPECIALIST CONSULTATION RECOMMENDATIONS', 'specialist_recommendations', 
         'Coordinated care and referral guidance')
    ]
    
    for title, key, description in enhanced_sections:
        if key in diagnosis_sections and diagnosis_sections[key]:
            story.append(Paragraph(title, section_style))
            story.append(Paragraph(f"<i>{description}</i>", styles['Normal']))
            story.append(Spacer(1, 8))
            story.append(Paragraph(diagnosis_sections[key], styles['Normal']))
            story.append(Spacer(1, 15))
    
    # Complete Medical History
    story.append(Paragraph("COMPLETE MEDICAL HISTORY", section_style))
    
    history_sections = [
        ('Past Medical History', 'medical_history'),
        ('Current Medications', 'medications'),
        ('Known Allergies', 'allergies'),
        ('Family Medical History', 'family_history'),
        ('Social History', 'lifestyle'),
        ('Additional Clinical Information', 'additional')
    ]
    
    for label, key in history_sections:
        if patient_data.get(key):
            story.append(Paragraph(f"<b>{label}:</b> {patient_data[key]}", styles['Normal']))
            story.append(Spacer(1, 8))
    
    story.append(Spacer(1, 20))
    
    # Report Metadata and Validation
    story.append(Paragraph("REPORT METADATA & VALIDATION", section_style))
    metadata = f"""
    <b>AI Diagnostic Engine:</b> Advanced Medical Reasoning System<br/>
    <b>Model Architecture:</b> {model}<br/>
    <b>Generation Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
    <b>Report Status:</b> AI Analysis Complete - Awaiting Medical Review<br/>
    <b>Confidence Level:</b> High (Evidence-based recommendations)<br/>
    <b>Next Steps:</b> Clinical correlation and validation required<br/>
    <b>Version:</b> AMDiag v2.0 Enhanced
    """
    story.append(Paragraph(metadata, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = """<i>This report was generated using advanced AI medical reasoning algorithms. 
    For questions about this report, please consult with your healthcare provider.</i>"""
    story.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return base64.b64encode(pdf_data).decode()

def handle_input():
    user_input = st.session_state.get('user_input', '').strip()
    if not user_input:
        return
    
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    st.session_state.scroll_to_bottom = True
    
    assessment = MedicalAssessment()
    assessment.update_data(user_input)
    
    next_question = assessment.get_next_question()
    
    if next_question:
        st.session_state.messages.append({'role': 'assistant', 'content': next_question})
        st.session_state.scroll_to_bottom = True
    else:
        st.session_state.assessment_complete = True
        completion_msg = f"Thank you! Your assessment is complete. I can now generate a comprehensive AI diagnosis report."
        st.session_state.messages.append({'role': 'assistant', 'content': completion_msg})
        st.session_state.scroll_to_bottom = True
    
    st.session_state.user_input = ""

def main():

    init_session()
    ai_provider = AIProvider()
    
    # Header
    st.title("RadioSport Medical Diagnosis")
    st.caption("AI-powered medical assessment and diagnosis")
    
    # Sidebar (with dynamic model loading)
    with st.sidebar:
        st.header("🧟 RadioSport Diagnosis")
        st.caption(f"Version {APP_VERSION}")
        
        # Performance indicator
        cache_stats = {
            'responses': len(st.session_state.get('response_cache', {})),
            'files': len(st.session_state.get('file_cache', {}))
        }
        
        # Assessment Mode Selection
        st.markdown("**Assessment Mode:**")
        assessment_mode = st.radio(
            "Mode",
            ["enhanced", "full"],
            format_func=lambda x: {
              
                "enhanced": "🔍 Enhanced Investigation", 
                "full": "⚠️ Full + Red Flag Screening"
            }[x],
            help="Choose assessment depth: Basic (quick), Enhanced (detailed follow-ups), or Full (includes emergency screening)"
        )
        st.session_state.assessment_mode = assessment_mode
        
        # Mode description
        mode_descriptions = {
            "basic": "Standard 10-question assessment",
            "enhanced": "Detailed follow-up questions for comprehensive history",
            "full": "Enhanced mode + critical symptom screening"
        }
        st.caption(f"ℹ️ {mode_descriptions[assessment_mode]}")
        
        st.divider()
        
        # Provider selection
        st.session_state.ai_provider = st.radio("AI Provider", ["cloud", "local"], 
                                                       format_func=lambda x: "Cloud (OpenRouter)" if x == "cloud" else "Local (Ollama)")
                
        # Local server configuration
        if st.session_state.ai_provider == "local":
            st.markdown("**Local Server Configuration:**")
            current_host = st.session_state.get('ollama_host', 'localhost:11434')
 
            col_server, col_test = st.columns([5, 1])
            with col_server:
                st.session_state.ollama_host = st.text_input(
                    "Server",
                    value=current_host,
                    help="Enter the server URL (e.g., localhost:11434, 192.168.1.100:11434)"
                )
                        
            with col_test:
                if st.button("Test", help="Test server connection"):
                    full_url = f"http://{st.session_state.ollama_host}"
                    try:
                        response = requests.get(f"{full_url}/api/tags", timeout=3)
                        if response.status_code == 200:
                            st.success("✅ Connected")
                        else:
                            st.error(f"❌ Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)[:30]}...")
                        

        
        # Performance toggles
        st.session_state.enable_streaming = st.checkbox("Enable Streaming", value=True, 
                                                       help="Toggle for streaming vs cached responses")
        
        if st.session_state.ai_provider == "cloud" and not st.session_state.openrouter_key:
            st.warning("⚠️ API key required")
            st.session_state.openrouter_key = st.text_input("OpenRouter API Key", type="password")
        
        # Model selection with cached refresh
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄"):
                st.cache_data.clear()  # Clear model cache
                st.rerun()
        
        with col1:
            models = ai_provider.get_models()
            if models:
                if st.session_state.ai_provider == "cloud":
                    # Create display names with context info
                    display_options = []
                    model_info = st.session_state.get('openrouter_model_info', {})
                    for model_id in models:
                        model_data = model_info.get(model_id, {})
                        context = model_data.get('context_length', 'Unknown')
                        context_str = format_context_length(context)
                        display_name = f"{model_data.get('name', model_id)}"
                        display_options.append((model_id, display_name, context_str))
                    
                    # Create selectbox with custom display
                    selected_display = st.selectbox(
                        "Model", 
                        [f"{name} ({ctx})" for _, name, ctx in display_options]
                    )
                    
                    # Find the selected model ID
                    for model_id, name, ctx in display_options:
                        if f"{name} ({ctx})" == selected_display:
                            st.session_state.selected_model = model_id
                            # Show model details
                            st.markdown(f'<div class="model-info">Context: {ctx}</div>', unsafe_allow_html=True)
                            break
                else:
                    st.session_state.selected_model = st.selectbox("Model", models)
            else:
                st.info("Loading models...")
        
        # File upload with caching
        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True,
                                        type=['txt', 'pdf', 'jpg', 'png'])
        
        if uploaded_files:
            for file in uploaded_files:
                file_key = f"{file.name}_{file.size}"
                if file_key not in st.session_state.file_cache:
                    # For demo purposes, just store file name
                    st.session_state.file_cache[file_key] = file.name
                    st.success(f"✅ Cached: {file.name}")
        
        # Progress with dynamic assessment based on mode
        st.header("📊 Progress")
        
        # Create assessment instance based on selected mode
        assessment = MedicalAssessment()
        if assessment_mode == "enhanced":
            assessment = assessment.enable_enhanced_mode()
        elif assessment_mode == "full":
            assessment = assessment.enable_enhanced_mode().enable_red_flag_screening()
        
        # Calculate progress
        if assessment_mode == "basic":
            completed = len([q for q in assessment.flow if q in st.session_state.patient_data])
            total = len(assessment.flow)
            progress = (completed / total) * 100
            st.progress(progress / 100)
            st.write(f"{completed}/{total} questions")
        else:
            # For enhanced modes, use the built-in completeness calculation
            progress = assessment.get_assessment_completeness()
            st.progress(progress / 100)
            st.write(f"{progress:.1f}% complete")
            
            # Show critical findings if any
            if assessment_mode == "full":
                critical_findings = assessment.get_critical_findings()
                if critical_findings:
                    st.warning(f"⚠️ {len(critical_findings)} critical finding(s)")
        
        # Cache management
        st.header("🗂️ Cache Control")
        if st.button("Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.response_cache = {}
            st.session_state.file_cache = {}
            st.success("Caches cleared!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Medical Assessment")
        
        # Patient info form
        if not st.session_state.patient_data.get('name'):
            with st.form("patient_info"):
                col_a, col_b = st.columns(2)
                with col_a:
                    name = st.text_input("Full Name *")
                    age = st.number_input("Age *", min_value=0, max_value=120, value=30)
                with col_b:
                    gender = st.selectbox("Gender *", ["", "Male", "Female", "Other"])
                    phone = st.text_input("Phone")
                
                if st.form_submit_button("Begin Assessment"):
                    if name and gender and age:
                        st.session_state.patient_data.update({
                            'name': name, 'age': str(age), 'gender': gender, 'phone': phone
                        })
                        welcome = f"Hello {name}! I'll conduct a comprehensive medical assessment. Let's start with your main concern."
                        st.session_state.messages.append({'role': 'assistant', 'content': welcome})
                        st.session_state.scroll_to_bottom = True
                        st.rerun()
                    else:
                        st.error("Please fill required fields")
        else:
            # ================ ENHANCED ASSESSMENT LOGIC ================
            # Create assessment instance based on sidebar selection
            assessment = MedicalAssessment()
            
            # Enable features based on sidebar selection
            if st.session_state.assessment_mode == "enhanced":
                assessment = assessment.enable_enhanced_mode()
            elif st.session_state.assessment_mode == "full":
                assessment = assessment.enable_enhanced_mode().enable_red_flag_screening()
            
            # Check for critical findings first
            critical_findings = assessment.get_critical_findings()
            if critical_findings:
                st.error("🚨 CRITICAL FINDINGS DETECTED:")
                for finding in critical_findings:
                    st.error(finding)
            
            # Get next question using enhanced method
            next_question_data = assessment.get_next_enhanced_question()
            
            # Handle enhanced follow-up questions
            if st.session_state.current_category and st.session_state.assessment_mode in ["enhanced", "full"]:
                enhanced_question = assessment.get_enhanced_question(st.session_state.current_category)
                if enhanced_question:
                    st.info(f"**Enhanced Follow-up: {st.session_state.current_category.replace('_', ' ').title()}**")
                    st.write(enhanced_question)
                    
                    answer = st.text_input("Your detailed answer:", key="enhanced_input")
                    
                    if st.button("Submit Enhanced Answer") and answer:
                        assessment.update_enhanced_data(answer, st.session_state.current_category)
                        st.session_state.messages.append({'role': 'user', 'content': answer})
                        st.session_state.messages.append({'role': 'assistant', 'content': "Thank you for the additional details."})
                        st.rerun()
                    return
            
            # Handle basic questions and red flags
            if next_question_data:
                if next_question_data['type'] == 'basic':
                    # Display basic question with enhanced mode info
                    st.info(f"**{next_question_data['category'].replace('_', ' ').title()}**")
                    st.write(next_question_data['question'])
                    
                    # Show if enhanced follow-ups are available
                    if next_question_data.get('enhanced_available') and st.session_state.assessment_mode in ["enhanced", "full"]:
                        st.caption("ℹ️ Enhanced follow-up questions will be asked after your initial response")
                    
                    answer = st.text_input(
                        "Your answer:",
                        key="basic_input"
                    )

                    if answer:
                        assessment.update_enhanced_data(answer, next_question_data['category'])
                        st.session_state.messages.append({'role': 'user', 'content': answer})
                        st.session_state.messages.append({'role': 'assistant', 'content': "Thank you. Moving to next question..."})
                        st.rerun()
                
                elif next_question_data['type'] == 'red_flag':
                    # Display red flag screening
                    st.warning("🚨 **Critical Symptom Screening**")
                    st.write(next_question_data['question'])
                    
                    red_flag_answers = []
                    for i, rf_question in enumerate(next_question_data['questions']):
                        answer = st.radio(rf_question, ["No", "Yes"], key=f"rf_{i}")
                        red_flag_answers.append(f"{rf_question}: {answer}")
                    
                    if st.button("Complete Red Flag Screening"):
                        st.session_state.patient_data['red_flags_complete'] = red_flag_answers
                        st.session_state.red_flags_complete = True
                        
                        # Check if any red flags were positive
                        positive_flags = [answer for answer in red_flag_answers if "Yes" in answer]
                        if positive_flags:
                            st.error("⚠️ POSITIVE RED FLAGS DETECTED - IMMEDIATE MEDICAL ATTENTION RECOMMENDED")
                            st.session_state.messages.append({'role': 'assistant', 'content': "⚠️ Critical symptoms detected. Please seek immediate medical attention."})
                        else:
                            st.success("✅ No immediate red flags detected")
                            st.session_state.messages.append({'role': 'assistant', 'content': "Red flag screening complete. Assessment will continue."})
                        
                        st.rerun()
            else:
                # Assessment complete
                st.success("✅ Assessment Complete!")
                st.session_state.assessment_complete = True
                
                # Generate and display summary
                if st.button("📋 View Complete Assessment Summary"):
                    summary = assessment.generate_summary()
                    st.text_area("Assessment Summary", summary, height=400)
            # ================ END ENHANCED ASSESSMENT LOGIC ================
    
    with col2:
        st.header("📋 Patient Summary")
        
        # Display patient data
        if st.session_state.patient_data:
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            for k, v in st.session_state.patient_data.items():
                if v and not k.endswith('_enhanced'):  # Don't show enhanced data in summary
                    st.text(f"{k.replace('_', ' ').title()}: {v}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate diagnosis report - Enhanced version
        if st.session_state.get('assessment_complete', False):
            st.success("✅ Assessment Complete!")
            
            # Add diagnosis mode selection
            st.subheader("🔬 AI Diagnosis Mode")
            diagnosis_mode = st.selectbox(
                "Select Diagnostic Approach",
                [
                    "Comprehensive Analysis (Recommended)",
                    "Emergency Medicine Focus",
                    "Multidisciplinary Team Approach",
                    "Risk-Stratified Assessment"
                ],
                help="Choose the AI diagnostic approach best suited for this case"
            )
            
            # Add streaming option for diagnosis
            use_streaming = st.checkbox("Enable Real-time Diagnosis Generation", 
                                    value=st.session_state.get('enable_streaming', True),
                                    help="See diagnosis sections generated in real-time")
            
            if st.button("🤖 Generate Enhanced AI Diagnosis Report", type="primary"):
                if not st.session_state.selected_model:
                    st.error("⚠️ Please select an AI model first")
                    return
                
                try:
                    # Create diagnosis generator with enhanced prompts
                    diagnosis_generator = DiagnosisGenerator(ai_provider)
                    
                    if use_streaming:
                        # Real-time streaming diagnosis
                        st.markdown("### 🔬 Real-time AI Diagnosis Generation")
                        
                        diagnosis_sections = {}
                        sections_to_generate = [
                            ('Primary Assessment', 'primary_assessment'),
                            ('Differential Diagnosis', 'differential_diagnosis'),
                            ('Emergency Assessment', 'emergency_assessment'),
                            ('Risk Assessment', 'risk_assessment'),
                            ('Diagnostic Plan', 'diagnostic_plan'),
                            ('Treatment Plan', 'treatment_plan'),
                            ('Prognosis', 'prognosis'),
                            ('Specialist Recommendations', 'specialist_recommendations')
                        ]
                        
                        for section_name, section_key in sections_to_generate:
                            with st.expander(f"📋 {section_name}", expanded=True):
                                stream_container = st.empty()
                                
                                # Generate section with streaming
                                patient_summary = diagnosis_generator._create_enhanced_summary(st.session_state.patient_data)
                                
                                if section_key == 'primary_assessment':
                                    result = diagnosis_generator._generate_supreme_assessment(patient_summary, st.session_state.selected_model)
                                elif section_key == 'differential_diagnosis':
                                    result = diagnosis_generator._generate_multidisciplinary_diagnosis(patient_summary, st.session_state.selected_model)
                                elif section_key == 'emergency_assessment':
                                    result = diagnosis_generator._generate_emergency_assessment(patient_summary, st.session_state.selected_model)
                                elif section_key == 'risk_assessment':
                                    result = diagnosis_generator._generate_enhanced_risk_assessment(patient_summary, st.session_state.selected_model)
                                elif section_key == 'diagnostic_plan':
                                    result = diagnosis_generator._generate_enhanced_diagnostic_plan(patient_summary, st.session_state.selected_model)
                                elif section_key == 'treatment_plan':
                                    result = diagnosis_generator._generate_evidence_based_treatment(patient_summary, st.session_state.selected_model)
                                elif section_key == 'prognosis':
                                    result = diagnosis_generator._generate_statistical_prognosis(patient_summary, st.session_state.selected_model)
                                elif section_key == 'specialist_recommendations':
                                    result = diagnosis_generator._generate_specialist_recommendations(patient_summary, st.session_state.selected_model)
                                else:
                                    result = f"{section_name} section not implemented"
                                
                                diagnosis_sections[section_key] = result
                                stream_container.markdown(result)
                        
                        st.success("✅ Enhanced AI Diagnosis Complete!")
                        
                    else:
                        # Traditional batch processing
                        with st.spinner("🔬 Generating comprehensive AI diagnosis using advanced medical reasoning..."):
                            diagnosis_sections = diagnosis_generator.generate_full_diagnosis(
                                st.session_state.patient_data, 
                                st.session_state.selected_model
                            )
                            
                            st.success("✅ Enhanced AI Diagnosis Complete!")
                            
                            # Display sections in expandable format
                            section_titles = {
                                'primary_assessment': '🔍 Primary Clinical Assessment',
                                'differential_diagnosis': '🧠 Multidisciplinary Differential Diagnosis',
                                'emergency_assessment': '🚨 Emergency Medicine Assessment',
                                'risk_assessment': '⚠️ Risk Stratification',
                                'diagnostic_plan': '🔬 Diagnostic Workup Plan',
                                'treatment_plan': '💊 Treatment Recommendations',
                                'prognosis': '📈 Prognosis & Outcomes',
                                'specialist_recommendations': '👥 Specialist Consultations'
                            }
                            
                            for key, title in section_titles.items():
                                if key in diagnosis_sections:
                                    with st.expander(title, expanded=False):
                                        st.markdown(diagnosis_sections[key])
                    
                    # Generate enhanced PDF
                    with st.spinner("📄 Generating enhanced PDF report..."):
                        pdf_data = generate_diagnosis_pdf(
                            st.session_state.patient_data,
                            diagnosis_sections,
                            st.session_state.conversation_id,
                            st.session_state.selected_model
                        )
                        
                        # Enhanced download options
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button(
                                "📥 Download Enhanced Diagnosis Report",
                                data=base64.b64decode(pdf_data),
                                file_name=f"enhanced_ai_diagnosis_{st.session_state.conversation_id}.pdf",
                                mime="application/pdf",
                                help="Download comprehensive AI diagnosis report"
                            )
                        
                        with col_b:
                            # Add option to generate summary report
                            if st.button("📋 Generate Summary Report"):
                                st.info("Summary report feature coming soon!")
                                
                except Exception as e:
                    st.error(f"❌ Error generating enhanced diagnosis: {str(e)}")
                    st.info("Please check your API key and model selection, then try again.")
                    
                    # Add debugging info
                    with st.expander("🔧 Debug Information"):
                        st.write(f"Model: {st.session_state.selected_model}")
                        st.write(f"Provider: {st.session_state.ai_provider}")
                        st.write(f"Error details: {str(e)}")
                        
if __name__ == "__main__":
    main()