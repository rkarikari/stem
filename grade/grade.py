import streamlit as st
import json
import time
import requests
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

# Grade level mapping
GRADE_MAPPING = {
    'K': 'Kindergarten',
    '1-3': 'Lower Primary (Basic 1–3)',
    '4-5': 'Upper Primary (Basic 4–6)', 
    '6-8': 'Junior High School (JHS 1–3)',
    '9-12': 'Senior High School (SHS 1–3)',
    'Post-12': 'University'
}

def calculate_age(birth_date):
    """Calculate age from birth date"""
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def auto_select_grade(birth_date):
    """Auto-select grade level based on date of birth"""
    age = calculate_age(birth_date)
    if age <= 5: return 'K'
    elif 6 <= age <= 8: return '1-3'
    elif 9 <= age <= 10: return '4-5'
    elif 11 <= age <= 13: return '6-8'
    elif 14 <= age <= 17: return '9-12'
    else: return 'Post-12'


# Page config
st.set_page_config(page_title="RadioSport SAS", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
def init_session_state():
    defaults = {
        'current_question': 0, 'answers': {}, 'current_assessment': None, 'student_name': '',
        'student_id': '', 'grade_level': '', 'birth_date': None, 'auto_grade': '',
        'assessments_complete': {'competence': False, 'iq': False, 'gpa': False},
        'all_assessments_done': False, 'report_generated': False, 'assessment_scores': {},
        'competence_score': 0, 'iq_score': 0, 'gpa_score': 0.0, 'strengths': [], 'weaknesses': [],
        'recommendations': [], 'ollama_host': 'http://localhost:11434', 'openrouter_key': '',
        'selected_provider': 'ollama', 'selected_model': 'llama2', 'connection_status': False,
        'available_models': [], 'questions_data': {}, 'current_response': '', 'assessment_started': False,
        'assessment_responses': {}, 'current_assessment_answers': [], 'generated_report': '',
        'generated_questions': {}, 'questions_generation_complete': {'competence': False, 'iq': False, 'gpa': False},
        'assessment_config': {'competence': 12, 'iq': 15, 'gpa': 10}  # Add this line
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_ollama_connection():
    """Check connection to Ollama server with improved error handling"""
    try:
        response = requests.get(f"{st.session_state.ollama_host}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            models = [model['name'] for model in models_data if 'name' in model]
            st.session_state.available_models = models if models else ['llama2']
            st.session_state.connection_status = True
            return True, ''
        else:
            st.session_state.connection_status = False
            return False, f"Server responded with status {response.status_code}"
    except requests.exceptions.Timeout:
        st.session_state.connection_status = False
        return False, "Connection timeout - check if Ollama is running"
    except requests.exceptions.ConnectionError:
        st.session_state.connection_status = False
        return False, "Connection refused - check Ollama host address"
    except Exception as e:
        st.session_state.connection_status = False
        return False, f"Connection error: {str(e)}"

def load_openrouter_key():
    """Load OpenRouter API key from secrets file"""
    try:
        # Try to load from Streamlit secrets
        if hasattr(st, 'secrets') and 'openrouter' in st.secrets and 'api_key' in st.secrets['openrouter']:
            return st.secrets['general']['api_key']
        
        # Try to load from .streamlit/secrets.toml
        import os
        secrets_path = os.path.join('.streamlit', 'secrets.toml')
        if os.path.exists(secrets_path):
            import toml
            secrets = toml.load(secrets_path)
            return secrets.get('general', {}).get('api_key', '')
        
        return ''
    except Exception:
        return ''

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_free_openrouter_models():
    """Dynamically load OpenRouter free models with enhanced display info"""
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
        
        # Filter for free models - KEEP FULL MODEL OBJECTS with enhanced info
        for model in models_data.get('data', []):
            pricing = model.get('pricing', {})
            prompt_price = float(pricing.get('prompt', '0'))
            completion_price = float(pricing.get('completion', '0'))
            
            if prompt_price == 0 and completion_price == 0:
                model_id = model.get('id', '')
                if model_id:
                    # Enhanced model object with display-friendly info
                    enhanced_model = {
                        'id': model_id,
                        'name': model.get('name', model_id),
                        'description': model.get('description', 'No description available'),
                        'context_length': model.get('context_length', 'Unknown'),
                        'max_output': model.get('max_output', 'Unknown'),
                        'architecture': model.get('architecture', {}).get('tokenizer', 'Unknown'),
                        'owned_by': model.get('owned_by', 'Unknown'),
                        'created': model.get('created', None),
                        'display_name': f"{model.get('name', model_id)} (Context: {format_context_length(model.get('context_length', 0))})",
                        'full_model_data': model  # Keep original for API calls
                    }
                    free_models.append(enhanced_model)
        
        if not free_models:
            st.error("No free models found in OpenRouter API response.")
            return []
        
        # Sort alphabetically by model name
        free_models.sort(key=lambda x: x.get('name', '').lower())
        return free_models
        
    except Exception as e:
        st.error(f"Error loading OpenRouter models: {str(e)}")
        return []

def format_context_length(context_length):
    """Format context length for display"""
    if not context_length or context_length == 0:
        return "Unknown"
    
    # Convert to more readable format
    if context_length >= 1000000:
        return f"{context_length // 1000000}M"
    elif context_length >= 1000:
        return f"{context_length // 1000}K"
    else:
        return str(context_length)

def display_model_info(model):
    """Display detailed model information in Streamlit"""
    with st.expander(f"📋 {model['name']}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model ID:** `{model['id']}`")
            st.markdown(f"**Owner:** {model['owned_by']}")
            st.markdown(f"**Architecture:** {model['architecture']}")
        
        with col2:
            st.markdown(f"**Context Length:** {format_context_length(model['context_length'])}")
            st.markdown(f"**Max Output:** {format_context_length(model['max_output'])}")
            if model['created']:
                created_date = datetime.fromtimestamp(model['created']).strftime('%Y-%m-%d')
                st.markdown(f"**Created:** {created_date}")
        
        if model['description'] != 'No description available':
            st.markdown(f"**Description:** {model['description']}")
        
        # Display context length prominently at the bottom
        st.markdown("---")
        context_display = format_context_length(model['context_length'])

def generate_ai_response(prompt: str, stream: bool = True):
    try:
        if st.session_state.selected_provider == 'ollama':
            return generate_ollama_response(prompt, stream)
        else:
            return generate_openrouter_response(prompt, stream)
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_ollama_response(prompt: str, stream: bool = True):
    try:
        data = {
            "model": st.session_state.selected_model,
            "prompt": prompt,
            "stream": stream
        }
        
        if stream:
            response = requests.post(f"{st.session_state.ollama_host}/api/generate", 
                                   json=data, stream=True, timeout=30)
            full_response = ""
            placeholder = st.empty()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            full_response += chunk['response']
                            placeholder.markdown(full_response + "▋")
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            placeholder.markdown(full_response)
            return full_response
        else:
            response = requests.post(f"{st.session_state.ollama_host}/api/generate", 
                                   json=data, timeout=60)
            return response.json().get('response', 'No response received')
    except Exception as e:
        return f"Ollama error: {str(e)}"

def generate_openrouter_response(prompt: str, stream: bool = True):
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.openrouter_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": st.session_state.selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream
        }
        
        if stream:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                   headers=headers, json=data, stream=True, timeout=30)
            full_response = ""
            placeholder = st.empty()
            
            for line in response.iter_lines():
                if line and line.startswith(b'data: '):
                    try:
                        chunk_data = line[6:].decode('utf-8')
                        if chunk_data.strip() == '[DONE]':
                            break
                        chunk = json.loads(chunk_data)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                full_response += delta['content']
                                placeholder.markdown(full_response + "▋")
                    except json.JSONDecodeError:
                        continue
            
            placeholder.markdown(full_response)
            return full_response
        else:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                   headers=headers, json=data, timeout=60)
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"OpenRouter error: {str(e)}"

# Assessment Questions functions
def generate_assessment_questions(assessment_type, grade_level, num_questions):
    """Generate AI-powered assessment questions based on type and grade level"""
    
    # Define assessment prompts based on type
    prompts = {
        'competence': f"""
        Generate {num_questions} diverse competence assessment questions for a {grade_level} student.
        Focus on: problem-solving, critical thinking, communication skills, teamwork, learning strategies, and academic confidence.
        
        Return ONLY a valid JSON array with this exact format:
        [
            {{"q": "question text", "type": "text", "weight": 0.15}},
            {{"q": "question text", "type": "slider", "min": 1, "max": 10, "weight": 0.1}},
            {{"q": "question text", "type": "radio", "options": ["option1", "option2", "option3"], "weight": 0.1}}
        ]
        
        Use these question types:
        - "text" for open-ended questions (weight: 0.1-0.2)
        - "slider" for 1-10 ratings (weight: 0.05-0.15)
        - "radio" for multiple choice (weight: 0.05-0.15)
        - "number" for numeric answers (weight: 0.05-0.1)
        
        Ensure weights sum to approximately 1.0. Make questions age-appropriate and engaging.
        """,
        
        'iq': f"""
        Generate {num_questions} diverse IQ assessment questions for a {grade_level} student.
        Include: logical reasoning, pattern recognition, mathematical problems, verbal reasoning, spatial thinking, and sequence completion.
        
        Return ONLY a valid JSON array with this exact format:
        [
            {{"q": "question text", "type": "radio", "options": ["A", "B", "C", "D"], "correct": "A", "weight": 0.1}},
            {{"q": "question text", "type": "number", "correct": 42, "weight": 0.15}},
            {{"q": "question text", "type": "text", "weight": 0.1}}
        ]
        
        For questions with definitive answers, include "correct" field.
        Make questions progressively challenging and age-appropriate.
        Ensure weights sum to approximately 1.0.
        """,
        
        'gpa': f"""
        Generate {num_questions} diverse GPA prediction questions for a {grade_level} student.
        Focus on: current grades, study habits, attendance, assignment completion, class participation, time management, and academic challenges.
        
        Return ONLY a valid JSON array with this exact format:
        [
            {{"q": "question text", "type": "number", "weight": 0.2}},
            {{"q": "question text", "type": "slider", "min": 1, "max": 10, "weight": 0.15}},
            {{"q": "question text", "type": "radio", "options": ["Always", "Usually", "Sometimes", "Rarely"], "weight": 0.1}}
        ]
        
        Include questions about actual performance metrics and study behaviors.
        Ensure weights sum to approximately 1.0.
        """
    }
    
    try:
        prompt = prompts.get(assessment_type, "")
        if not prompt:
            return []
        
        response = generate_ai_response(prompt, stream=False)
        
        # Extract JSON from response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        if response.startswith('```'):
            response = response[3:]
        
        # Find JSON array in response
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            questions = json.loads(json_str)
            
            # Validate and clean questions
            validated_questions = []
            for q in questions:
                if 'q' in q and 'type' in q and 'weight' in q:
                    validated_questions.append(q)
            
            return validated_questions
        else:
            st.error(f"Could not parse questions for {assessment_type}")
            return []
            
    except Exception as e:
        st.error(f"Error generating {assessment_type} questions: {str(e)}")
        return []

def get_assessment_questions(assessment_type):
    """Get questions for assessment, generating them if not already cached"""
    if assessment_type not in st.session_state.generated_questions:
        num_questions = st.session_state.assessment_config[assessment_type]
        grade_level = st.session_state.grade_level
        
        with st.spinner(f"Generating {assessment_type} questions..."):
            questions = generate_assessment_questions(assessment_type, grade_level, num_questions)
            st.session_state.generated_questions[assessment_type] = questions
            st.session_state.questions_generation_complete[assessment_type] = True
    
    return st.session_state.generated_questions.get(assessment_type, [])

# SCORING ALGORITHM REVIEW - IDENTIFIED ISSUES AND FIXES

## CORRECTED CALCULATE_SCORES FUNCTION
def calculate_scores(assessment_type):
    """Fixed version with improved scoring logic"""
    if assessment_type not in st.session_state.assessment_responses:
        return
    
    answers = st.session_state.assessment_responses[assessment_type]
    questions = get_assessment_questions(assessment_type)
    
    total_score = 0
    total_weight = 0
    
    for i, answer in enumerate(answers):
        if i < len(questions):
            weight = questions[i]["weight"]
            q_type = questions[i]["type"]
            
            # FIX 1: Improved Slider Scoring
            if q_type == "slider":
                max_val = questions[i].get("max", 10)
                min_val = questions[i].get("min", 1)
                
                # Use proportional scoring instead of 0-1 normalization
                # This gives partial credit for lower selections
                if max_val > min_val:
                    # Score ranges from 0.3 (minimum) to 1.0 (maximum)
                    score_range = 0.7  # 1.0 - 0.3
                    normalized = (float(answer) - min_val) / (max_val - min_val)
                    score = 0.3 + (normalized * score_range)
                else:
                    score = 0.7  # Default for invalid ranges
                
            # FIX 2: Improved Number Scoring
            elif q_type == "number":
                if "correct" in questions[i]:
                    # IQ questions with correct answers - enhanced tolerance
                    tolerance = questions[i].get("tolerance", 0)
                    correct_val = float(questions[i]["correct"])
                    try:
                        user_val = float(answer)
                        if abs(user_val - correct_val) <= tolerance:
                            score = 1.0
                        else:
                            # More generous partial credit
                            max_diff = max(correct_val * 0.8, 10)  # 80% tolerance or minimum 10
                            diff = abs(user_val - correct_val)
                            score = max(0.1, 1.0 - (diff / max_diff))
                    except (ValueError, TypeError):
                        score = 0.2  # Some credit for attempting
                else:
                    # Enhanced numeric scoring for non-IQ questions
                    try:
                        numeric_answer = float(answer)
                        if assessment_type == "gpa":
                            # Smart GPA-related scoring
                            if numeric_answer >= 95:     # A+ territory
                                score = 1.0
                            elif numeric_answer >= 90:   # A range
                                score = 0.95
                            elif numeric_answer >= 85:   # B+ range
                                score = 0.85
                            elif numeric_answer >= 80:   # B range
                                score = 0.8
                            elif numeric_answer >= 75:   # B- range
                                score = 0.75
                            elif numeric_answer >= 70:   # C range
                                score = 0.65
                            elif numeric_answer >= 60:   # D range
                                score = 0.5
                            else:                        # Below passing
                                score = 0.3
                        else:
                            # For competence numeric questions
                            if numeric_answer >= 90:
                                score = 1.0
                            elif numeric_answer >= 80:
                                score = 0.9
                            elif numeric_answer >= 70:
                                score = 0.8
                            elif numeric_answer >= 60:
                                score = 0.7
                            elif numeric_answer >= 50:
                                score = 0.6
                            else:
                                score = 0.4
                    except (ValueError, TypeError):
                        score = 0.3  # Some credit for attempting
                        
            # FIX 3: Enhanced Radio Button Scoring
            elif q_type == "radio":
                if "correct" in questions[i]:
                    # IQ questions with definitive correct answers
                    score = 1.0 if str(answer) == str(questions[i]["correct"]) else 0.15
                else:
                    # Enhanced radio scoring with more comprehensive mappings
                    answer_str = str(answer).lower()
                    
                    if assessment_type == "gpa":
                        gpa_scores = {
                            "always": 1.0, "very often": 0.9, "often": 0.8, "frequently": 0.85,
                            "usually": 0.8, "sometimes": 0.6, "occasionally": 0.55,
                            "rarely": 0.35, "never": 0.1, "seldom": 0.3,
                            "excellent": 1.0, "very good": 0.9, "good": 0.8, "satisfactory": 0.7,
                            "fair": 0.6, "average": 0.6, "below average": 0.4, "poor": 0.25,
                            "very poor": 0.1, "outstanding": 1.0, "above average": 0.8,
                            "yes": 0.8, "no": 0.3, "mostly": 0.75, "partly": 0.5
                        }
                        score = gpa_scores.get(answer_str, 0.6)  # Default 0.6 instead of 0.5
                        
                    elif assessment_type == "competence":
                        competence_scores = {
                            "strongly agree": 1.0, "agree": 0.85, "somewhat agree": 0.7,
                            "neutral": 0.55, "somewhat disagree": 0.4, "disagree": 0.25,
                            "strongly disagree": 0.1, "always": 1.0, "very often": 0.9,
                            "often": 0.8, "sometimes": 0.6, "rarely": 0.35, "never": 0.15,
                            "excellent": 1.0, "very good": 0.9, "good": 0.8, "average": 0.6,
                            "below average": 0.4, "poor": 0.25, "very confident": 1.0,
                            "confident": 0.8, "somewhat confident": 0.6, "not confident": 0.3,
                            "yes": 0.8, "no": 0.4, "mostly": 0.75, "partly": 0.5
                        }
                        score = competence_scores.get(answer_str, 0.65)  # Higher default
                        
                    else:  # IQ assessment
                        # For IQ radio without correct answers, more generous scoring
                        score = 0.6  # More generous default for IQ
                        
            # FIX 4: Enhanced Text Response Evaluation
            else:  # Text responses
                try:
                    # Attempt AI evaluation with better prompt
                    evaluation_prompt = f"""
                    Evaluate this {assessment_type} assessment response on a scale of 0.0 to 1.0.
                    
                    Question: {questions[i]['q']}
                    Student Answer: {answer}
                    
                    Scoring Guide:
                    - 0.9-1.0: Exceptional insight, comprehensive understanding
                    - 0.8-0.9: Strong understanding, well-articulated
                    - 0.7-0.8: Good understanding, clear communication
                    - 0.6-0.7: Adequate understanding, basic communication
                    - 0.5-0.6: Some understanding, limited communication
                    - 0.4-0.5: Minimal understanding, poor communication
                    - 0.0-0.4: Little to no understanding
                    
                    Return only the numerical score (e.g., 0.75).
                    """
                    
                    ai_response = generate_ai_response(evaluation_prompt, stream=False)
                    # Extract number from response
                    import re
                    numbers = re.findall(r'0?\.\d+|[01]\.0*', ai_response)
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.0, min(1.0, score))
                    else:
                        # Enhanced fallback scoring
                        score = evaluate_text_response_enhanced(questions[i]['q'], answer, assessment_type)
                except Exception as e:
                    # Enhanced fallback scoring
                    score = evaluate_text_response_enhanced(questions[i]['q'], answer, assessment_type)
            
            total_score += score * weight
            total_weight += weight
    
    # Calculate final scores with improved scaling
    if total_weight > 0:
        final_score = total_score / total_weight
    else:
        final_score = 0.6  # More generous default
    
    # FIX 5: Improved Final Score Scaling
    if assessment_type == "competence":
        # Scale to 0-100 with better distribution
        competence_score = int(final_score * 100)
        # Apply reasonable floor (40 instead of 30)
        competence_score = max(competence_score, 40)
        st.session_state.competence_score = competence_score
        st.session_state.assessment_scores['competence'] = competence_score
        
    elif assessment_type == "iq":
        # Improved IQ scaling: 80-130 range for educational context
        iq_score = int(80 + (final_score * 50))  # Scale to 80-130 range
        # Apply reasonable bounds
        iq_score = max(85, min(130, iq_score))
        st.session_state.iq_score = iq_score
        st.session_state.assessment_scores['iq'] = iq_score
        
    elif assessment_type == "gpa":
        # Enhanced GPA scaling with bonus for effort
        gpa_score = final_score * 4.0
        # Apply reasonable minimum (1.5 instead of 1.0)
        gpa_score = max(gpa_score, 1.5)
        st.session_state.gpa_score = round(gpa_score, 2)
        st.session_state.assessment_scores['gpa'] = round(gpa_score, 2)


## ENHANCED TEXT EVALUATION FUNCTION
def evaluate_text_response_enhanced(question, answer, assessment_type):
    """Enhanced text response evaluation with better scoring"""
    answer_text = str(answer).strip()
    
    # Base scoring by length with better thresholds
    if len(answer_text) >= 150:
        length_score = 0.4
    elif len(answer_text) >= 100:
        length_score = 0.35
    elif len(answer_text) >= 50:
        length_score = 0.3
    elif len(answer_text) >= 20:
        length_score = 0.25
    else:
        length_score = 0.15
    
    # Enhanced content quality scoring
    quality_indicators = {
        'reasoning': ['because', 'therefore', 'since', 'due to', 'as a result', 'consequently'],
        'examples': ['example', 'instance', 'such as', 'like', 'including', 'for example'],
        'analysis': ['analyze', 'compare', 'contrast', 'evaluate', 'consider', 'examine'],
        'complexity': ['however', 'although', 'nevertheless', 'furthermore', 'moreover', 'additionally'],
        'personal': ['i think', 'i believe', 'in my opinion', 'personally', 'my experience'],
        'academic': ['research', 'study', 'theory', 'according to', 'evidence', 'data']
    }
    
    content_score = 0
    answer_lower = answer_text.lower()
    
    for category, indicators in quality_indicators.items():
        if any(indicator in answer_lower for indicator in indicators):
            content_score += 0.1  # Each category adds 0.1
    
    # Bonus for comprehensive responses
    if len(answer_text) >= 100 and content_score >= 0.2:
        content_score += 0.1
    
    # Combine scores with higher maximum potential
    total_score = length_score + content_score
    
    # Remove artificial cap, allow full range
    return min(total_score, 1.0)  # Allow up to 1.0 instead of 0.9


def generate_assessment_report():
    completed_assessments = [k for k, v in st.session_state.assessments_complete.items() if v]
    
    scores_text = ""
    if 'competence' in completed_assessments:
        scores_text += f"- Competence Level: {st.session_state.competence_score}/100\n"
    if 'iq' in completed_assessments:
        scores_text += f"- IQ Score: {st.session_state.iq_score}\n"
    if 'gpa' in completed_assessments:
        scores_text += f"- GPA Prediction: {st.session_state.gpa_score:.2f}/4.0\n"
    
    # Collect all responses for context
    all_responses = ""
    for assessment in completed_assessments:
        if assessment in st.session_state.assessment_responses:
            responses = st.session_state.assessment_responses[assessment][:3]  # First 3 responses
            all_responses += f"{assessment.upper()}: {responses}\n"
    
    prompt = f"""
    Generate a comprehensive student assessment report based on the following data:
    
    Student: {st.session_state.student_name} (ID: {st.session_state.student_id})
    Grade Level: {st.session_state.grade_level}
    Assessment Date: {datetime.now().strftime('%Y-%m-%d')}
    Completed Assessments: {', '.join(completed_assessments).upper()}
    
    Scores:
    {scores_text}
    
    Sample Student Responses: {all_responses}
    
    Please provide a detailed report including:
    1. Executive Summary
    2. Individual Assessment Analysis (for each completed assessment)
    3. Key Strengths (4-5 specific points)
    4. Areas for Improvement (4-5 specific points)
    5. Detailed Recommendations for Teachers (6-8 actionable items)
    6. Suggested Learning Strategies for Student
    7. Academic Support Recommendations
    8. Next Steps and Follow-up Actions
    
    Format as a comprehensive professional educational assessment report.
    """
    
    return generate_ai_response(prompt, stream=True)

def create_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                fontSize=18, textColor=colors.darkblue, alignment=1)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], 
                                  fontSize=14, textColor=colors.darkblue)
    
    # Title
    story.append(Paragraph("COMPREHENSIVE STUDENT ASSESSMENT REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Student Information Table
    student_data = [
        ['Student Name:', st.session_state.student_name],
        ['Student ID:', st.session_state.student_id],
        ['Grade Level:', st.session_state.grade_level],
        ['Assessment Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Completed Assessments:', ', '.join([k.upper() for k, v in st.session_state.assessments_complete.items() if v])]
    ]
    
    student_table = Table(student_data, colWidths=[2*inch, 4*inch])
    student_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(student_table)
    story.append(Spacer(1, 20))
    
    # Assessment Scores
    story.append(Paragraph("ASSESSMENT SCORES", heading_style))
    story.append(Spacer(1, 10))
    
    scores_data = [['Assessment Type', 'Score', 'Performance Level']]
    
    if st.session_state.assessments_complete.get('competence', False):
        level = 'Excellent' if st.session_state.competence_score >= 80 else 'Good' if st.session_state.competence_score >= 60 else 'Needs Improvement'
        scores_data.append(['Competence Level', f"{st.session_state.competence_score}/100", level])
    
    if st.session_state.assessments_complete.get('iq', False):
        level = 'Above Average' if st.session_state.iq_score >= 110 else 'Average' if st.session_state.iq_score >= 90 else 'Below Average'
        scores_data.append(['IQ Score', str(st.session_state.iq_score), level])
    
    if st.session_state.assessments_complete.get('gpa', False):
        level = 'Excellent' if st.session_state.gpa_score >= 3.5 else 'Good' if st.session_state.gpa_score >= 2.5 else 'Needs Improvement'
        scores_data.append(['GPA Prediction', f"{st.session_state.gpa_score:.2f}/4.0", level])
    
    scores_table = Table(scores_data, colWidths=[2*inch, 1.5*inch, 2*inch])
    scores_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(scores_table)
    story.append(Spacer(1, 20))
    
    # AI Generated Report Content
    if hasattr(st.session_state, 'generated_report') and st.session_state.generated_report:
        story.append(Paragraph("DETAILED ANALYSIS", heading_style))
        story.append(Spacer(1, 10))
        
        # Split the AI report into paragraphs
        report_lines = st.session_state.generated_report.split('\n')
        for line in report_lines:
            if line.strip():
                if line.startswith('#') or line.isupper() or any(keyword in line.lower() for keyword in ['summary', 'strengths', 'improvement', 'recommendations']):
                    story.append(Paragraph(line.strip(), heading_style))
                else:
                    story.append(Paragraph(line.strip(), styles['Normal']))
                story.append(Spacer(1, 6))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Report generated by AI-Powered Student Assessment System", styles['Italic']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def render_question(question_data, question_idx):
    q = question_data["q"]
    q_type = question_data["type"]
    
    st.markdown(f"### Question {question_idx + 1}")
    st.markdown(f"**{q}**")
    
    answer = None
    
    if q_type == "text":
        answer = st.text_area(f"Your answer:", key=f"q_{question_idx}", height=100)
    elif q_type == "number":
        answer = st.number_input(f"Enter number:", key=f"q_{question_idx}")
    elif q_type == "slider":
        answer = st.slider(f"Rate (1-{question_data['max']}):", 
                          min_value=question_data["min"], 
                          max_value=question_data["max"], 
                          key=f"q_{question_idx}")
    elif q_type == "radio":
        answer = st.radio(f"Select one:", question_data["options"], key=f"q_{question_idx}")
    
    return answer

def main():
    init_session_state()
    
    # Replace the sidebar section with this corrected version:
    with st.sidebar:
        st.title("🧟RadioSport Grade")
        
        # Provider Selection with Radio Buttons
        provider_options = ["Local", "Cloud"]
        provider_labels = {"Local": "ollama", "Cloud": "openrouter"}
        
        selected_provider_label = st.radio(
            "AI Provider:", 
            provider_options,
            index=0 if st.session_state.selected_provider == "ollama" else 1
        )
        st.session_state.selected_provider = provider_labels[selected_provider_label]
        
        if st.session_state.selected_provider == "ollama":
            st.session_state.ollama_host = st.text_input("Ollama Host:", value=st.session_state.ollama_host)
            
            # Improved connection testing
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔗 Test Connection", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        success, message = check_ollama_connection()
                        if success:
                            #st.success(f" {message}")
                            st.session_state.connection_test_result = "success"
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.connection_test_result = "failed"
            
            with col2:
                # Connection status indicator
                if st.session_state.connection_status:
                    st.markdown("🟢")
                elif hasattr(st.session_state, 'connection_test_result'):
                    st.markdown("🔴")
                else:
                    st.markdown("⚪")
            
            # Model selection (only show if connected)
            if st.session_state.connection_status and st.session_state.available_models:
                st.session_state.selected_model = st.selectbox(
                    "Available Models:", 
                    st.session_state.available_models,
                    index=0 if st.session_state.selected_model not in st.session_state.available_models else 
                          st.session_state.available_models.index(st.session_state.selected_model)
                )
            elif st.session_state.selected_provider == "ollama":
                st.info("⚠️ Connect to see available models")
        
        else:  # OpenRouter (Cloud)
                    # Auto-load API key from secrets if not already loaded
                    if not st.session_state.openrouter_key:
                        loaded_key = load_openrouter_key()
                        if loaded_key:
                            st.session_state.openrouter_key = loaded_key
                    
                    # Show key input only if auto-load failed
                    if not st.session_state.openrouter_key:
                        st.session_state.openrouter_key = st.text_input("OpenRouter API Key:", type="password")
                        
                    
                    if st.session_state.openrouter_key:
                        free_models = get_free_openrouter_models()
                        
                        if free_models:
                            # Create model options with display names
                            model_options = {model['display_name']: model['id'] for model in free_models}
                            
                            # Select model
                            selected_display_name = st.selectbox("Free Models:", 
                                                               options=list(model_options.keys()),
                                                               key="openrouter_model_select")
                            
                            # Set selected model ID
                            if selected_display_name:
                                st.session_state.selected_model = model_options[selected_display_name]
                                
                                # Find the full model data for context info
                                selected_model_data = next(
                                    (model for model in free_models if model['display_name'] == selected_display_name), 
                                    None
                                )
                                
                                # Display context length info under the model selection
                                if selected_model_data:
                                    context_length = selected_model_data.get('context_length', 'Unknown')
                                    max_output = selected_model_data.get('max_output', 'Unknown')
                                    
                                    st.markdown("---")
                                    st.markdown("**Model Info:**")
                                    st.markdown(f"🔤 **Context:** {format_context_length(context_length)}")
                                    if max_output != 'Unknown':
                                        st.markdown(f"📤 **Max Output:** {format_context_length(max_output)}")
                                    
                                    # Optional: Show model description in a smaller text
                                    if selected_model_data.get('description') and selected_model_data['description'] != 'No description available':
                                        with st.expander("📋 Model Details"):
                                            st.caption(selected_model_data['description'])
                        else:
                            st.warning("No free models available. Please check your connection.")
        
        st.markdown("---")
        
        # Assessment Configuration
        with st.expander("⚙️ Assessment Configuration", expanded=False):
            st.session_state.assessment_config['competence'] = st.slider(
                "Competence Questions:", 8, 20, st.session_state.assessment_config['competence'], 
                key="competence_slider_config")
            st.session_state.assessment_config['iq'] = st.slider(
                "IQ Questions:", 10, 25, st.session_state.assessment_config['iq'],
                key="iq_slider_config")
            st.session_state.assessment_config['gpa'] = st.slider(
                "GPA Questions:", 6, 15, st.session_state.assessment_config['gpa'],
                key="gpa_slider_config")
        
        st.markdown("---")
        
        # Overall connection status
        st.markdown("**System Status:**")
        if st.session_state.selected_provider == "ollama":
            if st.session_state.connection_status:
                st.success("")
            else:
                st.warning("🟡 Disconnected")
        else:
            if st.session_state.openrouter_key:
                st.success("🟢 Ready")
            else:
                st.warning("🟡 API Key Required")


    
    # Main Application
    st.title("🎓RadioSport Student Assessment System")
    
    # Student Information
# Student Information
    if not st.session_state.assessment_started:
        st.markdown("## Student Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.student_name = st.text_input("Student Name:")
            st.session_state.student_id = st.text_input("Student ID:")
            
            # Date of birth input
            st.session_state.birth_date = st.date_input(
                "Date of Birth:",
                value=st.session_state.birth_date if st.session_state.birth_date else date(2010, 1, 1),
                min_value=date(1990, 1, 1),
                max_value=date.today()
            )
        
        with col2:
            # Auto-select grade based on birth date
            if st.session_state.birth_date:
                auto_grade = auto_select_grade(st.session_state.birth_date)
                st.session_state.auto_grade = auto_grade
                st.info(f"Suggested grade level based on age: **{GRADE_MAPPING[auto_grade]}**")
            
            # Manual grade selection with mapping
            grade_options = list(GRADE_MAPPING.keys())
            grade_labels = [f"{k} - {v}" for k, v in GRADE_MAPPING.items()]
            
            selected_idx = grade_options.index(st.session_state.auto_grade) if st.session_state.auto_grade in grade_options else 0
            selected_grade = st.selectbox(
                "Grade Level:", 
                options=grade_options,
                format_func=lambda x: f"{x} - {GRADE_MAPPING[x]}",
                index=selected_idx
            )
            st.session_state.grade_level = f"{selected_grade} - {GRADE_MAPPING[selected_grade]}"
        
        if st.button("Begin Assessment Process", type="primary"):
            if st.session_state.student_name and st.session_state.student_id and st.session_state.birth_date:
                st.session_state.assessment_started = True
                st.rerun()
            else:
                st.error("Please fill in all student information including date of birth.")
    
    # Assessment Selection and Progress
    elif st.session_state.assessment_started and not st.session_state.all_assessments_done:
        st.markdown("## Assessment Progress")
        
        # Progress indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅ Complete" if st.session_state.assessments_complete['competence'] else "⏳ Pending"
            st.markdown(f"**Competence Assessment:** {status}")
        
        with col2:
            status = "✅ Complete" if st.session_state.assessments_complete['iq'] else "⏳ Pending"
            st.markdown(f"**IQ Assessment:** {status}")
        
        with col3:
            status = "✅ Complete" if st.session_state.assessments_complete['gpa'] else "⏳ Pending"
            st.markdown(f"**GPA Assessment:** {status}")
        
        st.markdown("---")
        
        # Current Assessment Selection
        if st.session_state.current_assessment is None:
            st.markdown("## Choose Next Assessment")
            
            available_assessments = []
            if not st.session_state.assessments_complete['competence']:
                available_assessments.append(('competence', 'Competence Level Assessment'))
            if not st.session_state.assessments_complete['iq']:
                available_assessments.append(('iq', 'IQ Assessment'))
            if not st.session_state.assessments_complete['gpa']:
                available_assessments.append(('gpa', 'GPA Prediction Assessment'))
            
            if available_assessments:
                for assessment_key, assessment_name in available_assessments:
                    if st.button(f"Start {assessment_name}", key=f"btn_{assessment_key}"):
                        st.session_state.current_assessment = assessment_key
                        st.session_state.current_question = 0
                        st.session_state.current_assessment_answers = []
                        st.rerun()
            
            # Option to generate report with completed assessments
            completed_count = sum(st.session_state.assessments_complete.values())
            if completed_count > 0:
                st.markdown("---")
                st.markdown(f"**{completed_count} assessment(s) completed.** You can:")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Continue with remaining assessments"):
                        pass  # Just refresh to show available assessments
                with col2:
                    if st.button("Generate report with current data"):
                        st.session_state.all_assessments_done = True
                        st.rerun()
        

        # Assessment Questions
        else:
            current_assessment = st.session_state.current_assessment
            questions = get_assessment_questions(current_assessment)  # Changed this line
            
            if not questions:
                st.error("Failed to generate questions. Please try again.")
                if st.button("Retry Question Generation"):
                    if current_assessment in st.session_state.generated_questions:
                        del st.session_state.generated_questions[current_assessment]
                    st.rerun()
                return
            
            current_q = st.session_state.current_question
            
            if current_q < len(questions):
                st.markdown(f"## {current_assessment.upper()} Assessment")
                st.progress((current_q + 1) / len(questions))
                st.markdown(f"**Question {current_q + 1} of {len(questions)}**")
                
                question_data = questions[current_q]
                answer = render_question(question_data, current_q)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("Previous", disabled=current_q == 0):
                        st.session_state.current_question -= 1
                        st.rerun()
                
                with col2:
                    if st.button("Skip Assessment"):
                        st.session_state.current_assessment = None
                        st.rerun()
                
                with col3:
                    if st.button("Next", type="primary", disabled=not answer and answer != 0):
                        # Store answer
                        if len(st.session_state.current_assessment_answers) <= current_q:
                            st.session_state.current_assessment_answers.append(answer)
                        else:
                            st.session_state.current_assessment_answers[current_q] = answer
                        
                        st.session_state.current_question += 1
                        st.rerun()
            
            else:
                # Current Assessment Complete
                st.session_state.assessment_responses[current_assessment] = st.session_state.current_assessment_answers
                st.session_state.assessments_complete[current_assessment] = True
                calculate_scores(current_assessment)
                st.session_state.current_assessment = None
                
                st.success(f"✅ {current_assessment.upper()} Assessment completed!")
                st.info(f"Answered {len(st.session_state.current_assessment_answers)} questions")
                
                # Check if all assessments are done
                if all(st.session_state.assessments_complete.values()):
                    st.session_state.all_assessments_done = True
                
                if st.button("Continue"):
                    st.rerun()
    
    # Results and Report
    elif st.session_state.all_assessments_done or all(st.session_state.assessments_complete.values()):
        st.session_state.all_assessments_done = True
        st.markdown("## Assessment Results")
        
        # Display Scores for completed assessments
        completed_assessments = [k for k, v in st.session_state.assessments_complete.items() if v]
        cols = st.columns(len(completed_assessments))
        
        for i, assessment in enumerate(completed_assessments):
            with cols[i]:
                if assessment == 'competence':
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=st.session_state.competence_score,
                        title={'text': "Competence Level"}, domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"},
                                        {'range': [80, 100], 'color': "green"}]}))
                elif assessment == 'iq':
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=st.session_state.iq_score,
                        title={'text': "IQ Score"}, domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [70, 130]}, 'bar': {'color': "purple"}}))
                elif assessment == 'gpa':
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=st.session_state.gpa_score,
                        title={'text': "GPA Prediction"}, domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 4]}, 'bar': {'color': "red"}}))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Generate Report
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Comprehensive Report", type="primary"):
                st.markdown("## 📊 Comprehensive Assessment Report")
                with st.spinner("Generating AI-powered report..."):
                    report = generate_assessment_report()
                    st.session_state.generated_report = report
                st.session_state.report_generated = True
        
        with col2:
            if st.session_state.report_generated and hasattr(st.session_state, 'generated_report'):
                pdf_buffer = create_pdf_report()
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"Student_Assessment_Report_{st.session_state.student_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="secondary"
                )
        
        # Reset Assessment
        st.markdown("---")
        if st.button("Start New Assessment"):
            for key in list(st.session_state.keys()):
                if key not in ['ollama_host', 'openrouter_key', 'selected_provider', 'selected_model', 'available_models', 'connection_status']:
                    del st.session_state[key]
        


if __name__ == "__main__":
    main()