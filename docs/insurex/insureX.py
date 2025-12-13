import streamlit as st
import requests
import json
import base64
import shutil
from io import BytesIO
import PyPDF2
import os
import re
from datetime import datetime, timedelta
import hashlib
from functools import lru_cache
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import io
import cv2
from typing import List, Dict, Any, Optional, Tuple
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Version tracking
APP_VERSION = "2.4.3"  # Fixed pie chart error and claim evaluation logic

# API Settings
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Ollama settings
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_HOST = DEFAULT_OLLAMA_HOST
DEFAULT_MODEL = "gemma3:4b"

# Performance settings
CACHE_TTL = 600
MAX_CACHE_ENTRIES = 100

# Data storage settings
DATA_FILE = "insureX_data.json"
BACKUP_DIR = "backups"

# Enhanced Insurance claim rules with AI decision points
CLAIM_RULES = {
    "auto": {
        "max_amount": 500000,
        "auto_approve_threshold": 20000,
        "ai_review_threshold": 75000,
        "required_docs": ["incident_report", "photos", "estimate", "police_report"],
        "risk_factors": ["weather_conditions", "driver_history", "vehicle_age", "location"],
        "fraud_indicators": ["inconsistent_story", "multiple_claims", "suspicious_timing"],
        "deny_conditions": ["no_coverage", "fraud_confirmed", "expired_policy", "excluded_driver"]
    },
    "health": {
        "max_amount": 150000,
        "auto_approve_threshold": 1500,
        "ai_review_threshold": 25000,
        "required_docs": ["medical_report", "receipts", "prescription", "diagnosis"],
        "risk_factors": ["treatment_type", "provider_network", "pre_authorization"],
        "fraud_indicators": ["billing_anomalies", "unnecessary_procedures", "provider_flags"],
        "deny_conditions": ["pre_existing_exclusion", "experimental_treatment", "expired_policy"]
    },
    "property": {
        "max_amount": 500000,
        "auto_approve_threshold": 8000,
        "ai_review_threshold": 50000,
        "required_docs": ["damage_photos", "repair_estimate", "incident_report", "ownership_proof"],
        "risk_factors": ["property_location", "cause_of_damage", "property_age", "maintenance_history"],
        "fraud_indicators": ["staged_damage", "inflated_estimates", "suspicious_claims_history"],
        "deny_conditions": ["act_of_war", "nuclear_hazard", "maintenance_neglect", "expired_policy"]
    },
    "life": {
        "max_amount": 2000000,
        "auto_approve_threshold": 0,
        "ai_review_threshold": 10000,
        "required_docs": ["death_certificate", "policy_docs", "beneficiary_proof", "medical_records"],
        "risk_factors": ["cause_of_death", "policy_age", "premium_status", "beneficiary_relationship"],
        "fraud_indicators": ["suspicious_death", "recent_policy_changes", "beneficiary_flags"],
        "deny_conditions": ["suicide_clause_active", "fraud_confirmed", "lapsed_policy", "contestable_period"]
    }
}

# Enhanced AI Decision Matrix with explicit rejection requirements
AI_DECISION_PROMPTS = {
    "risk_assessment": """
    As an expert insurance claims adjuster, analyze this claim for risk factors:
    
    Claim: {claim_type} - ${amount:,}
    Details: {description}
    Documents: {documents}
    
    Evaluate:
    1. Risk Level (Low/Medium/High)
    2. Fraud Probability (0-100%)
    3. Policy Compliance
    4. Documentation Completeness
    5. Immediate Action Required
    
    Provide structured analysis with confidence scores.
    """,
    
    "fraud_detection": """
    Analyze this insurance claim for potential fraud indicators:
    
    Claim Details: {claim_details}
    Timeline: {timeline}
    Previous Claims: {claim_history}
    
    Red Flags to Check:
    - Inconsistent information
    - Suspicious timing patterns
    - Unusual claim amounts
    - Documentation anomalies
    - Behavioral indicators
    
    Rate fraud probability (0-100%) and provide detailed reasoning.
    """,
    
    "final_decision": """
    Make a final decision on this insurance claim:
    
    Claim Summary: {summary}
    Risk Assessment: {risk_data}
    Fraud Analysis: {fraud_data}
    Policy Coverage: {coverage_data}
    
    Based on all factors, provide:
    1. Decision: APPROVE/DENY/INVESTIGATE
    2. Confidence Level: 0-100%
    3. Detailed Reasoning with specific evidence
    4. Primary Rejection Reason (if denied) - MUST be specific and evidence-based
    5. Policy clause or rule violated (if denied)
    6. Changes Needed for Approval (if denied)
    7. Conditions (if approved)
    8. Next Steps Required
    
    IMPORTANT: If denying claim, rejection reason MUST:
    - Reference specific policy clause or rule
    - Cite evidence from claim details or documents
    - Avoid generic phrases like "policy violation"
    - Be actionable and specific
    """
}

def is_hosted_online():
    """Detect if app is running in a hosted environment (not local)"""
    try:
        # Check for common cloud hosting environment variables
        hosting_indicators = [
            os.environ.get('STREAMLIT_SHARING'),  # Streamlit Cloud
            os.environ.get('DYNO'),  # Heroku
            os.environ.get('WEBSITE_HOSTNAME'),  # Azure
            os.environ.get('VERCEL'),  # Vercel
            os.environ.get('RENDER'),  # Render
            os.environ.get('RAILWAY_ENVIRONMENT'),  # Railway
        ]
        return any(hosting_indicators)
    except:
        return False

def initialize_ui():
    st.set_page_config(
        page_title="RadioSport Claims Eval",
        page_icon="🏥",
        layout="wide",
        menu_items={
            'Report a Bug': "https://github.com/rkarikari/stem",
            'About': "RadioSport Insurance Claims Processing System"
        }
    )

    # Determine default API provider based on hosting environment
    default_api_provider = "Cloud" if is_hosted_online() else "Local"

    # Initialize session state
    session_defaults = {
        "claims_processed": 0,
        "claims_approved": 0,
        "claims_denied": 0,
        "claims_pending": 0,
        "total_approved_amount": 0,
        "total_denied_amount": 0,
        "messages": [],
        "ollama_models": [],
        "api_provider": default_api_provider,
        "ollama_host": DEFAULT_OLLAMA_HOST, 
        "api_keys": {"openrouter": ""},
        "selected_online_model": "mistralai/mistral-7b-instruct",
        "current_claim": None,
        "claim_history": [],
        "ai_enabled": True,
        "pdf_buffer": None,
        "override_attempts": 0,
        "last_override_attempt": None,
        "auto_switched_to_cloud": False
    }
    
    # Load saved data if available
    saved_data = load_data()
    if "override_attempts" not in st.session_state:
        st.session_state.override_attempts = 0
    if "last_override_attempt" not in st.session_state:
        st.session_state.last_override_attempt = None
    
    # Initialize ALL session state keys with defaults first
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Then override with saved data if available
    if saved_data:
        for key in session_defaults:
            if key in saved_data:
                st.session_state[key] = saved_data[key]

    # Enhanced CSS
    st.markdown("""
    <style>
    .claim-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .approved { border-left-color: #28a745 !important; background: #d4edda !important; }
    .denied { border-left-color: #dc3545 !important; background: #f8d7da !important; }
    .pending { border-left-color: #ffc107 !important; background: #fff3cd !important; }
    .investigate { border-left-color: #6f42c1 !important; background: #e2d9f3 !important; }
    
    .ai-analysis {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .decision-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.9em;
    }
    
    .badge-approved { background: #28a745; color: white; }
    .badge-denied { background: #dc3545; color: white; }
    .badge-pending { background: #ffc107; color: #212529; }
    .badge-investigate { background: #6f42c1; color: white; }
    
    .rejection-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .requirements-box {
        background: #d1ecf1;
        border-left: 4px solid #0dcaf0;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .conversion-success {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤖RadioSport Insurance Claims Eval")
    st.markdown("*Intelligent claim processing with advanced AI analysis*")

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_ollama_models_cached():
    try:
        # Use session state instead of constant
        host = st.session_state.ollama_host
        response = requests.get(f"{host}/api/tags", timeout=30)
        response.raise_for_status()
        
        data = response.json()
        models = [model_info.get("name", "") for model_info in data.get("models", [])]
        
        # Filter out embedding models
        embedding_prefixes = ("nomic-embed", "all-minilm", "mxbai-embed", "bge-", "gte-", "e5-")
        return [model for model in models if model and not any(model.lower().startswith(prefix) for prefix in embedding_prefixes)]
        
    except requests.ConnectionError:
        # Only show error if running locally
        if not is_hosted_online():
            st.error(f"Ollama server unavailable at {st.session_state.ollama_host}")
        return []
    except Exception as e:
        # Only show error if running locally
        if not is_hosted_online():
            st.error(f"Error fetching Ollama models: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_free_openrouter_models():
    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        response.raise_for_status()
        
        models_data = response.json()
        free_models = []
        
        for model in models_data.get('data', []):
            pricing = model.get('pricing', {})
            if float(pricing.get('prompt', '0')) == 0 and float(pricing.get('completion', '0')) == 0:
                free_models.append(model)
        
        return sorted(free_models, key=lambda x: x.get('id', ''))
        
    except Exception as e:
        st.error(f"Error loading OpenRouter models: {str(e)}")
        return []

def get_api_key(provider: str) -> str:
    try:
        if provider.lower() == "openrouter":
            try:
                return st.secrets["openrouter"]["api_key"]
            except:
                try:
                    return st.secrets["general"]["OPENROUTER_API_KEY"]
                except:
                    pass
    except:
        pass
    return ""

def call_ai_api(messages, model, api_key=None):
    """Enhanced AI API call with better error handling"""
    if st.session_state.api_provider == "Local":
        try:
            # Use session state instead of constant
            host = st.session_state.ollama_host
            response = requests.post(
                f"{host}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 2048
                    }
                },
                timeout=900
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            st.error(f"Ollama API error: {str(e)}")
            return None
    else:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/insurance-ai/claims-evaluator",
                "X-Title": "AI Insurance Claims Evaluator"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"OpenRouter API error: {str(e)}")
            return None

def get_security_code():
    try:
        # First try to get from secrets
        return st.secrets["security"]["override_code"]
    except:
        try:
            # Then try environment variables
            return os.environ.get("CLAIM_OVERRIDE_CODE", "DEFAULTCODE123")
        except:
            return "DEFAULTCODE123"  # Fallback for demo purposes

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

# Enhanced PDF text extraction with pdfplumber fallback
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using PyMuPDF first and pdfplumber as fallback"""
    text_content = ""
    try:
        # Reset file pointer to beginning
        pdf_file.seek(0)
        
        # First try with PyMuPDF for speed
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in doc:
                text_content += page.get_text() + "\n"
            # If we got text, return it
            if text_content.strip():
                return text_content
        except Exception as e:
            st.warning(f"PyMuPDF extraction warning: {str(e)}")
        
        # If PyMuPDF failed or returned no text, try pdfplumber
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() or "" + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return text_content

# OCR function for image text extraction
def extract_text_from_image(image_bytes) -> str:
    """Extract text from images using OCR"""
    try:
        import pytesseract
        from PIL import Image
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(image)
        return text.strip()
    except ImportError:
        st.warning("OCR capabilities not available (pytesseract not installed)")
        return ""
    except Exception as e:
        st.error(f"OCR processing error: {str(e)}")
        return ""

def extract_images_from_pdf(pdf_file) -> List[Dict]:
    """Extract images from PDF files using PyMuPDF"""
    import fitz  # PyMuPDF
    images = []
    try:
        # Reset file pointer to beginning
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    images.append({
                        'page': page_num + 1,
                        'image_num': img_index + 1,
                        'data': image_bytes,
                        'format': base_image["ext"]
                    })
                except Exception as e:
                    st.warning(f"Could not extract image {img_index + 1} from page {page_num + 1}: {str(e)}")
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
    return images

# Update the process_image_file function
def process_image_file(image_file) -> Dict:
    """Process uploaded image files"""
    try:
        # Read image using PIL
        from PIL import Image
        image = Image.open(image_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image info
        width, height = image.size
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        
        return {
            'filename': image_file.name,
            'width': width,
            'height': height,
            'format': 'png',
            'size': len(img_data),
            'data': img_data,
            'image_obj': image
        }
    except Exception as e:
        st.error(f"Error processing image {image_file.name}: {str(e)}")
        return None

def analyze_document_with_vision(document_content: str, images: List[Dict], claim_context: Dict, selected_model: str) -> str:
    """Analyze documents using vision-capable models"""
    
    # Get claim type specific requirements
    claim_type = claim_context.get('type', 'Unknown').lower()
    required_docs = CLAIM_RULES.get(claim_type, {}).get('required_docs', [])
    required_docs_str = ", ".join(required_docs) if required_docs else "None specified"
    
    # Vision-enhanced prompts
    vision_prompt = f"""
    You are an expert insurance claims adjuster with advanced document analysis capabilities.
    
    CLAIM CONTEXT:
    - Claim Type: {claim_context.get('type', 'Unknown')}
    - Claim Amount: ${claim_context.get('amount', 0):,}
    - Incident Date: {claim_context.get('incident_date', 'Unknown')}
    - Policy Number: {claim_context.get('policy_number', 'Unknown')}
    - REQUIRED DOCUMENTS: {required_docs_str}
    
    DOCUMENT TEXT CONTENT:
    {document_content}
    
    ANALYSIS INSTRUCTIONS:
    1. Extract all key facts, dates, locations, and parties involved
    2. Identify any inconsistencies or red flags
    3. Assess damage descriptions and cost estimates
    4. Evaluate supporting evidence quality
    5. Check for completeness of information
    6. Only note missing documents if they are in the REQUIRED DOCUMENTS list
    7. Do NOT mention documents that are not required for this claim type
    
    For images provided:
    - Analyze damage patterns and severity
    - Verify consistency with written descriptions
    - Assess authenticity and timing of photos
    - Identify any staged or suspicious elements
    - Extract text from images if visible
    
    Provide a comprehensive analysis with:
    - Key findings summary
    - Risk assessment (Low/Medium/High)
    - Fraud indicators (if any)
    - Recommendations for claim processing
    """
    
    # Prepare messages for API call
    messages = [{"role": "user", "content": vision_prompt}]
    
    # Add image analysis if images are present
    if images:
        image_analysis_prompt = f"""
        ADDITIONAL IMAGE ANALYSIS:
        
        I have {len(images)} image(s) related to this claim. Please analyze each image for:
        
        1. DAMAGE ASSESSMENT:
        - Type and extent of damage visible
        - Consistency with claim description
        - Estimated repair complexity
        
        2. AUTHENTICITY VERIFICATION:
        - Photo quality and metadata indicators
        - Lighting and environmental consistency
        - Signs of manipulation or staging
        
        3. SUPPORTING EVIDENCE:
        - Correlation with incident date/time
        - Location verification if possible
        - Additional context from surroundings
        
        4. TEXT EXTRACTION:
        - Any visible text, signs, or documents
        - License plates, addresses, or identifiers
        - Timestamps or date information
        
        Provide specific findings for each image and overall assessment.
        """
        
        messages.append({"role": "user", "content": image_analysis_prompt})

    # Add OCR text to analysis
    if claim_context.get('ocr_text'):
        ocr_prompt = f"""
        OCR EXTRACTED TEXT FROM IMAGES:
        {" ".join(claim_context['ocr_text'])[:5000]}
        """
        messages.append({"role": "user", "content": ocr_prompt})

    # Call AI API
    api_key = None
    if st.session_state.api_provider == "Cloud":
        api_key = st.session_state.api_keys.get("openrouter")
    
    analysis_result = call_ai_api(messages, selected_model, api_key)
    
    return analysis_result if analysis_result else "Document analysis unavailable"

def process_uploaded_documents(uploaded_files: List, claim_context: Dict, selected_model: str) -> Dict:
    """Process all uploaded documents comprehensively with OCR and parallel processing"""
    processed_documents = {
        'text_content': [],
        'images': [],
        'summary': {},
        'ocr_text': []  # New field for OCR extracted text
    }
    
    # Process files in parallel for efficiency
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_file(file):
        file_info = {
            'filename': file.name,
            'type': file.type,
            'size': file.size,
            'content': None,
            'images': [],
            'ocr_text': None,
            'analysis': None
        }
        
        try:
            # Process based on file type
            if file.type == 'application/pdf':
                # Extract text from PDF
                text_content = extract_text_from_pdf(file)
                file_info['content'] = text_content
                
                # Extract images from PDF
                file.seek(0)
                pdf_images = extract_images_from_pdf(file)
                file_info['images'] = pdf_images
                
                # Extract text from PDF images using OCR
                for img in pdf_images:
                    ocr_text = extract_text_from_image(img['data'])
                    if ocr_text:
                        processed_documents['ocr_text'].append(ocr_text)
                
            elif file.type in ['image/jpeg', 'image/png', 'image/jpg']:
                # Process image file
                image_data = process_image_file(file)
                if image_data:
                    file_info['images'] = [image_data]
                    
                    # Extract text from image using OCR
                    ocr_text = extract_text_from_image(image_data['data'])
                    if ocr_text:
                        file_info['ocr_text'] = ocr_text
                        processed_documents['ocr_text'].append(ocr_text)
            
            elif file.type in ['text/plain', 'text/csv']:
                # Process text files
                text_content = str(file.read(), 'utf-8')
                file_info['content'] = text_content
            
            return file_info
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            file_info['error'] = str(e)
            return file_info
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file) for file in uploaded_files]
        for future in as_completed(futures):
            file_info = future.result()
            if file_info.get('content'):
                processed_documents['text_content'].append(file_info['content'])
            if file_info.get('images'):
                processed_documents['images'].extend(file_info['images'])
            if file_info.get('ocr_text'):
                processed_documents['ocr_text'].append(file_info['ocr_text'])
    
    # Generate comprehensive summary
    if st.session_state.ai_enabled and selected_model:
        # Combine all text sources for analysis
        all_text = "\n\n".join(
            processed_documents['text_content'] + 
            processed_documents['ocr_text']
        )
        
        summary_prompt = f"""
        Provide a comprehensive summary of all document analysis results:
        
        CLAIM DETAILS:
        - Type: {claim_context.get('type', 'Unknown')}
        - Amount: ${claim_context.get('amount', 0):,}
        - Files Processed: {len(uploaded_files)}
        
        DOCUMENT CONTENT:
        {all_text[:15000]}  # Truncate to avoid token limits
        
        SUMMARY REQUIRED:
        1. Overall document quality and completeness
        2. Key evidence supporting the claim
        3. Any red flags or inconsistencies found
        4. Recommendations for claim processing
        5. Confidence level in documentation (0-100%)
        """
        
        api_key = None
        if st.session_state.api_provider == "Cloud":
            api_key = st.session_state.api_keys.get("openrouter")
        
        summary = call_ai_api([{"role": "user", "content": summary_prompt}], selected_model, api_key)
        processed_documents['summary'] = summary if summary else "Summary generation failed"
    
    return processed_documents

# Enhanced intelligent claim evaluation with document processing
def intelligent_claim_evaluation(claim_data, selected_model, uploaded_files=None):
    """Enhanced AI-driven claim evaluation with comprehensive document analysis"""
    
    # Process uploaded documents if provided
    document_analysis = None
    if uploaded_files:
        document_analysis = process_uploaded_documents(
            uploaded_files, 
            claim_data, 
            selected_model if is_vision_model(selected_model) else None  # Only process images with vision models
        )
        
        # Add document insights to claim data
        claim_data['document_analysis'] = document_analysis
    
    # Enhanced risk assessment with document insights
    risk_prompt = f"""
    As an expert insurance claims adjuster, analyze this claim comprehensively:
    
    CLAIM DETAILS:
    - Type: {claim_data.get('type', 'Unknown')}
    - Amount: ${claim_data.get('amount', 0):,}
    - Description: {claim_data.get('description', 'No description')}
    - Incident Date: {claim_data.get('incident_date', 'Unknown')}
    - Policy Number: {claim_data.get('policy_number', 'Unknown')}
    
    DOCUMENT ANALYSIS RESULTS:
    {document_analysis['summary'] if document_analysis else 'No documents processed'}
    
    COMPREHENSIVE EVALUATION:
    1. Risk Level Assessment (Low/Medium/High)
    2. Fraud Probability (0-100%)
    3. Policy Compliance Check
    4. Documentation Quality (0-100%)
    5. Evidence Strength (0-100%)
    6. Recommended Action
    
    Consider all available information including document content, image analysis, and claim details.
    Provide detailed reasoning for each assessment.
    """
    
    api_key = None
    if st.session_state.api_provider == "Cloud":
        api_key = st.session_state.api_keys.get("openrouter")
    
    comprehensive_analysis = call_ai_api([{"role": "user", "content": risk_prompt}], selected_model, api_key)
    
    # Enhanced decision making
    if comprehensive_analysis:
        analysis_lower = comprehensive_analysis.lower()
        
        # More sophisticated decision logic
        if "high risk" in analysis_lower or "fraud" in analysis_lower:
            if "approve" in analysis_lower:
                status = "investigate"
                reason = "High risk detected - requires investigation before approval"
            else:
                status = "denied"
                reason = "High risk or fraud indicators detected"
        elif "approve" in analysis_lower and "deny" not in analysis_lower:
            status = "approved"
            reason = "Comprehensive analysis supports approval"
        elif "deny" in analysis_lower:
            status = "denied"
            reason = "Analysis indicates denial recommended"
        else:
            status = "pending"
            reason = "Requires additional review"
            
        # Extract primary rejection reason and approval requirements for denied claims
        primary_rejection = ""
        approval_requirements = ""
        policy_clause = ""
        
        if status == "denied":
            # Enhanced extraction of rejection reasons
            reason_patterns = [
                r"Primary Rejection Reason:\s*(.+)",
                r"Denial Reason:\s*(.+)",
                r"Claim rejected because\s*(.+)",
                r"Violation of policy clause\s*(.+)",
                r"Specific reason for denial:\s*(.+)"
            ]
            
            # First try to extract explicitly labeled reason
            for pattern in reason_patterns:
                match = re.search(pattern, comprehensive_analysis, re.IGNORECASE)
                if match:
                    primary_rejection = match.group(1).strip()
                    break
            
            # If no explicit reason found, try to extract from reasoning
            if not primary_rejection:
                # Look for key phrases indicating rejection reasons
                phrases = [
                    "does not comply with",
                    "violates policy section",
                    "excluded by clause",
                    "not covered under",
                    "outside policy period",
                    "ineligible due to",
                    "pre-existing condition"
                ]
                
                for phrase in phrases:
                    if phrase in analysis_lower:
                        # Extract the sentence containing the phrase
                        match = re.search(r"([^.]*?" + re.escape(phrase) + r"[^.]*\.)", comprehensive_analysis, re.IGNORECASE)
                        if match:
                            primary_rejection = match.group(1).strip()
                            break
                
                # Fallback to first sentence of reasoning if still not found
                if not primary_rejection:
                    match = re.search(r"Reasoning:\s*(.*?\.)", comprehensive_analysis, re.IGNORECASE)
                    if match:
                        primary_rejection = match.group(1).strip()
                    else:
                        primary_rejection = "Specific rejection reason not provided - manual review required"
            
            # Extract policy clause if available
            clause_match = re.search(r"Policy clause violated:\s*(.+)", comprehensive_analysis, re.IGNORECASE)
            if clause_match:
                policy_clause = clause_match.group(1).strip()
            else:
                # Try to find policy section references
                clause_match = re.search(r"Section\s+[\d\.]+", comprehensive_analysis)
                if clause_match:
                    policy_clause = clause_match.group(0)
                else:
                    policy_clause = "Not specified"
            
            # Extract approval requirements
            req_match = re.search(r"Changes Needed for Approval:\s*(.+)", comprehensive_analysis, re.IGNORECASE)
            if req_match:
                approval_requirements = req_match.group(1).strip()
            else:
                # If not explicitly labeled, try to extract requirements from analysis
                requirements_match = re.search(r"Requirements:\s*(.+)", comprehensive_analysis, re.IGNORECASE)
                if requirements_match:
                    approval_requirements = requirements_match.group(1).strip()
                else:
                    # Generate requirements based on rejection reason
                    if "document" in primary_rejection.lower():
                        approval_requirements = "Submit missing or corrected documentation"
                    elif "policy" in primary_rejection.lower():
                        approval_requirements = "Provide evidence of policy coverage"
                    else:
                        approval_requirements = "Submit additional documentation or correct policy violations"
        
        # Return AI details
        ai_details = {
            "comprehensive_analysis": comprehensive_analysis,
            "document_analysis": document_analysis,
            "confidence_score": extract_confidence_score(comprehensive_analysis),
            "primary_rejection": primary_rejection if status == "denied" else "",
            "approval_requirements": approval_requirements if status == "denied" else "",
            "policy_clause": policy_clause if status == "denied" else ""
        }
        
        return status, reason, ai_details
    
    else:
        return basic_rule_evaluation(claim_data)

def clean_claim_for_storage(claim_data):
    """Remove non-serializable elements from claim data before storage"""
    # Create a copy to avoid modifying original
    cleaned = claim_data.copy()
    
    # Clean uploaded_files
    if 'uploaded_files' in cleaned:
        for file_info in cleaned['uploaded_files']:
            if 'image_obj' in file_info:
                del file_info['image_obj']
            if 'data' in file_info:
                del file_info['data']
    
    # Clean document_analysis
    if 'document_analysis' in cleaned:
        doc_analysis = cleaned['document_analysis']
        if 'images' in doc_analysis:
            for image_info in doc_analysis['images']:
                if 'image_obj' in image_info:
                    del image_info['image_obj']
                if 'data' in image_info:
                    del image_info['data']
    
    return cleaned

def basic_rule_evaluation(claim_data):
    """Basic rule-based claim evaluation with rejection reasons"""
    claim_type = claim_data.get("type", "").lower()
    claim_amount = float(claim_data.get("amount", 0))
    ai_details = {}
    
    if claim_type not in CLAIM_RULES:
        return "pending", "Unknown claim type - requires manual review", ai_details
    
    rules = CLAIM_RULES[claim_type]
    
    # Check hard deny conditions
    for condition in rules["deny_conditions"]:
        if claim_data.get(condition, False):
            # Provide detailed rejection reason and requirements
            primary_rejection = f"{condition.replace('_', ' ').title()} detected"
            
            # Add specific messages for common conditions
            if condition == "no_coverage":
                primary_rejection = "Policy does not cover this type of claim"
            elif condition == "fraud_confirmed":
                primary_rejection = "Evidence of fraud detected"
            elif condition == "expired_policy":
                primary_rejection = "Policy was expired at time of incident"
            
            approval_requirements = "Contact policy administrator for resolution options"
            
            ai_details = {
                "primary_rejection": primary_rejection,
                "approval_requirements": approval_requirements,
                "policy_clause": condition
            }
            
            return "denied", primary_rejection, ai_details
    
    # Check maximum amount
    if claim_amount > rules["max_amount"]:
        primary_rejection = f"Amount exceeds maximum limit of ${rules['max_amount']:,} for {claim_type} claims"
        approval_requirements = f"Reduce claim amount to below ${rules['max_amount']:,} or provide justification for exception"
        
        ai_details = {
            "primary_rejection": primary_rejection,
            "approval_requirements": approval_requirements,
            "policy_clause": "Maximum coverage limit"
        }
        
        return "denied", primary_rejection, ai_details
    
    # Check for auto-approval
    if claim_amount <= rules["auto_approve_threshold"]:
        return "approved", f"Auto-approved: Amount under threshold (${rules['auto_approve_threshold']:,})", ai_details
    
    # Check if AI review is needed
    if claim_amount >= rules["ai_review_threshold"]:
        return "investigate", f"High-value claim requires AI analysis (${rules['ai_review_threshold']:,}+)", ai_details
    
    return "pending", "Standard review required", ai_details

def extract_confidence_score(analysis_text: str) -> int:
    """Extract confidence score from analysis text"""
    import re
    
    # Look for confidence patterns
    confidence_patterns = [
        r'confidence[:\s]+(\d+)%',
        r'(\d+)%\s+confidence',
        r'confidence[:\s]+(\d+)'
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, analysis_text.lower())
        if match:
            return int(match.group(1))
    
    return 75  # Default confidence

# Enhanced PDF report generation with document insights
def generate_enhanced_pdf_report(claim_data):
    """Generate comprehensive PDF report with document analysis"""
    buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2c3e50')
    )
    story.append(Paragraph("COMPREHENSIVE INSURANCE CLAIM EVALUATION REPORT", title_style))
    story.append(Spacer(1, 12))
    
    # Claim Details Table
    claim_details = [
        ['Claim ID:', claim_data.get('id', 'N/A')],
        ['Claim Type:', claim_data.get('type', 'N/A').title()],
        ['Policy Number:', claim_data.get('policy_number', 'N/A')],
        ['Claimant Name:', claim_data.get('claimant_name', 'N/A')],
        ['Claim Amount:', f"₵{claim_data.get('amount', 0):,.2f}"],
        ['Incident Date:', claim_data.get('incident_date', 'N/A')],
        ['Submission Date:', claim_data.get('submitted_date', 'N/A')],
        ['Status:', claim_data.get('status', 'N/A').upper()],
    ]
    
    table = Table(claim_details, colWidths=[2*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Decision Summary
    decision_style = ParagraphStyle(
        'Decision',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#e74c3c') if claim_data.get('status') == 'denied' else colors.HexColor('#27ae60')
    )
    
    story.append(Paragraph(f"<b>DECISION: {claim_data.get('status', 'N/A').upper()}</b>", decision_style))
    story.append(Paragraph(f"<b>Reason:</b> {claim_data.get('reason', 'N/A')}", styles['Normal']))
    
    # Add rejection details for denied claims
    if claim_data.get('status') == 'denied':
        ai_details = claim_data.get('ai_details', {})
        
        # Primary Rejection Reason
        if ai_details.get('primary_rejection'):
            story.append(Spacer(1, 15))
            story.append(Paragraph("<b>PRIMARY REJECTION REASON</b>", styles['Heading3']))
            story.append(Paragraph(ai_details['primary_rejection'], styles['Normal']))
        
        # Policy Clause
        if ai_details.get('policy_clause'):
            story.append(Spacer(1, 5))
            story.append(Paragraph("<b>POLICY CLAUSE VIOLATED</b>", styles['Heading3']))
            story.append(Paragraph(ai_details['policy_clause'], styles['Normal']))
        
        # Approval Requirements
        if ai_details.get('approval_requirements'):
            story.append(Spacer(1, 10))
            story.append(Paragraph("<b>APPROVAL REQUIREMENTS</b>", styles['Heading3']))
            story.append(Paragraph(ai_details['approval_requirements'], styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Document Analysis Section
    if claim_data.get('document_analysis'):
        story.append(Paragraph("<b>DOCUMENT ANALYSIS SUMMARY</b>", styles['Heading2']))
        doc_analysis = claim_data['document_analysis']
        
        # Files processed
        files_processed = len(doc_analysis.get('text_content', [])) + len(doc_analysis.get('images', []))
        story.append(Paragraph(f"<b>Files Processed:</b> {files_processed}", styles['Normal']))
        
        # Document summary
        summary_text = doc_analysis.get('summary', 'No summary available')
        if len(summary_text) > 3000:
            summary_text = summary_text[:3000] + "..."
        story.append(Paragraph(f"<b>Analysis Summary:</b><br/>{summary_text}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Comprehensive AI Analysis
    if claim_data.get('ai_details', {}).get('comprehensive_analysis'):
        story.append(Paragraph("<b>COMPREHENSIVE AI ANALYSIS</b>", styles['Heading2']))
        ai_text = claim_data['ai_details']['comprehensive_analysis']
        if len(ai_text) > 4000:
            ai_text = ai_text[:4000] + "..."
        story.append(Paragraph(ai_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Claim Description
    story.append(Paragraph("<b>INCIDENT DESCRIPTION</b>", styles['Heading2']))
    story.append(Paragraph(claim_data.get('description', 'No description provided'), styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Supporting Documents
    if claim_data.get('uploaded_files'):
        story.append(Paragraph("<b>SUPPORTING DOCUMENTS</b>", styles['Heading2']))
        for file_info in claim_data['uploaded_files']:
            story.append(Paragraph(f"• {file_info.get('filename', 'Unknown')} ({file_info.get('type', 'Unknown type')})", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Confidence Score
    confidence = claim_data.get('ai_details', {}).get('confidence_score', 75)
    story.append(Paragraph(f"<b>ANALYSIS CONFIDENCE SCORE: {confidence}%</b>", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Enhanced AI Insurance Claims Evaluator v{APP_VERSION}", footer_style))
    
    pdf_doc.build(story)
    buffer.seek(0)
    return buffer

def create_backup():
    """Create a timestamped backup file"""
    # Create backup directory if it doesn't exist
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"insureX_backup_{timestamp}.json")
    
    try:
        shutil.copyfile(DATA_FILE, backup_file)
        return backup_file
    except Exception as e:
        st.error(f"Backup failed: {str(e)}")
        return None

def save_data():
    """Save application data to file"""
    # Clean all claim history before saving
    cleaned_history = [clean_claim_for_storage(claim) for claim in st.session_state.claim_history]
    
    data = {
        "claims_processed": st.session_state.claims_processed,
        "claims_approved": st.session_state.claims_approved,
        "claims_denied": st.session_state.claims_denied,
        "claims_pending": st.session_state.claims_pending,
        "total_approved_amount": st.session_state.total_approved_amount,
        "total_denied_amount": st.session_state.total_denied_amount,
        "claim_history": cleaned_history
    }
    
    try:
        # Handle case where DATA_FILE is in current directory
        dir_path = os.path.dirname(DATA_FILE)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def load_data():
    """Load application data from file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Attempt to recover by creating a new file
        try:
            with open(DATA_FILE, "w") as f:
                json.dump({}, f)
            st.info("Created new data file due to load error")
        except:
            pass
    return None

def create_backup():
    """Create a timestamped backup file"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"insureX_backup_{timestamp}.json")
    
    try:
        shutil.copyfile(DATA_FILE, backup_file)
        return backup_file
    except Exception as e:
        st.error(f"Backup failed: {str(e)}")
        return None


def main():
    initialize_ui()
    
    # Sidebar
    with st.sidebar:
        st.header("🧟 RadioSport Insurance")
        st.caption(f"Version {APP_VERSION}")
        
        # Add this expander for AI server settings
        with st.expander("🔧 AI Server Settings"):
            new_host = st.text_input(
                "Ollama Server URL:", 
                value=st.session_state.ollama_host,
                help="Change if your Ollama server is running elsewhere"
            )
            
            if new_host != st.session_state.ollama_host:
                st.session_state.ollama_host = new_host
                st.success(f"Server updated to: {new_host}")
                get_ollama_models_cached.clear()  # Clear model cache
            
            if st.button("Reset to Default"):
                st.session_state.ollama_host = DEFAULT_OLLAMA_HOST
                st.success(f"Reset to default: {DEFAULT_OLLAMA_HOST}")
                get_ollama_models_cached.clear()
        
        # API Provider
        # Get current default based on environment
        current_default = "Cloud" if is_hosted_online() else "Local"
        api_provider = st.radio(
            "AI Provider:", 
            ["Local", "Cloud"], 
            key="api_provider",
            index=1 if current_default == "Cloud" else 0
        )
        
        # AI Toggle
        st.session_state.ai_enabled = st.checkbox("Enable AI Analysis", value=True)
        
        # Model Selection
        if api_provider == "Local":
            if st.button("🔄 Refresh Models"):
                get_ollama_models_cached.clear()
                st.rerun()
            
            ollama_models = get_ollama_models_cached()
            if ollama_models:
                default_idx = ollama_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in ollama_models else 0
                selected_model = st.selectbox("Local Model:", ollama_models, index=default_idx)
            else:
                # Only show warning if running locally
                if not is_hosted_online():
                    st.warning("No local models available")
                else:
                    st.info("💡 Local models unavailable. Please switch to Cloud provider above.")
                selected_model = None
                
        # Auto-switch to Cloud if hosted online and Local has no models
        if api_provider == "Local" and is_hosted_online() and not ollama_models:
            if "auto_switched_to_cloud" not in st.session_state:
                st.session_state.api_provider = "Cloud"
                st.session_state.auto_switched_to_cloud = True
                st.rerun()        
                
        else:
            api_key = get_api_key("openrouter")
            if api_key:
                st.session_state.api_keys["openrouter"] = api_key
                st.success("🔑 API Key loaded from secrets")
            else:
                st.session_state.api_keys["openrouter"] = st.text_input("OpenRouter API Key:", type="password")
            
            openrouter_models = get_free_openrouter_models()
            if openrouter_models:
                model_names = [f"{m.get('name', m['id'])} (Free)" for m in openrouter_models]
                selected_idx = st.selectbox("Cloud Model:", range(len(model_names)), format_func=lambda x: model_names[x])
                selected_model = openrouter_models[selected_idx]['id']
            else:
                selected_model = None
        
        st.divider()
        
        # Statistics
        st.markdown("## 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Claims", st.session_state.claims_processed)
            st.metric("Approved", st.session_state.claims_approved, delta=f"${st.session_state.total_approved_amount:,.0f}")
        with col2:
            st.metric("Denied", st.session_state.claims_denied, delta=f"${st.session_state.total_denied_amount:,.0f}")
            st.metric("Pending", st.session_state.claims_pending)
        
        if st.session_state.claims_processed > 0:
            approval_rate = (st.session_state.claims_approved / st.session_state.claims_processed) * 100
            st.progress(approval_rate / 100)
            st.caption(f"Approval Rate: {approval_rate:.1f}%")
        
        st.divider()
        
        # Inside the sidebar clear data button:
        if st.button("🗑️ Clear All Data", type="secondary"):
            for key in ["claims_processed", "claims_approved", "claims_denied", "claims_pending", 
                       "total_approved_amount", "total_denied_amount", "claim_history"]:
                st.session_state[key] = 0 if key.endswith("_amount") or key.endswith("_processed") or key.endswith("_approved") or key.endswith("_denied") or key.endswith("_pending") else []
            
            # Save empty state
            save_data()
            st.rerun()

    # Main tabs definition:
    tab1, tab2, tab3, tab4 = st.tabs(["📝 New Claim", "📊 Analytics", "📋 History", "💾 Backup"])
    
    with tab1:
        st.header("Submit New Insurance Claim")
        
        with st.form("claim_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                claim_type = st.selectbox("Claim Type:", ["Auto", "Health", "Property", "Life"])
                policy_number = st.text_input("Policy Number:", placeholder="POL-2024-001")
                claim_amount = st.number_input("Claim Amount (₵):", min_value=0.0, format="%.2f", step=100.0)
                incident_date = st.date_input("Incident Date:")
                
            with col2:
                claimant_name = st.text_input("Claimant Name:", placeholder="John Doe")
                claimant_phone = st.text_input("Phone:", placeholder="+1-555-0123")
                claimant_email = st.text_input("Email:", placeholder="john.doe@email.com")
                description = st.text_area("Incident Description:", height=100, placeholder="Detailed description of the incident...")
            
            st.subheader("Supporting Documents")
            
            # File upload widgets
            col1, col2 = st.columns(2)
            
            with col1:
                incident_report_file = st.file_uploader("Incident Report", type=['pdf', 'doc', 'docx', 'txt'], key="incident_report")
                photos_files = st.file_uploader("Photos/Evidence", type=['jpg', 'jpeg', 'png', 'pdf'], accept_multiple_files=True, key="photos")
                estimates_file = st.file_uploader("Repair Estimates", type=['pdf', 'doc', 'docx', 'txt'], key="estimates")
                
            with col2:
                medical_records_file = st.file_uploader("Medical Records", type=['pdf', 'doc', 'docx'], key="medical_records")
                police_report_file = st.file_uploader("Police Report", type=['pdf', 'doc', 'docx', 'txt'], key="police_report")
                receipts_files = st.file_uploader("Receipts", type=['pdf', 'jpg', 'jpeg', 'png'], accept_multiple_files=True, key="receipts")
            
            # Document checklist display
            st.markdown("**Uploaded Documents:**")
            uploaded_docs = []
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if incident_report_file:
                    st.success("✅ Incident Report")
                    uploaded_docs.append("incident_report")
                else:
                    st.info("📄 Incident Report")
                
                if photos_files:
                    st.success(f"✅ Photos ({len(photos_files)} files)")
                    uploaded_docs.append("photos")
                else:
                    st.info("📷 Photos/Evidence")
            
            with col2:
                if estimates_file:
                    st.success("✅ Repair Estimates")
                    uploaded_docs.append("estimate")
                else:
                    st.info("💰 Repair Estimates")
                
                if medical_records_file:
                    st.success("✅ Medical Records")
                    uploaded_docs.append("medical_report")
                else:
                    st.info("🏥 Medical Records")
            
            with col3:
                if police_report_file:
                    st.success("✅ Police Report")
                    uploaded_docs.append("police_report")
                else:
                    st.info("🚔 Police Report")
                
                if receipts_files:
                    st.success(f"✅ Receipts ({len(receipts_files)} files)")
                    uploaded_docs.append("receipts")
                else:
                    st.info("🧾 Receipts")
            
            # Show upload summary
            if uploaded_docs:
                st.success(f"📎 {len(uploaded_docs)} document type(s) uploaded")
            else:
                st.warning("⚠️ No documents uploaded yet")
            
            submitted = st.form_submit_button("🚀 Submit Claim", type="primary")
            
            if submitted:
                if not all([claim_type, policy_number, claim_amount, claimant_name, description]):
                    st.error("❌ Please fill in all required fields")
                elif claim_amount <= 0:
                    st.error("❌ Claim amount must be positive")
                else:
                    # Collect all uploaded files
                    all_uploaded_files = []
                    
                    # Process each file type
                    file_mappings = {
                        'incident_report': incident_report_file,
                        'photos': photos_files,
                        'estimates': estimates_file,
                        'medical_records': medical_records_file,
                        'police_report': police_report_file,
                        'receipts': receipts_files
                    }
                    
                    processed_files = []
                    
                    for file_type, files in file_mappings.items():
                        if files:
                            if isinstance(files, list):
                                for file in files:
                                    all_uploaded_files.append(file)
                                    processed_files.append({
                                        "type": file_type,
                                        "filename": file.name,
                                        "size": file.size,
                                        "mime_type": file.type
                                    })
                            else:
                                all_uploaded_files.append(files)
                                processed_files.append({
                                    "type": file_type,
                                    "filename": files.name,
                                    "size": files.size,
                                    "mime_type": files.type
                                })
                    
                    # Prepare claim data
                    claim_data = {
                        "id": f"CLM-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
                        "type": claim_type.lower(),
                        "policy_number": policy_number,
                        "amount": claim_amount,
                        "incident_date": str(incident_date),
                        "claimant_name": claimant_name,
                        "claimant_phone": claimant_phone,
                        "claimant_email": claimant_email,
                        "description": description,
                        "documents": uploaded_docs,
                        "uploaded_files": processed_files,
                        "submitted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Enhanced AI-powered evaluation with document processing
                    with st.spinner("🤖 AI analyzing claim and processing documents..."):
                        
                        # Show processing status
                        status_container = st.empty()
                        
                        # Check for auto-approval threshold first
                        rules = CLAIM_RULES.get(claim_data['type'], {})
                        auto_threshold = rules.get("auto_approve_threshold", 0)
                        
                        if claim_amount <= auto_threshold and auto_threshold > 0:
                            status_container.success("✅ Auto-approving claim under threshold")
                            status = "approved"
                            reason = f"Auto-approved: Amount under threshold (₵{auto_threshold:,})"
                            ai_details = {}
                        else:
                            if st.session_state.ai_enabled and selected_model and all_uploaded_files:
                                status_container.info("📄 Processing uploaded documents...")
                                
                                # Use enhanced evaluation with document processing
                                status, reason, ai_details = intelligent_claim_evaluation(
                                    claim_data, 
                                    selected_model, 
                                    all_uploaded_files
                                )
                                
                            elif st.session_state.ai_enabled and selected_model:
                                status_container.info("🤖 AI analyzing claim...")
                                
                                # Use standard evaluation without document processing
                                status, reason, ai_details = intelligent_claim_evaluation(
                                    claim_data, 
                                    selected_model
                                )
                            
                            else:
                                status_container.info("📋 Using rule-based evaluation...")
                                status, reason, ai_details = basic_rule_evaluation(claim_data)
                        
                        status_container.empty()
                        
                        claim_data["status"] = status
                        claim_data["reason"] = reason
                        claim_data["ai_details"] = ai_details
                        
                        # Update statistics
                        cleaned_claim = clean_claim_for_storage(claim_data)

                        # Update statistics
                        st.session_state.claims_processed += 1
                        if status == "approved":
                            st.session_state.claims_approved += 1
                            st.session_state.total_approved_amount += claim_amount
                        elif status == "denied":
                            st.session_state.claims_denied += 1
                            st.session_state.total_denied_amount += claim_amount
                        else:
                            st.session_state.claims_pending += 1

                        # Store cleaned claim data
                        st.session_state.claim_history.append(cleaned_claim)
                        st.session_state.current_claim = cleaned_claim
                    
                    # Display results
                    status_emoji = {"approved": "✅", "denied": "❌", "pending": "⏳", "investigate": "🔍"}
                    status_color = {"approved": "success", "denied": "error", "pending": "warning", "investigate": "info"}
                    
                    getattr(st, status_color.get(status, "info"))(
                        f"{status_emoji.get(status, '📋')} **Claim {claim_data['id']}** - {status.upper()}\n\n{reason}"
                    )
                    
                    # Enhanced AI Analysis display
                    if claim_data.get("ai_details", {}).get("comprehensive_analysis"):
                        with st.expander("🤖 Comprehensive AI Analysis", expanded=True):
                            st.markdown(f"""
                            <div class="ai-analysis">
                                <h4>🧠 AI Decision Analysis</h4>
                                <p>{claim_data['ai_details']['comprehensive_analysis']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show confidence score
                            confidence = claim_data['ai_details'].get('confidence_score', 75)
                            st.metric("Analysis Confidence", f"{confidence}%")
                    
                    # Document Processing Results
                    if claim_data.get("document_analysis"):
                        with st.expander("📄 Document Analysis Results", expanded=True):
                            doc_analysis = claim_data["document_analysis"]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Files Processed", 
                                    len(doc_analysis.get('text_content', [])) + len(doc_analysis.get('images', [])))
                                st.metric("Images Analyzed", len(doc_analysis.get('images', [])))
                            
                            with col2:
                                st.metric("Text Documents", len(doc_analysis.get('text_content', [])))
                                st.metric("OCR Extractions", len(doc_analysis.get('ocr_text', [])))
                            
                            # Show document summary
                            st.subheader("Comprehensive Document Summary")
                            st.write(doc_analysis.get('summary', 'No summary available'))
                    
                    # Display rejection details for denied claims
                    if status == "denied":
                        ai_details = claim_data.get("ai_details", {})
                        
                        if ai_details.get("primary_rejection"):
                            st.markdown(f"""
                            <div class="rejection-box">
                                <h4>❌ Primary Rejection Reason</h4>
                                <p>{ai_details['primary_rejection']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if ai_details.get("policy_clause"):
                            st.markdown(f"""
                            <div class="rejection-box">
                                <h4>📜 Policy Clause Violated</h4>
                                <p>{ai_details['policy_clause']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if ai_details.get("approval_requirements"):
                            st.markdown(f"""
                            <div class="requirements-box">
                                <h4>✅ Approval Requirements</h4>
                                <p>{ai_details['approval_requirements']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Save data after processing
                    save_data()

        # Enhanced PDF Generation with document insights
        if st.session_state.get('current_claim'):
            st.subheader("📄 Enhanced Claim Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Generate Comprehensive PDF Report"):
                    pdf_buffer = generate_enhanced_pdf_report(st.session_state.current_claim)
                    st.download_button(
                        label="📥 Download Enhanced PDF Report",
                        data=pdf_buffer,
                        file_name=f"comprehensive_claim_report_{st.session_state.current_claim['id']}.pdf",
                        mime="application/pdf"
                    )
            
            with col2:
                if st.button("📧 Email Report"):
                    st.info("Email functionality would send the comprehensive report including document analysis")
            
            with col3:
                if st.button("🔍 Re-analyze Documents"):
                    if st.session_state.current_claim.get('document_analysis'):
                        st.success("Document re-analysis would be performed here")
                    else:
                        st.warning("No documents to re-analyze")
    
    with tab2:
        st.header("📊 Claims Analytics Dashboard")
        
        if st.session_state.claims_processed > 0:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Claims",
                    st.session_state.claims_processed,
                    delta=f"${st.session_state.total_approved_amount + st.session_state.total_denied_amount:,.0f}"
                )
            
            with col2:
                approval_rate = (st.session_state.claims_approved / st.session_state.claims_processed) * 100
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
            
            with col3:
                avg_claim_amount = np.mean([claim['amount'] for claim in st.session_state.claim_history])
                st.metric("Avg Claim Amount", f"₵{avg_claim_amount:,.0f}")
            
            with col4:
                fraud_rate = len([c for c in st.session_state.claim_history if 'fraud' in c.get('reason', '').lower()]) / st.session_state.claims_processed * 100
                st.metric("Fraud Detection Rate", f"{fraud_rate:.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Status pie chart
                fig, ax = plt.subplots(figsize=(10, 8))
                sizes = [
                    max(0, st.session_state.claims_approved),
                    max(0, st.session_state.claims_denied),
                    max(0, st.session_state.claims_pending)
                ]
                labels = ['Approved', 'Denied', 'Pending']
                colors_pie = ['#28a745', '#dc3545', '#ffc107']
                
                # Only show pie chart if there are claims
                if sum(sizes) > 0:
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Claims Status Distribution', fontsize=16, fontweight='bold')
                    
                    # Make text more readable
                    for text in texts:
                        text.set_fontsize(12)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    st.pyplot(fig)
                else:
                    st.info("No claims data to display")
            
            with col2:
                # Claim types bar chart
                if st.session_state.claim_history:
                    claim_types = [claim['type'].title() for claim in st.session_state.claim_history]
                    type_counts = pd.Series(claim_types).value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = ax.bar(type_counts.index, type_counts.values, color=['#007bff', '#6f42c1', '#fd7e14', '#20c997'])
                    ax.set_title('Claims by Type', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Claim Type')
                    ax.set_ylabel('Number of Claims')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontweight='bold')
                    
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Amount analysis
            st.subheader("💰 Financial Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Amount distribution
                amounts = [claim['amount'] for claim in st.session_state.claim_history]
                if amounts:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(amounts, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                    ax.set_title('Claim Amount Distribution', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Amount ($)')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            with col2:
                # Processing time simulation
                if len(st.session_state.claim_history) > 0:
                    processing_times = [random.randint(1, 10) for _ in st.session_state.claim_history]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(1, len(processing_times) + 1), processing_times, marker='o', linewidth=2, markersize=6)
                    ax.set_title('Processing Time Trend', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Claim Number')
                    ax.set_ylabel('Processing Time (Days)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            # Recent activity
            st.subheader("📈 Recent Activity")
            
            recent_claims = st.session_state.claim_history[-5:]
            for claim in reversed(recent_claims):
                status_class = claim['status']
                badge_class = f"badge-{status_class}"
                
                st.markdown(f"""
                <div class="claim-card {status_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4>{claim['id']} - {claim['claimant_name']}</h4>
                            <p><strong>${claim['amount']:,.2f}</strong> | {claim['type'].title()} | {claim['submitted_date']}</p>
                        </div>
                        <div>
                            <span class="decision-badge {badge_class}">{claim['status']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("📊 No claims data available yet. Submit some claims to see analytics!")
    
    with tab3:
        st.header("📋 Claim History & Management")
        
        if st.session_state.claim_history:
            # Search and filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_term = st.text_input("🔍 Search Claims:", placeholder="Claim ID, name, or policy number")
            
            with col2:
                status_filter = st.selectbox("Filter by Status:", ["All", "Approved", "Denied", "Pending", "Investigate"])
            
            with col3:
                type_filter = st.selectbox("Filter by Type:", ["All", "Auto", "Health", "Property", "Life"])
            
            # Filter claims
            filtered_claims = st.session_state.claim_history
            
            if search_term:
                filtered_claims = [
                    claim for claim in filtered_claims 
                    if search_term.lower() in claim['id'].lower() or 
                       search_term.lower() in claim['claimant_name'].lower() or
                       search_term.lower() in claim['policy_number'].lower()
                ]
            
            if status_filter != "All":
                filtered_claims = [claim for claim in filtered_claims if claim['status'] == status_filter.lower()]
            
            if type_filter != "All":
                filtered_claims = [claim for claim in filtered_claims if claim['type'] == type_filter.lower()]
            
            st.write(f"Showing {len(filtered_claims)} of {len(st.session_state.claim_history)} claims")
            
            # Display claims
            for claim in reversed(filtered_claims):
                status_class = claim['status']
                badge_class = f"badge-{status_class}"
                
                with st.container():
                    st.markdown(f"""
                    <div class="claim-card {status_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h4>{claim['id']} - {claim['claimant_name']}</h4>
                            <span class="decision-badge {badge_class}">{claim['status']}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div>
                                <p><strong>Type:</strong> {claim['type'].title()}</p>
                                <p><strong>Amount:</strong> ${claim['amount']:,.2f}</p>
                                <p><strong>Policy:</strong> {claim['policy_number']}</p>
                            </div>
                            <div>
                                <p><strong>Incident Date:</strong> {claim['incident_date']}</p>
                                <p><strong>Submitted:</strong> {claim['submitted_date']}</p>
                                <p><strong>Contact:</strong> {claim['claimant_phone']}</p>
                            </div>
                        </div>
                        <p><strong>Description:</strong> {claim['description']}</p>
                        <p><strong>Decision Reason:</strong> {claim['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expandable sections
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if claim.get('ai_details', {}).get('comprehensive_analysis'):
                            with st.expander("🤖 AI Analysis"):
                                st.write(claim['ai_details']['comprehensive_analysis'])
                                
                                # Show rejection details for denied claims
                                if claim['status'] == "denied":
                                    ai_details = claim.get('ai_details', {})
                                    if ai_details.get("primary_rejection"):
                                        st.markdown(f"""
                                        <div class="rejection-box">
                                            <h5>❌ Primary Rejection Reason</h5>
                                            <p>{ai_details['primary_rejection']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    if ai_details.get("policy_clause"):
                                        st.markdown(f"""
                                        <div class="rejection-box">
                                            <h5>📜 Policy Clause Violated</h5>
                                            <p>{ai_details['policy_clause']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    if ai_details.get("approval_requirements"):
                                        st.markdown(f"""
                                        <div class="requirements-box">
                                            <h5>✅ Approval Requirements</h5>
                                            <p>{ai_details['approval_requirements']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if claim.get('documents'):
                            with st.expander("📎 Documents"):
                                for doc in claim['documents']:
                                    st.write(f"• {doc}")
                    
                    with col3:
                        with st.expander("🔧 Actions"):
                            if st.button(f"📄 Generate PDF", key=f"pdf_{claim['id']}"):
                                pdf_buffer = generate_enhanced_pdf_report(claim)
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=pdf_buffer,
                                    file_name=f"claim_report_{claim['id']}.pdf",
                                    mime="application/pdf",
                                    key=f"download_{claim['id']}"
                                )
                            
                            if st.button(f"🔄 Re-evaluate", key=f"reeval_{claim['id']}"):
                                with st.spinner("Re-evaluating..."):
                                    # Get the index of this claim in history
                                    claim_idx = next((i for i, c in enumerate(st.session_state.claim_history) 
                                                     if c['id'] == claim['id']), -1)
                                    
                                    if claim_idx == -1:
                                        st.error("Claim not found in history")
                                        return
                                    
                                    # Get current claim data
                                    current_claim = st.session_state.claim_history[claim_idx]
                                    original_status = current_claim['status']
                                    
                                    if st.session_state.ai_enabled and selected_model:
                                        status, reason, ai_details = intelligent_claim_evaluation(current_claim, selected_model)
                                    else:
                                        status, reason, ai_details = basic_rule_evaluation(current_claim)
                                    
                                    # Update claim data
                                    current_claim['status'] = status
                                    current_claim['reason'] = reason
                                    current_claim['ai_details'] = ai_details
                                    current_claim['reevaluated_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    # Update statistics if status changed
                                    if original_status != status:
                                        # Handle conversion from denied to approved
                                        if original_status == "denied" and status == "approved":
                                            # Ensure counts are valid before updating
                                            if st.session_state.claims_denied > 0:
                                                st.session_state.claims_denied -= 1
                                            st.session_state.claims_approved += 1
                                            st.session_state.total_denied_amount = max(0, st.session_state.total_denied_amount - current_claim['amount'])
                                            st.session_state.total_approved_amount += current_claim['amount']
                                            st.success(f"✅ Claim converted from DENIED to APPROVED!")
                                        
                                        # Handle conversion from approved to denied
                                        elif original_status == "approved" and status == "denied":
                                            # Ensure counts are valid before updating
                                            if st.session_state.claims_approved > 0:
                                                st.session_state.claims_approved -= 1
                                            st.session_state.claims_denied += 1
                                            st.session_state.total_approved_amount = max(0, st.session_state.total_approved_amount - current_claim['amount'])
                                            st.session_state.total_denied_amount += current_claim['amount']
                                            st.error(f"❌ Claim converted from APPROVED to DENIED!")
                                        
                                        # Handle pending/investigate conversions
                                        elif original_status in ["pending", "investigate"]:
                                            # Ensure counts are valid before updating
                                            if st.session_state.claims_pending > 0:
                                                st.session_state.claims_pending -= 1
                                            if status == "approved":
                                                st.session_state.claims_approved += 1
                                                st.session_state.total_approved_amount += current_claim['amount']
                                            elif status == "denied":
                                                st.session_state.claims_denied += 1
                                                st.session_state.total_denied_amount += current_claim['amount']
                                        
                                        # Show conversion success message
                                        st.markdown(f"""
                                        <div class="conversion-success">
                                            <h4>🔄 Status Changed: {original_status.upper()} → {status.upper()}</h4>
                                            <p><strong>New Reason:</strong> {reason}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Save updated data
                                    save_data()
                                    st.rerun()
                    
#---7 tabs Override
                            # Add to the Actions expander in the History tab (tab3)
                            # Inside the "Actions" expander for each claim, after the re-evaluate button
                            if claim['status'] in ['pending', 'denied']:
                                st.markdown("---")
                                st.subheader("🔓 Security Override")
                                
                                # Rate limiting (max 3 attempts per minute)
                                if st.session_state.override_attempts >= 3 and \
                                   st.session_state.last_override_attempt and \
                                   (datetime.now() - st.session_state.last_override_attempt) < timedelta(minutes=1):
                                    st.error("⚠️ Too many failed attempts. Please wait 1 minute before trying again.")
                                else:
                                    security_code = st.text_input("Enter Security Code:", 
                                                                 type="password",
                                                                 key=f"security_{claim['id']}")
                                    
                                    if st.button("🔑 Override and Approve", 
                                                key=f"override_{claim['id']}",
                                                help="Approve this claim with security override"):
                                        if security_code == get_security_code():
                                            # Get the index of this claim in history
                                            claim_idx = next((i for i, c in enumerate(st.session_state.claim_history) 
                                                            if c['id'] == claim['id']), -1)
                                            
                                            if claim_idx == -1:
                                                st.error("Claim not found in history")
                                            else:
                                                # Get current claim data
                                                current_claim = st.session_state.claim_history[claim_idx]
                                                original_status = current_claim['status']
                                                
                                                # Update claim data
                                                current_claim['status'] = "approved"
                                                current_claim['reason'] = "Manually approved by security override"
                                                current_claim['overridden'] = True
                                                current_claim['override_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                
                                                # Update statistics
                                                if original_status == "denied":
                                                    st.session_state.claims_denied -= 1
                                                    st.session_state.total_denied_amount -= current_claim['amount']
                                                elif original_status == "pending":
                                                    st.session_state.claims_pending -= 1
                                                
                                                st.session_state.claims_approved += 1
                                                st.session_state.total_approved_amount += current_claim['amount']
                                                
                                                # Reset override attempts
                                                st.session_state.override_attempts = 0
                                                
                                                # Save updated data
                                                save_data()
                                                st.success("✅ Claim approved by security override!")
                                                st.balloons()
                                                time.sleep(1)
                                                st.rerun()
                                        else:
                                            st.session_state.override_attempts += 1
                                            st.session_state.last_override_attempt = datetime.now()
                                            st.error("❌ Invalid security code")
#--- Override end
                    
                    st.divider()
        
        else:
            st.info("📋 No claims submitted yet. Use the 'New Claim' tab to submit your first claim!")

# Data Backup and Restore tab
    with tab4:
        st.header("💾 Data Backup & Restore")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Backup Data")
            st.info("Create a backup of all claims data and analytics")
            if st.button("🔄 Create Backup", help="Save a timestamped backup file"):
                backup_file = create_backup()
                if backup_file:
                    st.success(f"✅ Backup created: {os.path.basename(backup_file)}")
                    with open(backup_file, "rb") as f:
                        st.download_button(
                            label="📥 Download Backup",
                            data=f,
                            file_name=os.path.basename(backup_file),
                            mime="application/json"
                        )
        
        with col2:
            st.subheader("Restore Data")
            st.warning("Restoring data will overwrite current information!")
            
            uploaded_file = st.file_uploader("Select backup file", type=["json"])
            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    
                    # Validate backup structure
                    required_keys = ["claims_processed", "claims_approved", "claims_denied", 
                                    "claims_pending", "total_approved_amount", 
                                    "total_denied_amount", "claim_history"]
                    
                    if all(key in data for key in required_keys):
                        st.success("✅ Valid backup file structure detected")
                        
                        if st.button("🚨 Restore from Backup", type="primary"):
                            for key in required_keys:
                                st.session_state[key] = data[key]
                            
                            save_data()
                            st.success("Data restored successfully!")
                            st.rerun()
                    else:
                        st.error("Invalid backup file format")
                except Exception as e:
                    st.error(f"Error processing backup file: {str(e)}")
        
        st.divider()
        st.subheader("Current Data Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Claims", st.session_state.claims_processed)
            st.metric("Data File Size", f"{os.path.getsize(DATA_FILE)/1024:.1f} KB" if os.path.exists(DATA_FILE) else "N/A")
        
        with col2:
            st.metric("Backups Available", len(os.listdir(BACKUP_DIR)) if os.path.exists(BACKUP_DIR) else 0)
            if st.button("🗑️ Delete All Backups", type="secondary"):
                if os.path.exists(BACKUP_DIR):
                    for file in os.listdir(BACKUP_DIR):
                        os.remove(os.path.join(BACKUP_DIR, file))
                    st.success("All backups deleted")
    
    # Footer
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666; padding: 20px;'>AI Insurance Claims Evaluator v{APP_VERSION} | Powered by Advanced AI Technology</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()