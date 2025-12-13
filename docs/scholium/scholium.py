import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import requests
import toml
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import hashlib
import uuid
from enum import Enum
import time
import io
from PIL import Image
import base64

# Version tracking
APP_VERSION = "1.5.0"  # AI Promotion

def initialize_ui():
    st.set_page_config(
        page_title="RadioSport Scholium",
        page_icon="🧟",
        layout="wide",
        menu_items={
            'Report a Bug': "https://github.com/rkarikari/RadioSport-chat",
            'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."
        }
    )

# Configuration and Data Models
class AIProvider(Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"

@dataclass
class Student:
    student_id: str
    name: str
    age: int
    grade: str
    admission_date: str
    gpa: float = 0.0
    attendance_rate: float = 100.0
    behavior_score: float = 100.0
    parent_contact: str = ""
    medical_info: str = ""
    photo: Optional[str] = None
    status: str = "active"
    stream_id: Optional[str] = None

@dataclass
class Teacher:
    teacher_id: str
    name: str
    subject: str
    department: str
    hire_date: str
    performance_score: float = 100.0
    salary: float = 0.0
    status: str = "active"
    is_class_teacher: bool = False

@dataclass
class Course:
    course_id: str
    name: str
    teacher_id: str
    grade_level: str
    credits: int
    schedule: str

@dataclass
class Stream:
    stream_id: str
    grade_level: str
    stream_type: str
    max_capacity: int = 36
    class_teacher_id: Optional[str] = None
    subject_teachers: Dict[str, str] = field(default_factory=dict)  # New: subject -> teacher_id

    def __post_init__(self):
        # Convert subject_teachers from JSON string if needed
        if isinstance(self.subject_teachers, str):
            try:
                self.subject_teachers = json.loads(self.subject_teachers)
            except json.JSONDecodeError:
                self.subject_teachers = {}

class SchoolAI:
    def __init__(self):
        provider_value = st.session_state.get('ai_provider', AIProvider.OLLAMA)
        self.provider_str = self._get_provider_string_direct(provider_value)
        
        if isinstance(provider_value, AIProvider):
            self.provider = provider_value
        else:
            self.provider = AIProvider.OLLAMA
            
        self.ollama_url = st.session_state.get('ollama_url', 'http://localhost:11434')
        self.openrouter_key = self._load_openrouter_key()
        self.model = st.session_state.get('ai_model', None)
        
        if 'openrouter_models' not in st.session_state:
            st.session_state.openrouter_models = []
        if 'ollama_models' not in st.session_state:
            st.session_state.ollama_models = []

    def _get_provider_string_direct(self, provider_value) -> str:
        """Direct mapping of provider to string - handles all cases"""
        try:
            provider_str = str(provider_value).lower()
            
            if 'ollama' in provider_str:
                return 'ollama'
            elif 'openrouter' in provider_str:
                return 'openrouter'
            else:
                return 'ollama'
        except Exception:
            return 'ollama'

    def _extract_provider_string(self, provider_enum) -> str:
        """Extract clean provider string from enum, handling various enum definitions"""
        try:
            if hasattr(provider_enum, 'value') and isinstance(provider_enum.value, str):
                value = provider_enum.value.lower()
                if value in ['ollama', 'openrouter']:
                    return value
            
            if hasattr(provider_enum, 'name'):
                name = provider_enum.name.lower()
                if name in ['ollama', 'openrouter']:
                    return name
            
            enum_str = str(provider_enum)
            if 'OLLAMA' in enum_str.upper():
                return 'ollama'
            elif 'OPENROUTER' in enum_str.upper():
                return 'openrouter'
            
            return 'ollama'
        except Exception:
            return 'ollama'
    
    def _clean_provider_string(self, provider_str: str) -> str:
        """Clean provider string from various formats"""
        try:
            provider_str = provider_str.lower()
            
            if 'ollama' in provider_str:
                return 'ollama'
            elif 'openrouter' in provider_str:
                return 'openrouter'
            
            return 'ollama'
        except Exception:
            return 'ollama'

    def _load_openrouter_key(D):
        'Load OpenRouter API key from Streamlit Cloud secrets or local secrets.toml'
        A='api_key'
        try:
            # For Streamlit Cloud (streamlit.app) - check secrets first
            if hasattr(st, 'secrets'):
                # Try nested format: [openrouter] api_key = "..."
                if 'openrouter' in st.secrets:
                    key = st.secrets.openrouter.get(A, '')
                    if key:
                        return key
                
                # Try direct format: OPENROUTER_API_KEY = "..."
                if 'OPENROUTER_API_KEY' in st.secrets:
                    key = st.secrets['OPENROUTER_API_KEY']
                    if key:
                        return key
            
            # For local development - read from file
            with open('.streamlit/secrets.toml', 'r') as B:
                C = toml.load(B)
                return C.get('openrouter', {}).get(A, '')
                
        except FileNotFoundError:
            # Local file doesn't exist, return empty (expected on Streamlit Cloud)
            return ''
        except Exception:
            return ''

    def get_openrouter_free_models(self) -> List[Dict]:
        """Dynamically load OpenRouter free models"""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get('data', [])
                free_models = [
                    model for model in models 
                    if model.get('pricing', {}).get('prompt', '0') == '0'
                ]
                st.session_state.openrouter_models = free_models
                return free_models
            else:
                return st.session_state.openrouter_models
        except Exception:
            return st.session_state.openrouter_models

    def get_ollama_models(self, ollama_url: str = None) -> List[str]:
        """Dynamically load available models from Ollama server"""
        url = ollama_url or self.ollama_url
        try:
            response = requests.get(f"{url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                model_names = [model['name'] for model in models_data]
                st.session_state.ollama_models = model_names
                return model_names
            else:
                return st.session_state.ollama_models
        except Exception:
            return st.session_state.ollama_models

    def sync_with_session_state(self):
        """Sync instance variables with session state properly"""
        try:
            if 'ai_provider' in st.session_state:
                provider_value = st.session_state.ai_provider
                self.provider_str = self._get_provider_string_direct(provider_value)
                
                if isinstance(provider_value, AIProvider):
                    self.provider = provider_value
                        
            if 'ollama_url' in st.session_state:
                self.ollama_url = st.session_state.ollama_url
                
            if 'ai_model' in st.session_state:
                self.model = st.session_state.ai_model
                
        except Exception:
            pass

    def get_configuration_summary(self) -> Dict[str, str]:
        """Get current configuration summary for display"""
        provider_display = self.provider_str.title() if self.provider_str else "Unknown"
        
        return {
            "provider": provider_display,
            "model": self.model or "Not selected",
            "server_url": self.ollama_url if self.provider_str == "ollama" else "N/A",
            "api_status": "Configured" if self.openrouter_key else "Not configured",
            "connection_status": self._test_connection()
        }

    def _test_connection(self) -> str:
        """Test connection to current provider"""
        try:
            if self.provider_str == "ollama":
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                return "✅ Connected" if response.status_code == 200 else "❌ Failed"
            elif self.provider_str == "openrouter":
                if not self.openrouter_key:
                    return "❌ No API Key"
                response = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {self.openrouter_key}"},
                    timeout=5
                )
                return "✅ Connected" if response.status_code == 200 else "❌ Failed"
            else:
                return "❌ Unknown provider"
        except Exception:
            return "❌ Connection Error"

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate AI response using selected provider"""
        try:
            provider_str_lower = str(self.provider_str).lower()
            
            if 'ollama' in provider_str_lower:
                return self._ollama_request(prompt, context)
            elif 'openrouter' in provider_str_lower:
                return self._openrouter_request(prompt, context)
            else:
                return f"Unknown provider: {self.provider_str}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _ollama_request(self, prompt: str, context: str) -> str:
        """Make request to local Ollama instance"""
        try:
            if not self.model:
                return "Ollama Error: No model selected"
            
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json().get('response', 'No response')
                return result.strip() if result else "No response"
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', f'HTTP {response.status_code}')
                    return f"Ollama Error: {error_msg}"
                except:
                    return f"Ollama Error: HTTP {response.status_code}"
                    
        except requests.exceptions.ConnectionError:
            return f"Ollama Error: Cannot connect to {self.ollama_url}"
        except requests.exceptions.Timeout:
            return "Ollama Error: Request timeout (120s)"
        except Exception as e:
            return f"Ollama Error: {str(e)}"

    def _openrouter_request(self, prompt: str, context: str) -> str:
        """Make request to OpenRouter API"""
        try:
            if not self.openrouter_key:
                return "OpenRouter Error: API key not configured"
            
            if not self.model:
                return "OpenRouter Error: No model selected"
                
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": full_prompt}]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                return result.strip() if result else "No response"
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
                    return f"OpenRouter Error: {error_msg}"
                except:
                    return f"OpenRouter Error: HTTP {response.status_code}"
                    
        except requests.exceptions.ConnectionError:
            return "OpenRouter Error: No internet connection"
        except requests.exceptions.Timeout:
            return "OpenRouter Error: Request timeout (60s)"
        except Exception as e:
            return f"OpenRouter Error: {str(e)}"

    def get_provider_enum(self) -> AIProvider:
        """Get provider as enum"""
        return self.provider
    
    def get_provider_string(self) -> str:
        """Get provider as string"""
        return self.provider_str
    
    def is_ollama(self) -> bool:
        """Check if current provider is Ollama"""
        return self.provider_str == "ollama"
    
    def is_openrouter(self) -> bool:
        """Check if current provider is OpenRouter"""
        return self.provider_str == "openrouter"

class DatabaseManager:
    def __init__(self, db_path="school_management.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    grade TEXT,
                    admission_date TEXT,
                    gpa REAL DEFAULT 0.0,
                    attendance_rate REAL DEFAULT 100.0,
                    behavior_score REAL DEFAULT 100.0,
                    parent_contact TEXT,
                    medical_info TEXT,
                    photo TEXT,
                    status TEXT DEFAULT 'active',
                    stream_id TEXT,
                    date_of_birth TEXT,
                    FOREIGN KEY (stream_id) REFERENCES streams (stream_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS teachers (
                    teacher_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    subject TEXT,
                    department TEXT,
                    hire_date TEXT,
                    performance_score REAL DEFAULT 100.0,
                    salary REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'active',
                    is_class_teacher BOOLEAN DEFAULT FALSE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS courses (
                    course_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    teacher_id TEXT,
                    grade_level TEXT,
                    credits INTEGER,
                    schedule TEXT,
                    FOREIGN KEY (teacher_id) REFERENCES teachers (teacher_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS grades (
                    grade_id TEXT PRIMARY KEY,
                    student_id TEXT,
                    course_id TEXT,
                    grade REAL,
                    semester TEXT,
                    year INTEGER,
                    FOREIGN KEY (student_id) REFERENCES students (student_id),
                    FOREIGN KEY (course_id) REFERENCES courses (course_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    attendance_id TEXT PRIMARY KEY,
                    student_id TEXT,
                    date TEXT,
                    status TEXT,
                    FOREIGN KEY (student_id) REFERENCES students (student_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS finance (
                    transaction_id TEXT PRIMARY KEY,
                    student_id TEXT,
                    amount REAL,
                    type TEXT,
                    description TEXT,
                    date TEXT,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (student_id) REFERENCES students (student_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS streams (
                    stream_id TEXT PRIMARY KEY,
                    grade_level TEXT NOT NULL,
                    stream_type TEXT NOT NULL,
                    max_capacity INTEGER DEFAULT 36,
                    class_teacher_id TEXT,
                    subject_teachers TEXT DEFAULT '{}',
                    FOREIGN KEY (class_teacher_id) REFERENCES teachers (teacher_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stream_config (
                    grade_level TEXT PRIMARY KEY,
                    stream_types TEXT NOT NULL,
                    max_capacity INTEGER DEFAULT 36,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()

    def add_student(self, student_data):
        """Add new student"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            columns = list(student_data.keys())
            placeholders = ['?' for _ in columns]
            values = list(student_data.values())
            
            query = f'''
                INSERT INTO students ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            '''
            
            cursor.execute(query, values)
            conn.commit()

    def update_student(self, student_data):
        """Update student information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            set_clauses = []
            values = []
            
            for key, value in student_data.items():
                if key != 'student_id':
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            
            values.append(student_data['student_id'])
            
            query = f'''
                UPDATE students 
                SET {', '.join(set_clauses)}
                WHERE student_id = ?
            '''
            
            cursor.execute(query, values)
            conn.commit()
            
            if cursor.rowcount == 0:
                raise Exception(f"Student not found: {student_data['student_id']}")

    def get_all_students(self):
        """Get all students"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students")
            return [dict(row) for row in cursor.fetchall()]

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute a query and return results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_update(self, query: str, params: tuple = ()) -> None:
        """Execute an update/insert query"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

class StreamManager:
    def __init__(self, db_manager, default_streams=None, default_max_capacity=36):
        self.db = db_manager
        self.ideal_class_size = 35
        self.default_max_capacity = default_max_capacity
        self.max_class_size = default_max_capacity  # Alias for backward compatibility
        
        # Default stream configuration - maintains backward compatibility
        if default_streams is None:
            self.default_streams = ['R', 'C', 'S']
        else:
            self.default_streams = default_streams
        
        # Initialize stream configuration table if it doesn't exist
        self._initialize_stream_config()
    
    def _initialize_stream_config(self):
        """Initialize stream configuration table for per-grade stream settings"""
        self.db.execute_update('''
            CREATE TABLE IF NOT EXISTS stream_config (
                grade_level TEXT PRIMARY KEY,
                stream_types TEXT NOT NULL,  -- JSON array of stream types
                max_capacity INTEGER DEFAULT 36,  -- Per-grade max capacity
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    def set_streams_for_grade(self, grade_level: str, stream_types: list, max_capacity: int = None):
        """Set custom stream types and max capacity for a specific grade level"""
        import json
        
        # Validate stream types
        if not stream_types or len(stream_types) == 0:
            raise ValueError("At least one stream type is required")
        
        # Use provided max_capacity or default
        if max_capacity is None:
            max_capacity = self.default_max_capacity
        
        # Validate max_capacity
        if max_capacity <= 0:
            raise ValueError("Max capacity must be greater than 0")
        
        # Remove duplicates while preserving order
        stream_types = list(dict.fromkeys(stream_types))
        
        # Store configuration
        query = '''
            INSERT INTO stream_config (grade_level, stream_types, max_capacity)
            VALUES (?, ?, ?)
            ON CONFLICT(grade_level) DO UPDATE SET 
                stream_types = ?, 
                max_capacity = ?
        '''
        stream_types_json = json.dumps(stream_types)
        self.db.execute_update(query, (grade_level, stream_types_json, max_capacity, 
                                     stream_types_json, max_capacity))
    
    def get_streams_config_for_grade(self, grade_level: str) -> dict:
        """Get stream configuration for a grade level"""
        import json
        
        query = "SELECT stream_types, max_capacity FROM stream_config WHERE grade_level = ?"
        result = self.db.execute_query(query, (grade_level,))
        
        if result:
            stream_types = json.loads(result[0][0])
            max_capacity = result[0][1] if result[0][1] is not None else self.default_max_capacity
            return {
                'stream_types': stream_types,
                'max_capacity': max_capacity
            }
        else:
            # Return default configuration if no custom config exists
            return {
                'stream_types': self.default_streams.copy(),
                'max_capacity': self.default_max_capacity
            }
    
    def get_max_capacity_for_grade(self, grade_level: str) -> int:
        """Get max capacity setting for a specific grade level"""
        config = self.get_streams_config_for_grade(grade_level)
        return config['max_capacity']
    
    def set_max_capacity_for_grade(self, grade_level: str, max_capacity: int):
        """Set max capacity for a specific grade level (keeps existing stream types)"""
        if max_capacity <= 0:
            raise ValueError("Max capacity must be greater than 0")
        
        # Get current config to preserve stream types
        current_config = self.get_streams_config_for_grade(grade_level)
        
        # Update with new max capacity
        self.set_streams_for_grade(grade_level, current_config['stream_types'], max_capacity)
        
        # Update existing streams in this grade
        self._update_existing_streams_capacity(grade_level, max_capacity)
    
    def set_global_max_capacity(self, max_capacity: int):
        """Set default max capacity for all new streams"""
        if max_capacity <= 0:
            raise ValueError("Max capacity must be greater than 0")
        
        self.default_max_capacity = max_capacity
        self.max_class_size = max_capacity  # Update alias as well
    
    def _update_existing_streams_capacity(self, grade_level: str, new_max_capacity: int):
        """Update max capacity for existing streams in a grade level"""
        query = "UPDATE streams SET max_capacity = ? WHERE grade_level = ?"
        self.db.execute_update(query, (new_max_capacity, grade_level))
    
    def create_streams_for_grade(self, grade_level: str, custom_streams: list = None, max_capacity: int = None):
        """Create streams for a grade level with configurable stream types and max capacity"""
        # Use custom streams if provided, otherwise get stored config, fallback to default
        if custom_streams or max_capacity is not None:
            # If either custom_streams or max_capacity is provided, update the config
            config = self.get_streams_config_for_grade(grade_level)
            stream_types = custom_streams if custom_streams else config['stream_types']
            capacity = max_capacity if max_capacity is not None else config['max_capacity']
            
            # Save this configuration for future use
            self.set_streams_for_grade(grade_level, stream_types, capacity)
        else:
            # Get existing configuration
            config = self.get_streams_config_for_grade(grade_level)
            stream_types = config['stream_types']
            capacity = config['max_capacity']
        
        # Create streams
        for stream_type in stream_types:
            stream_id = f"{grade_level.replace(' ', '')}{stream_type}"
            query = '''
                INSERT INTO streams (stream_id, grade_level, stream_type, max_capacity)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(stream_id) DO UPDATE SET max_capacity = ?
            '''
            self.db.execute_update(query, (stream_id, grade_level, stream_type, capacity, capacity))
    
    def get_streams_for_grade(self, grade_level: str):
        """Get all streams for a grade level"""
        query = "SELECT * FROM streams WHERE grade_level = ? ORDER BY stream_type"
        results = self.db.execute_query(query, (grade_level,))
        return [Stream(*row) for row in results]
    
    def get_stream_by_id(self, stream_id: str):
        """Get stream by ID"""
        query = "SELECT * FROM streams WHERE stream_id = ?"
        result = self.db.execute_query(query, (stream_id,))
        return Stream(*result[0]) if result else None
    
    def assign_class_teacher(self, stream_id: str, teacher_id: str):
        """Assign class teacher to a stream"""
        query = "UPDATE streams SET class_teacher_id = ? WHERE stream_id = ?"
        self.db.execute_update(query, (teacher_id, stream_id))
        
        # Mark teacher as class teacher
        query = "UPDATE teachers SET is_class_teacher = TRUE WHERE teacher_id = ?"
        self.db.execute_update(query, (teacher_id,))
    
    def get_students_in_stream(self, stream_id):
        """Get all students in a specific stream"""
        try:
            # Get table schema
            schema_query = "PRAGMA table_info(students)"
            columns_info = self.db.execute_query(schema_query)
            available_columns = [col[1] for col in columns_info]  # col[1] is the column name
            
            # Basic required fields that should exist
            basic_fields = ['student_id', 'name', 'grade']
            
            # Optional fields - only include if they exist
            optional_fields = ['date_of_birth', 'phone_number', 'parent_guardian_name', 
                              'parent_guardian_phone', 'parent_guardian_email',
                              'emergency_contact_name', 'emergency_contact_phone', 
                              'medical_info', 'status']
            
            # Build the SELECT clause with only existing columns
            select_fields = []
            for field in basic_fields:
                if field in available_columns:
                    select_fields.append(field)
            
            for field in optional_fields:
                if field in available_columns:
                    select_fields.append(field)
            
            if not select_fields:
                return []
            
            query = f"""
            SELECT {', '.join(select_fields)}
            FROM students 
            WHERE stream_id = ? AND status = 'active'
            ORDER BY name
            """
            
            results = self.db.execute_query(query, (stream_id,))
            
            students = []
            for row in results:
                try:
                    # Create a minimal student object with available data
                    class StreamStudent:
                        def __init__(self, data_dict):
                            for key, value in data_dict.items():
                                setattr(self, key, value)
                    
                    # Create dictionary from row data
                    student_data = {}
                    for i, field in enumerate(select_fields):
                        student_data[field] = row[i] if i < len(row) else None
                    
                    student = StreamStudent(student_data)
                    students.append(student)
                    
                except Exception:
                    # Create absolute minimal student object
                    class MinimalStudent:
                        def __init__(self, student_id, name):
                            self.student_id = student_id if len(row) > 0 else "Unknown"
                            self.name = name if len(row) > 1 else "Unknown Student"
                    
                    students.append(MinimalStudent(row[0] if len(row) > 0 else "Unknown", 
                                                  row[1] if len(row) > 1 else "Unknown Student"))
            
            return students
            
        except Exception:
            return []
    
    def assign_student_to_stream(self, student_id: str, assignment_method: str = "performance") -> str:
        """
        Assign student to a stream based on specified method
        Options: "performance", "random", "balanced"
        Returns: stream_id that student was assigned to
        """
        student = self._get_student(student_id)
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
        
        streams = self.get_streams_for_grade(student.grade)
        
        if not streams:
            self.create_streams_for_grade(student.grade)
            streams = self.get_streams_for_grade(student.grade)
        
        if assignment_method == "performance":
            return self._performance_based_assignment(student, streams)
        elif assignment_method == "random":
            return self._random_assignment(student, streams)
        elif assignment_method == "balanced":
            return self._balanced_assignment(student, streams)
        else:
            # Default to performance-based assignment
            return self._performance_based_assignment(student, streams)
    
    def _performance_based_assignment(self, student, streams) -> str:
        """Assign based on GPA thresholds - adaptable to any number of streams"""
        # Sort streams by type to ensure consistent assignment
        sorted_streams = sorted(streams, key=lambda s: s.stream_type)
        num_streams = len(sorted_streams)
        
        if num_streams == 1:
            # Only one stream available
            target_stream = sorted_streams[0]
        elif num_streams == 2:
            # Two streams: high performers in first, others in second
            if student.gpa >= 3.25:
                target_stream = sorted_streams[0]
            else:
                target_stream = sorted_streams[1]
        elif num_streams == 3:
            # Original 3-stream logic (backward compatible)
            if student.gpa >= 3.5:
                # Find 'S' stream or first stream
                target_stream = next((s for s in sorted_streams if s.stream_type == 'S'), sorted_streams[0])
            elif student.gpa >= 3.0:
                # Find 'C' stream or middle stream
                target_stream = next((s for s in sorted_streams if s.stream_type == 'C'), 
                                   sorted_streams[len(sorted_streams)//2])
            else:
                # Find 'R' stream or last stream
                target_stream = next((s for s in sorted_streams if s.stream_type == 'R'), sorted_streams[-1])
        else:
            # More than 3 streams: distribute evenly based on performance
            # Calculate which quintile/quartile the student falls into
            if student.gpa >= 3.8:
                target_index = 0  # Top stream
            elif student.gpa >= 3.4:
                target_index = min(1, num_streams - 1)
            elif student.gpa >= 3.0:
                target_index = min(num_streams // 2, num_streams - 1)
            elif student.gpa >= 2.5:
                target_index = min(num_streams - 2, num_streams - 1)
            else:
                target_index = num_streams - 1  # Bottom stream
            
            target_stream = sorted_streams[target_index]
        
        # Check capacity and assign to available stream if target is full
        if self.get_student_count(target_stream.stream_id) >= target_stream.max_capacity:
            # Find stream with available capacity
            available_streams = [s for s in streams if self.get_student_count(s.stream_id) < s.max_capacity]
            if available_streams:
                target_stream = min(available_streams, key=lambda s: self.get_student_count(s.stream_id))
        
        self._update_student_stream(student.student_id, target_stream.stream_id)
        return target_stream.stream_id
    
    def _random_assignment(self, student, streams) -> str:
        """Randomly assign to a stream"""
        import random
        available_streams = [s for s in streams if self.get_student_count(s.stream_id) < s.max_capacity]
        if not available_streams:
            # If all streams are full, find least full stream
            available_streams = sorted(streams, key=lambda s: self.get_student_count(s.stream_id))
        
        stream = random.choice(available_streams)
        self._update_student_stream(student.student_id, stream.stream_id)
        return stream.stream_id
    
    def _balanced_assignment(self, student, streams) -> str:
        """Assign to the stream with most available capacity"""
        stream_capacities = [
            (s, s.max_capacity - self.get_student_count(s.stream_id))
            for s in streams
        ]
        # Sort by available capacity descending, then by stream_type for consistency
        stream_capacities.sort(key=lambda x: (x[1], x[0].stream_type), reverse=True)
        
        # Choose stream with most available capacity
        stream = stream_capacities[0][0]
        self._update_student_stream(student.student_id, stream.stream_id)
        return stream.stream_id
    
    def get_student_count(self, stream_id: str) -> int:
        """Get number of students in a stream"""
        query = "SELECT COUNT(*) FROM students WHERE stream_id = ? AND status = 'active'"
        result = self.db.execute_query(query, (stream_id,))
        return result[0][0] if result else 0
    
    def _update_student_stream(self, student_id: str, stream_id: str):
        """Update student's stream assignment"""
        query = "UPDATE students SET stream_id = ? WHERE student_id = ?"
        self.db.execute_update(query, (stream_id, student_id))
    
    def _get_student(self, student_id: str):
        """Get student by ID"""
        query = "SELECT * FROM students WHERE student_id = ?"
        result = self.db.execute_query(query, (student_id,))
        if result:
            # Handle the case where we might have different column counts
            row = result[0]
            if len(row) >= 10:  # Minimum required columns
                return Student(
                    student_id=row[0],
                    name=row[1],
                    age=row[2],
                    grade=row[3],
                    admission_date=row[4],
                    gpa=row[5] if len(row) > 5 else 0.0,
                    attendance_rate=row[6] if len(row) > 6 else 100.0,
                    behavior_score=row[7] if len(row) > 7 else 100.0,
                    parent_contact=row[8] if len(row) > 8 else "",
                    medical_info=row[9] if len(row) > 9 else "",
                    photo=row[10] if len(row) > 10 else None,
                    status=row[11] if len(row) > 11 else "active",
                    stream_id=row[12] if len(row) > 12 else None
                )
        return None

    def assign_subject_teacher(self, stream_id: str, subject: str, teacher_id: str):
        """Assign subject teacher to a stream"""
        # First get the current stream data
        stream = self.get_stream_by_id(stream_id)
        if not stream:
            return
        
        # Update subject teachers dictionary
        stream.subject_teachers[subject] = teacher_id
        
        # Convert dictionary to JSON string for storage
        import json
        subject_teachers_json = json.dumps(stream.subject_teachers)
        
        # Update database
        query = "UPDATE streams SET subject_teachers = ? WHERE stream_id = ?"
        self.db.execute_update(query, (subject_teachers_json, stream_id))
    
    def get_subject_teacher(self, stream_id: str, subject: str):
        """Get subject teacher for a stream"""
        stream = self.get_stream_by_id(stream_id)
        if not stream:
            return None
        return stream.subject_teachers.get(subject, None)
    
    def enforce_class_sizes(self, grade_level: str = None):
        """Enforce maximum class sizes by rebalancing if needed"""
        if grade_level:
            streams = self.get_streams_for_grade(grade_level)
        else:
            # Get all streams
            query = "SELECT * FROM streams"
            results = self.db.execute_query(query)
            streams = [Stream(*row) for row in results]
        
        for stream in streams:
            student_count = self.get_student_count(stream.stream_id)
            if student_count > stream.max_capacity:
                self._rebalance_stream(stream)
    
    def _rebalance_stream(self, overloaded_stream):
        """Rebalance an overloaded stream by moving students to other streams"""
        students_in_stream = self.get_students_in_stream(overloaded_stream.stream_id)
        excess_students = len(students_in_stream) - overloaded_stream.max_capacity
        
        if excess_students <= 0:
            return
        
        # Get other streams in the same grade
        other_streams = [
            s for s in self.get_streams_for_grade(overloaded_stream.grade_level)
            if s.stream_id != overloaded_stream.stream_id
        ]
        
        # Sort students by GPA (move lowest performing students first for balanced redistribution)
        students_to_move = sorted(students_in_stream, key=lambda s: s.gpa)[:excess_students]
        
        for student in students_to_move:
            # Find stream with most available capacity
            available_streams = [
                (s, s.max_capacity - self.get_student_count(s.stream_id))
                for s in other_streams
                if self.get_student_count(s.stream_id) < s.max_capacity
            ]
            
            if available_streams:
                # Sort by available capacity descending
                available_streams.sort(key=lambda x: x[1], reverse=True)
                target_stream = available_streams[0][0]
                self._update_student_stream(student.student_id, target_stream.stream_id)
    
    def get_stream_statistics(self) -> dict:
        """Get statistics about all streams"""
        query = "SELECT * FROM streams ORDER BY grade_level, stream_type"
        results = self.db.execute_query(query)
        streams = [Stream(*row) for row in results]
        
        stats = {
            'total_streams': len(streams),
            'streams_by_grade': {},
            'streams_by_type': {},
            'capacity_utilization': [],
            'capacity_settings': {}  # Track max capacity settings
        }
        
        for stream in streams:
            student_count = self.get_student_count(stream.stream_id)
            utilization = (student_count / stream.max_capacity) * 100 if stream.max_capacity > 0 else 0
            
            # By grade
            if stream.grade_level not in stats['streams_by_grade']:
                stats['streams_by_grade'][stream.grade_level] = 0
            stats['streams_by_grade'][stream.grade_level] += 1
            
            # By type
            if stream.stream_type not in stats['streams_by_type']:
                stats['streams_by_type'][stream.stream_type] = 0
            stats['streams_by_type'][stream.stream_type] += 1
            
            # Capacity settings by grade
            if stream.grade_level not in stats['capacity_settings']:
                stats['capacity_settings'][stream.grade_level] = stream.max_capacity
            
            # Capacity utilization
            stats['capacity_utilization'].append({
                'stream_id': stream.stream_id,
                'grade': stream.grade_level,
                'type': stream.stream_type,
                'current_students': student_count,
                'max_capacity': stream.max_capacity,
                'utilization_percent': round(utilization, 1)
            })
        
        return stats
    
    def delete_stream(self, stream_id: str, reassign_method: str = "balanced"):
        """
        Delete a stream and reassign its students to other streams in the same grade
        """
        stream = self.get_stream_by_id(stream_id)
        if not stream:
            return False
        
        # Get students in this stream
        students = self.get_students_in_stream(stream_id)
        
        # Get other streams in the same grade
        other_streams = [
            s for s in self.get_streams_for_grade(stream.grade_level)
            if s.stream_id != stream_id
        ]
        
        if not other_streams:
            raise ValueError("Cannot delete the only stream in a grade level")
        
        # Reassign students
        for student in students:
            if reassign_method == "balanced":
                self._balanced_assignment(student, other_streams)
            elif reassign_method == "performance":
                self._performance_based_assignment(student, other_streams)
            else:  # random
                self._random_assignment(student, other_streams)
        
        # Delete the stream
        query = "DELETE FROM streams WHERE stream_id = ?"
        self.db.execute_update(query, (stream_id,))
        
        return True
    
    def get_available_stream_types(self) -> list:
        """Get list of all stream types that have been used"""
        query = "SELECT DISTINCT stream_type FROM streams ORDER BY stream_type"
        results = self.db.execute_query(query)
        used_types = [row[0] for row in results]
        
        # Combine with default types
        all_types = list(set(self.default_streams + used_types))
        return sorted(all_types)

class SchoolManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.ai = SchoolAI()
        self.stream_manager = StreamManager(self.db)
        
        self.admission_criteria = {
            'min_gpa': 2.0,
            'max_age_deviation': 2,
            'required_fields': ['name', 'age', 'parent_contact']
        }

    def auto_admit_student(self, application_data: Dict) -> Dict:
        """AI-powered automatic student admission with clear criteria"""
        
        try:
            # Validate required fields
            missing_fields = []
            for field in self.admission_criteria['required_fields']:
                if not application_data.get(field):
                    missing_fields.append(field)
            
            if missing_fields:
                return {
                    "status": "REJECTED",
                    "reason": f"Missing required fields: {', '.join(missing_fields)}",
                    "ai_reasoning": "Application incomplete - missing required information"
                }
            
            # Determine appropriate grade based on age
            age = application_data.get('age', 6)
            recommended_grade = self._determine_grade_by_age(age)
            typical_age_for_grade = self._get_typical_age_for_grade(recommended_grade)
            age_deviation = abs(age - typical_age_for_grade)
            
            # Process GPA
            previous_gpa = application_data.get('previous_gpa', 3.0)
            if isinstance(previous_gpa, str):
                try:
                    previous_gpa = float(previous_gpa)
                except ValueError:
                    previous_gpa = 3.0
            
            # Automatic rejection criteria
            if previous_gpa < self.admission_criteria['min_gpa']:
                return {
                    "status": "REJECTED",
                    "reason": f"GPA {previous_gpa} below minimum requirement of {self.admission_criteria['min_gpa']}",
                    "ai_reasoning": f"Academic performance below school standards. Minimum GPA required: {self.admission_criteria['min_gpa']}"
                }
            
            if age_deviation > self.admission_criteria['max_age_deviation']:
                return {
                    "status": "REJECTED",
                    "reason": f"Age {age} not appropriate for available grade levels",
                    "ai_reasoning": f"Student age ({age}) significantly outside typical range for grade placement"
                }
            
            # AI evaluation prompt
            evaluation_prompt = f"""
            SCHOOL ADMISSION EVALUATION

            Student Information:
            - Name: {application_data.get('name')}
            - Age: {age} years old
            - Previous GPA: {previous_gpa} (if applicable)
            - Extracurricular Activities: {application_data.get('activities', 'None listed')}
            - Parent Contact: {application_data.get('parent_contact', 'Provided')}
            - Medical Information: {application_data.get('medical_info', 'None listed')}

            Recommended Grade Level: {recommended_grade}
            (Typical age for this grade: {typical_age_for_grade})

            ADMISSION CRITERIA MET:
            ✓ Required fields completed
            ✓ GPA ({previous_gpa}) meets minimum requirement ({self.admission_criteria['min_gpa']})
            ✓ Age ({age}) within acceptable range for grade placement
            
            INSTRUCTIONS:
            Based on the information provided and criteria met, evaluate this application.
            
            The student meets all basic requirements. Consider:
            1. Academic readiness (GPA indicates capability)
            2. Age-appropriate placement (recommended grade: {recommended_grade})
            3. Any special considerations from medical information
            4. Overall suitability for school environment
            
            Respond with:
            DECISION: ACCEPT or REJECT
            GRADE: {recommended_grade} (unless special circumstances require adjustment)
            REASONING: Brief explanation of your decision
            
            Note: Student has already passed initial screening criteria.
            """
            
            ai_response = self.ai.generate_response(evaluation_prompt)
            
            # Check for AI errors
            if any(error in ai_response for error in ["Ollama Error:", "OpenRouter Error:", "AI Error:"]):
                raise Exception(ai_response)
            
            # Parse AI response
            decision = "ACCEPTED" if "ACCEPT" in ai_response.upper() else "REJECTED"
            
            # Extract grade from AI response
            ai_grade = recommended_grade
            if "GRADE:" in ai_response:
                try:
                    grade_line = [line for line in ai_response.split('\n') if 'GRADE:' in line][0]
                    ai_grade = grade_line.split('GRADE:')[1].strip().split()[0]
                    if ai_grade not in self._get_all_grade_levels():
                        ai_grade = recommended_grade
                except:
                    ai_grade = recommended_grade
            
            if decision == "ACCEPTED":
                # Generate student ID
                student_id = f"STU{len(self.get_all_students()) + 1:04d}"
                
                # Handle admission date
                admission_date = application_data.get('admission_date')
                if not admission_date:
                    from datetime import datetime
                    admission_date = datetime.now().strftime('%Y-%m-%d')
                
                # Create Student object
                student = Student(
                    student_id=student_id,
                    name=application_data.get('name'),
                    age=age,
                    grade=ai_grade,
                    admission_date=admission_date,
                    gpa=previous_gpa,
                    attendance_rate=100.0,
                    behavior_score=100.0,
                    parent_contact=application_data.get('parent_contact', ''),
                    medical_info=application_data.get('medical_info', ''),
                    photo=None
                )
                
                # Save student to database first
                self._save_student(student)
                
                # Assign to stream
                assignment_method = application_data.get('assignment_method', 'performance')
                stream_id = self.stream_manager.assign_student_to_stream(
                    student_id, 
                    assignment_method
                )
                
                return {
                    "status": "ACCEPTED",
                    "student_id": student_id,
                    "grade": ai_grade,
                    "stream_id": stream_id,
                    "ai_reasoning": ai_response,
                    "recommended_grade": recommended_grade
                }
            else:
                return {
                    "status": "REJECTED",
                    "ai_reasoning": ai_response,
                    "note": "Decision made after meeting basic requirements"
                }
                
        except Exception as e:
            # Clean error handling
            error_msg = str(e)
            if "OpenRouter Error:" in error_msg and hasattr(self.ai, 'provider') and "OLLAMA" in str(self.ai.provider):
                error_msg = error_msg.replace("OpenRouter Error: ", "")
            raise Exception(error_msg)

    def _determine_grade_by_age(self, age: int) -> str:
        """Determine appropriate grade level based on age"""
        grade_mapping = {
            5: "Kindergarten", 6: "Grade 1", 7: "Grade 2", 8: "Grade 3",
            9: "Grade 4", 10: "Grade 5", 11: "Grade 6", 12: "JHS 1",
            13: "JHS 2", 14: "JHS 3"
        }
        return grade_mapping.get(age, "Grade 1")
    
    def _get_typical_age_for_grade(self, grade: str) -> int:
        """Get typical age for a grade level"""
        age_mapping = {
            "Kindergarten": 5, "Grade 1": 6, "Grade 2": 7, "Grade 3": 8,
            "Grade 4": 9, "Grade 5": 10, "Grade 6": 11, "JHS 1": 12,
            "JHS 2": 13, "JHS 3": 14
        }
        return age_mapping.get(grade, 6)
    
    def _get_all_grade_levels(self) -> List[str]:
        """Get list of all valid grade levels"""
        return ["Kindergarten", "Grade 1", "Grade 2", "Grade 3", "Grade 4", 
                "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"]

    def _save_student(self, student):
        """Save student to database"""
        query = '''
            INSERT INTO students (student_id, name, age, grade, admission_date, 
                                gpa, attendance_rate, behavior_score, parent_contact, medical_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            student.student_id, student.name, student.age, student.grade,
            student.admission_date, student.gpa, student.attendance_rate,
            student.behavior_score, student.parent_contact, student.medical_info
        )
        self.db.execute_update(query, params)

    def get_admission_stats(self) -> Dict:
        """Get admission statistics"""
        try:
            query = "SELECT status, COUNT(*) FROM applications GROUP BY status"
            results = self.db.execute_query(query)
            return dict(results) if results else {'Accepted': 0, 'Rejected': 0, 'Pending': 0}
        except:
            return {'Accepted': 0, 'Rejected': 0, 'Pending': 0}

    def auto_grade_promotion(self) -> Dict:
        """Automatically promote students based on AI evaluation"""
        try:
            students = self.get_all_students()
            promotions = []
            
            for student in students:
                # Clear promotion criteria
                min_gpa = 2.0
                min_attendance = 75
                min_behavior = 60
                
                promotion_prompt = f"""
                GRADE PROMOTION EVALUATION
                
                Student: {student['name']} (ID: {student['student_id']})
                Current Grade: {student['grade']}
                Academic Performance:
                - GPA: {student['gpa']}
                - Attendance Rate: {student['attendance_rate']}%
                - Behavior Score: {student['behavior_score']}
                
                PROMOTION REQUIREMENTS:
                - Minimum GPA: {min_gpa} (Current: {student['gpa']} - {'✓ MET' if student['gpa'] >= min_gpa else '✗ NOT MET'})
                - Minimum Attendance: {min_attendance}% (Current: {student['attendance_rate']}% - {'✓ MET' if student['attendance_rate'] >= min_attendance else '✗ NOT MET'})
                - Minimum Behavior Score: {min_behavior} (Current: {student['behavior_score']} - {'✓ MET' if student['behavior_score'] >= min_behavior else '✗ NOT MET'})
                
                INSTRUCTIONS:
                If ALL requirements are met, respond with: PROMOTE
                If any requirement is not met, respond with: RETAIN
                
                Provide brief reasoning for your decision.
                
                DECISION: PROMOTE or RETAIN
                REASONING: [Your explanation]
                """
                
                ai_decision = self.ai.generate_response(promotion_prompt)
                
                if "PROMOTE" in ai_decision.upper():
                    new_grade = self._get_next_grade(student['grade'])
                    if new_grade != "GRADUATED":
                        self._update_student_grade(student['student_id'], new_grade)
                        
                        # Update stream for new grade
                        self.stream_manager.assign_student_to_stream(
                            student['student_id'], 
                            'balanced'
                        )
                        
                        promotions.append({
                            "student_id": student['student_id'],
                            "name": student['name'],
                            "from_grade": student['grade'],
                            "to_grade": new_grade,
                            "reasoning": ai_decision,
                            "status": "promoted"
                        })
                    else:
                        self._graduate_student(student['student_id'])
                        promotions.append({
                            "student_id": student['student_id'],
                            "name": student['name'],
                            "from_grade": student['grade'],
                            "to_grade": "GRADUATED",
                            "reasoning": "Student has completed all requirements and is ready for graduation",
                            "status": "graduated"
                        })
                else:
                    promotions.append({
                        "student_id": student['student_id'],
                        "name": student['name'],
                        "from_grade": student['grade'],
                        "to_grade": student['grade'],
                        "reasoning": ai_decision,
                        "status": "retained"
                    })
            
            return {"promotions": promotions}
            
        except Exception as e:
            return {"error": f"Promotion process failed: {str(e)}", "promotions": []}

    def _get_next_grade(self, current_grade: str) -> str:
        """Get the next grade level"""
        grade_progression = {
            "Kindergarten": "Grade 1", "Grade 1": "Grade 2", "Grade 2": "Grade 3",
            "Grade 3": "Grade 4", "Grade 4": "Grade 5", "Grade 5": "Grade 6",
            "Grade 6": "JHS 1", "JHS 1": "JHS 2", "JHS 2": "JHS 3",
            "JHS 3": "GRADUATED"
        }
        return grade_progression.get(current_grade, "GRADUATED")

    def _update_student_grade(self, student_id: str, new_grade: str):
        """Update student's grade in database"""
        query = "UPDATE students SET grade = ? WHERE student_id = ?"
        self.db.execute_update(query, (new_grade, student_id))

    def _graduate_student(self, student_id: str):
        """Mark student as graduated"""
        query = "UPDATE students SET status = 'graduated' WHERE student_id = ?"
        self.db.execute_update(query, (student_id,))

    def get_all_students(self) -> List[Dict]:
        """Get all active students with stream info"""
        try:
            query = """
                SELECT s.*, st.stream_id, st.grade_level as stream_grade, st.stream_type
                FROM students s
                LEFT JOIN streams st ON s.stream_id = st.stream_id
                WHERE s.status != 'graduated'
            """
            results = self.db.execute_query(query)
            columns = [
                'student_id', 'name', 'age', 'grade', 'admission_date', 'gpa', 
                'attendance_rate', 'behavior_score', 'parent_contact', 
                'medical_info', 'photo', 'status', 'stream_id',
                'stream_grade', 'stream_type'
            ]
            return [dict(zip(columns, row)) for row in results] if results else []
        except Exception as e:
            return []

    def generate_ai_insights(self) -> str:
        """Generate AI-powered school insights"""
        try:
            students = self.get_all_students()
            
            total_students = len(students)
            if total_students == 0:
                return "No active students found. Unable to generate insights."
            
            avg_gpa = sum(s['gpa'] for s in students) / total_students
            avg_attendance = sum(s['attendance_rate'] for s in students) / total_students
            
            insight_prompt = f"""
            SCHOOL PERFORMANCE ANALYSIS
            
            Current School Statistics:
            - Total Active Students: {total_students}
            - Average GPA: {avg_gpa:.2f}
            - Average Attendance Rate: {avg_attendance:.1f}%
            
            ANALYSIS REQUEST:
            Please provide actionable insights and recommendations based on this data:
            
            1. ACADEMIC PERFORMANCE
               - Assessment of current GPA trends
               - Recommendations for improvement strategies
            
            2. ATTENDANCE PATTERNS
               - Evaluation of attendance rates
               - Suggestions for improving attendance
            
            3. RESOURCE ALLOCATION
               - Where should the school focus resources?
               - What programs might be most beneficial?
            
            4. EARLY INTERVENTION
               - Which students might need additional support?
               - What warning signs should we monitor?
            
            Provide specific, actionable recommendations that the school can implement.
            """
            
            return self.ai.generate_response(insight_prompt)
            
        except Exception as e:
            return f"Unable to generate insights: {str(e)}"
        
    def update_student(self, update_data: Dict) -> bool:
        """Update student information"""
        try:
            self.db.update_student(update_data)
            return True
        except Exception as e:
            return False

class GPACalculator:
    """
    Handles GPA calculation and academic performance tracking
    Separate from DatabaseManager to maintain clean separation of concerns
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def calculate_gpa(self, student_id: str, semester: str = None, year: int = None) -> float:
        """
        Calculate GPA for a student based on their course grades
        
        Args:
            student_id: Student ID
            semester: Specific semester (optional, calculates for all if None)
            year: Specific year (optional, calculates for all if None)
            
        Returns:
            float: Calculated GPA on 4.0 scale
        """
        try:
            # Build query with optional filters
            query = """
                SELECT g.grade, c.credits
                FROM grades g
                JOIN courses c ON g.course_id = c.course_id
                WHERE g.student_id = ?
            """
            params = [student_id]
            
            if semester:
                query += " AND g.semester = ?"
                params.append(semester)
                
            if year:
                query += " AND g.year = ?"
                params.append(year)
                
            grades_data = self.db_manager.execute_query(query, tuple(params))
            
            if not grades_data:
                return 0.0
                
            total_grade_points = 0.0
            total_credits = 0
            
            for grade, credits in grades_data:
                if grade is not None and credits is not None:
                    grade_points = self._convert_to_4_scale(grade)
                    total_grade_points += grade_points * credits
                    total_credits += credits
            
            return round(total_grade_points / total_credits, 2) if total_credits > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def _convert_to_4_scale(self, percentage_grade: float) -> float:
        """Convert percentage grade to 4.0 GPA scale"""
        if percentage_grade >= 75:
            return 4.0
        elif percentage_grade >= 70:
            return 3.5
        elif percentage_grade >= 65:
            return 3.0
        elif percentage_grade >= 60:
            return 2.5
        elif percentage_grade >= 55:
            return 2.0
        elif percentage_grade >= 50:
            return 1.5
        elif percentage_grade >= 45:
            return 1.0
        elif percentage_grade >= 40:
            return 0.5
        else:
            return 0.0
    
    def update_student_gpa(self, student_id: str, semester: str = None, year: int = None) -> float:
        """
        Calculate and update student's GPA in the database
        
        Returns:
            float: The calculated GPA that was stored
        """
        try:
            gpa = self.calculate_gpa(student_id, semester, year)
            
            # Update student's GPA in database
            self.db_manager.execute_update(
                "UPDATE students SET gpa = ? WHERE student_id = ?",
                (gpa, student_id)
            )
            
            return gpa
            
        except Exception as e:
            return 0.0
    
    def bulk_update_all_gpas(self) -> Dict[str, float]:
        """
        Update GPAs for all active students
        
        Returns:
            Dict[str, float]: Dictionary mapping student_id to their calculated GPA
        """
        try:
            # Get all active students
            students = self.db_manager.execute_query(
                "SELECT student_id, name FROM students WHERE status = 'active'"
            )
            
            updated_gpas = {}
            
            if students:
                for student_id, name in students:
                    try:
                        gpa = self.update_student_gpa(student_id)
                        updated_gpas[student_id] = gpa
                    except Exception:
                        # Continue processing other students if one fails
                        continue
                        
            return updated_gpas
            
        except Exception as e:
            return {}
    
    def get_gpa_by_semester(self, student_id: str) -> List[Dict]:
        """
        Get GPA breakdown by semester for a student
        
        Returns:
            List of dictionaries with semester, year, and GPA
        """
        try:
            # Get all semester/year combinations for the student
            semesters_query = """
                SELECT DISTINCT semester, year
                FROM grades
                WHERE student_id = ?
                ORDER BY year DESC, 
                    CASE semester
                        WHEN 'Trinity' THEN 1
                        WHEN 'Lent' THEN 2
                        WHEN 'Advent' THEN 3
                    END
            """
            
            semesters = self.db_manager.execute_query(semesters_query, (student_id,))
            
            semester_gpas = []
            if semesters:
                for semester, year in semesters:
                    gpa = self.calculate_gpa(student_id, semester, year)
                    semester_gpas.append({
                        'semester': semester,
                        'year': year,
                        'gpa': gpa
                    })
                
            return semester_gpas
            
        except Exception as e:
            return []
    
    def get_class_gpa_statistics(self, grade_level: str = None) -> Dict:
        """
        Get GPA statistics for a class or entire school
        
        Returns:
            Dictionary with average, min, max GPAs and distribution
        """
        try:
            query = """
                SELECT s.student_id, s.name, s.gpa, s.grade
                FROM students s
                WHERE s.status = 'active' AND s.gpa IS NOT NULL
            """
            params = []
            
            if grade_level:
                query += " AND s.grade = ?"
                params.append(grade_level)
                
            students_data = self.db_manager.execute_query(query, tuple(params))
            
            if not students_data:
                return {
                    'count': 0,
                    'average_gpa': 0.0,
                    'min_gpa': 0.0,
                    'max_gpa': 0.0,
                    'distribution': {}
                }
            
            gpas = [gpa for _, _, gpa, _ in students_data if gpa is not None]
            
            if not gpas:
                return {
                    'count': 0,
                    'average_gpa': 0.0,
                    'min_gpa': 0.0,
                    'max_gpa': 0.0,
                    'distribution': {}
                }
            
            # Calculate distribution
            distribution = {
                'A (3.5-4.0)': len([gpa for gpa in gpas if gpa >= 3.5]),
                'B (2.5-3.49)': len([gpa for gpa in gpas if 2.5 <= gpa < 3.5]),
                'C (1.5-2.49)': len([gpa for gpa in gpas if 1.5 <= gpa < 2.5]),
                'D (1.0-1.49)': len([gpa for gpa in gpas if 1.0 <= gpa < 1.5]),
                'F (0.0-0.99)': len([gpa for gpa in gpas if gpa < 1.0])
            }
            
            return {
                'count': len(gpas),
                'average_gpa': round(sum(gpas) / len(gpas), 2),
                'min_gpa': min(gpas),
                'max_gpa': max(gpas),
                'distribution': distribution
            }
            
        except Exception as e:
            return {
                'count': 0,
                'average_gpa': 0.0,
                'min_gpa': 0.0,
                'max_gpa': 0.0,
                'distribution': {}
            }


class AIHelperFunctions:
    """Class containing AI helper functions for data compilation"""
    
    def __init__(self, db_manager):
        """Initialize with database manager instance"""
        self.db = db_manager
    
    def compile_school_data(self):
        """Compile comprehensive school data from all tables"""
        try:
            # Get students data
            students = self.db.execute_query("SELECT COUNT(*) as total, AVG(gpa) as avg_gpa, AVG(attendance_rate) as avg_attendance FROM students WHERE status = 'active'")
            
            # Get teachers data
            teachers = self.db.execute_query("SELECT COUNT(*) as total, AVG(performance_score) as avg_performance FROM teachers WHERE status = 'active'")
            
            # Get courses data
            courses = self.db.execute_query("SELECT COUNT(*) as total FROM courses")
            
            # Get recent grades
            grades = self.db.execute_query("SELECT AVG(grade) as avg_grade FROM grades WHERE year = (SELECT MAX(year) FROM grades)")
            
            # Get attendance summary
            attendance = self.db.execute_query("SELECT status, COUNT(*) as count FROM attendance GROUP BY status")
            
            # Get financial summary
            finance = self.db.execute_query("SELECT type, SUM(amount) as total FROM finance GROUP BY type")
            
            # Process student data
            total_students = students[0][0] if students and len(students) > 0 else 0
            avg_gpa = students[0][1] if students and len(students) > 0 and students[0][1] is not None else 0
            avg_attendance = students[0][2] if students and len(students) > 0 and students[0][2] is not None else 0
            
            # Process teacher data
            total_teachers = teachers[0][0] if teachers and len(teachers) > 0 else 0
            avg_performance = teachers[0][1] if teachers and len(teachers) > 0 and teachers[0][1] is not None else 0
            
            # Process course data
            total_courses = courses[0][0] if courses and len(courses) > 0 else 0
            
            # Process grades data
            avg_grade = grades[0][0] if grades and len(grades) > 0 and grades[0][0] is not None else 0
            
            # Format attendance summary
            attendance_summary = "\n".join([f"- {row[0]}: {row[1]}" for row in attendance]) if attendance else "No attendance data"
            
            # Format financial summary
            financial_summary = "\n".join([f"- {row[0]}: ${row[1]:,.2f}" for row in finance]) if finance else "No financial data"
            
            data_summary = f"""
            STUDENT STATISTICS:
            - Total Active Students: {total_students}
            - Average GPA: {avg_gpa:.2f}
            - Average Attendance Rate: {avg_attendance:.1f}%
            
            TEACHER STATISTICS:
            - Total Active Teachers: {total_teachers}
            - Average Performance Score: {avg_performance:.1f}
            
            ACADEMIC DATA:
            - Total Courses: {total_courses}
            - Average Grade: {avg_grade:.2f}
            
            ATTENDANCE SUMMARY:
            {attendance_summary}
            
            FINANCIAL SUMMARY:
            {financial_summary}
            """
            
            return data_summary
            
        except Exception as e:
            return f"Error compiling school data: {str(e)}"

    def compile_report_data(self):
        """Compile data for reporting"""
        return self.compile_school_data()  # Use same comprehensive data

    def compile_trends_data(self):
        """Compile historical data for trend analysis"""
        try:
            # Get grade trends
            grade_trends = self.db.execute_query("SELECT year, semester, AVG(grade) as avg_grade FROM grades GROUP BY year, semester ORDER BY year, semester")
            
            # Get enrollment trends
            enrollment_trends = self.db.execute_query("SELECT substr(admission_date, 1, 4) as year, COUNT(*) as enrolled FROM students WHERE admission_date IS NOT NULL GROUP BY substr(admission_date, 1, 4) ORDER BY year")
            
            # Format grade trends
            grade_trends_summary = "\n".join([f"- {row[0]} {row[1]}: {row[2]:.2f}" for row in grade_trends]) if grade_trends else "No grade trend data"
            
            # Format enrollment trends
            enrollment_trends_summary = "\n".join([f"- {row[0]}: {row[1]} students" for row in enrollment_trends]) if enrollment_trends else "No enrollment trend data"
            
            trends_summary = f"""
            GRADE TRENDS:
            {grade_trends_summary}
            
            ENROLLMENT TRENDS:
            {enrollment_trends_summary}
            """
            
            return trends_summary
            
        except Exception as e:
            return f"Error compiling trends data: {str(e)}"

    def compile_risk_data(self):
        """Compile data for risk assessment"""
        try:
            # Get at-risk students
            at_risk_students = self.db.execute_query("SELECT COUNT(*) FROM students WHERE (gpa < 2.0 OR attendance_rate < 80 OR behavior_score < 70) AND status = 'active'")
            
            # Get overdue finances
            overdue_finances = self.db.execute_query("SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM finance WHERE status = 'pending' AND date < date('now', '-30 days')")
            
            # Process risk data
            at_risk_count = at_risk_students[0][0] if at_risk_students and len(at_risk_students) > 0 else 0
            overdue_count = overdue_finances[0][0] if overdue_finances and len(overdue_finances) > 0 else 0
            overdue_amount = overdue_finances[0][1] if overdue_finances and len(overdue_finances) > 0 and overdue_finances[0][1] is not None else 0
            
            risk_summary = f"""
            ACADEMIC RISKS:
            - At-risk students (low GPA/attendance/behavior): {at_risk_count}
            
            FINANCIAL RISKS:
            - Overdue payments: {overdue_count}
            - Total overdue amount: ${overdue_amount:,.2f}
            """
            
            return risk_summary
            
        except Exception as e:
            return f"Error compiling risk data: {str(e)}"

    def compile_academic_data(self):
        """Compile academic performance data"""
        try:
            # Get grade distribution
            grade_dist = self.db.execute_query("""
                SELECT 
                    CASE 
                        WHEN grade >= 90 THEN 'A' 
                        WHEN grade >= 80 THEN 'B' 
                        WHEN grade >= 70 THEN 'C' 
                        WHEN grade >= 60 THEN 'D' 
                        ELSE 'F' 
                    END as letter_grade, 
                    COUNT(*) 
                FROM grades 
                WHERE grade IS NOT NULL 
                GROUP BY letter_grade
            """)
            
            # Get course performance
            course_perf = self.db.execute_query("""
                SELECT c.name, AVG(g.grade) as avg_grade 
                FROM courses c 
                JOIN grades g ON c.course_id = g.course_id 
                WHERE g.grade IS NOT NULL
                GROUP BY c.course_id, c.name 
                ORDER BY avg_grade DESC
            """)
            
            # Format grade distribution
            grade_dist_summary = "\n".join([f"- Grade {row[0]}: {row[1]} students" for row in grade_dist]) if grade_dist else "No grade distribution data"
            
            # Format course performance (limit to top 10)
            course_perf_summary = "\n".join([f"- {row[0]}: {row[1]:.2f}" for row in course_perf[:10]]) if course_perf else "No course performance data"
            
            academic_summary = f"""
            GRADE DISTRIBUTION:
            {grade_dist_summary}
            
            COURSE PERFORMANCE:
            {course_perf_summary}
            """
            
            return academic_summary
            
        except Exception as e:
            return f"Error compiling academic data: {str(e)}"

    def compile_admin_data(self):
        """Compile administrative data"""
        try:
            # Get stream utilization
            stream_util = self.db.execute_query("""
                SELECT s.grade_level, s.stream_type, 
                       COUNT(st.student_id) as enrolled, 
                       s.max_capacity 
                FROM streams s 
                LEFT JOIN students st ON s.stream_id = st.stream_id AND st.status = 'active'
                GROUP BY s.stream_id, s.grade_level, s.stream_type, s.max_capacity
            """)
            
            # Get teacher workload
            teacher_load = self.db.execute_query("""
                SELECT t.name, COUNT(c.course_id) as courses 
                FROM teachers t 
                LEFT JOIN courses c ON t.teacher_id = c.teacher_id 
                WHERE t.status = 'active'
                GROUP BY t.teacher_id, t.name
            """)
            
            # Format stream utilization
            stream_util_summary = "\n".join([
                f"- {row[0]} {row[1]}: {row[2]}/{row[3]} ({(row[2]/row[3]*100) if row[3] > 0 else 0:.1f}%)" 
                for row in stream_util
            ]) if stream_util else "No stream data"
            
            # Format teacher workload (limit to top 10)
            teacher_load_summary = "\n".join([f"- {row[0]}: {row[1]} courses" for row in teacher_load[:10]]) if teacher_load else "No teacher workload data"
            
            admin_summary = f"""
            STREAM UTILIZATION:
            {stream_util_summary}
            
            TEACHER WORKLOAD:
            {teacher_load_summary}
            """
            
            return admin_summary
            
        except Exception as e:
            return f"Error compiling administrative data: {str(e)}"

    def compile_financial_data(self):
        """Compile financial data"""
        try:
            # Get financial summary by type
            fin_summary = self.db.execute_query("""
                SELECT type, status, COUNT(*) as count, SUM(amount) as total 
                FROM finance 
                WHERE amount IS NOT NULL
                GROUP BY type, status
            """)
            
            # Get monthly trends
            monthly_trends = self.db.execute_query("""
                SELECT substr(date, 1, 7) as month, SUM(amount) as total 
                FROM finance 
                WHERE date IS NOT NULL AND amount IS NOT NULL
                GROUP BY substr(date, 1, 7) 
                ORDER BY month DESC 
                LIMIT 12
            """)
            
            # Format financial summary
            fin_summary_text = "\n".join([f"- {row[0]} ({row[1]}): {row[2]} transactions, ${row[3]:,.2f}" for row in fin_summary]) if fin_summary else "No financial summary data"
            
            # Format monthly trends
            monthly_trends_text = "\n".join([f"- {row[0]}: ${row[1]:,.2f}" for row in monthly_trends]) if monthly_trends else "No monthly trend data"
            
            financial_summary = f"""
            FINANCIAL SUMMARY BY TYPE AND STATUS:
            {fin_summary_text}
            
            MONTHLY FINANCIAL TRENDS (Last 12 months):
            {monthly_trends_text}
            """
            
            return financial_summary
            
        except Exception as e:
            return f"Error compiling financial data: {str(e)}"

    def compile_infrastructure_data(self):
        """Compile infrastructure-related data"""
        try:
            # Get capacity utilization - Fixed query
            capacity = self.db.execute_query("""
                SELECT 
                    (SELECT SUM(max_capacity) FROM streams) as total_capacity,
                    (SELECT COUNT(*) FROM students WHERE status = 'active') as total_students
            """)
            
            # Get stream distribution
            stream_dist = self.db.execute_query("""
                SELECT grade_level, COUNT(*) as stream_count, SUM(max_capacity) as total_capacity 
                FROM streams 
                GROUP BY grade_level
            """)
            
            # Process capacity data
            total_capacity = capacity[0][0] if capacity and len(capacity) > 0 and capacity[0][0] is not None else 0
            total_students = capacity[0][1] if capacity and len(capacity) > 0 and capacity[0][1] is not None else 0
            utilization_rate = (total_students / total_capacity * 100) if total_capacity > 0 else 0
            
            # Format stream distribution
            stream_dist_summary = "\n".join([f"- {row[0]}: {row[1]} streams, {row[2]} capacity" for row in stream_dist]) if stream_dist else "No stream distribution data"
            
            infrastructure_summary = f"""
            CAPACITY UTILIZATION:
            - Total Capacity: {total_capacity}
            - Current Students: {total_students}
            - Utilization Rate: {utilization_rate:.1f}%
            
            STREAM DISTRIBUTION:
            {stream_dist_summary}
            """
            
            return infrastructure_summary
            
        except Exception as e:
            return f"Error compiling infrastructure data: {str(e)}"

    def compile_context_data(self):
        """Compile essential context data for AI chat"""
        try:
            # Get basic stats
            student_count = self.db.execute_query("SELECT COUNT(*) FROM students WHERE status = 'active'")
            teacher_count = self.db.execute_query("SELECT COUNT(*) FROM teachers WHERE status = 'active'")
            course_count = self.db.execute_query("SELECT COUNT(*) FROM courses")
            
            # Process context data
            total_students = student_count[0][0] if student_count and len(student_count) > 0 else 0
            total_teachers = teacher_count[0][0] if teacher_count and len(teacher_count) > 0 else 0
            total_courses = course_count[0][0] if course_count and len(course_count) > 0 else 0
            
            context_summary = f"""
            SCHOOL OVERVIEW:
            - Active Students: {total_students}
            - Active Teachers: {total_teachers}
            - Total Courses: {total_courses}
            """
            
            return context_summary
            
        except Exception as e:
            return f"Error compiling context data: {str(e)}"

    def get_data_by_type(self, data_type):
        """Get data based on specified type"""
        data_type_methods = {
            'school': self.compile_school_data,
            'report': self.compile_report_data,
            'trends': self.compile_trends_data,
            'risk': self.compile_risk_data,
            'academic': self.compile_academic_data,
            'admin': self.compile_admin_data,
            'financial': self.compile_financial_data,
            'infrastructure': self.compile_infrastructure_data,
            'context': self.compile_context_data
        }
        
        method = data_type_methods.get(data_type.lower())
        if method:
            return method()
        else:
            return f"Unknown data type: {data_type}. Available types: {', '.join(data_type_methods.keys())}"


def calculate_age(birth_date):
    """Calculate age from birth date."""
    today = datetime.now().date()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
       
def get_next_grade(current_grade):
    """Helper function to determine next grade level"""
    grade_progression = {
        "Grade 1": "Grade 2", "Grade 2": "Grade 3", "Grade 3": "Grade 4",
        "Grade 4": "Grade 5", "Grade 5": "Grade 6", "Grade 6": "JHS 1",
        "JHS 1": "JHS 2", "JHS 2": "JHS 3", "JHS 3": "GRADUATED"
    }
    return grade_progression.get(current_grade, "GRADUATED")    

def main():  
    initialize_ui()
    
    # Initialize session state
    if 'school_manager' not in st.session_state:
        st.session_state.school_manager = SchoolManager()
    
    # Auto-detect environment and set appropriate default provider
    if 'ai_provider' not in st.session_state:
        # Check if running on Streamlit Cloud by trying Ollama connection
        try:
            test_response = requests.get("http://localhost:11434/api/tags", timeout=2)
            # If successful, we're in local environment
            st.session_state.ai_provider = AIProvider.OLLAMA
            st.session_state.environment = "local"
        except:
            # If failed, we're likely on Streamlit Cloud
            st.session_state.ai_provider = AIProvider.OPENROUTER
            st.session_state.environment = "cloud"
        
    # Sidebar for AI Configuration
    st.sidebar.header("🧟 RadioSport Scholium")
    st.sidebar.text(f' Version {APP_VERSION}')

    # Show environment indicator
    if st.session_state.get('environment') == 'cloud':
        st.sidebar.caption("☁️ Cloud ")
    else:
        st.sidebar.caption("💻 Local ")

    # Initialize SchoolAI instance
    if 'school_ai' not in st.session_state:
        st.session_state.school_ai = SchoolAI()

    school_ai = st.session_state.school_ai

    # Provider selection - show only available options
    if st.session_state.get('environment') == 'cloud':
        # On Streamlit Cloud, only show Cloud option
        provider_display = st.sidebar.radio(
            "AI Provider",
            ["Cloud"],
            key="ai_provider_select"
        )
    else:
        # In local environment, show both options
        provider_display = st.sidebar.radio(
            "AI Provider",
            ["Local", "Cloud"],
            key="ai_provider_select"
        )
    
    # Map display names to actual provider values
    provider_mapping = {
        "Local": AIProvider.OLLAMA.value,
        "Cloud": AIProvider.OPENROUTER.value
    }
    provider = provider_mapping[provider_display]
    st.session_state.ai_provider = AIProvider(provider)
    school_ai.provider = st.session_state.ai_provider

    if st.session_state.ai_provider == AIProvider.OLLAMA:
        # Only show Ollama configuration in local environment
        if st.session_state.get('environment') != 'cloud':
            # Ollama Configuration
            st.sidebar.subheader("🔧 Ollama Settings")
            
            # Server URL configuration
            with st.sidebar.expander("Server Configuration", expanded=False):
                new_ollama_url = st.text_input(
                "Ollama Server URL", 
                value=st.session_state.get('ollama_url', 'http://localhost:11434'),
                placeholder="http://localhost:11434",
                help="Change if running Ollama on different host/port"
            )
            
            if new_ollama_url != st.session_state.get('ollama_url', 'http://localhost:11434'):
                st.session_state.ollama_url = new_ollama_url
                school_ai.ollama_url = new_ollama_url
                st.session_state.ollama_models = []
                if 'ai_model' in st.session_state:
                    del st.session_state.ai_model
            
            # Connection test
#            if st.button("🔄 Test Connection", key="sidebar_test_ollama"):
#                models = school_ai.get_ollama_models()
#                if models:
#                    st.success(f"✅ Connected! {len(models)} models found")
#                else:
#                    st.error("❌ Connection failed")
        
        # Model selection
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if not st.session_state.get('ollama_models', []):
                school_ai.get_ollama_models()
            
            available_models = st.session_state.get('ollama_models', [])
            if available_models:
                current_model = st.session_state.get('ai_model')
                if not current_model or current_model not in available_models:
                    current_model = available_models[0]
                    st.session_state.ai_model = current_model
                
                selected_model = st.selectbox(
                    "Available Models",
                    available_models,
                    index=available_models.index(current_model),
                    key="sidebar_ollama_model"
                )
                st.session_state.ai_model = selected_model
                school_ai.model = selected_model
            else:
                # Only show warning in local environment
                if st.session_state.get('environment') != 'cloud':
                    st.warning("⚠️ Models Unavailable.")
                else:
                    st.info("💡 Ollama not available. Using Cloud AI provider.")
                
                if 'ai_model' in st.session_state:
                    del st.session_state.ai_model
                school_ai.model = None
        
        with col2:
            if st.button("🔄", key="sidebar_refresh_ollama", help="Refresh model list"):
                st.session_state.ollama_models = []
                if 'ai_model' in st.session_state:
                    del st.session_state.ai_model
                school_ai.get_ollama_models()
                st.rerun()

    else:
        # OpenRouter Configuration
        st.sidebar.subheader("🌐 OpenRouter Settings")
        
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if not st.session_state.get('openrouter_models', []):
                school_ai.get_openrouter_free_models()
            
            available_models = st.session_state.get('openrouter_models', [])
            if available_models:
                model_display = [f"{model.get('name', model['id'])}" for model in available_models]
                model_ids = [model['id'] for model in available_models]
                
                current_model = st.session_state.get('ai_model')
                if not current_model or current_model not in model_ids:
                    current_model = model_ids[0] if model_ids else None
                    if current_model:
                        st.session_state.ai_model = current_model
                
                if current_model:
                    current_index = model_ids.index(current_model)
                    
                    selected_index = st.selectbox(
                        "Free Models",
                        range(len(model_display)),
                        format_func=lambda x: model_display[x],
                        index=current_index,
                        key="sidebar_openrouter_model",
                        help="Only free models are shown"
                    )
                    
                    selected_model_id = model_ids[selected_index]
                    st.session_state.ai_model = selected_model_id
                    school_ai.model = selected_model_id
                    
                    # Model details
                    selected_model_info = available_models[selected_index]
                    st.sidebar.caption(f"ID: {selected_model_id}")
                    if 'context_length' in selected_model_info:
                        st.sidebar.caption(f"Context: {selected_model_info['context_length']:,} tokens")
            else:
                st.sidebar.warning("⚠️ No free models available")
                if school_ai.openrouter_key:
                    st.sidebar.info("Check your internet connection or try refreshing.")
                if 'ai_model' in st.session_state:
                    del st.session_state.ai_model
                school_ai.model = None
        
        with col2:
            if st.button("🔄", key="sidebar_refresh_openrouter", help="Refresh model list"):
                st.session_state.openrouter_models = []
                if 'ai_model' in st.session_state:
                    del st.session_state.ai_model
                school_ai.get_openrouter_free_models()
                st.rerun()

    # Current Configuration Summary
    # Get current model state (defined after both Ollama and OpenRouter blocks)
    current_model = st.session_state.get('ai_model', 'None selected')
    
    # Only show error if model not selected AND we're in proper environment
    if current_model == 'None selected':
        if st.session_state.get('environment') == 'cloud':
            # On cloud, just show info message
            st.sidebar.info("ℹ️ Select a model above to enable AI features")
        else:
            # On local, show error
            st.sidebar.error("⚠️ Please select a model to continue")
    
    # Main Interface
    st.title("🏫 RadioSport Scholium")
   
    # Main Navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Dashboard", "👥 Admissions", "🎓 Students", "👨‍🏫 Teachers", 
        "📚 Academics", "💰 Finance", "📈 Analytics", "🤖 AI Insights", "🏫 Streams",
        "📝 Data Input"
    ])
    
    with tab1:
        dashboard_page()
    
    with tab2:
        admissions_page()
    
    with tab3:
        students_page()
    
    with tab4:
        teachers_page()
    
    with tab5:
        academics_page()
    
    with tab6:
        finance_page()
    
    with tab7:
        analytics_page()
    
    with tab8:
        ai_insights_page()
        
    with tab9:
        streams_page()
        
    with tab10:
        data_input_page()
                   
def dashboard_page():
    #st.header("📊 School Dashboard")
    
    db_manager = DatabaseManager()
    
    # Get currency symbol from session state
    currency_symbol = st.session_state.get('currency_symbol', '₵')
    
    # Get comprehensive data from database
    try:
        # Student metrics
        students_query = """
            SELECT student_id, name, age, grade, gpa, behavior_score, status, stream_id
            FROM students 
            WHERE status = 'active'
        """
        students_data = db_manager.execute_query(students_query)
        
        # Teacher metrics
        teachers_query = """
            SELECT teacher_id, name, subject, department, performance_score, salary, status
            FROM teachers 
            WHERE status = 'active'
        """
        teachers_data = db_manager.execute_query(teachers_query)
        
        # Course metrics
        courses_query = "SELECT COUNT(*) FROM courses"
        total_courses = db_manager.execute_query(courses_query)[0][0]
        
        # Stream metrics
        streams_query = """
            SELECT s.stream_id, s.grade_level, s.stream_type, s.max_capacity,
                   COUNT(st.student_id) as current_students
            FROM streams s
            LEFT JOIN students st ON s.stream_id = st.stream_id AND st.status = 'active'
            GROUP BY s.stream_id, s.grade_level, s.stream_type, s.max_capacity
        """
        streams_data = db_manager.execute_query(streams_query)
        
        # Financial metrics
        finance_query = """
            SELECT 
                SUM(CASE WHEN type = 'tuition' THEN amount ELSE 0 END) as tuition_revenue,
                SUM(CASE WHEN type = 'fees' THEN amount ELSE 0 END) as fees_revenue,
                SUM(amount) as total_revenue,
                COUNT(*) as total_transactions,
                date,
                type,
                amount
            FROM finance
        """
        finance_data = db_manager.execute_query(finance_query)
        
        # Monthly financial data for trends
        monthly_finance_query = """
            SELECT 
                strftime('%Y-%m', date) as month,
                SUM(amount) as monthly_revenue,
                COUNT(*) as monthly_transactions
            FROM finance
            GROUP BY strftime('%Y-%m', date)
            ORDER BY month DESC
            LIMIT 12
        """
        monthly_finance_data = db_manager.execute_query(monthly_finance_query)
        
        # Recent attendance data
        recent_attendance_query = """
            SELECT 
                COUNT(DISTINCT student_id) as students_with_records,
                AVG(CASE WHEN status = 'present' THEN 100.0 ELSE 0.0 END) as avg_attendance_rate
            FROM attendance
            WHERE date >= date('now', '-7 days')
        """
        attendance_data = db_manager.execute_query(recent_attendance_query)
        
        # Daily attendance trends
        daily_attendance_query = """
            SELECT 
                date,
                COUNT(*) as total_records,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present_count,
                ROUND(AVG(CASE WHEN status = 'present' THEN 100.0 ELSE 0.0 END), 1) as attendance_rate
            FROM attendance
            WHERE date >= date('now', '-30 days')
            GROUP BY date
            ORDER BY date
        """
        daily_attendance_data = db_manager.execute_query(daily_attendance_query)
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        return
    
    # Calculate attendance rates for students
    attendance_rates = {}
    if students_data:
        for student in students_data:
            student_id = student[0]
            total_query = "SELECT COUNT(*) FROM attendance WHERE student_id = ?"
            total_days = db_manager.execute_query(total_query, (student_id,))[0][0]
            
            if total_days > 0:
                present_query = "SELECT COUNT(*) FROM attendance WHERE student_id = ? AND status = 'present'"
                present_days = db_manager.execute_query(present_query, (student_id,))[0][0]
                attendance_rates[student_id] = (present_days / total_days) * 100
            else:
                attendance_rates[student_id] = 100.0
    
    # Pre-calculate all metrics
    total_students = len(students_data) if students_data else 0
    avg_gpa = sum(student[4] for student in students_data) / len(students_data) if students_data else 0
    avg_attendance = sum(attendance_rates.values()) / len(attendance_rates) if attendance_rates else 0
    total_teachers = len(teachers_data) if teachers_data else 0
    high_performers = len([s for s in students_data if s[4] >= 3.5]) if students_data else 0
    at_risk = len([s for s in students_data if s[4] < 2.5 or attendance_rates.get(s[0], 100) < 75]) if students_data and attendance_rates else 0
    total_revenue = finance_data[0][2] if finance_data and finance_data[0][2] else 0
    tuition_revenue = finance_data[0][0] if finance_data and finance_data[0][0] else 0
    fees_revenue = finance_data[0][1] if finance_data and finance_data[0][1] else 0
    total_transactions = finance_data[0][3] if finance_data and finance_data[0][3] else 0
    
    # Key Performance Indicators
    st.subheader("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Students", total_students)
    
    with col2:
        st.metric("Average GPA", f"{avg_gpa:.2f}")
    
    with col3:
        st.metric("Average Attendance", f"{avg_attendance:.1f}%")
    
    with col4:
        st.metric("Active Teachers", total_teachers)
    
    with col5:
        st.metric("Total Courses", total_courses)
    
    # Secondary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_streams = len(streams_data) if streams_data else 0
        st.metric("Active Streams", total_streams)
    
    with col2:
        st.metric("High Performers", high_performers, delta=f"{(high_performers/total_students)*100:.1f}%" if total_students > 0 else "0%")
    
    with col3:
        st.metric("At-Risk Students", at_risk, delta=f"{(at_risk/total_students)*100:.1f}%" if total_students > 0 else "0%")
    
    with col4:
        st.metric("Total Revenue", f"{currency_symbol}{total_revenue:,.2f}")
    
    st.divider()
    
    # Visual Analytics Section
    st.subheader("📈 Visual Analytics")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Academic", "👥 Demographics", "💰 Financial", "📅 Attendance", "🏢 Operations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # GPA Distribution Chart
            if students_data:
                gpa_ranges = ["4.0+", "3.5-3.9", "3.0-3.4", "2.5-2.9", "2.0-2.4", "<2.0"]
                gpa_counts = [
                    len([s for s in students_data if s[4] >= 4.0]),
                    len([s for s in students_data if 3.5 <= s[4] < 4.0]),
                    len([s for s in students_data if 3.0 <= s[4] < 3.5]),
                    len([s for s in students_data if 2.5 <= s[4] < 3.0]),
                    len([s for s in students_data if 2.0 <= s[4] < 2.5]),
                    len([s for s in students_data if s[4] < 2.0])
                ]
                
                gpa_df = pd.DataFrame({
                    'GPA Range': gpa_ranges,
                    'Number of Students': gpa_counts
                })
                
                fig_gpa = px.bar(gpa_df, x='GPA Range', y='Number of Students', 
                               title="GPA Distribution",
                               color='Number of Students',
                               color_continuous_scale='viridis')
                fig_gpa.update_layout(height=400)
                st.plotly_chart(fig_gpa, width="stretch")
        
        with col2:
            # Performance Categories Pie Chart
            if students_data:
                excellent = len([s for s in students_data if s[4] >= 3.7])
                good = len([s for s in students_data if 3.0 <= s[4] < 3.7])
                satisfactory = len([s for s in students_data if 2.5 <= s[4] < 3.0])
                needs_improvement = len([s for s in students_data if s[4] < 2.5])
                
                performance_df = pd.DataFrame({
                    'Category': ['Excellent', 'Good', 'Satisfactory', 'Needs Improvement'],
                    'Count': [excellent, good, satisfactory, needs_improvement]
                })
                
                fig_performance = px.pie(performance_df, values='Count', names='Category',
                                       title="Academic Performance Distribution",
                                       color_discrete_sequence=px.colors.qualitative.Set3)
                fig_performance.update_layout(height=400)
                st.plotly_chart(fig_performance, width="stretch")
        
        # Grade-wise Performance Analysis
        if students_data:
            grade_performance = {}
            for student in students_data:
                grade = student[3]
                gpa = student[4]
                if grade not in grade_performance:
                    grade_performance[grade] = []
                grade_performance[grade].append(gpa)
            
            grade_avg_gpa = {grade: sum(gpas)/len(gpas) for grade, gpas in grade_performance.items()}
            
            grade_df = pd.DataFrame({
                'Grade': list(grade_avg_gpa.keys()),
                'Average GPA': list(grade_avg_gpa.values())
            })
            
            fig_grade = px.line(grade_df, x='Grade', y='Average GPA',
                              title="Average GPA by Grade Level",
                              markers=True, line_shape='spline')
            fig_grade.update_layout(height=400)
            st.plotly_chart(fig_grade, width="stretch")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution
            if students_data:
                ages = [student[2] for student in students_data]
                age_df = pd.DataFrame({'Age': ages})
                
                fig_age = px.histogram(age_df, x='Age', nbins=15,
                                     title="Student Age Distribution",
                                     color_discrete_sequence=['#FF6B6B'])
                fig_age.update_layout(height=400)
                st.plotly_chart(fig_age, width="stretch")
        
        with col2:
            # Grade Level Distribution
            if students_data:
                grades = [student[3] for student in students_data]
                grade_counts = pd.Series(grades).value_counts().sort_index()
                
                grade_df = pd.DataFrame({
                    'Grade': grade_counts.index,
                    'Students': grade_counts.values
                })
                
                fig_grades = px.bar(grade_df, x='Grade', y='Students',
                                  title="Students by Grade Level",
                                  color='Students',
                                  color_continuous_scale='blues')
                fig_grades.update_layout(height=400)
                st.plotly_chart(fig_grades, width="stretch")
        
        # Teacher Department Distribution
        if teachers_data:
            departments = [teacher[3] for teacher in teachers_data if teacher[3]]
            if departments:
                dept_counts = pd.Series(departments).value_counts()
                
                dept_df = pd.DataFrame({
                    'Department': dept_counts.index,
                    'Teachers': dept_counts.values
                })
                
                fig_dept = px.pie(dept_df, values='Teachers', names='Department',
                                title="Teacher Distribution by Department",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_dept.update_layout(height=400)
                st.plotly_chart(fig_dept, width="stretch")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue Breakdown
            if tuition_revenue > 0 or fees_revenue > 0:
                revenue_df = pd.DataFrame({
                    'Revenue Type': ['Tuition', 'Fees'],
                    'Amount': [tuition_revenue, fees_revenue]
                })
                
                fig_revenue = px.pie(revenue_df, values='Amount', names='Revenue Type',
                                   title="Revenue Breakdown",
                                   color_discrete_sequence=['#4CAF50', '#2196F3'])
                fig_revenue.update_layout(height=400)
                st.plotly_chart(fig_revenue, width="stretch")
        
        with col2:
            # Monthly Revenue Trend
            if monthly_finance_data:
                monthly_df = pd.DataFrame(monthly_finance_data, 
                                        columns=['Month', 'Revenue', 'Transactions'])
                monthly_df = monthly_df.sort_values('Month')
                
                fig_monthly = px.line(monthly_df, x='Month', y='Revenue',
                                    title="Monthly Revenue Trend",
                                    markers=True, line_shape='spline')
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, width="stretch")
        
        # Financial Health Indicators
        if total_revenue > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Average transaction amount gauge
                avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_transaction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Avg Transaction ({currency_symbol})"},
                    gauge = {'axis': {'range': [None, avg_transaction * 2]},
                             'bar': {'color': "darkblue"},
                             'steps': [
                                 {'range': [0, avg_transaction * 0.5], 'color': "lightgray"},
                                 {'range': [avg_transaction * 0.5, avg_transaction * 1.5], 'color': "gray"}],
                             'threshold': {'line': {'color': "red", 'width': 4},
                                         'thickness': 0.75, 'value': avg_transaction * 1.2}}))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, width="stretch")
            
            with col2:
                # Revenue growth indicator
                if len(monthly_finance_data) >= 2:
                    recent_revenue = monthly_finance_data[0][1]
                    previous_revenue = monthly_finance_data[1][1]
                    
                    fig_growth = go.Figure(go.Indicator(
                        mode = "number+delta",
                        value = recent_revenue,
                        delta = {'reference': previous_revenue, 'valueformat': '.0f'},
                        title = {'text': f"Monthly Revenue ({currency_symbol})"},
                    ))
                    fig_growth.update_layout(height=300)
                    st.plotly_chart(fig_growth, width="stretch")
            
            with col3:
                st.metric("Total Transactions", total_transactions)
                st.metric("Revenue per Student", f"{currency_symbol}{total_revenue/total_students:.2f}" if total_students > 0 else "N/A")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Attendance Rate Distribution
            if attendance_rates:
                attendance_ranges = ["95-100%", "90-94%", "85-89%", "80-84%", "75-79%", "<75%"]
                attendance_counts = [
                    len([r for r in attendance_rates.values() if r >= 95]),
                    len([r for r in attendance_rates.values() if 90 <= r < 95]),
                    len([r for r in attendance_rates.values() if 85 <= r < 90]),
                    len([r for r in attendance_rates.values() if 80 <= r < 85]),
                    len([r for r in attendance_rates.values() if 75 <= r < 80]),
                    len([r for r in attendance_rates.values() if r < 75])
                ]
                
                attendance_dist_df = pd.DataFrame({
                    'Attendance Range': attendance_ranges,
                    'Number of Students': attendance_counts
                })
                
                fig_att_dist = px.bar(attendance_dist_df, x='Attendance Range', y='Number of Students',
                                    title="Attendance Rate Distribution",
                                    color='Number of Students',
                                    color_continuous_scale='RdYlGn')
                fig_att_dist.update_layout(height=400)
                st.plotly_chart(fig_att_dist, width="stretch")
        
        with col2:
            # Daily Attendance Trend
            if daily_attendance_data:
                daily_df = pd.DataFrame(daily_attendance_data, 
                                      columns=['Date', 'Total Records', 'Present Count', 'Attendance Rate'])
                
                fig_daily = px.line(daily_df, x='Date', y='Attendance Rate',
                                  title="Daily Attendance Trend (Last 30 Days)",
                                  markers=True, line_shape='spline')
                fig_daily.update_layout(height=400)
                st.plotly_chart(fig_daily, width="stretch")
        
        # Attendance vs Academic Performance Correlation
        if students_data and attendance_rates:
            correlation_data = []
            for student in students_data:
                student_id = student[0]
                gpa = student[4]
                attendance = attendance_rates.get(student_id, 100)
                correlation_data.append({'GPA': gpa, 'Attendance Rate': attendance})
            
            corr_df = pd.DataFrame(correlation_data)
            
            fig_corr = px.scatter(corr_df, x='Attendance Rate', y='GPA',
                                title="Attendance vs Academic Performance Correlation",
                                trendline="ols")
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, width="stretch")
    
    with tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stream Capacity Utilization
            if streams_data:
                stream_df = pd.DataFrame(streams_data, 
                                       columns=['Stream ID', 'Grade Level', 'Stream Type', 'Max Capacity', 'Current Students'])
                stream_df['Utilization %'] = (stream_df['Current Students'] / stream_df['Max Capacity'] * 100).round(1)
                
                fig_streams = px.bar(stream_df, x='Stream ID', y='Utilization %',
                                   title="Stream Capacity Utilization",
                                   color='Utilization %',
                                   color_continuous_scale='RdYlGn_r')
                fig_streams.add_hline(y=100, line_dash="dash", line_color="red", 
                                    annotation_text="Full Capacity")
                fig_streams.update_layout(height=400)
                st.plotly_chart(fig_streams, width="stretch")
        
        with col2:
            # Teacher Performance Distribution
            if teachers_data:
                performance_scores = [teacher[4] for teacher in teachers_data if teacher[4] is not None]
                if performance_scores:
                    perf_df = pd.DataFrame({'Performance Score': performance_scores})
                    
                    fig_perf = px.histogram(perf_df, x='Performance Score', nbins=10,
                                          title="Teacher Performance Score Distribution",
                                          color_discrete_sequence=['#FF9800'])
                    fig_perf.update_layout(height=400)
                    st.plotly_chart(fig_perf, width="stretch")
        
        # System Health Dashboard
        st.subheader("🏥 System Health Dashboard")
        
        health_col1, health_col2, health_col3, health_col4 = st.columns(4)
        
        with health_col1:
            # Student-Teacher Ratio Gauge
            teacher_student_ratio = total_students / total_teachers if total_teachers > 0 else 0
            
            fig_ratio = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = teacher_student_ratio,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Student:Teacher Ratio"},
                gauge = {'axis': {'range': [None, 40]},
                         'bar': {'color': "darkgreen" if teacher_student_ratio <= 20 else "orange" if teacher_student_ratio <= 30 else "red"},
                         'steps': [
                             {'range': [0, 20], 'color': "lightgreen"},
                             {'range': [20, 30], 'color': "yellow"},
                             {'range': [30, 40], 'color': "lightcoral"}],
                         'threshold': {'line': {'color': "red", 'width': 4},
                                     'thickness': 0.75, 'value': 25}}))
            fig_ratio.update_layout(height=300)
            st.plotly_chart(fig_ratio, width="stretch")
        
        with health_col2:
            # Overall Capacity Utilization
            if streams_data:
                total_capacity = sum(stream[3] for stream in streams_data)
                total_enrolled = sum(stream[4] for stream in streams_data)
                utilization = (total_enrolled / total_capacity * 100) if total_capacity > 0 else 0
                
                fig_util = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = utilization,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Capacity Utilization %"},
                    gauge = {'axis': {'range': [None, 120]},
                             'bar': {'color': "darkblue"},
                             'steps': [
                                 {'range': [0, 75], 'color': "lightgray"},
                                 {'range': [75, 90], 'color': "yellow"},
                                 {'range': [90, 100], 'color': "lightgreen"},
                                 {'range': [100, 120], 'color': "red"}],
                             'threshold': {'line': {'color': "red", 'width': 4},
                                         'thickness': 0.75, 'value': 100}}))
                fig_util.update_layout(height=300)
                st.plotly_chart(fig_util, width="stretch")
        
        with health_col3:
            # Average GPA Gauge
            fig_gpa_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_gpa,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average GPA"},
                gauge = {'axis': {'range': [None, 4.0]},
                         'bar': {'color': "purple"},
                         'steps': [
                             {'range': [0, 2.0], 'color': "lightcoral"},
                             {'range': [2.0, 3.0], 'color': "yellow"},
                             {'range': [3.0, 4.0], 'color': "lightgreen"}],
                         'threshold': {'line': {'color': "red", 'width': 4},
                                     'thickness': 0.75, 'value': 3.5}}))
            fig_gpa_gauge.update_layout(height=300)
            st.plotly_chart(fig_gpa_gauge, width="stretch")
        
        with health_col4:
            # Attendance Rate Gauge
            fig_att_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_attendance,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Avg Attendance %"},
                gauge = {'axis': {'range': [None, 100]},
                         'bar': {'color': "teal"},
                         'steps': [
                             {'range': [0, 70], 'color': "lightcoral"},
                             {'range': [70, 85], 'color': "yellow"},
                             {'range': [85, 100], 'color': "lightgreen"}],
                         'threshold': {'line': {'color': "red", 'width': 4},
                                     'thickness': 0.75, 'value': 90}}))
            fig_att_gauge.update_layout(height=300)
            st.plotly_chart(fig_att_gauge, width="stretch")
    
    st.divider()
    
    # System Health & Status
    st.subheader("🏥 System Health & Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Stream capacity utilization
        if streams_data:
            total_capacity = sum(stream[3] for stream in streams_data)
            total_enrolled = sum(stream[4] for stream in streams_data)
            utilization = (total_enrolled / total_capacity * 100) if total_capacity > 0 else 0
            
            if utilization > 90:
                st.metric("Capacity Utilization", f"{utilization:.1f}%", delta="Near Full", delta_color="inverse")
            elif utilization > 75:
                st.metric("Capacity Utilization", f"{utilization:.1f}%", delta="Good", delta_color="normal")
            else:
                st.metric("Capacity Utilization", f"{utilization:.1f}%", delta="Available", delta_color="normal")
        else:
            st.metric("Capacity Utilization", "0%")
    
    with col2:
        # Teacher workload
        teacher_student_ratio = total_students / total_teachers if total_teachers > 0 else 0
        if teacher_student_ratio > 25:
            st.metric("Student-Teacher Ratio", f"{teacher_student_ratio:.1f}:1", delta="High Load", delta_color="inverse")
        elif teacher_student_ratio > 20:
            st.metric("Student-Teacher Ratio", f"{teacher_student_ratio:.1f}:1", delta="Moderate", delta_color="normal")
        else:
            st.metric("Student-Teacher Ratio", f"{teacher_student_ratio:.1f}:1", delta="Optimal", delta_color="normal")
    
    with col3:
        # System activity
        if attendance_data and attendance_data[0][0]:
            students_with_recent_activity = attendance_data[0][0]
            activity_rate = (students_with_recent_activity / total_students * 100) if total_students > 0 else 0
            st.metric("Recent Activity", f"{activity_rate:.1f}%", delta="Last 7 days")
        else:
            st.metric("Recent Activity", "0%", delta="No recent data")
    
    st.divider()
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
#---------------------------------------------    
    with col1:
            if st.button("📊 Generate Analytics Report", type="primary", width="stretch"):
                with st.spinner("Generating comprehensive analytics report..."):
                    report_data = {
                        'academic_performance': {
                            'total_students': total_students,
                            'average_gpa': round(avg_gpa, 2),
                            'high_performers': high_performers,
                            'high_performer_percentage': round((high_performers/total_students)*100, 1) if total_students > 0 else 0,
                            'at_risk_students': at_risk,
                            'at_risk_percentage': round((at_risk/total_students)*100, 1) if total_students > 0 else 0
                        },
                        'attendance_metrics': {
                            'average_attendance_rate': round(avg_attendance, 1),
                            'students_with_perfect_attendance': len([rate for rate in attendance_rates.values() if rate >= 95]) if attendance_rates else 0,
                            'students_with_low_attendance': len([rate for rate in attendance_rates.values() if rate < 75]) if attendance_rates else 0
                        },
                        'staffing_metrics': {
                            'total_teachers': total_teachers,
                            'student_teacher_ratio': round(teacher_student_ratio, 1),
                            'capacity_utilization': round(utilization, 1) if streams_data else 0
                        },
                        'financial_summary': {
                            'total_revenue': total_revenue,
                            'tuition_revenue': tuition_revenue,
                            'fees_revenue': fees_revenue,
                            'total_transactions': total_transactions,
                            'average_transaction_amount': round(total_revenue / total_transactions, 2) if total_transactions > 0 else 0
                        },
                        'system_health': {
                            'total_courses': total_courses,
                            'active_streams': total_streams,
                            'recent_activity_rate': round(activity_rate, 1) if attendance_data and attendance_data[0][0] else 0
                        }
                    }
                    
                    st.success("✅ Analytics report generated successfully!")
                    
                    with st.expander("📋 Comprehensive Analytics Report", expanded=True):
                        st.markdown("### 📚 Academic Performance")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Students", report_data['academic_performance']['total_students'])
                            st.metric("Average GPA", report_data['academic_performance']['average_gpa'])
                        with col_b:
                            st.metric("High Performers", 
                                    f"{report_data['academic_performance']['high_performers']} ({report_data['academic_performance']['high_performer_percentage']}%)")
                        with col_c:
                            st.metric("At-Risk Students", 
                                    f"{report_data['academic_performance']['at_risk_students']} ({report_data['academic_performance']['at_risk_percentage']}%)")
                        
                        st.markdown("### 📅 Attendance Analysis")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Average Attendance", f"{report_data['attendance_metrics']['average_attendance_rate']}%")
                        with col_b:
                            st.metric("Perfect Attendance", report_data['attendance_metrics']['students_with_perfect_attendance'])
                        with col_c:
                            st.metric("Low Attendance", report_data['attendance_metrics']['students_with_low_attendance'])
                        
                        st.markdown("### 💰 Financial Summary")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Total Revenue", f"{currency_symbol}{report_data['financial_summary']['total_revenue']:,.2f}")
                            st.metric("Tuition Revenue", f"{currency_symbol}{report_data['financial_summary']['tuition_revenue']:,.2f}")
                        with col_b:
                            st.metric("Fees Revenue", f"{currency_symbol}{report_data['financial_summary']['fees_revenue']:,.2f}")
                            st.metric("Avg Transaction", f"{currency_symbol}{report_data['financial_summary']['average_transaction_amount']:.2f}")
                        
                        st.markdown("### 🏥 System Health")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Student-Teacher Ratio", f"{report_data['staffing_metrics']['student_teacher_ratio']}:1")
                        with col_b:
                            st.metric("Capacity Utilization", f"{report_data['staffing_metrics']['capacity_utilization']}%")
                        with col_c:
                            st.metric("Recent Activity", f"{report_data['system_health']['recent_activity_rate']}%")
                        
                        st.markdown("### 📥 Export Options")
                        report_json = json.dumps(report_data, indent=2)
                        st.download_button(
                            label="📥 Download Report as JSON",
                            data=report_json,
                            file_name=f"school_analytics_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        
    with col2:
        if st.button("🎯 Identify At-Risk Students", width="stretch"):
            if students_data:
                at_risk_students = []
                for student in students_data:
                    student_id, name, age, grade, gpa, behavior_score = student[:6]
                    attendance_rate = attendance_rates.get(student_id, 100)
                    
                    if gpa < 2.5 or attendance_rate < 75 or behavior_score < 70:
                        at_risk_students.append({
                            'name': name,
                            'grade': grade,
                            'gpa': gpa,
                            'attendance': attendance_rate,
                            'behavior': behavior_score
                        })
                
                if at_risk_students:
                    st.warning(f"⚠️ Found {len(at_risk_students)} at-risk students")
                    with st.expander("View At-Risk Students"):
                        at_risk_df = pd.DataFrame(at_risk_students)
                        
                        fig_atrisk = px.scatter(at_risk_df, x='attendance', y='gpa', 
                                              size='behavior', hover_name='name',
                                              title="At-Risk Students Analysis",
                                              labels={'attendance': 'Attendance Rate (%)', 'gpa': 'GPA'},
                                              color='grade')
                        fig_atrisk.add_hline(y=2.5, line_dash="dash", line_color="red", 
                                           annotation_text="Minimum GPA")
                        fig_atrisk.add_vline(x=75, line_dash="dash", line_color="red", 
                                           annotation_text="Minimum Attendance")
                        st.plotly_chart(fig_atrisk, width="stretch")
                        
                        for student in at_risk_students[:5]:
                            st.write(f"• **{student['name']}** (Grade {student['grade']}) - GPA: {student['gpa']:.2f}, Attendance: {student['attendance']:.1f}%")
                        if len(at_risk_students) > 5:
                            st.write(f"... and {len(at_risk_students) - 5} more")
                else:
                    st.success("✅ No at-risk students identified!")
            else:
                st.info("No student data available")
    
    with col3:
        if st.button("💰 Financial Summary", width="stretch"):
            st.success("💰 Financial summary generated!")
            with st.expander("💳 Financial Breakdown"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Tuition Revenue", f"{currency_symbol}{tuition_revenue:,.2f}")
                    st.metric("Total Transactions", total_transactions)
                with col_b:
                    st.metric("Fees Revenue", f"{currency_symbol}{fees_revenue:,.2f}")
                    avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
                    st.metric("Avg Transaction", f"{currency_symbol}{avg_transaction:.2f}")
                
                if monthly_finance_data:
                    monthly_df = pd.DataFrame(monthly_finance_data, 
                                            columns=['Month', 'Revenue', 'Transactions'])
                    monthly_df = monthly_df.sort_values('Month')
                    
                    fig_trend = px.area(monthly_df, x='Month', y='Revenue',
                                      title="Revenue Trend Over Time",
                                      color_discrete_sequence=['#4CAF50'])
                    st.plotly_chart(fig_trend, width="stretch")
    
    with col4:
        if st.button("🔄 System Health Check", width="stretch"):
            with st.spinner("Running system health check..."):
                health_issues = []
                
                if not students_data:
                    health_issues.append("⚠️ No active students found")
                if not teachers_data:
                    health_issues.append("⚠️ No active teachers found")
                if not streams_data:
                    health_issues.append("⚠️ No streams configured")
                
                if students_data and attendance_rates:
                    low_attendance_count = len([rate for rate in attendance_rates.values() if rate < 70])
                    if low_attendance_count > total_students * 0.2:
                        health_issues.append(f"⚠️ High absenteeism: {low_attendance_count} students")
                
                if students_data:
                    low_gpa_count = len([s for s in students_data if s[4] < 2.0])
                    if low_gpa_count > total_students * 0.15:
                        health_issues.append(f"⚠️ Academic concerns: {low_gpa_count} students with GPA < 2.0")
                
                if health_issues:
                    st.warning(f"🚨 Found {len(health_issues)} system issues")
                    with st.expander("View Health Issues"):
                        for issue in health_issues:
                            st.write(issue)
                        
                        health_categories = ['Data Integrity', 'Academic Performance', 'Attendance', 'System Resources']
                        health_scores = [
                            100 if students_data and teachers_data and streams_data else 50,
                            (100 - (low_gpa_count / total_students * 100)) if students_data else 100,
                            avg_attendance if attendance_rates else 100,
                            min(100, (100 - utilization)) if streams_data else 100
                        ]
                        
                        health_df = pd.DataFrame({
                            'Category': health_categories,
                            'Health Score': health_scores
                        })
                        
                        fig_health = px.bar(health_df, x='Category', y='Health Score',
                                          title="System Health Breakdown",
                                          color='Health Score',
                                          color_continuous_scale='RdYlGn')
                        fig_health.add_hline(y=75, line_dash="dash", line_color="orange", 
                                           annotation_text="Warning Threshold")
                        st.plotly_chart(fig_health, width="stretch")
                else:
                    st.success("✅ All systems operating normally!")
                    
                    health_metrics = {
                        'System Uptime': '99.9%',
                        'Data Quality': 'Excellent',
                        'Performance': 'Optimal',
                        'Security': 'Secure'
                    }
                    
                    for metric, value in health_metrics.items():
                        st.metric(metric, value)
    
    st.divider()
    
    st.subheader("📈 Recent Activity & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📚 Academic Performance Insights**")
        if students_data:
            top_students = sorted(students_data, key=lambda x: x[4], reverse=True)[:5]
            
            top_df = pd.DataFrame({
                'Student': [f"{student[1][:15]}..." if len(student[1]) > 15 else student[1] for student in top_students],
                'GPA': [student[4] for student in top_students]
            })
            
            fig_top = px.bar(top_df, x='GPA', y='Student', orientation='h',
                           title="Top 5 Performers",
                           color='GPA', color_continuous_scale='viridis')
            fig_top.update_layout(height=300)
            st.plotly_chart(fig_top, width="stretch")
        else:
            st.write("No student data available")
    
    with col2:
        st.markdown("**👥 System Statistics & Trends**")
        if streams_data:
            stream_util_df = pd.DataFrame({
                'Stream': [f"Stream {stream[0]}" for stream in streams_data[:5]],
                'Utilization': [(stream[4] / stream[3] * 100) if stream[3] > 0 else 0 for stream in streams_data[:5]]
            })
            
            fig_util_mini = px.line(stream_util_df, x='Stream', y='Utilization',
                                  title="Stream Utilization Trend",
                                  markers=True)
            fig_util_mini.add_hline(y=100, line_dash="dash", line_color="red")
            fig_util_mini.update_layout(height=300)
            st.plotly_chart(fig_util_mini, width="stretch")
        else:
            st.write("No stream data available")
    
    st.subheader("🔮 Predictive Insights & Recommendations")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("**📊 Enrollment Projections**")
        if streams_data:
            total_capacity = sum(stream[3] for stream in streams_data)
            total_enrolled = sum(stream[4] for stream in streams_data)
            projected_enrollment = total_enrolled * 1.05
            
            projection_df = pd.DataFrame({
                'Period': ['Current', 'Projected (Next Term)'],
                'Enrollment': [total_enrolled, min(projected_enrollment, total_capacity)]
            })
            
            fig_proj = px.bar(projection_df, x='Period', y='Enrollment',
                            title="Enrollment Projection",
                            color='Enrollment')
            fig_proj.add_hline(y=total_capacity, line_dash="dash", line_color="red",
                             annotation_text="Max Capacity")
            st.plotly_chart(fig_proj, width="stretch")
    
    with insight_col2:
        st.markdown("**💡 Academic Recommendations**")
        recommendations = []
        
        if students_data:
            low_performers = len([s for s in students_data if s[4] < 2.5])
            if low_performers > 0:
                recommendations.append(f"• Focus on {low_performers} students with GPA < 2.5")
            
            if attendance_rates:
                low_attendance = len([r for r in attendance_rates.values() if r < 80])
                if low_attendance > 0:
                    recommendations.append(f"• Implement attendance intervention for {low_attendance} students")
            
            if avg_gpa < 3.0:
                recommendations.append("• Consider additional academic support programs")
            
        if not recommendations:
            recommendations.append("• Maintain current excellent performance standards")
            recommendations.append("• Consider advanced programs for high achievers")
        
        for rec in recommendations[:4]:
            st.write(rec)
    
    with insight_col3:
        st.markdown("**🎯 Action Items**")
        action_items = []
        
        if streams_data:
            overcrowded = [s for s in streams_data if (s[4] / s[3]) > 0.95]
            if overcrowded:
                action_items.append(f"• Review capacity for {len(overcrowded)} streams")
        
        if teacher_student_ratio > 25:
            action_items.append("• Consider hiring additional teachers")
        
        if monthly_finance_data and len(monthly_finance_data) >= 2:
            recent_revenue = monthly_finance_data[0][1]
            previous_revenue = monthly_finance_data[1][1]
            if recent_revenue < previous_revenue * 0.9:
                action_items.append("• Investigate revenue decline")
        
        if not action_items:
            action_items.append("• Continue monitoring key metrics")
            action_items.append("• Plan for next academic term")
        
        for item in action_items[:4]:
            st.write(item)
    
    if students_data or teachers_data:
        st.subheader("🔔 Smart Alerts & Notifications")
        
        alerts = []
        
        if students_data:
            failing_students = len([s for s in students_data if s[4] < 2.0])
            if failing_students > 0:
                alerts.append(f"🚨 {failing_students} students with GPA below 2.0 need immediate attention")
            
            if attendance_rates:
                chronic_absentees = len([rate for rate in attendance_rates.values() if rate < 60])
                if chronic_absentees > 0:
                    alerts.append(f"⚠️ {chronic_absentees} students with attendance below 60%")
        
        if streams_data:
            overcrowded_streams = [s for s in streams_data if s[4] > s[3]]
            if overcrowded_streams:
                alerts.append(f"📊 {len(overcrowded_streams)} streams are over capacity")
        
        if teachers_data:
            if teacher_student_ratio > 30:
                alerts.append("👥 Teacher-student ratio is critically high")
        
        if monthly_finance_data and len(monthly_finance_data) >= 2:
            recent_revenue = monthly_finance_data[0][1]
            previous_revenue = monthly_finance_data[1][1]
            if recent_revenue < previous_revenue * 0.85:
                alerts.append("💰 Significant revenue decline detected")
        
        if alerts:
            alert_col1, alert_col2 = st.columns([3, 1])
            with alert_col1:
                for alert in alerts:
                    st.warning(alert)
            with alert_col2:
                alert_types = ['Academic', 'Attendance', 'Capacity', 'Financial']
                alert_counts = [
                    len([a for a in alerts if '🚨' in a or 'GPA' in a]),
                    len([a for a in alerts if 'attendance' in a.lower()]),
                    len([a for a in alerts if 'capacity' in a.lower() or 'streams' in a.lower()]),
                    len([a for a in alerts if '💰' in a or 'revenue' in a.lower()])
                ]
                
                if sum(alert_counts) > 0:
                    alert_df = pd.DataFrame({
                        'Type': alert_types,
                        'Count': alert_counts
                    })
                    alert_df = alert_df[alert_df['Count'] > 0]
                    
                    fig_alerts = px.pie(alert_df, values='Count', names='Type',
                                      title="Alert Categories")
                    fig_alerts.update_layout(height=200)
                    st.plotly_chart(fig_alerts, width="stretch")
        else:
            st.success("✅ No critical alerts at this time")
            
            positive_metrics = []
            if avg_gpa >= 3.0:
                positive_metrics.append("📚 Excellent academic performance maintained")
            if avg_attendance >= 85:
                positive_metrics.append("📅 Strong attendance rates across all grades")
            if teacher_student_ratio <= 25:
                positive_metrics.append("👥 Optimal teacher-student ratio")
            
            for metric in positive_metrics:
                st.success(metric)
    
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.caption(f"📅 Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with footer_col2:
        st.caption(f"📊 Data points: {total_students + total_teachers + total_courses}")
    
    with footer_col3:
        st.caption(f"🔄 System status: {'🟢 Online' if students_data or teachers_data else '🔴 No Data'}")

def diagnose_ai_issues(manager):
    """Helper function to diagnose AI configuration issues"""
    issues = []
    suggestions = []
    
    if not hasattr(manager, 'ai') or not manager.ai:
        issues.append("AI system not initialized")
        suggestions.append("Restart the application")
        return issues, suggestions
    
    ai = manager.ai
    
    connection_status = ai._test_connection()
    if "❌" in connection_status:
        issues.append(f"Connection failed: {connection_status}")
        
        if hasattr(ai, 'provider') and ai.provider.name == 'OLLAMA':
            suggestions.extend([
                "Start Ollama server: `ollama serve`",
                f"Verify model exists: `ollama pull {ai.model}`",
                "Check server URL in settings"
            ])
        else:
            suggestions.extend([
                "Check API key configuration",
                "Verify internet connection",
                "Confirm model availability"
            ])
    
    if hasattr(ai, 'provider') and ai.provider.name == 'OLLAMA':
        try:
            models = ai.get_ollama_models()
            if ai.model not in models:
                issues.append(f"Model '{ai.model}' not found in available models")
                suggestions.append(f"Pull the model: `ollama pull {ai.model}`")
        except:
            issues.append("Could not retrieve model list")
            suggestions.append("Check Ollama server connection")
    
    return issues, suggestions

def admissions_page():
    st.header("👥 AI-Powered Admissions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 New Application")
        
        if 'school_manager' not in st.session_state:
            st.error("School management system not initialized. Please restart the application.")
            return
        
        manager = st.session_state.school_manager
        
        # Ensure AI is properly configured
        if hasattr(manager, 'ai') and manager.ai:
            manager.ai.sync_with_session_state()
        else:
            st.error("AI system not initialized.")
            return
        
        with st.form("admission_form"):
            name = st.text_input("Student Name*", placeholder="Enter full name")
            
            # Date of Birth and Age section
            col_dob, col_age = st.columns([2, 1])
            
            with col_dob:
                from datetime import datetime, date
                # Default to 10 years ago for convenience
                default_dob = date.today().replace(year=date.today().year - 10)
                date_of_birth = st.date_input(
                    "Date of Birth*", 
                    value=default_dob,
                    min_value=date(1990, 1, 1),
                    max_value=date.today(),
                    help="Student's actual date of birth"
                )
            
            with col_age:
                # Calculate age automatically from DOB
                if date_of_birth:
                    today = date.today()
                    age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
                    st.metric("Age", f"{age} years")
                else:
                    age = 6
            
            previous_gpa = st.number_input("Previous School GPA", min_value=0.0, max_value=4.0, value=3.0)
            activities = st.text_area("Extracurricular Activities", placeholder="List activities")
            parent_contact = st.text_input("Parent Contact*", placeholder="Email or phone")
            medical_info = st.text_area("Medical Information", placeholder="Any medical conditions")
            
            submitted = st.form_submit_button("🤖 AI Auto-Evaluate Application")
        
            # Add stream assignment method
            assignment_method = st.selectbox(
                "Stream Assignment Method",
                ["Performance-based", "Random", "Balanced"],
                help="How students are assigned to class streams",
                index=2  # Default to Balanced
            )
        
            if submitted:
                # Validate required fields
                validation_errors = []
                if not name:
                    validation_errors.append("Student Name is required")
                if not parent_contact:
                    validation_errors.append("Parent Contact is required")
                if not date_of_birth:
                    validation_errors.append("Date of Birth is required")
                if age < 3 or age > 25:
                    validation_errors.append("Age must be between 3 and 25")
                
                # Additional DOB validation
                if date_of_birth and date_of_birth > date.today():
                    validation_errors.append("Date of Birth cannot be in the future")
                
                if validation_errors:
                    for error in validation_errors:
                        st.error(error)
                else:
                    # Create application data with proper date format
                    from datetime import datetime
                    
                    application_data = {
                        'name': name,
                        'age': age,
                        'date_of_birth': date_of_birth.strftime('%Y-%m-%d'),
                        'previous_gpa': previous_gpa,
                        'activities': activities,
                        'parent_contact': parent_contact,
                        'medical_info': medical_info,
                        'admission_date': datetime.now().strftime('%Y-%m-%d'),
                        'assignment_method': assignment_method.lower().replace('-', '')
                    }
                    
                    with st.spinner("AI is evaluating the application..."):
                        try:
                            result = manager.auto_admit_student(application_data)
                            
                            if not isinstance(result, dict):
                                raise ValueError("Invalid AI response format")
                            
                            status = result.get('status', 'UNKNOWN').upper()
                            reasoning = result.get('ai_reasoning', 'No reasoning provided')
                            
                            if status == 'ACCEPTED':
                                st.success("🎉 Application ACCEPTED!")
                                st.write(f"**Student ID:** {result.get('student_id', 'N/A')}")
                                st.write(f"**Assigned Grade:** {result.get('grade', 'N/A')}")
                                st.write(f"**Date of Birth:** {date_of_birth.strftime('%B %d, %Y')}")
                                st.write(f"**Age:** {age} years")
                                with st.expander("AI Reasoning"):
                                    st.markdown(reasoning)
                            elif status == 'REJECTED':
                                st.error("❌ Application REJECTED")
                                with st.expander("AI Reasoning"):
                                    st.markdown(reasoning)
                            else:
                                st.warning(f"⚠️ Unclear evaluation result: {status}")
                                with st.expander("AI Reasoning"):
                                    st.markdown(reasoning)
                                    
                        except Exception as e:
                            st.error(f"Evaluation failed: {str(e)}")
    
    with col2:
        st.subheader("📊 Admission Statistics")
        
        # Student status statistics
        try:
            students = manager.db.execute_query("SELECT status FROM students")
            status_counts = {
                'active': 0,
                'pending': 0,
                'rejected': 0,
                'graduated': 0
            }
            
            for row in students:
                status = row[0]
                if status in status_counts:
                    status_counts[status] += 1
            
            admission_data = {
                'Status': ['Active (Accepted)', 'Pending', 'Rejected', 'Graduated'],
                'Count': [
                    status_counts.get('active', 0),
                    status_counts.get('pending', 0),
                    status_counts.get('rejected', 0),
                    status_counts.get('graduated', 0)
                ]
            }
        except Exception as e:
            st.warning(f"Could not fetch admission statistics: {str(e)}")
            admission_data = {
                'Status': ['Active (Accepted)', 'Pending', 'Rejected', 'Graduated'],
                'Count': [0, 0, 0, 0]
            }
        
        if sum(admission_data['Count']) > 0:
            fig = px.pie(
                values=admission_data['Count'], 
                names=admission_data['Status'],
                title="Student Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("📈 No admission data available yet. Submit applications to see statistics.")
        
        # Quick metrics
        st.subheader("📈 Quick Stats")
        col_a, col_b = st.columns(2)
        
        total_students = sum(admission_data['Count'])
        
        with col_a:
            st.metric("Total Students", total_students)
        
        with col_b:
            if total_students > 0:
                accepted = admission_data['Count'][0]  # Active count
                acceptance_rate = round((accepted / total_students) * 100, 1)
                st.metric("Acceptance Rate", f"{acceptance_rate}%")
            else:
                st.metric("Acceptance Rate", "N/A")
        
        # Age distribution statistics
        if total_students > 0:
            st.subheader("👶 Age Distribution")
            try:
                age_data = manager.db.execute_query("SELECT age FROM students WHERE status = 'active'")
                if age_data:
                    ages = [row[0] for row in age_data if row[0] is not None]
                    if ages:
                        age_counts = {}
                        for age in ages:
                            age_counts[age] = age_counts.get(age, 0) + 1
                        
                        age_df = pd.DataFrame(list(age_counts.items()), columns=['Age', 'Count'])
                        age_df = age_df.sort_values('Age')
                        
                        fig_age = px.bar(age_df, x='Age', y='Count', 
                                       title="Student Age Distribution",
                                       color='Count',
                                       color_continuous_scale='Blues')
                        st.plotly_chart(fig_age, width="stretch")
                    else:
                        st.info("No age data available for visualization.")
                else:
                    st.info("No student age data found.")
            except Exception as e:
                st.warning(f"Could not generate age distribution: {str(e)}")
        
        # AI model status
        st.subheader("🤖 AI Model")
        if hasattr(manager, 'ai') and manager.ai:
            provider_str = manager.ai.provider
            if isinstance(provider_str, AIProvider):
                provider_str = provider_str.value.title()
                
            st.write(f"**Model:** {st.session_state.get('ai_model', 'None selected')}")
            st.write(f"**Provider:** {provider_str}")
            st.write(f"**Connection Status:** {manager.ai._test_connection()}")
            
            if st.button("🧪 Test Model"):
                with st.spinner("Testing AI model..."):
                    try:
                        manager.ai.sync_with_session_state()
                        response = manager.ai.generate_response("Say 'Hello, this is a test response.'")
                        if response and response.strip():
                            st.success(f"✅ Model working! Response: {response}")
                        else:
                            st.warning("⚠️ Empty or invalid response from AI model.")
                    except Exception as e:
                        st.error(f"❌ Test failed: {str(e)}")
        else:
            st.error("AI system not configured. Check AI settings in the sidebar.")

def students_page():
    st.header("🎓 Student Management")
    
    manager = st.session_state.school_manager
    students = manager.get_all_students()
    
    if not students:
        st.info("No students found. Add students through the Admissions tab.")
        return
    
    # Sync all stored ages with calculated ages
    updated = False
    for student in students:
        if student.get('date_of_birth'):
            try:
                dob_date = datetime.strptime(student['date_of_birth'], '%Y-%m-%d').date()
                calculated_age = calculate_age(dob_date)
                if student.get('age') != calculated_age:
                    student['age'] = calculated_age
                    # Update in database
                    manager.update_student({
                        'student_id': student['student_id'],
                        'age': calculated_age
                    })
                    updated = True
            except:
                pass
    
    # Reload students if any ages were updated
    if updated:
        students = manager.get_all_students()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📊 View Students", "✏️ Edit Student Data"])
    
    # Convert to DataFrame for display
    df = pd.DataFrame(students)
    
    # Add stream display column (combine stream_id and stream_type)
    df['stream'] = df.apply(lambda row: 
        f"{row.get('stream_id', 'N/A')} ({row.get('stream_type', 'N/A')})" 
        if row.get('stream_id') and row.get('stream_type') 
        else "Not assigned", axis=1)
    
    with tab1:
        # Student filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            grade_filter = st.selectbox("Filter by Grade", ["All"] + list(df['grade'].unique()))
        
        with col2:
            # Stream filter - dynamically filter based on selected grade
            if grade_filter != "All":
                # Get streams only for the selected grade
                grade_filtered_df = df[df['grade'] == grade_filter]
                
                # Get actual stream_ids that exist (not null/empty) for this grade
                streams_with_ids = grade_filtered_df[grade_filtered_df['stream_id'].notna() & (grade_filtered_df['stream_id'] != '')]
                
                if len(streams_with_ids) > 0:
                    # Build stream options from actual stream_id and stream_type data
                    actual_streams = []
                    for _, row in streams_with_ids.iterrows():
                        stream_id = row.get('stream_id', '')
                        stream_type = row.get('stream_type', '')
                        
                        if stream_id and stream_type and stream_type != '0':
                            stream_display = f"{stream_id} ({stream_type})"
                        elif stream_id:
                            stream_display = stream_id
                        else:
                            continue
                            
                        if stream_display not in actual_streams:
                            actual_streams.append(stream_display)
                    
                    stream_options = ["All"] + actual_streams + ["Not assigned"]
                else:
                    # No streams found for this grade
                    stream_options = ["All", "Not assigned"]
                
            else:
                # Show all streams if no grade is selected
                streams_with_ids = df[df['stream_id'].notna() & (df['stream_id'] != '')]
                
                actual_streams = []
                for _, row in streams_with_ids.iterrows():
                    stream_id = row.get('stream_id', '')
                    stream_type = row.get('stream_type', '')
                    
                    if stream_id and stream_type and stream_type != '0':
                        stream_display = f"{stream_id} ({stream_type})"
                    elif stream_id:
                        stream_display = stream_id
                    else:
                        continue
                        
                    if stream_display not in actual_streams:
                        actual_streams.append(stream_display)
                
                stream_options = ["All"] + actual_streams + ["Not assigned"]
            
            stream_filter = st.selectbox("Filter by Stream", stream_options)
        
        with col3:
            min_gpa = st.slider("Minimum GPA", 0.0, 4.0, 0.0)
        
        with col4:
            search_name = st.text_input("Search by Name")
        
        # Apply filters
        filtered_df = df.copy()
        if grade_filter != "All":
            filtered_df = filtered_df[filtered_df['grade'] == grade_filter]
        
        if stream_filter != "All":
            if stream_filter == "Not assigned":
                # Show students with no stream assigned
                filtered_df = filtered_df[filtered_df['stream_id'].isna() | (filtered_df['stream_id'] == '')]
            else:
                # Extract the actual stream_id from the display format
                # Handle both "Grade1R" and "Grade1R (Type)" formats
                if "(" in stream_filter and stream_filter.endswith(")"):
                    # Format: "Grade1R (Type)" - extract just "Grade1R"
                    actual_stream_id = stream_filter.split(" (")[0]
                else:
                    # Format: "Grade1R" - use as is
                    actual_stream_id = stream_filter
                
                # Filter by the actual stream_id column, not the combined stream column
                filtered_df = filtered_df[filtered_df['stream_id'] == actual_stream_id]
        
        if min_gpa > 0:
            filtered_df = filtered_df[filtered_df['gpa'] >= min_gpa]
        if search_name:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False)]
        
        # Display students with stream column included
        st.dataframe(
            filtered_df[['student_id', 'name', 'age', 'grade', 'stream', 'gpa', 'attendance_rate', 'behavior_score']],
            width="stretch"
        )
        
        # Student details
        if not filtered_df.empty:
            selected_student = st.selectbox("Select Student for Details", filtered_df['name'].tolist())
            student_data = filtered_df[filtered_df['name'] == selected_student].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📋 {student_data['name']} Details")
                st.write(f"**Student ID:** {student_data['student_id']}")
                st.write(f"**Grade:** {student_data['grade']}")
                st.write(f"**Age:** {student_data['age']}")
                st.write(f"**GPA:** {student_data['gpa']}")
                st.write(f"**Attendance:** {student_data['attendance_rate']}%")
                st.write(f"**Behavior Score:** {student_data['behavior_score']}")
                st.write(f"**Stream:** {student_data['stream']}")
                st.write(f"**Status:** {student_data.get('status', 'Active')}")
            
            with col2:
                st.subheader("📊 Performance Chart")
                
                performance_data = {
                    'Metric': ['GPA', 'Attendance', 'Behavior'],
                    'Score': [student_data['gpa']*25, student_data['attendance_rate'], student_data['behavior_score']],
                    'Max': [100, 100, 100]
                }
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Current', x=performance_data['Metric'], y=performance_data['Score']))
                fig.add_trace(go.Bar(name='Maximum', x=performance_data['Metric'], y=performance_data['Max'], opacity=0.3))
                fig.update_layout(title="Student Performance Metrics", barmode='overlay')
                
                st.plotly_chart(fig, width="stretch")
    
    with tab2:
        st.subheader("✏️ Edit Student Information")
        
        # Student selection for editing
        col1, col2 = st.columns([2, 1])
        
        with col1:
            edit_student = st.selectbox(
                "Select Student to Edit", 
                df['name'].tolist(),
                key="edit_student_select"
            )
        
        with col2:
            st.metric("Total Students", len(df))
        
        if edit_student:
            student_data = df[df['name'] == edit_student].iloc[0]
            
            # Create form for editing
            with st.form("edit_student_form"):
                st.markdown(f"### Editing: {student_data['name']} (ID: {student_data['student_id']})")
                
                # Organize fields in columns for better space utilization
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📚 Academic Information**")
                    new_gpa = st.number_input(
                        "GPA", 
                        min_value=0.0, 
                        max_value=4.0, 
                        value=float(student_data['gpa']),
                        step=0.1,
                        format="%.2f"
                    )
                    
                    new_grade = st.selectbox(
                        "Grade Level",
                        options=sorted(df['grade'].unique()),
                        index=list(sorted(df['grade'].unique())).index(student_data['grade'])
                    )
                    
                    # Convert student_data to dict for consistent access
                    student_dict = student_data.to_dict()
                    
                    # Get ALL available streams from the dataframe, not just current grade
                    all_streams = df['stream'].unique()
                    available_streams = [s for s in all_streams if s != "Not assigned"]
                    
                    # If current student has a stream that's not in the list, add it
                    current_stream_id = student_dict.get('stream_id')
                    current_stream_type = student_dict.get('stream_type')
                    
                    if current_stream_id and current_stream_type:
                        current_stream_display = f"{current_stream_id} ({current_stream_type})"
                    elif current_stream_id:
                        current_stream_display = current_stream_id
                    else:
                        current_stream_display = "Not assigned"
                    
                    # Add current stream to options if it's not already there
                    if current_stream_display != "Not assigned" and current_stream_display not in available_streams:
                        available_streams.append(current_stream_display)
                    
                    stream_options_edit = ["Not assigned"] + list(available_streams)              
      
                    current_stream_index = 0
                    if current_stream_display in stream_options_edit:
                        current_stream_index = stream_options_edit.index(current_stream_display)
                    else:
                        # If exact match not found, try to find just the stream_id
                        if current_stream_id:
                            for i, option in enumerate(stream_options_edit):
                                if option.startswith(current_stream_id):
                                    current_stream_index = i
                                    break
                    
                    new_stream = st.selectbox(
                        "Stream Assignment",
                        options=stream_options_edit,
                        index=current_stream_index
                    )
                    
                with col2:
                    st.markdown("**📊 Performance Metrics**")
                    new_attendance = st.number_input(
                        "Attendance Rate (%)", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=float(student_data['attendance_rate']),
                        step=0.1,
                        format="%.1f"
                    )
                    
                    new_behavior = st.number_input(
                        "Behavior Score", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=float(student_data['behavior_score']),
                        step=0.1,
                        format="%.1f"
                    )
                    
                    new_status = st.selectbox(
                        "Student Status",
                        options=["active", "inactive", "suspended", "graduated"],
                        index=["active", "inactive", "suspended", "graduated"].index(
                            student_data.get('status', 'active')
                        )
                    )
                
                with col3:
                    st.markdown("**👨‍👩‍👧‍👦 Personal Information**")
                    
                    # Handle current date of birth
                    current_dob = student_data.get('date_of_birth')
                    current_age = student_data.get('age', 10)
                    
                    if current_dob:
                        try:
                            if isinstance(current_dob, str):
                                current_dob_date = datetime.strptime(current_dob, '%Y-%m-%d').date()
                            else:
                                current_dob_date = current_dob
                        except (ValueError, TypeError):
                            current_dob_date = datetime.now().date() - timedelta(days=365 * int(current_age))
                    else:
                        current_dob_date = datetime.now().date() - timedelta(days=365 * int(current_age))
                    
                    # Date of birth input
                    new_dob = st.date_input(
                        "Date of Birth",
                        value=current_dob_date,
                        min_value=datetime.now().date() - timedelta(days=365*25),
                        max_value=datetime.now().date() - timedelta(days=365*3),
                        help="Age will be automatically calculated from this date"
                    )
                    
                    # Calculate and display age
                    calculated_age = calculate_age(new_dob)
                    current_stored_age = int(student_data['age'])
                    
                    if calculated_age != current_stored_age:
                        st.write(f"**Updated age:** {calculated_age} years")
                    else:
                        st.write(f"**Age:** {calculated_age} years")
                    
                    new_parent_contact = st.text_input(
                        "Parent Contact",
                        value=student_data.get('parent_contact', ''),
                        placeholder="Phone number or email"
                    )
                    
                    new_medical_info = st.text_area(
                        "Medical Information",
                        value=student_data.get('medical_info', ''),
                        height=80,
                        placeholder="Any relevant medical information..."
                    )
                
                # Form submission buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    submit_button = st.form_submit_button("Save Changes", type="primary")
                
                with col2:
                    reset_button = st.form_submit_button("Reset Form")
                
                # Handle form submission
                if submit_button:
                    try:
                        # Calculate final age
                        final_age = calculate_age(new_dob)
                        
                        # Prepare update data
                        update_data = {
                            'student_id': student_data['student_id'],
                            'gpa': new_gpa,
                            'grade': new_grade,
                            'attendance_rate': new_attendance,
                            'behavior_score': new_behavior,
                            'status': new_status,
                            'date_of_birth': new_dob.strftime('%Y-%m-%d'),
                            'age': final_age,
                            'parent_contact': new_parent_contact,
                            'medical_info': new_medical_info
                        }
                        
                        # Handle stream assignment
                        if new_stream != "Not assigned":
                            stream_id = new_stream.split(' (')[0] if ' (' in new_stream else new_stream
                            update_data['stream_id'] = stream_id
                        else:
                            update_data['stream_id'] = None
                        
                        # Update student in database
                        manager.update_student(update_data)
                        
                        st.success(f"Successfully updated {student_data['name']}'s information")
                        
                        # Refresh the page data
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error updating student: {str(e)}")
                
                if reset_button:
                    st.rerun()
            
            # Display current vs new values comparison
            st.markdown("---")
            st.subheader("📋 Current Student Summary")
            
            # Convert pandas Series to dict for consistent access
            student_dict = student_data.to_dict()
            
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Current GPA", f"{student_dict['gpa']:.2f}")
                st.metric("Current Attendance", f"{student_dict['attendance_rate']:.1f}%")
            
            with summary_col2:
                st.metric("Current Behavior Score", f"{student_dict['behavior_score']:.1f}")
                st.metric("Current Grade", student_dict['grade'])
            
            with summary_col3:
                # Fix stream assignment display
                current_stream_id = student_dict.get('stream_id')
                current_stream_type = student_dict.get('stream_type')
                
                if current_stream_id and current_stream_type:
                    current_stream_display = f"{current_stream_id} ({current_stream_type})"
                elif current_stream_id:
                    current_stream_display = current_stream_id
                else:
                    current_stream_display = "Not assigned"
                
                st.metric("Current Stream", current_stream_display)
                st.metric("Current Status", student_dict.get('status', 'active').title())
            
            with summary_col4:
                # Display synchronized age
                display_age = student_dict.get('age', 'Unknown')
                st.metric("Age", f"{display_age} years" if display_age != 'Unknown' else "Unknown")
                
                # Admission date display
                admission_date = student_dict.get('admission_date', 'Not recorded')
                st.metric("Admission Date", admission_date if admission_date else "Not recorded")

def teachers_page():
    """Main teacher management page with basic CRUD operations"""
    st.header("👨‍🏫 Teacher Management")
    
    # Initialize database manager
    db = DatabaseManager()
    
    try:
        # Fetch actual teacher data from database with stream assignment info
        query = """
        SELECT 
            t.teacher_id,
            t.name,
            t.subject,
            t.department,
            t.performance_score,
            t.status,
            t.is_class_teacher,
            CASE 
                WHEN t.is_class_teacher = 1 THEN 
                    COALESCE(s.grade_level || ' - ' || s.stream_type, 'Not Assigned')
                ELSE 
                    'Not Class Teacher'
            END as assigned_class
        FROM teachers t
        LEFT JOIN streams s ON t.teacher_id = s.class_teacher_id
        WHERE t.status = 'active'
        ORDER BY t.name
        """
        
        teachers_raw = db.execute_query(query)
        
        if not teachers_raw:
            st.warning("No teacher data found in database. Please add teachers first.")
            
            # Add teacher form
            st.subheader("➕ Add New Teacher")
            with st.form("add_teacher"):
                col1, col2 = st.columns(2)
                with col1:
                    teacher_id = st.text_input("Teacher ID", placeholder="T001")
                    name = st.text_input("Name", placeholder="Dr. John Smith")
                    subject = st.selectbox("Subject", [
                        'Mathematics', 'Science', 'English', 'History', 'Art', 'Music', 
                        'Physical Education', 'Computer Science', 'Biology', 'Chemistry', 
                        'Physics', 'Literature', 'Geography', 'Economics', 'Psychology', 
                        'Foreign Languages', 'Drama', 'Health Education'
                    ])
                
                with col2:
                    department = st.selectbox("Department", [
                        'STEM', 'Languages', 'Social Studies', 'Arts', 'Health & PE', 'General'
                    ])
                    performance_score = st.slider("Performance Score", 0.0, 100.0, 85.0)
                    salary = st.number_input("Salary", min_value=0.0, value=50000.0)
                
                hire_date = st.date_input("Hire Date")
                is_class_teacher = st.checkbox("Is Class Teacher?", value=False)
                
                if st.form_submit_button("Add Teacher"):
                    if teacher_id and name:
                        try:
                            insert_query = """
                            INSERT INTO teachers 
                            (teacher_id, name, subject, department, hire_date, performance_score, salary, status, is_class_teacher)
                            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                            """
                            db.execute_update(insert_query, (
                                teacher_id, name, subject, department, 
                                hire_date.strftime('%Y-%m-%d'), performance_score, salary, is_class_teacher
                            ))
                            st.success(f"Teacher {name} added successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding teacher: {str(e)}")
                    else:
                        st.error("Please fill in Teacher ID and Name")
            return
        
        # Convert to DataFrame
        teachers_data = {
            'Teacher ID': [row[0] for row in teachers_raw],
            'Name': [row[1] for row in teachers_raw],
            'Subject': [row[2] for row in teachers_raw],
            'Department': [row[3] for row in teachers_raw],
            'Performance': [row[4] for row in teachers_raw],
            'Status': [row[5] for row in teachers_raw],
            'Class Teacher': ["✅" if row[6] else "❌" for row in teachers_raw],
            'Assigned Stream': [row[7] for row in teachers_raw]
        }
        
        # Get student counts for each teacher
        student_counts = []
        for teacher_id in teachers_data['Teacher ID']:
            # Check if teacher is a class teacher first
            is_class_teacher_query = """
            SELECT is_class_teacher FROM teachers WHERE teacher_id = ?
            """
            is_class_teacher_result = db.execute_query(is_class_teacher_query, (teacher_id,))
            is_class_teacher = is_class_teacher_result[0][0] if is_class_teacher_result else False
            
            if is_class_teacher:
                # For class teachers, count students in their assigned stream
                stream_student_count_query = """
                SELECT COUNT(DISTINCT st.student_id) as student_count
                FROM streams s
                JOIN students st ON s.stream_id = st.stream_id
                WHERE s.class_teacher_id = ? AND st.status = 'active'
                """
                result = db.execute_query(stream_student_count_query, (teacher_id,))
                stream_count = result[0][0] if result and result[0][0] else 0
                
                # Also count students from courses they teach
                course_student_count_query = """
                SELECT COUNT(DISTINCT g.student_id) as student_count
                FROM courses c
                LEFT JOIN grades g ON c.course_id = g.course_id
                WHERE c.teacher_id = ?
                """
                result = db.execute_query(course_student_count_query, (teacher_id,))
                course_count = result[0][0] if result and result[0][0] else 0
                
                # For class teachers, prioritize stream count, but show total if they teach courses too
                total_count = max(stream_count, course_count)
                if stream_count > 0 and course_count > 0 and course_count > stream_count:
                    total_count = course_count
                else:
                    total_count = stream_count
                    
            else:
                # For regular teachers, count students from courses they teach
                course_student_count_query = """
                SELECT COUNT(DISTINCT g.student_id) as student_count
                FROM courses c
                LEFT JOIN grades g ON c.course_id = g.course_id
                WHERE c.teacher_id = ?
                """
                result = db.execute_query(course_student_count_query, (teacher_id,))
                total_count = result[0][0] if result and result[0][0] else 0
            
            student_counts.append(total_count)
        
        teachers_data['Students'] = student_counts
        df = pd.DataFrame(teachers_data)
        
    except Exception as e:
        st.error(f"Error fetching teacher data: {str(e)}")
        return
    
    # Create main tabs
    tab1, tab2, tab3 = st.tabs(["👨‍🏫 Basic Management", "🚀 Advanced Features", "⚙️ Teacher Updates"])
    
    # ==================== BASIC MANAGEMENT TAB ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Teacher Overview")
            
            # Add search and filter options
            search_term = st.text_input("🔍 Search Teachers", placeholder="Search by name, subject, department, or stream...")
            
            # Filter options
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                dept_filter = st.selectbox("Filter by Department", ["All"] + sorted(df['Department'].unique().tolist()))
            with col_filter2:
                class_teacher_filter = st.selectbox("Filter by Class Teacher Status", ["All", "Class Teachers Only", "Non-Class Teachers"])
            
            # Apply filters
            filtered_df = df.copy()
            
            if search_term:
                mask = (
                    df['Name'].str.contains(search_term, case=False, na=False) |
                    df['Subject'].str.contains(search_term, case=False, na=False) |
                    df['Department'].str.contains(search_term, case=False, na=False) |
                    df['Assigned Stream'].str.contains(search_term, case=False, na=False)
                )
                filtered_df = filtered_df[mask]
            
            if dept_filter != "All":
                filtered_df = filtered_df[filtered_df['Department'] == dept_filter]
            
            if class_teacher_filter == "Class Teachers Only":
                filtered_df = filtered_df[filtered_df['Class Teacher'] == "✅"]
            elif class_teacher_filter == "Non-Class Teachers":
                filtered_df = filtered_df[filtered_df['Class Teacher'] == "❌"]
            
            # Reorder columns to show Assigned Stream after Class Teacher
            column_order = ['Teacher ID', 'Name', 'Subject', 'Department', 'Performance', 'Status', 'Class Teacher', 'Assigned Stream', 'Students']
            filtered_df = filtered_df[column_order]
            
            st.dataframe(filtered_df, width="stretch")
            
            # Export functionality
            if st.button("📥 Export Teacher Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="teachers_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("🎯 Performance Distribution")
            if len(df) > 0:
                fig = px.histogram(df, x='Performance', nbins=min(15, len(df)), 
                                  title="Teacher Performance Distribution",
                                  color_discrete_sequence=['#636EFA'])
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data to display")
            
            # Department Overview
            st.subheader("📊 Department Overview")
            if len(df) > 0:
                # Department distribution
                dept_counts = df['Department'].value_counts()
                fig2 = px.pie(values=dept_counts.values, names=dept_counts.index, 
                             title="Teachers by Department")
                st.plotly_chart(fig2, width="stretch")
                
                # Key metrics
                st.metric("Total Teachers", len(df))
                st.metric("Average Performance", f"{df['Performance'].mean():.1f}%" if len(df) > 0 else "N/A")
                st.metric("Total Students", f"{df['Students'].sum():,}")
                st.metric("Avg Students/Teacher", f"{df['Students'].mean():.0f}" if len(df) > 0 else "N/A")
            else:
                st.info("No teacher data available for metrics.")
    
    # ==================== ADVANCED FEATURES TAB ====================
    with tab2:
        st.subheader("🚀 Advanced Teacher Management Features")
        
        # Create sub-tabs for advanced features
        subtab1, subtab2, subtab3 = st.tabs(["🏫 Stream Assignment", "🤖 AI Teacher Assignment", "📊 Generate Reports"])
        
        # ==================== STREAM ASSIGNMENT MANAGEMENT ====================
        with subtab1:
            st.subheader("🏫 Stream Assignment Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Assign Teacher to Stream**")
                if len(df) > 0:
                    # Get available teachers and classes with department info
                    class_teachers_df = df[df['Class Teacher'] == "✅"].copy()
                    
                    if not class_teachers_df.empty:
                        # Create teacher options with department info
                        teacher_options = {}
                        for _, teacher in class_teachers_df.iterrows():
                            display_name = f"{teacher['Name']} ({teacher['Department']} - {teacher['Subject']})"
                            teacher_options[display_name] = teacher['Name']
                        
                        selected_teacher_display = st.selectbox(
                            "Select Class Teacher", 
                            list(teacher_options.keys()), 
                            key="assign_teacher",
                            help="Format: Teacher Name (Department - Subject)"
                        )
                        selected_teacher_name = teacher_options[selected_teacher_display]
                        
                        # Show additional teacher info
                        teacher_info = class_teachers_df[class_teachers_df['Name'] == selected_teacher_name].iloc[0]
                        
                        # Display teacher details in an info box
                        st.info(f"""
                        **Selected Teacher Details:**
                        - **Name:** {teacher_info['Name']}
                        - **Department:** {teacher_info['Department']}
                        - **Subject:** {teacher_info['Subject']}
                        - **Performance:** {teacher_info['Performance']:.1f}%
                        - **Current Students:** {teacher_info['Students']}
                        - **Current Assignment:** {teacher_info['Assigned Stream']}
                        """)
                        
                        # Fetch available streams (unassigned ones)
                        try:
                            streams_query = """
                            SELECT stream_id, grade_level, stream_type, max_capacity,
                                   (SELECT COUNT(*) FROM students WHERE stream_id = s.stream_id AND status = 'active') as current_students
                            FROM streams s
                            WHERE class_teacher_id IS NULL OR class_teacher_id = ''
                            ORDER BY grade_level, stream_type
                            """
                            available_streams = db.execute_query(streams_query)
                            
                            if available_streams:
                                stream_options = {}
                                for row in available_streams:
                                    stream_id, grade_level, stream_type, max_capacity, current_students = row
                                    current_students = current_students or 0
                                    display_text = f"{grade_level} - {stream_type} (Capacity: {max_capacity}, Current: {current_students})"
                                    stream_options[display_text] = stream_id
                                
                                selected_stream_display = st.selectbox(
                                    "Select Stream", 
                                    list(stream_options.keys()), 
                                    key="assign_stream",
                                    help="Format: Grade - Stream Type (Capacity info)"
                                )
                                selected_stream_id = stream_options[selected_stream_display]
                                
                                # Show stream details
                                stream_parts = selected_stream_display.split(" (")
                                stream_name = stream_parts[0]
                                capacity_info = stream_parts[1].rstrip(")")
                                
                                st.info(f"""
                                **Selected Stream Details:**
                                - **Stream:** {stream_name}
                                - **Stream ID:** {selected_stream_id}
                                - **{capacity_info}**
                                """)
                                
                                # Check compatibility
                                if teacher_info['Department'] in ['STEM', 'General'] or 'Science' in teacher_info['Subject'] or 'Math' in teacher_info['Subject']:
                                    st.success("✅ Teacher's background is suitable for this stream assignment")
                                else:
                                    st.warning("⚠️ Consider if teacher's department aligns with stream requirements")
                                
                                if st.button("Assign Stream", key="assign_btn"):
                                    teacher_id = df[df['Name'] == selected_teacher_name]['Teacher ID'].iloc[0]
                                    
                                    try:
                                        # Update stream with teacher assignment
                                        update_query = "UPDATE streams SET class_teacher_id = ? WHERE stream_id = ?"
                                        db.execute_update(update_query, (teacher_id, selected_stream_id))
                                        st.success(f"✅ Successfully assigned {selected_teacher_name} ({teacher_info['Department']}) to {stream_name}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error assigning stream: {str(e)}")
                            else:
                                st.info("No unassigned streams available")
                                
                        except Exception as e:
                            st.error(f"Error fetching streams: {str(e)}")
                    else:
                        st.info("No class teachers available")
            
            with col2:
                st.write("**Stream Assignment Summary**")
                if len(df) > 0:
                    class_teachers = df[df['Class Teacher'] == "✅"]
                    assigned_count = len(class_teachers[class_teachers['Assigned Stream'] != 'Not Assigned'])
                    unassigned_count = len(class_teachers[class_teachers['Assigned Stream'] == 'Not Assigned'])
                    
                    st.metric("Class Teachers", len(class_teachers))
                    st.metric("Assigned", assigned_count)
                    st.metric("Unassigned", unassigned_count)
                    
                    if unassigned_count > 0:
                        st.warning(f"{unassigned_count} class teachers need stream assignment")
                        st.write("**Unassigned Class Teachers:**")
                        unassigned_teachers = class_teachers[class_teachers['Assigned Stream'] == 'Not Assigned']
                        for _, teacher in unassigned_teachers.iterrows():
                            st.write(f"• **{teacher['Name']}** - {teacher['Department']} ({teacher['Subject']})")
                    
                    # Department breakdown of class teachers
                    st.write("**Class Teachers by Department:**")
                    dept_breakdown = class_teachers['Department'].value_counts()
                    for dept, count in dept_breakdown.items():
                        assigned_in_dept = len(class_teachers[
                            (class_teachers['Department'] == dept) & 
                            (class_teachers['Assigned Stream'] != 'Not Assigned')
                        ])
                        st.write(f"• **{dept}:** {count} total ({assigned_in_dept} assigned)")
            
            # Stream Assignment Overview
            st.subheader("📋 Stream Assignment Overview")
            
            # Show all stream assignments with teacher department info
            try:
                stream_assignment_query = """
                SELECT 
                    s.stream_id,
                    s.grade_level,
                    s.stream_type,
                    s.max_capacity,
                    (SELECT COUNT(*) FROM students WHERE stream_id = s.stream_id AND status = 'active') as current_students,
                    COALESCE(t.name, 'Unassigned') as teacher_name,
                    COALESCE(t.teacher_id, '') as teacher_id,
                    COALESCE(t.department, 'N/A') as teacher_department,
                    COALESCE(t.subject, 'N/A') as teacher_subject,
                    COALESCE(t.performance_score, 0) as teacher_performance
                FROM streams s
                LEFT JOIN teachers t ON s.class_teacher_id = t.teacher_id
                ORDER BY s.grade_level, s.stream_type
                """
                
                stream_data = db.execute_query(stream_assignment_query)
                
                if stream_data:
                    # Process data to create a more informative display
                    processed_data = []
                    for row in stream_data:
                        stream_id, grade_level, stream_type, max_capacity, current_students, teacher_name, teacher_id, teacher_dept, teacher_subject, teacher_perf = row
                        current_students = current_students or 0
                        
                        # Create teacher info string
                        if teacher_name != 'Unassigned':
                            teacher_info = f"{teacher_name} ({teacher_dept} - {teacher_subject})"
                            performance_display = f"{teacher_perf:.1f}%"
                        else:
                            teacher_info = "Unassigned"
                            performance_display = "N/A"
                        
                        # Calculate capacity utilization
                        utilization = (current_students / max_capacity * 100) if max_capacity > 0 else 0
                        
                        processed_data.append([
                            stream_id,
                            f"{grade_level} - {stream_type}",
                            f"{current_students}/{max_capacity} ({utilization:.1f}%)",
                            teacher_info,
                            teacher_id,
                            performance_display
                        ])
                    
                    stream_df = pd.DataFrame(processed_data, columns=[
                        'Stream ID', 'Stream', 'Enrollment (Utilization)', 'Assigned Teacher (Dept - Subject)', 'Teacher ID', 'Performance Score'
                    ])
                    
                    # Color code based on assignment status
                    def highlight_unassigned(row):
                        if 'Unassigned' in str(row['Assigned Teacher (Dept - Subject)']):
                            return ['background-color: #ffcccc'] * len(row)
                        else:
                            return ['background-color: #ccffcc'] * len(row)
                    
                    styled_df = stream_df.style.apply(highlight_unassigned, axis=1)
                    st.dataframe(styled_df, width="stretch")
                    
                    # Show summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_streams = len(stream_data)
                        st.metric("Total Streams", total_streams)
                    with col2:
                        assigned_streams = len([row for row in stream_data if row[5] != 'Unassigned'])
                        st.metric("Assigned Streams", assigned_streams)
                    with col3:
                        unassigned_streams = total_streams - assigned_streams
                        st.metric("Unassigned Streams", unassigned_streams)
                    with col4:
                        total_capacity = sum([row[3] for row in stream_data if row[3]])
                        total_enrolled = sum([row[4] or 0 for row in stream_data])
                        overall_utilization = (total_enrolled / total_capacity * 100) if total_capacity > 0 else 0
                        st.metric("Overall Utilization", f"{overall_utilization:.1f}%")
                    
                    # Export stream assignments
                    if st.button("📥 Export Stream Assignments", key="export_streams"):
                        csv = stream_df.to_csv(index=False)
                        st.download_button(
                            label="Download Stream Assignments CSV",
                            data=csv,
                            file_name=f"stream_assignments_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_streams"
                        )
                else:
                    st.info("No streams found in the database")
                    
            except Exception as e:
                st.error(f"Error fetching stream assignments: {str(e)}")
                
            # Additional insights section
            st.subheader("🔍 Assignment Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Department-Stream Compatibility**")
                if len(df) > 0:
                    # Show which departments are best suited for different stream types
                    compatibility_info = {
                        "STEM": "Excellent for Science, Math-focused streams",
                        "Languages": "Ideal for Language Arts, Literature streams", 
                        "Social Studies": "Perfect for History, Geography streams",
                        "Arts": "Great for Creative, Arts-focused streams",
                        "Health & PE": "Suitable for Physical Education programs",
                        "General": "Flexible for various stream types"
                    }
                    
                    for dept, description in compatibility_info.items():
                        dept_teachers = len(df[(df['Department'] == dept) & (df['Class Teacher'] == "✅")])
                        if dept_teachers > 0:
                            st.write(f"**{dept}** ({dept_teachers} class teachers): {description}")
            
            with col2:
                st.write("**Assignment Recommendations**")
                try:
                    # Get unassigned streams and suggest suitable teachers
                    unassigned_streams_query = """
                    SELECT stream_id, grade_level, stream_type 
                    FROM streams 
                    WHERE class_teacher_id IS NULL OR class_teacher_id = ''
                    LIMIT 3
                    """
                    unassigned_streams = db.execute_query(unassigned_streams_query)
                    
                    if unassigned_streams:
                        st.write("**Suggested Assignments:**")
                        for stream in unassigned_streams:
                            stream_id, grade_level, stream_type = stream
                            
                            # Suggest teachers based on stream type
                            if 'Science' in stream_type or 'Math' in stream_type:
                                suggested_dept = 'STEM'
                            elif 'Language' in stream_type or 'English' in stream_type:
                                suggested_dept = 'Languages'
                            elif 'Arts' in stream_type:
                                suggested_dept = 'Arts'
                            else:
                                suggested_dept = 'General'
                            
                            available_teachers = df[
                                (df['Department'] == suggested_dept) & 
                                (df['Class Teacher'] == "✅") & 
                                (df['Assigned Stream'] == 'Not Assigned')
                            ]
                            
                            if not available_teachers.empty:
                                best_teacher = available_teachers.loc[available_teachers['Performance'].idxmax()]
                                st.write(f"**{grade_level} - {stream_type}**")
                                st.write(f"→ Suggested: {best_teacher['Name']} ({best_teacher['Department']}, {best_teacher['Performance']:.1f}%)")
                            else:
                                st.write(f"**{grade_level} - {stream_type}**: No suitable teachers available")
                    else:
                        st.success("✅ All streams have been assigned!")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
        
        # ==================== AI TEACHER ASSIGNMENT ====================
        # ==================== AI TEACHER ASSIGNMENT ====================
                with subtab2:
                    st.subheader("🤖 AI Teacher Assignment")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**AI Recommendation Engine**")
                        
                        # Get available subjects from both teachers and courses tables
                        try:
                            # Get subjects from teachers table
                            teacher_subjects_query = "SELECT DISTINCT subject FROM teachers WHERE status = 'active' AND subject IS NOT NULL AND subject != ''"
                            teacher_subjects_raw = db.execute_query(teacher_subjects_query)
                            teacher_subjects = [row[0] for row in teacher_subjects_raw] if teacher_subjects_raw else []
                            
                            # Get course names from courses table (as potential subjects)
                            course_subjects_query = "SELECT DISTINCT name FROM courses WHERE name IS NOT NULL AND name != ''"
                            course_subjects_raw = db.execute_query(course_subjects_query)
                            course_subjects = [row[0] for row in course_subjects_raw] if course_subjects_raw else []
                            
                            # Combine and deduplicate subjects
                            all_subjects = list(set(teacher_subjects + course_subjects))
                            
                            # If no subjects found, provide default subject list
                            if not all_subjects:
                                all_subjects = [
                                    'Mathematics', 'Science', 'English', 'History', 'Art', 'Music', 
                                    'Physical Education', 'Computer Science', 'Biology', 'Chemistry', 
                                    'Physics', 'Literature', 'Geography', 'Economics', 'Psychology', 
                                    'Foreign Languages', 'Drama', 'Health Education'
                                ]
                            
                            st.info(f"📚 Found {len(all_subjects)} subjects: {', '.join(sorted(all_subjects)[:5])}{'...' if len(all_subjects) > 5 else ''}")
                            
                            subject = st.selectbox("Subject", sorted(all_subjects), key="ai_subject")
                            grade_level = st.selectbox("Grade Level", ["Kindergarten", "Lower Primary", "Upper Primary", "Junior High School"], key="ai_grade")
                            class_size = st.number_input("Expected Class Size", min_value=1, max_value=50, value=35, key="ai_class_size")
                            
                            if st.button("🎯 AI Recommend Teacher", key="ai_recommend"):
                                try:
                                    # Filter teachers by subject or find teachers who can teach courses with similar names
                                    subject_match_query = """
                                    SELECT DISTINCT t.teacher_id, t.name, t.subject, t.department, t.performance_score, 
                                           t.is_class_teacher,
                                           COALESCE(student_counts.student_count, 0) as current_students,
                                           CASE 
                                               WHEN t.is_class_teacher = 1 THEN 
                                                   COALESCE(s.grade_level || ' - ' || s.stream_type, 'Not Assigned')
                                               ELSE 
                                                   'Not Class Teacher'
                                           END as assigned_class
                                    FROM teachers t
                                    LEFT JOIN streams s ON t.teacher_id = s.class_teacher_id
                                    LEFT JOIN (
                                        SELECT c.teacher_id, COUNT(DISTINCT g.student_id) as student_count
                                        FROM courses c
                                        LEFT JOIN grades g ON c.course_id = g.course_id
                                        WHERE c.teacher_id IS NOT NULL AND c.teacher_id != ''
                                        GROUP BY c.teacher_id
                                    ) student_counts ON t.teacher_id = student_counts.teacher_id
                                    WHERE t.status = 'active' 
                                    AND (LOWER(t.subject) = LOWER(?) OR EXISTS (
                                        SELECT 1 FROM courses c WHERE c.teacher_id = t.teacher_id AND LOWER(c.name) = LOWER(?)
                                    ))
                                    ORDER BY t.performance_score DESC, t.name ASC
                                    """
                                    
                                    available_teachers_raw = db.execute_query(subject_match_query, (subject, subject))
                                    
                                    if available_teachers_raw:
                                        # Convert to DataFrame for easier manipulation
                                        teachers_df = pd.DataFrame(available_teachers_raw, columns=[
                                            'Teacher ID', 'Name', 'Subject', 'Department', 'Performance', 
                                            'Is Class Teacher', 'Students', 'Assigned Stream'
                                        ])
                                        
                                        # Calculate recommendation score
                                        teachers_df['load_factor'] = 1 - (teachers_df['Students'] / 200)  # Assume max 200 students
                                        teachers_df['recommendation_score'] = (
                                            teachers_df['Performance'] * 0.7 + 
                                            teachers_df['load_factor'] * 100 * 0.3
                                        )
                                        
                                        # Find best teacher
                                        best_teacher = teachers_df.loc[teachers_df['recommendation_score'].idxmax()]
                                        
                                        # Check if teacher can handle additional students
                                        current_load = int(best_teacher['Students'])
                                        if current_load + class_size <= 200:  # Assume max 200 students per teacher
                                            st.success(f"**Recommended Teacher: {best_teacher['Name']}**")
                                            st.write(f"- Teacher ID: {best_teacher['Teacher ID']}")
                                            st.write(f"- Subject: {best_teacher['Subject']}")
                                            st.write(f"- Performance Score: {best_teacher['Performance']:.1f}%")
                                            st.write(f"- Current Students: {current_load}")
                                            st.write(f"- Department: {best_teacher['Department']}")
                                            st.write(f"- Class Teacher: {'Yes' if best_teacher['Is Class Teacher'] else 'No'}")
                                            st.write(f"- Assigned Stream: {best_teacher['Assigned Stream']}")
                                            st.write(f"- Recommendation Score: {best_teacher['recommendation_score']:.1f}")
                                            
                                            # Show reasoning
                                            st.info("**Recommendation Reasoning:**")
                                            st.write(f"- High performance score ({best_teacher['Performance']:.1f}%)")
                                            st.write(f"- Manageable current workload ({current_load} students)")
                                            st.write(f"- Can accommodate {class_size} additional students")
                                            
                                        else:
                                            st.warning(f"{best_teacher['Name']} is overloaded ({current_load} students). Consider redistributing classes.")
                                            
                                            # Show alternative recommendations
                                            alternative_teachers = teachers_df[
                                                teachers_df['Students'] + class_size <= 200
                                            ].sort_values('recommendation_score', ascending=False)
                                            
                                            if not alternative_teachers.empty:
                                                st.info("**Alternative recommendations:**")
                                                for idx, teacher in alternative_teachers.head(3).iterrows():
                                                    st.write(f"• **{teacher['Name']}** (Score: {teacher['recommendation_score']:.1f}, "
                                                           f"Students: {int(teacher['Students'])}, Stream: {teacher['Assigned Stream']})")
                                            else:
                                                st.error("No teachers available with sufficient capacity.")
                                    else:
                                        st.warning(f"No teachers found for **{subject}**. Consider:")
                                        st.write("- 🆕 Hiring a new teacher")
                                        st.write("- 📚 Cross-training existing teachers")
                                        st.write("- 🔄 Reassigning from other subjects")
                                        
                                        # Show all available teachers for reference
                                        all_teachers_query = """
                                        SELECT t.name, t.subject, t.performance_score, 
                                               COALESCE(student_counts.student_count, 0) as current_students
                                        FROM teachers t
                                        LEFT JOIN (
                                            SELECT c.teacher_id, COUNT(DISTINCT g.student_id) as student_count
                                            FROM courses c
                                            LEFT JOIN grades g ON c.course_id = g.course_id
                                            WHERE c.teacher_id IS NOT NULL AND c.teacher_id != ''
                                            GROUP BY c.teacher_id
                                        ) student_counts ON t.teacher_id = student_counts.teacher_id
                                        WHERE t.status = 'active'
                                        ORDER BY t.performance_score DESC
                                        LIMIT 5
                                        """
                                        
                                        all_teachers_raw = db.execute_query(all_teachers_query)
                                        if all_teachers_raw:
                                            st.info("**Top performing teachers (other subjects):**")
                                            for teacher in all_teachers_raw:
                                                st.write(f"• {teacher[0]} - {teacher[1]} (Performance: {teacher[2]:.1f}%, Students: {teacher[3]})")
                                        
                                except Exception as e:
                                    st.error("Error generating recommendation. Please try again.")
                                    
                        except Exception as e:
                            st.error("Error loading subjects. Using default subject list.")
                            # Fallback to default subjects
                            subject = st.selectbox("Subject", [
                                'Mathematics', 'Science', 'English', 'History', 'Art', 'Music', 
                                'Physical Education', 'Computer Science', 'Biology', 'Chemistry', 
                                'Physics', 'Literature', 'Geography', 'Economics', 'Psychology', 
                                'Foreign Languages', 'Drama', 'Health Education'
                            ], key="ai_subject_fallback")
                    
                    with col2:
                        st.write("**Teacher Workload Analysis**")
                        if len(df) > 0:
                            # Workload distribution
                            fig_workload = px.bar(df, x='Name', y='Students', 
                                                title="Teacher Workload (Students per Teacher)",
                                                color='Performance',
                                                color_continuous_scale='RdYlGn')
                            fig_workload.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_workload, width="stretch")
                            
                            # Workload statistics
                            st.write("**Workload Statistics:**")
                            st.write(f"- Average students per teacher: {df['Students'].mean():.1f}")
                            st.write(f"- Max students (single teacher): {df['Students'].max()}")
                            st.write(f"- Min students (single teacher): {df['Students'].min()}")
                            
                            # Identify overloaded teachers
                            overloaded = df[df['Students'] > 150]  # Assume 150+ is overloaded
                            if len(overloaded) > 0:
                                st.warning("⚠️ **Overloaded Teachers:**")
                                for _, teacher in overloaded.iterrows():
                                    st.write(f"• {teacher['Name']}: {teacher['Students']} students")
                            else:
                                st.success("✅ No teachers are currently overloaded")
                        else:
                            st.info("📊 Add teachers to view workload analysis")
                            
                        # Additional insights
                        st.write("**📈 System Insights**")
                        try:
                            # Get total courses
                            total_courses_query = "SELECT COUNT(*) FROM courses"
                            total_courses = db.execute_query(total_courses_query)
                            total_courses_count = total_courses[0][0] if total_courses else 0
                            
                            # Get courses without assigned teachers
                            unassigned_courses_query = "SELECT COUNT(*) FROM courses WHERE teacher_id IS NULL OR teacher_id = ''"
                            unassigned_courses = db.execute_query(unassigned_courses_query)
                            unassigned_count = unassigned_courses[0][0] if unassigned_courses else 0
                            
                            col_insight1, col_insight2 = st.columns(2)
                            with col_insight1:
                                st.metric("📚 Total Courses", total_courses_count)
                            with col_insight2:
                                st.metric("❓ Unassigned Courses", unassigned_count)
                                
                            if unassigned_count > 0:
                                st.warning(f"There are {unassigned_count} courses without assigned teachers.")
                                
                        except Exception as e:
                            st.error("Error fetching course insights.")
                
                # ==================== GENERATE REPORTS ====================
                with subtab3:
                    st.subheader("📊 Generate Reports")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Performance Reports**")
                        if st.button("📈 Performance Analysis", key="perf_analysis"):
                            if len(df) > 0:
                                # Generate performance summary
                                high_performers = df[df['Performance'] >= 90]
                                medium_performers = df[(df['Performance'] >= 70) & (df['Performance'] < 90)]
                                low_performers = df[df['Performance'] < 70]
                                
                                st.write("**📈 Performance Analysis Report**")
                                st.write(f"- **High Performers (≥90%):** {len(high_performers)} teachers ({len(high_performers)/len(df)*100:.1f}%)")
                                st.write(f"- **Medium Performers (70-89%):** {len(medium_performers)} teachers ({len(medium_performers)/len(df)*100:.1f}%)")
                                st.write(f"- **Low Performers (<70%):** {len(low_performers)} teachers ({len(low_performers)/len(df)*100:.1f}%)")
                                st.write(f"- **Average Performance:** {df['Performance'].mean():.1f}%")
                                st.write(f"- **Performance Standard Deviation:** {df['Performance'].std():.1f}")
                                
                                if len(low_performers) > 0:
                                    st.warning("**Teachers needing improvement:**")
                                    for _, teacher in low_performers.iterrows():
                                        st.write(f"• {teacher['Name']} ({teacher['Performance']:.1f}%) - {teacher['Subject']}")
                                
                                # Performance by department
                                dept_performance = df.groupby('Department')['Performance'].agg(['mean', 'count']).round(1)
                                st.write("**Performance by Department:**")
                                st.dataframe(dept_performance)
                    
                    with col2:
                        st.write("**Workload Reports**")
                        if st.button("👥 Workload Analysis", key="workload_analysis"):
                            if len(df) > 0:
                                st.write("**👥 Teacher Workload Report**")
                                st.write(f"- **Total Students:** {df['Students'].sum():,}")
                                st.write(f"- **Average Students per Teacher:** {df['Students'].mean():.1f}")
                                st.write(f"- **Student Distribution Range:** {df['Students'].min()} - {df['Students'].max()}")
                                
                                # Workload categories
                                light_load = df[df['Students'] < 50]
                                medium_load = df[(df['Students'] >= 50) & (df['Students'] < 100)]
                                heavy_load = df[df['Students'] >= 100]
                                
                                st.write(f"- **Light Load (<50 students):** {len(light_load)} teachers")
                                st.write(f"- **Medium Load (50-99 students):** {len(medium_load)} teachers")
                                st.write(f"- **Heavy Load (≥100 students):** {len(heavy_load)} teachers")
                                
                                if len(heavy_load) > 0:
                                    st.warning("**Teachers with heavy workload:**")
                                    for _, teacher in heavy_load.iterrows():
                                        st.write(f"• {teacher['Name']}: {teacher['Students']} students ({teacher['Subject']})")
                    
                    with col3:
                        st.write("**Department Reports**")
                        if st.button("🏢 Department Summary", key="dept_summary"):
                            if len(df) > 0:
                                st.write("**🏢 Department Summary Report**")
                                
                                dept_summary = df.groupby('Department').agg({
                                    'Name': 'count',
                                    'Performance': ['mean', 'std'],
                                    'Students': ['sum', 'mean'],
                                    'Class Teacher': lambda x: (x == '✅').sum()
                                }).round(1)
                                
                                dept_summary.columns = ['Total Teachers', 'Avg Performance', 'Performance StdDev', 
                                                      'Total Students', 'Avg Students/Teacher', 'Class Teachers']
                                
                                st.dataframe(dept_summary)
                                
                                # Department with most/least teachers
                                max_dept = dept_summary['Total Teachers'].idxmax()
                                min_dept = dept_summary['Total Teachers'].idxmin()
                                
                                st.write(f"- **Largest Department:** {max_dept} ({dept_summary.loc[max_dept, 'Total Teachers']} teachers)")
                                st.write(f"- **Smallest Department:** {min_dept} ({dept_summary.loc[min_dept, 'Total Teachers']} teachers)")
                                
                                # Best performing department
                                best_dept = dept_summary['Avg Performance'].idxmax()
                                st.write(f"- **Best Performing Department:** {best_dept} ({dept_summary.loc[best_dept, 'Avg Performance']:.1f}% avg)")
                    
                    # Comprehensive Report Export
                    st.write("---")
                    st.write("**📋 Export Comprehensive Reports**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📊 Export All Teacher Data", key="export_all"):
                            # Create comprehensive report
                            comprehensive_data = df.copy()
                            
                            # Add additional calculated fields
                            comprehensive_data['Performance Category'] = pd.cut(
                                comprehensive_data['Performance'], 
                                bins=[0, 70, 90, 100], 
                                labels=['Low', 'Medium', 'High']
                            )
                            
                            comprehensive_data['Workload Category'] = pd.cut(
                                comprehensive_data['Students'], 
                                bins=[0, 50, 100, float('inf')], 
                                labels=['Light', 'Medium', 'Heavy']
                            )
                            
                            csv = comprehensive_data.to_csv(index=False)
                            st.download_button(
                                label="Download Comprehensive Teacher Report",
                                data=csv,
                                file_name=f"teacher_comprehensive_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_comprehensive"
                            )
                    
                    with col2:
                        if st.button("📈 Export Performance Summary", key="export_perf"):
                            # Create performance summary
                            summary_data = {
                                'Metric': [
                                    'Total Teachers',
                                    'Average Performance',
                                    'High Performers (≥90%)',
                                    'Medium Performers (70-89%)',
                                    'Low Performers (<70%)',
                                    'Total Students',
                                    'Average Students per Teacher',
                                    'Class Teachers',
                                    'Assigned Class Teachers'
                                ],
                                'Value': [
                                    len(df),
                                    f"{df['Performance'].mean():.1f}%",
                                    len(df[df['Performance'] >= 90]),
                                    len(df[(df['Performance'] >= 70) & (df['Performance'] < 90)]),
                                    len(df[df['Performance'] < 70]),
                                    df['Students'].sum(),
                                    f"{df['Students'].mean():.1f}",
                                    len(df[df['Class Teacher'] == '✅']),
                                    len(df[(df['Class Teacher'] == '✅') & (df['Assigned Stream'] != 'Not Assigned')])
                                ]
                            }
                            
                            summary_df = pd.DataFrame(summary_data)
                            csv = summary_df.to_csv(index=False)
                            st.download_button(
                                label="Download Performance Summary",
                                data=csv,
                                file_name=f"teacher_performance_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_summary"
                            )

            # ==================== TEACHER UPDATES TAB ====================
            with tab3:
                st.subheader("⚙️ Teacher Management")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Update Teacher Performance**")
                    if len(df) > 0:
                        selected_teacher = st.selectbox("Select Teacher", df['Name'].tolist(), key="update_teacher")
                        new_performance = st.slider("New Performance Score", 0.0, 100.0, 85.0, key="update_performance")
                        
                        if st.button("Update Performance"):
                            teacher_id = df[df['Name'] == selected_teacher]['Teacher ID'].iloc[0]
                            try:
                                update_query = "UPDATE teachers SET performance_score = ? WHERE teacher_id = ?"
                                db.execute_update(update_query, (new_performance, teacher_id))
                                st.success(f"Updated {selected_teacher}'s performance to {new_performance}%")
                                st.rerun()
                            except Exception as e:
                                st.error("Error updating performance. Please try again.")
                
                with col2:
                    st.write("**Add New Teacher**")
                    if st.button("➕ Add Teacher"):
                        st.session_state.show_add_form = True

def academics_page():
    st.header("📚 Academic Management")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    gpa_calculator = GPACalculator(db_manager)
    
    def log_promotion_activity(student_id, student_name, from_grade, to_grade, action, 
                          gpa, attendance_rate, behavior_score, reason, academic_year=None):
        """Log promotion activity to history table"""
        if academic_year is None:
            from datetime import datetime
            current_year = datetime.now().year
            academic_year = f"{current_year}-{str(current_year + 1)[2:]}"
        
        try:
            # Remove any existing record for this student in this academic year to avoid duplicates
            db_manager.execute_update("""
                DELETE FROM promotion_history 
                WHERE student_id = ? AND academic_year = ?
            """, (student_id, academic_year))
            
            # Insert new record
            db_manager.execute_update("""
                INSERT INTO promotion_history 
                (academic_year, student_id, student_name, from_grade, to_grade, action, 
                 promotion_date, gpa, attendance_rate, behavior_score, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (academic_year, student_id, student_name, from_grade, to_grade, action,
                  datetime.now().strftime('%Y-%m-%d'), gpa, attendance_rate, behavior_score, reason))
        except Exception as e:
            st.error(f"Error logging promotion activity: {e}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Courses", "📝 Grades", "📅 Schedule", "🤖 Promotion"])
    
    with tab1:
        st.subheader("Course Catalog")
        
        # Add new course form
        with st.expander("➕ Add New Course"):
            with st.form("add_course_form"):
                col1, col2 = st.columns(2)
                with col1:
                    course_id = st.text_input("Course ID")
                    course_name = st.text_input("Course Name")
                    grade_level = st.selectbox("Grade Level", 
                        ["Grade 1", "Grade 2", "Grade 3", "Grade 4",
                         "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"
                        ])
                with col2:
                    # Check if it's a primary grade (1-6) or JHS
                    is_primary = grade_level in ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6"]
                    
                    if is_primary:
                        # For primary grades, teacher is optional
                        teachers_query = "SELECT teacher_id, name FROM teachers WHERE status = 'active'"
                        teachers_data = db_manager.execute_query(teachers_query)
                        
                        teacher_names = ["No Teacher (Optional for Primary)"] + [name for teacher_id, name in teachers_data]
                        teacher_id_map = {name: teacher_id for teacher_id, name in teachers_data}
                        
                        selected_teacher = st.selectbox("Teacher (Optional for Primary Grades)", teacher_names)
                        teacher_id_to_use = teacher_id_map.get(selected_teacher, None)
                    else:
                        # For JHS, teacher is required
                        teachers_query = "SELECT teacher_id, name FROM teachers WHERE status = 'active'"
                        teachers_data = db_manager.execute_query(teachers_query)
                        
                        teacher_names = ["Select a teacher..."] + [name for teacher_id, name in teachers_data]
                        teacher_id_map = {name: teacher_id for teacher_id, name in teachers_data}
                        
                        selected_teacher = st.selectbox("Teacher (Required for JHS)", teacher_names)
                        teacher_id_to_use = teacher_id_map.get(selected_teacher, None)
                    
                    credits = st.number_input("Credits", min_value=1, max_value=6, value=3)
                    schedule = st.text_input("Schedule (e.g., MWF 9:00-10:00)")
                
                if st.form_submit_button("Add Course"):
                    # Validation logic
                    if not course_id or not course_name:
                        st.error("Course ID and Course Name are required")
                    elif not is_primary and (not selected_teacher or selected_teacher == "Select a teacher..."):
                        st.error("Teacher is required for JHS courses")
                    else:
                        # For primary grades, teacher can be None
                        if is_primary and selected_teacher == "No Teacher (Optional for Primary)":
                            teacher_id_to_use = None
                        
                        insert_query = """
                            INSERT INTO courses (course_id, name, teacher_id, grade_level, credits, schedule)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """
                        try:
                            db_manager.execute_update(insert_query, 
                                (course_id, course_name, teacher_id_to_use, grade_level, credits, schedule))
                            
                            teacher_info = selected_teacher if teacher_id_to_use else "No Teacher Assigned"
                            st.success(f"Course '{course_name}' added successfully! Teacher: {teacher_info}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding course: {str(e)}")
        
        # Display courses from database
        courses_query = """
            SELECT c.course_id, c.name, COALESCE(t.name, 'No Teacher Assigned') as teacher_name, 
                   c.grade_level, c.credits, c.schedule,
                   COUNT(g.student_id) as enrolled_count
            FROM courses c
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            LEFT JOIN grades g ON c.course_id = g.course_id
            GROUP BY c.course_id, c.name, t.name, c.grade_level, c.credits, c.schedule
            ORDER BY c.name
        """
        
        courses_data = db_manager.execute_query(courses_query)
        
        if courses_data:
            courses_df = pd.DataFrame(courses_data, columns=[
                'Course ID', 'Course Name', 'Teacher', 'Grade Level', 
                'Credits', 'Schedule', 'Enrolled Students'
            ])
            st.dataframe(courses_df, width="stretch", hide_index=True)
        else:
            st.info("No courses found. Add some courses to get started!")
            
    with tab2:
        st.subheader("Grade Management")
        
        # Create sub-tabs for Grade Entry and GPA Management
        grade_tab1, grade_tab2 = st.tabs(["📝 Grade Entry", "📊 GPA Management"])
        
        # GPA Scale Information Display
        with st.expander("📊 GPA Scale Reference", expanded=False):
            gpa_scale_data = [
                ["A1", "75-100", "4.0", "Excellent"],
                ["B2", "70-74", "3.5", "Very Good"],
                ["B3", "65-69", "3.0", "Good"],
                ["C4", "60-64", "2.5", "Credit"],
                ["C5", "55-59", "2.0", "Credit"],
                ["C6", "50-54", "1.5", "Credit"],
                ["D7", "45-49", "1.0", "Pass"],
                ["E8", "40-44", "0.5", "Pass"],
                ["F9", "0-39", "0.0", "Fail"]
            ]
            gpa_scale_df = pd.DataFrame(gpa_scale_data, columns=['Grade', 'Percentage Range', 'GPA Points', 'Remark'])
            st.dataframe(gpa_scale_df, width="stretch", hide_index=True)
        
        # ====================== GRADE ENTRY TAB ======================
        with grade_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Grade Entry**")
                
                # Add grade form
                with st.form("add_grade_form"):
                    # Get students for dropdown
                    students_query = "SELECT student_id, name FROM students WHERE status = 'active'"
                    students_data = db_manager.execute_query(students_query)
                    student_options = {f"{name} ({student_id})": student_id for student_id, name in students_data}
                    
                    # Get courses for dropdown
                    courses_query = "SELECT course_id, name FROM courses"
                    courses_data = db_manager.execute_query(courses_query)
                    course_options = {f"{name} ({course_id})": course_id for course_id, name in courses_data}
                    
                    selected_student = st.selectbox("Select Student", list(student_options.keys()) if student_options else ["No students available"])
                    selected_course = st.selectbox("Select Course", list(course_options.keys()) if course_options else ["No courses available"])
                    grade_value = st.number_input("Grade (%)", min_value=0.0, max_value=100.0, step=0.1)
                    semester = st.selectbox("Semester", ["Advent", "Lent", "Trinity"])
                    year = st.number_input("Year", min_value=2025, max_value=2095, value=2025)
                    
                    # Show GPA equivalent
                    def get_gpa_points(percentage):
                        if percentage >= 75:
                            return 4.0, "A1", "Excellent"
                        elif percentage >= 70:
                            return 3.5, "B2", "Very Good"
                        elif percentage >= 65:
                            return 3.0, "B3", "Good"
                        elif percentage >= 60:
                            return 2.5, "C4", "Credit"
                        elif percentage >= 55:
                            return 2.0, "C5", "Credit"
                        elif percentage >= 50:
                            return 1.5, "C6", "Credit"
                        elif percentage >= 45:
                            return 1.0, "D7", "Pass"
                        elif percentage >= 40:
                            return 0.5, "E8", "Pass"
                        else:
                            return 0.0, "F9", "Fail"
                    
                    if grade_value > 0:
                        gpa_points, letter_grade, remark = get_gpa_points(grade_value)
                        st.info(f"**GPA Equivalent:** {gpa_points} ({letter_grade} - {remark})")
                    
                    if st.form_submit_button("Add Grade"):
                        if selected_student != "No students available" and selected_course != "No courses available":
                            import uuid
                            grade_id = str(uuid.uuid4())
                            student_id = student_options[selected_student]
                            course_id = course_options[selected_course]
                            
                            insert_grade_query = """
                                INSERT INTO grades (grade_id, student_id, course_id, grade, semester, year)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """
                            try:
                                db_manager.execute_update(insert_grade_query, 
                                    (grade_id, student_id, course_id, grade_value, semester, year))
                                st.success("Grade added successfully!")
                                
                                # Auto-update student GPA after adding grade
                                try:
                                    updated_gpa = gpa_calculator.update_student_gpa(student_id)
                                    st.info(f"Student GPA updated to: {updated_gpa:.2f}")
                                except Exception as e:
                                    st.warning(f"Grade added but GPA update failed: {e}")
                                
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error adding grade: {str(e)}")
            
            with col2:
                # Grade distribution chart from actual data
                st.write("**Grade Distribution**")
                grade_dist_query = """
                    SELECT 
                        CASE 
                            WHEN grade >= 75 THEN 'A1 (75-100)'
                            WHEN grade >= 70 THEN 'B2 (70-74)'
                            WHEN grade >= 65 THEN 'B3 (65-69)'
                            WHEN grade >= 60 THEN 'C4 (60-64)'
                            WHEN grade >= 55 THEN 'C5 (55-59)'
                            WHEN grade >= 50 THEN 'C6 (50-54)'
                            WHEN grade >= 45 THEN 'D7 (45-49)'
                            WHEN grade >= 40 THEN 'E8 (40-44)'
                            ELSE 'F9 (0-39)'
                        END as letter_grade,
                        COUNT(*) as count
                    FROM grades
                    GROUP BY letter_grade
                    ORDER BY 
                        CASE 
                            WHEN grade >= 75 THEN 1
                            WHEN grade >= 70 THEN 2
                            WHEN grade >= 65 THEN 3
                            WHEN grade >= 60 THEN 4
                            WHEN grade >= 55 THEN 5
                            WHEN grade >= 50 THEN 6
                            WHEN grade >= 45 THEN 7
                            WHEN grade >= 40 THEN 8
                            ELSE 9
                        END
                """
                
                grade_dist_data = db_manager.execute_query(grade_dist_query)
                
                if grade_dist_data:
                    import plotly.express as px  #
                    grades_df = pd.DataFrame(grade_dist_data, columns=['Grade', 'Count'])
                    fig = px.bar(grades_df, x='Grade', y='Count', title="Grade Distribution")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, width="stretch")
                    
                    # Calculate and display average grade and GPA
                    avg_query = "SELECT AVG(grade) as avg_grade FROM grades"
                    avg_result = db_manager.execute_query(avg_query)
                    if avg_result and avg_result[0][0]:
                        avg_percentage = avg_result[0][0]
                        avg_gpa, avg_letter, avg_remark = get_gpa_points(avg_percentage)
                        
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.metric("Average Grade", f"{avg_percentage:.1f}%")
                        with col2_2:
                            st.metric("Average GPA", f"{avg_gpa:.2f}")
                        st.info(f"Class Average: {avg_letter} - {avg_remark}")
                else:
                    st.info("No grades data available for chart")
            
            # Recent grades table
            st.markdown("---")
            st.subheader("Recent Grades")
            recent_grades_query = """
                SELECT s.name as student_name, c.name as course_name, 
                       g.grade, g.semester, g.year
                FROM grades g
                JOIN students s ON g.student_id = s.student_id
                JOIN courses c ON g.course_id = c.course_id
                ORDER BY g.year DESC, g.semester DESC
                LIMIT 10
            """
            
            recent_grades = db_manager.execute_query(recent_grades_query)
            if recent_grades:
                # Convert to DataFrame and add GPA columns
                recent_grades_data = []
                for student_name, course_name, grade, semester, year in recent_grades:
                    gpa_points, letter_grade, remark = get_gpa_points(grade)
                    recent_grades_data.append([
                        student_name, course_name, f"{grade:.1f}%", letter_grade, 
                        f"{gpa_points:.1f}", remark, semester, year
                    ])
                
                recent_grades_df = pd.DataFrame(recent_grades_data, columns=[
                    'Student', 'Course', 'Grade (%)', 'Letter Grade', 'GPA Points', 'Remark', 'Semester', 'Year'
                ])
                
                # Add color coding for grades
                def highlight_grades(row):
                    grade_val = float(row['Grade (%)'].replace('%', ''))
                    if grade_val >= 75:
                        return ['background-color: #d4edda; color: #155724'] * len(row)
                    elif grade_val >= 65:
                        return ['background-color: #d1ecf1; color: #0c5460'] * len(row)
                    elif grade_val >= 50:
                        return ['background-color: #fff3cd; color: #856404'] * len(row)
                    elif grade_val >= 40:
                        return ['background-color: #f8d7da; color: #721c24'] * len(row)
                    else:
                        return ['background-color: #f5c6cb; color: #721c24'] * len(row)
                
                styled_grades = recent_grades_df.style.apply(highlight_grades, axis=1)
                st.dataframe(styled_grades, width="stretch")
                
                # Add legend
                st.markdown("""
                **Grade Legend:**
                - 🟢 **A1 (75-100%)**: 4.0 GPA - Excellent
                - 🔵 **B2/B3 (65-74%)**: 3.5-3.0 GPA - Very Good/Good  
                - 🟡 **C4/C5/C6 (50-64%)**: 2.5-1.5 GPA - Credit
                - 🔴 **D7/E8 (40-49%)**: 1.0-0.5 GPA - Pass
                - ⚫ **F9 (0-39%)**: 0.0 GPA - Fail
                """)
            else:
                st.info("No recent grades found")
        
        # ====================== GPA MANAGEMENT TAB ======================
        with grade_tab2:
            st.markdown("### 📊 GPA Management Center")
            
            # GPA Calculation and Updates Section
            st.markdown("#### 🔄 GPA Updates")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Bulk GPA Operations**")
                
                # Bulk GPA update
                if st.button("🔄 Update All Student GPAs", type="primary"):
                    with st.spinner("Calculating GPAs for all students..."):
                        try:
                            updated_gpas = gpa_calculator.bulk_update_all_gpas()
                            st.success(f"Updated GPAs for {len(updated_gpas)} students!")
                            
                            # Show summary
                            if updated_gpas:
                                avg_gpa = sum(updated_gpas.values()) / len(updated_gpas)
                                st.info(f"Average GPA: {avg_gpa:.2f}")
                                
                                # Show top performers
                                sorted_gpas = sorted(updated_gpas.items(), key=lambda x: x[1], reverse=True)
                                
                                if len(sorted_gpas) >= 3:
                                    st.markdown("**Top Performers:**")
                                    for i, (student_name, gpa) in enumerate(sorted_gpas[:3]):
                                        # Get letter grade for GPA
                                        if gpa >= 3.75:
                                            letter = "A1"
                                        elif gpa >= 3.25:
                                            letter = "B2"
                                        elif gpa >= 2.75:
                                            letter = "B3"
                                        elif gpa >= 2.25:
                                            letter = "C4"
                                        elif gpa >= 1.75:
                                            letter = "C5"
                                        elif gpa >= 1.25:
                                            letter = "C6"
                                        elif gpa >= 0.75:
                                            letter = "D7"
                                        elif gpa >= 0.25:
                                            letter = "E8"
                                        else:
                                            letter = "F9"
                                        
                                        st.write(f"{i+1}. {student_name}: {gpa:.2f} ({letter})")
                                
                                st.rerun()
                            else:
                                st.warning("No students found or no grades available")
                                
                        except Exception as e:
                            st.error(f"Error updating GPAs: {str(e)}")
                
                # Individual student GPA update
                st.markdown("**Individual Student GPA Update**")
                students_query = "SELECT student_id, name FROM students WHERE status = 'active'"
                students_data = db_manager.execute_query(students_query)
                
                if students_data:
                    student_options = {f"{name} ({student_id})": student_id for student_id, name in students_data}
                    selected_student_gpa = st.selectbox("Select Student for GPA Update", 
                                                      ["Select a student..."] + list(student_options.keys()),
                                                      key="gpa_update_student")
                    
                    if selected_student_gpa != "Select a student...":
                        student_id = student_options[selected_student_gpa]
                        student_name = selected_student_gpa.split(' (')[0]
                        
                        if st.button(f"Update GPA for {student_name}", key="update_individual_gpa"):
                            try:
                                updated_gpa = gpa_calculator.update_student_gpa(student_id)
                                st.success(f"GPA updated to {updated_gpa:.2f}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating GPA: {str(e)}")
                else:
                    st.info("No active students found")
            
            with col2:
                st.markdown("**Class GPA Statistics**")
                
                # Get and display GPA statistics
                try:
                    stats = gpa_calculator.get_class_gpa_statistics()
                    
                    if stats['count'] > 0:
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.metric("Average GPA", f"{stats['average_gpa']:.2f}")
                            st.metric("Students", stats['count'])
                        with col2_2:
                            st.metric("Highest GPA", f"{stats['max_gpa']:.2f}")
                            st.metric("Lowest GPA", f"{stats['min_gpa']:.2f}")
                        
                        # GPA distribution
                        gpa_distribution = stats['distribution']
                        
                        # GPA Distribution pie chart
                        if any(gpa_distribution.values()):
                            dist_df = pd.DataFrame(list(gpa_distribution.items()), columns=['GPA Range', 'Count'])
                            dist_df = dist_df[dist_df['Count'] > 0]  # Filter out zero counts
                            
                            if not dist_df.empty:
                                fig_pie = px.pie(dist_df, values='Count', names='GPA Range', 
                                                title="GPA Distribution by Grade Range")
                                st.plotly_chart(fig_pie, width="stretch")
                            else:
                                st.info("No GPA distribution data available")
                        else:
                            st.info("No GPA distribution data available")
                    else:
                        st.info("No GPA data available. Add grades and update GPAs.")
                except Exception as e:
                    st.error(f"Error loading GPA statistics: {str(e)}")
            
            # Individual Student GPA Analysis Section
            st.markdown("---")
            st.markdown("### 👤 Individual Student GPA Analysis")
            
            students_query = "SELECT student_id, name FROM students WHERE status = 'active'"
            students_data = db_manager.execute_query(students_query)
            
            if students_data:
                student_options = {f"{name} ({student_id})": student_id for student_id, name in students_data}
                selected_student_analysis = st.selectbox("Select Student for Detailed Analysis", 
                                                        ["Select a student..."] + list(student_options.keys()),
                                                        key="gpa_analysis_student")
                
                if selected_student_analysis != "Select a student...":
                    student_id = student_options[selected_student_analysis]
                    student_name = selected_student_analysis.split(' (')[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Current overall GPA
                        st.markdown("**Current Performance**")
                        try:
                            current_gpa = gpa_calculator.calculate_gpa(student_id)
                            st.metric("Overall GPA", f"{current_gpa:.2f}/4.0")
                            
                            # Letter grade equivalent and performance level
                            if current_gpa >= 3.5:
                                letter_grade = "A"
                                grade_color = "🟢"
                                performance = "Excellent/Very Good"
                            elif current_gpa >= 2.5:
                                letter_grade = "B"
                                grade_color = "🟡"
                                performance = "Good"
                            elif current_gpa >= 1.5:
                                letter_grade = "C"
                                grade_color = "🟠"
                                performance = "Credit"
                            elif current_gpa >= 0.5:
                                letter_grade = "D-E"
                                grade_color = "🔴"
                                performance = "Pass"
                            else:
                                letter_grade = "F"
                                grade_color = "⚫"
                                performance = "Fail"
                            
                            st.write(f"{grade_color} **Grade:** {letter_grade}")
                            st.write(f"**Performance:** {performance}")
                            
                            # GPA progress indicator
                            progress = min(current_gpa / 4.0, 1.0)
                            st.progress(progress)
                            
                        except Exception as e:
                            st.error(f"Error calculating GPA: {str(e)}")
                            current_gpa = 0.0
                    
                    with col2:
                        # Semester breakdown
                        st.markdown("**Semester Performance**")
                        try:
                            semester_gpas = gpa_calculator.get_gpa_by_semester(student_id)
                            
                            if semester_gpas:
                                for sem_data in semester_gpas:
                                    semester_gpa = sem_data['gpa']
                                    # Add trend indicator
                                    if len(semester_gpas) > 1:
                                        current_index = semester_gpas.index(sem_data)
                                        if current_index > 0:
                                            prev_gpa = semester_gpas[current_index - 1]['gpa']
                                            if semester_gpa > prev_gpa:
                                                trend = "📈"
                                            elif semester_gpa < prev_gpa:
                                                trend = "📉"
                                            else:
                                                trend = "➡️"
                                        else:
                                            trend = ""
                                    else:
                                        trend = ""
                                    
                                    # Get letter grade for semester GPA
                                    if semester_gpa >= 3.5:
                                        sem_letter = "A"
                                    elif semester_gpa >= 2.5:
                                        sem_letter = "B"
                                    elif semester_gpa >= 1.5:
                                        sem_letter = "C"
                                    elif semester_gpa >= 0.5:
                                        sem_letter = "D-E"
                                    else:
                                        sem_letter = "F"
                                    
                                    st.write(f"{sem_data['semester']} {sem_data['year']}: {semester_gpa:.2f} ({sem_letter}) {trend}")
                            else:
                                st.info("No semester data available")
                        except Exception as e:
                            st.error(f"Error loading semester data: {str(e)}")
                    
                    with col3:
                        # Student course grades
                        st.markdown("**Recent Course Grades**")
                        try:
                            course_grades_query = """
                                SELECT c.name as course_name, g.grade, g.semester, g.year
                                FROM grades g
                                JOIN courses c ON g.course_id = c.course_id
                                WHERE g.student_id = ?
                                ORDER BY g.year DESC, g.semester DESC, c.name
                            """
                            
                            course_grades = db_manager.execute_query(course_grades_query, (student_id,))
                            
                            if course_grades:
                                # Show most recent grades (limit to 5)
                                for course_name, grade, semester, year in course_grades[:5]:
                                    # Get GPA points and letter grade for this grade
                                    def get_gpa_points_local(percentage):
                                        if percentage >= 75:
                                            return 4.0, "A1", "Excellent"
                                        elif percentage >= 70:
                                            return 3.5, "B2", "Very Good"
                                        elif percentage >= 65:
                                            return 3.0, "B3", "Good"
                                        elif percentage >= 60:
                                            return 2.5, "C4", "Credit"
                                        elif percentage >= 55:
                                            return 2.0, "C5", "Credit"
                                        elif percentage >= 50:
                                            return 1.5, "C6", "Credit"
                                        elif percentage >= 45:
                                            return 1.0, "D7", "Pass"
                                        elif percentage >= 40:
                                            return 0.5, "E8", "Pass"
                                        else:
                                            return 0.0, "F9", "Fail"
                                    
                                    gpa_points, letter_grade, remark = get_gpa_points_local(grade)
                                    
#################################   # Color code grades based on scale
                                    if grade >= 75:
                                        grade_emoji = "🟢"
                                    elif grade >= 65:
                                        grade_emoji = "🟡"
                                    elif grade >= 50:
                                        grade_emoji = "🟠"
                                    elif grade >= 40:
                                        grade_emoji = "🔴"
                                    else:
                                        grade_emoji = "⚫"
                                    
                                    st.write(f"{grade_emoji} **{course_name}**: {grade:.1f}% ({letter_grade})")
                                
                                if len(course_grades) > 5:
                                    st.write(f"... and {len(course_grades) - 5} more grades")
                            else:
                                st.info("No grades available for this student")
                                
                        except Exception as e:
                            st.error(f"Error loading course grades: {str(e)}")
                    
                    # GPA Trend Analysis
                    st.markdown("---")
                    st.markdown("#### 📈 GPA Trend Analysis")
                    
                    try:
                        semester_gpas = gpa_calculator.get_gpa_by_semester(student_id)
                        
                        if semester_gpas and len(semester_gpas) > 1:
                            # Create trend chart
                            trend_data = []
                            for sem_data in semester_gpas:
                                semester_label = f"{sem_data['semester']} {sem_data['year']}"
                                trend_data.append({
                                    'Semester': semester_label,
                                    'GPA': sem_data['gpa']
                                })
                            
                            trend_df = pd.DataFrame(trend_data)
                            
                            fig_line = px.line(trend_df, x='Semester', y='GPA', 
                                             title=f"GPA Trend for {student_name}",
                                             markers=True)
                            fig_line.update_yaxis(range=[0, 4.0])
                            fig_line.add_hline(y=2.0, line_dash="dash", line_color="red", 
                                             annotation_text="Minimum Pass GPA")
                            fig_line.add_hline(y=3.0, line_dash="dash", line_color="green", 
                                             annotation_text="Good Performance")
                            
                            st.plotly_chart(fig_line, width="stretch")
                            
                            # Trend analysis
                            if len(semester_gpas) >= 2:
                                recent_gpa = semester_gpas[0]['gpa']
                                previous_gpa = semester_gpas[1]['gpa']
                                
                                if recent_gpa > previous_gpa:
                                    st.success(f"📈 **Improving Performance**: GPA increased by {recent_gpa - previous_gpa:.2f} points")
                                elif recent_gpa < previous_gpa:
                                    st.warning(f"📉 **Declining Performance**: GPA decreased by {previous_gpa - recent_gpa:.2f} points")
                                else:
                                    st.info("➡️ **Stable Performance**: GPA remained consistent")
                        else:
                            st.info("Need at least 2 semesters of data to show trend analysis")
                            
                    except Exception as e:
                        st.error(f"Error creating trend analysis: {str(e)}")
            else:
                st.info("No active students found")
            
            # GPA Leaderboard
            st.markdown("---")
            st.markdown("### 🏆 GPA Leaderboard")
            
            try:
                # Query for top students by GPA
                leaderboard_query = """
                    SELECT s.name, s.gpa, s.student_id
                    FROM students s
                    WHERE s.status = 'active' AND s.gpa IS NOT NULL
                    ORDER BY s.gpa DESC
                    LIMIT 10
                """
                
                leaderboard_data = db_manager.execute_query(leaderboard_query)
                
                if leaderboard_data:
                    leaderboard_list = []
                    for i, (name, gpa, student_id) in enumerate(leaderboard_data):
                        # Get letter grade for GPA
                        if gpa >= 3.5:
                            letter_grade = "A"
                            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                        elif gpa >= 2.5:
                            letter_grade = "B"
                            medal = "🏅"
                        elif gpa >= 1.5:
                            letter_grade = "C"
                            medal = "📚"
                        else:
                            letter_grade = "D/F"
                            medal = "📝"
                        
                        leaderboard_list.append({
                            'Rank': i + 1,
                            'Student': name,
                            'GPA': f"{gpa:.2f}",
                            'Grade': letter_grade,
                            'Medal': medal
                        })
                    
                    leaderboard_df = pd.DataFrame(leaderboard_list)
                    
                    # Display as a nice table
                    st.dataframe(leaderboard_df, width="stretch", hide_index=True)
                    
                    # Quick stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        honor_students = len([gpa for _, gpa, _ in leaderboard_data if gpa >= 3.5])
                        st.metric("Honor Students (GPA ≥ 3.5)", honor_students)
                    with col2:
                        good_students = len([gpa for _, gpa, _ in leaderboard_data if 2.5 <= gpa < 3.5])
                        st.metric("Good Standing (GPA 2.5-3.4)", good_students)
                    with col3:
                        at_risk_students = len([gpa for _, gpa, _ in leaderboard_data if gpa < 2.0])
                        st.metric("At Risk (GPA < 2.0)", at_risk_students)
                else:
                    st.info("No GPA data available for leaderboard")
                    
            except Exception as e:
                st.error(f"Error loading leaderboard: {str(e)}")
#--------------------------------
    with tab3:
            st.subheader("Class Schedule")
            
            # Display schedule based on courses
            schedule_query = """
                SELECT c.name as course_name, COALESCE(t.name, 'No Teacher Assigned') as teacher_name, 
                       c.schedule, c.grade_level
                FROM courses c
                LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
                WHERE c.schedule IS NOT NULL AND c.schedule != ''
                ORDER BY c.grade_level, c.schedule
            """
            
            schedule_data = db_manager.execute_query(schedule_query)
            
            if schedule_data:
                schedule_df = pd.DataFrame(schedule_data, columns=[
                    'Course', 'Teacher', 'Schedule', 'Grade Level'
                ])
                
                # Group by grade level
                for grade in schedule_df['Grade Level'].unique():
                    st.write(f"**{grade} Schedule:**")
                    grade_schedule = schedule_df[schedule_df['Grade Level'] == grade]
                    st.dataframe(grade_schedule[['Course', 'Teacher', 'Schedule']], 
                               width="stretch", hide_index=True)
                    st.write("")
            else:
                st.info("No scheduled courses found. Add course schedules to see them here!")
            
            # Schedule management
            st.subheader("Schedule Management")
            
            with st.expander("📅 Update Course Schedule"):
                courses_query = "SELECT course_id, name FROM courses"
                courses_data = db_manager.execute_query(courses_query)
                
                if courses_data:
                    course_options = {f"{name} ({course_id})": course_id for course_id, name in courses_data}
                    
                    selected_course = st.selectbox("Select Course to Schedule", list(course_options.keys()))
                    new_schedule = st.text_input("Schedule (e.g., MWF 9:00-10:00 AM)")
                    
                    if st.button("Update Schedule"):
                        if new_schedule:
                            course_id = course_options[selected_course]
                            update_query = "UPDATE courses SET schedule = ? WHERE course_id = ?"
                            try:
                                db_manager.execute_update(update_query, (new_schedule, course_id))
                                st.success("Schedule updated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating schedule: {str(e)}")
                        else:
                            st.error("Please enter a schedule")
                else:
                    st.info("No courses available to schedule")

    with tab4:
        st.subheader("🤖 AI-Powered Grade Promotion System")
        
        # Promotion criteria display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Promotion Criteria")
            criteria_info = """
            **Academic Requirements:**
            - Minimum GPA: 2.0
            - Minimum Attendance: 75%
            - Minimum Behavior Score: 60
            
            **AI Evaluation Factors:**
            - Overall academic performance
            - Attendance patterns
            - Behavioral assessments
            - Individual progress tracking
            """
            st.info(criteria_info)
            
            # Manual promotion settings
            st.markdown("### ⚙️ Promotion Settings")
            min_gpa = st.number_input("Minimum GPA for Promotion", 
                                    min_value=0.0, max_value=4.0, value=2.0, step=0.1)
            min_attendance = st.number_input("Minimum Attendance (%)", 
                                           min_value=0, max_value=100, value=75)
            min_behavior = st.number_input("Minimum Behavior Score", 
                                         min_value=0, max_value=100, value=60)
        
        with col2:
            st.markdown("### 📊 Promotion Statistics")
            
            # Get current academic year promotion stats
            promotion_stats_query = """
                SELECT 
                    grade,
                    COUNT(*) as total_students,
                    AVG(gpa) as avg_gpa,
                    AVG(attendance_rate) as avg_attendance,
                    AVG(behavior_score) as avg_behavior,
                    COUNT(CASE WHEN gpa >= 2.0 AND attendance_rate >= 75 AND behavior_score >= 60 THEN 1 END) as eligible_for_promotion
                FROM students 
                WHERE status = 'active'
                GROUP BY grade
                ORDER BY grade
            """
            
            stats_data = db_manager.execute_query(promotion_stats_query)
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data, columns=[
                    'Grade', 'Total Students', 'Avg GPA', 'Avg Attendance', 
                    'Avg Behavior', 'Eligible for Promotion'
                ])
                
                # Calculate promotion rate
                stats_df['Promotion Rate %'] = (stats_df['Eligible for Promotion'] / stats_df['Total Students'] * 100).round(1)
                
                st.dataframe(stats_df, width="stretch", hide_index=True)
            else:
                st.info("No student data available for promotion analysis")
        
        st.markdown("---")
        
        # Individual student promotion review
        st.markdown("### 👥 Individual Student Review")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get students for review
            students_query = """
                SELECT s.student_id, s.name, s.grade, s.gpa, s.attendance_rate, s.behavior_score,
                       CASE 
                           WHEN s.gpa >= ? AND s.attendance_rate >= ? AND s.behavior_score >= ? 
                           THEN 'Eligible' 
                           ELSE 'Review Needed' 
                       END as promotion_status
                FROM students s
                WHERE s.status = 'active'
                ORDER BY s.grade, s.name
            """
            
            students_data = db_manager.execute_query(students_query, (min_gpa, min_attendance, min_behavior))
            
            if students_data:
                students_df = pd.DataFrame(students_data, columns=[
                    'Student ID', 'Name', 'Current Grade', 'GPA', 'Attendance %', 
                    'Behavior Score', 'Status'
                ])
                
                # Color coding for status
                def highlight_status(val):
                    if val == 'Eligible':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'Review Needed':
                        return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                styled_df = students_df.style.map(highlight_status, subset=['Status'])
                st.dataframe(styled_df, width="stretch", hide_index=True)
                
                # Select student for detailed AI analysis
                student_names = [f"{row[1]} ({row[0]})" for row in students_data]
                selected_student = st.selectbox("Select Student for AI Analysis", student_names)
                
                if selected_student:
                    student_id = selected_student.split('(')[1].replace(')', '')
                    
                    if st.button("🤖 Get AI Promotion Recommendation"):
                        with st.spinner("AI analyzing student performance..."):
                            # Get detailed student data
                            student_detail_query = """
                                SELECT s.*, 
                                       AVG(g.grade) as avg_course_grade,
                                       COUNT(g.grade_id) as total_grades
                                FROM students s
                                LEFT JOIN grades g ON s.student_id = g.student_id
                                WHERE s.student_id = ?
                                GROUP BY s.student_id
                            """
                            
                            student_detail = db_manager.execute_query(student_detail_query, (student_id,))
                            
                            if student_detail:
                                student = student_detail[0]
                                
                                # Safely handle ALL potential None values
                                student_id = student[0] if student[0] is not None else "N/A"
                                student_name = student[1] if student[1] is not None else "N/A"
                                current_grade = student[3] if student[3] is not None else "N/A"
                                gpa = student[5] if student[5] is not None else 0.0
                                attendance_rate = student[6] if student[6] is not None else 0.0
                                behavior_score = student[7] if student[7] is not None else 0
                                avg_course_grade = student[-2] if student[-2] is not None else None
                                
                                # Create safe display strings
                                gpa_display = f"{gpa:.2f}" if gpa is not None else "N/A"
                                attendance_display = f"{attendance_rate:.1f}" if attendance_rate is not None else "N/A"
                                behavior_display = f"{behavior_score}" if behavior_score is not None else "N/A"
                                avg_course_display = f"{avg_course_grade:.2f}" if avg_course_grade is not None else "N/A"
                                
                                # AI Promotion Analysis with safe variables
                                ai_analysis = f"""
**AI PROMOTION ANALYSIS**

**Student:** {student_name} (ID: {student_id})
**Current Grade:** {current_grade}
**Academic Performance:**
- Overall GPA: {gpa_display}
- Average Course Grade: {avg_course_display}
- Attendance Rate: {attendance_display}%
- Behavior Score: {behavior_display}/100

**Criteria Assessment:**
- GPA Requirement (≥{min_gpa}): {'✅ MET' if (gpa is not None and gpa >= min_gpa) else '❌ NOT MET'}
- Attendance Requirement (≥{min_attendance}%): {'✅ MET' if (attendance_rate is not None and attendance_rate >= min_attendance) else '❌ NOT MET'}
- Behavior Requirement (≥{min_behavior}): {'✅ MET' if (behavior_score is not None and behavior_score >= min_behavior) else '❌ NOT MET'}

**AI RECOMMENDATION:**
"""
                                
                                # Safe criteria checking
                                meets_criteria = (
                                    gpa is not None and gpa >= min_gpa and
                                    attendance_rate is not None and attendance_rate >= min_attendance and 
                                    behavior_score is not None and behavior_score >= min_behavior
                                )
                                
                                if meets_criteria:
                                    next_grade = get_next_grade(current_grade)
                                    ai_analysis += f"""
**PROMOTE** to {next_grade}

**Reasoning:** Student demonstrates consistent academic performance above minimum requirements. Strong attendance and behavior scores indicate readiness for next grade level challenges.

**Recommendations for Success:**
- Continue monitoring academic progress
- Maintain current attendance patterns
- Consider advanced placement opportunities
"""
                                else:
                                    ai_analysis += f"""
**RETAIN** in current grade

**Reasoning:** Student has not met all promotion criteria. Additional support needed before advancement.

**Intervention Recommendations:**
"""
                                    
                                    if gpa is None or gpa < min_gpa:
                                        ai_analysis += "\n- Academic tutoring and remedial support"
                                    if attendance_rate is None or attendance_rate < min_attendance:
                                        ai_analysis += "\n- Attendance intervention program"
                                    if behavior_score is None or behavior_score < min_behavior:
                                        ai_analysis += "\n- Behavioral counseling and support"
                                
                                st.markdown(ai_analysis)
                       
        with col2:
            st.markdown("### 🎯 Quick Actions")
            
            # Bulk promotion with logging
            if st.button("🚀 Run Bulk AI Promotion", type="primary"):
                with st.spinner("Processing AI-powered promotions for all students..."):
                    try:
                        # Create promotion history table if it doesn't exist
                        db_manager.execute_update("""
                            CREATE TABLE IF NOT EXISTS promotion_history (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                academic_year TEXT NOT NULL,
                                student_id TEXT NOT NULL,
                                student_name TEXT NOT NULL,
                                from_grade TEXT NOT NULL,
                                to_grade TEXT NOT NULL,
                                action TEXT NOT NULL,
                                promotion_date TEXT NOT NULL,
                                gpa REAL,
                                attendance_rate REAL,
                                behavior_score REAL,
                                reason TEXT
                            )
                        """)
                        
                        # Get all students for processing
                        students_to_process = db_manager.execute_query("""
                            SELECT student_id, name, grade, gpa, attendance_rate, behavior_score
                            FROM students 
                            WHERE status = 'active'
                            ORDER BY grade, name
                        """)
                        
                        promoted_count = 0
                        retained_count = 0
                        graduated_count = 0
                        promotion_results = []
                        
                        for student in students_to_process:
                            student_id, name, current_grade, gpa, attendance_rate, behavior_score = student
                            
                            # Check promotion criteria
                            meets_gpa = gpa >= min_gpa
                            meets_attendance = attendance_rate >= min_attendance
                            meets_behavior = behavior_score >= min_behavior
                            
                            if meets_gpa and meets_attendance and meets_behavior:
                                # Student meets criteria - promote
                                next_grade = get_next_grade(current_grade)
                                
                                if next_grade == "GRADUATED":
                                    # Graduate the student
                                    db_manager.execute_update(
                                        "UPDATE students SET status = 'graduated' WHERE student_id = ?",
                                        (student_id,)
                                    )
                                    graduated_count += 1
                                    action = 'graduated'
                                    to_grade = 'GRADUATED'
                                    reason = 'Met all promotion criteria - Graduated'
                                else:
                                    # Promote to next grade
                                    db_manager.execute_update(
                                        "UPDATE students SET grade = ? WHERE student_id = ?",
                                        (next_grade, student_id)
                                    )
                                    promoted_count += 1
                                    action = 'promoted'
                                    to_grade = next_grade
                                    reason = 'Met all promotion criteria'
                                
                                # Log the promotion activity
                                log_promotion_activity(
                                    student_id=student_id,
                                    student_name=name,
                                    from_grade=current_grade,
                                    to_grade=to_grade,
                                    action=action,
                                    gpa=gpa,
                                    attendance_rate=attendance_rate,
                                    behavior_score=behavior_score,
                                    reason=reason
                                )
                                
                                promotion_results.append({
                                    'Name': name,
                                    'Action': action.title(),
                                    'From Grade': current_grade,
                                    'To Grade': to_grade,
                                    'Reason': reason
                                })
                            else:
                                # Student doesn't meet criteria - retain
                                retained_count += 1
                                reasons = []
                                if not meets_gpa:
                                    reasons.append(f"GPA {gpa:.2f} < {min_gpa}")
                                if not meets_attendance:
                                    reasons.append(f"Attendance {attendance_rate:.1f}% < {min_attendance}%")
                                if not meets_behavior:
                                    reasons.append(f"Behavior {behavior_score} < {min_behavior}")
                                
                                reason = '; '.join(reasons)
                                
                                # Log the retention
                                log_promotion_activity(
                                    student_id=student_id,
                                    student_name=name,
                                    from_grade=current_grade,
                                    to_grade=current_grade,
                                    action='retained',
                                    gpa=gpa,
                                    attendance_rate=attendance_rate,
                                    behavior_score=behavior_score,
                                    reason=reason
                                )
                                
                                promotion_results.append({
                                    'Name': name,
                                    'Action': 'Retained',
                                    'From Grade': current_grade,
                                    'To Grade': current_grade,
                                    'Reason': reason
                                })
                        
                        # Display results
                        total_processed = len(students_to_process)
                        
                        st.success(f"""
                        **Bulk Promotion Complete!**
                        - Total Students Processed: {total_processed}
                        - Students Promoted: {promoted_count}
                        - Students Graduated: {graduated_count}
                        - Students Retained: {retained_count}
                        """)
                        
                        # Show detailed results
                        with st.expander("View Detailed Results", expanded=True):
                            if promotion_results:
                                results_df = pd.DataFrame(promotion_results)
                                
                                # Color code the results
                                def highlight_action(val):
                                    if val == 'Promoted':
                                        return 'background-color: #d4edda; color: #155724'
                                    elif val == 'Graduated':
                                        return 'background-color: #d1ecf1; color: #0c5460'
                                    elif val == 'Retained':
                                        return 'background-color: #f8d7da; color: #721c24'
                                    return ''
                                
                                styled_results = results_df.style.map(highlight_action, subset=['Action'])
                                st.dataframe(styled_results, width="stretch", hide_index=True)
                            else:
                                st.info("No students to process")
                        
                        # Force refresh to show updated data
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during bulk promotion: {str(e)}")
            
            st.markdown("---")

            # Historical promotion analysis
            st.markdown("### 📊 Historical Promotion Analysis")

            try:
                # Create promotion tracking table if it doesn't exist
                db_manager.execute_update("""
                    CREATE TABLE IF NOT EXISTS promotion_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        academic_year TEXT NOT NULL,
                        student_id TEXT NOT NULL,
                        student_name TEXT NOT NULL,
                        from_grade TEXT NOT NULL,
                        to_grade TEXT NOT NULL,
                        action TEXT NOT NULL,
                        promotion_date TEXT NOT NULL,
                        gpa REAL,
                        attendance_rate REAL,
                        behavior_score REAL,
                        reason TEXT
                    )
                """)
                
                # Get current academic year
                from datetime import datetime
                current_year = datetime.now().year
                current_academic_year = f"{current_year}-{str(current_year + 1)[2:]}"
                
                # Check if we have promotion data, if not, create initial data from current students
                existing_data = db_manager.execute_query("""
                    SELECT COUNT(*) FROM promotion_history 
                    WHERE academic_year = ?
                """, (current_academic_year,))
                
                if existing_data[0][0] == 0:
                    # No data exists, initialize with current student status
                    current_students = db_manager.execute_query("""
                        SELECT student_id, name, grade, gpa, attendance_rate, behavior_score, status
                        FROM students
                    """)
                    
                    # Add current students as baseline data
                    for student in current_students:
                        student_id, name, grade, gpa, attendance, behavior, status = student
                        if status == 'graduated':
                            action = 'graduated'
                            to_grade = 'GRADUATED'
                        else:
                            action = 'active'
                            to_grade = grade
                        
                        db_manager.execute_update("""
                            INSERT INTO promotion_history 
                            (academic_year, student_id, student_name, from_grade, to_grade, action, 
                             promotion_date, gpa, attendance_rate, behavior_score, reason)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (current_academic_year, student_id, name, grade, to_grade, action,
                              datetime.now().strftime('%Y-%m-%d'), gpa, attendance, behavior, 
                              'Initial enrollment data'))
                
                # Get historical promotion statistics
                historical_query = """
                    SELECT 
                        academic_year,
                        COUNT(*) as total_students,
                        COUNT(CASE WHEN action = 'promoted' THEN 1 END) as promoted,
                        COUNT(CASE WHEN action = 'retained' THEN 1 END) as retained,
                        COUNT(CASE WHEN action = 'graduated' THEN 1 END) as graduated,
                        ROUND(
                            (COUNT(CASE WHEN action = 'promoted' THEN 1 END) * 100.0 / 
                             NULLIF(COUNT(CASE WHEN action IN ('promoted', 'retained') THEN 1 END), 0)), 1
                        ) as promotion_rate
                    FROM promotion_history
                    WHERE action IN ('promoted', 'retained', 'graduated')
                    GROUP BY academic_year
                    ORDER BY academic_year DESC
                    LIMIT 5
                """
                
                historical_data = db_manager.execute_query(historical_query)
                
                if historical_data:
                    # Convert to DataFrame
                    historical_df = pd.DataFrame(historical_data, columns=[
                        'Academic Year', 'Total Students', 'Promoted', 'Retained', 'Graduated', 'Promotion Rate %'
                    ])
                    
                    # Fill NaN values with 0
                    historical_df = historical_df.fillna(0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(historical_df, width="stretch", hide_index=True)
                        
                        # Show detailed breakdown for current year
                        current_year_detail = db_manager.execute_query("""
                            SELECT 
                                from_grade,
                                COUNT(*) as total,
                                COUNT(CASE WHEN action = 'promoted' THEN 1 END) as promoted,
                                COUNT(CASE WHEN action = 'retained' THEN 1 END) as retained,
                                COUNT(CASE WHEN action = 'graduated' THEN 1 END) as graduated
                            FROM promotion_history
                            WHERE academic_year = ? AND action IN ('promoted', 'retained', 'graduated')
                            GROUP BY from_grade
                            ORDER BY 
                                CASE from_grade
                                    WHEN 'Grade 1' THEN 1
                                    WHEN 'Grade 2' THEN 2
                                    WHEN 'Grade 3' THEN 3
                                    WHEN 'Grade 4' THEN 4
                                    WHEN 'Grade 5' THEN 5
                                    WHEN 'Grade 6' THEN 6
                                    WHEN 'JHS 1' THEN 7
                                    WHEN 'JHS 2' THEN 8
                                    WHEN 'JHS 3' THEN 9
                                    ELSE 10
                                END
                        """, (current_academic_year,))
                        
                        if current_year_detail:
                            st.markdown(f"#### Current Year ({current_academic_year}) by Grade")
                            detail_df = pd.DataFrame(current_year_detail, columns=[
                                'Grade', 'Total', 'Promoted', 'Retained', 'Graduated'
                            ])
                            st.dataframe(detail_df, width="stretch", hide_index=True)
                    
                    with col2:
                        # Create promotion rate trend chart
                        if len(historical_df) > 1:
                            import plotly.express as px
                            fig = px.line(historical_df, x='Academic Year', y='Promotion Rate %', 
                                         title='Promotion Rate Trend', markers=True)
                            fig.update_layout(yaxis_range=[0, 100])
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.info("Need more historical data to show trend chart")
                        
                        # Show recent promotion activities
                        recent_promotions = db_manager.execute_query("""
                            SELECT student_name, from_grade, to_grade, action, promotion_date
                            FROM promotion_history
                            WHERE academic_year = ? AND action IN ('promoted', 'retained', 'graduated')
                            ORDER BY promotion_date DESC
                            LIMIT 10
                        """, (current_academic_year,))
                        
                        if recent_promotions:
                            st.markdown("#### Recent Promotion Activities")
                            recent_df = pd.DataFrame(recent_promotions, columns=[
                                'Student', 'From Grade', 'To Grade', 'Action', 'Date'
                            ])
                            
                            # Color code the actions
                            def highlight_recent_action(val):
                                if val == 'promoted':
                                    return 'background-color: #d4edda; color: #155724'
                                elif val == 'graduated':
                                    return 'background-color: #d1ecf1; color: #0c5460'
                                elif val == 'retained':
                                    return 'background-color: #f8d7da; color: #721c24'
                                return ''
                            
                            styled_recent = recent_df.style.map(highlight_recent_action, subset=['Action'])
                            st.dataframe(styled_recent, width="stretch", hide_index=True)
                
                else:
                    st.info("No historical promotion data available yet. Data will be populated after running promotions.")

            except Exception as e:
                st.error(f"Error loading historical data: {str(e)}")

def finance_page():
    st.header("💰 Financial Management")
    
    # Initialize database manager
    db = DatabaseManager()
    
    # Get currency setting from session state
    currency_options = {
        "₵ (Ghana Cedi)": "₵",
        "$ (US Dollar)": "$",
        "€ (Euro)": "€",
        "£ (British Pound)": "£",
        "₦ (Nigerian Naira)": "₦"
    }
    
    # Use currency from session state if available, otherwise default to Ghana Cedi
    if 'selected_currency_name' in st.session_state:
        selected_currency_name = st.session_state.selected_currency_name
        currency_symbol = currency_options.get(selected_currency_name, "₵")
    else:
        selected_currency_name = "₵ (Ghana Cedi)"
        currency_symbol = "₵"
    
    # Create main tabs for the finance page
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Financial Overview", "➕ Add Transaction", "💳 Fee Management"])
    
    with main_tab1:
        st.subheader("📊 Financial Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Financial Metrics")
            
            try:
                # Calculate total revenue
                revenue_query = """
                    SELECT SUM(amount) FROM finance 
                    WHERE type IN ('tuition', 'fees', 'grants', 'donations') 
                    AND status = 'completed'
                """
                revenue_result = db.execute_query(revenue_query)
                total_revenue = revenue_result[0][0] if revenue_result[0][0] else 0
                
                # Calculate total expenses
                expense_query = """
                    SELECT SUM(amount) FROM finance 
                    WHERE type IN ('salary', 'utilities', 'supplies', 'maintenance', 'other_expenses')
                    AND status = 'completed'
                """
                expense_result = db.execute_query(expense_query)
                total_expenses = expense_result[0][0] if expense_result[0][0] else 0
                
                net_profit = total_revenue - total_expenses
                profit_margin = (net_profit/total_revenue*100) if total_revenue > 0 else 0
                
                st.metric("Total Revenue", f"{currency_symbol}{total_revenue:,.2f}")
                st.metric("Total Expenses", f"{currency_symbol}{total_expenses:,.2f}")
                st.metric("Net Profit", f"{currency_symbol}{net_profit:,.2f}", delta=f"{profit_margin:.1f}%")
                
                # Revenue breakdown
                revenue_breakdown_query = """
                    SELECT 
                        CASE 
                            WHEN type = 'tuition' THEN 'Tuition'
                            WHEN type = 'grants' THEN 'Grants'
                            WHEN type = 'donations' THEN 'Donations'
                            WHEN type = 'fees' THEN 'Fees'
                            ELSE 'Other'
                        END as source,
                        SUM(amount) as total_amount
                    FROM finance 
                    WHERE type IN ('tuition', 'fees', 'grants', 'donations') 
                    AND status = 'completed'
                    GROUP BY source
                    HAVING total_amount > 0
                """
                
                revenue_data = db.execute_query(revenue_breakdown_query)
                
                if revenue_data:
                    sources = [row[0] for row in revenue_data]
                    amounts = [row[1] for row in revenue_data]
                    
                    fig = px.pie(values=amounts, names=sources, title="Revenue Sources")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No revenue data available yet.")
                    
            except Exception as e:
                st.error(f"Error loading financial data: {str(e)}")
                total_revenue = 0
                total_expenses = 0
                net_profit = 0
        
        with col2:
            st.subheader("🤖 AI Financial Insights")
            
            if st.button("Generate Financial Report"):
                ai = SchoolAI()
                
                try:
                    # Pending payments
                    pending_query = """
                        SELECT COUNT(*), COALESCE(SUM(amount), 0) 
                        FROM finance 
                        WHERE status = 'pending' AND type IN ('tuition', 'fees')
                    """
                    pending_result = db.execute_query(pending_query)
                    pending_count, pending_amount = pending_result[0] if pending_result else (0, 0)
                    
                    # Monthly revenue trend
                    monthly_query = """
                        SELECT 
                            strftime('%Y-%m', date) as month,
                            SUM(amount) as monthly_revenue
                        FROM finance 
                        WHERE type IN ('tuition', 'fees', 'grants', 'donations') 
                        AND status = 'completed'
                        AND date >= date('now', '-6 months')
                        GROUP BY month
                        ORDER BY month
                    """
                    monthly_data = db.execute_query(monthly_query)
                    
                    financial_prompt = f"""
                    Analyze the school's financial data (Currency: {selected_currency_name}):
                    - Total Revenue: {currency_symbol}{total_revenue:,.2f}
                    - Total Expenses: {currency_symbol}{total_expenses:,.2f}
                    - Net Profit: {currency_symbol}{net_profit:,.2f}
                    - Net Profit Margin: {profit_margin:.1f}%
                    - Pending Payments: {pending_count} transactions worth {currency_symbol}{pending_amount:,.2f}
                    - Recent Revenue Trend: {len(monthly_data)} months of data available
                    
                    Provide insights on:
                    1. Financial health assessment
                    2. Cost optimization opportunities
                    3. Revenue enhancement strategies
                    4. Cash flow management recommendations
                    5. Collection efficiency improvements
                    """
                    
                    with st.spinner("AI analyzing financial data..."):
                        insights = ai.generate_response(financial_prompt)
                        st.write(insights)
                        
                except Exception as e:
                    st.error(f"Error generating AI insights: {str(e)}")
        
        # Financial Analytics Section
        st.subheader("📈 Financial Analytics")
        
        try:
            # Monthly revenue trend
            monthly_trend_query = """
                SELECT 
                    strftime('%Y-%m', date) as month,
                    SUM(CASE WHEN type IN ('tuition', 'fees', 'grants', 'donations') THEN amount ELSE 0 END) as revenue,
                    SUM(CASE WHEN type IN ('salary', 'utilities', 'supplies', 'maintenance', 'other_expenses') THEN amount ELSE 0 END) as expenses
                FROM finance 
                WHERE status = 'completed'
                AND date >= date('now', '-12 months')
                GROUP BY month
                ORDER BY month
            """
            monthly_data = db.execute_query(monthly_trend_query)
            
            if monthly_data:
                months = [row[0] for row in monthly_data]
                revenues = [row[1] for row in monthly_data]
                expenses = [row[2] for row in monthly_data]
                
                # Create DataFrame for plotting
                trend_df = pd.DataFrame({
                    'Month': months,
                    'Revenue': revenues,
                    'Expenses': expenses,
                    'Profit': [r - e for r, e in zip(revenues, expenses)]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Revenue trend chart
                    fig_revenue = px.line(trend_df, x='Month', y='Revenue', title=f"Monthly Revenue Trend ({currency_symbol})")
                    st.plotly_chart(fig_revenue, width="stretch")
                
                with col2:
                    # Profit/Loss chart
                    fig_profit = px.line(trend_df, x='Month', y='Profit', title=f"Monthly Profit/Loss Trend ({currency_symbol})")
                    st.plotly_chart(fig_profit, width="stretch")
                
                # Revenue vs Expenses comparison
                fig_comparison = px.bar(trend_df, x='Month', y=['Revenue', 'Expenses'], 
                                      title=f"Revenue vs Expenses Comparison ({currency_symbol})", barmode='group')
                st.plotly_chart(fig_comparison, width="stretch")
                
            else:
                st.info("No financial trend data available yet.")
                
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")

    with main_tab2:
        st.subheader("➕ Add Financial Transaction")
        
        st.info("💡 Use this section to record all financial transactions including tuition payments, fees, expenses, and other income.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Transaction Details")
            transaction_type = st.selectbox("Transaction Type", 
                                          ['tuition', 'fees', 'grants', 'donations', 'salary', 'utilities', 'supplies', 'maintenance', 'other_expenses'])
            amount = st.number_input(f"Amount ({currency_symbol})", min_value=0.0, step=0.01, format="%.2f")
            description = st.text_input("Description", placeholder="Enter transaction description...")
            
        with col2:
            st.subheader("🎯 Additional Information")
            # Get student list for student-related transactions
            if transaction_type in ['tuition', 'fees']:
                students_query = "SELECT student_id, name FROM students WHERE status = 'active' ORDER BY name"
                students = db.execute_query(students_query)
                student_options = [''] + [f"{row[1]} ({row[0]})" for row in students]
                selected_student = st.selectbox("Student (if applicable)", student_options)
            else:
                selected_student = ''
                st.info("💼 Business expense - no student selection needed")
                
            status = st.selectbox("Status", ['pending', 'completed', 'overdue'])
            transaction_date = st.date_input("Date", value=datetime.now().date())
        
        # Display amount preview
        if amount > 0:
            st.success(f"💰 Transaction Preview: **{currency_symbol}{amount:,.2f}** - {transaction_type.title()}")
            
            if transaction_type in ['tuition', 'fees']:
                st.info("📚 This is a revenue transaction (increases school income)")
            elif transaction_type in ['grants', 'donations']:
                st.info("🎁 This is an income transaction (external funding)")
            else:
                st.warning("💸 This is an expense transaction (reduces school funds)")
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("💾 Save Transaction", type="primary", width="stretch"):
                if amount <= 0:
                    st.error("⚠️ Please enter a valid amount greater than 0")
                elif not description.strip():
                    st.error("⚠️ Please enter a description for the transaction")
                else:
                    try:
                        student_id = selected_student.split('(')[1].rstrip(')') if selected_student and '(' in selected_student else None
                        transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        description_with_currency = f"{description} ({currency_symbol})" if description else f"Transaction ({currency_symbol})"
                        
                        insert_query = """
                            INSERT INTO finance (transaction_id, student_id, amount, type, description, date, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """
                        
                        db.execute_update(insert_query, (
                            transaction_id, student_id, amount, transaction_type, 
                            description_with_currency, transaction_date.isoformat(), status
                        ))
                        
                        st.success(f"✅ Transaction recorded successfully!")
                        st.success(f"💰 **{currency_symbol}{amount:,.2f}** - {description}")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error saving transaction: {str(e)}")
        
        with col2:
            if st.button("🔄 Clear Form", width="stretch"):
                st.rerun()
        
        with col3:
            if st.button("📊 View Overview", width="stretch"):
                st.info("💡 Switch to the 'Financial Overview' tab to see updated metrics and charts!")
        
        # Recent transactions preview
        st.subheader("📋 Recent Transactions")
        
        try:
            recent_query = """
                SELECT transaction_id, type, amount, description, date, status
                FROM finance 
                ORDER BY date DESC, rowid DESC
                LIMIT 5
            """
            recent_transactions = db.execute_query(recent_query)
            
            if recent_transactions:
                recent_df = pd.DataFrame(recent_transactions, 
                                       columns=['ID', 'Type', 'Amount', 'Description', 'Date', 'Status'])
                
                recent_df['Amount'] = recent_df['Amount'].apply(
                    lambda x: f"{currency_symbol}{x:,.2f}" if pd.notnull(x) else f"{currency_symbol}0.00"
                )
                
                recent_df['Type'] = recent_df['Type'].str.title()
                recent_df['Status'] = recent_df['Status'].str.title()
                
                st.dataframe(recent_df, width="stretch", hide_index=True)
            else:
                st.info("📝 No transactions recorded yet. Use the form above to add your first transaction!")
                
        except Exception as e:
            st.error(f"Error loading recent transactions: {str(e)}")

    with main_tab3:
        st.subheader("💳 Fee Management")
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📋 Fee Structure", "💰 Collections", "📊 Payment Analytics"])
        
        with sub_tab1:
            st.subheader("Current Fee Structure")
            
            try:
                # Get unique grade levels from students
                grade_query = "SELECT DISTINCT grade FROM students WHERE status = 'active' ORDER BY grade"
                grades = db.execute_query(grade_query)
                grade_levels = [grade[0] for grade in grades] if grades else ['Kindergarten', 'Grade 1', 'Grade 2', 'Grade 3']
                
                # Create fee structure
                fee_data = []
                for grade in grade_levels:
                    fee_query = """
                        SELECT amount FROM finance 
                        WHERE description LIKE ? AND type = 'tuition'
                        ORDER BY date DESC LIMIT 1
                    """
                    fee_result = db.execute_query(fee_query, (f"%{grade}%",))
                    base_fee = fee_result[0][0] if fee_result else 10000
                    
                    fee_data.append({
                        'Grade Level': grade,
                        f'Tuition ({currency_symbol})': base_fee,
                        f'Activity Fee ({currency_symbol})': base_fee * 0.05,
                        f'Technology Fee ({currency_symbol})': base_fee * 0.03,
                        f'Total ({currency_symbol})': base_fee * 1.08
                    })
                
                fee_df = pd.DataFrame(fee_data)
                
                # Format currency columns
                currency_columns = [col for col in fee_df.columns if currency_symbol in col]
                for col in currency_columns:
                    fee_df[col] = fee_df[col].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                
                st.dataframe(fee_df, width="stretch")
                
                if st.button("🤖 AI Optimize Fee Structure"):
                    with st.spinner("AI analyzing market rates and costs..."):
                        student_count_query = """
                            SELECT grade, COUNT(*) as student_count
                            FROM students 
                            WHERE status = 'active'
                            GROUP BY grade
                        """
                        student_counts = db.execute_query(student_count_query)
                        
                        ai_prompt = f"""
                        Analyze our current fee structure and student distribution (Currency: {selected_currency_name}):
                        Grade Distribution: {dict(student_counts) if student_counts else 'No data'}
                        Current Revenue: {currency_symbol}{total_revenue:,.2f}
                        
                        Provide recommendations for:
                        1. Competitive fee pricing for each grade level
                        2. Optimal fee structure to maximize revenue while remaining affordable
                        3. Additional fee categories that could be introduced
                        4. Discount strategies for families with multiple children
                        """
                        
                        ai = SchoolAI()
                        recommendations = ai.generate_response(ai_prompt)
                        st.success("✅ Fee structure analysis complete!")
                        st.write(recommendations)
                        
            except Exception as e:
                st.error(f"Error loading fee structure: {str(e)}")
        
        with sub_tab2:
            st.subheader("Collection Status")
            
            try:
                collection_query = """
                    SELECT 
                        status,
                        COUNT(*) as count,
                        COALESCE(SUM(amount), 0) as total_amount
                    FROM finance 
                    WHERE type IN ('tuition', 'fees')
                    GROUP BY status
                """
                collection_data = db.execute_query(collection_query)
                
                if collection_data:
                    statuses = [row[0] for row in collection_data]
                    counts = [row[1] for row in collection_data]
                    amounts = [row[2] for row in collection_data]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:  
                        fig = px.bar(x=statuses, y=counts, title="Payment Status Count")
                        st.plotly_chart(fig, width="stretch")
                    
                    with col2:
                        fig = px.bar(x=statuses, y=amounts, title=f"Payment Status Amount ({currency_symbol})")
                        st.plotly_chart(fig, width="stretch")
                    
                    # Show overdue payments
                    overdue_query = """
                        SELECT s.name, f.amount, f.description, f.date
                        FROM finance f
                        JOIN students s ON f.student_id = s.student_id
                        WHERE f.status = 'overdue' OR (f.status = 'pending' AND f.date < date('now', '-30 days'))
                        ORDER BY f.date
                    """
                    overdue_payments = db.execute_query(overdue_query)
                    
                    if overdue_payments:
                        st.subheader("⚠️ Overdue Payments")
                        overdue_df = pd.DataFrame(overdue_payments, columns=['Student', 'Amount', 'Description', 'Due Date'])
                        
                        overdue_df['Amount'] = overdue_df['Amount'].apply(
                            lambda x: f"{currency_symbol}{x:,.2f}" if pd.notnull(x) else f"{currency_symbol}0.00"
                        )
                        
                        try:
                            overdue_df['Due Date'] = pd.to_datetime(overdue_df['Due Date']).dt.date
                        except:
                            pass
                        st.dataframe(overdue_df, width="stretch")
                    
                else:
                    st.info("No payment data available yet.")
                    
                if st.button("🤖 AI Send Payment Reminders"):
                    try:
                        pending_query = """
                            SELECT DISTINCT s.name, s.parent_contact, f.amount, f.description
                            FROM finance f
                            JOIN students s ON f.student_id = s.student_id
                            WHERE f.status IN ('pending', 'overdue')
                        """
                        pending_students = db.execute_query(pending_query)
                        
                        with st.spinner("AI generating personalized payment reminders..."):
                            ai = SchoolAI()
                            reminder_prompt = f"""
                            Generate personalized payment reminder messages for {len(pending_students)} students.
                            Use currency: {selected_currency_name} ({currency_symbol})
                            Create templates that are:
                            1. Professional but friendly
                            2. Include payment details
                            3. Offer payment plan options
                            4. Maintain positive school-parent relationship
                            """
                            
                            reminders = ai.generate_response(reminder_prompt)
                            st.success(f"✅ {len(pending_students)} personalized payment reminders generated!")
                            st.write(reminders)
                            
                    except Exception as e:
                        st.error(f"Error generating reminders: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error loading collection data: {str(e)}")
        
        with sub_tab3:
            st.subheader("Payment Analytics by Grade")
            
            try:
                student_payment_query = """
                    SELECT 
                        s.grade,
                        COUNT(DISTINCT s.student_id) as total_students,
                        COUNT(DISTINCT CASE WHEN f.status = 'completed' THEN f.student_id END) as paid_students,
                        COALESCE(AVG(f.amount), 0) as avg_payment
                    FROM students s
                    LEFT JOIN finance f ON s.student_id = f.student_id AND f.type IN ('tuition', 'fees')
                    WHERE s.status = 'active'
                    GROUP BY s.grade
                    ORDER BY s.grade
                """
                payment_analytics = db.execute_query(student_payment_query)
                
                if payment_analytics:
                    analytics_df = pd.DataFrame(payment_analytics, 
                                              columns=['Grade', 'Total Students', 'Paid Students', f'Avg Payment ({currency_symbol})'])
                    analytics_df['Payment Rate %'] = (analytics_df['Paid Students'] / analytics_df['Total Students'] * 100).round(2)
                    
                    analytics_df[f'Avg Payment ({currency_symbol})'] = analytics_df[f'Avg Payment ({currency_symbol})'].apply(
                        lambda x: f"{currency_symbol}{x:,.2f}"
                    )
                    
                    st.dataframe(analytics_df, width="stretch")
                    
                    # Payment rate visualization
                    fig_payment_rate = px.bar(analytics_df, x='Grade', y='Payment Rate %', 
                                            title="Payment Rate by Grade Level")
                    st.plotly_chart(fig_payment_rate, width="stretch")
                    
                else:
                    st.info("No payment analytics data available yet.")
                    
            except Exception as e:
                st.error(f"Error loading payment analytics: {str(e)}")
    
def analytics_page():
    """Advanced Analytics Dashboard for School Management System"""
    st.header("📈 Advanced Analytics")
    
    db_manager = DatabaseManager()
    
    # Get all students from database
    students_query = """
        SELECT student_id, name, age, grade, gpa, behavior_score, status, stream_id
        FROM students 
        WHERE status = 'active'
    """
    
    try:
        students_data = db_manager.execute_query(students_query)
    except Exception as e:
        st.error("Unable to load student data. Please check database connection.")
        return
    
    if not students_data:
        st.warning("No student data available for analytics.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(students_data, columns=[
        'student_id', 'name', 'age', 'grade', 'gpa', 'behavior_score', 'status', 'stream_id'
    ])
    
    # Calculate attendance rates for each student
    attendance_rates = {}
    for student_id in df['student_id']:
        try:
            total_query = "SELECT COUNT(*) FROM attendance WHERE student_id = ?"
            total_days = db_manager.execute_query(total_query, (student_id,))[0][0]
            
            if total_days > 0:
                present_query = """
                    SELECT COUNT(*) FROM attendance 
                    WHERE student_id = ? AND status = 'present'
                """
                present_days = db_manager.execute_query(present_query, (student_id,))[0][0]
                attendance_rates[student_id] = (present_days / total_days) * 100
            else:
                attendance_rates[student_id] = 100.0
        except Exception:
            attendance_rates[student_id] = 0.0
    
    df['attendance_rate'] = df['student_id'].map(attendance_rates)
    
    # Create analytics tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Overview KPIs", 
        "📊 Student Analytics", 
        "🏫 Stream Analysis", 
        "💰 Financial Analytics", 
        "👨‍🏫 Teacher Analytics", 
        "🤖 AI Predictions", 
        "📈 Trends & Stats"
    ])
    
    with tab1:
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_gpa = df['gpa'].mean()
            st.metric("Average GPA", f"{avg_gpa:.2f}")
        
        with col2:
            avg_attendance = df['attendance_rate'].mean()
            st.metric("Average Attendance", f"{avg_attendance:.1f}%")
        
        with col3:
            high_performers = len(df[df['gpa'] >= 3.5])
            st.metric("High Performers (GPA ≥ 3.5)", high_performers)
        
        with col4:
            at_risk = len(df[(df['gpa'] < 2.0) | (df['attendance_rate'] < 75)])
            st.metric("At-Risk Students", at_risk)
        
        st.divider()
        
        st.subheader("📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            with col1:
                total_students = db_manager.execute_query("SELECT COUNT(*) FROM students WHERE status = 'active'")[0][0]
                st.metric("Active Students", total_students)
            
            with col2:
                total_teachers = db_manager.execute_query("SELECT COUNT(*) FROM teachers WHERE status = 'active'")[0][0]
                st.metric("Active Teachers", total_teachers)
            
            with col3:
                total_courses = db_manager.execute_query("SELECT COUNT(*) FROM courses")[0][0]
                st.metric("Total Courses", total_courses)
            
            with col4:
                total_streams = db_manager.execute_query("SELECT COUNT(*) FROM streams")[0][0]
                st.metric("Total Streams", total_streams)
        except Exception:
            st.error("Unable to load system metrics.")
    
    with tab2:
        st.subheader("📊 Student Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            grade_counts = df['grade'].value_counts()
            fig = px.bar(x=grade_counts.index, y=grade_counts.values, 
                        title="Students by Grade Level",
                        labels={'x': 'Grade Level', 'y': 'Number of Students'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            fig = px.histogram(df, x='gpa', nbins=20, title="GPA Distribution",
                             labels={'x': 'GPA', 'y': 'Number of Students'})
            st.plotly_chart(fig, width="stretch")
        
        st.divider()
        
        st.subheader("🔗 Performance Correlation Analysis")
        
        numeric_cols = ['age', 'gpa', 'attendance_rate', 'behavior_score']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Performance Metrics Correlation Matrix",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, width="stretch")
        
        st.subheader("⚠️ At-Risk Students")
        
        at_risk_students = df[
            (df['gpa'] < 2.5) | 
            (df['attendance_rate'] < 80) | 
            (df['behavior_score'] < 70)
        ]
        
        if len(at_risk_students) > 0:
            st.warning(f"**{len(at_risk_students)} students identified as at-risk:**")
            
            for _, student in at_risk_students.iterrows():
                risk_factors = []
                if student['gpa'] < 2.5:
                    risk_factors.append(f"Low GPA ({student['gpa']:.2f})")
                if student['attendance_rate'] < 80:
                    risk_factors.append(f"Poor Attendance ({student['attendance_rate']:.1f}%)")
                if student['behavior_score'] < 70:
                    risk_factors.append(f"Behavior Issues ({student['behavior_score']:.0f})")
                
                st.write(f"⚠️ **{student['name']}** - {', '.join(risk_factors)}")
        else:
            st.success("No students currently identified as at-risk!")
    
    with tab3:
        st.subheader("🏫 Stream Performance Analysis")
        
        streams_query = """
            SELECT s.stream_id, s.grade_level, s.stream_type, s.max_capacity,
                   COUNT(st.student_id) as current_students,
                   AVG(st.gpa) as avg_gpa,
                   t.name as class_teacher
            FROM streams s
            LEFT JOIN students st ON s.stream_id = st.stream_id AND st.status = 'active'
            LEFT JOIN teachers t ON s.class_teacher_id = t.teacher_id
            GROUP BY s.stream_id, s.grade_level, s.stream_type, s.max_capacity, t.name
        """
        
        try:
            streams_data = db_manager.execute_query(streams_query)
        except Exception:
            st.error("Unable to load stream data.")
            streams_data = []
        
        if streams_data:
            streams_df = pd.DataFrame(streams_data, columns=[
                'stream_id', 'grade_level', 'stream_type', 'max_capacity', 
                'current_students', 'avg_gpa', 'class_teacher'
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(streams_df, x='stream_id', y='current_students',
                            title="Students per Stream", color='grade_level',
                            labels={'x': 'Stream ID', 'y': 'Number of Students'})
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = px.bar(streams_df, x='stream_id', y='avg_gpa',
                            title="Average GPA by Stream", color='stream_type',
                            labels={'x': 'Stream ID', 'y': 'Average GPA'})
                st.plotly_chart(fig, width="stretch")
            
            st.divider()
            
            streams_df['capacity_utilization'] = (streams_df['current_students'] / streams_df['max_capacity']) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(streams_df, x='stream_id', y='capacity_utilization',
                            title="Stream Capacity Utilization (%)", color='stream_type',
                            labels={'x': 'Stream ID', 'y': 'Capacity Utilization (%)'})
                fig.add_hline(y=100, line_dash="dash", line_color="red", 
                             annotation_text="Full Capacity")
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                st.subheader("Stream Details")
                display_df = streams_df[['stream_id', 'grade_level', 'stream_type', 
                                       'current_students', 'max_capacity', 'class_teacher']].copy()
                display_df.columns = ['Stream ID', 'Grade', 'Type', 'Current', 'Max Capacity', 'Teacher']
                st.dataframe(display_df, width="stretch", hide_index=True)
        else:
            st.info("No stream data available.")
    
    with tab4:
        st.subheader("💰 Financial Analytics")
        
        finance_query = """
            SELECT type, SUM(amount) as total_amount, COUNT(*) as transaction_count
            FROM finance
            GROUP BY type
        """
        
        try:
            finance_data = db_manager.execute_query(finance_query)
        except Exception:
            st.error("Unable to load financial data.")
            finance_data = []
        
        if finance_data:
            finance_df = pd.DataFrame(finance_data, columns=['type', 'total_amount', 'transaction_count'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(finance_df, values='total_amount', names='type',
                            title="Revenue by Transaction Type")
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = px.bar(finance_df, x='type', y='transaction_count',
                            title="Transaction Count by Type",
                            labels={'x': 'Transaction Type', 'y': 'Count'})
                st.plotly_chart(fig, width="stretch")
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            total_revenue = finance_df['total_amount'].sum()
            total_transactions = finance_df['transaction_count'].sum()
            avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
            
            with col1:
                st.metric("Total Revenue", f"${total_revenue:,.2f}")
            
            with col2:
                st.metric("Total Transactions", total_transactions)
            
            with col3:
                st.metric("Avg Transaction Value", f"${avg_transaction:.2f}")
            
            st.subheader("💳 Financial Breakdown")
            display_df = finance_df.copy()
            display_df.columns = ['Transaction Type', 'Total Amount ($)', 'Transaction Count']
            display_df['Total Amount ($)'] = display_df['Total Amount ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, width="stretch", hide_index=True)
        else:
            st.info("No financial data available.")
    
    with tab5:
        st.subheader("👨‍🏫 Teacher Performance Analytics")
        
        teachers_query = """
            SELECT name, subject, department, performance_score, salary
            FROM teachers
            WHERE status = 'active'
        """
        
        try:
            teachers_data = db_manager.execute_query(teachers_query)
        except Exception:
            st.error("Unable to load teacher data.")
            teachers_data = []
        
        if teachers_data:
            teachers_df = pd.DataFrame(teachers_data, columns=[
                'name', 'subject', 'department', 'performance_score', 'salary'
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                dept_performance = teachers_df.groupby('department')['performance_score'].mean().reset_index()
                fig = px.bar(dept_performance, x='department', y='performance_score',
                            title="Average Performance by Department",
                            labels={'x': 'Department', 'y': 'Performance Score'})
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = px.scatter(teachers_df, x='performance_score', y='salary',
                               color='department', title="Performance vs Salary",
                               labels={'x': 'Performance Score', 'y': 'Salary ($)'})
                st.plotly_chart(fig, width="stretch")
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_performance = teachers_df['performance_score'].mean()
                st.metric("Average Performance Score", f"{avg_performance:.1f}")
            
            with col2:
                top_performers = len(teachers_df[teachers_df['performance_score'] >= 90])
                st.metric("Top Performers (≥90)", top_performers)
            
            with col3:
                avg_salary = teachers_df['salary'].mean()
                st.metric("Average Salary", f"${avg_salary:,.2f}")
            
            st.subheader("📚 Department Analysis")
            dept_summary = teachers_df.groupby('department').agg({
                'performance_score': 'mean',
                'salary': 'mean',
                'name': 'count'
            }).round(2)
            dept_summary.columns = ['Avg Performance', 'Avg Salary ($)', 'Teacher Count']
            dept_summary['Avg Salary ($)'] = dept_summary['Avg Salary ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(dept_summary, width="stretch")
        else:
            st.info("No teacher data available.")
    
    with tab6:
        st.subheader("🤖 AI Predictive Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Student Performance Prediction**")
            st.write("Analyze student data patterns to predict academic outcomes and identify intervention needs.")
            
            if st.button("🤖 Generate Performance Predictions", key="predict_performance"):
                try:
                    ai = SchoolAI()
                    
                    numeric_cols = ['age', 'gpa', 'attendance_rate', 'behavior_score']
                    summary_stats = df[numeric_cols].describe()
                    
                    prediction_prompt = f"""
                    Based on student performance data, provide academic predictions and recommendations:
                    
                    Data Summary:
                    - Total Students: {len(df)}
                    - Average GPA: {df['gpa'].mean():.2f}
                    - Average Attendance: {df['attendance_rate'].mean():.1f}%
                    - High Performers (GPA ≥ 3.5): {len(df[df['gpa'] >= 3.5])}
                    - At-Risk Students: {len(df[(df['gpa'] < 2.0) | (df['attendance_rate'] < 75)])}
                    
                    Provide concise predictions for:
                    1. Students needing academic intervention
                    2. Expected graduation rates
                    3. Performance trends for next semester
                    4. Recommended interventions
                    """
                    
                    with st.spinner("Analyzing performance patterns..."):
                        predictions = ai.generate_response(prediction_prompt)
                        st.success("**AI Predictions Generated:**")
                        st.write(predictions)
                except Exception:
                    st.error("AI prediction service temporarily unavailable.")
        
        with col2:
            st.info("**Risk Assessment Analysis**")
            st.write("Comprehensive risk analysis for at-risk students with intervention strategies.")
            
            if st.button("🎯 Generate Risk Assessment", key="risk_assessment"):
                try:
                    ai = SchoolAI()
                    
                    at_risk_students = df[
                        (df['gpa'] < 2.5) | 
                        (df['attendance_rate'] < 80) | 
                        (df['behavior_score'] < 70)
                    ]
                    
                    risk_prompt = f"""
                    Analyze at-risk student data and provide intervention recommendations:
                    
                    Risk Analysis:
                    - Total Students: {len(df)}
                    - At-Risk Students: {len(at_risk_students)}
                    - Low GPA (<2.5): {len(df[df['gpa'] < 2.5])} students
                    - Poor Attendance (<80%): {len(df[df['attendance_rate'] < 80])} students
                    - Behavior Issues (<70): {len(df[df['behavior_score'] < 70])} students
                    
                    Provide:
                    1. Risk assessment summary
                    2. Intervention strategies
                    3. Resource allocation recommendations
                    4. Implementation timeline
                    """
                    
                    with st.spinner("Generating risk assessment..."):
                        risk_analysis = ai.generate_response(risk_prompt)
                        st.success("**Risk Assessment Complete:**")
                        st.write(risk_analysis)
                except Exception:
                    st.error("AI analysis service temporarily unavailable.")
        
        st.divider()
        
        st.subheader("💡 Quick AI Insights")
        
        if st.button("📈 Generate Performance Insights", key="performance_insights"):
            try:
                ai = SchoolAI()
                
                insights_prompt = f"""
                Provide 3-5 key actionable insights about student performance:
                - Average GPA: {df['gpa'].mean():.2f}
                - Average Attendance: {df['attendance_rate'].mean():.1f}%
                - High Performers: {len(df[df['gpa'] >= 3.5])} students
                - At-Risk Students: {len(df[(df['gpa'] < 2.0) | (df['attendance_rate'] < 75)])} students
                
                Keep insights concise and actionable for school administrators.
                """
                
                with st.spinner("Generating insights..."):
                    insights = ai.generate_response(insights_prompt)
                    st.info(f"**AI Insights:** {insights}")
            except Exception:
                st.error("AI insights service temporarily unavailable.")
    
    with tab7:
        st.subheader("📈 Attendance Trends & Statistics")
        
        recent_attendance_query = """
            SELECT date, 
                   COUNT(*) as total_students,
                   SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present_students
            FROM attendance
            WHERE date >= date('now', '-30 days')
            GROUP BY date
            ORDER BY date
        """
        
        try:
            attendance_trend_data = db_manager.execute_query(recent_attendance_query)
        except Exception:
            st.error("Unable to load attendance trend data.")
            attendance_trend_data = []
        
        if attendance_trend_data:
            trend_df = pd.DataFrame(attendance_trend_data, columns=['date', 'total_students', 'present_students'])
            trend_df['attendance_rate'] = (trend_df['present_students'] / trend_df['total_students']) * 100
            trend_df['date'] = pd.to_datetime(trend_df['date'])
            
            fig = px.line(trend_df, x='date', y='attendance_rate',
                         title="Daily Attendance Rate (Last 30 Days)",
                         labels={'x': 'Date', 'y': 'Attendance Rate (%)'})
            fig.add_hline(y=trend_df['attendance_rate'].mean(), line_dash="dash", 
                         line_color="green", annotation_text="Average Rate")
            st.plotly_chart(fig, width="stretch")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_daily_attendance = trend_df['attendance_rate'].mean()
                st.metric("Average Daily Attendance", f"{avg_daily_attendance:.1f}%")
            
            with col2:
                best_day = trend_df.loc[trend_df['attendance_rate'].idxmax()]
                st.metric("Best Attendance Day", f"{best_day['attendance_rate']:.1f}%")
            
            with col3:
                worst_day = trend_df.loc[trend_df['attendance_rate'].idxmin()]
                st.metric("Lowest Attendance Day", f"{worst_day['attendance_rate']:.1f}%")
        else:
            st.info("No recent attendance data available for trend analysis.")
        
        st.divider()
        
        st.subheader("📊 Performance Trends")
        
        fig = px.box(df, x='grade', y='gpa', title="GPA Distribution by Grade Level",
                     labels={'x': 'Grade Level', 'y': 'GPA'})
        st.plotly_chart(fig, width="stretch")
        
        fig = px.scatter(df, x='attendance_rate', y='gpa', 
                        color='grade', size='behavior_score',
                        title="Attendance Rate vs GPA by Grade",
                        labels={'x': 'Attendance Rate (%)', 'y': 'GPA'},
                        hover_data=['name'])
        st.plotly_chart(fig, width="stretch")

def ai_insights_page():
    st.header("🤖 AI-Powered Insights")
    
    manager = st.session_state.school_manager
    db = manager.db
    
    # Initialize AI Helper Functions
    ai_helper = AIHelperFunctions(db)
    
    # AI Insights Dashboard
    st.subheader("🧠 Comprehensive School Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Complete AI Analysis", type="primary"):
            with st.spinner("AI is analyzing all school data..."):
                ai_instance = SchoolAI()
                
                if 'ai_provider' in st.session_state:
                    ai_instance.provider = st.session_state.ai_provider
                if 'ai_model' in st.session_state:
                    ai_instance.model = st.session_state.ai_model
                if 'ollama_url' in st.session_state:
                    ai_instance.ollama_url = st.session_state.ollama_url
                
                school_data = ai_helper.get_data_by_type('school')
                
                analysis_prompt = f"""
                Based on the following real school data, provide a comprehensive analysis:
                
                {school_data}
                
                Please analyze:
                1. Overall performance assessment
                2. Key strengths and weaknesses
                3. Priority areas for improvement
                4. Strategic recommendations
                5. Implementation roadmap
                """
                
                try:
                    insights = ai_instance.generate_response(analysis_prompt)
                    
                    if "Error:" in insights:
                        st.error(f"{insights}")
                    else:
                        st.write("### 📊 AI Analysis Results")
                        st.write(insights)
                        
                        st.session_state.last_ai_analysis = insights
                        st.success("Complete AI analysis generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating AI analysis: {str(e)}")
    
    with col2:
        st.subheader("🎯 Quick AI Actions")
        
        if st.button("📝 Generate Report"):
            with st.spinner("Generating comprehensive report..."):
                ai_instance = SchoolAI()
                
                if 'ai_provider' in st.session_state:
                    ai_instance.provider = st.session_state.ai_provider
                if 'ai_model' in st.session_state:
                    ai_instance.model = st.session_state.ai_model
                if 'ollama_url' in st.session_state:
                    ai_instance.ollama_url = st.session_state.ollama_url
                
                report_data = ai_helper.get_data_by_type('report')
                
                report_prompt = f"""
                Generate a detailed school performance report based on this data:
                
                {report_data}
                
                Include:
                1. Executive summary
                2. Key performance indicators
                3. Student achievement metrics
                4. Financial overview
                5. Recommendations for improvement
                """
                
                try:
                    report = ai_instance.generate_response(report_prompt)
                    
                    if "Error:" in report:
                        st.error(f"{report}")
                    else:
                        st.success("Comprehensive report generated!")
                        
                        with st.expander("📄 View Generated Report"):
                            st.write(report)
                        
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        if st.button("🔍 Identify Trends"):
            with st.spinner("Analyzing trends..."):
                ai_instance = SchoolAI()
                
                if 'ai_provider' in st.session_state:
                    ai_instance.provider = st.session_state.ai_provider
                if 'ai_model' in st.session_state:
                    ai_instance.model = st.session_state.ai_model
                if 'ollama_url' in st.session_state:
                    ai_instance.ollama_url = st.session_state.ollama_url
                
                trends_data = ai_helper.get_data_by_type('trends')
                
                trends_prompt = f"""
                Analyze trends based on this historical school data:
                
                {trends_data}
                
                Identify:
                1. Academic performance trends over time
                2. Enrollment patterns and demographics
                3. Financial trends and patterns
                4. Staffing and resource utilization trends
                5. Emerging challenges and opportunities
                """
                
                try:
                    trends = ai_instance.generate_response(trends_prompt)
                    
                    if "Error:" in trends:
                        st.error(f"{trends}")
                    else:
                        st.info("Trend analysis completed!")
                        
                        with st.expander("📊 View Trend Analysis"):
                            st.write(trends)
                        
                except Exception as e:
                    st.error(f"Error analyzing trends: {str(e)}")
        
        if st.button("⚠️ Risk Assessment"):
            with st.spinner("Conducting risk assessment..."):
                ai_instance = SchoolAI()
                
                if 'ai_provider' in st.session_state:
                    ai_instance.provider = st.session_state.ai_provider
                if 'ai_model' in st.session_state:
                    ai_instance.model = st.session_state.ai_model
                if 'ollama_url' in st.session_state:
                    ai_instance.ollama_url = st.session_state.ollama_url
                
                risk_data = ai_helper.get_data_by_type('risk')
                
                risk_prompt = f"""
                Conduct a comprehensive risk assessment based on this data:
                
                {risk_data}
                
                Assess:
                1. Academic performance risks
                2. Financial risks and concerns
                3. Operational and safety risks
                4. Compliance and regulatory risks
                5. Risk mitigation strategies and priorities
                """
                
                try:
                    risk_assessment = ai_instance.generate_response(risk_prompt)
                    
                    if "Error:" in risk_assessment:
                        st.error(f"{risk_assessment}")
                    else:
                        st.warning("Risk assessment completed!")
                        
                        with st.expander("⚠️ View Risk Assessment"):
                            st.write(risk_assessment)
                    
                except Exception as e:
                    st.error(f"Error conducting risk assessment: {str(e)}")
    
    # Show previous analysis if available
    if 'last_ai_analysis' in st.session_state:
        st.subheader("📋 Previous Analysis")
        with st.expander("View Last Generated Analysis"):
            st.write(st.session_state.last_ai_analysis)
    
    # AI-Powered Recommendations
    st.subheader("💡 AI Recommendations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎓 Academic", "👥 Administrative", "💰 Financial", "🏫 Infrastructure"])
    
    with tab1:
        st.write("### Academic Recommendations")
        
        if st.button("🤖 Analyze Academic Performance"):
            ai_instance = SchoolAI()
            
            if 'ai_provider' in st.session_state:
                ai_instance.provider = st.session_state.ai_provider
            if 'ai_model' in st.session_state:
                ai_instance.model = st.session_state.ai_model
            if 'ollama_url' in st.session_state:
                ai_instance.ollama_url = st.session_state.ollama_url
            
            academic_data = ai_helper.get_data_by_type('academic')
            
            academic_prompt = f"""
            Based on this academic data, provide specific recommendations:
            
            {academic_data}
            
            Focus on:
            1. Curriculum optimization suggestions
            2. Teaching methodology improvements
            3. Student support programs
            4. Assessment strategy enhancements
            5. Technology integration opportunities
            """
            
            with st.spinner("AI analyzing academic data..."):
                try:
                    recommendations = ai_instance.generate_response(academic_prompt)
                    
                    if "Error:" in recommendations:
                        st.error(f"{recommendations}")
                    else:
                        st.write(recommendations)
                        st.success("Academic analysis completed!")
                except Exception as e:
                    st.error(f"Error generating academic recommendations: {str(e)}")
    
    with tab2:
        st.write("### Administrative Efficiency")
        
        if st.button("🔧 Optimize Operations"):
            ai_instance = SchoolAI()
            
            if 'ai_provider' in st.session_state:
                ai_instance.provider = st.session_state.ai_provider
            if 'ai_model' in st.session_state:
                ai_instance.model = st.session_state.ai_model
            if 'ollama_url' in st.session_state:
                ai_instance.ollama_url = st.session_state.ollama_url
            
            admin_data = ai_helper.get_data_by_type('admin')
            
            admin_prompt = f"""
            Based on this administrative data, suggest improvements:
            
            {admin_data}
            
            Address:
            1. Process automation opportunities
            2. Communication improvements
            3. Resource allocation optimization
            4. Staff productivity enhancements
            5. Digital transformation initiatives
            """
            
            with st.spinner("AI optimizing operations..."):
                try:
                    admin_recommendations = ai_instance.generate_response(admin_prompt)
                    
                    if "Error:" in admin_recommendations:
                        st.error(f"{admin_recommendations}")
                    else:
                        st.write(admin_recommendations)
                        st.success("Operational optimization completed!")
                except Exception as e:
                    st.error(f"Error generating administrative recommendations: {str(e)}")
    
    with tab3:
        st.write("### Financial Optimization")
        
        if st.button("💡 Financial Insights"):
            ai_instance = SchoolAI()
            
            if 'ai_provider' in st.session_state:
                ai_instance.provider = st.session_state.ai_provider
            if 'ai_model' in st.session_state:
                ai_instance.model = st.session_state.ai_model
            if 'ollama_url' in st.session_state:
                ai_instance.ollama_url = st.session_state.ollama_url
            
            financial_data = ai_helper.get_data_by_type('financial')
            
            financial_prompt = f"""
            Based on this financial data, provide optimization recommendations:
            
            {financial_data}
            
            Include:
            1. Cost reduction strategies
            2. Revenue enhancement opportunities
            3. Budget allocation improvements
            4. Investment priorities
            5. Financial risk management
            """
            
            with st.spinner("AI analyzing financial data..."):
                try:
                    financial_insights = ai_instance.generate_response(financial_prompt)
                    
                    if "Error:" in financial_insights:
                        st.error(f"{financial_insights}")
                    else:
                        st.write(financial_insights)
                        st.success("Financial analysis completed!")
                except Exception as e:
                    st.error(f"Error generating financial insights: {str(e)}")
    
    with tab4:
        st.write("### Infrastructure Planning")
        
        if st.button("🏗️ Infrastructure Analysis"):
            ai_instance = SchoolAI()
            
            if 'ai_provider' in st.session_state:
                ai_instance.provider = st.session_state.ai_provider
            if 'ai_model' in st.session_state:
                ai_instance.model = st.session_state.ai_model
            if 'ollama_url' in st.session_state:
                ai_instance.ollama_url = st.session_state.ollama_url
            
            infrastructure_data = ai_helper.get_data_by_type('infrastructure')
            
            infrastructure_prompt = f"""
            Based on this infrastructure data, analyze needs and provide recommendations:
            
            {infrastructure_data}
            
            Cover:
            1. Facility utilization optimization
            2. Technology infrastructure upgrades
            3. Safety and security improvements
            4. Future expansion planning
            5. Maintenance and sustainability strategies
            """
            
            with st.spinner("AI assessing infrastructure..."):
                try:
                    infrastructure_analysis = ai_instance.generate_response(infrastructure_prompt)
                    
                    if "Error:" in infrastructure_analysis:
                        st.error(f"{infrastructure_analysis}")
                    else:
                        st.write(infrastructure_analysis)
                        st.success("Infrastructure analysis completed!")
                except Exception as e:
                    st.error(f"Error generating infrastructure analysis: {str(e)}")
    
    # Data Explorer Section
    st.subheader("🔍 Data Explorer")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("### Available Data Types")
        data_types = ['school', 'report', 'trends', 'risk', 'academic', 
                     'admin', 'financial', 'infrastructure', 'context']
        
        selected_data_type = st.selectbox("Select data type to preview:", data_types)
        
        if st.button("🔎 Preview Selected Data"):
            with st.spinner(f"Loading {selected_data_type} data..."):
                try:
                    preview_data = ai_helper.get_data_by_type(selected_data_type)
                    
                    with st.expander(f"📋 {selected_data_type.title()} Data Preview"):
                        st.text(preview_data)
                        
                except Exception as e:
                    st.error(f"Error loading {selected_data_type} data: {str(e)}")
    
    with col6:
        st.write("### Custom Analysis")
        
        analysis_type = st.selectbox(
            "Choose analysis focus:",
            ["Performance Overview", "Trend Analysis", "Risk Assessment", 
             "Financial Review", "Academic Deep Dive", "Operational Efficiency"]
        )
        
        if st.button("🎯 Run Custom Analysis"):
            ai_instance = SchoolAI()
            
            if 'ai_provider' in st.session_state:
                ai_instance.provider = st.session_state.ai_provider
            if 'ai_model' in st.session_state:
                ai_instance.model = st.session_state.ai_model
            if 'ollama_url' in st.session_state:
                ai_instance.ollama_url = st.session_state.ollama_url
            
            data_mapping = {
                "Performance Overview": 'school',
                "Trend Analysis": 'trends',
                "Risk Assessment": 'risk',
                "Financial Review": 'financial',
                "Academic Deep Dive": 'academic',
                "Operational Efficiency": 'admin'
            }
            
            selected_data = ai_helper.get_data_by_type(data_mapping[analysis_type])
            
            custom_prompt = f"""
            Perform a focused {analysis_type.lower()} based on this data:
            
            {selected_data}
            
            Provide detailed insights, actionable recommendations, and specific next steps 
            for school administrators to implement improvements.
            """
            
            with st.spinner(f"Running {analysis_type}..."):
                try:
                    custom_analysis = ai_instance.generate_response(custom_prompt)
                    
                    if "Error:" in custom_analysis:
                        st.error(f"{custom_analysis}")
                    else:
                        st.write(f"### {analysis_type} Results")
                        st.write(custom_analysis)
                        st.success(f"{analysis_type} completed successfully!")
                        
                except Exception as e:
                    st.error(f"Error running custom analysis: {str(e)}")
    
    # Additional utility section
    st.subheader("🛠️ AI Utilities")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🔄 Clear AI Cache"):
            if 'school_ai' in st.session_state:
                del st.session_state.school_ai
            if 'last_ai_analysis' in st.session_state:
                del st.session_state.last_ai_analysis
            st.success("AI cache cleared successfully!")
    
    with col4:
        if st.button("📤 Export AI Results"):
            if 'last_ai_analysis' in st.session_state:
                st.download_button(
                    label="💾 Download Analysis",
                    data=st.session_state.last_ai_analysis,
                    file_name="ai_school_analysis.txt",
                    mime="text/plain"
                )
            else:
                st.info("No analysis available to export. Generate an analysis first.")
    
    # AI Chat Interface  
    st.subheader("💬 AI Assistant Chat")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**AI Assistant:** {message['content']}")
    
    # Chat input
    user_question = st.chat_input("Ask the AI Assistant anything about your school...")
    
    if user_question:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        ai = SchoolAI()
        
        if 'ai_provider' in st.session_state:
            ai.provider = st.session_state.ai_provider
        if 'ai_model' in st.session_state:
            ai.model = st.session_state.ai_model
        if 'ollama_url' in st.session_state:
            ai.ollama_url = st.session_state.ollama_url
        
        context_data = ai_helper.get_data_by_type('context')
        context = f"""You are an AI assistant for a comprehensive school management system. 
        Here is the current school data for context:
        
        {context_data}
        
        Provide helpful, accurate, and actionable advice for school administrators based on this real data."""
        
        with st.spinner("AI is thinking..."):
            try:
                ai_response = ai.generate_response(user_question, context)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': ai_response
                })
                
            except Exception as e:
                error_response = f"I apologize, but I encountered an error: {str(e)}"
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': error_response
                })
        
        st.rerun()
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def data_input_page():
    """Page for manual data input for students, teachers, courses, grades, attendance, and finance"""
    st.header("📝 Data Input")
    
    manager = st.session_state.school_manager
    
    # Sub-tabs for different data types
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "👨‍🎓 Students", "👨‍🏫 Teachers", "📚 Courses", 
        "📝 Grades", "📅 Attendance", "💰 Financial Transactions", "📁 Bulk Operations"
    ])
    
    with tab1:
        st.subheader("Students Management")
        
        # Display existing students in editable table
        students = manager.get_all_students()
        if students:
            st.write("**Existing Students:**")
            
            # Convert to DataFrame for editing
            df_students = pd.DataFrame(students)
            
            # Convert admission_date from string to datetime for proper handling
            if 'admission_date' in df_students.columns:
                df_students['admission_date'] = pd.to_datetime(df_students['admission_date'], errors='coerce').dt.date
            
            # Configure column config for better editing
            column_config = {
                "student_id": st.column_config.TextColumn("Student ID", disabled=True),
                "name": st.column_config.TextColumn("Name", required=True),
                "age": st.column_config.NumberColumn("Age", min_value=3, max_value=25),
                "grade": st.column_config.SelectboxColumn("Grade", options=[
                    "Kindergarten", "Grade 1", "Grade 2", "Grade 3", "Grade 4", 
                    "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3" 
                ]),
                "gpa": st.column_config.NumberColumn("GPA", min_value=0.0, max_value=4.0),
                "attendance_rate": st.column_config.NumberColumn("Attendance %", min_value=0.0, max_value=100.0),
                "behavior_score": st.column_config.NumberColumn("Behavior Score", min_value=0.0, max_value=100.0),
                "parent_contact": st.column_config.TextColumn("Parent Contact"),
                "medical_info": st.column_config.TextColumn("Medical Info"),
                "admission_date": st.column_config.DateColumn("Admission Date"),
                "status": st.column_config.SelectboxColumn("Status", options=["active", "inactive"])
            }
            
            edited_students = st.data_editor(
                df_students,
                column_config=column_config,
                num_rows="dynamic",
                width="stretch",
                key="students_editor"
            )
            
            # Save changes button
            if st.button("💾 Save Student Changes", key="save_students"):
                try:
                    for index, row in edited_students.iterrows():
                        # Convert date back to string format for database storage
                        admission_date_str = row['admission_date'].strftime('%Y-%m-%d') if pd.notna(row['admission_date']) else None
                        
                        query = '''
                            UPDATE students SET name=?, age=?, grade=?, gpa=?, attendance_rate=?, 
                            behavior_score=?, parent_contact=?, medical_info=?, admission_date=?, status=?
                            WHERE student_id=?
                        '''
                        params = (
                            row['name'], row['age'], row['grade'], row['gpa'], row['attendance_rate'],
                            row['behavior_score'], row['parent_contact'], row['medical_info'], 
                            admission_date_str, row['status'], row['student_id']
                        )
                        manager.db.execute_update(query, params)
                    st.success("✅ Student records updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error updating students: {str(e)}")
        else:
            st.info("No students found. Use the form below to add new students.")
        
        # Collapsible form for adding new students
        with st.expander("➕ Add New Student", expanded=False):
            with st.form("new_student_form"):
                student_name = st.text_input("Student Name*", placeholder="Enter student name")
                student_age = st.number_input("Age*", min_value=3, max_value=25, value=6)
                student_grade = st.selectbox("Grade Level*", [
                    "Kindergarten", "Grade 1", "Grade 2", "Grade 3", "Grade 4", 
                    "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3" 
                ])
                student_gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=0.0)
                student_attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=100.0)
                student_behavior = st.number_input("Behavior Score", min_value=0.0, max_value=100.0, value=100.0)
                parent_contact = st.text_input("Parent Contact", placeholder="Email or phone")
                medical_info = st.text_area("Medical Information", placeholder="Any medical conditions or notes")
                admission_date = st.date_input("Admission Date", value=datetime.now())
                
                submitted = st.form_submit_button("➕ Add Student")
                
                if submitted and student_name:
                    student_id = str(uuid.uuid4())[:8]
                    student = Student(
                        student_id=student_id,
                        name=student_name,
                        age=student_age,
                        grade=student_grade,
                        admission_date=admission_date.strftime('%Y-%m-%d'),
                        gpa=student_gpa,
                        attendance_rate=student_attendance,
                        behavior_score=student_behavior,
                        parent_contact=parent_contact,
                        medical_info=medical_info
                    )
                    
                    try:
                        manager._save_student(student)
                        st.success(f"✅ Student {student_name} added successfully! ID: {student_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding student: {str(e)}")

    with tab2:
        st.subheader("Teachers Management")
        
        # Display existing teachers in editable table
        teachers = manager.db.execute_query("SELECT * FROM teachers ORDER BY name")
        if teachers:
            st.write("**Existing Teachers:**")
            
            # Create DataFrame with the exact column names from your schema
            teachers_df = pd.DataFrame(teachers, columns=[
                'teacher_id', 'name', 'subject', 'department', 'hire_date', 
                'performance_score', 'salary', 'status', 'is_class_teacher'
            ])
            
            # Convert hire_date to string format for editor compatibility
            teachers_df['hire_date'] = pd.to_datetime(teachers_df['hire_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            teachers_df['hire_date'] = teachers_df['hire_date'].fillna('')
            
            # Convert is_class_teacher to boolean
            teachers_df['is_class_teacher'] = teachers_df['is_class_teacher'].astype(bool)
            
            # Configure column types
            column_config = {
                "teacher_id": st.column_config.TextColumn("Teacher ID", disabled=True),
                "name": st.column_config.TextColumn("Name", required=True),
                "subject": st.column_config.TextColumn("Subject"),
                "department": st.column_config.TextColumn("Department"),
                "hire_date": st.column_config.TextColumn(
                    "Hire Date", 
                    help="Format: YYYY-MM-DD"
                ),
                "performance_score": st.column_config.NumberColumn(
                    "Performance Score", 
                    min_value=0, 
                    max_value=100, 
                    step=0.1
                ),
                "salary": st.column_config.NumberColumn("Salary", min_value=0, step=100),
                "status": st.column_config.SelectboxColumn(
                    "Status", 
                    options=["active", "inactive", "on_leave"]
                ),
                "is_class_teacher": st.column_config.CheckboxColumn(
                    "Class Teacher",
                    help="Is this teacher a class teacher?"
                )
            }
            
            # Data editor with column configuration
            edited_teachers = st.data_editor(
                teachers_df,
                column_config=column_config,
                num_rows="dynamic",
                width="stretch",
                key="teachers_data_editor"
            )
            
            # Action buttons in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Save changes button
                if st.button("💾 Save Teacher Changes", key="save_teacher_changes"):
                    try:
                        # Validate date format for each row
                        for idx, row in edited_teachers.iterrows():
                            if 'hire_date' in row and row['hire_date'] and row['hire_date'].strip():
                                try:
                                    datetime.strptime(row['hire_date'], '%Y-%m-%d')
                                except ValueError:
                                    st.error(f"❌ Invalid date format in row {idx + 1}. Use YYYY-MM-DD format.")
                                    st.stop()
                        
                        # Update each teacher record
                        for idx, row in edited_teachers.iterrows():
                            if 'teacher_id' in row:
                                update_query = """
                                    UPDATE teachers 
                                    SET name=?, subject=?, department=?, hire_date=?, 
                                        performance_score=?, salary=?, status=?, is_class_teacher=?
                                    WHERE teacher_id=?
                                """
                                # Convert boolean back to integer for database
                                is_class_teacher_int = 1 if row['is_class_teacher'] else 0
                                params = (
                                    row['name'], row['subject'], row['department'], 
                                    row['hire_date'], row['performance_score'], 
                                    row['salary'], row['status'], is_class_teacher_int, row['teacher_id']
                                )
                                manager.db.execute_update(update_query, params)
                        
                        st.success("✅ Teacher data updated successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error updating teacher data: {str(e)}")
            
            with col2:
                # Delete selected records button
                if st.button("🗑️ Delete Selected Records", key="delete_teachers"):
                    # Get selected rows (rows that were removed from the editor)
                    original_ids = set(teachers_df['teacher_id'].tolist())
                    current_ids = set(edited_teachers['teacher_id'].tolist()) if not edited_teachers.empty else set()
                    deleted_ids = original_ids - current_ids
                    
                    if deleted_ids:
                        try:
                            for teacher_id in deleted_ids:
                                delete_query = "DELETE FROM teachers WHERE teacher_id=?"
                                manager.db.execute_update(delete_query, (teacher_id,))
                            
                            st.success(f"✅ {len(deleted_ids)} teacher record(s) deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting teacher records: {str(e)}")
                    else:
                        st.warning("⚠️ No records selected for deletion. Remove rows from the table above to delete them.")
        else:
            st.info("No teachers found. Use the form below to add new teachers.")
        
        # Add new teacher form
        with st.expander("➕ Add New Teacher", expanded=False):
            with st.form("new_teacher_form"):
                teacher_name = st.text_input("Teacher Name*", placeholder="Enter teacher name")
                teacher_subject = st.text_input("Subject*", placeholder="e.g., Mathematics")
                teacher_department = st.text_input("Department*", placeholder="e.g., STEM")
                teacher_salary = st.number_input("Salary", min_value=0.0, value=0.0)
                performance_score = st.number_input("Performance Score", min_value=0.0, max_value=100.0, value=100.0)
                hire_date = st.date_input("Hire Date", value=datetime.now())
                is_class_teacher = st.checkbox("Is Class Teacher", value=False)
                
                submitted = st.form_submit_button("➕ Add Teacher")
                
                if submitted and teacher_name and teacher_subject and teacher_department:
                    teacher_id = str(uuid.uuid4())[:8]
                    
                    query = '''
                        INSERT INTO teachers (teacher_id, name, subject, department, hire_date, 
                                            performance_score, salary, status, is_class_teacher)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    '''
                    # Convert boolean to integer for database
                    is_class_teacher_int = 1 if is_class_teacher else 0
                    params = (
                        teacher_id, teacher_name, teacher_subject, teacher_department,
                        hire_date.strftime('%Y-%m-%d'), performance_score, teacher_salary, 'active', is_class_teacher_int
                    )
                    
                    try:
                        manager.db.execute_update(query, params)
                        st.success(f"✅ Teacher {teacher_name} added successfully! ID: {teacher_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding teacher: {str(e)}")
    
    with tab3:
        st.subheader("Courses Management")
        
        # Display existing courses in editable table
        courses_query = '''
            SELECT c.course_id, c.name, c.teacher_id, t.name as teacher_name, 
                   c.grade_level, c.credits, c.schedule
            FROM courses c
            LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
            ORDER BY c.name
        '''
        courses = manager.db.execute_query(courses_query)
        
        if courses:
            st.write("**Existing Courses:**")
            
            # Create DataFrame with only necessary columns
            df_courses = pd.DataFrame(courses, columns=[
                'course_id', 'name', 'teacher_id', 'teacher_name', 
                'grade_level', 'credits', 'schedule'
            ])
            
            # Drop the teacher_name column as it's redundant and create display DataFrame
            display_df = df_courses[['course_id', 'name', 'teacher_id', 'grade_level', 'credits', 'schedule']].copy()
            
            # Get teachers for selectbox
            teachers = manager.db.execute_query("SELECT teacher_id, name FROM teachers WHERE status = 'active'")
            teacher_options = {t[1]: t[0] for t in teachers}  # {teacher_name: teacher_id}
            
            # Create reverse mapping for display purposes
            teacher_id_to_name = {t[0]: t[1] for t in teachers}  # {teacher_id: teacher_name}
            
            # Map current teacher_ids to teacher names for display
            display_df['teacher_id'] = display_df['teacher_id'].map(teacher_id_to_name)
            
            column_config = {
                "course_id": st.column_config.TextColumn("Course ID", disabled=True),
                "name": st.column_config.TextColumn("Course Name", required=True),
                "teacher_id": st.column_config.SelectboxColumn(
                    "Teacher", 
                    options=list(teacher_options.keys()),  # Show teacher names in dropdown
                    help="Select the teacher for this course"
                ),
                "grade_level": st.column_config.SelectboxColumn("Grade Level", options=[
                    "Kindergarten", "Grade 1", "Grade 2", "Grade 3", "Grade 4", 
                    "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"
                ]),
                "credits": st.column_config.NumberColumn("Credits", min_value=1, max_value=10),
                "schedule": st.column_config.TextColumn("Schedule")
            }
            
            edited_courses = st.data_editor(
                display_df,
                column_config=column_config,
                num_rows="dynamic",
                width="stretch",
                key="courses_editor"
            )
            
            # Action buttons in columns
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("💾 Save Course Changes", key="save_courses"):
                    try:
                        for index, row in edited_courses.iterrows():
                            # Convert teacher name back to teacher_id for database update
                            teacher_name = row['teacher_id']  # This is now a teacher name
                            teacher_id = teacher_options[teacher_name]  # Convert to teacher_id
                            
                            query = '''
                                UPDATE courses SET name=?, teacher_id=?, grade_level=?, credits=?, schedule=?
                                WHERE course_id=?
                            '''
                            params = (
                                row['name'], teacher_id, row['grade_level'], 
                                row['credits'], row['schedule'], row['course_id']
                            )
                            manager.db.execute_update(query, params)
                        st.success("✅ Course records updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error updating courses: {str(e)}")
            
            with col2:
                # Delete course functionality
                course_options = ["Select course to delete..."] + [f"{row['name']} (ID: {row['course_id']})" for _, row in display_df.iterrows()]
                
                with st.form("delete_course_form"):
                    selected_course = st.selectbox(
                        "Choose course to delete:",
                        options=course_options,
                        key="course_delete_select"
                    )
                    
                    delete_confirm = st.checkbox("⚠️ I confirm I want to delete this course")
                    delete_submitted = st.form_submit_button("🗑️ Delete Course", type="secondary")
                    
                    if delete_submitted and selected_course != "Select course to delete..." and delete_confirm:
                        try:
                            # Extract course_id from the selected option
                            course_id_to_delete = selected_course.split("ID: ")[1].rstrip(")")
                            course_name_to_delete = selected_course.split(" (ID:")[0]
                            
                            # Check for enrollments
                            try:
                                enrollment_check = manager.db.execute_query(
                                    "SELECT COUNT(*) FROM enrollments WHERE course_id = ?", 
                                    (course_id_to_delete,)
                                )
                                
                                if enrollment_check and enrollment_check[0][0] > 0:
                                    st.error(f"❌ Cannot delete course '{course_name_to_delete}'. It has {enrollment_check[0][0]} enrolled students. Remove enrollments first.")
                                    return
                            except Exception:
                                # enrollments table doesn't exist, skip the check
                                pass
                            
                            # Delete the course
                            delete_query = "DELETE FROM courses WHERE course_id = ?"
                            manager.db.execute_update(delete_query, (course_id_to_delete,))
                            st.success(f"✅ Course '{course_name_to_delete}' deleted successfully!")
                            st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Error deleting course: {str(e)}")
                    
                    elif delete_submitted and selected_course == "Select course to delete...":
                        st.warning("⚠️ Please select a course to delete.")
                    
                    elif delete_submitted and not delete_confirm:
                        st.warning("⚠️ Please check the confirmation box to delete the course.")
        else:
            st.info("No courses found. Use the form below to add new courses.")
        
        with st.expander("➕ Add New Course", expanded=False):
            # Get available teachers for dropdown
            teachers = manager.db.execute_query("SELECT teacher_id, name FROM teachers WHERE status = 'active'")
            teacher_options = ["Select a teacher..."] + [t[1] for t in teachers]  # Show teacher names
            teacher_id_map = {t[1]: t[0] for t in teachers}  # Map names to IDs
            
            with st.form("new_course_form"):
                course_name = st.text_input("Course Name*", placeholder="e.g., Algebra I")
                teacher_name = st.selectbox("Teacher*", teacher_options, help="Select a teacher for this course")
                grade_level = st.selectbox("Grade Level*", [
                    "Kindergarten", "Grade 1", "Grade 2", "Grade 3", "Grade 4", 
                    "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"
                ])
                credits = st.number_input("Credits*", min_value=1, max_value=10, value=3)
                schedule = st.text_input("Schedule", placeholder="e.g., Mon/Wed 9:00-10:00")
                
                submitted = st.form_submit_button("➕ Add Course")
                
                if submitted:
                    if not course_name:
                        st.error("❌ Please enter a course name.")
                    elif teacher_name == "Select a teacher...":
                        st.error("❌ Please select a teacher.")
                    else:
                        course_id = str(uuid.uuid4())[:8]
                        teacher_id = teacher_id_map[teacher_name]
                        
                        course = Course(
                            course_id=course_id,
                            name=course_name,
                            teacher_id=teacher_id,
                            grade_level=grade_level,
                            credits=credits,
                            schedule=schedule
                        )
                        
                        query = '''
                            INSERT INTO courses (course_id, name, teacher_id, grade_level, credits, schedule)
                            VALUES (?, ?, ?, ?, ?, ?)
                        '''
                        params = (
                            course.course_id, course.name, course.teacher_id,
                            course.grade_level, course.credits, course.schedule
                        )
                        
                        try:
                            manager.db.execute_update(query, params)
                            st.success(f"✅ Course '{course_name}' assigned to '{teacher_name}' added successfully! ID: {course_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error adding course: {str(e)}")
                        
    with tab4:
        st.subheader("Grades Management")
        
        # Display existing grades in editable table
        grades_query = '''
            SELECT g.grade_id, g.student_id, s.name as student_name, 
                   g.course_id, c.name as course_name, g.grade, g.semester, g.year
            FROM grades g
            LEFT JOIN students s ON g.student_id = s.student_id
            LEFT JOIN courses c ON g.course_id = c.course_id
            ORDER BY g.year DESC, g.semester, s.name
        '''
        grades = manager.db.execute_query(grades_query)
        
        if grades:
            st.write("**Existing Grades:**")
            
            df_grades = pd.DataFrame(grades, columns=[
                'grade_id', 'student_id', 'student_name', 'course_id', 
                'course_name', 'grade', 'semester', 'year'
            ])
            
            # Store original data for comparison
            if 'original_grades_df' not in st.session_state:
                st.session_state.original_grades_df = df_grades.copy()
            
            # Get students and courses for selectboxes
            students = manager.get_all_students()
            student_options = {s['name']: s['student_id'] for s in students}
            courses = manager.db.execute_query("SELECT course_id, name FROM courses")
            course_options = {c[1]: c[0] for c in courses}
            
            column_config = {
                "grade_id": st.column_config.TextColumn("Grade ID", disabled=True),
                "student_id": st.column_config.SelectboxColumn("Student", options=student_options),
                "student_name": st.column_config.TextColumn("Student Name", disabled=True),
                "course_id": st.column_config.SelectboxColumn("Course", options=course_options),
                "course_name": st.column_config.TextColumn("Course Name", disabled=True),
                "grade": st.column_config.NumberColumn("Grade", min_value=0.0, max_value=100.0),
                "semester": st.column_config.TextColumn("Semester"),
                "year": st.column_config.NumberColumn("Year", min_value=2000, max_value=2100)
            }
            
            edited_grades = st.data_editor(
                df_grades,
                column_config=column_config,
                num_rows="dynamic",
                width="stretch",
                key="grades_editor"
            )
            
            if st.button("💾 Save Grade Changes", key="save_grades"):
                try:
                    # Get original grade IDs
                    original_grade_ids = set(st.session_state.original_grades_df['grade_id'].tolist())
                    current_grade_ids = set(edited_grades['grade_id'].tolist())
                    
                    # Find deleted records
                    deleted_grade_ids = original_grade_ids - current_grade_ids
                    
                    # Delete removed records
                    for grade_id in deleted_grade_ids:
                        delete_query = "DELETE FROM grades WHERE grade_id = ?"
                        manager.db.execute_update(delete_query, (grade_id,))
                    
                    # Update existing records
                    for index, row in edited_grades.iterrows():
                        # Skip if this is a new row (grade_id not in original data)
                        if row['grade_id'] in original_grade_ids:
                            query = '''
                                UPDATE grades SET student_id=?, course_id=?, grade=?, semester=?, year=?
                                WHERE grade_id=?
                            '''
                            params = (
                                row['student_id'], row['course_id'], row['grade'], 
                                row['semester'], row['year'], row['grade_id']
                            )
                            manager.db.execute_update(query, params)
                        else:
                            # This is a new record added through the data editor
                            # Insert new record
                            insert_query = '''
                                INSERT INTO grades (grade_id, student_id, course_id, grade, semester, year)
                                VALUES (?, ?, ?, ?, ?, ?)
                            '''
                            params = (
                                row['grade_id'], row['student_id'], row['course_id'], 
                                row['grade'], row['semester'], row['year']
                            )
                            manager.db.execute_update(insert_query, params)
                    
                    # Clear the original data from session state to refresh it
                    if 'original_grades_df' in st.session_state:
                        del st.session_state.original_grades_df
                    
                    st.success("✅ Grade records updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error updating grades: {str(e)}")
        else:
            st.info("No grades found. Use the form below to add new grades.")
        
        with st.expander("➕ Add New Grade", expanded=False):
            # Get available students and courses
            students = manager.get_all_students()
            student_options = {s['name']: s['student_id'] for s in students}
            courses = manager.db.execute_query("SELECT course_id, name FROM courses")
            course_options = {c[1]: c[0] for c in courses}
            
            with st.form("new_grade_form"):
                student_name = st.selectbox("Student*", list(student_options.keys()), help="Select a student")
                course_name = st.selectbox("Course*", list(course_options.keys()), help="Select a course")
                grade_value = st.number_input("Grade*", min_value=0.0, max_value=100.0, value=0.0)
                semester = st.text_input("Semester*", placeholder="e.g., Fall 2025")
                year = st.number_input("Year*", min_value=2000, max_value=2100, value=datetime.now().year)
                
                submitted = st.form_submit_button("➕ Add Grade")
                
                if submitted and student_name and course_name and semester:
                    grade_id = str(uuid.uuid4())[:8]
                    student_id = student_options[student_name]
                    course_id = course_options[course_name]
                    
                    query = '''
                        INSERT INTO grades (grade_id, student_id, course_id, grade, semester, year)
                        VALUES (?, ?, ?, ?, ?, ?)
                    '''
                    params = (grade_id, student_id, course_id, grade_value, semester, year)
                    
                    try:
                        manager.db.execute_update(query, params)
                        st.success(f"✅ Grade added for {student_name} in {course_name}!")
                        # Clear the original data from session state to refresh it
                        if 'original_grades_df' in st.session_state:
                            del st.session_state.original_grades_df
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding grade: {str(e)}")
#-----------------------
#-----------------------    
    with tab5:
            st.subheader("Attendance Management")
            
            # Display existing attendance records in editable table
            attendance_query = '''
                SELECT a.attendance_id, a.student_id, s.name as student_name, 
                       a.date, a.status
                FROM attendance a
                LEFT JOIN students s ON a.student_id = s.student_id
                ORDER BY a.date DESC, s.name
            '''
            attendance = manager.db.execute_query(attendance_query)
            
            if attendance:
                st.write("**Existing Attendance Records:**")
                
                df_attendance = pd.DataFrame(attendance, columns=[
                    'attendance_id', 'student_id', 'student_name', 'date', 'status'
                ])
                
                # Convert date column from TEXT to actual date objects for proper editing
                df_attendance['date'] = pd.to_datetime(df_attendance['date'], errors='coerce').dt.date
                
                # Get students for selectbox
                students = manager.get_all_students()
                student_options = {s['name']: s['student_id'] for s in students}
                
                column_config = {
                    "attendance_id": st.column_config.TextColumn("Attendance ID", disabled=True),
                    "student_id": st.column_config.SelectboxColumn("Student", options=student_options),
                    "student_name": st.column_config.TextColumn("Student Name", disabled=True),
                    "date": st.column_config.DateColumn("Date"),
                    "status": st.column_config.SelectboxColumn("Status", options=["Present", "Absent", "Tardy", "Excused"])
                }
                
                edited_attendance = st.data_editor(
                    df_attendance,
                    column_config=column_config,
                    num_rows="dynamic",
                    width="stretch",
                    key="attendance_editor"
                )
                
                if st.button("💾 Save Attendance Changes", key="save_attendance"):
                    try:
                        for index, row in edited_attendance.iterrows():
                            # Convert date object back to string format for database storage
                            date_str = row['date'].strftime('%Y-%m-%d') if row['date'] else None
                            
                            query = '''
                                UPDATE attendance SET student_id=?, date=?, status=?
                                WHERE attendance_id=?
                            '''
                            params = (
                                row['student_id'], date_str, row['status'], row['attendance_id']
                            )
                            manager.db.execute_update(query, params)
                        st.success("✅ Attendance records updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error updating attendance: {str(e)}")
            else:
                st.info("No attendance records found. Use the form below to add new records.")
            
            with st.expander("➕ Add Attendance Record", expanded=False):
                students = manager.get_all_students()
                student_options = {s['name']: s['student_id'] for s in students}
                
                with st.form("new_attendance_form"):
                    student_name = st.selectbox("Student*", list(student_options.keys()), help="Select a student")
                    attendance_date = st.date_input("Date*", value=datetime.now())
                    status = st.selectbox("Status*", ["Present", "Absent", "Tardy", "Excused"])
                    
                    submitted = st.form_submit_button("➕ Add Attendance")
                    
                    if submitted and student_name:
                        attendance_id = str(uuid.uuid4())[:8]
                        student_id = student_options[student_name]
                        
                        query = '''
                            INSERT INTO attendance (attendance_id, student_id, date, status)
                            VALUES (?, ?, ?, ?)
                        '''
                        params = (attendance_id, student_id, attendance_date.strftime('%Y-%m-%d'), status)
                        
                        try:
                            manager.db.execute_update(query, params)
                            st.success(f"✅ Attendance recorded for {student_name} on {attendance_date.strftime('%Y-%m-%d')}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error adding attendance: {str(e)}")

    with tab6:
        st.subheader("Financial Transactions Management")
        
        # Currency selection
        col1, col2 = st.columns([1, 3])
        with col1:
            currency_options = {
                "₵ (Ghana Cedi)": "₵",
                "$ (US Dollar)": "$",
                "€ (Euro)": "€",
                "£ (British Pound)": "£",
                "₦ (Nigerian Naira)": "₦"
            }
            selected_currency_name = st.selectbox(
                "Currency", 
                list(currency_options.keys()),
                index=0 if 'selected_currency_name' not in st.session_state else list(currency_options.keys()).index(st.session_state.get('selected_currency_name', "₵ (Ghana Cedi)")),
                help="Select currency for display and new transactions",
                key="currency_selector"
            )
            currency_symbol = currency_options[selected_currency_name]
            
            # Store currency selection in session state for use in other pages
            st.session_state.selected_currency_name = selected_currency_name
            st.session_state.currency_symbol = currency_symbol
        
        # Display existing transactions in editable table
        finance_query = '''
            SELECT f.transaction_id, f.student_id, s.name as student_name, 
                   f.amount, f.type, f.description, f.date, f.status
            FROM finance f
            LEFT JOIN students s ON f.student_id = s.student_id
            ORDER BY f.date DESC
        '''
        transactions = manager.db.execute_query(finance_query)
        
        if transactions:
            st.write("**Existing Financial Transactions:**")
            
            df_transactions = pd.DataFrame(transactions, columns=[
                'transaction_id', 'student_id', 'student_name', 'amount', 
                'type', 'description', 'date', 'status'
            ])
            
            # Add a delete column with checkboxes
            df_transactions['delete'] = False
            
            # Format amount column with selected currency
            df_transactions['formatted_amount'] = df_transactions['amount'].apply(
                lambda x: f"{currency_symbol}{x:,.2f}" if pd.notnull(x) else f"{currency_symbol}0.00"
            )
            
            # Convert date column from string to datetime for proper editing
            try:
                df_transactions['date'] = pd.to_datetime(df_transactions['date']).dt.date
            except Exception:
                st.warning("Date conversion issue. Date column will be text-editable.")
            
            # Get students for selectbox
            students = manager.get_all_students()
            student_options = {s['name']: s['student_id'] for s in students}
            student_options['None'] = None  # Allow no student selection
            
            column_config = {
                "delete": st.column_config.CheckboxColumn("Delete", help="Check to mark for deletion"),
                "transaction_id": st.column_config.TextColumn("Transaction ID", disabled=True),
                "student_id": st.column_config.SelectboxColumn("Student", options=student_options),
                "student_name": st.column_config.TextColumn("Student Name", disabled=True),
                "amount": st.column_config.NumberColumn(f"Amount ({currency_symbol})", min_value=0.0, format=f"{currency_symbol}%.2f"),
                "formatted_amount": None,  # Hide the formatted amount column
                "type": st.column_config.SelectboxColumn("Type", options=["Tuition", "Fee", "Donation", "Refund", "Other"]),
                "description": st.column_config.TextColumn("Description"),
                "status": st.column_config.SelectboxColumn("Status", options=["Pending", "Completed", "Failed"])
            }
            
            # Only add DateColumn if date conversion was successful
            if df_transactions['date'].dtype == 'object' and not df_transactions['date'].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').all():
                column_config["date"] = st.column_config.TextColumn("Date (YYYY-MM-DD)")
            else:
                column_config["date"] = st.column_config.DateColumn("Date")
            
            # Reorder columns to put delete first
            column_order = ['delete', 'transaction_id', 'student_id', 'student_name', 'amount', 'type', 'description', 'date', 'status']
            df_transactions = df_transactions[column_order]
            
            edited_transactions = st.data_editor(
                df_transactions,
                column_config=column_config,
                num_rows="dynamic",
                width="stretch",
                key="transactions_editor"
            )
            
            # Action buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("💾 Save Transaction Changes", key="save_transactions"):
                    try:
                        # Only update rows that are NOT marked for deletion
                        rows_to_update = edited_transactions[~edited_transactions['delete']]
                        
                        for index, row in rows_to_update.iterrows():
                            # Handle date conversion for database storage
                            date_value = row['date']
                            if isinstance(date_value, str):
                                # If it's still a string, validate format
                                try:
                                    datetime.strptime(date_value, '%Y-%m-%d')
                                    formatted_date = date_value
                                except ValueError:
                                    st.error(f"Invalid date format in row {index + 1}. Please use YYYY-MM-DD format.")
                                    continue
                            else:
                                # If it's a date object, convert to string
                                formatted_date = date_value.strftime('%Y-%m-%d')
                            
                            query = '''
                                UPDATE finance SET student_id=?, amount=?, type=?, description=?, date=?, status=?
                                WHERE transaction_id=?
                            '''
                            params = (
                                row['student_id'], row['amount'], row['type'], row['description'],
                                formatted_date, row['status'], row['transaction_id']
                            )
                            manager.db.execute_update(query, params)
                        st.success("✅ Transaction records updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error updating transactions: {str(e)}")
            
            with col2:
                # Check if any rows are marked for deletion
                rows_to_delete = edited_transactions[edited_transactions['delete']]
                
                if not rows_to_delete.empty:
                    if st.button(f"🗑️ Delete Selected ({len(rows_to_delete)} items)", key="delete_transactions", type="secondary"):
                        # Store the transaction IDs to delete in session state
                        st.session_state.transactions_to_delete = rows_to_delete['transaction_id'].tolist()
                        st.session_state.confirm_delete = True
                else:
                    st.button("🗑️ Delete Selected (0 items)", key="delete_transactions_disabled", disabled=True)
            
            # Show confirmation dialog if needed
            if st.session_state.get('confirm_delete', False):
                # Get transaction details for preview
                transaction_ids_to_delete = st.session_state.get('transactions_to_delete', [])
                rows_to_delete = edited_transactions[edited_transactions['transaction_id'].isin(transaction_ids_to_delete)]
                
                st.warning(f"⚠️ Are you sure you want to delete {len(transaction_ids_to_delete)} transaction(s)? This action cannot be undone.")
                
                # Show details of transactions to be deleted
                st.write("**Transactions to be deleted:**")
                delete_preview = rows_to_delete[['transaction_id', 'student_name', 'amount', 'type', 'description', 'date']].copy()
                delete_preview['amount'] = delete_preview['amount'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                st.dataframe(delete_preview, width="stretch")
                
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Confirm Delete", key="confirm_delete_btn", type="primary"):
                        try:
                            deleted_count = 0
                            for transaction_id in transaction_ids_to_delete:
                                delete_query = 'DELETE FROM finance WHERE transaction_id = ?'
                                manager.db.execute_update(delete_query, (transaction_id,))
                                deleted_count += 1
                            
                            st.success(f"✅ Successfully deleted {deleted_count} transaction(s)!")
                            # Clear session state
                            if 'confirm_delete' in st.session_state:
                                del st.session_state.confirm_delete
                            if 'transactions_to_delete' in st.session_state:
                                del st.session_state.transactions_to_delete
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting transactions: {str(e)}")
                            # Clear session state on error too
                            if 'confirm_delete' in st.session_state:
                                del st.session_state.confirm_delete
                            if 'transactions_to_delete' in st.session_state:
                                del st.session_state.transactions_to_delete
                
                with col_cancel:
                    if st.button("❌ Cancel", key="cancel_delete_btn"):
                        # Clear session state
                        if 'confirm_delete' in st.session_state:
                            del st.session_state.confirm_delete
                        if 'transactions_to_delete' in st.session_state:
                            del st.session_state.transactions_to_delete
                        st.rerun()
                    
        else:
            st.info("No financial transactions found. Use the form below to add new transactions.")
        
        with st.expander("➕ Add Financial Transaction", expanded=False):
            students = manager.get_all_students()
            student_options = {s['name']: s['student_id'] for s in students}
            
            with st.form("new_transaction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    student_name = st.selectbox("Student", list(student_options.keys()), help="Select a student (optional)")
                    amount = st.number_input(f"Amount* ({currency_symbol})", min_value=0.0, value=0.0, format="%.2f")
                    transaction_type = st.selectbox("Type*", ["Tuition", "Fee", "Donation", "Refund", "Other"])
                
                with col2:
                    description = st.text_input("Description*", placeholder="e.g., Tuition payment for Fall 2025")
                    transaction_date = st.date_input("Date*", value=datetime.now())
                    transaction_status = st.selectbox("Status*", ["Pending", "Completed", "Failed"])
                
                # Display amount preview with selected currency
                if amount > 0:
                    st.info(f"💰 Amount Preview: {currency_symbol}{amount:,.2f}")
                
                submitted = st.form_submit_button("➕ Add Transaction")
                
                if submitted and amount and description:
                    transaction_id = str(uuid.uuid4())[:8]
                    student_id = student_options[student_name] if student_name else None
                    
                    # Store currency symbol with the transaction for future reference
                    description_with_currency = f"{description} ({currency_symbol})"
                    
                    query = '''
                        INSERT INTO finance (transaction_id, student_id, amount, type, description, date, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    '''
                    params = (
                        transaction_id, student_id, amount, transaction_type,
                        description_with_currency, transaction_date.strftime('%Y-%m-%d'), transaction_status
                    )
                    
                    try:
                        manager.db.execute_update(query, params)
                        st.success(f"✅ Transaction added: {currency_symbol}{amount:,.2f} - {description}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding transaction: {str(e)}")  
#----------------------------------------
#----------------------------------------
    with tab7:
            st.subheader("Bulk Data Operations")
            
            # Create sub-sections for different bulk operations
            bulk_tab1, bulk_tab2, bulk_tab3, bulk_tab4 = st.tabs([
                "📤 Export Data", "📥 Import Data", "💾 Backup", "🔄 Restore"
            ])
            
            with bulk_tab1:
                st.write("### Export School Data")
                
                # Data type selection
                export_type = st.selectbox("Select data to export:", [
                    "Students", "Teachers", "Courses", "Grades", "Attendance", "Financial Transactions", "All Data"
                ])
                
                if st.button("🔄 Generate Export Data", key="generate_export"):
                    with st.spinner("Preparing data for export..."):
                        try:
                            if export_type == "Students":
                                data = manager.get_all_students()
                                filename_base = "students_data"
                            elif export_type == "Teachers":
                                teachers = manager.db.execute_query("SELECT * FROM teachers")
                                data = [dict(zip(['teacher_id', 'name', 'subject', 'department', 'hire_date', 'performance_score', 'salary', 'status'], row)) for row in teachers]
                                filename_base = "teachers_data"
                            elif export_type == "Courses":
                                courses = manager.db.execute_query("""
                                    SELECT c.course_id, c.name, c.teacher_id, t.name as teacher_name, 
                                           c.grade_level, c.credits, c.schedule
                                    FROM courses c
                                    LEFT JOIN teachers t ON c.teacher_id = t.teacher_id
                                """)
                                data = [dict(zip(['course_id', 'name', 'teacher_id', 'teacher_name', 'grade_level', 'credits', 'schedule'], row)) for row in courses]
                                filename_base = "courses_data"
                            elif export_type == "Grades":
                                grades = manager.db.execute_query("""
                                    SELECT g.grade_id, g.student_id, s.name as student_name, 
                                           g.course_id, c.name as course_name, g.grade, g.semester, g.year
                                    FROM grades g
                                    LEFT JOIN students s ON g.student_id = s.student_id
                                    LEFT JOIN courses c ON g.course_id = c.course_id
                                """)
                                data = [dict(zip(['grade_id', 'student_id', 'student_name', 'course_id', 'course_name', 'grade', 'semester', 'year'], row)) for row in grades]
                                filename_base = "grades_data"
                            elif export_type == "Attendance":
                                attendance = manager.db.execute_query("""
                                    SELECT a.attendance_id, a.student_id, s.name as student_name, a.date, a.status
                                    FROM attendance a
                                    LEFT JOIN students s ON a.student_id = s.student_id
                                """)
                                data = [dict(zip(['attendance_id', 'student_id', 'student_name', 'date', 'status'], row)) for row in attendance]
                                filename_base = "attendance_data"
                            elif export_type == "Financial Transactions":
                                transactions = manager.db.execute_query("""
                                    SELECT f.transaction_id, f.student_id, s.name as student_name, 
                                           f.amount, f.type, f.description, f.date, f.status
                                    FROM finance f
                                    LEFT JOIN students s ON f.student_id = s.student_id
                                """)
                                data = [dict(zip(['transaction_id', 'student_id', 'student_name', 'amount', 'type', 'description', 'date', 'status'], row)) for row in transactions]
                                filename_base = "finance_data"
                            else:  # All Data
                                st.write("**All Data Export - Multiple formats available:**")
                                
                                # Get all data
                                students_data = manager.get_all_students()
                                teachers_data = manager.db.execute_query("SELECT * FROM teachers")
                                courses_data = manager.db.execute_query("SELECT * FROM courses")
                                grades_data = manager.db.execute_query("SELECT * FROM grades")
                                attendance_data = manager.db.execute_query("SELECT * FROM attendance")
                                finance_data = manager.db.execute_query("SELECT * FROM finance")
                                
                                # Create multi-sheet Excel
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    if students_data:
                                        pd.DataFrame(students_data).to_excel(writer, sheet_name='Students', index=False)
                                    if teachers_data:
                                        pd.DataFrame(teachers_data, columns=['teacher_id', 'name', 'subject', 'department', 'hire_date', 'performance_score', 'salary', 'status']).to_excel(writer, sheet_name='Teachers', index=False)
                                    if courses_data:
                                        pd.DataFrame(courses_data, columns=['course_id', 'name', 'teacher_id', 'grade_level', 'credits', 'schedule']).to_excel(writer, sheet_name='Courses', index=False)
                                    if grades_data:
                                        pd.DataFrame(grades_data, columns=['grade_id', 'student_id', 'course_id', 'grade', 'semester', 'year']).to_excel(writer, sheet_name='Grades', index=False)
                                    if attendance_data:
                                        pd.DataFrame(attendance_data, columns=['attendance_id', 'student_id', 'date', 'status']).to_excel(writer, sheet_name='Attendance', index=False)
                                    if finance_data:
                                        pd.DataFrame(finance_data, columns=['transaction_id', 'student_id', 'amount', 'type', 'description', 'date', 'status']).to_excel(writer, sheet_name='Finance', index=False)
                                
                                excel_data = buffer.getvalue()
                                st.download_button(
                                    label="📊 Download Complete Database (Excel)",
                                    data=excel_data,
                                    file_name=f"school_complete_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                
                                # Combined JSON export
                                all_data = {
                                    'students': students_data,
                                    'teachers': [dict(zip(['teacher_id', 'name', 'subject', 'department', 'hire_date', 'performance_score', 'salary', 'status'], row)) for row in teachers_data],
                                    'courses': [dict(zip(['course_id', 'name', 'teacher_id', 'grade_level', 'credits', 'schedule'], row)) for row in courses_data],
                                    'grades': [dict(zip(['grade_id', 'student_id', 'course_id', 'grade', 'semester', 'year'], row)) for row in grades_data],
                                    'attendance': [dict(zip(['attendance_id', 'student_id', 'date', 'status'], row)) for row in attendance_data],
                                    'finance': [dict(zip(['transaction_id', 'student_id', 'amount', 'type', 'description', 'date', 'status'], row)) for row in finance_data]
                                }
                                
                                json_data = json.dumps(all_data, indent=2, default=str)
                                st.download_button(
                                    label="📋 Download Complete Database (JSON)",
                                    data=json_data,
                                    file_name=f"school_complete_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                                return
                            
                            if data:
                                df = pd.DataFrame(data)
                                
                                # Create download buttons for different formats
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="📄 Download CSV",
                                        data=csv,
                                        file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    # Convert to Excel
                                    buffer = io.BytesIO()
                                    df.to_excel(buffer, index=False, engine='openpyxl')
                                    excel_data = buffer.getvalue()
                                    
                                    st.download_button(
                                        label="📊 Download Excel",
                                        data=excel_data,
                                        file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                
                                with col3:
                                    json_data = df.to_json(orient='records', indent=2)
                                    st.download_button(
                                        label="📋 Download JSON",
                                        data=json_data,
                                        file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json"
                                    )
                                
                                st.success(f"✅ {export_type} data prepared for download ({len(data)} records)")
                            else:
                                st.warning(f"⚠️ No {export_type.lower()} data found to export")
                                
                        except Exception as e:
                            st.error(f"❌ Error preparing export data: {str(e)}")
            
            with bulk_tab2:
                st.write("### Import Data from Files")
                
                # Import type selection
                import_type = st.selectbox("Select data type to import:", [
                    "Students", "Teachers", "Courses", "Grades", "Attendance", "Financial Transactions"
                ], key="import_type_select")
                
                # File upload
                uploaded_file = st.file_uploader(
                    f"Choose {import_type} file to import", 
                    type=['csv', 'xlsx', 'json'],
                    help="Upload CSV, Excel, or JSON file with the appropriate data structure"
                )
                
                if uploaded_file is not None:
                    try:
                        # Read file based on type
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.xlsx'):
                            df = pd.read_excel(uploaded_file)
                        elif uploaded_file.name.endswith('.json'):
                            df = pd.read_json(uploaded_file)
                        
                        # Convert date columns to proper format for data display
                        date_columns = []
                        if import_type == "Students" and 'admission_date' in df.columns:
                            date_columns.append('admission_date')
                        elif import_type == "Teachers" and 'hire_date' in df.columns:
                            date_columns.append('hire_date')
                        elif import_type in ["Attendance", "Financial Transactions"] and 'date' in df.columns:
                            date_columns.append('date')
                        
                        # Convert date columns to string format for consistent handling
                        for col in date_columns:
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                                df[col] = df[col].fillna('')
                        
                        st.write("**Preview of uploaded data:**")
                        st.dataframe(df.head())
                        st.write(f"Total records: {len(df)}")
                        
                        # Import options
                        col1, col2 = st.columns(2)
                        with col1:
                            import_mode = st.radio("Import mode:", ["Add new records", "Replace existing data"])
                        with col2:
                            validate_data = st.checkbox("Validate data before import", value=True)
                        
                        # Initialize session state for confirmation
                        if "replace_confirmed" not in st.session_state:
                            st.session_state.replace_confirmed = False
                        
                        # Handle Replace existing data confirmation
                        if import_mode == "Replace existing data" and not st.session_state.replace_confirmed:
                            st.warning("⚠️ This will replace all existing data!")
                            if st.button("⚠️ Confirm Replace", key="confirm_replace"):
                                st.session_state.replace_confirmed = True
                                st.rerun()
                        else:
                            if st.button(f"📥 Import {import_type} Data", key="import_data_btn"):
                                with st.spinner(f"Importing {import_type.lower()} data..."):
                                    try:
                                        imported_count = 0
                                        errors = []
                                        
                                        # Clear existing data if replace mode is confirmed
                                        if import_mode == "Replace existing data" and st.session_state.replace_confirmed:
                                            table_map = {
                                                "Students": "students",
                                                "Teachers": "teachers", 
                                                "Courses": "courses",
                                                "Grades": "grades",
                                                "Attendance": "attendance",
                                                "Financial Transactions": "finance"
                                            }
                                            manager.db.execute_update(f"DELETE FROM {table_map[import_type]}")
                                            st.session_state.replace_confirmed = False  # Reset confirmation
                                        
                                        # Import data row by row
                                        for index, row in df.iterrows():
                                            try:
                                                if import_type == "Students":
                                                    student_id = str(uuid.uuid4())[:8] if 'student_id' not in row or pd.isna(row['student_id']) else str(row['student_id'])
                                                    # Ensure admission_date is properly formatted
                                                    admission_date = row.get('admission_date', datetime.now().strftime('%Y-%m-%d'))
                                                    if pd.isna(admission_date) or admission_date == '' or admission_date == 'NaT':
                                                        admission_date = datetime.now().strftime('%Y-%m-%d')
                                                    
                                                    query = '''
                                                        INSERT OR REPLACE INTO students 
                                                        (student_id, name, age, grade, admission_date, gpa, attendance_rate, behavior_score, parent_contact, medical_info, status)
                                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                                    '''
                                                    params = (
                                                        student_id, 
                                                        str(row.get('name', '')), 
                                                        int(float(row.get('age', 0))) if pd.notna(row.get('age', 0)) else 0, 
                                                        str(row.get('grade', '')),
                                                        str(admission_date), 
                                                        float(row.get('gpa', 0.0)) if pd.notna(row.get('gpa', 0.0)) else 0.0, 
                                                        float(row.get('attendance_rate', 100.0)) if pd.notna(row.get('attendance_rate', 100.0)) else 100.0, 
                                                        float(row.get('behavior_score', 100.0)) if pd.notna(row.get('behavior_score', 100.0)) else 100.0, 
                                                        str(row.get('parent_contact', '')), 
                                                        str(row.get('medical_info', '')), 
                                                        str(row.get('status', 'active'))
                                                    )
                                                    
                                                elif import_type == "Teachers":
                                                    teacher_id = str(uuid.uuid4())[:8] if 'teacher_id' not in row or pd.isna(row['teacher_id']) else str(row['teacher_id'])
                                                    # Ensure hire_date is properly formatted
                                                    hire_date = row.get('hire_date', datetime.now().strftime('%Y-%m-%d'))
                                                    if pd.isna(hire_date) or hire_date == '' or hire_date == 'NaT':
                                                        hire_date = datetime.now().strftime('%Y-%m-%d')
                                                        
                                                    query = '''
                                                        INSERT OR REPLACE INTO teachers 
                                                        (teacher_id, name, subject, department, hire_date, performance_score, salary, status)
                                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                    '''
                                                    params = (
                                                        teacher_id, 
                                                        str(row.get('name', '')), 
                                                        str(row.get('subject', '')), 
                                                        str(row.get('department', '')),
                                                        str(hire_date), 
                                                        float(row.get('performance_score', 100.0)) if pd.notna(row.get('performance_score', 100.0)) else 100.0, 
                                                        float(row.get('salary', 0.0)) if pd.notna(row.get('salary', 0.0)) else 0.0, 
                                                        str(row.get('status', 'active'))
                                                    )
                                                    
                                                elif import_type == "Courses":
                                                    course_id = str(uuid.uuid4())[:8] if 'course_id' not in row or pd.isna(row['course_id']) else str(row['course_id'])
                                                    query = '''
                                                        INSERT OR REPLACE INTO courses 
                                                        (course_id, name, teacher_id, grade_level, credits, schedule)
                                                        VALUES (?, ?, ?, ?, ?, ?)
                                                    '''
                                                    params = (
                                                        course_id, 
                                                        str(row.get('name', '')), 
                                                        str(row.get('teacher_id', '')), 
                                                        str(row.get('grade_level', '')), 
                                                        int(float(row.get('credits', 0))) if pd.notna(row.get('credits', 0)) else 0, 
                                                        str(row.get('schedule', ''))
                                                    )
                                                    
                                                elif import_type == "Grades":
                                                    grade_id = str(uuid.uuid4())[:8] if 'grade_id' not in row or pd.isna(row['grade_id']) else str(row['grade_id'])
                                                    query = '''
                                                        INSERT OR REPLACE INTO grades 
                                                        (grade_id, student_id, course_id, grade, semester, year)
                                                        VALUES (?, ?, ?, ?, ?, ?)
                                                    '''
                                                    params = (
                                                        grade_id, 
                                                        str(row.get('student_id', '')), 
                                                        str(row.get('course_id', '')), 
                                                        str(row.get('grade', '')), 
                                                        str(row.get('semester', '')), 
                                                        int(float(row.get('year', datetime.now().year))) if pd.notna(row.get('year', datetime.now().year)) else datetime.now().year
                                                    )
                                                    
                                                elif import_type == "Attendance":
                                                    attendance_id = str(uuid.uuid4())[:8] if 'attendance_id' not in row or pd.isna(row['attendance_id']) else str(row['attendance_id'])
                                                    # Ensure date is properly formatted
                                                    attendance_date = row.get('date', datetime.now().strftime('%Y-%m-%d'))
                                                    if pd.isna(attendance_date) or attendance_date == '' or attendance_date == 'NaT':
                                                        attendance_date = datetime.now().strftime('%Y-%m-%d')
                                                        
                                                    query = '''
                                                        INSERT OR REPLACE INTO attendance 
                                                        (attendance_id, student_id, date, status)
                                                        VALUES (?, ?, ?, ?)
                                                    '''
                                                    params = (
                                                        attendance_id, 
                                                        str(row.get('student_id', '')), 
                                                        str(attendance_date), 
                                                        str(row.get('status', 'present'))
                                                    )
                                                    
                                                elif import_type == "Financial Transactions":
                                                    transaction_id = str(uuid.uuid4())[:8] if 'transaction_id' not in row or pd.isna(row['transaction_id']) else str(row['transaction_id'])
                                                    # Ensure date is properly formatted
                                                    transaction_date = row.get('date', datetime.now().strftime('%Y-%m-%d'))
                                                    if pd.isna(transaction_date) or transaction_date == '' or transaction_date == 'NaT':
                                                        transaction_date = datetime.now().strftime('%Y-%m-%d')
                                                        
                                                    query = '''
                                                        INSERT OR REPLACE INTO finance 
                                                        (transaction_id, student_id, amount, type, description, date, status)
                                                        VALUES (?, ?, ?, ?, ?, ?, ?)
                                                    '''
                                                    params = (
                                                        transaction_id, 
                                                        str(row.get('student_id', '')), 
                                                        float(row.get('amount', 0.0)) if pd.notna(row.get('amount', 0.0)) else 0.0, 
                                                        str(row.get('type', '')), 
                                                        str(row.get('description', '')), 
                                                        str(transaction_date), 
                                                        str(row.get('status', 'pending'))
                                                    )
                                                
                                                manager.db.execute_update(query, params)
                                                imported_count += 1
                                                
                                            except Exception as row_error:
                                                errors.append(f"Row {index + 1}: {str(row_error)}")
                                        
                                        if imported_count > 0:
                                            st.success(f"✅ Successfully imported {imported_count} {import_type.lower()} records!")
                                        
                                        if errors:
                                            st.warning(f"⚠️ {len(errors)} rows had errors:")
                                            for error in errors[:5]:  # Show first 5 errors
                                                st.write(f"• {error}")
                                            if len(errors) > 5:
                                                st.write(f"... and {len(errors) - 5} more errors")
                                        
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"❌ Import failed: {str(e)}")
                                        
                    except Exception as e:
                        st.error(f"❌ Error reading file: {str(e)}")
            
            with bulk_tab3:
                st.write("### Database Backup")
                
                # Backup options
                backup_type = st.radio("Backup type:", ["Complete Database", "Selected Tables"])
                
                if backup_type == "Selected Tables":
                    tables_to_backup = st.multiselect("Select tables to backup:", [
                        "students", "teachers", "courses", "grades", "attendance", "finance"
                    ])
                
                backup_format = st.selectbox("Backup format:", ["SQLite Database (.db)", "JSON Export", "Excel Workbook"])
                
                if st.button("💾 Create Backup"):
                    with st.spinner("Creating backup..."):
                        try:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            if backup_format == "SQLite Database (.db)":
                                # Create a copy of the database file
                                backup_path = f"school_backup_{timestamp}.db"
                                
                                # Read the current database
                                with open(manager.db.db_path, 'rb') as f:
                                    db_data = f.read()
                                
                                st.download_button(
                                    label="📥 Download Database Backup",
                                    data=db_data,
                                    file_name=backup_path,
                                    mime="application/octet-stream"
                                )
                                
                            elif backup_format == "JSON Export":
                                # Export all data as JSON
                                backup_data = {}
                                tables = ["students", "teachers", "courses", "grades", "attendance", "finance"]
                                
                                if backup_type == "Selected Tables":
                                    tables = tables_to_backup
                                
                                for table in tables:
                                    try:
                                        data = manager.db.execute_query(f"SELECT * FROM {table}")
                                        # Get column names
                                        columns_info = manager.db.execute_query(f"PRAGMA table_info({table})")
                                        columns = [col[1] for col in columns_info]  # col[1] contains the column name
                                        backup_data[table] = [dict(zip(columns, row)) for row in data]
                                    except Exception as table_error:
                                        st.warning(f"⚠️ Could not backup table {table}: {str(table_error)}")
                                        backup_data[table] = []
                                
                                json_backup = json.dumps(backup_data, indent=2, default=str)
                                st.download_button(
                                    label="📥 Download JSON Backup",
                                    data=json_backup,
                                    file_name=f"school_backup_{timestamp}.json",
                                    mime="application/json"
                                )
                                
                            elif backup_format == "Excel Workbook":
                                # Export all data as Excel workbook
                                buffer = io.BytesIO()
                                tables = ["students", "teachers", "courses", "grades", "attendance", "finance"]
                                
                                if backup_type == "Selected Tables":
                                    tables = tables_to_backup
                                
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    for table in tables:
                                        try:
                                            data = manager.db.execute_query(f"SELECT * FROM {table}")
                                            if data:
                                                columns_info = manager.db.execute_query(f"PRAGMA table_info({table})")
                                                columns = [col[1] for col in columns_info]
                                                df = pd.DataFrame(data, columns=columns)
                                                df.to_excel(writer, sheet_name=table.capitalize(), index=False)
                                        except Exception as table_error:
                                            st.warning(f"⚠️ Could not backup table {table}: {str(table_error)}")
                                
                                excel_data = buffer.getvalue()
                                st.download_button(
                                    label="📥 Download Excel Backup",
                                    data=excel_data,
                                    file_name=f"school_backup_{timestamp}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            st.success(f"✅ Backup created successfully! Timestamp: {timestamp}")
                            st.info("💡 Store your backup file in a safe location")
                            
                        except Exception as e:
                            st.error(f"❌ Backup failed: {str(e)}")
#===========            
            with bulk_tab4:
                            st.write("### Database Restore")
                            
                            st.warning("⚠️ **Warning**: Restoring will replace current data. Make sure to backup first!")
                            
                            restore_mode = st.radio("Restore mode:", ["Complete Database", "Merge with Existing Data"])
                            
                            uploaded_backup = st.file_uploader(
                                "Choose backup file to restore", 
                                type=['db', 'json', 'xlsx'],
                                help="Upload a .db database file, .json backup file, or .xlsx Excel backup"
                            )
                            
                            if uploaded_backup is not None:
                                st.write("**Backup file details:**")
                                st.write(f"• Filename: {uploaded_backup.name}")
                                st.write(f"• Size: {len(uploaded_backup.getvalue())} bytes")
                                
                                confirm_restore = st.checkbox("I understand this will modify/replace current data")
                                
                                if confirm_restore and st.button("🔄 Restore from Backup"):
                                    with st.spinner("Restoring from backup..."):
                                        try:
                                            if uploaded_backup.name.endswith('.db'):
                                                if restore_mode == "Complete Database":
                                                    backup_data = uploaded_backup.getvalue()
                                                    with open(manager.db.db_path, 'wb') as f:
                                                        f.write(backup_data)
                                                    st.success("✅ Database restored successfully!")
                                                    st.info("🔄 Please refresh the page to see the restored data.")
                                                else:
                                                    st.error("❌ Merge mode not supported for .db files")
                                                    
                                            elif uploaded_backup.name.endswith('.json'):
                                                backup_data = json.loads(uploaded_backup.getvalue().decode('utf-8'))
                                                
                                                restored_tables = []
                                                for table_name, table_data in backup_data.items():
                                                    if table_data and isinstance(table_data, list):
                                                        try:
                                                            if restore_mode == "Complete Database":
                                                                manager.db.execute_update(f"DELETE FROM {table_name}")
                                                            
                                                            for record in table_data:
                                                                if isinstance(record, dict):
                                                                    columns = list(record.keys())
                                                                    values = [str(v) if v is not None else '' for v in record.values()]
                                                                    placeholders = ', '.join(['?' for _ in columns])
                                                                    
                                                                    query = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                                                                    manager.db.execute_update(query, values)
                                                            
                                                            restored_tables.append(f"{table_name} ({len(table_data)} records)")
                                                        except Exception:
                                                            st.error(f"Failed to restore table: {table_name}")
                                                
                                                if restored_tables:
                                                    st.success("✅ Data restored successfully!")
                                                    st.info("**Restored tables:**")
                                                    for table_info in restored_tables:
                                                        st.write(f"• {table_info}")
                                                else:
                                                    st.error("❌ No data was restored. Please check the backup file format.")
                                                    
                                            elif uploaded_backup.name.endswith('.xlsx'):
                                                excel_file = pd.ExcelFile(uploaded_backup)
                                                restored_tables = []
                                                
                                                for sheet_name in excel_file.sheet_names:
                                                    try:
                                                        df = pd.read_excel(uploaded_backup, sheet_name=sheet_name)
                                                        table_name = sheet_name.lower()
                                                        
                                                        if restore_mode == "Complete Database":
                                                            manager.db.execute_update(f"DELETE FROM {table_name}")
                                                        
                                                        date_columns = ['hire_date', 'admission_date', 'date']
                                                        for col in date_columns:
                                                            if col in df.columns:
                                                                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                                                                df[col] = df[col].fillna('')
                                                        
                                                        for _, row in df.iterrows():
                                                            columns = list(row.index)
                                                            values = [str(v) if pd.notna(v) else '' for v in row.values]
                                                            placeholders = ', '.join(['?' for _ in columns])
                                                            query = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                                                            manager.db.execute_update(query, values)
                                                        
                                                        restored_tables.append(f"{table_name} ({len(df)} records)")
                                                    except Exception:
                                                        st.error(f"Failed to restore sheet: {sheet_name}")
                                                
                                                if restored_tables:
                                                    st.success("✅ Data restored successfully!")
                                                    st.info("**Restored tables:**")
                                                    for table_info in restored_tables:
                                                        st.write(f"• {table_info}")
                                                else:
                                                    st.error("❌ No data was restored. Please check the backup file format.")
                                            
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"❌ Restore failed: {str(e)}")
                                            st.error("Please check that the backup file is valid and not corrupted.")

def streams_page():
    st.header("🏫 Class Stream Management")
    manager = st.session_state.school_manager
    
    # Stream Configuration Section
    with st.expander("🔧 Stream Configuration", expanded=False):
        st.subheader("Configure Streams per Grade")
        
        config_grade = st.selectbox("Select Grade to Configure", [
            "Kindergarten", "Grade 1", "Grade 2", "Grade 3", 
            "Grade 4", "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"
        ], key="config_grade_select")
        
        # Get current configuration
        current_streams = manager.stream_manager.get_streams_config_for_grade(config_grade)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Current streams for {config_grade}:** {', '.join(current_streams)}")
            
            # Stream configuration options
            stream_option = st.radio("Stream Configuration", [
                "Use Default (R, C, S)",
                "Custom Configuration"
            ], key="stream_config_option")
            
            if stream_option == "Custom Configuration":
                # Get available stream types
                available_types = manager.stream_manager.get_available_stream_types()
                suggested_types = ['A', 'B', 'C', 'D', 'E', 'R', 'C', 'S', 'Gold', 'Silver', 'Bronze']
                all_suggested = list(set(available_types + suggested_types))
                
                # Multi-select for stream types
                selected_streams = st.multiselect(
                    "Select Stream Types",
                    options=sorted(all_suggested),
                    default=current_streams,
                    help="Select stream types for this grade. Order will be preserved for performance-based assignment."
                )
                
                # Option to add custom stream type
                custom_stream = st.text_input("Add Custom Stream Type", placeholder="e.g., Alpha, Beta, Gamma")
                if custom_stream and st.button("Add Custom Stream"):
                    if custom_stream not in selected_streams:
                        selected_streams.append(custom_stream)
                        st.success(f"Added '{custom_stream}' to stream types")
                        st.rerun()
                
                # Show preview
                if selected_streams:
                    st.info(f"Preview: {config_grade} will have streams: {', '.join(selected_streams)}")
                    
                    if st.button("Apply Configuration"):
                        try:
                            manager.stream_manager.set_streams_for_grade(config_grade, selected_streams)
                            st.success(f"Stream configuration updated for {config_grade}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating configuration: {e}")
                else:
                    st.warning("Please select at least one stream type")
            
            else:  # Use Default
                if st.button("Reset to Default (R, C, S)"):
                    try:
                        manager.stream_manager.set_streams_for_grade(config_grade, ['R', 'C', 'S'])
                        st.success(f"Reset {config_grade} to default streams (R, C, S)!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error resetting configuration: {e}")
        
        with col2:
            st.info("""
            **Stream Types Guide:**
            - **R**: Regular/General
            - **C**: Commerce/Creative
            - **S**: Science/Special
            - **A, B, C**: Alphabetical
            - **1, 2, 3**: Numerical
            - Custom names allowed
            """)
    
    # Bulk stream creation
    st.subheader("🚀 Quick Setup")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✨ Initialize Streams for All Grades (Default)"):
            for grade in ["Kindergarten", "Grade 1", "Grade 2", "Grade 3", 
                          "Grade 4", "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"]:
                manager.stream_manager.create_streams_for_grade(grade)
            st.success("Default streams (R, C, S) created for all grades!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Recreate Streams with Current Config"):
            success_count = 0
            for grade in ["Kindergarten", "Grade 1", "Grade 2", "Grade 3", 
                          "Grade 4", "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"]:
                try:
                    manager.stream_manager.create_streams_for_grade(grade)
                    success_count += 1
                except Exception as e:
                    st.error(f"Error creating streams for {grade}: {e}")
            
            if success_count > 0:
                st.success(f"Streams created/updated for {success_count} grades using stored configurations!")
                st.rerun()
    
    st.divider()
    
    # Grade selection for management
    st.subheader("📋 Stream Management")
    grade_level = st.selectbox("Select Grade Level", [
        "Kindergarten", "Grade 1", "Grade 2", "Grade 3", 
        "Grade 4", "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"
    ], key="stream_grade_select")
    
    streams = manager.stream_manager.get_streams_for_grade(grade_level)
    stream_config = manager.stream_manager.get_streams_config_for_grade(grade_level)
    
    if not streams:
        st.warning(f"No streams exist for {grade_level}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Create Default Streams for {grade_level}"):
                manager.stream_manager.create_streams_for_grade(grade_level)
                st.rerun()
        with col2:
            if st.button(f"Create Configured Streams for {grade_level}"):
                manager.stream_manager.create_streams_for_grade(grade_level, stream_config)
                st.rerun()
        return
    
    # Display stream configuration info
    st.info(f"**{grade_level}** has **{len(streams)} streams**: {', '.join([s.stream_type for s in streams])} | Class size limit: {manager.stream_manager.max_class_size} students")
    
    # Stream management tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stream Overview", "👥 Student Assignment", "👨‍🏫 Teacher Assignment", "⚙️ Capacity Management"])
    
    with tab1:
        # Display streams
        for i, stream in enumerate(streams):
            student_count = manager.stream_manager.get_student_count(stream.stream_id)
            capacity_percent = min(100, (student_count / stream.max_capacity) * 100)
            
            with st.expander(f"{stream.stream_id} Stream ({student_count}/{stream.max_capacity} students) - {capacity_percent:.1f}% full"):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Capacity indicator
                    st.progress(capacity_percent / 100)
                    st.caption(f"Capacity: {student_count}/{stream.max_capacity} students")
                    
                    # Student list
                    students = manager.stream_manager.get_students_in_stream(stream.stream_id)
                    if students:
                        st.write("**Students:**")
                        for student in students[:10]:
                            st.write(f"- {student.name} (ID: {student.student_id})")
                        if len(students) > 10:
                            st.write(f"... and {len(students) - 10} more")
                    else:
                        st.info("No students assigned yet")
                
                with col2:
                    # Class teacher assignment (only for primary grades)
                    if "Grade" in grade_level or grade_level == "Kindergarten":
                        teachers = manager.db.execute_query(
                            "SELECT teacher_id, name FROM teachers WHERE status = 'active'"
                        )
                        teacher_options = {f"{name} ({teacher_id})": teacher_id for teacher_id, name in teachers}
                        
                        current_teacher = stream.class_teacher_id
                        current_teacher_name = next(
                            (name for tid, name in teachers if tid == current_teacher), 
                            "Not assigned"
                        )
                        
                        st.write(f"**Class Teacher:** {current_teacher_name}")
                        selected_teacher = st.selectbox(
                            "Assign Class Teacher", 
                            ["Select teacher"] + list(teacher_options.keys()),
                            key=f"teacher_{stream.stream_id}"
                        )
                        
                        if st.button("Assign Teacher", key=f"assign_{stream.stream_id}") and selected_teacher != "Select teacher":
                            teacher_id = teacher_options[selected_teacher]
                            manager.stream_manager.assign_class_teacher(stream.stream_id, teacher_id)
                            st.success("Teacher assigned!")
                            st.rerun()
                
                with col3:
                    # Stream management options
                    st.write("**Actions:**")
                    
                    # Delete stream option (only if more than 1 stream exists)
                    if len(streams) > 1:
                        if st.button("🗑️ Delete Stream", key=f"delete_{stream.stream_id}"):
                            if student_count > 0:
                                reassign_method = st.selectbox(
                                    "Reassignment method for students:",
                                    ["balanced", "performance", "random"],
                                    key=f"reassign_method_{stream.stream_id}"
                                )
                                if st.button("Confirm Delete & Reassign", key=f"confirm_delete_{stream.stream_id}"):
                                    try:
                                        manager.stream_manager.delete_stream(stream.stream_id, reassign_method)
                                        st.success(f"Stream {stream.stream_id} deleted and students reassigned!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting stream: {e}")
                            else:
                                if st.button("Confirm Delete", key=f"confirm_delete_empty_{stream.stream_id}"):
                                    try:
                                        manager.stream_manager.delete_stream(stream.stream_id)
                                        st.success(f"Stream {stream.stream_id} deleted!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting stream: {e}")
                    else:
                        st.info("Cannot delete the only stream")
    
    with tab2:
        # Student assignment controls
        st.subheader("Student Assignment")
        
        # Show current unassigned students for this grade
        try:
            unassigned_students = manager.db.execute_query(
                """SELECT student_id, name, grade 
                   FROM students 
                   WHERE grade = ? AND (stream_id IS NULL OR stream_id = '') AND status = 'active'""",
                (grade_level,)
            )
            
            assigned_students = manager.db.execute_query(
                """SELECT student_id, name, grade 
                   FROM students 
                   WHERE grade = ? AND stream_id IS NOT NULL AND stream_id != '' AND status = 'active'""",
                (grade_level,)
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if unassigned_students:
                    st.info(f"Found {len(unassigned_students)} unassigned students in {grade_level}")
                    
                    # Show first few unassigned students
                    if len(unassigned_students) <= 5:
                        st.write("**Unassigned students:**")
                        for student_id, name, grade in unassigned_students:
                            st.write(f"- {name} (ID: {student_id})")
                    else:
                        st.write(f"**Unassigned students:** {unassigned_students[0][1]}, {unassigned_students[1][1]}, and {len(unassigned_students)-2} others")
                else:
                    st.success(f"All students in {grade_level} are assigned to streams")
            
            with col2:
                if assigned_students:
                    st.info(f"Found {len(assigned_students)} assigned students in {grade_level}")
                    
                    # Initialize session state for confirmation
                    if f"confirm_reset_{grade_level}" not in st.session_state:
                        st.session_state[f"confirm_reset_{grade_level}"] = False
                    
                    # Reset/Un-assign button
                    if not st.session_state[f"confirm_reset_{grade_level}"]:
                        if st.button("🔄 Reset Student Assignments", key="reset_assignments", 
                                    type="secondary", help="This will unassign ALL students in this grade"):
                            st.session_state[f"confirm_reset_{grade_level}"] = True
                            st.rerun()
                    
                    # Show confirmation if button was clicked
                    if st.session_state[f"confirm_reset_{grade_level}"]:
                        st.warning(f"⚠️ This will unassign ALL {len(assigned_students)} students from their streams in {grade_level}!")
                        
                        col_confirm1, col_confirm2 = st.columns(2)
                        
                        with col_confirm1:
                            if st.button("✅ Confirm Reset", key="confirm_reset", type="primary"):
                                try:
                                    # Clear stream assignments for all students in this grade
                                    result = manager.db.execute_update(
                                        "UPDATE students SET stream_id = NULL WHERE grade = ? AND status = 'active'",
                                        (grade_level,)
                                    )
                                    
                                    st.success(f"Successfully unassigned all students from {grade_level} streams!")
                                    st.info("All students are now in the 'Unassigned' category and can be reassigned.")
                                    
                                    # Reset confirmation state
                                    st.session_state[f"confirm_reset_{grade_level}"] = False
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Error resetting student assignments: {e}")
                                    st.session_state[f"confirm_reset_{grade_level}"] = False
                        
                        with col_confirm2:
                            if st.button("❌ Cancel", key="cancel_reset"):
                                st.session_state[f"confirm_reset_{grade_level}"] = False
                                st.info("Reset cancelled.")
                                st.rerun()
                else:
                    st.info(f"No assigned students found in {grade_level}")
                    
        except Exception as e:
            st.error(f"Error checking student assignments: {e}")
        
        st.divider()
        
        assignment_method = st.radio("Assignment Method", 
            ["Performance-based", "Random", "Balanced"],
            index=2, key="stream_assignment_method",
            help="Performance-based: Assigns students based on GPA to appropriate streams. Random: Randomly distributes students. Balanced: Fills streams evenly."
        )
        
        method_map = {
            "Performance-based": "performance",
            "Random": "random", 
            "Balanced": "balanced"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Assign Unassigned Students ({assignment_method})", key="assign_unassigned"):
                try:
                    unassigned = manager.db.execute_query(
                        """SELECT student_id, name, grade 
                           FROM students 
                           WHERE grade = ? AND (stream_id IS NULL OR stream_id = '') AND status = 'active'""",
                        (grade_level,)
                    )
                    
                    if unassigned:
                        success_count = 0
                        error_count = 0
                        
                        for student_id, name, grade in unassigned:
                            try:
                                if hasattr(manager.stream_manager, 'assign_student_to_stream'):
                                    result = manager.stream_manager.assign_student_to_stream(
                                        student_id, method_map[assignment_method]
                                    )
                                    if result:
                                        success_count += 1
                                    else:
                                        error_count += 1
                                else:
                                    st.error("Stream assignment method not found")
                                    break
                            except Exception as e:
                                st.error(f"Error assigning {name}: {e}")
                                error_count += 1
                        
                        if success_count > 0:
                            st.success(f"Successfully assigned {success_count} students using {assignment_method} method!")
                        if error_count > 0:
                            st.warning(f"Failed to assign {error_count} students")
                        
                        if success_count > 0:
                            st.rerun()
                    else:
                        st.info("No unassigned students found for this grade")
                        
                except Exception as e:
                    st.error(f"Error during student assignment: {e}")
        
        with col2:
            if st.button(f"Reassign ALL Students ({assignment_method})", key="reassign_all"):
                try:
                    # First clear existing assignments for this grade
                    manager.db.execute_update(
                        "UPDATE students SET stream_id = NULL WHERE grade = ?",
                        (grade_level,)
                    )
                    
                    # Get all students for this grade
                    all_students = manager.db.execute_query(
                        "SELECT student_id, name, grade FROM students WHERE grade = ? AND status = 'active'",
                        (grade_level,)
                    )
                    
                    if all_students:
                        success_count = 0
                        error_count = 0
                        
                        for student_id, name, grade in all_students:
                            try:
                                if hasattr(manager.stream_manager, 'assign_student_to_stream'):
                                    result = manager.stream_manager.assign_student_to_stream(
                                        student_id, method_map[assignment_method]
                                    )
                                    if result:
                                        success_count += 1
                                    else:
                                        error_count += 1
                                else:
                                    st.error("Stream assignment method not found")
                                    break
                            except Exception as e:
                                st.error(f"Error reassigning {name}: {e}")
                                error_count += 1
                        
                        if success_count > 0:
                            st.success(f"Successfully reassigned {success_count} students using {assignment_method} method!")
                        if error_count > 0:
                            st.warning(f"Failed to reassign {error_count} students")
                        
                        if success_count > 0:
                            st.rerun()
                    else:
                        st.info("No students found for this grade")
                        
                except Exception as e:
                    st.error(f"Error during student reassignment: {e}")
        
        # Alternative: Simple Reset Button
        st.divider()
        st.subheader("🔄 Quick Reset Options")
        
        col_reset1, col_reset2 = st.columns(2)
        
        with col_reset1:
            # Simple reset with text confirmation
            reset_confirmation = st.text_input(
                f"Type 'RESET {grade_level}' to confirm unassigning all students:",
                key="reset_confirmation_text"
            )
            
            if st.button("Reset All Assignments", key="simple_reset", type="secondary"):
                if reset_confirmation == f"RESET {grade_level}":
                    try:
                        # Clear stream assignments for all students in this grade
                        result = manager.db.execute_update(
                            "UPDATE students SET stream_id = NULL WHERE grade = ? AND status = 'active'",
                            (grade_level,)
                        )
                        
                        st.success(f"✅ Successfully reset all student assignments for {grade_level}!")
                        st.info("All students are now unassigned and ready for re-assignment.")
                        
                        # Clear the confirmation text
                        st.session_state.reset_confirmation_text = ""
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error resetting assignments: {e}")
                else:
                    if reset_confirmation:
                        st.error(f"Please type exactly: RESET {grade_level}")
                    else:
                        st.warning("Please enter the confirmation text above")
        
        with col_reset2:
            # Show what will be reset
            try:
                current_assignments = manager.db.execute_query(
                    """SELECT s.name, s.stream_id 
                       FROM students s 
                       WHERE s.grade = ? AND s.stream_id IS NOT NULL AND s.stream_id != '' AND s.status = 'active'
                       ORDER BY s.stream_id, s.name""",
                    (grade_level,)
                )
                
                if current_assignments:
                    st.write(f"**Current assignments ({len(current_assignments)} students):**")
                    
                    # Group by stream
                    from collections import defaultdict
                    by_stream = defaultdict(list)
                    for name, stream_id in current_assignments:
                        by_stream[stream_id].append(name)
                    
                    # Show first few from each stream
                    for stream_id, students in by_stream.items():
                        if len(students) <= 3:
                            st.write(f"- **{stream_id}**: {', '.join(students)}")
                        else:
                            st.write(f"- **{stream_id}**: {students[0]}, {students[1]}, +{len(students)-2} more")
                else:
                    st.info("No students currently assigned to streams")
                    
            except Exception as e:
                st.error(f"Error loading current assignments: {e}")
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⚖️ Balance Class Sizes", key="balance_classes"):
                try:
                    if hasattr(manager.stream_manager, 'enforce_class_sizes'):
                        manager.stream_manager.enforce_class_sizes(grade_level)
                        st.success("Class sizes balanced across all streams!")
                        st.rerun()
                    else:
                        st.error("Class size balancing method not implemented")
                except Exception as e:
                    st.error(f"Error balancing class sizes: {e}")
        
        with col2:
            # Add stream option
            if st.button("➕ Add New Stream", key="add_stream"):
                current_config = manager.stream_manager.get_streams_config_for_grade(grade_level)
                new_stream_type = st.text_input("Enter new stream type (e.g., D, Alpha, etc.)", key="new_stream_input")
                
                if new_stream_type and st.button("Create Stream", key="create_new_stream"):
                    try:
                        # Add to configuration
                        updated_config = current_config + [new_stream_type]
                        manager.stream_manager.set_streams_for_grade(grade_level, updated_config)
                        
                        # Create the actual stream
                        manager.stream_manager.create_streams_for_grade(grade_level)
                        
                        st.success(f"Stream {new_stream_type} added to {grade_level}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding stream: {e}")

    with tab3:
        # Subject teacher assignment for all grades
        st.subheader("Subject Teacher Assignment")
        
        # Get all available subjects
        try:
            subjects = manager.db.execute_query("SELECT DISTINCT name FROM courses")
            subject_options = [subj[0] for subj in subjects] if subjects else []
        except Exception as e:
            st.error(f"Error fetching subjects: {e}")
            # Fall back to predefined subjects based on Ghanaian curriculum
            if "JHS" in grade_level:
                subject_options = [
                    "Mathematics", "English Language", "Science", "Social Studies", 
                    "Information Communication Technology", "Creative Arts", 
                    "Physical Education", "Ghanaian Language", "French"
                ]
            else:
                subject_options = [
                    "Mathematics", "English Language", "Science", "Social Studies",
                    "Creative Arts", "Physical Education", "Ghanaian Language"
                ]
        
        if subject_options:
            selected_subject = st.selectbox("Select Subject", subject_options)
            
            # Get available teachers for the selected subject
            try:
                subject_teachers = manager.db.execute_query(
                    "SELECT DISTINCT t.teacher_id, t.name FROM teachers t "
                    "JOIN courses c ON t.teacher_id = c.teacher_id "
                    "WHERE c.name = ? AND t.status = 'active'",
                    (selected_subject,)
                )
                # If no teachers found for this specific subject, get all active teachers
                if not subject_teachers:
                    subject_teachers = manager.db.execute_query(
                        "SELECT teacher_id, name FROM teachers WHERE status = 'active'"
                    )
            except Exception as e:
                # Fallback to all active teachers
                subject_teachers = manager.db.execute_query(
                    "SELECT teacher_id, name FROM teachers WHERE status = 'active'"
                )
            
            teacher_options = {f"{name} ({teacher_id})": teacher_id for teacher_id, name in subject_teachers}
            
            if teacher_options:
                selected_teacher = st.selectbox(
                    f"Select Teacher for {selected_subject}",
                    list(teacher_options.keys())
                )
                
                if st.button(f"Assign to All {grade_level} Streams"):
                    success_count = 0
                    for stream in streams:
                        try:
                            manager.stream_manager.assign_subject_teacher(
                                stream.stream_id,
                                selected_subject,
                                teacher_options[selected_teacher]
                            )
                            success_count += 1
                        except AttributeError:
                            # If assign_subject_teacher method doesn't exist, show a message
                            st.warning("Subject teacher assignment method not implemented yet")
                            break
                        except Exception as e:
                            st.error(f"Error assigning to {stream.stream_id}: {e}")
                    
                    if success_count > 0:
                        st.success(f"{selected_teacher} assigned to teach {selected_subject} in {success_count} {grade_level} streams!")
            else:
                st.warning(f"No teachers available for {selected_subject}")
        else:
            st.warning("No subjects found in the database")
#========================    
    with tab4:
            st.subheader("⚙️ Stream Capacity Management")
            
            # Global capacity settings section
            with st.expander("🌍 Global Capacity Settings", expanded=False):
                st.write("**Set default capacity for new streams across the school**")
                
                # Get current default capacity
                current_default = manager.stream_manager.default_max_capacity
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    new_default_capacity = st.number_input(
                        "Default Maximum Capacity for New Streams",
                        min_value=1,
                        max_value=100,
                        value=current_default,
                        step=1,
                        help="This will be used as the default capacity when creating new streams"
                    )
                    
                    if st.button("Update Global Default Capacity"):
                        try:
                            manager.stream_manager.set_global_max_capacity(new_default_capacity)
                            st.success(f"Global default capacity updated to {new_default_capacity} students!")
                            st.info("This will apply to all newly created streams. Existing streams are not affected.")
                        except Exception as e:
                            st.error(f"Error updating global capacity: {e}")
                
                with col2:
                    st.info(f"""
                    **Current Global Default:**
                    {current_default} students per stream
                    
                    **Recommended Ranges:**
                    - Kindergarten: 20-25
                    - Primary: 30-35
                    - JHS: 35-40
                    """)
            
            st.divider()
            
            # Grade-specific capacity management
            st.subheader(f"📋 Grade-Specific Capacity: {grade_level}")
            
            # Get current capacity configuration for selected grade
            try:
                grade_config = manager.stream_manager.get_streams_config_for_grade(grade_level)
                current_capacity = grade_config['max_capacity']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Current capacity for {grade_level}:** {current_capacity} students per stream")
                    
                    new_capacity = st.number_input(
                        f"Set Maximum Capacity for {grade_level}",
                        min_value=1,
                        max_value=100,
                        value=current_capacity,
                        step=1,
                        key=f"capacity_{grade_level}",
                        help=f"This will update the capacity for all streams in {grade_level}"
                    )
                    
                    # Show what will be affected
                    current_streams = manager.stream_manager.get_streams_for_grade(grade_level)
                    if current_streams:
                        st.info(f"This will affect {len(current_streams)} existing streams: {', '.join([s.stream_id for s in current_streams])}")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button(f"Update {grade_level} Capacity", key=f"update_capacity_{grade_level}"):
                            try:
                                manager.stream_manager.set_max_capacity_for_grade(grade_level, new_capacity)
                                st.success(f"Capacity for {grade_level} updated to {new_capacity} students!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating capacity: {e}")
                    
                    with col_btn2:
                        if st.button(f"Reset to Global Default ({current_default})", key=f"reset_capacity_{grade_level}"):
                            try:
                                manager.stream_manager.set_max_capacity_for_grade(grade_level, current_default)
                                st.success(f"Capacity for {grade_level} reset to global default ({current_default})!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error resetting capacity: {e}")
                
                with col2:
                    # Capacity recommendations based on grade level
                    if grade_level == "Kindergarten":
                        recommended = "20-25"
                        reason = "Younger children need more individual attention"
                    elif grade_level in ["Grade 1", "Grade 2", "Grade 3"]:
                        recommended = "25-30"
                        reason = "Lower primary requires smaller class sizes"
                    elif grade_level in ["Grade 4", "Grade 5", "Grade 6"]:
                        recommended = "30-35"
                        reason = "Upper primary can handle slightly larger classes"
                    else:  # JHS
                        recommended = "35-40"
                        reason = "Secondary students can work in larger groups"
                    
                    st.info(f"""
                    **Recommended for {grade_level}:**
                    {recommended} students
                    
                    **Reason:** {reason}
                    
                    **Current Setting:** {current_capacity}
                    """)
                    
                    # Show capacity impact
                    if current_streams:
                        total_students = sum(manager.stream_manager.get_student_count(s.stream_id) for s in current_streams)
                        total_capacity = len(current_streams) * new_capacity
                        utilization = (total_students / total_capacity * 100) if total_capacity > 0 else 0
                        
                        st.metric("Current Utilization", f"{utilization:.1f}%")
                        if utilization > 90:
                            st.warning("⚠️ High utilization - consider adding more streams")
                        elif utilization < 50:
                            st.info("ℹ️ Low utilization - streams have plenty of space")
            
            except Exception as e:
                st.error(f"Error loading capacity configuration: {e}")
            
            st.divider()
            
            # Individual stream capacity override
            st.subheader("🎯 Individual Stream Capacity Override")
            st.write("Override capacity for specific streams (advanced usage)")
            
            if streams:
                # Stream selection for individual override
                stream_options = {f"{s.stream_id} (Current: {s.max_capacity})": s.stream_id for s in streams}
                selected_stream_display = st.selectbox(
                    "Select Stream to Modify",
                    list(stream_options.keys()),
                    key="individual_stream_select"
                )
                
                if selected_stream_display:
                    selected_stream_id = stream_options[selected_stream_display]
                    selected_stream = next(s for s in streams if s.stream_id == selected_stream_id)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        current_students = manager.stream_manager.get_student_count(selected_stream_id)
                        
                        individual_capacity = st.number_input(
                            f"Capacity for {selected_stream_id}",
                            min_value=max(1, current_students),  # Can't set below current student count
                            max_value=100,
                            value=selected_stream.max_capacity,
                            step=1,
                            key="individual_capacity",
                            help=f"Cannot be set below current student count ({current_students})"
                        )
                        
                        if individual_capacity < current_students:
                            st.warning(f"⚠️ Cannot set capacity below current student count ({current_students})")
                        
                        if st.button("Update Individual Stream Capacity", disabled=individual_capacity < current_students):
                            try:
                                # Direct database update for individual stream
                                query = "UPDATE streams SET max_capacity = ? WHERE stream_id = ?"
                                manager.db.execute_update(query, (individual_capacity, selected_stream_id))
                                st.success(f"Capacity for {selected_stream_id} updated to {individual_capacity}!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating individual stream capacity: {e}")
                    
                    with col2:
                        utilization = (current_students / selected_stream.max_capacity * 100) if selected_stream.max_capacity > 0 else 0
                        new_utilization = (current_students / individual_capacity * 100) if individual_capacity > 0 else 0
                        
                        st.metric("Current Students", current_students)
                        st.metric("Current Capacity", selected_stream.max_capacity)
                        st.metric("Current Utilization", f"{utilization:.1f}%")
                        
                        if individual_capacity != selected_stream.max_capacity:
                            st.write("**After Update:**")
                            st.metric("New Utilization", f"{new_utilization:.1f}%")
            else:
                st.info("No streams available for individual modification")
            
            st.divider()
            
            # Bulk capacity operations
            st.subheader("🔧 Bulk Capacity Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Apply Recommended Capacities**")
                st.write("Set all grade levels to their recommended capacity ranges")
                
                if st.button("Apply All Recommended Capacities"):
                    try:
                        recommendations = {
                            "Kindergarten": 25,
                            "Grade 1": 28, "Grade 2": 28, "Grade 3": 28,
                            "Grade 4": 32, "Grade 5": 32, "Grade 6": 32,
                            "JHS 1": 38, "JHS 2": 38, "JHS 3": 38
                        }
                        
                        updated_count = 0
                        for grade, capacity in recommendations.items():
                            try:
                                manager.stream_manager.set_max_capacity_for_grade(grade, capacity)
                                updated_count += 1
                            except Exception as e:
                                st.error(f"Error updating {grade}: {e}")
                        
                        if updated_count > 0:
                            st.success(f"Updated capacity for {updated_count} grade levels to recommended values!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error applying recommended capacities: {e}")
            
            with col2:
                st.write("**Standardize All Capacities**")
                standard_capacity = st.number_input(
                    "Set same capacity for all grades",
                    min_value=1,
                    max_value=100,
                    value=current_default,
                    step=1,
                    key="standard_capacity"
                )
                
                if st.button("Apply to All Grades"):
                    try:
                        grades = ["Kindergarten", "Grade 1", "Grade 2", "Grade 3", 
                                 "Grade 4", "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"]
                        
                        updated_count = 0
                        for grade in grades:
                            try:
                                manager.stream_manager.set_max_capacity_for_grade(grade, standard_capacity)
                                updated_count += 1
                            except Exception as e:
                                st.error(f"Error updating {grade}: {e}")
                        
                        if updated_count > 0:
                            st.success(f"Standardized capacity to {standard_capacity} for {updated_count} grade levels!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error standardizing capacities: {e}")
            
            st.divider()
            
            # Capacity overview table
            st.subheader("📊 Capacity Overview - All Grades")
            
            try:
                # Get capacity information for all grades
                grades = ["Kindergarten", "Grade 1", "Grade 2", "Grade 3", 
                         "Grade 4", "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"]
                
                capacity_data = []
                for grade in grades:
                    try:
                        config = manager.stream_manager.get_streams_config_for_grade(grade)
                        streams_in_grade = manager.stream_manager.get_streams_for_grade(grade)
                        
                        total_students = sum(manager.stream_manager.get_student_count(s.stream_id) for s in streams_in_grade)
                        total_capacity = len(streams_in_grade) * config['max_capacity'] if streams_in_grade else 0
                        utilization = (total_students / total_capacity * 100) if total_capacity > 0 else 0
                        
                        capacity_data.append({
                            'Grade': grade,
                            'Streams': len(streams_in_grade),
                            'Capacity per Stream': config['max_capacity'],
                            'Total Capacity': total_capacity,
                            'Current Students': total_students,
                            'Utilization (%)': f"{utilization:.1f}%",
                            'Available Spots': max(0, total_capacity - total_students)
                        })
                    except Exception as e:
                        capacity_data.append({
                            'Grade': grade,
                            'Streams': 0,
                            'Capacity per Stream': 'Error',
                            'Total Capacity': 'Error',
                            'Current Students': 'Error',
                            'Utilization (%)': 'Error',
                            'Available Spots': 'Error'
                        })
                
                # Display as dataframe
                import pandas as pd
                df_capacity = pd.DataFrame(capacity_data)
                st.dataframe(df_capacity, hide_index=True, width="stretch")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate totals (excluding error rows)
                valid_data = [row for row in capacity_data if isinstance(row['Current Students'], int)]
                
                total_students_all = sum(row['Current Students'] for row in valid_data)
                total_capacity_all = sum(row['Total Capacity'] for row in valid_data)
                total_streams_all = sum(row['Streams'] for row in valid_data)
                overall_utilization = (total_students_all / total_capacity_all * 100) if total_capacity_all > 0 else 0
                
                with col1:
                    st.metric("Total Students", total_students_all)
                with col2:
                    st.metric("Total Capacity", total_capacity_all)
                with col3:
                    st.metric("Total Streams", total_streams_all)
                with col4:
                    st.metric("Overall Utilization", f"{overall_utilization:.1f}%")
            
            except Exception as e:
                st.error(f"Error loading capacity overview: {e}")
            
            # Capacity enforcement section
            st.divider()
            st.subheader("⚖️ Capacity Enforcement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Check for Overcapacity Issues**")
                if st.button("Scan for Overcapacity Streams"):
                    try:
                        overcapacity_streams = []
                        
                        # Check all streams for overcapacity
                        all_grades = ["Kindergarten", "Grade 1", "Grade 2", "Grade 3", 
                                     "Grade 4", "Grade 5", "Grade 6", "JHS 1", "JHS 2", "JHS 3"]
                        
                        for grade in all_grades:
                            streams_in_grade = manager.stream_manager.get_streams_for_grade(grade)
                            for stream in streams_in_grade:
                                student_count = manager.stream_manager.get_student_count(stream.stream_id)
                                if student_count > stream.max_capacity:
                                    overcapacity_streams.append({
                                        'stream_id': stream.stream_id,
                                        'grade': grade,
                                        'current': student_count,
                                        'capacity': stream.max_capacity,
                                        'excess': student_count - stream.max_capacity
                                    })
                        
                        if overcapacity_streams:
                            st.warning(f"Found {len(overcapacity_streams)} overcapacity streams:")
                            for stream_info in overcapacity_streams:
                                st.write(f"- **{stream_info['stream_id']}**: {stream_info['current']}/{stream_info['capacity']} ({stream_info['excess']} excess)")
                        else:
                            st.success("✅ All streams are within capacity limits!")
                            
                    except Exception as e:
                        st.error(f"Error scanning for overcapacity: {e}")
            
            with col2:
                st.write("**Auto-Rebalance Overcapacity**")
                if st.button("Enforce Capacity Limits"):
                    try:
                        # Use the existing enforce_class_sizes method
                        manager.stream_manager.enforce_class_sizes()
                        st.success("✅ Capacity limits enforced! Excess students have been rebalanced.")
                        st.info("Students from overcapacity streams have been moved to available streams in the same grade.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error enforcing capacity limits: {e}")
        
        
            # Stream Statistics
            st.divider()
            st.subheader("📊 Stream Statistics")
            
            try:
                # Show statistics for current grade
                total_students = manager.db.execute_query(
                    "SELECT COUNT(*) FROM students WHERE grade = ? AND status = 'active'",
                    (grade_level,)
                )[0][0]
                
                assigned_students = manager.db.execute_query(
                    "SELECT COUNT(*) FROM students WHERE grade = ? AND stream_id IS NOT NULL AND stream_id != '' AND status = 'active'",
                    (grade_level,)
                )[0][0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Students", total_students)
                with col2:
                    st.metric("Assigned Students", assigned_students)
                with col3:
                    st.metric("Unassigned Students", total_students - assigned_students)
                with col4:
                    st.metric("Number of Streams", len(streams))
                
                # Show per-stream breakdown
                if streams:
                    st.write("**Per-Stream Breakdown:**")
                    stream_data = []
                    for stream in streams:
                        student_count = manager.stream_manager.get_student_count(stream.stream_id)
                        utilization = (student_count / stream.max_capacity) * 100
                        stream_data.append({
                            'Stream': stream.stream_id,
                            'Type': stream.stream_type,
                            'Students': student_count,
                            'Capacity': stream.max_capacity,
                            'Utilization': f"{utilization:.1f}%"
                        })
                    
                    # Display as a table
                    import pandas as pd
                    df = pd.DataFrame(stream_data)
                    st.dataframe(df, hide_index=True)
                    
            except Exception as e:
                st.error(f"Error loading statistics: {e}")
            
            # Global stream statistics
            with st.expander("🌍 School-wide Stream Statistics"):
                try:
                    stats = manager.stream_manager.get_stream_statistics()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Streams by Grade Level:**")
                        for grade, count in stats['streams_by_grade'].items():
                            st.write(f"- {grade}: {count} streams")
                    
                    with col2:
                        st.write("**Streams by Type:**")
                        for stream_type, count in stats['streams_by_type'].items():
                            st.write(f"- {stream_type}: {count} streams")
                    
                    # Capacity utilization chart
                    if stats['capacity_utilization']:
                        st.write("**Capacity Utilization by Stream:**")
                        utilization_data = []
                        for item in stats['capacity_utilization']:
                            utilization_data.append({
                                'Stream': item['stream_id'],
                                'Grade': item['grade'],
                                'Utilization %': item['utilization_percent']
                            })
                        
                        df_util = pd.DataFrame(utilization_data)
                        st.bar_chart(df_util.set_index('Stream')['Utilization %'])
                
                except Exception as e:
                    st.error(f"Error loading global statistics: {e}")


# Additional utility functions
def export_data():
    """Export school data to various formats"""
    manager = st.session_state.school_manager
    students = manager.get_all_students()
    
    if students:
        df = pd.DataFrame(students)
        
        # Create download buttons for different formats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name="students_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Convert to Excel
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name="students_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name="students_data.json",
                mime="application/json"
            )

def backup_restore():
    """Backup and restore functionality"""
    st.subheader("💾 Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Backup")
        if st.button("🔄 Create Backup"):
            manager = st.session_state.school_manager
            backup_filename = f"school_backup_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.db"
            
            try:
                # Create backup of current database
                backup_data = {
                    'students': manager.get_all_students(),
                    'backup_date': datetime.now().isoformat(),
                    'version': '1.0'
                }
                
                # Convert to JSON for download
                backup_json = json.dumps(backup_data, indent=2)
                
                st.download_button(
                    label="💾 Download Backup File",
                    data=backup_json,
                    file_name=backup_filename.replace('.db', '.json'),
                    mime="application/json"
                )
                
                st.success("✅ Backup created successfully!")
                
            except Exception as e:
                st.error(f"❌ Backup failed: {str(e)}")
    
    with col2:
        st.write("### Restore")
        uploaded_file = st.file_uploader("Choose backup file", type=['json', 'db'])
        
        if uploaded_file:
            if st.button("📥 Restore from Backup"):
                try:
                    # Read backup file
                    backup_content = uploaded_file.read().decode('utf-8')
                    backup_data = json.loads(backup_content)
                    
                    # Validate backup structure
                    if 'students' in backup_data:
                        manager = st.session_state.school_manager
                        
                        # Clear existing data and restore
                        for student_data in backup_data['students']:
                            manager.add_student(
                                student_data['name'],
                                student_data['student_id'],
                                student_data['grade'],
                                student_data['subjects']
                            )
                        
                        st.success("✅ Data restored successfully!")
                        st.experimental_rerun()
                        
                    else:
                        st.error("❌ Invalid backup file format")
                        
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format in backup file")
                except Exception as e:
                    st.error(f"❌ Restore failed: {str(e)}")

# Run the application
if __name__ == "__main__":
    main()