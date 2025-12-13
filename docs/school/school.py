import pickle
import streamlit as st
import pandas as pd
import sqlite3
import requests
import json
import base64
from datetime import datetime, timedelta, date
import hashlib
import os
import io
from io import BytesIO, StringIO
import plotly.express as px
from functools import lru_cache
import time
import re
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import random
from typing import Dict, Optional, List, Tuple
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Spacer, Image
# Import chatX.py components
from chatX import get_ollama_models, is_vision_model, is_qwen3_model, split_message, OLLAMA_HOST, get_ollama_models_cached, get_ollama_host
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Image as PlatypusImage  # Add this import for PlatypusImage
import openai  # NEW IMPORT

# Version tracking
APP_VERSION = "1.8.0"
#warnings.filterwarnings('ignore')

st.set_page_config(page_title="RadioSport School", page_icon="🧟", layout="wide", 
                   menu_items={'Report a Bug': "https://github.com/rkarikari/stem",
                              'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."})


# Database setup
DB_NAME = "school_admin.db"

# Performance settings
CACHE_TTL = 300  # Cache TTL in seconds (5 minutes)
MAX_RECORDS = 10000  # Limit for displayed records

# Email validation regex (more permissive for subdomains and TLDs)
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?$')

# CSS for professional sleek look
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
}
.main-title {
    font-size: 28px !important;
    font-weight: 700;
    color: #1a3c5e;
}
.sidebar-title {
    font-size: 22px !important;
    font-weight: bold;
    color: #1a3c5e;
    margin-bottom: 10px !important;
}
.section-header {
    font-size: 20px;
    font-weight: 600;
    color: #2c5282;
    margin-top: 20px;
    margin-bottom: 10px;
}
.stButton>button {
    background-color: #2b6cb0;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
}
.stButton>button:hover {
    background-color: #2c5282;
}
.dataframe {
    border-radius: 8px;
    overflow: hidden;
}
.card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.cache-stats {
    font-size: 12px;
    color: #666;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 6px;
}
.student-row {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.ai-status {
    padding: 8px;
    border-radius: 6px;
    margin: 8px 0;
}
.ai-connected {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.ai-disconnected {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
.email-status {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.email-success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.email-error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
.db-connection {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    background-color: #e6f7ff;
    border: 1px solid #91d5ff;
}
.column-mapping {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
}
.db-field-table {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
}
.db-field-row {
    display: flex;
    margin-bottom: 8px;
    align-items: center;
}
.db-field-input {
    flex: 1;
    margin-right: 10px;
}
.action-button {
    margin: 0 3px;
    padding: 4px 8px;
}
.student-report {
    background-color: #f0f9ff;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}
.financial-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.financial-metric {
    font-size: 24px;
    font-weight: 700;
    margin-top: 10px;
}
.financial-label {
    font-size: 14px;
    color: #4a5568;
}
.positive-value {
    color: #38a169;
}
.negative-value {
    color: #e53e3e;
}
/* Add these styles from chatX.py */
.thinking-display {
    position: fixed;
    top: 10px;
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
    max-height: 250px;
    overflow-y: auto;
    line-height: 1.4;
    word-wrap: break-word;
}
.thinking-content::-webkit-scrollbar {
    width: 4px;
}
.thinking-content::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 2px;
}
.thinking-content::-webkit-scrollbar-thumb {
    background: #007bff;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# NEW CONSTANTS
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = [
    "moonshotai/kimi-dev-72b:free",
    "openrouter/auto",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
    "mistralai/mistral-7b-instruct",
    "google/gemma-7b-it",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4-turbo",
    "meta-llama/llama-3-70b-instruct"
]

# Database initialization
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Students table (UPDATED)
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            grade TEXT,                -- CHANGED: From INTEGER to TEXT to support streams like "1R", "1C"
            room TEXT,
            dob DATE,
            phone TEXT,
            address TEXT,
            enrollment_date DATE,
            email TEXT,
            parent_name TEXT,
            medical_notes TEXT,
            scholarship_status TEXT,
            status TEXT DEFAULT 'Active',  -- NEW: Added status column for graduation tracking
            FOREIGN KEY (room) REFERENCES classes(room) 
        )
    ''')

    # Courses table (UPDATED)
    c.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            teacher_id INTEGER,
            term_id INTEGER,
            schedule TEXT,
            min_promotion_score REAL DEFAULT 60.0,  -- UPDATED: Added default value
            FOREIGN KEY (teacher_id) REFERENCES staff(id),
            FOREIGN KEY (term_id) REFERENCES terms(id)
        )
    ''')

    # Enrollments table
    c.execute('''
        CREATE TABLE IF NOT EXISTS enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            course_id INTEGER,
            enrollment_date DATE,
            status TEXT DEFAULT 'Active',  -- UPDATED: Added default status
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (course_id) REFERENCES courses(id)
        )
    ''')

    # Behavior table
    c.execute('''
        CREATE TABLE IF NOT EXISTS behavior (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date DATE,
            incident_type TEXT,
            description TEXT,
            action_taken TEXT,
            resolved BOOLEAN DEFAULT 0,  -- UPDATED: Added default value
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')

    # Library table
    c.execute('''
        CREATE TABLE IF NOT EXISTS library (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_title TEXT NOT NULL,
            author TEXT,
            isbn TEXT,
            category TEXT,
            status TEXT DEFAULT 'Available',  -- UPDATED: Added default status
            borrower_id INTEGER,
            checkout_date DATE,
            due_date DATE,
            FOREIGN KEY (borrower_id) REFERENCES students(id)
        )
    ''')
    
    # Staff table
    c.execute('''
        CREATE TABLE IF NOT EXISTS staff (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            staff_id INTEGER,
            name TEXT NOT NULL,
            role TEXT,
            department TEXT,
            phone TEXT,
            hire_date DATE,
            salary REAL
        )
    ''')
  
    # Classes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            room TEXT UNIQUE,
            teacher_id INTEGER,
            term_id INTEGER,
            FOREIGN KEY (teacher_id) REFERENCES staff(id),
            FOREIGN KEY (term_id) REFERENCES terms(id)
        )
    ''')   

    # Class_Members table
    c.execute('''
        CREATE TABLE IF NOT EXISTS class_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            term_id INTEGER NOT NULL,
            UNIQUE(student_id, term_id),
            FOREIGN KEY (class_id) REFERENCES classes(id),
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (term_id) REFERENCES terms(id)
        )
    ''')     
    
    # Attendance table
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date DATE,
            status TEXT DEFAULT 'Present',  -- UPDATED: Added default status
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    
    # Test results table
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            subject TEXT,
            score REAL,
            test_date DATE,
            test_type TEXT,  -- NEW: Added test type field
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    
    # Fees table
    c.execute('''
        CREATE TABLE IF NOT EXISTS fees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            amount REAL,
            due_date DATE,
            status TEXT DEFAULT 'Pending',  -- UPDATED: Added default status
            payment_date DATE,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    
    # Expenditures table
    c.execute('''
        CREATE TABLE IF NOT EXISTS expenditures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            description TEXT,
            amount REAL NOT NULL,
            date DATE NOT NULL,
            vendor TEXT,
            notes TEXT
        )
    ''')
    
    # Terms table
    c.execute('''
        CREATE TABLE IF NOT EXISTS terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            status TEXT DEFAULT 'Active'  -- NEW: Added status for terms
        )
    ''')
    
    # Student_reports table
    c.execute('''
        CREATE TABLE IF NOT EXISTS student_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            term_id INTEGER NOT NULL,
            report_content TEXT NOT NULL,
            generated_date DATE NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (term_id) REFERENCES terms(id)
        )
    ''')
    
    # Lunch Payments table
    c.execute('''
        CREATE TABLE IF NOT EXISTS lunch_payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            payment_date DATE NOT NULL,
            status TEXT CHECK( status IN ('paid','unpaid') ) NOT NULL DEFAULT 'unpaid',
            term_id INTEGER,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (term_id) REFERENCES terms(id)
        )
    ''')
    
    # NEW: Promotion History table to track promotions
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS promotion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            from_term_id INTEGER NOT NULL,
            to_term_id INTEGER NOT NULL,
            from_grade INTEGER NOT NULL,
            to_grade INTEGER NOT NULL,
            action TEXT NOT NULL, -- 'Promote', 'Repeat', 'Transfer', 'Graduate'
            promotion_date DATE NOT NULL,
            notes TEXT,
            FOREIGN KEY (student_id) REFERENCES students (id),
            FOREIGN KEY (from_term_id) REFERENCES terms (id),
            FOREIGN KEY (to_term_id) REFERENCES terms (id)
        )
    ''');

    c.execute('''
        CREATE TABLE IF NOT EXISTS promotion_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_grade INTEGER NOT NULL,
            to_grade INTEGER NOT NULL,
            minimum_score REAL,
            required_subjects TEXT, -- JSON array of subject requirements
            auto_promote BOOLEAN DEFAULT FALSE,
            created_date DATE DEFAULT CURRENT_DATE
        )
    ''');
    
    conn.commit()
    conn.close()


# Get table schema
def get_table_schema(table_name):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    schema = c.fetchall()
    conn.close()
    return schema

# Add column to table
def add_column_to_table(table_name, column_name, column_type):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Use direct string formatting for table/column names
        # (safe in this context since it's an admin tool)
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()
        return True, f"Column '{column_name}' added to '{table_name}'"
    except sqlite3.OperationalError as e:
        return False, str(e)
    finally:
        conn.close()

# Rename column in table
def rename_column(table_name, old_name, new_name):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # SQLite doesn't support RENAME COLUMN directly, so we need to recreate the table
        c.execute(f"PRAGMA table_info({table_name})")
        columns = c.fetchall()
        
        # Create new table name
        temp_table = f"{table_name}_temp"
        
        # Create new table with renamed column
        create_sql = f"CREATE TABLE {temp_table} ("
        for col in columns:
            name = new_name if col[1] == old_name else col[1]
            create_sql += f"{name} {col[2]}, "
        create_sql = create_sql.rstrip(", ") + ")"
        
        c.execute(create_sql)
        
        # Copy data
        c.execute(f"INSERT INTO {temp_table} SELECT * FROM {table_name}")
        
        # Drop old table
        c.execute(f"DROP TABLE {table_name}")
        
        # Rename new table
        c.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
        
        conn.commit()
        return True, f"Column '{old_name}' renamed to '{new_name}' in '{table_name}'"
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        conn.close()

# Delete column from table
def delete_column(table_name, column_name):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
        c.execute(f"PRAGMA table_info({table_name})")
        columns = c.fetchall()
        
        # Filter out the column to delete
        remaining_cols = [col for col in columns if col[1] != column_name]
        
        # Create new table name
        temp_table = f"{table_name}_temp"
        
        # Create new table without the column
        create_sql = f"CREATE TABLE {temp_table} ("
        for col in remaining_cols:
            create_sql += f"{col[1]} {col[2]}, "
        create_sql = create_sql.rstrip(", ") + ")"
        
        c.execute(create_sql)
        
        # Copy data (only remaining columns)
        cols_str = ", ".join([col[1] for col in remaining_cols])
        c.execute(f"INSERT INTO {temp_table} ({cols_str}) SELECT {cols_str} FROM {table_name}")
        
        # Drop old table
        c.execute(f"DROP TABLE {table_name}")
        
        # Rename new table
        c.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
        
        conn.commit()
        return True, f"Column '{column_name}' deleted from '{table_name}'"
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        conn.close()

# Reasoning Window functions 
def show_reasoning_window():
    """Initialize and show the reasoning window"""
    if not st.session_state.reasoning_window:
        st.session_state.reasoning_window = st.empty()
    
    st.session_state.reasoning_window.markdown(
        '''
        <div id="thinking-display" class="thinking-display visible">
            <div class="thinking-header">🤔 Reasoning Process</div>
            <div class="thinking-content">Thinking...</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

def update_reasoning_window(content):
    """Update the reasoning window with new content"""
    if st.session_state.reasoning_window:
        if len(content) > 1500:
            content = "..." + content[-1500:]
        
        st.session_state.reasoning_window.markdown(
            f'''
            <div id="thinking-display" class="thinking-display visible">
                <div class="thinking-header">🤔 Reasoning Process</div>
                <div class="thinking-content">{content}</div>
            </div>
            <style>
                .thinking-content {{
                    animation: scrollToBottom 0.1s ease-out;
                }}
                @keyframes scrollToBottom {{
                    to {{ scroll-behavior: smooth; }}
                }}
            </style>
            ''',
            unsafe_allow_html=True
        )

def hide_reasoning_window():
    """Hide the reasoning window"""
    if st.session_state.reasoning_window:
        st.session_state.reasoning_window.empty()
        st.session_state.reasoning_window = None

# Cache database queries
@lru_cache(maxsize=128) # Cache for 5 minutes
def query_db(query: str, params: tuple = ()) -> Tuple[List[Tuple], List[str]]:
    """Execute a SQL query and return results and column names."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(query, params)
        results = c.fetchall()
        columns = [desc[0] for desc in c.description] if c.description else []
        conn.close()
        
        st.session_state.cache_stats["db_cache_hits"] += 1
        return results, columns
    except sqlite3.Error as e:
        st.session_state.cache_stats["db_cache_misses"] += 1
        st.error(f"Database query error: {str(e)}")
        return [], []
    except Exception as e:
        st.session_state.cache_stats["db_cache_misses"] += 1
        st.error(f"Unexpected error: {str(e)}")
        return [], []
        
def get_db_context():
    context = []
    tables = ["students", "staff", "attendance", "test_results", "fees",
            "expenditures", "terms", "lunch_payments", "classes", "class_members"]
    
    for table in tables:
        try:
            # Get table schema
            schema = get_table_schema(table)
            schema_str = ", ".join([f"{col[1]} ({col[2]})" for col in schema])
            
            # Get row count
            count_query = f"SELECT COUNT(*) FROM {table}"
            count_result, _ = query_db(count_query)
            total_count = count_result[0][0] if count_result else 0
            
            # Get table summary based on table type
            summary = ""
            if table == "students":
                results, cols = query_db("SELECT grade, COUNT(*) as count FROM students GROUP BY grade")
                summary = f"Grade distribution: {', '.join([f'{row[0]}:{row[1]}' for row in results])}"
            elif table == "staff":
                results, cols = query_db("SELECT role, COUNT(*) as count FROM staff GROUP BY role")
                summary = f"Roles: {', '.join([f'{row[0]}:{row[1]}' for row in results])}"
            elif table == "fees":
                results, cols = query_db("SELECT status, SUM(amount) as total FROM fees GROUP BY status")
                summary = f"Fee status: {', '.join([f'{row[0]}:₡{row[1]:.2f}' for row in results])}"
            elif table == "test_results":
                results, cols = query_db("SELECT subject, AVG(score) as avg_score FROM test_results GROUP BY subject")
                summary = f"Subject averages: {', '.join([f'{row[0]}:{row[1]:.1f}' for row in results])}"
            elif table == "attendance":
                results, cols = query_db("SELECT status, COUNT(*) as count FROM attendance GROUP BY status")
                summary = f"Attendance status: {', '.join([f'{row[0]}:{row[1]}' for row in results])}"
            elif table == "expenditures":
                results, cols = query_db("SELECT category, SUM(amount) as total FROM expenditures GROUP BY category")
                summary = f"Expenditures: {', '.join([f'{row[0]}:₡{row[1]:.2f}' for row in results])}"
            elif table == "terms":
                results, cols = query_db("SELECT name, start_date, end_date FROM terms ORDER BY start_date DESC")
                summary = f"Terms: {', '.join([row[0] for row in results])}"
            
            # Get sample data (more samples for smaller tables)
            sample_size = 25 if total_count < 100 else 0
            sample_query = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_size}"
            sample_results, sample_cols = query_db(sample_query)
            
            # Format sample data
            sample_str = ""
            if sample_results:
                sample_df = pd.DataFrame(sample_results, columns=sample_cols)
                sample_str = f"\nSample data:\n{sample_df.to_markdown(index=False)}"
            
            context.append(
                f"### {table.upper()} TABLE\n"
                f"- Schema: {schema_str}\n"
                f"- Total records: {total_count}\n"
                f"- Summary: {summary}{sample_str}\n"
            )
        except Exception as e:
            context.append(f"⚠️ Error processing {table}: {str(e)}")
    
    return "\n".join(context)

# Principal Name
def get_principal_name():
    """Get the name of the principal from staff table"""
    try:
        query = """
            SELECT name 
            FROM staff 
            WHERE role IN ('Principal', 'Headmaster', 'Headmistress')
            LIMIT 1
        """
        results, _ = query_db(query)
        return results[0][0] if results else "School Principal"
    except:
        return "School Principal"

def get_teacher_for_student(student_grade):
    try:
        query = """
            SELECT s.name 
            FROM staff s
            JOIN classes c ON c.teacher_id = s.id
            WHERE c.room = ?
            LIMIT 1
        """
        results, _ = query_db(query, (student_grade,))
        return results[0][0] if results else "Not assigned"
    except:
        return "Error fetching"

# New function for SQL execution
def execute_sql(query: str) -> Tuple[Optional[List[Tuple]], Optional[List[str]], Optional[str]]:
    """Execute SQL query and return results, columns, or error message"""
    try:
        # Validate query - allow whitespace before SELECT
        cleaned_query = query.strip().rstrip(';')
        if not re.match(r'^\s*SELECT', cleaned_query, re.IGNORECASE):
            return None, None, "Only SELECT queries are allowed"
        
        # Check for forbidden keywords
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE']
        if any(f.upper() in cleaned_query.upper() for f in forbidden):
            return None, None, "Query contains forbidden operation"
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(cleaned_query)
        results = c.fetchall()
        columns = [desc[0] for desc in c.description] if c.description else []
        return results, columns, None
    except sqlite3.Error as e:
        return None, None, f"SQL Error: {str(e)}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"
    finally:
        conn.close()


# Generate PDF reports
def generate_report_pdf(report_title, student_name, term, student_info, performance_table, report_content, graph_path=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    content_style = styles["BodyText"]
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=1,  # centered
        spaceAfter=20
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=18,
        alignment=1,
        spaceAfter=40
    )
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=10,
        spaceAfter=10
    )
    
    elements = []
    
    # Title Page
    elements.append(Paragraph(report_title, title_style))
    elements.append(Paragraph(f"Student: {student_name}", subtitle_style))
    elements.append(Paragraph(f"Term: {term}", subtitle_style))
    elements.append(Paragraph(f"Generated: {date.today().strftime('%Y-%m-%d')}", subtitle_style))
    elements.append(PageBreak())
    
    # Student Information
    elements.append(Paragraph("Student Information", heading2_style))
    
    teacher_name = get_teacher_for_student(student_info['grade'])
    
    # Create student info table
    student_info_data = [
        ["Name", student_info['name']],
        ["Grade", student_info['grade']],
        ["Class Teacher", teacher_name],
        ["Overall Position", f"{student_info.get('overall_position', 'N/A')}"]
    ]
    student_info_table = Table(student_info_data, colWidths=[80, 300])
    student_info_table.setStyle(TableStyle([
        ('FONT', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (0,-1), 'RIGHT'),
        ('ALIGN', (1,0), (1,-1), 'LEFT'),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
    ]))
    elements.append(student_info_table)
    elements.append(Spacer(1, 10))
    
    # Academic Performance
    if performance_table and len(performance_table) > 0:
        try:
            elements.append(Paragraph("Academic Performance", heading2_style))
    
            # Create performance table
            performance_data = [["Subject", "Score", "Class Average", "Position in Class", "Remarks"]] + performance_table
            perf_table = Table(performance_data, repeatRows=1)
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(perf_table)
            elements.append(Spacer(1, 10))
        except Exception as e:
            print(f"Error creating performance table: {str(e)}")
    
    # Performance graph
    if graph_path and os.path.exists(graph_path):
        try:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Performance Over Time", heading2_style))
            
            # Create an Image object from the file path
            img = PlatypusImage(graph_path, width=500, height=280)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 12))
        except Exception as e:
            print(f"Error including performance graph: {str(e)}")
    
    # Report Content
    elements.append(Paragraph("Report Summary", heading2_style))

    for line in report_content.split('\n'):
        if line.strip():
            elements.append(Paragraph(line, content_style))
            elements.append(Spacer(1, 6))
    
    # Add principal's signature section at the end - commented out as it's duplicating AI gen..
    #elements.append(Spacer(1, 20))
    #elements.append(Paragraph("Best regards,", content_style))
    #elements.append(Spacer(1, 5))
    #elements.append(Paragraph(f"{get_principal_name()}", styles['Heading3']))
    #elements.append(Paragraph("School Head", styles['BodyText']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer


def call_openrouter_api(messages, model, api_key, max_retries=3):
    """Calls OpenRouter API for chat completions with rate limiting and retry logic"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rkarikari/RadioSport-chat",
        "X-Title": "RadioSport AI"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
    for attempt in range(max_retries):
        try:
            # Add exponential backoff delay
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0, 1)
                st.info(f"Rate limited. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    st.warning(f"Rate limit hit. Waiting {retry_after} seconds before retry...")
                    time.sleep(retry_after)
                    continue
                else:
                    st.error("Rate limit exceeded. Please wait a few minutes before trying again.")
                    return None
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                continue  # Will be handled above
            elif response.status_code == 403:
                st.error("API key invalid or insufficient credits.")
                return None
            elif response.status_code >= 500:
                if attempt < max_retries - 1:
                    st.warning(f"Server error {response.status_code}. Retrying...")
                    continue
                else:
                    st.error(f"Server error {response.status_code}. Please try again later.")
                    return None
            else:
                st.error(f"HTTP error {response.status_code}: {str(e)}")
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning("Request timed out. Retrying...")
                continue
            else:
                st.error("Request timed out multiple times. Please check your connection.")
                return None
                
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                st.warning("Connection error. Retrying...")
                time.sleep(2)
                continue
            else:
                st.error("Connection failed. Please check your internet connection.")
                return None
                
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    return None

# Alternative: Rate-limited wrapper for any API calls
class RateLimiter:
    def __init__(self, calls_per_minute=10):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        if len(self.call_times) >= self.calls_per_minute:
            # Need to wait
            oldest_call = min(self.call_times)
            wait_time = 60 - (now - oldest_call)
            if wait_time > 0:
                st.info(f"Rate limiting: waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        self.call_times.append(now)


# Initialize session state
if "cache_stats" not in st.session_state:
    st.session_state.cache_stats = {
        "db_cache_hits": 0,
        "db_cache_misses": 0,
        "model_cache_hits": 0,
        "model_cache_misses": 0,
        "document_cache_hits": 0,
        "document_cache_misses": 0,
        "last_reset": datetime.now()
    }
else:
    # Ensure all cache keys exist for compatibility
    for key in ["db_cache_hits", "db_cache_misses", "model_cache_hits", "model_cache_misses"]:
        if key not in st.session_state.cache_stats:
            st.session_state.cache_stats[key] = 0

# Reasoning Window session state initialization
if "reasoning_window" not in st.session_state:
    st.session_state.reasoning_window = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_section" not in st.session_state:
    st.session_state.current_section = "Dashboard"

# Ticket Generator specific session state
if "ticket_messages" not in st.session_state:
    st.session_state.ticket_messages = []
    
if "ticket_file_uploader_key" not in st.session_state:
    st.session_state.ticket_file_uploader_key = "uploader_0"
    
if "gmail_connected" not in st.session_state:
    st.session_state.gmail_connected = False
    
if "last_connection_message" not in st.session_state:
    st.session_state.last_connection_message = None
    
if "email_status" not in st.session_state:
    st.session_state.email_status = {"individual": {}, "bulk": None}
    
if "current_reminders" not in st.session_state:
    st.session_state.current_reminders = []
    
if "chat_cleared" not in st.session_state:
    st.session_state.chat_cleared = False
    
if "db_connection" not in st.session_state:
    st.session_state.db_connection = None
    
if "db_file_path" not in st.session_state:
    st.session_state.db_file_path = None

# Initialize database
init_db()

class GmailConnector:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = ""
        self.password = ""
        self.connected = False
        self.connection = None
        self.settings_file = "gmail_settings.pkl"
    
    def save_settings(self):
        """Save settings to a file"""
        settings = {
            'username': self.username,
            'password': self.password,
            'sender_name': st.session_state.get('gmail_sender_name_input', '')
        }
        with open(self.settings_file, 'wb') as f:
            pickle.dump(settings, f)
        return True
    
    def load_settings(self):
        """Load settings from file or secrets"""
        # First try to load from pickle file
        try:
            with open(self.settings_file, 'rb') as f:
                settings = pickle.load(f)
            self.username = settings.get('username', '')
            self.password = settings.get('password', '')
            # Handle streamlit session state safely
            try:
                import streamlit as st
                st.session_state.gmail_sender_name_input = settings.get('sender_name', 'School Admin')
            except:
                pass  # Ignore if streamlit not available
            return True
        except FileNotFoundError:
            # Try to load from streamlit secrets if pickle file doesn't exist
            try:
                import streamlit as st
                gmail_config = st.secrets.get('gmail', {})
                username = gmail_config.get('username', '')
                password = gmail_config.get('password', '')
                sender_name = gmail_config.get('sender_name', 'School Admin')
                
                if username and password:
                    self.username = username
                    self.password = password
                    st.session_state.gmail_sender_name_input = sender_name
                    return True
            except:
                pass
        except Exception:
            pass

    
    def connect(self, username: str, password: str) -> Tuple[bool, str]:
        """Connect to Gmail SMTP server"""
        try:
            self.username = username
            self.password = password
            
            # Test connection
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10)
            server.starttls()
            server.login(username, password)
            server.quit()
            
            self.connected = True
            st.session_state['gmail_connected'] = True
            return True, "✅ Successfully connected to Gmail"
            
        except smtplib.SMTPAuthenticationError:
            self.connected = False
            st.session_state['gmail_connected'] = False
            return False, "❌ Authentication failed. Please check your email and app password."
        except smtplib.SMTPConnectError:
            self.connected = False
            st.session_state['gmail_connected'] = False
            return False, "❌ Could not connect to Gmail servers. Check internet connection."
        except Exception as e:
            self.connected = False
            st.session_state['gmail_connected'] = False
            return False, f"❌ Connection error: {str(e)}"
    
    def send_email(self, to_email: str, subject: str, body: str, 
                   sender_name: str = None, attachment_data: bytes = None, 
                   attachment_filename: str = None) -> Tuple[bool, str]:
        """Send email via Gmail"""
        if not self.connected:
            success, message = self.connect(self.username, self.password)
            if not success:
                return False, "Not connected to Gmail"
        
        # Re-validate email
        if not EMAIL_PATTERN.match(to_email):
            return False, f"Invalid email format: {to_email}"
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{sender_name} <{self.username}>" if sender_name else self.username
            msg['To'] = to_email
            msg['Subject'] = sanitize_text(subject)
            
            # Attach body
            sanitized_body = sanitize_text(body)
            msg.attach(MIMEText(sanitized_body, 'plain', 'utf-8'))
            
            # Add attachment if provided
            if attachment_data and attachment_filename:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(attachment_data)
                encoders.encode_base64(attachment)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename={sanitize_text(attachment_filename)}'
                )
                msg.attach(attachment)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10)
            server.starttls()
            server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.username, to_email, text)
            server.quit()
            
            return True, "✅ Email sent successfully"
            
        except smtplib.SMTPRecipientsRefused as e:
            error_msg = f"❌ Recipient refused: {str(e)}"
            return False, error_msg
        except smtplib.SMTPException as e:
            error_msg = f"❌ SMTP error: {str(e)}"
            return False, error_msg
        except Exception as e:
            error_msg = f"❌ Failed to send email: {str(e)}"
            return False, error_msg
    
    def send_bulk_reminders(self, reminders: list, sender_name: str = None) -> dict:
        """Send bulk payment reminder emails by reusing send_email for each"""
        results = {"success": 0, "failed": 0, "errors": []}
        
        # Reconnect to ensure valid connection
        if not self.connected:
            success, message = self.test_connection()
            if not success:
                results["errors"].append(message)
                return results
        
        for reminder in reminders:
            if not reminder.get('has_email'):
                results["failed"] += 1
                results["errors"].append(f"{reminder['name']}: No valid email address")
                continue
            
            subject = f"Lunch Payment Reminder - {reminder['name']}"
            body = reminder['reminder']
            
            # Reuse the same send_email method that works for test emails
            success, message = self.send_email(
                reminder['email'],
                subject,
                body,
                sender_name
            )
            
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
                error_msg = f"Failed to send to {reminder['email']}: {message}"
                results["errors"].append(f"{reminder['name']}: {error_msg}")
        
        return results
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test current connection status"""
        if not self.username or not self.password:
            return False, "No credentials configured"
        
        return self.connect(self.username, self.password)

class TicketGenerator:
    def __init__(self):
        self.students_df = pd.DataFrame()
        self.payments_df = pd.DataFrame()
        if 'gmail_connector' not in st.session_state:
            st.session_state['gmail_connector'] = GmailConnector()
        self.gmail = st.session_state['gmail_connector']
        self.ollama_model = "granite3.3:2b"  # Set default model

    def fetch_db_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch student and payment data from school database"""
        try:
            # Fetch students
            results, columns = query_db("SELECT * FROM students")
            if not results:
                return pd.DataFrame(), pd.DataFrame()
                
            students_df = pd.DataFrame(results, columns=columns)
            
            # Handle missing columns
            if 'email' not in students_df.columns:
                students_df['email'] = 'No email provided'
            if 'phone' not in students_df.columns:
                students_df['phone'] = 'No phone provided'
            if 'parent_name' not in students_df.columns:
                students_df['parent_name'] = 'Parent/Guardian'
                
            # Fetch payments
            results, columns = query_db("SELECT * FROM fees")
            if not results:
                return students_df, pd.DataFrame()
                
            payments_df = pd.DataFrame(results, columns=columns)
            
            # Apply payment status mapping
            payments_df['status'] = payments_df['status'].str.lower() if payments_df['status'].dtype == "object" else payments_df['status'].astype(str).str.lower()

            return students_df, payments_df
        except Exception as e:
            st.error(f"Error fetching database data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def process_combined_csv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process combined CSV with student, payment, and contact data"""
        try:
            # Required columns: student_id, name, class, payment_status
            # Optional columns: phone, email, parent_name
            required_cols = ['student_id', 'name', 'class', 'payment_status']
            optional_cols = ['phone', 'email', 'parent_name']
            
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Create separate dataframes with contact info preserved
            student_cols = ['student_id', 'name', 'class']
            # Add optional contact columns if they exist
            for col in optional_cols:
                if col in df.columns:
                    student_cols.append(col)
            
            students_df = df[student_cols].copy()
            students_df['email'] = students_df.get('email', pd.Series('No email provided', index=students_df.index))
            students_df['parent_name'] = students_df.get('parent_name', pd.Series('Parent/Guardian', index=students_df.index))
            payments_df = df[['student_id', 'payment_status']].copy()
            
            return students_df, payments_df
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def filter_paid_students(self, students_df: pd.DataFrame, payments_df: pd.DataFrame) -> pd.DataFrame:
        """Filter students who have paid"""
        try:
            if students_df.empty or payments_df.empty:
                st.warning("No data available to filter paid students")
                return pd.DataFrame()
                
            if 'student_id' not in students_df.columns:
                raise ValueError("Missing 'student_id' column in students data")
            if 'student_id' not in payments_df.columns:
                raise ValueError("Missing 'student_id' column in payments data")
                
            # Get unique paid student IDs
            paid_ids = payments_df[payments_df['payment_status'] == 'paid']['student_id'].unique()
            paid_students = students_df[students_df['student_id'].isin(paid_ids)]
            
            # If no paid students found, show status values for debugging
            if paid_students.empty:
                status_values = payments_df['payment_status'].unique()
                st.warning(f"No paid students found. Payment status values: {status_values}")
                
            return paid_students
        except Exception as e:
            st.error(f"Error filtering students: {str(e)}")
            return pd.DataFrame()

    def generate_pdf(self, students_data: pd.DataFrame, school_info: dict, ticket_date: date, validity_info: dict) -> io.BytesIO:
        """Generate PDF tickets with 3x5 grid (15 tickets per page)"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm, leftMargin=15*mm, rightMargin=15*mm)
        tickets_per_page = 15
        story = []

        for page_start in range(0, len(students_data), tickets_per_page):
            page_students = students_data.iloc[page_start:page_start + tickets_per_page]
            table_data = []
            for row in range(5):
                row_data = []
                for col in range(3):
                    idx = row * 3 + col
                    if idx < len(page_students):
                        student = page_students.iloc[idx]
                        content = self._create_ticket_content(student, school_info, ticket_date, validity_info)
                        row_data.append(content)
                    else:
                        row_data.append(Paragraph("", getSampleStyleSheet()['Normal']))
                table_data.append(row_data)

            page_width = A4[0] - 30*mm
            page_height = A4[1] - 30*mm
            ticket_width = page_width / 3
            ticket_height = (page_height - 20*mm) / 5

            table = Table(table_data, colWidths=[ticket_width]*3, rowHeights=[ticket_height]*5)
            table.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 2, colors.black),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('LINEBELOW', (0, 0), (-1, 3), 2, colors.black, None, (4, 2)),
                ('LINEAFTER', (0, 0), (1, -1), 2, colors.black, None, (4, 2)),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ]))

            story.append(table)
            story.append(Paragraph("<br/>", getSampleStyleSheet()['Normal']))
            cutting_style = ParagraphStyle('CuttingInstructions', parent=getSampleStyleSheet()['Normal'], fontSize=10, textColor=colors.black, alignment=1, fontName='Helvetica-Bold')
            story.append(Paragraph("✂️ Cut along lines to separate tickets (15 per page)", cutting_style))
            if page_start + tickets_per_page < len(students_data):
                story.append(PageBreak())

        doc.build(story)
        buffer.seek(0)
        return buffer

    def _create_ticket_content(self, student, school_info, ticket_date, validity_info):
        """Create individual ticket content"""
        styles = getSampleStyleSheet()
        header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], fontSize=10, leading=11, alignment=1, fontName='Helvetica-Bold', spaceAfter=3)
        ticket_title_style = ParagraphStyle('TicketTitleStyle', parent=styles['Normal'], fontSize=9, leading=10, alignment=1, fontName='Helvetica-Bold', spaceAfter=4)
        content_style = ParagraphStyle('ContentStyle', parent=styles['Normal'], fontSize=8, leading=9, spaceAfter=2, leftIndent=4)
        footer_style = ParagraphStyle('FooterStyle', parent=styles['Normal'], fontSize=6, leading=7, alignment=1, textColor=colors.gray, spaceAfter=1)

        ticket_num = f"LT{random.randint(100000, 999999)}"
        student_name = str(student.get('name', 'N/A')).strip()
        student_class = str(student.get('class', 'N/A')).strip()
        student_id = str(student.get('student_id', 'N/A')).strip()
        school_name = str(school_info.get('name', 'School')).strip()
        issue_date = ticket_date.strftime('%m/%d/%Y')
        validity_display = str(validity_info.get('display', 'Valid Today')).strip()

        ticket_elements = [
            Paragraph(school_name, header_style),
            Paragraph("LUNCH TICKET", ticket_title_style),
            Paragraph(f"<b>Student:</b> {student_name}", content_style),
            Paragraph(f"<b>Class:</b> {student_class}", content_style),
            Paragraph(f"<b>ID:</b> {student_id}", content_style),
            Paragraph(f"<b>Issued:</b> {issue_date}", content_style),
            Paragraph(f"<b>Valid:</b> {validity_display}", content_style),
            Paragraph("------------------------", footer_style),
            Paragraph("This ticket is non-transferable", footer_style),
            Paragraph("Present to cafeteria staff", footer_style),
            Paragraph(f"#{ticket_num}", footer_style)
        ]

        mini_table = Table([[elem] for elem in ticket_elements], colWidths=[None])
        mini_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        return mini_table

    def generate_reminders(self, unpaid_df: pd.DataFrame, sender_info: dict) -> list:
        """Generate payment reminders using chatX.py's AI"""
        if unpaid_df.empty:
            return []
        reminders = []
        total_students = len(unpaid_df.head(10))
        progress = st.progress(0)
        for idx, (_, student) in enumerate(unpaid_df.head(10).iterrows()):
            progress.progress((idx + 1) / total_students)
            
            # improved version:
            phone = str(student.get('phone', 'No phone provided')).strip()
            email = str(student.get('email', 'No email provided')).strip().lower()
            parent_name = str(student.get('parent_name', 'Parent/Guardian')).strip()

            # Phone validation
            has_valid_phone = False
            if phone != 'No phone provided' and phone != 'nan' and phone:
                # Remove non-digit characters
                clean_phone = re.sub(r'\D', '', phone)
                if clean_phone.isdigit() and len(clean_phone) >= 7:
                    has_valid_phone = True
                else:
                    phone = 'Invalid phone format'

            # Email validation
            has_valid_email = False
            if email != 'No email provided' and email != 'nan' and email:
                if EMAIL_PATTERN.match(email):
                    has_valid_email = True
                else:
                    # More permissive check
                    if '@' in email and '.' in email.split('@')[-1]:
                        has_valid_email = True

            prompt = f"""
Write the body of a polite payment reminder email (without subject line) for:
Student: {student.get('name', 'Student')}
Class: {student.get('class', 'N/A')}
Parent/Guardian: {parent_name}

Create a respectful message about their child's lunch payment.
Include that payment is due by {sender_info.get('due_date', 'soon')}.
Keep it under 100 words, professional but friendly.
DO NOT INCLUDE A SUBJECT LINE.

End the message with:

Best regards,
{sender_info['name']}
{sender_info['position']}
{sender_info['date']}

For questions, please contact the school office.
"""
            system_prompt = "You are writing friendly payment reminders to parents."
            reminder_text = "AI disabled for testing."
            if self.ollama_model:
                api_payload = {
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 300}
                }
                try:
                    response = requests.post(
                        f"{get_ollama_host()}/api/chat",
                        json=api_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=60
                    )
                    response.raise_for_status()
                    reminder_text = response.json().get('message', {}).get('content', 'No response generated')
                    reminder_text = sanitize_text(reminder_text)
                except requests.exceptions.Timeout:
                    reminder_text = "Error: AI server timed out after 30 seconds."
                except Exception:
                    reminder_text = "Generation error: Please check AI connection"

            reminders.append({
                'name': student.get('name'),
                'class': student.get('class'),
                'parent_name': parent_name,
                'phone': phone,
                'email': email,
                'reminder': reminder_text,
                'has_phone': phone not in ['No phone provided', 'Invalid phone format'] and pd.notna(phone),
                'has_email': has_valid_email
            })
        progress.progress(1.0)
        return reminders

# Helper function to sanitize text for email
def sanitize_text(text: str) -> str:
    """Sanitize text to ensure it's safe for email sending."""
    if not text:
        return "This is a payment reminder from RadioSport SchoolSync."
    try:
        # Remove any null bytes and encode/decode to handle special characters
        text = text.replace('\x00', '')
        return text.encode('utf-8', errors='ignore').decode('utf-8').strip()
    except Exception:
        return "This is a payment reminder from RadioSport SchoolSync."

# Function to import data from CSV/TXT
def import_data(table_name, file):
    try:
        # Read file
        if file.name.endswith('.csv'):
            # Handle CSV files with extra commas by skipping bad lines
            df = pd.read_csv(file, skipinitialspace=True, on_bad_lines='skip')
        else:  # Assume TXT with tab separator
            df = pd.read_csv(file, sep='\t', skipinitialspace=True, on_bad_lines='skip')
        
        # Clean column names by stripping whitespace and converting to lowercase
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Connect to database
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get existing columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        # Create table if not exists (should already exist, but just in case)
        if not existing_columns:
            st.error(f"Table {table_name} not found in database")
            return False
        
        # Filter columns to those that exist in the table
        valid_columns = [col for col in df.columns if col in existing_columns]
        
        if not valid_columns:
            st.error("No matching columns found between file and database table")
            return False
            
        # Prepare data for insertion
        placeholders = ', '.join(['?'] * len(valid_columns))
        columns_str = ', '.join(valid_columns)
        
        # Insert data
        for _, row in df.iterrows():
            # Skip rows with empty name if table requires name
            if 'name' in valid_columns and (pd.isna(row.get('name')) or row.get('name') == ''):
                continue
                
            values = [row[col] for col in valid_columns]
            cursor.execute(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})", values)
        
        conn.commit()
        conn.close()
        query_db.cache_clear()
        return True
    except Exception as e:
        st.error(f"Error importing data: {str(e)}")
        return False

# Function to delete record
def delete_record(table_name, record_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        query_db.cache_clear()
        return True
    except Exception as e:
        st.error(f"Error deleting record: {str(e)}")
        return False

# Function to update record
def update_record(table_name, record_id, updates):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Build update query
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [record_id]
        
        cursor.execute(f"UPDATE {table_name} SET {set_clause} WHERE id = ?", values)
        conn.commit()
        conn.close()
        query_db.cache_clear()
        return True
    except Exception as e:
        st.error(f"Error updating record: {str(e)}")
        return False

# Helper function for report generation
def format_as_table(results, headers):
    """Format query results as markdown table"""
    if not results:
        return "No data available"
    
    # Create table header
    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---"] * len(headers)) + "|\n"
    
    # Add rows
    for row in results:
        table += "| " + " | ".join(str(x) for x in row) + " |\n"
    
    return table

# Update navigation to include new section
sections = [
    "Dashboard", "Students", "Staff", "Courses", "Attendance", "Test Results", "Fees", 
    "Financials", "Reports", "End of Term Reports", "AI Assistant", 
    "Ticket Generator", "Database Management"  ,"Tools"
]


# Main app
st.markdown('<div class="main-title">RadioSport SchoolSync 🏫</div>', unsafe_allow_html=True)
st.markdown(f'<div style="font-size: 14px; color: #666;">Version {APP_VERSION}</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">RadioSport SchoolSync</div>', unsafe_allow_html=True)
    
    # Navigation
    sections = ["Dashboard", "Students", "Staff", "Courses", "Attendance", "Test Results", "Fees", "Financials", "Reports", "End of Term Reports",
                "AI Assistant", "Ticket Generator", "Database Management", "Tools"]
    selected_section = st.selectbox(
        "Navigate",
        sections,
        index=sections.index(st.session_state.current_section),
        key="navigate_selectbox"
    )

    # Update session state with the new selection
    if selected_section != st.session_state.current_section:
        st.session_state.current_section = selected_section
        st.rerun()
    
    # AI Provider Selection - Auto-detect and default to Cloud if Ollama unavailable
    ollama_available = bool(get_ollama_models())  # Check if Ollama is accessible

    # Set default based on Ollama availability
    default_provider_index = 0 if ollama_available else 1

    ai_provider = st.radio(
        "AI Provider",
        ["Local", "Cloud"],
        index=default_provider_index,
        key="ai_provider_select",
        help="Local: Uses Ollama (if available) | Cloud: Uses OpenRouter API"
    )
    
    # Initialize model variables
    OLLAMA_MODEL = None
    OPENROUTER_MODEL = "anthropic/claude-3-haiku"
    
    # Function to get free OpenRouter models dynamically
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_free_openrouter_models(api_key=None):
        """Fetch available free models from OpenRouter API, ordered by coding performance"""
        # Curated list of best free models for coding/SQL (fallback and default)
        fallback_models = [
            "deepseek/deepseek-chat-v3-0324:free",
            "deepseek/deepseek-r1-0528:free", 
            "moonshotai/kimi-dev-72b:free",
            "openrouter/auto"
        ]
        
        if not api_key:
            return fallback_models
        
        try:
            # Add rate limiting delay
            import time
            time.sleep(0.5)  # 500ms delay to respect rate limits
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rkarikari/RadioSport-chat",
                "X-Title": "RadioSport AI"
            }
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=15)
            
            if response.status_code == 429:
                st.info("Rate limit reached. Using curated model list.")
                return fallback_models
                
            response.raise_for_status()
            
            models_data = response.json()
            free_models = []
            
            # Extract free models (pricing.prompt = 0)
            for model in models_data.get('data', []):
                if (model.get('pricing', {}).get('prompt', '0') == '0' or 
                    ':free' in model.get('id', '') or
                    model.get('id') == 'openrouter/auto'):
                    free_models.append(model.get('id'))
            
            # Order by coding performance (best known models first)
            performance_order = [
                "deepseek/deepseek-chat-v3-0324:free",
                "deepseek/deepseek-r1-0528:free",
                "moonshotai/kimi-dev-72b:free",
                "openrouter/auto"
            ]
            
            # Sort models: known good models first, then others alphabetically
            ordered_models = []
            for model in performance_order:
                if model in free_models:
                    ordered_models.append(model)
                    free_models.remove(model)
            
            # Add remaining free models alphabetically
            ordered_models.extend(sorted(free_models))
            
            return ordered_models if ordered_models else fallback_models
            
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                st.info("Rate limit reached. Using curated model list.")
            else:
                st.info("Could not fetch latest models. Using curated list.")
            return fallback_models
        except Exception as e:
            return fallback_models
    
    # Try to load OpenRouter API key from secrets
    try:
        openrouter_api_key = st.secrets["general"]["OPENROUTER_API_KEY"]
    except KeyError:
        # Fallback to session state or input if secrets not configured
        if "openrouter_api_key" not in st.session_state:
            st.session_state.openrouter_api_key = ""
        openrouter_api_key = None
    
    if ai_provider == "Local":
        # Model Selection section
        st.subheader("Model Selection")
        
        # Refresh models controls
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Refresh Models", use_container_width=True):
                get_ollama_models_cached.clear()
                st.session_state.ollama_models = get_ollama_models()
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto", help="Auto-refresh models")
        
        # Get models (with caching logic)
        if not st.session_state.get('ollama_models') or auto_refresh:
            st.session_state.ollama_models = get_ollama_models()
        else:
            st.session_state.cache_stats["model_cache_hits"] += 1
        
        ollama_models = st.session_state.ollama_models
        
        if ollama_models:
            default_model = 'granite3.3:2b'
            default_index = 0
            if default_model in ollama_models:
                default_index = ollama_models.index(default_model)
                
            OLLAMA_MODEL = st.selectbox(
                "Select a model:",
                ollama_models,
                index=default_index,
                help=f"Available models: {len(ollama_models)}",
                key="ollama_model_selectbox"
            )
        else:
            # Silent fallback - no warning message when online
            OLLAMA_MODEL = None
            if not ollama_available:
                # Automatically switch to cloud without showing error
                st.info("💡 Local AI unavailable. Using Cloud AI provider.")
        
        # Ollama Host Configuration (collapsed by default)
        with st.expander("🔧 Local Server Configuration", expanded=False):
            # Initialize OLLAMA_HOST in session state if not present
            if "ollama_host" not in st.session_state:
                st.session_state.ollama_host = getattr(globals(), 'DEFAULT_OLLAMA_HOST', 'http://localhost:11434')
            
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
                if 'update_all_ollama_host_references' in globals():
                    update_all_ollama_host_references(new_host)
                
                # Clear model cache when host changes
                if 'get_ollama_models_cached' in globals():
                    get_ollama_models_cached.clear()
                st.session_state.ollama_models = []
                
                # Force refresh of models from new host
                try:
                    # Debug: Check what host is being used
                   # print(f"DEBUG: Current OLLAMA_HOST value: {globals().get('OLLAMA_HOST', 'NOT SET')}")
                   # print(f"DEBUG: About to fetch models from: {new_host}")
                    
                    # Fetch models from the new host immediately
                    if 'get_ollama_models_cached' in globals():
                        new_models = get_ollama_models_cached()
                    else:
                        new_models = get_ollama_models()
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
            
    else:  # Cloud
        if not openrouter_api_key:
            # Fallback input for OpenRouter API key
            openrouter_api_key = st.text_input(
                "OpenRouter API Key",
                value=st.session_state.openrouter_api_key,
                type="password",
                key="openrouter_api_key_input",
                help="Get your key from https://openrouter.ai/keys or add it to .streamlit/secrets.toml"
            )
            st.session_state.openrouter_api_key = openrouter_api_key
            if not openrouter_api_key:
                st.warning("Enter OpenRouter API key to select models")
                OPENROUTER_MODEL = None
            else:
                # Get dynamic list of free models
                free_models = get_free_openrouter_models(openrouter_api_key)
                default_index = 0
                if "deepseek/deepseek-chat-v3-0324:free" in free_models:
                    default_index = free_models.index("deepseek/deepseek-chat-v3-0324:free")
                
                OPENROUTER_MODEL = st.selectbox(
                    "Select AI Model",
                    free_models,
                    index=default_index,
                    key="openrouter_model_select"
                )
        else:
            # Get dynamic list of free models
            free_models = get_free_openrouter_models(openrouter_api_key)
            default_index = 0
            if "deepseek/deepseek-chat-v3-0324:free" in free_models:
                default_index = free_models.index("deepseek/deepseek-chat-v3-0324:free")
            
            OPENROUTER_MODEL = st.selectbox(
                "Select AI Model",
                free_models,
                index=default_index,
                key="openrouter_model_select"
            )
 #---------------------------------------------------   
    # Cache stats
    with st.expander("📊 Cache Statistics"):
        try:
            stats = st.session_state.cache_stats
            total_db_requests = stats["db_cache_hits"] + stats["db_cache_misses"]
            total_model_requests = stats["model_cache_hits"] + stats["model_cache_misses"]
            db_hit_rate = (stats["db_cache_hits"] / max(total_db_requests, 1)) * 100
            model_hit_rate = (stats["model_cache_hits"] / max(total_model_requests, 1)) * 100
            uptime = datetime.now() - stats["last_reset"]
            uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
            
            stats_html = f"""
            <div class="cache-stats">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Uptime</td><td>{uptime_str}</td></tr>
                    <tr><td>DB Cache Hit Rate</td><td>{db_hit_rate:.1f}%</td></tr>
                    <tr><td>DB Cache Hits</td><td>{stats["db_cache_hits"]}</td></tr>
                    <tr><td>DB Cache Misses</td><td>{stats["db_cache_misses"]}</td></tr>
                    <tr><td>Model Cache Hit Rate</td><td>{model_hit_rate:.1f}%</td></tr>
                    <tr><td>Model Cache Hits</td><td>{stats["model_cache_hits"]}</td></tr>
                    <tr><td>Model Cache Misses</td><td>{stats["model_cache_misses"]}</td></tr>
                </table>
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)
        except KeyError as e:
            st.error(f"Cache stats error: {e}. Resetting stats.")
            st.session_state.cache_stats = {
                "db_cache_hits": 0,
                "db_cache_misses": 0,
                "model_cache_hits": 0,
                "model_cache_misses": 0,
                "last_reset": datetime.now()
            }
            st.rerun()
        
        if st.button("Reset Stats", key="reset_stats_button"):
            st.session_state.cache_stats = {
                "db_cache_hits": 0,
                "db_cache_misses": 0,
                "model_cache_hits": 0,
                "model_cache_misses": 0,
                "last_reset": datetime.now()
            }
            query_db.cache_clear()
            st.rerun()

# Dashboard
if st.session_state.current_section == "Dashboard":
    st.markdown('<div class="section-header">Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        results, _ = query_db("SELECT COUNT(*) FROM students")
        st.markdown('<div class="card">Total Students<br><b>{}</b></div>'.format(results[0][0]), unsafe_allow_html=True)
    
    with col2:
        results, _ = query_db("SELECT COUNT(*) FROM staff")
        st.markdown('<div class="card">Total Staff<br><b>{}</b></div>'.format(results[0][0]), unsafe_allow_html=True)
    
    with col3:
        results, _ = query_db("SELECT COUNT(*) FROM fees WHERE LOWER(status) = 'unpaid'")
        st.markdown('<div class="card">Unpaid Fees<br><b>{}</b></div>'.format(results[0][0]), unsafe_allow_html=True)
    
    with col4:
        results, _ = query_db("SELECT SUM(amount) FROM fees WHERE LOWER(status) = 'paid'")
        total_revenue = results[0][0] or 0  # Handle None case
        formatted_revenue = "₡ : {:,.2f}".format(total_revenue)  # Format as currency
        st.markdown('<div class="card">Total Revenue<br><b>{}</b></div>'.format(formatted_revenue), unsafe_allow_html=True)
    
    # Visualizations section
    st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Fee Status Distribution pie chart
        st.markdown('<div class="section-header">Fee Status Distribution</div>', unsafe_allow_html=True)
        results, columns = query_db("""
            SELECT 
                CASE 
                    WHEN LOWER(status) = 'paid' THEN 'Paid' 
                    WHEN LOWER(status) = 'unpaid' THEN 'Unpaid'
                    ELSE 'Other'
                END AS status,
                COUNT(*) AS student_count
            FROM fees
            GROUP BY LOWER(status)
        """)
        
        if results:
            df = pd.DataFrame(results, columns=columns)
            fig = px.pie(df, names="status", values="student_count", 
                         title="Fee Status Distribution",
                         hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fee records found")
    
    with col2:
        # Average performance per subject bar graph
        st.markdown('<div class="section-header">Average Performance per Subject</div>', unsafe_allow_html=True)
        results, columns = query_db("SELECT subject, AVG(score) as average_score FROM test_results GROUP BY subject")
        if results:
            df = pd.DataFrame(results, columns=columns)
            fig = px.bar(df, x="subject", y="average_score", 
                         title="Average Test Scores by Subject",
                         color="subject")
            fig.update_layout(showlegend=False)
            fig.update_yaxes(title="Average Score")
            fig.update_xaxes(title="Subject")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No test results found")

# Students
elif st.session_state.current_section == "Students":
    st.markdown('<div class="section-header">Student Management</div>', unsafe_allow_html=True)
    
    with st.expander("Add New Student", expanded=False):
        with st.form(key="student_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name", key="student_name_input")
                grade = st.text_input("Grade", key="student_grade_input", 
                                     help="Enter grade level (e.g., 1R, JHS-2S, KG-B)")
                room = st.text_input("Room", key="student_room_input")
                
                # Display teacher based on grade
                if grade:
                    teacher = get_teacher_for_student(grade)
                else:
                    teacher = "Enter grade first"
                st.text_input("Teacher", value=teacher, disabled=True, key="teacher_display")
                
            with col2:
                dob = st.date_input("Date of Birth", key="student_dob_input")
                phone = st.text_input("Phone", key="student_phone_input")
                address = st.text_area("Address", key="student_address_input")
                enrollment_date = st.date_input("Enrollment Date", key="student_enrollment_date_input")
                email = st.text_input("Email", key="student_email_input")
                parent_name = st.text_input("Parent/Guardian", key="student_parent_input")
            
            if st.form_submit_button("Add Student"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("INSERT INTO students (name, grade, room, dob, phone, address, enrollment_date, email, parent_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                         (name, grade, room, dob, phone, address, enrollment_date, email, parent_name))
                conn.commit()
                conn.close()
                query_db.cache_clear()
                st.success("Student added successfully!")
    
    # Import section
    st.markdown('<div class="section-header">Import Students</div>', unsafe_allow_html=True)
    with st.expander("📤 Import from CSV/TXT"):
        uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'], key="student_import_uploader")
        if uploaded_file is not None:
            if st.button("Import Students", key="student_import_button"):
                if import_data("students", uploaded_file):
                    st.success("Students imported successfully!")
    
    # Display records with edit/delete
    st.markdown('<div class="section-header">Student Records</div>', unsafe_allow_html=True)
    results, columns = query_db("SELECT * FROM students LIMIT ?", (MAX_RECORDS,))
    if results:
        # Add delete column to dataframe
        df = pd.DataFrame(results, columns=columns)
        df['delete'] = False  # Add delete checkbox column
        
        # Convert date columns from string to datetime.date objects
        if 'dob' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce').dt.date
        if 'enrollment_date' in df.columns:
            df['enrollment_date'] = pd.to_datetime(df['enrollment_date'], errors='coerce').dt.date
        
        # Ensure grade column is string type and handle NaN values
        if 'grade' in df.columns:
            df['grade'] = df['grade'].astype(str).replace('nan', '')
            df['grade'] = df['grade'].replace('None', '')
        
        # Add computed teacher column using existing function
        df['teacher'] = df['grade'].apply(lambda x: get_teacher_for_student(x) if x else "Not assigned")
        
        # Configure data editor with explicit column configurations
        column_config = {
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "delete": st.column_config.CheckboxColumn("Delete?", default=False),
            "name": st.column_config.TextColumn(
                "Name",
                help="Student full name",
                max_chars=100,
                required=True
            ),
            "grade": st.column_config.TextColumn(
                "Grade",
                help="Grade level (e.g., 1R, JHS-2S, KG-B)",
                max_chars=20,
                required=False
            ),
            "room": st.column_config.TextColumn(
                "Room",
                help="Classroom identifier"
            ),
            "teacher": st.column_config.TextColumn(
                "Teacher",
                disabled=True
            ),
            "phone": st.column_config.TextColumn(
                "Phone",
                help="Contact phone number",
                max_chars=20
            ),
            "email": st.column_config.TextColumn(
                "Email",
                help="Email address",
                max_chars=100
            ),
            "address": st.column_config.TextColumn(
                "Address",
                help="Home address",
                max_chars=200
            ),
            "parent_name": st.column_config.TextColumn(
                "Parent/Guardian",
                help="Parent or guardian name",
                max_chars=100
            ),
            "dob": st.column_config.DateColumn("Date of Birth"),
            "enrollment_date": st.column_config.DateColumn("Enrollment Date")
        }
        
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config=column_config,
            key="student_data_editor"
        )
        
        # Edit handling
        if st.button("Save Changes", key="save_student_changes"):
            # Create a mapping of IDs to original rows
            id_to_original = {row['id']: row for _, row in df.iterrows()}
            
            changes_made = False
            for _, edited_row in edited_df.iterrows():
                student_id = edited_row['id']
                original_row = id_to_original.get(student_id)
                if original_row is None:
                    continue  # Skip new rows (handled by separate form)
                    
                # Compare edited row with original
                updates = {}
                for col in df.columns:
                    if col not in ['id', 'delete', 'teacher']:  # Skip computed columns
                        edited_val = edited_row[col]
                        original_val = original_row[col]
                        
                        # Handle different data types properly
                        if col in ['dob', 'enrollment_date']:
                            # Handle NaT/NaN values properly
                            if pd.isna(edited_val) and pd.isna(original_val):
                                continue  # Both are NaN, no change
                            elif pd.isna(edited_val) and not pd.isna(original_val):
                                updates[col] = None  # Setting to NULL
                            elif not pd.isna(edited_val) and pd.isna(original_val):
                                if isinstance(edited_val, date):
                                    updates[col] = edited_val.strftime('%Y-%m-%d')
                                else:
                                    updates[col] = edited_val
                            elif not pd.isna(edited_val) and not pd.isna(original_val):
                                # Both have values, compare them
                                if isinstance(edited_val, date) and isinstance(original_val, date):
                                    if edited_val != original_val:
                                        updates[col] = edited_val.strftime('%Y-%m-%d')
                                elif edited_val != original_val:
                                    if isinstance(edited_val, date):
                                        updates[col] = edited_val.strftime('%Y-%m-%d')
                                    else:
                                        updates[col] = edited_val
                        else:
                            # For text columns, handle NaN and None values
                            edited_str = str(edited_val) if not pd.isna(edited_val) else ""
                            original_str = str(original_val) if not pd.isna(original_val) else ""
                            
                            if edited_str != original_str:
                                updates[col] = edited_str if edited_str else None
                
                if updates:
                    if update_record("students", edited_row['id'], updates):
                        changes_made = True
            
            if changes_made:
                st.success("Changes saved successfully!")
                query_db.cache_clear() 
                st.rerun()  # Refresh to show updated data
        
        # Delete handling
        if st.button("Delete Selected", key="delete_students"):
            deleted_count = 0
            for index, row in edited_df.iterrows():
                if row['delete']:  # Check the delete column
                    if delete_record("students", row['id']):
                        deleted_count += 1
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} students")
                query_db.cache_clear()  # Clear cache after deletion
                st.rerun()  # Refresh to show updated list
    else:
        st.info("No student records found")
        
# Staff
elif st.session_state.current_section == "Staff":
    st.markdown('<div class="section-header">Staff Management</div>', unsafe_allow_html=True)
    
    with st.expander("Add New Staff", expanded=False):
        with st.form(key="staff_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name", key="staff_name_input")
                role = st.selectbox("Role", [
                    "Teacher", 
                    "Administrator", 
                    "Support Staff",
                    "Upper Primary Staff",
                    "Lower Primary Staff",
                    "Junior High School Staff"
                ], key="staff_role_selectbox")
                department = st.text_input("Department", key="staff_department_input")
            with col2:
                phone = st.text_input("Phone", key="staff_contact_input")
                hire_date = st.date_input("Hire Date", key="staff_hire_date_input")
                salary = st.number_input("Salary", min_value=0.0, key="staff_salary_input")
            
            if st.form_submit_button("Add Staff"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("INSERT INTO staff (name, role, department, phone, hire_date, salary) VALUES (?, ?, ?, ?, ?, ?)",
                         (name, role, department, phone, hire_date, salary))
                conn.commit()
                conn.close()
                query_db.cache_clear()
                st.success("Staff added successfully!")
    
    # Import section
    st.markdown('<div class="section-header">Import Staff</div>', unsafe_allow_html=True)
    with st.expander("📤 Import from CSV/TXT"):
        uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'], key="staff_import_uploader")
        if uploaded_file is not None:
            if st.button("Import Staff", key="staff_import_button"):
                if import_data("staff", uploaded_file):
                    st.success("Staff imported successfully!")
    
    # Display records with edit/delete
    st.markdown('<div class="section-header">Staff Records</div>', unsafe_allow_html=True)
    results, columns = query_db("SELECT * FROM staff LIMIT ?", (MAX_RECORDS,))
    if results:
        # Add delete column to dataframe
        df = pd.DataFrame(results, columns=columns)
        df['delete'] = False  # Add delete checkbox column
        
        # Configure data editor with checkbox column
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "delete": st.column_config.CheckboxColumn(
                    "Delete?",
                    help="Select rows to delete",
                    default=False,
                )
            }
        )
        
        # Edit handling
        if st.button("Save Changes", key="save_staff_changes"):
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if not row.equals(original_row):
                    updates = {}
                    for col in df.columns:
                        if col != 'id' and col != 'delete' and row[col] != original_row[col]:
                            updates[col] = row[col]
                    if updates:
                        if update_record("staff", row['id'], updates):
                            st.success(f"Updated staff ID {row['id']}")
        
        # Delete handling
        if st.button("Delete Selected", key="delete_staff"):
            deleted_count = 0
            for index, row in edited_df.iterrows():
                if row['delete']:  # Directly check the delete column
                    if delete_record("staff", row['id']):
                        deleted_count += 1
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} staff members")
                st.rerun()  # Refresh to show updated list
    else:
        st.info("No staff records found")

# Attendance
elif st.session_state.current_section == "Attendance":
    st.markdown('<div class="section-header">Attendance Tracking</div>', unsafe_allow_html=True)
   
    # Automatically assume everyone is present unless explicitly marked absent
    with st.expander("Bulk Attendance Management", expanded=True):
        # Date selection
        attendance_date = st.date_input("Select Date", key="bulk_attendance_date")
        
        # Load students
        results, _ = query_db("SELECT id, name FROM students")
        student_options = {row[1]: row[0] for row in results}
        
        if student_options:
            # Get existing attendance for this date
            existing_attendance = {}
            results, _ = query_db("""
                SELECT a.student_id, a.status 
                FROM attendance a 
                WHERE a.date = ?
            """, (attendance_date,))
            for row in results:
                existing_attendance[row[0]] = row[1]
            
            # Create editable DataFrame
            attendance_data = []
            for student_name, student_id in student_options.items():
                status = existing_attendance.get(student_id, "Present")  # Default to Present
                attendance_data.append({
                    "student_id": student_id,
                    "name": student_name,
                    "status": status
                })
            
            df = pd.DataFrame(attendance_data)
            
            # Display editor
            edited_df = st.data_editor(
                df,
                column_config={
                    "student_id": None,
                    "name": st.column_config.TextColumn("Student", disabled=True),
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        options=["Present", "Absent", "Late"],
                        required=True
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("Save Attendance", key="save_bulk_attendance"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                
                for _, row in edited_df.iterrows():
                    student_id = row["student_id"]
                    status = row["status"]
                    
                    # Delete existing record if status is Present (since it's default)
                    if status == "Present":
                        c.execute("DELETE FROM attendance WHERE student_id = ? AND date = ?", 
                                 (student_id, attendance_date))
                    else:
                        # Check if record exists
                        c.execute("SELECT 1 FROM attendance WHERE student_id = ? AND date = ?", 
                                 (student_id, attendance_date))
                        exists = c.fetchone()
                        
                        if exists:
                            # Update existing record
                            c.execute("""
                                UPDATE attendance 
                                SET status = ?
                                WHERE student_id = ? AND date = ?
                            """, (status, student_id, attendance_date))
                        else:
                            # Insert new record
                            c.execute("""
                                INSERT INTO attendance (student_id, date, status)
                                VALUES (?, ?, ?)
                            """, (student_id, attendance_date, status))
                
                conn.commit()
                conn.close()
                st.success("Attendance saved successfully!")
        else:
            st.info("No students found")
   
    with st.expander("Record Attendance", expanded=False):
        with st.form(key="attendance_form"):
            results, _ = query_db("SELECT id, name FROM students")
            student_options = {row[1]: row[0] for row in results}
            student_name = st.selectbox("Student", list(student_options.keys()), key="attendance_student_selectbox")
            date = st.date_input("Date", key="attendance_date_input")
            status = st.selectbox("Status", ["Present", "Absent", "Late"], key="attendance_status_selectbox")
            
            if st.form_submit_button("Record Attendance"):
                student_id = student_options[student_name]
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("INSERT INTO attendance (student_id, date, status) VALUES (?, ?, ?)",
                         (student_id, date, status))
                conn.commit()
                conn.close()
                query_db.cache_clear()
                st.success("Attendance recorded!")
    
    # Import section
    st.markdown('<div class="section-header">Import Attendance</div>', unsafe_allow_html=True)
    with st.expander("📤 Import from CSV/TXT"):
        uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'], key="attendance_import_uploader")
        if uploaded_file is not None:
            if st.button("Import Attendance", key="attendance_import_button"):
                if import_data("attendance", uploaded_file):
                    st.success("Attendance records imported successfully!")
        

    # Display records with edit/delete
    st.markdown('<div class="section-header">Attendance Records</div>', unsafe_allow_html=True)

    # Get detailed attendance records
    results, columns = query_db("SELECT a.id, s.name, a.date, a.status FROM attendance a JOIN students s ON a.student_id = s.id ORDER BY a.date DESC LIMIT ?", (MAX_RECORDS,))

    # Get attendance percentage data
    attendance_percentage_query = """
        SELECT 
            s.id AS student_id,
            s.name,
            COUNT(a.id) AS total_days,
            SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) AS present_days,
            ROUND((SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) * 100.0 / COUNT(a.id)), 2) AS attendance_percentage
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id
        GROUP BY s.id
    """

    # Fetch both detailed records and percentage data
    percentage_results, percentage_columns = query_db(attendance_percentage_query)

    if results:
        df = pd.DataFrame(results, columns=columns)
        
        # Merge attendance percentage with detailed records
        percentage_df = pd.DataFrame(percentage_results, columns=percentage_columns)
        df = df.merge(percentage_df[['name', 'attendance_percentage']], on='name', how='left')
        
        # Display records with percentage
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "attendance_percentage": st.column_config.ProgressColumn(
                    "Attendance %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )
        
        # Edit handling
        if st.button("Save Changes", key="save_attendance_changes"):
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if not row.equals(original_row):
                    updates = {}
                    for col in df.columns:
                        if col != 'id' and col != 'attendance_percentage' and row[col] != original_row[col]:
                            updates[col] = row[col]
                    if updates:
                        if update_record("attendance", row['id'], updates):
                            st.success(f"Updated attendance ID {row['id']}")
            st.rerun()
        
        # Delete handling
        if st.button("Delete Selected", key="delete_attendance"):
            deleted_count = 0
            for index, row in edited_df.iterrows():
                if 'delete' in row and row['delete']:
                    if delete_record("attendance", row['id']):
                        deleted_count += 1
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} attendance records")
                st.rerun()
    else:
        st.info("No attendance records found")

# Courses - Tabbed Interface
elif st.session_state.current_section == "Courses":
    st.markdown('<div class="section-header">Course Management System</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Terms", "📚 Courses", "🏫 Classes", "👥 Class Members", "🎓 Promotions"])
    
    # ==================== TAB 1: TERMS ====================
    with tab1:
        st.markdown("### Academic Term Management")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Existing Terms")
            results, columns = query_db("SELECT * FROM terms")
            if results:
                # Add delete column to dataframe
                df = pd.DataFrame(results, columns=columns)
                df['delete'] = False  # Add delete checkbox column
                
                # Configure data editor with checkbox column
                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "delete": st.column_config.CheckboxColumn(
                            "Delete?",
                            help="Select rows to delete",
                            default=False,
                        )
                    }
                )
                
                # Edit handling
                if st.button("Save Changes", key="save_term_changes"):
                    for index, row in edited_df.iterrows():
                        original_row = df.iloc[index]
                        if not row.equals(original_row):
                            updates = {}
                            for col in df.columns:
                                if col != 'id' and col != 'delete' and row[col] != original_row[col]:
                                    updates[col] = row[col]
                            if updates:
                                conn = sqlite3.connect(DB_NAME)
                                c = conn.cursor()
                                set_clause = ", ".join([f"{col} = ?" for col in updates.keys()])
                                values = list(updates.values()) + [row['id']]
                                c.execute(f"UPDATE terms SET {set_clause} WHERE id = ?", values)
                                conn.commit()
                                conn.close()
                                st.success(f"Updated term ID {row['id']}")
                
                # Delete handling
                if st.button("Delete Selected", key="delete_terms"):
                    deleted_count = 0
                    for index, row in edited_df.iterrows():
                        if row['delete']:  # Directly check the delete column
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            c.execute("DELETE FROM terms WHERE id = ?", (row['id'],))
                            conn.commit()
                            conn.close()
                            deleted_count += 1
                    if deleted_count > 0:
                        query_db.cache_clear()
                        st.success(f"Deleted {deleted_count} term(s)")
                        st.rerun()  # Refresh to show updated list
            else:
                st.info("No terms defined")
        
        with col2:
            st.subheader("Add New Term")
            with st.form("term_form"):
                name = st.text_input("Term Name", key="term_name")
                start_date = st.date_input("Start Date", key="term_start")
                end_date = st.date_input("End Date", key="term_end")
                
                if st.form_submit_button("Add Term"):
                    if name and start_date and end_date:
                        if start_date < end_date:
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            c.execute("INSERT INTO terms (name, start_date, end_date) VALUES (?, ?, ?)",
                                     (name, start_date, end_date))
                            conn.commit()
                            conn.close()
                            query_db.cache_clear()
                            st.success("Term added successfully!")
                            st.rerun()
                        else:
                            st.error("End date must be after start date")
                    else:
                        st.error("All fields are required")
    
    # ==================== TAB 2: COURSE MANAGEMENT ====================
    with tab2:
        st.markdown("### Course Management")
        
        # Create and Enroll Section
        with st.expander("Create Course & Enroll Students", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Create New Course")
                with st.form("course_form", clear_on_submit=True):
                    name = st.text_input("Course Name", key="course_name")
                    description = st.text_area("Description", key="course_description")
                    
                    # For term selection
                    term_results, _ = query_db("SELECT id, name FROM terms")
                    term_options = {row[1]: row[0] for row in term_results} if term_results else {}

                    if term_options:
                        term = st.selectbox("Term", list(term_options.keys()), key="course_term_selectbox")
                    else:
                        st.warning("No terms available. Please add terms first.")
                        term = None
                    
                    # Get available teachers
                    staff_results, _ = query_db("""
                        SELECT id, name 
                        FROM staff 
                        WHERE role IN (
                            'Teacher', 
                            'Upper Primary Staff', 
                            'Lower Primary Staff', 
                            'Junior High School Staff'
                        )
                    """)
                    teacher_options = {row[1]: row[0] for row in staff_results} if staff_results else {}

                    if teacher_options:
                        teacher = st.selectbox("Teacher", list(teacher_options.keys()), key="course_teacher_selectbox")
                    else:
                        st.warning("No teachers available. Please add staff with role 'Teacher' first.")
                        teacher = None

                    schedule = st.text_input("Schedule", key="course_schedule_input")
                    
                    if st.form_submit_button("Add Course"):
                        if teacher and term:
                            term_id = term_options[term]
                            teacher_id = teacher_options[teacher]
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO courses (name, description, teacher_id, term_id, schedule)
                                VALUES (?, ?, ?, ?, ?)
                            """, (name, description, teacher_id, term_id, schedule))
                            conn.commit()
                            conn.close()
                            st.success("Course added successfully!")
                        else:
                            st.error("Please select a teacher and term")
            
            with col2:
                st.subheader("Enroll Students")
                with st.form("enrollment_form", clear_on_submit=True):
                    # Get available courses
                    course_results, _ = query_db("SELECT id, name FROM courses")
                    course_options = {row[1]: row[0] for row in course_results}
                    
                    if course_options:
                        course = st.selectbox("Course", list(course_options.keys()), key="enroll_course_selectbox")
                        
                        # Get available students
                        student_results, _ = query_db("SELECT id, name FROM students")
                        student_options = {row[1]: row[0] for row in student_results}
                        
                        if student_options:
                            student = st.selectbox("Student", list(student_options.keys()), key="enroll_student_selectbox")
                            
                            enrollment_date = st.date_input("Enrollment Date", key="enrollment_date_input")
                            status = st.selectbox("Status", ["Active", "Completed"], key="enrollment_status_selectbox")
                            
                            if st.form_submit_button("Enroll Student"):
                                course_id = course_options[course]
                                student_id = student_options[student]
                                conn = sqlite3.connect(DB_NAME)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO enrollments (student_id, course_id, enrollment_date, status)
                                    VALUES (?, ?, ?, ?)
                                """, (student_id, course_id, enrollment_date, status))
                                conn.commit()
                                conn.close()
                                st.success("Student enrolled successfully!")
                        else:
                            st.info("No students available")
                    else:
                        st.info("No courses available. Create a course first.")

        # Existing Courses Table
        st.markdown("#### Existing Courses")
        
        # Fetch staff and terms for dropdowns
        staff_results, _ = query_db("SELECT id, name FROM staff")
        staff_options = {row[1]: row[0] for row in staff_results} if staff_results else {}
        
        term_results, _ = query_db("SELECT id, name FROM terms")
        term_options = {row[1]: row[0] for row in term_results} if term_results else {}
        
        # Display courses table with edit/delete
        results, columns = query_db("SELECT * FROM courses LIMIT ?", (MAX_RECORDS,))
        if results:
            # Add delete column to dataframe
            df = pd.DataFrame(results, columns=columns)
            df['delete'] = False  # Add delete checkbox column
            
            # Configure data editor with checkbox column
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "delete": st.column_config.CheckboxColumn(
                        "Delete?",
                        help="Select rows to delete",
                        default=False,
                    ),
                    "teacher_id": st.column_config.SelectboxColumn(
                        "Teacher",
                        help="Select teacher",
                        options=list(staff_options.keys()),
                        required=True,
                    ),
                    "term_id": st.column_config.SelectboxColumn(
                        "Term",
                        help="Select term",
                        options=list(term_options.keys()),
                        required=True,
                    )
                }
            )
            
            # Edit handling
            if st.button("Save Course Changes", key="save_course_changes"):
                # Create a mapping of IDs to original rows
                id_to_original = {row['id']: row for _, row in df.iterrows()}
                
                for _, edited_row in edited_df.iterrows():
                    original_row = id_to_original.get(edited_row['id'])
                    if original_row is None:
                        continue  # Skip new rows (handle separately if needed)
                        
                    # Compare edited row with original
                    updates = {}
                    for col in df.columns:
                        if col not in ['id', 'delete'] and edited_row[col] != original_row[col]:
                            # Convert teacher/term names to IDs
                            if col == 'teacher_id':
                                updates[col] = staff_options.get(edited_row[col], original_row[col])
                            elif col == 'term_id':
                                updates[col] = term_options.get(edited_row[col], original_row[col])
                            else:
                                updates[col] = edited_row[col]
                    
                    if updates:
                        if update_record("courses", edited_row['id'], updates):
                            st.success(f"Updated course ID {edited_row['id']}")
            
            # Delete handling
            if st.button("Delete Selected Courses", key="delete_courses"):
                deleted_count = 0
                for index, row in edited_df.iterrows():
                    if row['delete']:
                        if delete_record("courses", row['id']):
                            deleted_count += 1
                if deleted_count > 0:
                    st.success(f"Deleted {deleted_count} courses")
                    st.rerun()  # Refresh to show updated list
        else:
            st.info("No course records found")
    
    # ==================== TAB 3: CLASS MANAGEMENT ====================
    with tab3:
        st.markdown("### Class Management")
        
        # Create Class Section
        with st.expander("Create New Class", expanded=False):
            with st.form("class_form", clear_on_submit=True):
                class_name = st.text_input("Class Name", key="class_name_input")
                room = st.text_input("Room", key="class_room_input")
                
                # Get available teachers
                teacher_results, _ = query_db("SELECT id, name FROM staff")
                
                if teacher_results:
                    # Create teacher options dictionary
                    teacher_options = {row[1]: row[0] for row in teacher_results}
                    
                    # Get available terms
                    term_results, _ = query_db("SELECT id, name FROM terms")
                    term_options = {row[1]: row[0] for row in term_results} if term_results else {}
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        teacher_name = st.selectbox(
                            "Class Teacher", 
                            options=list(teacher_options.keys()),
                            key="class_teacher_selectbox"
                        )
                    with col2:
                        if term_options:
                            term_name = st.selectbox(
                                "Term", 
                                options=list(term_options.keys()),
                                key="class_term_selectbox"
                            )
                        else:
                            st.warning("No terms available")
                            term_name = None
                    
                    if st.form_submit_button("Create Class"):
                        if class_name and teacher_name and term_name:
                            teacher_id = teacher_options[teacher_name]
                            term_id = term_options[term_name]
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO classes (name, room, teacher_id, term_id)
                                VALUES (?, ?, ?, ?)
                            """, (class_name, room, teacher_id, term_id))
                            conn.commit()
                            conn.close()
                            st.success("Class created successfully!")
                            st.rerun()
                        else:
                            st.error("Class Name, Teacher, and Term are required")
                else:
                    st.warning("No staff available. Please add staff first.")

        # Existing Classes Table
        st.markdown("#### Existing Classes")

        # Fetch classes with teacher and term information
        class_query = """
            SELECT 
                c.id AS class_id,
                c.name AS class_name,
                c.room,
                s.name AS teacher_name,
                t.id AS term_id,
                t.name AS term_name
            FROM classes c
            JOIN staff s ON c.teacher_id = s.id
            JOIN terms t ON c.term_id = t.id
        """
        class_results, class_columns = query_db(class_query)

        if class_results:
            # Create DataFrame
            df = pd.DataFrame(class_results, columns=class_columns)
            df['delete'] = False
            
            # Display editable table
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "class_id": None,
                    "delete": st.column_config.CheckboxColumn("Delete?", default=False),
                    "teacher_name": st.column_config.SelectboxColumn(
                        "Teacher",
                        options=list(teacher_options.keys()) if 'teacher_options' in locals() else []
                    ),
                    "term_name": st.column_config.SelectboxColumn(
                        "Term",
                        options=list(term_options.keys()) if 'term_options' in locals() else []
                    )
                },
                hide_index=True
            )
            
            # Handle edits
            if st.button("Save Class Changes", key="save_class_changes"):
                for index, row in edited_df.iterrows():
                    original_row = df.iloc[index]
                    if not row.equals(original_row):
                        updates = {}
                        for col in df.columns:
                            if col not in ['class_id', 'delete'] and row[col] != original_row[col]:
                                updates[col] = row[col]
                        
                        if updates:
                            # Convert teacher name to ID
                            if 'teacher_name' in updates:
                                updates['teacher_id'] = teacher_options[updates['teacher_name']]
                                del updates['teacher_name']
                            
                            # Convert term name to ID
                            if 'term_name' in updates:
                                updates['term_id'] = term_options[updates['term_name']]
                                del updates['term_name']
                            
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            set_clause = ", ".join([f"{col} = ?" for col in updates.keys()])
                            values = list(updates.values()) + [row['class_id']]
                            c.execute(f"UPDATE classes SET {set_clause} WHERE id = ?", values)
                            conn.commit()
                            conn.close()
                
                st.success("Class changes saved!")
            
            # Handle deletes
            if st.button("Delete Selected Classes", key="delete_classes"):
                deleted_count = 0
                for index, row in edited_df.iterrows():
                    if row['delete']:
                        conn = sqlite3.connect(DB_NAME)
                        c = conn.cursor()
                        c.execute("DELETE FROM classes WHERE id = ?", (row['class_id'],))
                        conn.commit()
                        conn.close()
                        deleted_count += 1
                if deleted_count > 0:
                    st.success(f"Deleted {deleted_count} classes")
                    st.rerun()
                if deleted_count > 0:
                    st.success(f"Deleted {deleted_count} classes")
                    st.rerun()
        else:
            st.info("No classes defined yet")
    
    # ==================== TAB 4: CLASS MEMBERS MANAGEMENT ====================
# ==================== TAB 4: CLASS MEMBERS MANAGEMENT ====================
    with tab4:
        st.markdown("### Class Members Management")
        
        # Get all classes for selection
        class_query = """
            SELECT 
                c.id AS class_id,
                c.name AS class_name,
                c.room,
                s.name AS teacher_name,
                t.id AS term_id,
                t.name AS term_name
            FROM classes c
            JOIN staff s ON c.teacher_id = s.id
            JOIN terms t ON c.term_id = t.id
        """
        class_results, class_columns = query_db(class_query)
        
        if class_results:
            class_df = pd.DataFrame(class_results, columns=class_columns)
            
            # Class selection
            selected_class_name = st.selectbox(
                "Select Class to Manage Members", 
                class_df['class_name'].unique(),
                key="class_member_select"
            )
            
            if selected_class_name:
                selected_class_id = class_df[class_df['class_name'] == selected_class_name].iloc[0]['class_id']
                
                # Get term ID for the selected class
                class_row = class_df[class_df['class_name'] == selected_class_name].iloc[0]
                class_term_id = class_row['term_id']
                class_room = class_row['room']
                
                # Display class information
                st.info(f"**Class:** {selected_class_name} | **Teacher:** {class_row['teacher_name']} | **Term:** {class_row['term_name']} | **Room:** {class_room}")
                
                # Auto-sync students from room data to class_members if not already assigned
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                
                # Find students with matching room who aren't in class_members for this term
                sync_query = """
                    INSERT OR IGNORE INTO class_members (class_id, student_id, term_id)
                    SELECT ?, s.id, ?
                    FROM students s
                    WHERE TRIM(UPPER(COALESCE(s.room, ''))) = TRIM(UPPER(?))
                    AND s.id NOT IN (
                        SELECT student_id 
                        FROM class_members 
                        WHERE term_id = ?
                    )
                """
                c.execute(sync_query, (selected_class_id, class_term_id, class_room, class_term_id))
                synced_count = c.rowcount
                conn.commit()
                conn.close()
                
                if synced_count > 0:
                    st.success(f"Auto-synced {synced_count} students from room assignments to class members")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Query class members from class_members table (primary source)
                    member_query = """
                        SELECT 
                            s.id AS student_id,
                            s.name AS student_name,
                            s.id AS student_number,
                            s.grade,
                            s.phone,
                            s.address,
                            s.parent_name,
                            s.enrollment_date,
                            s.email,
                            s.dob
                        FROM students s
                        JOIN class_members cm ON s.id = cm.student_id
                        WHERE cm.class_id = ? AND cm.term_id = ?
                        ORDER BY s.name
                    """
                    member_results, member_columns = query_db(member_query, (selected_class_id, class_term_id))
                    
                    # Create member DataFrame
                    member_df = pd.DataFrame(member_results, columns=member_columns) if member_results else pd.DataFrame(columns=member_columns)
                    
                    st.subheader(f"Current Members ({len(member_df)} students)")
                    
                    if not member_df.empty:
                        member_df['remove'] = False
                        
                        # Display comprehensive student information
                        display_columns = ['student_name', 'student_number', 'grade', 'phone', 'address', 'parent_name', 'enrollment_date', 'email', 'dob', 'remove']
                        
                        edited_member_df = st.data_editor(
                            member_df[display_columns],
                            column_config={
                                "student_name": st.column_config.TextColumn("Student Name", disabled=True, width="medium"),
                                "student_number": st.column_config.NumberColumn("Student ID", disabled=True, width="small"),
                                "grade": st.column_config.NumberColumn("Grade", disabled=True, width="small"),
                                "phone": st.column_config.TextColumn("Phone", disabled=True, width="medium"),
                                "address": st.column_config.TextColumn("Address", disabled=True, width="large"),
                                "parent_name": st.column_config.TextColumn("Parent Name", disabled=True, width="medium"),
                                "enrollment_date": st.column_config.DateColumn("Enrollment Date", disabled=True, width="medium"),
                                "email": st.column_config.TextColumn("Email", disabled=True, width="medium"),
                                "dob": st.column_config.DateColumn("Date of Birth", disabled=True, width="medium"),
                                "remove": st.column_config.CheckboxColumn("Remove?", default=False, width="small")
                            },
                            hide_index=True,
                            use_container_width=True,
                            key="class_members_editor"
                        )
                        
                        # Handle member removal
                        if st.button("Remove Selected Students", key="remove_students_button", type="secondary"):
                            if edited_member_df['remove'].any():
                                conn = sqlite3.connect(DB_NAME)
                                c = conn.cursor()
                                removed_count = 0
                                try:
                                    for _, row in edited_member_df.iterrows():
                                        if row['remove']:
                                            # Remove from class_members table (primary)
                                            c.execute("DELETE FROM class_members WHERE student_id = ? AND class_id = ? AND term_id = ?", 
                                                    (row['student_number'], selected_class_id, class_term_id))
                                            # Also clear room field for consistency
                                            c.execute("UPDATE students SET room = NULL WHERE id = ?", (row['student_number'],))
                                            removed_count += 1
                                    conn.commit()
                                    if removed_count > 0:
                                        st.success(f"Removed {removed_count} students from {selected_class_name}")
                                        st.rerun()
                                    else:
                                        st.warning("No students were removed")
                                except sqlite3.IntegrityError as e:
                                    st.error(f"Error removing students: {str(e)}")
                                    conn.rollback()
                                finally:
                                    conn.close()
                            else:
                                st.warning("Please select at least one student to remove")
                    else:
                        st.info("No students found in this class")
                
                with col2:
                    # Add new members
                    st.subheader("Add New Members")
                    
                    # Get students not assigned to any class in this term
                    available_query = """
                        SELECT DISTINCT s.id, s.name, s.grade, s.phone, s.parent_name
                        FROM students s
                        WHERE s.id NOT IN (
                            SELECT cm.student_id 
                            FROM class_members cm 
                            WHERE cm.term_id = ?
                        )
                        ORDER BY s.name
                    """
                    available_results, _ = query_db(available_query, (class_term_id,))
                    
                    if available_results:
                        # Create a more informative display for students
                        student_display_options = [f"{row[1]} (ID: {row[0]}, Grade: {row[2]})" for row in available_results]
                        student_name_to_id = {f"{row[1]} (ID: {row[0]}, Grade: {row[2]})": row[0] for row in available_results}
                        
                        selected_students = st.multiselect(
                            "Select Students to Add",
                            student_display_options,
                            key="add_students_multiselect",
                            help=f"Available students not assigned to any class this term"
                        )
                        
                        if st.button("Add Selected Students", key="add_students_button", type="primary") and selected_students:
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            added_count = 0
                            errors = []
                            
                            for student_display in selected_students:
                                student_id = student_name_to_id[student_display]
                                student_name = student_display.split(" (ID:")[0]
                                try:
                                    # Add to class_members table (primary)
                                    c.execute("""
                                        INSERT OR REPLACE INTO class_members (class_id, student_id, term_id)
                                        VALUES (?, ?, ?)
                                    """, (selected_class_id, student_id, class_term_id))
                                    
                                    # Update student's room field for consistency
                                    c.execute("UPDATE students SET room = ? WHERE id = ?", (class_room, student_id))
                                    added_count += 1
                                except sqlite3.IntegrityError as e:
                                    errors.append(f"{student_name}: {str(e)}")
                            
                            conn.commit()
                            conn.close()
                            
                            if added_count > 0:
                                st.success(f"Added {added_count} students to {selected_class_name}")
                            
                            if errors:
                                for error in errors:
                                    st.warning(error)
                            
                            if added_count > 0:
                                st.rerun()
                    else:
                        st.info("All students are already assigned to classes this term")
                    
                    # Quick stats
                    st.markdown("---")
                    st.markdown("**Quick Stats:**")
                    total_students = len(query_db("SELECT id FROM students")[0])
                    
                    # Count assigned students for this term
                    assigned_query = """
                        SELECT COUNT(DISTINCT student_id) 
                        FROM class_members 
                        WHERE term_id = ?
                    """
                    assigned_students = query_db(assigned_query, (class_term_id,))[0][0][0]
                    unassigned_students = total_students - assigned_students
                    
                    st.metric("Total Students", total_students)
                    st.metric("Assigned This Term", assigned_students)
                    st.metric("Unassigned This Term", unassigned_students)
                    
                    # Show class member counts for this term
                    class_counts_query = """
                        SELECT c.name, COUNT(cm.student_id) as member_count
                        FROM classes c
                        LEFT JOIN class_members cm ON c.id = cm.class_id AND cm.term_id = ?
                        WHERE c.term_id = ?
                        GROUP BY c.id, c.name
                        ORDER BY c.name
                    """
                    class_counts = query_db(class_counts_query, (class_term_id, class_term_id))[0]
                    
                    if class_counts:
                        st.markdown("**Class Sizes This Term:**")
                        for class_name, count in class_counts:
                            st.write(f"  - {class_name}: {count} students")
        else:
            st.info("No classes available. Please create classes first in the Classes tab.")

        # Utility section for bulk sync
        st.markdown("---")
        with st.expander("🔄 Bulk Sync from Room Assignments", expanded=False):
            st.markdown("**Sync All Students from Room Data**")
            st.markdown("This will add students to class_members table based on their room assignments, only if they're not already assigned.")
            
            if st.button("Sync All Room Assignments", key="bulk_sync_rooms"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                
                # Get current term
                current_term_query = "SELECT id FROM terms ORDER BY id DESC LIMIT 1"
                current_term_result = query_db(current_term_query)
                if current_term_result[0]:
                    current_term_id = current_term_result[0][0][0]
                    
                    # Sync all students from room assignments to class_members
                    bulk_sync_query = """
                        INSERT OR IGNORE INTO class_members (class_id, student_id, term_id)
                        SELECT c.id, s.id, ?
                        FROM students s
                        JOIN classes c ON TRIM(UPPER(COALESCE(s.room, ''))) = TRIM(UPPER(c.room))
                        WHERE c.term_id = ?
                        AND s.id NOT IN (
                            SELECT student_id 
                            FROM class_members 
                            WHERE term_id = ?
                        )
                    """
                    c.execute(bulk_sync_query, (current_term_id, current_term_id, current_term_id))
                    synced_count = c.rowcount
                    conn.commit()
                    conn.close()
                    
                    if synced_count > 0:
                        st.success(f"Bulk synced {synced_count} students from room assignments to class members")
                        st.rerun()
                    else:
                        st.info("No students needed syncing - all room assignments already match class members")
                else:
                    st.error("No terms found. Please create terms first.")

# Add this as TAB 5 after the existing tabs in courses.py

# ==================== TAB 5: PROMOTIONS ====================
    with tab5:
        st.markdown("### Student Promotions & Graduation")
        
        # Get current and next terms
        term_query = "SELECT id, name, start_date, end_date FROM terms ORDER BY start_date DESC"
        term_results, _ = query_db(term_query)
        
        if not term_results:
            st.warning("No terms found. Please create terms first.")
        else:
            current_term = term_results[0]
            next_term = term_results[1] if len(term_results) > 1 else None
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Promotion Rules Section
                st.subheader("📋 Promotion Rules")
                
                with st.expander("Configure Promotion Rules", expanded=False):
                    with st.form("promotion_rules_form"):
                        from_grade = st.selectbox("From Grade", ["1", "2", "3", "4", "5", "6", "JHS1", "JHS2", "JHS3"])
                        to_grade = st.selectbox("To Grade", ["2", "3", "4", "5", "6", "JHS1", "JHS2", "JHS3", "Graduate"])
                        min_score = st.number_input("Minimum Average Score", min_value=0.0, max_value=100.0, value=60.0)
                        auto_promote = st.checkbox("Auto-promote at term end", value=True)
                        
                        if st.form_submit_button("Save Rule"):
                            try:
                                conn = sqlite3.connect(DB_NAME)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT OR REPLACE INTO promotion_rules 
                                    (from_grade, to_grade, minimum_score, auto_promote)
                                    VALUES (?, ?, ?, ?)
                                """, (from_grade, to_grade, min_score, auto_promote))
                                conn.commit()
                                conn.close()
                                st.success("Promotion rule saved!")
                            except Exception as e:
                                st.error(f"Error saving rule: {str(e)}")
                
                # Show existing promotion rules
                try:
                    rules_results, rules_columns = query_db("SELECT * FROM promotion_rules")
                    if rules_results:
                        rules_df = pd.DataFrame(rules_results, columns=rules_columns)
                        st.subheader("Current Promotion Rules")
                        st.dataframe(rules_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load promotion rules: {str(e)}")
                
                # Students Eligible for Promotion
                st.subheader("🎓 Students Eligible for Promotion")
                
                try:
                    # Get students with their current scores and promotion eligibility
                    eligibility_query = """
                    SELECT DISTINCT
                        s.id,
                        s.name,
                        s.grade,
                        COALESCE(AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END), 0) as avg_score,
                        COALESCE(pr.minimum_score, 60.0) as minimum_score,
                        COALESCE(pr.to_grade, 'Review Required') as to_grade,
                        CASE 
                            WHEN AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END) >= COALESCE(pr.minimum_score, 60.0) THEN 'Eligible'
                            ELSE 'Needs Review'
                        END as status,
                        CASE 
                            WHEN s.grade = 'JHS3' AND AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END) >= COALESCE(pr.minimum_score, 60.0) THEN 'Graduate'
                            WHEN AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END) >= COALESCE(pr.minimum_score, 60.0) THEN COALESCE(pr.to_grade, 'Next Grade')
                            ELSE 'Repeat ' || s.grade
                        END as recommended_action
                    FROM students s
                    LEFT JOIN test_results tr ON s.id = tr.student_id
                    LEFT JOIN promotion_rules pr ON s.grade = pr.from_grade
                    WHERE s.status = 'Active' OR s.status IS NULL
                    GROUP BY s.id, s.name, s.grade, pr.minimum_score, pr.to_grade
                    ORDER BY s.grade, s.name
                    """
                    
                    eligibility_results, eligibility_columns = query_db(eligibility_query)
                    
                    if eligibility_results:
                        df = pd.DataFrame(eligibility_results, columns=eligibility_columns)
                        df['promote'] = df['status'] == 'Eligible'  # Default selection for eligible students
                        
                        edited_df = st.data_editor(
                            df,
                            column_config={
                                "id": None,  # Hide ID column
                                "name": st.column_config.TextColumn("Student Name", disabled=True),
                                "grade": st.column_config.TextColumn("Current Grade", disabled=True),
                                "avg_score": st.column_config.NumberColumn("Average Score", format="%.1f", disabled=True),
                                "minimum_score": st.column_config.NumberColumn("Required Score", disabled=True),
                                "to_grade": st.column_config.TextColumn("Next Grade", disabled=True),
                                "status": st.column_config.TextColumn("Status", disabled=True),
                                "recommended_action": st.column_config.TextColumn("Recommended Action", disabled=True),
                                "promote": st.column_config.CheckboxColumn("Promote/Graduate?", default=False)
                            },
                            hide_index=True,
                            use_container_width=True,
                            key="promotion_editor"
                        )
                        
                        # Process Promotions Button
                        if st.button("🚀 Process Selected Promotions", type="primary", key="process_promotions"):
                            try:
                                conn = sqlite3.connect(DB_NAME)
                                c = conn.cursor()
                                
                                promoted_count = 0
                                graduated_count = 0
                                repeated_count = 0
                                
                                for _, row in edited_df.iterrows():
                                    if row['promote']:
                                        student_id = row['id']
                                        current_grade = row['grade']
                                        recommended_action = row['recommended_action']
                                        
                                        if recommended_action == 'Graduate':
                                            # Graduate student
                                            c.execute("UPDATE students SET status = 'Graduate' WHERE id = ?", (student_id,))
                                            c.execute("""
                                                INSERT INTO promotion_history 
                                                (student_id, from_term_id, to_term_id, from_grade, to_grade, action, promotion_date, notes)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """, (student_id, current_term[0], current_term[0], current_grade, 'Graduate', 
                                                 'Graduate', datetime.now().date(), 'Graduated from JHS3'))
                                            graduated_count += 1
                                            
                                        elif not recommended_action.startswith('Repeat'):
                                            # Promote to next grade
                                            new_grade = recommended_action
                                            c.execute("UPDATE students SET grade = ? WHERE id = ?", (new_grade, student_id))
                                            
                                            # Add promotion history
                                            c.execute("""
                                                INSERT INTO promotion_history 
                                                (student_id, from_term_id, to_term_id, from_grade, to_grade, action, promotion_date, notes)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """, (student_id, current_term[0], next_term[0] if next_term else current_term[0], 
                                                 current_grade, new_grade, 'Promote', datetime.now().date(), 
                                                 f'Promoted from {current_grade} to {new_grade}'))
                                            promoted_count += 1
                                        
                                        else:
                                            # Student repeats grade
                                            c.execute("""
                                                INSERT INTO promotion_history 
                                                (student_id, from_term_id, to_term_id, from_grade, to_grade, action, promotion_date, notes)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """, (student_id, current_term[0], next_term[0] if next_term else current_term[0], 
                                                 current_grade, current_grade, 'Repeat', datetime.now().date(), 
                                                 f'Repeating {current_grade} - insufficient score'))
                                            repeated_count += 1
                                
                                conn.commit()
                                conn.close()
                                
                                if promoted_count > 0:
                                    st.success(f"✅ Promoted {promoted_count} students")
                                if graduated_count > 0:
                                    st.success(f"🎓 Graduated {graduated_count} students")
                                if repeated_count > 0:
                                    st.info(f"🔄 {repeated_count} students will repeat their grade")
                                
                                if promoted_count > 0 or graduated_count > 0 or repeated_count > 0:
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Error processing promotions: {str(e)}")
                    else:
                        st.info("No students found for promotion evaluation")
                        
                except Exception as e:
                    st.error(f"Error loading student data: {str(e)}")
            
            with col2:
                # Quick Stats
                st.subheader("📊 Promotion Stats")
                
                try:
                    # Count students by grade
                    grade_query = """
                    SELECT grade, COUNT(*) as count
                    FROM students 
                    WHERE status = 'Active' OR status IS NULL
                    GROUP BY grade
                    ORDER BY 
                        CASE 
                            WHEN grade LIKE 'JHS%' THEN 10 + CAST(SUBSTR(grade, 4) AS INTEGER)
                            ELSE CAST(grade AS INTEGER)
                        END
                    """
                    grade_results, _ = query_db(grade_query)
                    
                    if grade_results:
                        st.markdown("**Current Grade Distribution:**")
                        for grade, count in grade_results:
                            st.write(f"  - Grade {grade}: {count} students")
                            
                except Exception as e:
                    st.warning(f"Could not load grade stats: {str(e)}")
                
                # Recent Promotions History
                st.markdown("---")
                st.subheader("📜 Recent Promotions")
                
                try:
                    history_query = """
                    SELECT s.name, ph.from_grade, ph.to_grade, ph.action, ph.promotion_date
                    FROM promotion_history ph
                    JOIN students s ON ph.student_id = s.id
                    ORDER BY ph.promotion_date DESC
                    LIMIT 10
                    """
                    history_results, _ = query_db(history_query)
                    
                    if history_results:
                        for name, from_grade, to_grade, action, date in history_results:
                            if action == 'Graduate':
                                st.write(f"🎓 **{name}** graduated from {from_grade}")
                            elif action == 'Promote':
                                st.write(f"⬆️ **{name}** {from_grade} → {to_grade}")
                            elif action == 'Repeat':
                                st.write(f"🔄 **{name}** repeating {from_grade}")
                            st.caption(f"   {date}")
                    else:
                        st.info("No promotion history yet")
                        
                except Exception as e:
                    st.warning(f"Could not load promotion history: {str(e)}")


# Test Results
elif st.session_state.current_section == "Test Results":
    st.markdown('<div class="section-header">Test Results</div>', unsafe_allow_html=True)
    
    with st.expander("Record Result", expanded=False):
        with st.form(key="test_results_form"):
            results, _ = query_db("SELECT id, name FROM students")
            student_options = {row[1]: row[0] for row in results}
            student_name = st.selectbox("Student", list(student_options.keys()), key="test_results_student_selectbox")
            subject = st.text_input("Subject", key="test_results_subject_input")
            score = st.number_input("Score", min_value=0.0, max_value=100.0, key="test_results_score_input")
            test_date = st.date_input("Test Date", key="test_results_date_input")
            
            if st.form_submit_button("Record Result"):
                student_id = student_options[student_name]
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("INSERT INTO test_results (student_id, subject, score, test_date) VALUES (?, ?, ?, ?)",
                         (student_id, subject, score, test_date))
                conn.commit()
                conn.close()
                query_db.cache_clear()
                st.success("Test result recorded!")
    
    # Import section
    st.markdown('<div class="section-header">Import Test Results</div>', unsafe_allow_html=True)
    with st.expander("📤 Import from CSV/TXT"):
        uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'], key="test_import_uploader")
        if uploaded_file is not None:
            if st.button("Import Test Results", key="test_import_button"):
                if import_data("test_results", uploaded_file):
                    st.success("Test results imported successfully!")
    
    # Display records with edit/delete
    st.markdown('<div class="section-header">Test Result Records</div>', unsafe_allow_html=True)
    results, columns = query_db("SELECT * FROM test_results LIMIT ?", (MAX_RECORDS,))
    if results:
        # Add delete column to dataframe
        df = pd.DataFrame(results, columns=columns)
        df['delete'] = False  # Add delete checkbox column
        
        # Configure data editor with checkbox column
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "delete": st.column_config.CheckboxColumn(
                    "Delete?",
                    help="Select rows to delete",
                    default=False,
                )
            }
        )
        
        # Edit handling
        if st.button("Save Changes", key="save_test_changes"):
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if not row.equals(original_row):
                    updates = {}
                    for col in df.columns:
                        if col != 'id' and col != 'delete' and row[col] != original_row[col]:
                            updates[col] = row[col]
                    if updates:
                        if update_record("test_results", row['id'], updates):
                            st.success(f"Updated test result ID {row['id']}")
        
        # FIXED: Proper delete handling
        if st.button("Delete Selected", key="delete_test_results"):
            deleted_count = 0
            for index, row in edited_df.iterrows():
                if row['delete']:  # Directly check the delete column
                    if delete_record("test_results", row['id']):
                        deleted_count += 1
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} test results")
                st.rerun()  # Refresh to show updated list
    else:
        st.info("No test results records found")
    
    # Performance visualization
    if results:
        fig = px.line(df, x="test_date", y="score", color="subject", title="Test Scores Over Time")
        st.plotly_chart(fig, use_container_width=True)

# Fees
elif st.session_state.current_section == "Fees":
    st.markdown('<div class="section-header">Fees Management</div>', unsafe_allow_html=True)
    
    with st.expander("Record Fee", expanded=False):
        with st.form(key="fees_form"):
            results, _ = query_db("SELECT id, name FROM students")
            student_options = {row[1]: row[0] for row in results}
            student_name = st.selectbox("Student", list(student_options.keys()), key="fees_student_selectbox")
            amount = st.number_input("Amount", min_value=0.0, key="fees_amount_input")
            due_date = st.date_input("Due Date", key="fees_due_date_input")
            status = st.selectbox("Status", ["Paid", "Unpaid"], key="fees_status_selectbox")
            payment_date = st.date_input("Payment Date", key="fees_payment_date_input") if status == "Paid" else None
            
            if st.form_submit_button("Record Fee"):
                student_id = student_options[student_name]
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("INSERT INTO fees (student_id, amount, due_date, status, payment_date) VALUES (?, ?, ?, ?, ?)",
                         (student_id, amount, due_date, status, payment_date))
                conn.commit()
                conn.close()
                query_db.cache_clear()
                st.success("Fee recorded!")
    
    # Import section
    st.markdown('<div class="section-header">Import Fees</div>', unsafe_allow_html=True)
    with st.expander("📤 Import from CSV/TXT"):
        uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'], key="fees_import_uploader")
        if uploaded_file is not None:
            if st.button("Import Fees", key="fees_import_button"):
                if import_data("fees", uploaded_file):
                    st.success("Fees imported successfully!")
    
    # Display records with edit/delete
    st.markdown('<div class="section-header">Fee Records</div>', unsafe_allow_html=True)
    results, columns = query_db("SELECT * FROM fees LIMIT ?", (MAX_RECORDS,))
    if results:
        # Add delete column to dataframe
        df = pd.DataFrame(results, columns=columns)
        df['delete'] = False  # Add delete checkbox column
        
        # Configure data editor with checkbox column
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "delete": st.column_config.CheckboxColumn(
                    "Delete?",
                    help="Select rows to delete",
                    default=False,
                )
            }
        )
        
        # Edit handling
        if st.button("Save Changes", key="save_fee_changes"):
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if not row.equals(original_row):
                    updates = {}
                    for col in df.columns:
                        if col != 'id' and col != 'delete' and row[col] != original_row[col]:
                            updates[col] = row[col]
                    if updates:
                        if update_record("fees", row['id'], updates):
                            st.success(f"Updated fees ID {row['id']}")
        
        # Delete handling
        if st.button("Delete Selected", key="delete_fees"):
            deleted_count = 0
            for index, row in edited_df.iterrows():
                if row['delete']:  # Directly check the delete column
                    if delete_record("fees", row['id']):
                        deleted_count += 1
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} fees")
                st.rerun()  # Refresh to show updated list
    else:
        st.info("No fee records found")

# Financials
elif st.session_state.current_section == "Financials":
    st.markdown('<div class="section-header">Financial Management</div>', unsafe_allow_html=True)
    
    
    # Expenditure Management
    st.markdown('<div class="section-header">Expenditure Tracking</div>', unsafe_allow_html=True)
    
    with st.expander("Record Expenditure", expanded=False):
        with st.form(key="expenditure_form"):
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category", ["Salaries", "Facilities", "Supplies", "Technology", "Transportation", "Food", "Other"], key="exp_category")
                description = st.text_area("Description", key="exp_description")
            with col2:
                amount = st.number_input("Amount", min_value=0.01, key="exp_amount")
                date = st.date_input("Date", key="exp_date")
                vendor = st.text_input("Vendor", key="exp_vendor")
                notes = st.text_area("Notes", key="exp_notes")
            
            if st.form_submit_button("Record Expenditure"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("INSERT INTO expenditures (category, description, amount, date, vendor, notes) VALUES (?, ?, ?, ?, ?, ?)",
                         (category, description, amount, date, vendor, notes))
                conn.commit()
                conn.close()
                query_db.cache_clear()
                st.success("Expenditure recorded!")
    
    # Import section
    st.markdown('<div class="section-header">Import Expenditures</div>', unsafe_allow_html=True)
    with st.expander("📤 Import from CSV/TXT"):
        uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'], key="exp_import_uploader")
        if uploaded_file is not None:
            if st.button("Import Expenditures", key="exp_import_button"):
                if import_data("expenditures", uploaded_file):
                    st.success("Expenditures imported successfully!")
    
    # Display records with edit/delete
    st.markdown('<div class="section-header">Expenditure Records</div>', unsafe_allow_html=True)
    results, columns = query_db("SELECT * FROM expenditures ORDER BY date DESC LIMIT ?", (MAX_RECORDS,))
    if results:
        df = pd.DataFrame(results, columns=columns)
        edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        
        # Edit handling
        if st.button("Save Changes", key="save_exp_changes"):
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if not row.equals(original_row):
                    updates = {}
                    for col in df.columns:
                        if col != 'id' and row[col] != original_row[col]:
                            updates[col] = row[col]
                    if updates:
                        if update_record("expenditures", row['id'], updates):
                            st.success(f"Updated expenditure ID {row['id']}")
            st.rerun()
        
        # Delete handling
        if st.button("Delete Selected", key="delete_exp"):
            deleted_count = 0
            for index, row in edited_df.iterrows():
                if 'delete' in row and row['delete']:
                    if delete_record("expenditures", row['id']):
                        deleted_count += 1
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} expenditures")
                st.rerun()
    else:
        st.info("No expenditure records found")
    
    # Expenditure Visualization
    if results:
        st.markdown('<div class="section-header">Expenditure Analysis</div>', unsafe_allow_html=True)
        fig = px.pie(df, names="category", values="amount", title="Expenditure by Category")
        st.plotly_chart(fig, use_container_width=True)

# Reports
elif st.session_state.current_section == "Reports":
    st.markdown('<div class="section-header">Reports</div>', unsafe_allow_html=True)
    
    report_type = st.selectbox("Report Type", 
                              ["Student Performance", "Individual Student Performance", "Attendance Summary", 
                               "Financial Overview", "Financial Report"],
                              key="report_type_selectbox")
    
    if report_type == "Student Performance":
        # Get distinct subjects
        subjects_query = "SELECT DISTINCT subject FROM test_results"
        subjects = [row[0] for row in query_db(subjects_query)[0]]
        
        # Build dynamic query for subject averages
        subject_selects = []
        for subject in subjects:
            subject_selects.append(
                f"ROUND(AVG(CASE WHEN t.subject = '{subject}' THEN t.score ELSE NULL END), 2) AS \"{subject}\""
            )
        
        # For empty database case..
        if not subject_selects:
            subject_selects = ["NULL AS no_subjects"]  # Handle empty case

        query = f"""
            SELECT 
                s.name,
                ROUND(AVG(t.score), 2) AS "AVG",
                {", ".join(subject_selects)}
            FROM test_results t
            JOIN students s ON t.student_id = s.id
            GROUP BY s.id
        """
        
        results, columns = query_db(query)
        df = pd.DataFrame(results, columns=columns)
        
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x="name", y="AVG", title="Average Student Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    # FIXED: Individual Student Performance Report
    elif report_type == "Individual Student Performance":
        st.markdown('<div class="section-header">Individual Student Performance Report</div>', unsafe_allow_html=True)
        
        # Get list of students
        results, _ = query_db("SELECT id, name FROM students")
        student_options = {row[1]: row[0] for row in results}
        
        if student_options:
            # Student selection
            selected_student = st.selectbox("Select Student", list(student_options.keys()), key="student_performance_select")
            
            if selected_student:
                student_id = student_options[selected_student]
                
                # Fetch student info - FIXED: Proper unpacking
                student_info = query_db("SELECT name, id, grade, enrollment_date FROM students WHERE id = ?", (student_id,))
                if student_info[0]:
                    # Unpack the first row of the result
                    name, db_id, grade, enrollment_date = student_info[0][0]
                
                    # Display student info
                    st.markdown(f"""
                    <div class="student-report">
                        <h3>{name}</h3>
                        <p><strong>ID No:</strong> {db_id}</p>
                        <p><strong>Grade:</strong> {grade}</p>
                        <p><strong>Enrollment Date:</strong> {enrollment_date}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Fetch test results
                results, columns = query_db(
                    "SELECT subject, score, test_date FROM test_results WHERE student_id = ? ORDER BY test_date",
                    (student_id,)
                )
                
                if results:
                    df = pd.DataFrame(results, columns=columns)
                    
                    # Performance over time
                    st.markdown('<div class="section-header">Performance Over Time</div>', unsafe_allow_html=True)
                    fig_time = px.line(df, x="test_date", y="score", color="subject", 
                                      title=f"Test Scores Over Time - {selected_student}",
                                      markers=True)
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    # Subject-wise performance
                    st.markdown('<div class="section-header">Subject-wise Performance</div>', unsafe_allow_html=True)
                    subject_avg = df.groupby("subject")["score"].mean().reset_index()
                    fig_subject = px.bar(subject_avg, x="subject", y="score", 
                                        title=f"Average Score by Subject - {selected_student}",
                                        color="subject")
                    fig_subject.update_layout(showlegend=False)
                    st.plotly_chart(fig_subject, use_container_width=True)
                    
                    # Latest test results
                    st.markdown('<div class="section-header">Latest Test Results</div>', unsafe_allow_html=True)
                    latest_results = df.sort_values("test_date", ascending=False).head(10)
                    st.dataframe(latest_results, use_container_width=True)
                    
                    # Performance summary
                    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    overall_avg = df["score"].mean()
                    best_subject = subject_avg.loc[subject_avg["score"].idxmax(), "subject"]
                    worst_subject = subject_avg.loc[subject_avg["score"].idxmin(), "subject"]
                    
                    col1.metric("Overall Average", f"{overall_avg:.1f}/100")
                    col2.metric("Best Subject", best_subject)
                    col3.metric("Improvement Needed", worst_subject)
                    
                    # Download report
                    report_data = {
                        "Student Name": name,
                        "Grade": grade,
                        "Enrollment Date": enrollment_date,
                        "Overall Average": overall_avg,
                        "Best Subject": best_subject,
                        "Worst Subject": worst_subject
                    }
                    report_df = pd.DataFrame([report_data])
                    csv = report_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Performance Summary",
                        data=csv,
                        file_name=f"student_performance_{name.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No test results found for {selected_student}")
        else:
            st.info("No students found in database")
    
    elif report_type == "Attendance Summary":
        results, columns = query_db("SELECT s.name, COUNT(CASE WHEN a.status = 'Present' THEN 1 END) as present_days, COUNT(*) as total_days FROM attendance a JOIN students s ON a.student_id = s.id GROUP BY s.id")
        df = pd.DataFrame(results, columns=columns)
        df["attendance_rate"] = (df["present_days"] / df["total_days"]) * 100
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x="name", y="attendance_rate", title="Attendance Rate")
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Financial Overview":
        results, columns = query_db("SELECT status, SUM(amount) as total FROM fees GROUP BY status")
        df = pd.DataFrame(results, columns=columns)
        st.dataframe(df, use_container_width=True)
        fig = px.pie(df, names="status", values="total", title="Fee Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Financial Report
    elif report_type == "Financial Report":
        st.markdown('<div class="section-header">Financial Report</div>', unsafe_allow_html=True)
        
        # Get terms for selection
        terms_results, _ = query_db("SELECT id, name, start_date, end_date FROM terms ORDER BY start_date DESC")
        terms_options = {f"{row[1]} ({row[2]} to {row[3]})": row[0] for row in terms_results}
        
        # Term selection
        if terms_options:
            selected_term_label = st.selectbox("Select Term", list(terms_options.keys()))
            term_id = terms_options[selected_term_label]
            
            # Get term dates  
            term_info = [t for t in terms_results if t[0] == term_id][0]
            term_name = term_info[1]
            start_date = term_info[2]
            end_date = term_info[3]
                        
            st.markdown(f"### Term: {term_name} ({start_date} to {end_date})")
            #  
            # Fetch revenue data for the term
            # Alternative using parameterized query (more secure):
            revenue_results, _ = query_db("""
                SELECT SUM(amount) FROM fees 
                WHERE LOWER(status) = 'paid' 
                AND payment_date >= ? 
                AND payment_date <= ?
            """, (start_date, end_date))
            
            total_revenue = revenue_results[0][0] or 0
            
            # Fetch expenditure data for the term
            expenditure_query = """
                SELECT category, SUM(amount) as total_expenditure 
                FROM expenditures 
                WHERE date BETWEEN ? AND ?
                GROUP BY category
            """
            expenditure_results, expenditure_columns = query_db(expenditure_query, (start_date, end_date))
            total_expenditure = sum(row[1] for row in expenditure_results) if expenditure_results else 0
            
            # Calculate profit
            profit = total_revenue - total_expenditure
            
            # Display financial metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="financial-card">'
                            f'<div class="financial-label">Total Revenue</div>'
                            f'<div class="financial-metric">₡ {total_revenue:,.2f}</div>'
                            f'</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="financial-card">'
                            f'<div class="financial-label">Total Expenditure</div>'
                            f'<div class="financial-metric">₡ {total_expenditure:,.2f}</div>'
                            f'</div>', unsafe_allow_html=True)
            with col3:
                profit_class = "positive-value" if profit >= 0 else "negative-value"
                st.markdown(f'<div class="financial-card">'
                            f'<div class="financial-label">Profit</div>'
                            f'<div class="financial-metric {profit_class}">₡ {profit:,.2f}</div>'
                            f'</div>', unsafe_allow_html=True)
            
            # Display expenditure breakdown
            if expenditure_results:
                st.markdown("### Expenditure Breakdown")
                exp_df = pd.DataFrame(expenditure_results, columns=expenditure_columns)
                exp_df.rename(columns={"category": "Category", "total_expenditure": "Amount"}, inplace=True)
                st.dataframe(exp_df, use_container_width=True)
                
                # Visualization
                fig = px.pie(exp_df, names="Category", values="Amount", 
                             title="Expenditure by Category")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No expenditures recorded for this term")
            
            # Download report
            if st.button("📥 Download Financial Report", key="download_financial_report"):
                # Create a comprehensive report
                report_data = {
                    "Term Name": [term_name],
                    "Start Date": [start_date],
                    "End Date": [end_date],
                    "Total Revenue": [total_revenue],
                    "Total Expenditure": [total_expenditure],
                    "Profit": [profit]
                }
                
                if expenditure_results:
                    for category, amount in expenditure_results:
                        report_data[f"Expenditure: {category}"] = [amount]
                
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download CSV Report",
                    data=csv,
                    file_name=f"financial_report_{term_name.replace(' ', '_')}_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No terms defined. Please add terms in the Financials section.")


# End of Term Report Generation Section - FIXED DATE PROCESSING
elif st.session_state.current_section == "End of Term Reports":
    st.markdown('<div class="section-header">End of Term Report Automation</div>', unsafe_allow_html=True)
    
    # Initialize Gmail connector
    if 'gmail_connector' not in st.session_state:
        st.session_state.gmail_connector = GmailConnector()
        st.session_state.gmail_connector.load_settings()

    # Report scope selection
    report_scope = st.radio("Report Scope", ["Individual", "Class", "All"], horizontal=True)
    
    # Term selection
    results, _ = query_db("SELECT id, name FROM terms ORDER BY end_date DESC")
    term_options = {row[1]: row[0] for row in results} if results else {}
    
    if not term_options:
        st.warning("No terms available. Please add terms first.")
        st.stop()
    
    selected_term_label = st.selectbox("Select Term", list(term_options.keys()))
    term_id = term_options[selected_term_label]
    
    # Student selection based on scope
    students_to_process = []
    
    if report_scope == "Individual":
        student_results, _ = query_db("SELECT id, name FROM students")
        student_options = {row[1]: row[0] for row in student_results}
        selected_student = st.selectbox("Select Student", list(student_options.keys()))
        students_to_process = [(student_options[selected_student], selected_student)]
    
    elif report_scope == "Class":
        # Fixed: Properly query distinct grades and handle text types
        try:
            class_results, _ = query_db("SELECT DISTINCT grade FROM students WHERE grade IS NOT NULL AND grade != '' ORDER BY grade")
            
            if class_results and len(class_results) > 0:
                # Ensure we extract the values correctly
                class_options = []
                for row in class_results:
                    if isinstance(row, (list, tuple)) and len(row) > 0:
                        class_options.append(str(row[0]).strip())
                    else:
                        class_options.append(str(row).strip())
                
                # Remove any empty strings and duplicates
                class_options = list(set([opt for opt in class_options if opt]))
                class_options.sort()
                
                if class_options:
                    selected_class = st.selectbox("Select Class", class_options)
                    
                    # Fixed: Properly query students by grade
                    class_students_results, _ = query_db(
                        "SELECT id, name FROM students WHERE grade = ?", 
                        (selected_class,)
                    )
                    
                    if class_students_results:
                        students_to_process = [(row[0], row[1]) for row in class_students_results]
                    else:
                        st.warning(f"No students found in class {selected_class}")
                        st.stop()
                else:
                    st.warning("No valid classes/grades found after processing.")
                    st.stop()
            else:
                st.warning("No classes/grades found in the students table.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error querying classes: {str(e)}")
            st.stop()
    
    elif report_scope == "All":
        all_students_results, _ = query_db("SELECT id, name FROM students")
        students_to_process = [(row[0], row[1]) for row in all_students_results]

    # HELPER FUNCTION FOR FLEXIBLE DATE COMPARISON
    def normalize_date(date_str):
        """Convert various date formats to YYYY-MM-DD for comparison"""
        if not date_str:
            return None
        
        try:
            # Handle different date formats
            date_formats = [
                '%Y-%m-%d',      # YYYY-MM-DD
                '%d/%m/%Y',      # DD/MM/YYYY
                '%m/%d/%Y',      # MM/DD/YYYY
                '%d-%m-%Y',      # DD-MM-YYYY
                '%Y/%m/%d',      # YYYY/MM/DD
                '%d.%m.%Y',      # DD.MM.YYYY
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(str(date_str).strip(), fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # If none of the formats work, return None
            return None
        except Exception:
            return None

    # Report generation section
    if st.button("Generate Report", key="generate_report_button"):
        if not students_to_process:
            st.error("No students selected for report generation")
            st.stop()
        
        # Use spinner for overall process indication
        with st.spinner("🔄 Initializing report generation..."):
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_container = st.empty()
            total_students = len(students_to_process)
            
            # Generate reports for all selected students
            generated_reports = {}
            temporary_files = []  # Track temporary files for cleanup
            
            try:
                for i, (student_id, student_name) in enumerate(students_to_process):
                    # Update progress
                    progress = (i + 1) / total_students
                    progress_bar.progress(progress)
                    status_container.markdown(f"**Generating report {i+1}/{total_students}: {student_name}**")
                    
                    # Individual student processing with spinner
                    with st.spinner(f"📝 Processing {student_name}'s report..."):
                        graph_path = None
                        try:
                            # Gather student data
                            student_data = query_db("SELECT * FROM students WHERE id = ?", (student_id,))[0][0]
                            
                            # EXTRACT STUDENT DETAILS HERE
                            name = student_data[1]  # Get name from student_data tuple
                            grade = student_data[2]  # Grade
                            dob = student_data[4]    # Date of birth
                            parent_name = student_data[9]  # Parent name
                            
                            # GET TEACHER NAME BASED ON STUDENT'S GRADE
                            teacher_name = get_teacher_for_student(grade)
                            
                            # Get term dates
                            term_info = query_db("SELECT start_date, end_date FROM terms WHERE id = ?", (term_id,))[0][0]
                            term_start = term_info[0]
                            term_end = term_info[1]
                            
                            # FIXED: Flexible date filtering for subjects
                            subjects_query = """
                                SELECT DISTINCT subject 
                                FROM test_results 
                                WHERE student_id = ?
                            """
                            all_subjects_results = query_db(subjects_query, (student_id,))[0]
                            
                            # Filter subjects based on date range with flexible date comparison
                            subjects = []
                            if all_subjects_results:
                                for subject_row in all_subjects_results:
                                    subject = subject_row[0]
                                    # Check if this subject has any test results in the term date range
                                    subject_test_query = """
                                        SELECT test_date FROM test_results 
                                        WHERE student_id = ? AND subject = ?
                                    """
                                    subject_dates = query_db(subject_test_query, (student_id, subject))[0]
                                    
                                    # Check if any test date falls within the term
                                    for date_row in subject_dates:
                                        test_date = normalize_date(date_row[0])
                                        normalized_start = normalize_date(term_start)
                                        normalized_end = normalize_date(term_end)
                                        
                                        if test_date and normalized_start and normalized_end:
                                            if normalized_start <= test_date <= normalized_end:
                                                subjects.append(subject)
                                                break
                            
                            table_data = []
                            for subject in subjects:
                                # FIXED: Student's average with flexible date comparison
                                student_scores = []
                                student_scores_query = """
                                    SELECT score, test_date
                                    FROM test_results
                                    WHERE student_id = ? AND subject = ?
                                """
                                score_results = query_db(student_scores_query, (student_id, subject))[0]
                                
                                for score_row in score_results:
                                    score, test_date = score_row[0], score_row[1]
                                    normalized_test_date = normalize_date(test_date)
                                    normalized_start = normalize_date(term_start)
                                    normalized_end = normalize_date(term_end)
                                    
                                    if normalized_test_date and normalized_start and normalized_end:
                                        if normalized_start <= normalized_test_date <= normalized_end:
                                            student_scores.append(score)
                                
                                student_avg = sum(student_scores) / len(student_scores) if student_scores else 0
                                
                                # FIXED: Class average with flexible date comparison
                                class_scores = []
                                class_scores_query = """
                                    SELECT score, test_date
                                    FROM test_results
                                    WHERE subject = ?
                                """
                                class_score_results = query_db(class_scores_query, (subject,))[0]
                                
                                for score_row in class_score_results:
                                    score, test_date = score_row[0], score_row[1]
                                    normalized_test_date = normalize_date(test_date)
                                    normalized_start = normalize_date(term_start)
                                    normalized_end = normalize_date(term_end)
                                    
                                    if normalized_test_date and normalized_start and normalized_end:
                                        if normalized_start <= normalized_test_date <= normalized_end:
                                            class_scores.append(score)
                                
                                class_avg = sum(class_scores) / len(class_scores) if class_scores else 0
                                
                                # FIXED: Student's position calculation
                                # Get all students' averages for this subject in the term
                                all_student_averages = {}
                                all_students_query = "SELECT DISTINCT student_id FROM test_results WHERE subject = ?"
                                all_students_results = query_db(all_students_query, (subject,))[0]
                                
                                for student_row in all_students_results:
                                    other_student_id = student_row[0]
                                    other_scores = []
                                    
                                    other_scores_query = """
                                        SELECT score, test_date
                                        FROM test_results
                                        WHERE student_id = ? AND subject = ?
                                    """
                                    other_score_results = query_db(other_scores_query, (other_student_id, subject))[0]
                                    
                                    for score_row in other_score_results:
                                        score, test_date = score_row[0], score_row[1]
                                        normalized_test_date = normalize_date(test_date)
                                        normalized_start = normalize_date(term_start)
                                        normalized_end = normalize_date(term_end)
                                        
                                        if normalized_test_date and normalized_start and normalized_end:
                                            if normalized_start <= normalized_test_date <= normalized_end:
                                                other_scores.append(score)
                                    
                                    if other_scores:
                                        all_student_averages[other_student_id] = sum(other_scores) / len(other_scores)
                                
                                # Calculate position
                                position = 1
                                for other_avg in all_student_averages.values():
                                    if other_avg > student_avg:
                                        position += 1
                                
                                # Remarks
                                if student_avg >= 90:
                                    remarks = "Excellent"
                                elif student_avg >= 80:
                                    remarks = "Very Good"
                                elif student_avg >= 70:
                                    remarks = "Good"
                                elif student_avg >= 60:
                                    remarks = "Satisfactory"
                                else:
                                    remarks = "Needs Improvement"
                                
                                table_data.append([subject, f"{student_avg:.1f}", f"{class_avg:.1f}", position, remarks])
                            
                            # FIXED: Overall class position calculation
                            all_student_overall_averages = {}
                            all_students_query = "SELECT DISTINCT student_id FROM test_results"
                            all_students_results = query_db(all_students_query)[0]
                            
                            for student_row in all_students_results:
                                other_student_id = student_row[0]
                                other_scores = []
                                
                                other_scores_query = """
                                    SELECT score, test_date
                                    FROM test_results
                                    WHERE student_id = ?
                                """
                                other_score_results = query_db(other_scores_query, (other_student_id,))[0]
                                
                                for score_row in other_score_results:
                                    score, test_date = score_row[0], score_row[1]
                                    normalized_test_date = normalize_date(test_date)
                                    normalized_start = normalize_date(term_start)
                                    normalized_end = normalize_date(term_end)
                                    
                                    if normalized_test_date and normalized_start and normalized_end:
                                        if normalized_start <= normalized_test_date <= normalized_end:
                                            other_scores.append(score)
                                
                                if other_scores:
                                    all_student_overall_averages[other_student_id] = sum(other_scores) / len(other_scores)
                            
                            # Calculate overall position
                            current_student_overall_avg = all_student_overall_averages.get(student_id, 0)
                            overall_position = 1
                            for other_avg in all_student_overall_averages.values():
                                if other_avg > current_student_overall_avg:
                                    overall_position += 1
                            
                            # FIXED: Performance trends visualization with flexible dates
                            academic_results = []
                            academic_query = """
                                SELECT subject, score, test_date 
                                FROM test_results 
                                WHERE student_id = ?
                                ORDER BY test_date
                            """
                            all_academic_results = query_db(academic_query, (student_id,))[0]
                            
                            # Filter results by term dates
                            for result_row in all_academic_results:
                                subject, score, test_date = result_row[0], result_row[1], result_row[2]
                                normalized_test_date = normalize_date(test_date)
                                normalized_start = normalize_date(term_start)
                                normalized_end = normalize_date(term_end)
                                
                                if normalized_test_date and normalized_start and normalized_end:
                                    if normalized_start <= normalized_test_date <= normalized_end:
                                        academic_results.append((subject, score, test_date))
                            
                            # Create temporary file for the graph
                            if academic_results and len(academic_results) > 0:
                                try:
                                    history_df = pd.DataFrame(academic_results, columns=["subject", "score", "test_date"])
                                    fig = px.line(
                                        history_df, 
                                        x="test_date", 
                                        y="score", 
                                        color="subject",
                                        title=f"Performance Over Time - {student_name}",
                                        markers=True,
                                        color_discrete_sequence=px.colors.qualitative.Plotly
                                    )
                                    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
                                    
                                    # Save to temporary file
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                                        fig.write_image(tmpfile.name, format='png', width=800, height=600, engine='kaleido', scale=2)
                                        graph_path = tmpfile.name
                                        temporary_files.append(graph_path)  # Track for cleanup
                                except Exception as e:
                                    st.warning(f"Graph generation failed for {student_name}: {str(e)}")
                                    graph_path = None
                            
                            # Get principal's name
                            principal_name = get_principal_name()
                    
                            # Prepare AI prompt - UPDATED WITH PRINCIPAL'S SIGNATURE
                            prompt = f"""
Generate a brief end-of-term report for student: {name}
Term: {selected_term_label}

Student Information:
- Grade: {grade}
- Class Teacher: {teacher_name}
- Date of Birth: {dob}
- Parent/Guardian: {parent_name}

Academic Performance Summary:
{format_as_table(table_data, ['Subject', 'Student Score', 'Class Average', 'Position', 'Remarks']) if table_data else 'No performance data'}

Performance Trends:
{"Performance graphs included for each subject showing trends over time" if graph_path else "No performance graphs available"}

Structure the report with these sections:
1. Overall academic performance summary
2. Detailed exam results summary
3. Attendance analysis
4. Behavioral assessment
5. Specific recommendations for improvement
6. Encouragement and next steps

Include specific examples from the data above. Highlight significant achievements and areas needing improvement.

Use professional educational language appropriate for parents. Maintain a balanced perspective that recognizes achievements while addressing challenges.

End the report with:
Best regards,
{principal_name}
School Head
"""
                            
                            # Generate report using AI with spinner for AI processing
                            with st.spinner(f"🤖 Generating AI report content for {student_name}..."):
                                report_content = ""
                                if st.session_state.get("ai_provider_select") == "Local (Ollama)" and OLLAMA_MODEL:
                                    try:
                                        api_payload = {
                                            "model": OLLAMA_MODEL,
                                            "messages": [
                                                {"role": "system", "content": "You are an experienced school administrator creating end-of-term reports."},
                                                {"role": "user", "content": prompt}
                                            ],
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
                                        
                                        report_content = ""
                                        for line in response.iter_lines():
                                            if line:
                                                data = json.loads(line.decode("utf-8"))
                                                if "message" in data and "content" in data["message"]:
                                                    report_content += data["message"]["content"]
                                    
                                    except Exception as e:
                                        report_content = f"Local AI Error: {str(e)}"

                                elif st.session_state.get("ai_provider_select") == "Cloud (OpenRouter)" and OPENROUTER_MODEL and openrouter_api_key:
                                    try:
                                        messages = [
                                            {"role": "system", "content": "You are an experienced school administrator creating end-of-term reports."},
                                            {"role": "user", "content": prompt}
                                        ]
                                        
                                        response = call_openrouter_api(messages, OPENROUTER_MODEL, openrouter_api_key)
                                        if response:
                                            # Process streaming response
                                            report_content = ""
                                            for chunk in response.iter_lines():
                                                if chunk and chunk != b'':
                                                    if chunk.startswith(b'data:'):
                                                        try:
                                                            data = json.loads(chunk.decode("utf-8")[5:])
                                                            if "choices" in data and len(data["choices"]) > 0:
                                                                delta = data["choices"][0].get("delta", {})
                                                                report_content += delta.get("content", "")
                                                        except json.JSONDecodeError:
                                                            pass
                                        else:
                                            report_content = "Report generation error: API call failed"
                                    except Exception as e:
                                        report_content = f"OpenRouter Error: {str(e)}"
                                else:
                                    report_content = "AI disabled or not configured"

                            # PDF generation with spinner
                            with st.spinner(f"📄 Creating PDF report for {student_name}..."):
                                # Prepare student info and generate PDF report
                                student_info_dict = {
                                    'name': student_data[1],
                                    'grade': student_data[2],
                                    'overall_position': overall_position
                                }
                                
                                # Generate PDF report
                                pdf_buffer = generate_report_pdf(
                                    "End of Term Report",
                                    student_name,
                                    selected_term_label,
                                    student_info_dict,
                                    table_data,  # Performance table data
                                    report_content,
                                    graph_path  # Performance graph file path
                                )
                                
                                # Store report in session state
                                generated_reports[student_id] = {
                                    'name': student_name,
                                    'pdf': pdf_buffer,
                                    'content': report_content
                                }

                        except Exception as e:
                            st.error(f"Report generation failed for {student_name}: {str(e)}")
                            continue
                        
                        finally:
                            # Clean up individual student's temporary file immediately
                            if graph_path and os.path.exists(graph_path):
                                try:
                                    os.unlink(graph_path)
                                    if graph_path in temporary_files:
                                        temporary_files.remove(graph_path)
                                except Exception as cleanup_error:
                                    st.warning(f"Could not clean up temporary file for {student_name}: {cleanup_error}")
            
            except Exception as e:
                st.error(f"Report generation process failed: {str(e)}")
            
            finally:
                # Final cleanup for any remaining temporary files
                for temp_file in temporary_files:
                    if os.path.exists(temp_file):
                        try:
                            os.unlink(temp_file)
                        except Exception as cleanup_error:
                            st.warning(f"Could not clean up temporary file {temp_file}: {cleanup_error}")
                
                # Clear progress indicators
                progress_bar.empty()
                status_container.empty()
            
            # Store reports in session state
            st.session_state.generated_reports = generated_reports
            st.session_state.current_term = term_id
            st.session_state.current_term_name = selected_term_label
            
            if generated_reports:
                st.success(f"✅ Successfully generated {len(generated_reports)} reports!")
            else:
                st.warning("No reports were generated successfully.")

    # Display generated reports
    if 'generated_reports' in st.session_state and st.session_state.generated_reports:
        st.markdown("---")
        st.subheader("Generated Reports")
        
        # Display reports for individual students
        if report_scope == "Individual":
            for student_id, report in st.session_state.generated_reports.items():
                st.markdown(f"### {report['name']}")
                st.download_button(
                    "📥 Download PDF Report",
                    data=report['pdf'].getvalue(),
                    file_name=f"{report['name']}_Term_Report_{st.session_state.current_term_name}.pdf",
                    mime="application/pdf"
                )
                with st.expander("View Report Content"):
                    st.write(report['content'])
        
        # Email automation section
        st.markdown("---")
        st.subheader("Email Automation")
        
        with st.expander("Configure Email Settings", expanded=True):
            email_subject = st.text_input(
                "Email Subject", 
                value=f"End of Term Report - {st.session_state.current_term_name}",
                key="email_subject"
            )
            email_body = st.text_area(
                "Email Body", 
                height=150,
                value="""Dear Parent/Guardian,

Please find attached the end of term report for your child.

Best regards,
School Administration""",
                key="email_body"
            )
            sender_name = st.text_input(
                "Sender Name", 
                value=st.session_state.get('gmail_sender_name_input', 'School Admin'),
                key="email_sender_name"
            )
        
        if st.button("📧 Email Reports", key="email_reports_button", type="primary"):
            if not st.session_state.gmail_connector.username or not st.session_state.gmail_connector.password:
                st.error("Gmail not configured. Please set up email in Ticket Generator section.")
                st.stop()
            
            # Email sending process with spinner
            with st.spinner("📧 Connecting to Gmail and sending reports..."):
                # Connect to Gmail
                success, message = st.session_state.gmail_connector.connect(
                    st.session_state.gmail_connector.username,
                    st.session_state.gmail_connector.password
                )
                if not success:
                    st.error(f"Gmail connection failed: {message}")
                    st.stop()
                
                # Email reports
                progress_bar = st.progress(0)
                status_container = st.empty()
                results = []
                
                for i, (student_id, report) in enumerate(st.session_state.generated_reports.items()):
                    progress = (i + 1) / len(st.session_state.generated_reports)
                    progress_bar.progress(progress)
                    status_container.markdown(f"**Sending {i+1}/{len(st.session_state.generated_reports)}: {report['name']}**")
                    
                    try:
                        # Get student email
                        student_email = query_db("SELECT email FROM students WHERE id = ?", (student_id,))[0][0][0]
                        
                        if not EMAIL_PATTERN.match(student_email):
                            results.append({
                                "student": report['name'],
                                "status": "Skipped",
                                "message": f"Invalid email: {student_email}"
                            })
                            continue
                        
                        # Personalize email
                        personalized_subject = email_subject.replace("{student}", report['name'])
                        personalized_body = email_body.replace("{student}", report['name'])
                        
                        # Send email with PDF attachment
                        success, message = st.session_state.gmail_connector.send_email(
                            student_email,
                            personalized_subject,
                            personalized_body,
                            sender_name,
                            report['pdf'].getvalue(),
                            f"{report['name']}_Report_{st.session_state.current_term_name}.pdf"
                        )
                        
                        results.append({
                            "student": report['name'],
                            "status": "Sent" if success else "Failed",
                            "message": message
                        })
                        
                    except Exception as e:
                        results.append({
                            "student": report['name'],
                            "status": "Error",
                            "message": str(e)
                        })
                
                # Display results
                progress_bar.empty()
                status_container.empty()
                st.success(f"Processed {len(st.session_state.generated_reports)} students")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Download summary
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results Summary",
                    data=csv,
                    file_name=f"report_results_{st.session_state.current_term_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# AI Assistant
# AI Assistant
elif st.session_state.current_section == "AI Assistant":
    st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar Chat Controls
    with st.sidebar:
        st.subheader("Chat Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.enable_reasoning = st.checkbox(
                "Reasoning",
                value=st.session_state.get("enable_reasoning", False),
                help="Enable reasoning for qwen3* models"
            )
        
        with col2:
            clear_chat = st.button("Clear", use_container_width=True)
    
    # Handle clear chat functionality
    if clear_chat:
        st.session_state.messages = []
        st.rerun()
    
    # Display chat history
    for msg_index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            parts = split_message(msg["content"])
            for code_index, part in enumerate(parts):
                if part["type"] == "text":
                    st.markdown(part["content"], unsafe_allow_html=True)
                elif part["type"] == "code":
                    st.code(part["code"], language=part["language"])
                    ext = part['language'] if part['language'] != 'text' else 'txt'
                    filename = f"code_{msg_index}_{code_index}.{ext}"
                    st.download_button(
                        label="💾 Download",
                        data=part["code"],
                        file_name=filename,
                        mime="text/plain",
                        key=f"download_code_{msg_index}_{code_index}",
                        use_container_width=False
                    )
    
    # Chat input
    prompt = st.chat_input("Ask the AI Assistant:", key="ai_assistant_chat_input")
    
    if prompt:
        # Refined system prompt with explicit schema - using the same comprehensive prompt as cloud
        db_context = get_db_context()
        system_prompt = f"""
You are an expert in SQLite and school administration. Your task is to answer user questions about the school database by generating and executing SQLite SELECT queries, then presenting the results in clear, conversational English. Assume all queries refer to the school database unless explicitly stated otherwise. Use the conversation history to provide context for follow-up questions, especially when referring to previously mentioned data (e.g., student IDs, scores).

Database Schema:
{db_context}

Instructions:
1. **Always generate a SQLite SELECT query** for questions about school data, such as students, grades, scores, or subjects, using the provided schema.
2. For questions about student performance (e.g., scores, grades, subjects like 'math'), always join the `students` table to include the `name` column in the results, unless explicitly not required.
3. For questions about subjects (e.g., 'math', 'science'), assume they refer to the `test_results` table's `subject` column unless otherwise specified.
4. Use exact table and column names from the schema (e.g., `students.id`, `test_results.score`).
5. Always include the primary key (e.g., `students.id`) in results when listing records.
6. Use SQLite date functions (e.g., `date()`, `strftime()`) for date operations.
7. Execute the query internally using the provided database connection and wait for results before responding.
8. Verify query results before presenting them. If no results are found, state this clearly and suggest an alternative query.
9. For follow-up questions (e.g., 'what is their name?'), use the conversation history to identify relevant context, such as previously mentioned student IDs, and generate a query to retrieve the requested information (e.g., `SELECT name FROM students WHERE id = <previously_mentioned_id>`). Do not ask for clarification if the history provides sufficient context (e.g., a student ID).
10. Present answers conversationally, as if explaining to a school administrator, including the student's name when relevant, without displaying the SQL query unless requested.
11. For calculations (e.g., averages, counts), include the exact results from the query in the response.
12. If the question clearly refers to external events (e.g., 'math Olympiad'), ask for clarification, but prioritize database queries for school-related terms.
13. Handle case sensitivity in `subject` values (e.g., 'Math' vs. 'math') by using `LOWER()` in queries.
14. Never modify data; only use SELECT queries.

Format your response as natural language explanations of the data, not code blocks, unless the user requests the SQL query.
"""
        
        # Handle both AI providers
        if st.session_state.get("ai_provider_select") == "Local (Ollama)" and OLLAMA_MODEL:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt, unsafe_allow_html=True)
            
            # Prepare user query with context
            user_query = f"School Administration Query: {prompt}"
            
            # Append /no_think if reasoning is disabled for qwen3* models
            if not st.session_state.get("enable_reasoning", False) and "qwen3" in OLLAMA_MODEL.lower():
                user_query += " /no_think"
            
            # Build messages array with full history INCLUDING the new user message
            messages_for_api = [{"role": "system", "content": system_prompt}]
            
            # Add all previous messages to context
            for msg in st.session_state.messages:
                if msg["content"].strip():
                    messages_for_api.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current user message to API messages
            messages_for_api.append({"role": "user", "content": user_query})
            
            # Prepare API payload
            api_payload = {
                "model": OLLAMA_MODEL,
                "messages": messages_for_api,
                "stream": True
            }
            
            with st.spinner("🤖️ AI thinking..."):
                response_placeholder = st.empty()
                accumulated_response = ""
                reasoning_buffer = ""
                in_thinking_block = False
                sql_query = None
                
                try:
                    response = requests.post(
                        f"{get_ollama_host()}/api/chat",
                        json=api_payload,
                        headers={"Content-Type": "application/json"},
                        stream=True,
                        timeout=600
                    )
                    response.raise_for_status()
                    
                    # Process response stream
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8"))
                            if "message" in data and "content" in data["message"]:
                                content_chunk = data["message"]["content"]
                                
                                if st.session_state.get("enable_reasoning", False):
                                    reasoning_buffer += content_chunk
                                    processed_content = ""
                                    
                                    while "<think>" in reasoning_buffer or "</think>" in reasoning_buffer:
                                        if not in_thinking_block and "<think>" in reasoning_buffer:
                                            start_idx = reasoning_buffer.find("<think>")
                                            if start_idx > 0:
                                                processed_content += reasoning_buffer[:start_idx]
                                            reasoning_buffer = reasoning_buffer[start_idx + 7:]
                                            in_thinking_block = True
                                            show_reasoning_window()
                                        elif in_thinking_block and "</think>" in reasoning_buffer:
                                            end_idx = reasoning_buffer.find("</think>")
                                            think_content = reasoning_buffer[:end_idx].strip()
                                            reasoning_buffer = reasoning_buffer[end_idx + 8:]
                                            in_thinking_block = False
                                            update_reasoning_window(think_content)
                                            hide_reasoning_window()
                                        else:
                                            break
                                    
                                    if in_thinking_block:
                                        update_reasoning_window(reasoning_buffer.strip())
                                    else:
                                        processed_content += reasoning_buffer
                                        reasoning_buffer = ""
                                    
                                    if processed_content:
                                        accumulated_response += processed_content
                                        response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                else:
                                    accumulated_response += content_chunk
                                    response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                    
                    # Extract SQL query from response
                    if "SELECT" in accumulated_response.upper():
                        match = re.search(r'SELECT.*?;', accumulated_response, re.IGNORECASE | re.DOTALL)
                        if match:
                            sql_query = match.group(0)
                            
                            # Execute the SQL query
                            results, columns, error = execute_sql(sql_query)
                            
                            if error:
                                accumulated_response += f"\n\n❌ Query Error: {error}"
                                response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                            elif results:
                                df = pd.DataFrame(results, columns=columns)
                                
                                if not df.empty:
                                    accumulated_response += f"\n\n🔍 Query returned {len(df)} rows:\n"
                                    accumulated_response += df.to_markdown(index=False)
                                    response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                    
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="💾 Download Results",
                                        data=csv,
                                        file_name="query_results.csv",
                                        mime="text/csv",
                                        key=f"download_results_{int(time.time())}"
                                    )
                                else:
                                    accumulated_response += "\n\nℹ️ Query returned no results"
                                    response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                            else:
                                accumulated_response += "\n\n⚠️ No results found for that query"
                                response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                    
                    # Add user message and assistant response to session state AFTER processing
                    st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()})
                    st.session_state.messages.append({"role": "assistant", "content": accumulated_response})
                
                except requests.exceptions.Timeout:
                    error_msg = "AI server timed out after 600 seconds."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()})
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()})
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                finally:
                    hide_reasoning_window()
        
        elif st.session_state.get("ai_provider_select") == "Cloud (OpenRouter)" and OPENROUTER_MODEL and openrouter_api_key:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt, unsafe_allow_html=True)
            
            # Prepare user query
            user_query = f"School Administration Query: {prompt}"
            
            # Build messages array with full history INCLUDING the new user message
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add all previous messages to context
            for msg in st.session_state.messages:
                if msg["content"].strip():
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current user message
            messages.append({"role": "user", "content": user_query})
            
            with st.spinner("🤖️ AI thinking..."):
                response_placeholder = st.empty()
                accumulated_response = ""
                
                # Call OpenRouter API
                response = call_openrouter_api(messages, OPENROUTER_MODEL, openrouter_api_key)
                
                if response:
                    try:
                        # Process streaming response
                        for chunk in response.iter_lines():
                            if chunk and chunk != b'':
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
                        
                        # Extract SQL query from response
                        if "SELECT" in accumulated_response.upper():
                            match = re.search(r'SELECT.*?;', accumulated_response, re.IGNORECASE | re.DOTALL)
                            if match:
                                sql_query = match.group(0)
                                
                                # Execute the SQL query
                                results, columns, error = execute_sql(sql_query)
                                
                                if error:
                                    accumulated_response += f"\n\n❌ Query Error: {error}"
                                    response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                elif results:
                                    df = pd.DataFrame(results, columns=columns)
                                    
                                    if not df.empty:
                                        accumulated_response += f"\n\n🔍 Query returned {len(df)} rows:\n"
                                        accumulated_response += df.to_markdown(index=False)
                                        response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                        
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="💾 Download Results",
                                            data=csv,
                                            file_name="query_results.csv",
                                            mime="text/csv",
                                            key=f"download_results_{int(time.time())}"
                                        )
                                    else:
                                        accumulated_response += "\n\nℹ️ Query returned no results"
                                        response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                else:
                                    accumulated_response += "\n\n⚠️ No results found for that query"
                                    response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                        
                        # Add user message and assistant response to session state AFTER processing
                        st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()})
                        st.session_state.messages.append({"role": "assistant", "content": accumulated_response})
                    
                    except Exception as e:
                        error_msg = f"Error processing response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()})
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = "Failed to get response from OpenRouter"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()})


# Ticket Generator
elif st.session_state.current_section == "Ticket Generator":
    st.markdown(f'<div class="main-title">Ticket Generator 🍽️</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 14px; color: #333;">Version {APP_VERSION}</div>', unsafe_allow_html=True)

    generator = TicketGenerator()
    ollama_models = get_ollama_models()
    
    # Set default model to granite3.3:2b if available
    if "granite3.3:2b" in ollama_models:
        generator.ollama_model = "granite3.3:2b"
    elif ollama_models:
        generator.ollama_model = ollama_models[0]

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)

        # Data Source Selection
        data_source = "School Database" 
        
        # Gmail Connector Section - 
        with st.expander("📧 Gmail Connector", expanded=False):
            gmail_enabled = st.checkbox("Enable Gmail Connector", key="gmail_enabled")
            
            if gmail_enabled:
                # Try to load credentials automatically
                credentials_loaded = False
                
                # Try loading from gmail_settings.pkl first
                try:
                    if generator.gmail.load_settings():
                        credentials_loaded = True
                        st.success(f"✅ Gmail credentials loaded for: {generator.gmail.username}")
                        # Auto-connect with loaded credentials
                        if not generator.gmail.connected:
                            success, message = generator.gmail.connect(generator.gmail.username, generator.gmail.password)
                            if success:
                                st.session_state['gmail_connected'] = True
                                generator.gmail.connected = True
                            else:
                                st.error(f"Connection failed: {message}")
                except Exception as e:
                    # Try loading from secrets.toml
                    try:
                        gmail_username = st.secrets.get("gmail", {}).get("username", "")
                        gmail_password = st.secrets.get("gmail", {}).get("password", "")
                        sender_name = st.secrets.get("gmail", {}).get("sender_name", "School Admin")
                        
                        if gmail_username and gmail_password:
                            generator.gmail.username = gmail_username
                            generator.gmail.password = gmail_password
                            st.session_state.gmail_sender_name_input = sender_name
                            credentials_loaded = True
                            st.success(f"✅ Gmail credentials loaded from secrets for: {gmail_username}")
                            
                            # Auto-connect with loaded credentials
                            if not generator.gmail.connected:
                                success, message = generator.gmail.connect(gmail_username, gmail_password)
                                if success:
                                    st.session_state['gmail_connected'] = True
                                    generator.gmail.connected = True
                                else:
                                    st.error(f"Connection failed: {message}")
                    except Exception:
                        pass
                
                # Only show manual input interface if credentials not loaded
                if not credentials_loaded:
                    st.warning("⚠️ No saved Gmail credentials found. Please configure manually.")
                    st.markdown("[📖 How to create Gmail App Password](https://support.google.com/accounts/answer/185833)")
                    
                    gmail_username = st.text_input(
                        "Gmail Address", 
                        placeholder="your-email@gmail.com", 
                        key="gmail_username",
                        help="Your full Gmail address. Use an App Password for authentication."
                    )
                    
                    gmail_password = st.text_input(
                        "App Password", 
                        type="password", 
                        key="gmail_password",
                        help="16-character app password from Google (not your regular password)"
                    )
                    
                    sender_display_name = st.text_input(
                        "Display Name", 
                        value="School Admin",
                        key="gmail_sender_name_input",
                        help="Name that appears as sender"
                    )
                    
                    col_connect, col_save = st.columns(2)
                    
                    with col_connect:
                        if st.button("🔗 Connect Gmail", use_container_width=True, key="connect_gmail"):
                            if gmail_username and gmail_password:
                                with st.spinner("Connecting to Gmail..."):
                                    success, message = generator.gmail.connect(gmail_username, gmail_password)
                                    if success:
                                        st.success(message)
                                        st.session_state['gmail_connected'] = True
                                        generator.gmail.connected = True
                                    else:
                                        st.error(message)
                                        st.session_state['gmail_connected'] = False
                                        generator.gmail.connected = False
                            else:
                                st.error("Please enter both email and app password")
                    
                    with col_save:
                        if st.button("💾 Save Settings", use_container_width=True, key="save_gmail"):
                            if gmail_username and gmail_password:
                                generator.gmail.username = gmail_username
                                generator.gmail.password = gmail_password
                                if generator.gmail.save_settings():
                                    st.success("Settings saved successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to save settings")
                            else:
                                st.error("Please enter credentials first")
                
                # Connection status display (always shown when enabled)
                if st.session_state.get('gmail_connected', False) and generator.gmail.connected:
                    st.success(f"🟢 Connected as: {generator.gmail.username}")
                else:
                    st.error("🔴 Not connected")
                
                # Send Test Email (always available when connected)
                if st.session_state.get('gmail_connected', False) and generator.gmail.connected:
                    st.subheader("📧 Send Test Email")
                    test_email = st.text_input("Test Email Address", placeholder="test@example.com", key="test_email_input")
                    if st.button("📤 Send Test Email", use_container_width=True):
                        if test_email:
                            sender_name = st.session_state.get('gmail_sender_name_input', 'School Admin')
                            success, message = generator.gmail.send_email(
                                test_email,
                                "Test Email from Ticket Generator",
                                f"This is a test email sent from the Ticket Generator app.\n\nConnection successful!\n\nBest regards,\n{sender_name}",
                                sender_name
                            )
                            if success:
                                st.success(f"Test email sent to {test_email}")
                            else:
                                st.error(message)
                        else:
                            st.error("Please enter a test email address")
        
        # Ticket Settings
        with st.expander("🎟 Ticket Settings", expanded=False):
            school_name = st.text_input("School Name", value="RadioSport")
            ticket_date = st.date_input("Issue Date", value=date.today())
            validity_type = st.selectbox("Validity Type", ["Single Day", "Date Range", "Weekly"])
            if validity_type == "Single Day":
                valid_date = st.date_input("Valid Date", value=date.today())
                validity_info = {'display': f"Valid: {valid_date.strftime('%m/%d/%Y')}"}
            elif validity_type == "Date Range":
                start_date = st.date_input("Start Date", value=date.today())
                end_date = st.date_input("End Date", value=date.today())
                validity_info = {'display': f"{start_date.strftime('%m/%d')} - {end_date.strftime('%m/%d')}"}
            else:
                week_start = st.date_input("Week Start", value=date.today())
                week_end = week_start + pd.Timedelta(days=6)
                validity_info = {'display': f"Week: {week_start.strftime('%m/%d')} - {week_end.strftime('%m/%d')}"}
        
        # Send All Reminders Button


        # AI Settings
        # AI Settings - UPDATED SECTION WITH AUTO-LOADING
        with st.expander("🤖 AI Settings", expanded=False):
            # AI Provider Selection - Auto-detect Ollama availability
            ollama_available_ticket = bool(get_ollama_models())
            default_ticket_provider = 0 if ollama_available_ticket else 1
            
            ai_provider = st.radio(
                "AI Provider",
                ["Local", "Cloud"],
                index=default_ticket_provider,
                key="ticket_ai_provider",
                help="Local: Ollama (if available) | Cloud: OpenRouter"
            )
            
            if ai_provider == "Local":
                if ollama_models:
                    # Set default selection to granite3.3:2b if available
                    default_index = ollama_models.index("granite3.3:2b") if "granite3.3:2b" in ollama_models else 0
                    generator.ollama_model = st.selectbox(
                        "Select AI Model", 
                        ollama_models, 
                        index=default_index,
                        key="ticket_ollama_model_select"
                    )
                    st.markdown('<div class="ai-status ai-connected">🟢 AI Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ai-status ai-disconnected">🔴 AI Disconnected</div>', unsafe_allow_html=True)
            else:  # Cloud (OpenRouter)
                # Try to auto-load API key from secrets.toml
                auto_loaded_key = ""
                try:
                    auto_loaded_key = st.secrets.get("general", {}).get("OPENROUTER_API_KEY", "")
                    if auto_loaded_key:
                        st.success("✅ OpenRouter API key loaded from secrets")
                except Exception:
                    pass
                
                openrouter_api_key = st.text_input(
                    "OpenRouter API Key",
                    type="password",
                    value=auto_loaded_key,  # Auto-populate from secrets
                    key="ticket_openrouter_key",
                    help="Get your key from https://openrouter.ai/keys"
                )
                
                if openrouter_api_key:
                    OPENROUTER_MODEL = st.selectbox(
                        "Select AI Model",
                        OPENROUTER_MODELS,
                        index=0,
                        key="ticket_openrouter_model"
                    )
                    st.markdown('<div class="ai-status ai-connected">🟢 OpenRouter Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ai-status ai-disconnected">🔴 Enter API Key</div>', unsafe_allow_html=True)
            
            # Clear Chat Button
            if st.button("🧹 Clear Chat", 
                         key="clear_chat_button", 
                         help="Clear all chat messages",
                         use_container_width=True):
                st.session_state.ticket_messages = []
                st.session_state.chat_cleared = True
                st.rerun()

        # Cache Stats
        with st.expander("📊 Cache Statistics"):
            try:
                stats = st.session_state.cache_stats
                total_db_requests = stats["db_cache_hits"] + stats["db_cache_misses"]
                total_model_requests = stats["model_cache_hits"] + stats["model_cache_misses"]
                db_hit_rate = (stats["db_cache_hits"] / max(total_db_requests, 1)) * 100
                model_hit_rate = (stats["model_cache_hits"] / max(total_model_requests, 1)) * 100
                uptime = datetime.now() - stats["last_reset"]
                uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
                
                stats_html = f"""
                <div class="cache-stats">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Uptime</td><td>{uptime_str}</td></tr>
                        <tr><td>DB Cache Hit Rate</td><td>{db_hit_rate:.1f}%</td></tr>
                        <tr><td>DB Cache Hits</td><td>{stats["db_cache_hits"]}</td></tr>
                        <tr><td>DB Cache Misses</td><td>{stats["db_cache_misses"]}</td></tr>
                        <tr><td>Model Cache Hit Rate</td><td>{model_hit_rate:.1f}%</td></tr>
                        <tr><td>Model Cache Hits</td><td>{stats["model_cache_hits"]}</td></tr>
                        <tr><td>Model Cache Misses</td><td>{stats["model_cache_misses"]}</td></tr>
                    </table>
                </div>
                """
                st.markdown(stats_html, unsafe_allow_html=True)
            except KeyError as e:
                st.error(f"Cache stats error: {e}. Resetting stats.")
                st.session_state.cache_stats = {
                    "db_cache_hits": 0,
                    "db_cache_misses": 0,
                    "model_cache_hits": 0,
                    "model_cache_misses": 0,
                    "last_reset": datetime.now()
                }
                st.rerun()
            
            if st.button("Reset Stats", key="ticket_reset_stats_button"):
                st.session_state.cache_stats = {
                    "db_cache_hits": 0,
                    "db_cache_misses": 0,
                    "model_cache_hits": 0,
                    "model_cache_misses": 0,
                    "last_reset": datetime.now()
                }
                query_db.cache_clear()
                st.rerun()

    tab1, tab2, tab3 = st.tabs(["📊 Data Management", "🤖 AI Assistant", "🎫 Generate Tickets"])

    with tab1:
        st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            if data_source == "School Database":
                st.markdown('<div class="section-header">Lunch Payment Records</div>', unsafe_allow_html=True)
                
                # Query lunch payment data
                lunch_query = """
                    SELECT 
                        lp.id,
                        s.name AS student_name,
                        s.grade AS class,
                        s.phone,  -- ADDED
                        s.email,  -- ADDED
                        s.parent_name,  -- ADDED
                        lp.amount,
                        lp.payment_date,
                        lp.status,
                        t.name AS term
                    FROM lunch_payments lp
                    JOIN students s ON lp.student_id = s.id
                    LEFT JOIN terms t ON lp.term_id = t.id
                """
                results, columns = query_db(lunch_query)
                
                if results:
                    # Create and display editable DataFrame
                    df = pd.DataFrame(results, columns=columns)
                    df['delete'] = False
                    
                    # Get all students for dropdown
                    all_students = query_db("SELECT id, name FROM students")[0]
                    student_options = {row[1]: row[0] for row in all_students}
                    
                    # Add new row functionality with student dropdown
                    with st.expander("➕ Add New Record", expanded=False):
                        # Searchable student dropdown
                        search_term = st.text_input("Search Student", key="student_search")
                        filtered_students = [name for name in student_options.keys() 
                                            if search_term.lower() in name.lower()] if search_term else list(student_options.keys())
                        
                        selected_student = st.selectbox("Select Student", filtered_students, key="new_student_select")
                        
                        # Get student details
                        student_details = query_db("SELECT grade FROM students WHERE id = ?", 
                                                  (student_options[selected_student],))[0][0]
                        
                        # Display auto-populated details
                        st.text_input("Class", value=student_details[0], disabled=True, key="auto_class")
                        
                        # Editable fields
                        amount = st.number_input("Amount", min_value=0.0, step=100.0, key="new_amount")
                        status = st.selectbox("Status", ["paid", "unpaid"], key="new_status")
                        payment_date = st.date_input("Payment Date", value=date.today(), key="new_payment_date")
                        
                        if st.button("Add Record", key="add_lunch_record"):
                            conn = sqlite3.connect(DB_NAME)
                            c = conn.cursor()
                            
                            # Get student ID
                            student_id = student_options[selected_student]
                            
                            # Check for existing records - PROPERLY UNPACK RESULTS
                            results, columns = query_db(
                                "SELECT id FROM lunch_payments WHERE student_id = ?",
                                (student_id,)
                            )
                            
                            if results:  # If results exist
                                # Get first ID value from first row
                                record_id = results[0][0]  # Access first element of first row
                                
                                # Update existing record
                                c.execute("""
                                    UPDATE lunch_payments 
                                    SET amount = ?, payment_date = ?, status = ?
                                    WHERE id = ?
                                """, (amount, payment_date, status, record_id))
                                action = "updated"
                            else:
                                # Insert new record
                                c.execute("""
                                    INSERT INTO lunch_payments (student_id, amount, payment_date, status)
                                    VALUES (?, ?, ?, ?)
                                """, (student_id, amount, payment_date, status))
                                action = "added"
                            
                            conn.commit()
                            conn.close()
                            st.success(f"Record {action} successfully!")
                            st.rerun()
                    
                    # Configure editor
                    edited_df = st.data_editor(
                        df,
                        use_container_width=True,
                        num_rows="dynamic",
                        column_config={
                            "delete": st.column_config.CheckboxColumn("Delete?", default=False),
                            "status": st.column_config.SelectboxColumn(
                                "Status", 
                                options=["paid", "unpaid"]
                            )
                        }
                    )
                    
                    # Set session state variables for data processing
                    st.session_state.students = df[['student_name', 'class', 'phone', 'email', 'parent_name']].rename(
                        columns={'student_name': 'name', 'class': 'class'}
                    )
                    st.session_state.payments = df[['status']].rename(columns={'status': 'payment_status'})
                    
                    # Add a student_id column using the DataFrame index
                    st.session_state.students['student_id'] = st.session_state.students.index
                    st.session_state.payments['student_id'] = st.session_state.students.index
                    
                    # Then create paid_students
                    st.session_state.paid_students = st.session_state.students[
                        st.session_state.payments['payment_status'] == 'paid'
                    ]
                    
                    # Add this to ensure contact info is properly processed
                    st.session_state.students['phone'] = st.session_state.students['phone'].fillna('No phone provided')
                    st.session_state.students['email'] = st.session_state.students['email'].fillna('No email provided')
                    st.session_state.students['parent_name'] = st.session_state.students['parent_name'].fillna('Parent/Guardian')
                    
                    # Handle edits
                    if st.button("Save Changes", key="save_lunch_changes"):
                        conn = sqlite3.connect(DB_NAME)
                        c = conn.cursor()
                        
                        # Create a mapping of row IDs to original rows
                        id_to_original = {row['id']: row for _, row in df.iterrows()}
                        
                        for _, edited_row in edited_df.iterrows():
                            original_row = id_to_original.get(edited_row['id'])
                            
                            # Skip if new row (no original exists)
                            if original_row is None:
                                continue
                                
                            updates = {}
                            for col in df.columns:
                                if col not in ['id', 'delete'] and edited_row[col] != original_row[col]:
                                    updates[col] = edited_row[col]
                            
                            if updates:
                                set_clause = ", ".join([f"{col} = ?" for col in updates.keys()])
                                values = list(updates.values()) + [edited_row['id']]
                                c.execute(f"UPDATE lunch_payments SET {set_clause} WHERE id = ?", values)
                        
                        conn.commit()
                        conn.close()
                        st.success("Changes saved!")
                        st.rerun()
                    
                    # Handle deletes
                    if st.button("Delete Selected", key="delete_lunch_records"):
                        deleted_count = 0
                        conn = sqlite3.connect(DB_NAME)
                        c = conn.cursor()
                        for index, row in edited_df.iterrows():
                            if row['delete']:
                                c.execute("DELETE FROM lunch_payments WHERE id = ?", (row['id'],))
                                deleted_count += 1
                        conn.commit()
                        conn.close()
                        
                        if deleted_count > 0:
                            st.success(f"Deleted {deleted_count} records")
                            st.rerun()
                else:
                    st.info("No lunch payment records found")

        with col2:
            st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
            if 'paid_students' in st.session_state and not st.session_state.paid_students.empty:
                paid = st.session_state.paid_students
                total_students = len(st.session_state.get('students', []))
                if total_students > 0:
                    payment_rate = (len(paid) / total_students) * 100
                    summary_data = {
                        "Metric": ["Total Students", "Paid Students", "Payment Rate"],
                        "Value": [str(total_students), str(len(paid)), f"{payment_rate:.1f}%"]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.table(summary_df)
                if 'class' in paid.columns:
                    classes = paid['class'].value_counts()
                    st.metric("Classes", len(classes))
                    with st.expander("Class Breakdown"):
                        for cls, count in classes.items():
                            st.write(f"**{cls}:** {count}")
            else:
                st.info("No student data loaded yet")

    with tab2:
        st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)
        
        # Determine AI provider for this tab
        current_ai_provider = st.session_state.get("ticket_ai_provider", "Local (Ollama)")
        openrouter_api_key = st.session_state.get("ticket_openrouter_key", "")
        openrouter_model = st.session_state.get("ticket_openrouter_model", "")
        
        if not ollama_models and current_ai_provider == "Local":
            # Silent handling - no warning displayed
            if not get_ollama_models():
                st.info("💡 Switching to Cloud AI - Local AI unavailable")
        elif current_ai_provider == "Cloud (OpenRouter)" and not openrouter_api_key:
            st.warning("⚠️ OpenRouter API key required for cloud AI")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Data Analysis")
            if st.button("Analyze Data", key="analyze_data_button", type="primary"):
                students = st.session_state.get('students', pd.DataFrame())
                payments = st.session_state.get('payments', pd.DataFrame())
                if not students.empty and not payments.empty:
                    total = len(students)
                    paid = len(payments[payments['payment_status'] == 'paid'])
                    unpaid_students = total - paid
                    payment_rate = (paid / total * 100) if total > 0 else 0
                    class_info = ""
                    if 'class' in students.columns:
                        class_counts = students['class'].value_counts()
                        class_info = f"\n- Classes: {', '.join([f'{cls} ({cnt})' for cls, cnt in class_counts.head(5).items()])}"
                    
                    prompt = f"""
Analyze this school lunch program data:
- Total Students: {total}
- Students Who Paid: {paid}
- Payment Rate: {payment_rate:.1f}%{class_info}

Provide a brief analysis covering:
1. Payment collection effectiveness
2. Key insights and recommendations
3. Any potential concerns or areas for improvement

Keep response concise and actionable.
"""
                    # Handle both AI providers
                    if current_ai_provider == "Local (Ollama)" and generator.ollama_model:
                        api_payload = {
                            "model": generator.ollama_model,
                            "messages": [
                                {"role": "system", "content": "You are a helpful school administration expert providing data insights."},
                                {"role": "user", "content": prompt}
                            ],
                            "stream": True
                        }
                        with st.spinner("🤖 AI analyzing data..."):
                            response_placeholder = st.empty()
                            accumulated_response = ""
                            try:
                                response = requests.post(f"{get_ollama_host()}/api/chat", json=api_payload, headers={"Content-Type": "application/json"}, stream=True, timeout=60)
                                response.raise_for_status()
                                for line in response.iter_lines():
                                    if line:
                                        data = json.loads(line.decode("utf-8"))
                                        if "message" in data and "content" in data["message"]:
                                            accumulated_response += data["message"]["content"]
                                            response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                st.session_state.ticket_messages.append({"role": "assistant", "content": accumulated_response})
                            except requests.exceptions.Timeout:
                                st.error("AI server timed out after 10 seconds.")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    
                    elif current_ai_provider == "Cloud (OpenRouter)" and openrouter_api_key and openrouter_model:
                        messages = [
                            {"role": "system", "content": "You are a helpful school administration expert providing data insights."},
                            {"role": "user", "content": prompt}
                        ]
                        with st.spinner("☁️ Cloud AI analyzing data..."):
                            response_placeholder = st.empty()
                            accumulated_response = ""
                            try:
                                response = call_openrouter_api(messages, openrouter_model, openrouter_api_key)
                                if response:
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
                                    st.session_state.ticket_messages.append({"role": "assistant", "content": accumulated_response})
                                else:
                                    st.error("Failed to get response from OpenRouter")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.info("📝 Please configure AI provider first")
                else:
                    st.info("📝 Please load student data first")
        
        with col2:
            st.subheader("📧 Payment Reminders")
            if st.button("✉️ Generate Reminders", type="primary", key="generate_reminders_button"):
                students = st.session_state.get('students', pd.DataFrame())
                payments = st.session_state.get('payments', pd.DataFrame())
                
                if not students.empty and not payments.empty:
                    # Find unpaid students
                    paid_ids = payments[payments['payment_status'] == 'paid']['student_id']
                    unpaid = students[~students['student_id'].isin(paid_ids)]
                    
                    if not unpaid.empty:
                        with st.spinner("✉️ Generating reminders..."):
                            sender_info = st.session_state.get('sender_info', {
                                'name': 'School Administrator',
                                'position': 'Administrator',
                                'date': date.today().strftime('%B %d, %Y'),
                                'due_date': (date.today() + pd.Timedelta(days=7)).strftime('%B %d, %Y')
                            })
                            
                            # REMINDER GENERATION - FIXED SECTION WITH PROPER AI INTEGRATION
                            # Generate and store reminders in session state
                            st.session_state.current_reminders = []
                            for _, student in unpaid.iterrows():
                                phone = str(student.get('phone', 'No phone provided')).strip()
                                email = str(student.get('email', 'No email provided')).strip().lower()
                                parent_name = str(student.get('parent_name', 'Parent/Guardian')).strip()

                                # Phone validation
                                has_valid_phone = False
                                if phone != 'No phone provided' and phone != 'nan' and phone:
                                    clean_phone = re.sub(r'\D', '', phone)
                                    if clean_phone.isdigit() and len(clean_phone) >= 7:
                                        has_valid_phone = True
                                    else:
                                        phone = 'Invalid phone format'

                                # Email validation
                                has_valid_email = False
                                if email != 'No email provided' and email != 'nan' and email:
                                    if EMAIL_PATTERN.match(email):
                                        has_valid_email = True
                                    elif '@' in email and '.' in email.split('@')[-1]:
                                        has_valid_email = True
                                
                                # Prepare prompt
                                prompt = f"""
                            Write the body of a polite payment reminder email (without subject line) for:
                            Student: {student.get('name', 'Student')}
                            Class: {student.get('class', 'N/A')}
                            Parent/Guardian: {parent_name}

                            Create a respectful message about their child's lunch payment.
                            Include that payment is due by {sender_info.get('due_date', 'soon')}.
                            Keep it under 100 words, professional but friendly.
                            DO NOT INCLUDE A SUBJECT LINE.

                            End the message with:

                            Best regards,
                            {sender_info['name']}
                            {sender_info['position']}
                            {sender_info['date']}

                            For questions, please contact the school office.
                            """
                                system_prompt = "You are writing friendly payment reminders to parents."
                                
                                # Initialize reminder_text with fallback
                                reminder_text = f"""Dear {parent_name},

                            This is a reminder that your child {student.get('name', 'Student')} from {student.get('class', 'N/A')} has an outstanding lunch payment.

                            Please submit the payment by {sender_info.get('due_date', 'soon')} to ensure uninterrupted meal service.

                            Best regards,
                            {sender_info['name']}
                            {sender_info['position']}
                            {sender_info['date']}

                            For questions, please contact the school office."""
                                
                                # Generate reminder based on selected provider
                                if current_ai_provider == "Local" and generator.ollama_model:
                                    api_payload = {
                                        "model": generator.ollama_model,
                                        "messages": [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": prompt}
                                        ],
                                        "stream": False,
                                        "options": {"temperature": 0.5, "num_predict": 300}
                                    }
                                    try:
                                        response = requests.post(
                                            f"{get_ollama_host()}/api/chat",
                                            json=api_payload,
                                            headers={"Content-Type": "application/json"},
                                            timeout=60
                                        )
                                        response.raise_for_status()
                                        ai_response = response.json().get('message', {}).get('content', '')
                                        if ai_response:
                                            reminder_text = sanitize_text(ai_response)
                                    except requests.exceptions.Timeout:
                                        pass  # Keep fallback reminder
                                    except Exception:
                                        pass  # Keep fallback reminder

                                elif current_ai_provider == "Cloud":
                                    # Auto-load API key if not already loaded
                                    if not openrouter_api_key:
                                        try:
                                            openrouter_api_key = st.secrets.get("general", {}).get("OPENROUTER_API_KEY", "")
                                        except Exception:
                                            pass
                                    
                                    if openrouter_api_key and openrouter_model:
                                        messages = [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": prompt}
                                        ]
                                        try:
                                            response = call_openrouter_api(messages, openrouter_model, openrouter_api_key)
                                            if response:
                                                ai_response = ""
                                                for chunk in response.iter_lines():
                                                    if chunk:
                                                        if chunk == b'':
                                                            continue
                                                        if chunk.startswith(b'data:'):
                                                            try:
                                                                data = json.loads(chunk.decode("utf-8")[5:])
                                                                if "choices" in data and len(data["choices"]) > 0:
                                                                    delta = data["choices"][0].get("delta", {})
                                                                    ai_response += delta.get("content", "")
                                                            except json.JSONDecodeError:
                                                                pass
                                                if ai_response:
                                                    reminder_text = ai_response
                                        except Exception:
                                            pass  # Keep fallback reminder
                                
                                # Add reminder to session state
                                st.session_state.current_reminders.append({
                                    'name': student.get('name'),
                                    'class': student.get('class'),
                                    'parent_name': parent_name,
                                    'phone': phone,
                                    'email': email,
                                    'reminder': reminder_text,
                                    'has_phone': has_valid_phone,
                                    'has_email': has_valid_email
                                })

                            st.session_state.email_status = {"individual": {}, "bulk": None}  # Reset status
                            st.rerun()
                    else:
                        st.success("🎉 All students have paid!")
                else:
                    st.info("📝 Please load student data first")
            
            # Display existing reminders if available
            if st.session_state.get('current_reminders'):
                reminders = st.session_state.current_reminders
                st.markdown("### 📝 Payment Reminders")
                
                # Summary stats
                total_reminders = len(reminders)
                with_phone = sum(1 for r in reminders if r['has_phone'])
                with_email = sum(1 for r in reminders if r['has_email'])
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                col_stats1.metric("Total Reminders", total_reminders)
                col_stats2.metric("📱 With Phone", with_phone)
                col_stats3.metric("📧 With Email", with_email)
                
                # Display reminders with contact info and actions
                for i, reminder in enumerate(reminders):
                    with st.expander(f"📧 {reminder['name']} ({reminder['class']})", expanded=False):
                        # Contact information display
                        col_contact, col_actions = st.columns([2, 1])
                        
                        with col_contact:
                            st.write(f"**Parent:** {reminder['parent_name']}")
                            if reminder['has_phone']:
                                st.write(f"📱 **Phone:** {reminder['phone']}")
                            else:
                                st.write("📱 Phone: Not available")
                            if reminder['has_email']:
                                st.write(f"📧 **Email:** {reminder['email']}")
                            else:
                                st.write("📧 Email: Not available")
                        
                        with col_actions:
                            if reminder['has_phone']:
                                if st.button(f"📱 SMS", key=f"sms_{i}", help="Send SMS reminder"):
                                    st.info(f"SMS feature coming soon for {reminder['phone']}")
                            
                            email_invalid = not EMAIL_PATTERN.match(reminder['email']) if reminder['has_email'] else True
                            email_disabled = not st.session_state.get('gmail_connected', False) or email_invalid
                            email_help = "Invalid email format" if email_invalid else "Send email reminder"
                            if st.button(f"📧 Email", key=f"email_{i}", help=email_help, disabled=email_disabled):
                                if not email_disabled:
                                    with st.spinner("Sending email..."):
                                        success, message = generator.gmail.send_email(
                                            reminder['email'],
                                            f"Lunch Payment Reminder - {reminder['name']}",
                                            reminder['reminder'],
                                            st.session_state.gmail_sender_name_input
                                        )
                                        # Store status in session state
                                        st.session_state.email_status["individual"][i] = (success, message)
                                        st.rerun()
                                else:
                                    st.error("Gmail not connected or email invalid")
                            
                            if st.button(f"📋 Copy", key=f"copy_{i}", help="Copy message to clipboard"):
                                st.code(reminder['reminder'], language=None)
                        
                        # Display email status if exists
                        if i in st.session_state.email_status["individual"]:
                            success, message = st.session_state.email_status["individual"][i]
                            if success:
                                st.markdown(f'<div class="email-status email-success">{message}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="email-status email-error">{message}</div>', unsafe_allow_html=True)
                        
                        st.markdown("**Message:**")
                        st.write(reminder['reminder'])
                        st.code(reminder['reminder'], language=None)
                
                # Bulk Actions Section
                st.markdown("### 📧 Bulk Actions")
                
                # Send All Reminders Button - Moved from sidebar
                col_bulk, col_download = st.columns(2)
                
                with col_bulk:
                    # Check if reminders exist and Gmail is connected
                    reminders_exist = bool(st.session_state.get('current_reminders', []))
                    gmail_ready = st.session_state.get('gmail_connected', False) and generator.gmail.connected
                    send_disabled = not (reminders_exist and gmail_ready)
                    
                    if st.button("📧 Send All Reminders", 
                                 key="send_all_reminders_main",
                                 disabled=send_disabled,
                                 help="Send all generated reminders via email",
                                 use_container_width=True,
                                 type="primary"):
                        if st.session_state.get('gmail_connected', False) and st.session_state.get('current_reminders'):
                            with st.spinner(f"Sending {len(st.session_state.current_reminders)} reminder emails..."):
                                email_reminders = [r for r in st.session_state.current_reminders if r['has_email']]
                                results = generator.gmail.send_bulk_reminders(
                                    email_reminders, 
                                    st.session_state.gmail_sender_name_input
                                )
                                st.session_state.email_status["bulk"] = results
                                st.success(f"✅ Sent {results['success']} emails successfully")
                                if results["failed"] > 0:
                                    st.error(f"❌ Failed to send {results['failed']} emails")
                                st.rerun()
                        else:
                            st.error("Gmail not connected or no reminders generated")
                
                with col_download:
                    # Download button
                    if st.button("📄 Download All", key="download_reminders_button", use_container_width=True):
                        reminder_text = "\n\n" + "="*50 + "\n\n"
                        reminder_text = reminder_text.join([
                            f"REMINDER FOR: {r['name']} ({r['class']})\n"
                            f"Parent: {r['parent_name']}\n"
                            f"Phone: {r['phone']}\n"
                            f"Email: {r['email']}\n\n"
                            f"MESSAGE:\n{r['reminder']}"
                            for r in reminders
                        ])
                        st.download_button(
                            "📥 Download Reminders",
                            data=reminder_text,
                            file_name=f"payment_reminders_{date.today().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                
                # Status display for bulk send
                if st.session_state.email_status.get("bulk"):
                    results = st.session_state.email_status["bulk"]
                    st.success(f"📊 Bulk Send Results: {results['success']} sent, {results['failed']} failed")

        st.subheader("💬 Chat with AI")
        for msg in st.session_state.ticket_messages:
            with st.chat_message(msg['role']):
                parts = split_message(msg['content'])
                for part in parts:
                    if part["type"] == "text":
                        st.markdown(part["content"], unsafe_allow_html=True)
                    elif part["type"] == "code":
                        st.code(part["code"], language=part["language"])
        if prompt := st.chat_input("Ask about your lunch program..."):
            st.session_state.ticket_messages.append({'role': 'user', 'content': prompt})
            students_df = st.session_state.get('students', pd.DataFrame())
            payments_df = st.session_state.get('payments', pd.DataFrame())
            if not students_df.empty and not payments_df.empty:
                total_students = len(students_df)
                paid_students_count = len(payments_df[payments_df['payment_status'] == 'paid'])
                unpaid_students = total_students - paid_students_count
                payment_rate = (paid_students_count / total_students * 100) if total_students > 0 else 0
                paid_students_df = st.session_state.get('paid_students', pd.DataFrame())
                paid_names = paid_students_df['name'].tolist() if not paid_students_df.empty else []
                paid_ids = payments_df[payments_df['payment_status'] == 'paid']['student_id']
                unpaid_names = students_df[~students_df['student_id'].isin(paid_ids)]['name'].tolist() if not students_df.empty else []
                class_info = ""
                if 'class' in students_df.columns:
                    class_counts = students_df['class'].value_counts()
                    paid_by_class = paid_students_df['class'].value_counts() if not paid_students_df.empty else pd.Series()
                    class_breakdown = [f"{cls}: {paid_by_class.get(cls, 0)}/{total} paid" for cls, total in class_counts.head(10).items()]
                    class_info = "\n- " + "\n".join(class_breakdown)
                
                prompt_text = f"""
Current Lunch Payment Data:
- Total Students: {total_students}
- Students Who Paid: {paid_students_count}
- Students Still Need to Pay: {unpaid_students}
- Payment Rate: {payment_rate:.1f}%

Students Who Have Paid (Names):
{', '.join(paid_names) if paid_names else 'None'}

Students Who Haven't Paid (Names):
{', '.join(unpaid_names) if unpaid_names else 'None'}

Class Breakdown:{class_info}

User Question: {prompt}
"""
                system_prompt = "You are analyzing actual school lunch payment data. Use the provided data to answer questions specifically and accurately. Be direct and practical in your responses."
                
                # Handle both AI providers
                if current_ai_provider == "Local (Ollama)" and generator.ollama_model:
                    api_payload = {
                        "model": generator.ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_text}
                        ],
                        "stream": True
                    }
                    with st.spinner("🤖️ AI thinking..."):
                        response_placeholder = st.empty()
                        accumulated_response = ""
                        try:
                            response = requests.post(f"{get_ollama_host()}/api/chat", json=api_payload, headers={"Content-Type": "application/json"}, stream=True, timeout=60)
                            response.raise_for_status()
                            for line in response.iter_lines():
                                if line:
                                    data = json.loads(line.decode("utf-8"))
                                    if "message" in data and "content" in data["message"]:
                                        accumulated_response += data["message"]["content"]
                                        response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                            st.session_state.ticket_messages.append({'role': 'assistant', 'content': accumulated_response})
                        except requests.exceptions.Timeout:
                            st.error("AI server timed out after 10 seconds.")
                            st.session_state.ticket_messages.append({'role': 'assistant', 'content': "AI server timed out after 10 seconds."})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.session_state.ticket_messages.append({'role': 'assistant', 'content': f"Error: {str(e)}"})
                        st.rerun()
                
                elif current_ai_provider == "Cloud (OpenRouter)" and openrouter_api_key and openrouter_model:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ]
                    with st.spinner("☁️ Cloud AI thinking..."):
                        response_placeholder = st.empty()
                        accumulated_response = ""
                        try:
                            response = call_openrouter_api(messages, openrouter_model, openrouter_api_key)
                            if response:
                                for chunk in response.iter_lines():
                                    if chunk:
                                        if chunk == b'':
                                            continue
                                        if chunk.startswith(b'data:'):
                                            try:
                                                data = json.loads(chunk.decode("utf-8")[5:])
                                                if "choices" in data and len(data["choices"]) > 0:
                                                    delta = data["choices"][0].get("delta", {})
                                                    accumulated_response += delta.get("content", "")
                                                    response_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
                                            except json.JSONDecodeError:
                                                pass
                                st.session_state.ticket_messages.append({'role': 'assistant', 'content': accumulated_response})
                            else:
                                error_msg = "Failed to get response from OpenRouter"
                                st.error(error_msg)
                                st.session_state.ticket_messages.append({'role': 'assistant', 'content': error_msg})
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.ticket_messages.append({'role': 'assistant', 'content': error_msg})
                        st.rerun()
                
                else:
                    st.info("📝 Please configure AI provider first")
            else:
                st.info("📝 Please load data for specific analysis.")

    with tab3:
        st.markdown('<div class="section-header">Generate Tickets</div>', unsafe_allow_html=True)
        if 'paid_students' in st.session_state and not st.session_state.paid_students.empty:
            paid_count = len(st.session_state.paid_students)
            st.success(f"✅ Ready to generate tickets for {paid_count} paid students")
            st.info(f"📅 {validity_info['display']}")
            with st.expander("👀 Preview Student List"):
                st.dataframe(st.session_state.paid_students[['name', 'class', 'student_id']], use_container_width=True)
            if st.button("🖨 Generate PDF Tickets", key="generate_pdf_tickets_button", type="primary", use_container_width=True):
                with st.spinner("🎫 Generating tickets..."):
                    try:
                        school_info = {'name': school_name}
                        pdf_buffer = generator.generate_pdf(st.session_state.paid_students, school_info, ticket_date, validity_info)
                        st.success("✅ Tickets generated successfully!")
                        filename = f"lunch_tickets_{school_name.replace(' ', '_')}_{ticket_date.strftime('%Y%m%d')}.pdf"
                        st.download_button("📥 Download PDF Tickets", data=pdf_buffer.getvalue(), file_name=filename, mime="application/pdf", key="download_pdf_tickets_button", type="primary", use_container_width=True)
                        st.info(f"📊 Generated {paid_count} tickets ({(paid_count + 14) // 15} pages)")
                    except Exception as e:
                        st.error(f"❌ Error generating tickets: {str(e)}")
        else:
            st.info("📝 Please load and process student data first to generate tickets")
            with st.expander("ℹ️ How to get started"):
                st.markdown("""
1. Go to the **Data Management** tab
2. Load or import your student and payment data
3. Ensure you have students marked as 'paid'
4. Click the 'Generate PDF Tickets' button
""")


# Database Management
elif st.session_state.current_section == "Database Management":
    st.markdown('<div class="section-header">Database Management</div>', unsafe_allow_html=True)
    
    # Table selection
    tables = ["students", "staff", "attendance", "test_results", "fees", "expenditures", 
                "terms", "courses", "enrollments", "library", "behaviour", "student_reports",
                "classes", "class_members", "lunch_payments"]  # Added new tables
    selected_table = st.selectbox("Select Table", tables, key="db_table_select")
  

  
    # Display current schema
    st.markdown('<div class="section-header">Current Schema</div>', unsafe_allow_html=True)
    schema = get_table_schema(selected_table)
    if schema:
        schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
        st.dataframe(schema_df[["name", "type"]], use_container_width=True)
    else:
        st.info(f"No schema found for table: {selected_table}")
    
    # Field management
    st.markdown('<div class="section-header">Field Management</div>', unsafe_allow_html=True)
    
    # Combined Field Management
    with st.expander("Manage Fields", expanded=False):
        with st.form(key="field_management_form"):
            # Operation selection
            operation = st.radio(
                "Select Operation", 
                ["Add Field", "Rename Field", "Delete Field"], 
                horizontal=True,
                key="field_operation"
            )
            
            if operation == "Add Field":
                st.subheader("Add New Field")
                col1, col2 = st.columns(2)
                with col1:
                    field_name = st.text_input("Field Name", key="field_name")
                with col2:
                    field_types = ["TEXT", "INTEGER", "REAL", "DATE", "BOOLEAN"]
                    field_type = st.selectbox("Field Type", field_types, key="field_type")
                    
            elif operation == "Rename Field" and schema:
                st.subheader("Rename Field")
                col1, col2 = st.columns(2)
                with col1:
                    old_name = st.selectbox("Select Field", [col[1] for col in schema], key="old_field_name")
                with col2:
                    new_name = st.text_input("New Name", key="new_field_name")
                    
            elif operation == "Delete Field" and schema:
                st.subheader("Delete Field")
                field_to_delete = st.selectbox("Select Field to Delete", [col[1] for col in schema], key="field_to_delete")
                
            # Submit button with dynamic label
            submit_label = f"{operation.split()[0]} Field"
            button_type = "primary" if operation == "Delete Field" else "secondary"
            
            if st.form_submit_button(submit_label, type=button_type):
                # Handle operations
                if operation == "Add Field":
                    if field_name:
                        success, message = add_column_to_table(selected_table, field_name, field_type)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(f"Error: {message}")
                    else:
                        st.error("Field name is required")
                        
                elif operation == "Rename Field":
                    if old_name and new_name:
                        success, message = rename_column(selected_table, old_name, new_name)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(f"Error: {message}")
                    else:
                        st.error("Both fields are required")
                        
                elif operation == "Delete Field":
                    if field_to_delete:
                        success, message = delete_column(selected_table, field_to_delete)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(f"Error: {message}")
                    else:
                        st.error("Please select a field to delete")


    # NEW EXPORT FEATURE STARTS HERE
    with st.expander("📤 Export Data", expanded=False):
        st.subheader("Export Table Data")
        st.info(f"Export all records from the '{selected_table}' table as CSV")
        
        if st.button(f"💾 Export {selected_table.capitalize()} Data", 
                     key="export_data_button",
                     type="secondary",
                     use_container_width=True):
            try:
                conn = sqlite3.connect(DB_NAME)
                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                conn.close()
                
                if not df.empty:
                    csv = df.to_csv(index=False).encode('utf-8')
                    today = date.today().strftime("%Y%m%d")
                    filename = f"{selected_table}_export_{today}.csv"
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv",
                        key="download_csv_button"
                    )
                    st.success(f"✅ Exported {len(df)} records")
                else:
                    st.info(f"No data found in {selected_table} table")
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")

    # Add this after the "Export Data" expander in the Database Management section
    with st.expander("📥 Import Data", expanded=False):
        st.subheader("Import Data into Table")
        st.info(f"Import data into the '{selected_table}' table from a CSV file")
        
        uploaded_file = st.file_uploader(
            f"Upload CSV file for {selected_table}",
            type=['csv'],
            key=f"import_{selected_table}_uploader"
        )
        
        if uploaded_file is not None:
            # Show file preview
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
               # with st.expander("👀 Preview Data (first 5 rows)"):
                st.dataframe(df.head(5))
                    
                # Column mapping interface
                st.subheader("Column Mapping")
                st.info("Map CSV columns to database columns")
                
                # Get table schema
                schema = get_table_schema(selected_table)
                db_columns = [col[1] for col in schema] if schema else []
                
                if not db_columns:
                    st.error(f"Could not retrieve schema for {selected_table}")
                else:
                    mapping = {}
                    for csv_col in df.columns:
                        st.markdown(f"**CSV Column:** `{csv_col}`")
                        selected_db_col = st.selectbox(
                            f"Map to database column:",
                            ["(Ignore)"] + db_columns,
                            key=f"map_{csv_col}",
                            index=0
                        )
                        if selected_db_col != "(Ignore)":
                            mapping[csv_col] = selected_db_col
                    
                    # Import button
                    if st.button("📥 Import Data", key=f"import_{selected_table}_button"):
                        if not mapping:
                            st.error("Please map at least one column")
                        else:
                            with st.spinner("Importing data..."):
                                try:
                                    # Create a new DataFrame with mapped columns
                                    mapped_df = df.rename(columns=mapping)[list(mapping.values())]
                                    
                                    conn = sqlite3.connect(DB_NAME)
                                    mapped_df.to_sql(
                                        name=selected_table, 
                                        con=conn, 
                                        if_exists='append', 
                                        index=False
                                    )
                                    conn.close()
                                    
                                    st.success(f"✅ Imported {len(df)} records into {selected_table}")
                                    query_db.cache_clear()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Import error: {str(e)}")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    # Delete All Data Feature
    st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
    with st.expander("⚠️ Delete All Data", expanded=False):
        st.warning("This will permanently delete ALL records from the selected table. This action cannot be undone!")
        
        # Confirmation mechanism
        confirm_text = st.text_input(
            f"Type 'DELETE {selected_table.upper()}' to confirm", 
            key="delete_confirm_input",
            placeholder=f"DELETE {selected_table.upper()}"
        )
        
        if st.button("🗑️ Delete All Records", 
                    type="primary", 
                    disabled=(confirm_text != f"DELETE {selected_table.upper()}"),
                    key="delete_all_data_button",
                    help="Confirm by typing the required text above"):
            try:
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute(f"DELETE FROM {selected_table}")
                conn.commit()
                
                # Reset autoincrement counter for SQLite
                if selected_table != "terms":  # terms table doesn't have autoincrement
                    c.execute(f"DELETE FROM sqlite_sequence WHERE name = '{selected_table}'")
                    conn.commit()
                
                conn.close()
                query_db.cache_clear()
                st.success(f"✅ All records deleted from {selected_table} table!")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting data: {str(e)}")
    



# Migration Script: Move from students.room to class_members table
# Add this as a separate section or run it once to migrate your data
#Tools
elif st.session_state.current_section == "Tools":
    st.markdown('<div class="section-header">Tools</div>', unsafe_allow_html=True)
    
    # Migration Script: Move from students.room to class_members table
    # Add this as a separate section or run it once to migrate your data

    def migrate_room_assignments_to_class_members():
        """
        Migrate existing room assignments from students table to class_members table
        """
        st.markdown("### 🔄 Data Migration Tool")
        st.markdown("This tool will migrate students from room-based assignments to the proper class_members table.")
        
        # Add diagnostic section
        if st.expander("🔍 Database Diagnostics", expanded=False):
            show_database_diagnostics()
        
        # Show current state
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Room Assignments")
            room_stats_query = """
                SELECT 
                    COALESCE(room, 'Unassigned') as room,
                    COUNT(*) as student_count
                FROM students 
                GROUP BY room
                ORDER BY room
            """
            room_stats_result, _ = query_db(room_stats_query)
            
            if room_stats_result:
                room_stats_df = pd.DataFrame(room_stats_result, columns=['Room', 'Students'])
                st.dataframe(room_stats_df)
            
        with col2:
            st.subheader("Class Members Table")
            class_members_stats_query = """
                SELECT 
                    COUNT(*) as total_memberships,
                    COUNT(DISTINCT student_id) as unique_students,
                    COUNT(DISTINCT class_id) as classes_with_students
                FROM class_members
            """
            cm_stats_result, _ = query_db(class_members_stats_query)
            
            if cm_stats_result:
                total_memberships, unique_students, classes_with_students = cm_stats_result[0]
                st.metric("Total Memberships", total_memberships)
                st.metric("Unique Students", unique_students)
                st.metric("Classes with Students", classes_with_students)
        
        st.markdown("---")
        
        # Migration options
        st.subheader("Migration Options")
        
        # Get available terms for migration
        term_results, _ = query_db("SELECT id, name FROM terms ORDER BY name")
        if term_results:
            term_options = {row[1]: row[0] for row in term_results}
            
            migration_term = st.selectbox(
                "Select term for migration", 
                list(term_options.keys()),
                key="migration_term_select",
                help="Students will be enrolled in classes for this term"
            )
            migration_term_id = term_options[migration_term]
            
            # Show migration status for selected term
            st.markdown("#### Migration Status")
            unmigrated_count, rooms_without_classes = check_migration_status(migration_term_id, migration_term)
            
            st.markdown("---")
            
            # Migration buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🏫 Create Missing Classes", key="create_missing_classes", 
                            disabled=rooms_without_classes == 0):
                    create_classes_for_rooms(migration_term_id, migration_term)
            
            with col2:
                if st.button("🔄 Migrate Students", key="migrate_rooms", type="primary",
                            disabled=unmigrated_count == 0):
                    migrate_students_by_room(migration_term_id, migration_term)
            
            with col3:
                if st.button("🧹 Clean Up Room Data", key="cleanup_rooms"):
                    cleanup_room_assignments()
                    
            # Help text
            if rooms_without_classes > 0:
                st.info("💡 Create missing classes first, then migrate students")
            elif unmigrated_count > 0:
                st.success("✅ All classes exist! Ready to migrate students")
            else:
                st.success("✅ All students are properly assigned to classes")
                
        else:
            st.warning("No terms available. Please create terms first.")

    def check_migration_status(term_id, term_name):
        """Check and display migration status"""
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Students with room assignments but not in class_members for this term
        unmigrated_query = """
            SELECT COUNT(*) 
            FROM students s
            WHERE s.room IS NOT NULL AND s.room != ''
            AND s.id NOT IN (
                SELECT student_id FROM class_members WHERE term_id = ?
            )
        """
        unmigrated_count = c.execute(unmigrated_query, (term_id,)).fetchone()[0]
        
        # Students already in class_members for this term
        migrated_query = """
            SELECT COUNT(DISTINCT cm.student_id)
            FROM class_members cm
            JOIN students s ON cm.student_id = s.id
            WHERE cm.term_id = ? AND s.room IS NOT NULL
        """
        migrated_count = c.execute(migrated_query, (term_id,)).fetchone()[0]
        
        # Classes without rooms vs rooms without classes
        classes_without_rooms = c.execute("""
            SELECT COUNT(*) FROM classes 
            WHERE term_id = ? AND (room IS NULL OR room = '')
        """, (term_id,)).fetchone()[0]
        
        # Check for rooms that actually need classes (more precise check)
        rooms_needing_classes = c.execute("""
            SELECT COUNT(DISTINCT s.room)
            FROM students s
            WHERE s.room IS NOT NULL AND s.room != ''
            AND s.room NOT IN (
                SELECT DISTINCT room FROM classes 
                WHERE term_id = ? AND room IS NOT NULL AND room != ''
            )
        """, (term_id,)).fetchone()[0]
        
        # Also check which specific rooms are missing classes for debugging
        missing_rooms = c.execute("""
            SELECT DISTINCT s.room
            FROM students s
            WHERE s.room IS NOT NULL AND s.room != ''
            AND s.room NOT IN (
                SELECT DISTINCT room FROM classes 
                WHERE term_id = ? AND room IS NOT NULL AND room != ''
            )
        """, (term_id,)).fetchall()
        
        conn.close()
        
        # Display status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Students to Migrate", unmigrated_count)
        with col2:
            st.metric("Already Migrated", migrated_count)
        with col3:
            st.metric("Classes w/o Rooms", classes_without_rooms)
        with col4:
            st.metric("Rooms w/o Classes", rooms_needing_classes)
        
        # Debug info
        if missing_rooms:
            st.info(f"Rooms missing classes: {', '.join([r[0] for r in missing_rooms])}")
        
        return unmigrated_count, rooms_needing_classes

    def create_classes_for_rooms(term_id, term_name):
        """Create classes for rooms that don't have corresponding classes"""
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Get unique rooms from students table
        rooms_result = c.execute("""
            SELECT DISTINCT room 
            FROM students 
            WHERE room IS NOT NULL AND room != ''
        """).fetchall()
        
        # Get existing classes for this term
        existing_classes = c.execute("""
            SELECT room FROM classes WHERE term_id = ? AND room IS NOT NULL
        """, (term_id,)).fetchall()
        existing_rooms = {row[0] for row in existing_classes}
        
        # Also check for classes with same room in ANY term (for debugging)
        all_classes_with_room = c.execute("""
            SELECT room, term_id, name FROM classes 
            WHERE room IS NOT NULL AND room != ''
        """).fetchall()
        
        # Get default teacher
        teacher_result = c.execute("SELECT id FROM staff LIMIT 1").fetchone()
        default_teacher_id = teacher_result[0] if teacher_result else 1
        
        created_count = 0
        updated_count = 0
        errors = []
        
        for (room,) in rooms_result:
            if room not in existing_rooms:
                # Check if class exists in another term
                existing_in_other_term = [cls for cls in all_classes_with_room if cls[0] == room]
                
                if existing_in_other_term:
                    # Class exists in another term, offer to update or create new one
                    other_term_class = existing_in_other_term[0]
                    st.warning(f"Room {room} already has a class '{other_term_class[2]}' in term {other_term_class[1]}")
                    
                    # Check if we can update the existing class to this term
                    try:
                        # Try to update the existing class to the current term
                        c.execute("""
                            UPDATE classes 
                            SET term_id = ? 
                            WHERE room = ? AND term_id = ?
                        """, (term_id, room, other_term_class[1]))
                        
                        if c.rowcount > 0:
                            updated_count += 1
                            st.info(f"Updated existing class for room {room} to {term_name}")
                        else:
                            # If update failed, try creating with unique name
                            unique_name = f"Class {room} ({term_name})"
                            c.execute("""
                                INSERT INTO classes (name, room, teacher_id, term_id)
                                VALUES (?, ?, ?, ?)
                            """, (unique_name, room, default_teacher_id, term_id))
                            created_count += 1
                            
                    except sqlite3.IntegrityError as e:
                        errors.append(f"Room {room}: {str(e)} - Class may already exist for this term")
                        continue
                else:
                    # No existing class, create new one
                    try:
                        class_name = f"Class {room}"
                        c.execute("""
                            INSERT INTO classes (name, room, teacher_id, term_id)
                            VALUES (?, ?, ?, ?)
                        """, (class_name, room, default_teacher_id, term_id))
                        created_count += 1
                    except sqlite3.IntegrityError as e:
                        errors.append(f"Room {room}: {str(e)}")
                        continue
        
        conn.commit()
        conn.close()
        
        # Show results
        if created_count > 0:
            st.success(f"Created {created_count} new classes for {term_name}")
        if updated_count > 0:
            st.success(f"Updated {updated_count} existing classes to {term_name}")
        if created_count == 0 and updated_count == 0:
            st.info("No new classes needed")
        
        if errors:
            with st.expander("Class Creation Errors", expanded=False):
                for error in errors:
                    st.warning(error)
                    
            # Show diagnostic info
            st.markdown("#### Diagnostic Information")
            st.markdown("**All classes with rooms:**")
            if all_classes_with_room:
                for room, term_id_val, name in all_classes_with_room:
                    st.write(f"- Room {room}: '{name}' (Term {term_id_val})")
        
        if created_count > 0 or updated_count > 0:
            st.rerun()

    def migrate_students_by_room(term_id, term_name):
        """Migrate students from room assignments to class_members table"""
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Get students with room assignments who aren't already in class_members for this term
        students_with_rooms = c.execute("""
            SELECT s.id, s.room
            FROM students s
            WHERE s.room IS NOT NULL AND s.room != ''
            AND s.id NOT IN (
                SELECT student_id FROM class_members WHERE term_id = ?
            )
        """, (term_id,)).fetchall()
        
        if not students_with_rooms:
            st.info("No students need migration")
            conn.close()
            return
        
        # Get existing classes for this term and build room-to-class mapping
        existing_classes = c.execute("""
            SELECT id, room FROM classes 
            WHERE term_id = ? AND room IS NOT NULL
        """, (term_id,)).fetchall()
        
        room_to_class = {room: class_id for class_id, room in existing_classes}
        
        # Get default teacher for creating new classes if needed
        teacher_result = c.execute("SELECT id FROM staff LIMIT 1").fetchone()
        default_teacher_id = teacher_result[0] if teacher_result else 1
        
        # Create missing classes for rooms that don't exist yet
        unique_rooms = set(room for _, room in students_with_rooms)
        classes_created = 0
        
        for room in unique_rooms:
            if room not in room_to_class:
                try:
                    class_name = f"Class {room}"
                    c.execute("""
                        INSERT INTO classes (name, room, teacher_id, term_id)
                        VALUES (?, ?, ?, ?)
                    """, (class_name, room, default_teacher_id, term_id))
                    room_to_class[room] = c.lastrowid
                    classes_created += 1
                except sqlite3.IntegrityError as e:
                    # If class creation fails, try to get the existing class
                    existing_class = c.execute("""
                        SELECT id FROM classes 
                        WHERE room = ? AND term_id = ?
                    """, (room, term_id)).fetchone()
                    
                    if existing_class:
                        room_to_class[room] = existing_class[0]
                    else:
                        st.error(f"Failed to create or find class for room {room}: {str(e)}")
                        continue
        
        # Migrate students to their respective classes
        migrated_count = 0
        errors = []
        
        for student_id, room in students_with_rooms:
            if room in room_to_class:
                try:
                    c.execute("""
                        INSERT INTO class_members (class_id, student_id, term_id)
                        VALUES (?, ?, ?)
                    """, (room_to_class[room], student_id, term_id))
                    migrated_count += 1
                except sqlite3.IntegrityError as e:
                    errors.append(f"Student {student_id} (room {room}): {str(e)}")
            else:
                errors.append(f"No class found for room {room}")
        
        conn.commit()
        conn.close()
        
        # Show results
        success_msg = f"Migrated {migrated_count} students to {term_name} classes"
        if classes_created > 0:
            success_msg += f" (created {classes_created} new classes)"
        
        st.success(success_msg)
        
        if errors:
            with st.expander(f"Migration Issues ({len(errors)})", expanded=False):
                for error in errors:
                    st.warning(error)
        
        if migrated_count > 0:
            st.rerun()

    def cleanup_room_assignments():
        """Clean up room assignments after successful migration"""
        
        # Show confirmation
        st.warning("⚠️ This will clear all room assignments from the students table!")
        st.markdown("**Only do this after confirming the migration was successful.**")
        
        # Show current class_members count
        cm_count_result, _ = query_db("SELECT COUNT(*) FROM class_members")
        cm_count = cm_count_result[0][0] if cm_count_result else 0
        
        st.info(f"Current class memberships: {cm_count}")
        
        confirm_cleanup = st.checkbox("I confirm that students are properly assigned in class_members table", key="confirm_cleanup")
        
        if confirm_cleanup and st.button("Clear Room Assignments", key="clear_rooms", type="secondary"):
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            
            # Clear room assignments
            result = c.execute("UPDATE students SET room = NULL")
            cleared_count = result.rowcount
            
            conn.commit()
            conn.close()
            
            st.success(f"Cleared room assignments for {cleared_count} students")
            st.info("Students are now managed through the class_members table only")
            st.rerun()

    def show_database_diagnostics():
        """Show diagnostic information about the database state"""
        st.subheader("Database Diagnostics")
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Check for duplicate rooms across terms
        st.markdown("**Classes by Room and Term:**")
        all_classes = c.execute("""
            SELECT c.room, c.term_id, t.name as term_name, c.name as class_name, c.id
            FROM classes c
            LEFT JOIN terms t ON c.term_id = t.id
            WHERE c.room IS NOT NULL AND c.room != ''
            ORDER BY c.room, c.term_id
        """).fetchall()
        
        if all_classes:
            classes_df = pd.DataFrame(all_classes, columns=['Room', 'Term ID', 'Term Name', 'Class Name', 'Class ID'])
            st.dataframe(classes_df)
            
            # Check for rooms with multiple classes
            room_counts = {}
            for room, term_id, term_name, class_name, class_id in all_classes:
                if room not in room_counts:
                    room_counts[room] = {}
                room_counts[room][term_id] = (term_name, class_name)
            
            duplicate_rooms = {room: terms for room, terms in room_counts.items() if len(terms) > 1}
            if duplicate_rooms:
                st.warning("**Rooms with classes in multiple terms:**")
                for room, terms in duplicate_rooms.items():
                    term_info = [f"Term {tid} ({tname})" for tid, (tname, _) in terms.items()]
                    st.write(f"- Room {room}: {', '.join(term_info)}")
        
        # Show students by room
        st.markdown("**Students by Room:**")
        students_by_room = c.execute("""
            SELECT 
                COALESCE(room, 'No Room') as room,
                COUNT(*) as student_count,
                GROUP_CONCAT(name, ', ') as students
            FROM students 
            GROUP BY room
            ORDER BY room
        """).fetchall()
        
        if students_by_room:
            for room, count, students in students_by_room:
                with st.expander(f"Room {room} ({count} students)"):
                    st.write(students)
        
        # Check database constraints
        st.markdown("**Database Schema Info:**")
        schema_info = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='classes'").fetchone()
        if schema_info:
            st.code(schema_info[0], language="sql")
        
        conn.close()

    # Call the migration function
    migrate_room_assignments_to_class_members()

