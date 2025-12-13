_DJ='SELECT id FROM staff LIMIT 1'
_DI='RESTORE DATABASE'
_DH='SQLite Database (.db)'
_DG='📥 Import Data'
_DF='student_reports'
_DE='enrollments'
_DD='%B %d, %Y'
_DC='📝 Please load student data first'
_DB='📝 Please configure AI provider first'
_DA='You are a helpful school administration expert providing data insights.'
_D9='<div class="section-header">Data Management</div>'
_D8='ticket_openrouter_model'
_D7='ticket_openrouter_key'
_D6='ticket_ai_provider'
_D5='Date Range'
_D4='Single Day'
_D3='School Database'
_D2='\n\n⚠️ No results found for that query'
_D1='\n\nℹ️ Query returned no results'
_D0='query_results.csv'
_C_='💾 Download Results'
_Cz='SELECT.*?;'
_Cy='<div class="section-header">AI Assistant</div>'
_Cx='{student}'
_Cw='application/pdf'
_Cv='You are an experienced school administrator creating end-of-term reports.'
_Cu='Select Term'
_Ct='attendance_rate'
_Cs='Best Subject'
_Cr='Overall Average'
_Cq='Financial Report'
_Cp='Financial Overview'
_Co='Attendance Summary'
_Cn='Individual Student Performance'
_Cm='Student Performance'
_Cl='Expenditure by Category'
_Ck='Record Expenditure'
_Cj='Payment Date'
_Ci='Record Fee'
_Ch='Record Result'
_Cg='recommended_action'
_Cf='SELECT id, name, start_date, end_date FROM terms ORDER BY start_date DESC'
_Ce='No terms found. Please create terms first.'
_Cd='Total Students'
_Cc='\n            SELECT \n                c.id AS class_id,\n                c.name AS class_name,\n                c.room,\n                s.name AS teacher_name,\n                t.id AS term_id,\n                t.name AS term_name\n            FROM classes c\n            JOIN staff s ON c.teacher_id = s.id\n            JOIN terms t ON c.term_id = t.id\n        '
_Cb='\n                                INSERT INTO classes (name, room, teacher_id, term_id)\n                                VALUES (?, ?, ?, ?)\n                            '
_Ca='Class Name'
_CZ='SELECT id, name FROM staff'
_CY='No terms available. Please add terms first.'
_CX='Description'
_CW='Record Attendance'
_CV='Administrator'
_CU='Average Score'
_CT='No fee records found'
_CS='Fee Status Distribution'
_CR='Reset Stats'
_CQ='📊 Cache Statistics'
_CP='openrouter_model_select'
_CO='OpenRouter API Key'
_CN='get_ollama_models_cached'
_CM='http://localhost:11434'
_CL='AI Provider'
_CK='num_predict'
_CJ='You are writing friendly payment reminders to parents.'
_CI='RadioSport AI'
_CH='https://github.com/rkarikari/RadioSport-chat'
_CG='HTTP-Referer'
_CF='Authorization'
_CE='BOTTOMPADDING'
_CD='Class Average'
_CC='overall_position'
_CB='Class Teacher'
_CA='Not assigned'
_C9='SELECT status, SUM(amount) as total FROM fees GROUP BY status'
_C8='Add Field'
_C7='%Y%m%d'
_C6='AI server timed out after 10 seconds.'
_C5='payments'
_C4='paid_students'
_C3='Failed to get response from OpenRouter'
_C2='</think>'
_C1='<think>'
_C0='🤖️ AI thinking...'
_B_='text/plain'
_Bz='enable_reasoning'
_By='student'
_Bx='Class'
_Bw='Individual'
_Bv='Select Student'
_Bu='Category'
_Bt='Repeat'
_Bs='\n                                                INSERT INTO promotion_history \n                                                (student_id, from_term_id, to_term_id, from_grade, to_grade, action, promotion_date, notes)\n                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)\n                                            '
_Br='promote'
_Bq='small'
_Bp='Student Name'
_Bo='class_name'
_Bn='teacher_id'
_Bm='SELECT id, name FROM terms'
_Bl='End Date'
_Bk='Start Date'
_Bj='Term Name'
_Bi='attendance_percentage'
_Bh='address'
_Bg='teacher'
_Bf='Email'
_Be='Address'
_Bd='Date of Birth'
_Bc='OPENROUTER_API_KEY'
_Bb='general'
_Ba='Cloud'
_BZ='Tools'
_BY='Database Management'
_BX='Ticket Generator'
_BW='AI Assistant'
_BV='End of Term Reports'
_BU='Reports'
_BT='Financials'
_BS='Test Results'
_BR='Attendance'
_BQ='Courses'
_BP='Staff'
_BO='Invalid phone format'
_BN='%m/%d/%Y'
_BM='temperature'
_BL='BACKGROUND'
_BK='VALIGN'
_BJ='class_members'
_BI='classes'
_BH='lunch_payments'
_BG='deepseek/deepseek-r1-0528:free'
_BF='moonshotai/kimi-dev-72b:free'
_BE='Replace Entire Database'
_BD='Rename Field'
_BC='%m/%d'
_BB='language'
_BA='secondary'
_B9='student_number'
_B8='student_name'
_B7='courses'
_B6='term_id'
_B5='Term'
_B4='Present'
_B3='Phone'
_B2='Select AI Model'
_B1='Students'
_B0='date'
_A_='position'
_Az='due_date'
_Ay='gmail_connector'
_Ax='success'
_Aw='gmail'
_Av='gmail_sender_name_input'
_Au='sender_name'
_At='username'
_As='current_reminders'
_Ar='Dashboard'
_Aq='Helvetica-Bold'
_Ap='Name'
_Ao='openrouter/auto'
_An='Delete Field'
_Am='code'
_Al='delta'
_Ak=b'data:'
_Aj='Cloud (OpenRouter)'
_Ai='Local (Ollama)'
_Ah='Amount'
_Ag='test_date'
_Af='Graduate'
_Ae='remove'
_Ad='teacher_name'
_Ac='class_id'
_Ab='Select rows to delete'
_Aa='Enrollment Date'
_AZ='Room'
_AY='ai_provider_select'
_AX='has_phone'
_AW='nan'
_AV='display'
_AU='bulk'
_AT='individual'
_AS='Subject'
_AR='Grade'
_AQ='terms'
_AP='timestamp'
_AO='type'
_AN='text/csv'
_AM='medium'
_AL='term_name'
_AK='Upload CSV/TXT file'
_AJ='📤 Import from CSV/TXT'
_AI='Teacher'
_AH='Local'
_AG='granite3.3:2b'
_AF='failed'
_AE='School Admin'
_AD='%Y-%m-%d'
_AC='Status'
_AB='enrollment_date'
_AA='dob'
_A9='txt'
_A8='csv'
_A7='Normal'
_A6='Parent/Guardian'
_A5='No phone provided'
_A4='No email provided'
_A3='has_email'
_A2='password'
_A1='last_reset'
_A0='stream'
_z='model'
_y='N/A'
_x='expenditures'
_w='fees'
_v='test_results'
_u='attendance'
_t='staff'
_s='deepseek/deepseek-chat-v3-0324:free'
_r='Delete Selected'
_q='Save Changes'
_p='Delete?'
_o='Student'
_n='paid'
_m='reminder'
_l='messages'
_k='application/json'
_j='Content-Type'
_i='score'
_h='SELECT id, name FROM students'
_g='dynamic'
_f='payment_status'
_e='model_cache_misses'
_d='subject'
_c='---'
_b='system'
_a='students'
_Z='db_cache_misses'
_Y='assistant'
_X='primary'
_W='model_cache_hits'
_V='db_cache_hits'
_U='gmail_connected'
_T='grade'
_S='choices'
_R='phone'
_Q='message'
_P='status'
_O='parent_name'
_N='student_id'
_M='utf-8'
_L='user'
_K='email'
_J='class'
_I=', '
_H='delete'
_G='id'
_F='role'
_E='name'
_D=None
_C='content'
_B=False
_A=True
import pickle,streamlit as st,pandas as pd,sqlite3,requests,json,base64
from datetime import datetime,timedelta,date
import hashlib,os,io
from io import BytesIO,StringIO
import plotly.express as px
from functools import lru_cache
import time,re,tempfile,smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import random
from typing import Dict,Optional,List,Tuple
from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,PageBreak,Spacer,Image
from chatX import get_ollama_models,is_vision_model,is_qwen3_model,split_message,OLLAMA_HOST,get_ollama_models_cached,get_ollama_host
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Image as PlatypusImage
import openai
APP_VERSION='1.8.0'
st.set_page_config(page_title='RadioSport School',page_icon='🧟',layout='wide',menu_items={'Report a Bug':'https://github.com/rkarikari/stem','About':'Copyright © RNK, 2025 RadioSport. All rights reserved.'})
DB_NAME='school_admin.db'
CACHE_TTL=300
MAX_RECORDS=10000
EMAIL_PATTERN=re.compile('^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}(?:\\.[a-zA-Z]{2,})?$')
st.markdown("\n<style>\nbody {\n    font-family: 'Arial', sans-serif;\n}\n.main-title {\n    font-size: 28px !important;\n    font-weight: 700;\n    color: #1a3c5e;\n}\n.sidebar-title {\n    font-size: 22px !important;\n    font-weight: bold;\n    color: #1a3c5e;\n    margin-bottom: 10px !important;\n}\n.section-header {\n    font-size: 20px;\n    font-weight: 600;\n    color: #2c5282;\n    margin-top: 20px;\n    margin-bottom: 10px;\n}\n.stButton>button {\n    background-color: #2b6cb0;\n    color: white;\n    border-radius: 8px;\n    padding: 8px 16px;\n}\n.stButton>button:hover {\n    background-color: #2c5282;\n}\n.dataframe {\n    border-radius: 8px;\n    overflow: hidden;\n}\n.card {\n    background: white;\n    padding: 15px;\n    border-radius: 10px;\n    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n    margin-bottom: 15px;\n}\n.cache-stats {\n    font-size: 12px;\n    color: #666;\n    padding: 10px;\n    background-color: #f8f9fa;\n    border-radius: 6px;\n}\n.student-row {\n    background: white;\n    padding: 15px;\n    border-radius: 10px;\n    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n    margin-bottom: 15px;\n}\n.ai-status {\n    padding: 8px;\n    border-radius: 6px;\n    margin: 8px 0;\n}\n.ai-connected {\n    background-color: #d4edda;\n    color: #155724;\n    border: 1px solid #c3e6cb;\n}\n.ai-disconnected {\n    background-color: #f8d7da;\n    color: #721c24;\n    border: 1px solid #f5c6cb;\n}\n.email-status {\n    padding: 10px;\n    border-radius: 5px;\n    margin: 10px 0;\n}\n.email-success {\n    background-color: #d4edda;\n    color: #155724;\n    border: 1px solid #c3e6cb;\n}\n.email-error {\n    background-color: #f8d7da;\n    color: #721c24;\n    border: 1px solid #f5c6cb;\n}\n.db-connection {\n    padding: 10px;\n    border-radius: 5px;\n    margin: 10px 0;\n    background-color: #e6f7ff;\n    border: 1px solid #91d5ff;\n}\n.column-mapping {\n    background-color: #f9f9f9;\n    padding: 15px;\n    border-radius: 8px;\n    margin-bottom: 15px;\n}\n.db-field-table {\n    background-color: #f8f9fa;\n    border-radius: 8px;\n    padding: 15px;\n    margin-bottom: 15px;\n}\n.db-field-row {\n    display: flex;\n    margin-bottom: 8px;\n    align-items: center;\n}\n.db-field-input {\n    flex: 1;\n    margin-right: 10px;\n}\n.action-button {\n    margin: 0 3px;\n    padding: 4px 8px;\n}\n.student-report {\n    background-color: #f0f9ff;\n    padding: 15px;\n    border-radius: 10px;\n    margin-bottom: 15px;\n}\n.financial-card {\n    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);\n    border-radius: 12px;\n    padding: 20px;\n    margin-bottom: 20px;\n    box-shadow: 0 4px 6px rgba(0,0,0,0.1);\n}\n.financial-metric {\n    font-size: 24px;\n    font-weight: 700;\n    margin-top: 10px;\n}\n.financial-label {\n    font-size: 14px;\n    color: #4a5568;\n}\n.positive-value {\n    color: #38a169;\n}\n.negative-value {\n    color: #e53e3e;\n}\n/* Add these styles from chatX.py */\n.thinking-display {\n    position: fixed;\n    top: 10px;\n    right: 10px;\n    z-index: 1000;\n    background: rgba(248,249,250,0.98);\n    backdrop-filter: blur(12px);\n    border: 1px solid #e9ecef;\n    border-radius: 12px;\n    padding: 0;\n    max-width: 400px;\n    max-height: 300px;\n    overflow: hidden;\n    font-size: 13px;\n    color: #495057;\n    box-shadow: 0 8px 25px rgba(0,0,0,0.15);\n    transition: all 0.3s ease;\n    opacity: 0;\n    transform: translateY(-10px);\n}\n.thinking-display.visible {\n    opacity: 1;\n    transform: translateY(0);\n}\n.thinking-header {\n    background: linear-gradient(135deg, #007bff, #0056b3);\n    color: white;\n    padding: 8px 12px;\n    font-weight: 600;\n    font-size: 12px;\n    border-radius: 11px 11px 0 0;\n}\n.thinking-content {\n    padding: 12px;\n    max-height: 250px;\n    overflow-y: auto;\n    line-height: 1.4;\n    word-wrap: break-word;\n}\n.thinking-content::-webkit-scrollbar {\n    width: 4px;\n}\n.thinking-content::-webkit-scrollbar-track {\n    background: #f1f1f1;\n    border-radius: 2px;\n}\n.thinking-content::-webkit-scrollbar-thumb {\n    background: #007bff;\n    border-radius: 2px;\n}\n</style>\n",unsafe_allow_html=_A)
OPENROUTER_API_URL='https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODELS=[_BF,_Ao,_s,_BG,'mistralai/mistral-7b-instruct','google/gemma-7b-it','openai/gpt-3.5-turbo','openai/gpt-4-turbo','meta-llama/llama-3-70b-instruct']
def init_db():conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('\n        CREATE TABLE IF NOT EXISTS students (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            name TEXT NOT NULL,\n            grade TEXT,                -- CHANGED: From INTEGER to TEXT to support streams like "1R", "1C"\n            room TEXT,\n            dob DATE,\n            phone TEXT,\n            address TEXT,\n            enrollment_date DATE,\n            email TEXT,\n            parent_name TEXT,\n            medical_notes TEXT,\n            scholarship_status TEXT,\n            status TEXT DEFAULT \'Active\',  -- NEW: Added status column for graduation tracking\n            FOREIGN KEY (room) REFERENCES classes(room) \n        )\n    ');c.execute('\n        CREATE TABLE IF NOT EXISTS courses (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            name TEXT NOT NULL,\n            description TEXT,\n            teacher_id INTEGER,\n            term_id INTEGER,\n            schedule TEXT,\n            min_promotion_score REAL DEFAULT 60.0,  -- UPDATED: Added default value\n            FOREIGN KEY (teacher_id) REFERENCES staff(id),\n            FOREIGN KEY (term_id) REFERENCES terms(id)\n        )\n    ');c.execute("\n        CREATE TABLE IF NOT EXISTS enrollments (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER,\n            course_id INTEGER,\n            enrollment_date DATE,\n            status TEXT DEFAULT 'Active',  -- UPDATED: Added default status\n            FOREIGN KEY (student_id) REFERENCES students(id),\n            FOREIGN KEY (course_id) REFERENCES courses(id)\n        )\n    ");c.execute('\n        CREATE TABLE IF NOT EXISTS behavior (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER,\n            date DATE,\n            incident_type TEXT,\n            description TEXT,\n            action_taken TEXT,\n            resolved BOOLEAN DEFAULT 0,  -- UPDATED: Added default value\n            FOREIGN KEY (student_id) REFERENCES students(id)\n        )\n    ');c.execute("\n        CREATE TABLE IF NOT EXISTS library (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            book_title TEXT NOT NULL,\n            author TEXT,\n            isbn TEXT,\n            category TEXT,\n            status TEXT DEFAULT 'Available',  -- UPDATED: Added default status\n            borrower_id INTEGER,\n            checkout_date DATE,\n            due_date DATE,\n            FOREIGN KEY (borrower_id) REFERENCES students(id)\n        )\n    ");c.execute('\n        CREATE TABLE IF NOT EXISTS staff (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            staff_id INTEGER,\n            name TEXT NOT NULL,\n            role TEXT,\n            department TEXT,\n            phone TEXT,\n            hire_date DATE,\n            salary REAL\n        )\n    ');c.execute('\n        CREATE TABLE IF NOT EXISTS classes (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            name TEXT NOT NULL,\n            room TEXT UNIQUE,\n            teacher_id INTEGER,\n            term_id INTEGER,\n            FOREIGN KEY (teacher_id) REFERENCES staff(id),\n            FOREIGN KEY (term_id) REFERENCES terms(id)\n        )\n    ');c.execute('\n        CREATE TABLE IF NOT EXISTS class_members (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            class_id INTEGER NOT NULL,\n            student_id INTEGER NOT NULL,\n            term_id INTEGER NOT NULL,\n            UNIQUE(student_id, term_id),\n            FOREIGN KEY (class_id) REFERENCES classes(id),\n            FOREIGN KEY (student_id) REFERENCES students(id),\n            FOREIGN KEY (term_id) REFERENCES terms(id)\n        )\n    ');c.execute("\n        CREATE TABLE IF NOT EXISTS attendance (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER,\n            date DATE,\n            status TEXT DEFAULT 'Present',  -- UPDATED: Added default status\n            FOREIGN KEY (student_id) REFERENCES students(id)\n        )\n    ");c.execute('\n        CREATE TABLE IF NOT EXISTS test_results (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER,\n            subject TEXT,\n            score REAL,\n            test_date DATE,\n            test_type TEXT,  -- NEW: Added test type field\n            FOREIGN KEY (student_id) REFERENCES students(id)\n        )\n    ');c.execute("\n        CREATE TABLE IF NOT EXISTS fees (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER,\n            amount REAL,\n            due_date DATE,\n            status TEXT DEFAULT 'Pending',  -- UPDATED: Added default status\n            payment_date DATE,\n            FOREIGN KEY (student_id) REFERENCES students(id)\n        )\n    ");c.execute('\n        CREATE TABLE IF NOT EXISTS expenditures (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            category TEXT NOT NULL,\n            description TEXT,\n            amount REAL NOT NULL,\n            date DATE NOT NULL,\n            vendor TEXT,\n            notes TEXT\n        )\n    ');c.execute("\n        CREATE TABLE IF NOT EXISTS terms (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            name TEXT NOT NULL,\n            start_date DATE NOT NULL,\n            end_date DATE NOT NULL,\n            status TEXT DEFAULT 'Active'  -- NEW: Added status for terms\n        )\n    ");c.execute('\n        CREATE TABLE IF NOT EXISTS student_reports (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER NOT NULL,\n            term_id INTEGER NOT NULL,\n            report_content TEXT NOT NULL,\n            generated_date DATE NOT NULL,\n            FOREIGN KEY (student_id) REFERENCES students(id),\n            FOREIGN KEY (term_id) REFERENCES terms(id)\n        )\n    ');c.execute("\n        CREATE TABLE IF NOT EXISTS lunch_payments (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER NOT NULL,\n            amount REAL NOT NULL,\n            payment_date DATE NOT NULL,\n            status TEXT CHECK( status IN ('paid','unpaid') ) NOT NULL DEFAULT 'unpaid',\n            term_id INTEGER,\n            FOREIGN KEY (student_id) REFERENCES students(id),\n            FOREIGN KEY (term_id) REFERENCES terms(id)\n        )\n    ");c.execute("\n        CREATE TABLE IF NOT EXISTS promotion_history (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            student_id INTEGER NOT NULL,\n            from_term_id INTEGER NOT NULL,\n            to_term_id INTEGER NOT NULL,\n            from_grade INTEGER NOT NULL,\n            to_grade INTEGER NOT NULL,\n            action TEXT NOT NULL, -- 'Promote', 'Repeat', 'Transfer', 'Graduate'\n            promotion_date DATE NOT NULL,\n            notes TEXT,\n            FOREIGN KEY (student_id) REFERENCES students (id),\n            FOREIGN KEY (from_term_id) REFERENCES terms (id),\n            FOREIGN KEY (to_term_id) REFERENCES terms (id)\n        )\n    ");c.execute('\n        CREATE TABLE IF NOT EXISTS promotion_rules (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            from_grade INTEGER NOT NULL,\n            to_grade INTEGER NOT NULL,\n            minimum_score REAL,\n            required_subjects TEXT, -- JSON array of subject requirements\n            auto_promote BOOLEAN DEFAULT FALSE,\n            created_date DATE DEFAULT CURRENT_DATE\n        )\n    ');conn.commit();conn.close()
def get_table_schema(table_name):conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute(f"PRAGMA table_info({table_name})");schema=c.fetchall();conn.close();return schema
def add_column_to_table(table_name,column_name,column_type):
	conn=sqlite3.connect(DB_NAME);c=conn.cursor()
	try:c.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}");conn.commit();return _A,f"Column '{column_name}' added to '{table_name}'"
	except sqlite3.OperationalError as e:return _B,str(e)
	finally:conn.close()
def rename_column(table_name,old_name,new_name):
	conn=sqlite3.connect(DB_NAME);c=conn.cursor()
	try:
		c.execute(f"PRAGMA table_info({table_name})");columns=c.fetchall();temp_table=f"{table_name}_temp";create_sql=f"CREATE TABLE {temp_table} ("
		for col in columns:name=new_name if col[1]==old_name else col[1];create_sql+=f"{name} {col[2]}, "
		create_sql=create_sql.rstrip(_I)+')';c.execute(create_sql);c.execute(f"INSERT INTO {temp_table} SELECT * FROM {table_name}");c.execute(f"DROP TABLE {table_name}");c.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}");conn.commit();return _A,f"Column '{old_name}' renamed to '{new_name}' in '{table_name}'"
	except sqlite3.Error as e:return _B,str(e)
	finally:conn.close()
def delete_column(table_name,column_name):
	conn=sqlite3.connect(DB_NAME);c=conn.cursor()
	try:
		c.execute(f"PRAGMA table_info({table_name})");columns=c.fetchall();remaining_cols=[col for col in columns if col[1]!=column_name];temp_table=f"{table_name}_temp";create_sql=f"CREATE TABLE {temp_table} ("
		for col in remaining_cols:create_sql+=f"{col[1]} {col[2]}, "
		create_sql=create_sql.rstrip(_I)+')';c.execute(create_sql);cols_str=_I.join([col[1]for col in remaining_cols]);c.execute(f"INSERT INTO {temp_table} ({cols_str}) SELECT {cols_str} FROM {table_name}");c.execute(f"DROP TABLE {table_name}");c.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}");conn.commit();return _A,f"Column '{column_name}' deleted from '{table_name}'"
	except sqlite3.Error as e:return _B,str(e)
	finally:conn.close()
def show_reasoning_window():
	'Initialize and show the reasoning window'
	if not st.session_state.reasoning_window:st.session_state.reasoning_window=st.empty()
	st.session_state.reasoning_window.markdown('\n        <div id="thinking-display" class="thinking-display visible">\n            <div class="thinking-header">🤔 Reasoning Process</div>\n            <div class="thinking-content">Thinking...</div>\n        </div>\n        ',unsafe_allow_html=_A)
def update_reasoning_window(content):
	'Update the reasoning window with new content'
	if st.session_state.reasoning_window:
		if len(content)>1500:content='...'+content[-1500:]
		st.session_state.reasoning_window.markdown(f'''
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
            ''',unsafe_allow_html=_A)
def hide_reasoning_window():
	'Hide the reasoning window'
	if st.session_state.reasoning_window:st.session_state.reasoning_window.empty();st.session_state.reasoning_window=_D
@lru_cache(maxsize=128)
def query_db(query,params=()):
	'Execute a SQL query and return results and column names.'
	try:conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute(query,params);results=c.fetchall();columns=[desc[0]for desc in c.description]if c.description else[];conn.close();st.session_state.cache_stats[_V]+=1;return results,columns
	except sqlite3.Error as e:st.session_state.cache_stats[_Z]+=1;st.error(f"Database query error: {str(e)}");return[],[]
	except Exception as e:st.session_state.cache_stats[_Z]+=1;st.error(f"Unexpected error: {str(e)}");return[],[]
def get_db_context():
	context=[];tables=[_a,_t,_u,_v,_w,_x,_AQ,_BH,_BI,_BJ]
	for table in tables:
		try:
			schema=get_table_schema(table);schema_str=_I.join([f"{col[1]} ({col[2]})"for col in schema]);count_query=f"SELECT COUNT(*) FROM {table}";count_result,_=query_db(count_query);total_count=count_result[0][0]if count_result else 0;summary=''
			if table==_a:results,cols=query_db('SELECT grade, COUNT(*) as count FROM students GROUP BY grade');summary=f"Grade distribution: {_I.join([f'{row[0]}:{row[1]}'for row in results])}"
			elif table==_t:results,cols=query_db('SELECT role, COUNT(*) as count FROM staff GROUP BY role');summary=f"Roles: {_I.join([f'{row[0]}:{row[1]}'for row in results])}"
			elif table==_w:results,cols=query_db(_C9);summary=f"Fee status: {_I.join([f'{row[0]}:₡{row[1]:.2f}'for row in results])}"
			elif table==_v:results,cols=query_db('SELECT subject, AVG(score) as avg_score FROM test_results GROUP BY subject');summary=f"Subject averages: {_I.join([f'{row[0]}:{row[1]:.1f}'for row in results])}"
			elif table==_u:results,cols=query_db('SELECT status, COUNT(*) as count FROM attendance GROUP BY status');summary=f"Attendance status: {_I.join([f'{row[0]}:{row[1]}'for row in results])}"
			elif table==_x:results,cols=query_db('SELECT category, SUM(amount) as total FROM expenditures GROUP BY category');summary=f"Expenditures: {_I.join([f'{row[0]}:₡{row[1]:.2f}'for row in results])}"
			elif table==_AQ:results,cols=query_db('SELECT name, start_date, end_date FROM terms ORDER BY start_date DESC');summary=f"Terms: {_I.join([row[0]for row in results])}"
			sample_size=25 if total_count<100 else 0;sample_query=f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_size}";sample_results,sample_cols=query_db(sample_query);sample_str=''
			if sample_results:sample_df=pd.DataFrame(sample_results,columns=sample_cols);sample_str=f"\nSample data:\n{sample_df.to_markdown(index=_B)}"
			context.append(f"### {table.upper()} TABLE\n- Schema: {schema_str}\n- Total records: {total_count}\n- Summary: {summary}{sample_str}\n")
		except Exception as e:context.append(f"⚠️ Error processing {table}: {str(e)}")
	return'\n'.join(context)
def get_principal_name():
	'Get the name of the principal from staff table';A='School Principal'
	try:query="\n            SELECT name \n            FROM staff \n            WHERE role IN ('Principal', 'Headmaster', 'Headmistress')\n            LIMIT 1\n        ";results,_=query_db(query);return results[0][0]if results else A
	except:return A
def get_teacher_for_student(student_grade):
	try:query='\n            SELECT s.name \n            FROM staff s\n            JOIN classes c ON c.teacher_id = s.id\n            WHERE c.room = ?\n            LIMIT 1\n        ';results,_=query_db(query,(student_grade,));return results[0][0]if results else _CA
	except:return'Error fetching'
def execute_sql(query):
	'Execute SQL query and return results, columns, or error message'
	try:
		cleaned_query=query.strip().rstrip(';')
		if not re.match('^\\s*SELECT',cleaned_query,re.IGNORECASE):return _D,_D,'Only SELECT queries are allowed'
		forbidden=['INSERT','UPDATE','DELETE','DROP','ALTER','CREATE']
		if any(f.upper()in cleaned_query.upper()for f in forbidden):return _D,_D,'Query contains forbidden operation'
		conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute(cleaned_query);results=c.fetchall();columns=[desc[0]for desc in c.description]if c.description else[];return results,columns,_D
	except sqlite3.Error as e:return _D,_D,f"SQL Error: {str(e)}"
	except Exception as e:return _D,_D,f"Error: {str(e)}"
	finally:conn.close()
def generate_report_pdf(report_title,student_name,term,student_info,performance_table,report_content,graph_path=_D):
	D='CENTER';C='FONTSIZE';B='ALIGN';A='Heading2';buffer=io.BytesIO();doc=SimpleDocTemplate(buffer,pagesize=A4);styles=getSampleStyleSheet();content_style=styles['BodyText'];title_style=ParagraphStyle('Title',parent=styles['Heading1'],fontSize=24,alignment=1,spaceAfter=20);subtitle_style=ParagraphStyle('Subtitle',parent=styles[A],fontSize=18,alignment=1,spaceAfter=40);heading2_style=ParagraphStyle(A,parent=styles[A],fontSize=16,spaceBefore=10,spaceAfter=10);elements=[];elements.append(Paragraph(report_title,title_style));elements.append(Paragraph(f"Student: {student_name}",subtitle_style));elements.append(Paragraph(f"Term: {term}",subtitle_style));elements.append(Paragraph(f"Generated: {date.today().strftime(_AD)}",subtitle_style));elements.append(PageBreak());elements.append(Paragraph('Student Information',heading2_style));teacher_name=get_teacher_for_student(student_info[_T]);student_info_data=[[_Ap,student_info[_E]],[_AR,student_info[_T]],[_CB,teacher_name],['Overall Position',f"{student_info.get(_CC,_y)}"]];student_info_table=Table(student_info_data,colWidths=[80,300]);student_info_table.setStyle(TableStyle([('FONT',(0,0),(-1,-1),'Helvetica'),(C,(0,0),(-1,-1),10),(_BK,(0,0),(-1,-1),'TOP'),(B,(0,0),(0,-1),'RIGHT'),(B,(1,0),(1,-1),'LEFT'),('BOX',(0,0),(-1,-1),1,colors.black),('GRID',(0,0),(-1,-1),1,colors.grey)]));elements.append(student_info_table);elements.append(Spacer(1,10))
	if performance_table and len(performance_table)>0:
		try:elements.append(Paragraph('Academic Performance',heading2_style));performance_data=[[_AS,'Score',_CD,'Position in Class','Remarks']]+performance_table;perf_table=Table(performance_data,repeatRows=1);perf_table.setStyle(TableStyle([(_BL,(0,0),(-1,0),colors.grey),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),(B,(0,0),(-1,-1),D),('FONTNAME',(0,0),(-1,0),_Aq),(C,(0,0),(-1,0),12),(_CE,(0,0),(-1,0),12),(_BL,(0,1),(-1,-1),colors.beige),('GRID',(0,0),(-1,-1),1,colors.black)]));elements.append(perf_table);elements.append(Spacer(1,10))
		except Exception as e:print(f"Error creating performance table: {str(e)}")
	if graph_path and os.path.exists(graph_path):
		try:elements.append(Spacer(1,12));elements.append(Paragraph('Performance Over Time',heading2_style));img=PlatypusImage(graph_path,width=500,height=280);img.hAlign=D;elements.append(img);elements.append(Spacer(1,12))
		except Exception as e:print(f"Error including performance graph: {str(e)}")
	elements.append(Paragraph('Report Summary',heading2_style))
	for line in report_content.split('\n'):
		if line.strip():elements.append(Paragraph(line,content_style));elements.append(Spacer(1,6))
	doc.build(elements);buffer.seek(0);return buffer
def call_openrouter_api(messages,model,api_key,max_retries=3):
	'Calls OpenRouter API for chat completions with rate limiting and retry logic';headers={_CF:f"Bearer {api_key}",_j:_k,_CG:_CH,'X-Title':_CI};payload={_z:model,_l:messages,_A0:_A,'max_tokens':2048,_BM:.7}
	for attempt in range(max_retries):
		try:
			if attempt>0:delay=2**attempt+random.uniform(0,1);st.info(f"Rate limited. Retrying in {delay:.1f} seconds... (Attempt {attempt+1}/{max_retries})");time.sleep(delay)
			response=requests.post(OPENROUTER_API_URL,headers=headers,json=payload,stream=_A,timeout=60)
			if response.status_code==429:
				if attempt<max_retries-1:retry_after=int(response.headers.get('Retry-After',60));st.warning(f"Rate limit hit. Waiting {retry_after} seconds before retry...");time.sleep(retry_after);continue
				else:st.error('Rate limit exceeded. Please wait a few minutes before trying again.');return
			response.raise_for_status();return response
		except requests.exceptions.HTTPError as e:
			if response.status_code==429:continue
			elif response.status_code==403:st.error('API key invalid or insufficient credits.');return
			elif response.status_code>=500:
				if attempt<max_retries-1:st.warning(f"Server error {response.status_code}. Retrying...");continue
				else:st.error(f"Server error {response.status_code}. Please try again later.");return
			else:st.error(f"HTTP error {response.status_code}: {str(e)}");return
		except requests.exceptions.Timeout:
			if attempt<max_retries-1:st.warning('Request timed out. Retrying...');continue
			else:st.error('Request timed out multiple times. Please check your connection.');return
		except requests.exceptions.ConnectionError:
			if attempt<max_retries-1:st.warning('Connection error. Retrying...');time.sleep(2);continue
			else:st.error('Connection failed. Please check your internet connection.');return
		except Exception as e:st.error(f"Unexpected error: {str(e)}");return
class RateLimiter:
	def __init__(self,calls_per_minute=10):self.calls_per_minute=calls_per_minute;self.call_times=[]
	def wait_if_needed(self):
		now=time.time();self.call_times=[t for t in self.call_times if now-t<60]
		if len(self.call_times)>=self.calls_per_minute:
			oldest_call=min(self.call_times);wait_time=60-(now-oldest_call)
			if wait_time>0:st.info(f"Rate limiting: waiting {wait_time:.1f} seconds...");time.sleep(wait_time)
		self.call_times.append(now)
if'cache_stats'not in st.session_state:st.session_state.cache_stats={_V:0,_Z:0,_W:0,_e:0,'document_cache_hits':0,'document_cache_misses':0,_A1:datetime.now()}
else:
	for key in[_V,_Z,_W,_e]:
		if key not in st.session_state.cache_stats:st.session_state.cache_stats[key]=0
if'reasoning_window'not in st.session_state:st.session_state.reasoning_window=_D
if _l not in st.session_state:st.session_state.messages=[]
if'current_section'not in st.session_state:st.session_state.current_section=_Ar
if'ticket_messages'not in st.session_state:st.session_state.ticket_messages=[]
if'ticket_file_uploader_key'not in st.session_state:st.session_state.ticket_file_uploader_key='uploader_0'
if _U not in st.session_state:st.session_state.gmail_connected=_B
if'last_connection_message'not in st.session_state:st.session_state.last_connection_message=_D
if'email_status'not in st.session_state:st.session_state.email_status={_AT:{},_AU:_D}
if _As not in st.session_state:st.session_state.current_reminders=[]
if'chat_cleared'not in st.session_state:st.session_state.chat_cleared=_B
if'db_connection'not in st.session_state:st.session_state.db_connection=_D
if'db_file_path'not in st.session_state:st.session_state.db_file_path=_D
init_db()
class GmailConnector:
	def __init__(self):self.smtp_server='smtp.gmail.com';self.smtp_port=587;self.username='';self.password='';self.connected=_B;self.connection=_D;self.settings_file='gmail_settings.pkl'
	def save_settings(self):
		'Save settings to a file';settings={_At:self.username,_A2:self.password,_Au:st.session_state.get(_Av,'')}
		with open(self.settings_file,'wb')as f:pickle.dump(settings,f)
		return _A
	def load_settings(self):
		'Load settings from file or secrets'
		try:
			with open(self.settings_file,'rb')as f:settings=pickle.load(f)
			self.username=settings.get(_At,'');self.password=settings.get(_A2,'')
			try:import streamlit as st;st.session_state.gmail_sender_name_input=settings.get(_Au,_AE)
			except:pass
			return _A
		except FileNotFoundError:
			try:
				import streamlit as st;gmail_config=st.secrets.get(_Aw,{});username=gmail_config.get(_At,'');password=gmail_config.get(_A2,'');sender_name=gmail_config.get(_Au,_AE)
				if username and password:self.username=username;self.password=password;st.session_state.gmail_sender_name_input=sender_name;return _A
			except:pass
		except Exception:pass
	def connect(self,username,password):
		'Connect to Gmail SMTP server'
		try:self.username=username;self.password=password;server=smtplib.SMTP(self.smtp_server,self.smtp_port,timeout=10);server.starttls();server.login(username,password);server.quit();self.connected=_A;st.session_state[_U]=_A;return _A,'✅ Successfully connected to Gmail'
		except smtplib.SMTPAuthenticationError:self.connected=_B;st.session_state[_U]=_B;return _B,'❌ Authentication failed. Please check your email and app password.'
		except smtplib.SMTPConnectError:self.connected=_B;st.session_state[_U]=_B;return _B,'❌ Could not connect to Gmail servers. Check internet connection.'
		except Exception as e:self.connected=_B;st.session_state[_U]=_B;return _B,f"❌ Connection error: {str(e)}"
	def send_email(self,to_email,subject,body,sender_name=_D,attachment_data=_D,attachment_filename=_D):
		'Send email via Gmail'
		if not self.connected:
			success,message=self.connect(self.username,self.password)
			if not success:return _B,'Not connected to Gmail'
		if not EMAIL_PATTERN.match(to_email):return _B,f"Invalid email format: {to_email}"
		try:
			msg=MIMEMultipart();msg['From']=f"{sender_name} <{self.username}>"if sender_name else self.username;msg['To']=to_email;msg[_AS]=sanitize_text(subject);sanitized_body=sanitize_text(body);msg.attach(MIMEText(sanitized_body,'plain',_M))
			if attachment_data and attachment_filename:attachment=MIMEBase('application','octet-stream');attachment.set_payload(attachment_data);encoders.encode_base64(attachment);attachment.add_header('Content-Disposition',f"attachment; filename={sanitize_text(attachment_filename)}");msg.attach(attachment)
			server=smtplib.SMTP(self.smtp_server,self.smtp_port,timeout=10);server.starttls();server.login(self.username,self.password);text=msg.as_string();server.sendmail(self.username,to_email,text);server.quit();return _A,'✅ Email sent successfully'
		except smtplib.SMTPRecipientsRefused as e:error_msg=f"❌ Recipient refused: {str(e)}";return _B,error_msg
		except smtplib.SMTPException as e:error_msg=f"❌ SMTP error: {str(e)}";return _B,error_msg
		except Exception as e:error_msg=f"❌ Failed to send email: {str(e)}";return _B,error_msg
	def send_bulk_reminders(self,reminders,sender_name=_D):
		'Send bulk payment reminder emails by reusing send_email for each';A='errors';results={_Ax:0,_AF:0,A:[]}
		if not self.connected:
			success,message=self.test_connection()
			if not success:results[A].append(message);return results
		for reminder in reminders:
			if not reminder.get(_A3):results[_AF]+=1;results[A].append(f"{reminder[_E]}: No valid email address");continue
			subject=f"Lunch Payment Reminder - {reminder[_E]}";body=reminder[_m];success,message=self.send_email(reminder[_K],subject,body,sender_name)
			if success:results[_Ax]+=1
			else:results[_AF]+=1;error_msg=f"Failed to send to {reminder[_K]}: {message}";results[A].append(f"{reminder[_E]}: {error_msg}")
		return results
	def test_connection(self):
		'Test current connection status'
		if not self.username or not self.password:return _B,'No credentials configured'
		return self.connect(self.username,self.password)
class TicketGenerator:
	def __init__(self):
		self.students_df=pd.DataFrame();self.payments_df=pd.DataFrame()
		if _Ay not in st.session_state:st.session_state[_Ay]=GmailConnector()
		self.gmail=st.session_state[_Ay];self.ollama_model=_AG
	def fetch_db_data(self):
		'Fetch student and payment data from school database'
		try:
			results,columns=query_db('SELECT * FROM students')
			if not results:return pd.DataFrame(),pd.DataFrame()
			students_df=pd.DataFrame(results,columns=columns)
			if _K not in students_df.columns:students_df[_K]=_A4
			if _R not in students_df.columns:students_df[_R]=_A5
			if _O not in students_df.columns:students_df[_O]=_A6
			results,columns=query_db('SELECT * FROM fees')
			if not results:return students_df,pd.DataFrame()
			payments_df=pd.DataFrame(results,columns=columns);payments_df[_P]=payments_df[_P].str.lower()if payments_df[_P].dtype=='object'else payments_df[_P].astype(str).str.lower();return students_df,payments_df
		except Exception as e:st.error(f"Error fetching database data: {str(e)}");return pd.DataFrame(),pd.DataFrame()
	def process_combined_csv(self,df):
		'Process combined CSV with student, payment, and contact data'
		try:
			required_cols=[_N,_E,_J,_f];optional_cols=[_R,_K,_O]
			if not all(col in df.columns for col in required_cols):missing=[col for col in required_cols if col not in df.columns];raise ValueError(f"Missing required columns: {missing}")
			student_cols=[_N,_E,_J]
			for col in optional_cols:
				if col in df.columns:student_cols.append(col)
			students_df=df[student_cols].copy();students_df[_K]=students_df.get(_K,pd.Series(_A4,index=students_df.index));students_df[_O]=students_df.get(_O,pd.Series(_A6,index=students_df.index));payments_df=df[[_N,_f]].copy();return students_df,payments_df
		except Exception as e:st.error(f"Error processing CSV: {str(e)}");return pd.DataFrame(),pd.DataFrame()
	def filter_paid_students(self,students_df,payments_df):
		'Filter students who have paid'
		try:
			if students_df.empty or payments_df.empty:st.warning('No data available to filter paid students');return pd.DataFrame()
			if _N not in students_df.columns:raise ValueError("Missing 'student_id' column in students data")
			if _N not in payments_df.columns:raise ValueError("Missing 'student_id' column in payments data")
			paid_ids=payments_df[payments_df[_f]==_n][_N].unique();paid_students=students_df[students_df[_N].isin(paid_ids)]
			if paid_students.empty:status_values=payments_df[_f].unique();st.warning(f"No paid students found. Payment status values: {status_values}")
			return paid_students
		except Exception as e:st.error(f"Error filtering students: {str(e)}");return pd.DataFrame()
	def generate_pdf(self,students_data,school_info,ticket_date,validity_info):
		'Generate PDF tickets with 3x5 grid (15 tickets per page)';buffer=io.BytesIO();doc=SimpleDocTemplate(buffer,pagesize=A4,topMargin=15*mm,bottomMargin=15*mm,leftMargin=15*mm,rightMargin=15*mm);tickets_per_page=15;story=[]
		for page_start in range(0,len(students_data),tickets_per_page):
			page_students=students_data.iloc[page_start:page_start+tickets_per_page];table_data=[]
			for row in range(5):
				row_data=[]
				for col in range(3):
					idx=row*3+col
					if idx<len(page_students):student=page_students.iloc[idx];content=self._create_ticket_content(student,school_info,ticket_date,validity_info);row_data.append(content)
					else:row_data.append(Paragraph('',getSampleStyleSheet()[_A7]))
				table_data.append(row_data)
			page_width=A4[0]-30*mm;page_height=A4[1]-30*mm;ticket_width=page_width/3;ticket_height=(page_height-20*mm)/5;table=Table(table_data,colWidths=[ticket_width]*3,rowHeights=[ticket_height]*5);table.setStyle(TableStyle([('BOX',(0,0),(-1,-1),2,colors.black),('GRID',(0,0),(-1,-1),1,colors.black),('LINEBELOW',(0,0),(-1,3),2,colors.black,_D,(4,2)),('LINEAFTER',(0,0),(1,-1),2,colors.black,_D,(4,2)),(_BK,(0,0),(-1,-1),'TOP'),('PADDING',(0,0),(-1,-1),4),(_BL,(0,0),(-1,-1),colors.white)]));story.append(table);story.append(Paragraph('<br/>',getSampleStyleSheet()[_A7]));cutting_style=ParagraphStyle('CuttingInstructions',parent=getSampleStyleSheet()[_A7],fontSize=10,textColor=colors.black,alignment=1,fontName=_Aq);story.append(Paragraph('✂️ Cut along lines to separate tickets (15 per page)',cutting_style))
			if page_start+tickets_per_page<len(students_data):story.append(PageBreak())
		doc.build(story);buffer.seek(0);return buffer
	def _create_ticket_content(self,student,school_info,ticket_date,validity_info):'Create individual ticket content';styles=getSampleStyleSheet();header_style=ParagraphStyle('HeaderStyle',parent=styles[_A7],fontSize=10,leading=11,alignment=1,fontName=_Aq,spaceAfter=3);ticket_title_style=ParagraphStyle('TicketTitleStyle',parent=styles[_A7],fontSize=9,leading=10,alignment=1,fontName=_Aq,spaceAfter=4);content_style=ParagraphStyle('ContentStyle',parent=styles[_A7],fontSize=8,leading=9,spaceAfter=2,leftIndent=4);footer_style=ParagraphStyle('FooterStyle',parent=styles[_A7],fontSize=6,leading=7,alignment=1,textColor=colors.gray,spaceAfter=1);ticket_num=f"LT{random.randint(100000,999999)}";student_name=str(student.get(_E,_y)).strip();student_class=str(student.get(_J,_y)).strip();student_id=str(student.get(_N,_y)).strip();school_name=str(school_info.get(_E,'School')).strip();issue_date=ticket_date.strftime(_BN);validity_display=str(validity_info.get(_AV,'Valid Today')).strip();ticket_elements=[Paragraph(school_name,header_style),Paragraph('LUNCH TICKET',ticket_title_style),Paragraph(f"<b>Student:</b> {student_name}",content_style),Paragraph(f"<b>Class:</b> {student_class}",content_style),Paragraph(f"<b>ID:</b> {student_id}",content_style),Paragraph(f"<b>Issued:</b> {issue_date}",content_style),Paragraph(f"<b>Valid:</b> {validity_display}",content_style),Paragraph('------------------------',footer_style),Paragraph('This ticket is non-transferable',footer_style),Paragraph('Present to cafeteria staff',footer_style),Paragraph(f"#{ticket_num}",footer_style)];mini_table=Table([[elem]for elem in ticket_elements],colWidths=[_D]);mini_table.setStyle(TableStyle([(_BK,(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),2),(_CE,(0,0),(-1,-1),2)]));return mini_table
	def generate_reminders(self,unpaid_df,sender_info):
		"Generate payment reminders using chatX.py's AI"
		if unpaid_df.empty:return[]
		reminders=[];total_students=len(unpaid_df.head(10));progress=st.progress(0)
		for(idx,(_,student))in enumerate(unpaid_df.head(10).iterrows()):
			progress.progress((idx+1)/total_students);phone=str(student.get(_R,_A5)).strip();email=str(student.get(_K,_A4)).strip().lower();parent_name=str(student.get(_O,_A6)).strip();has_valid_phone=_B
			if phone!=_A5 and phone!=_AW and phone:
				clean_phone=re.sub('\\D','',phone)
				if clean_phone.isdigit()and len(clean_phone)>=7:has_valid_phone=_A
				else:phone=_BO
			has_valid_email=_B
			if email!=_A4 and email!=_AW and email:
				if EMAIL_PATTERN.match(email):has_valid_email=_A
				elif'@'in email and'.'in email.split('@')[-1]:has_valid_email=_A
			prompt=f"""
Write the body of a polite payment reminder email (without subject line) for:
Student: {student.get(_E,_o)}
Class: {student.get(_J,_y)}
Parent/Guardian: {parent_name}

Create a respectful message about their child's lunch payment.
Include that payment is due by {sender_info.get(_Az,"soon")}.
Keep it under 100 words, professional but friendly.
DO NOT INCLUDE A SUBJECT LINE.

End the message with:

Best regards,
{sender_info[_E]}
{sender_info[_A_]}
{sender_info[_B0]}

For questions, please contact the school office.
""";system_prompt=_CJ;reminder_text='AI disabled for testing.'
			if self.ollama_model:
				api_payload={_z:self.ollama_model,_l:[{_F:_b,_C:system_prompt},{_F:_L,_C:prompt}],_A0:_B,'options':{_BM:.5,_CK:300}}
				try:response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_j:_k},timeout=60);response.raise_for_status();reminder_text=response.json().get(_Q,{}).get(_C,'No response generated');reminder_text=sanitize_text(reminder_text)
				except requests.exceptions.Timeout:reminder_text='Error: AI server timed out after 30 seconds.'
				except Exception:reminder_text='Generation error: Please check AI connection'
			reminders.append({_E:student.get(_E),_J:student.get(_J),_O:parent_name,_R:phone,_K:email,_m:reminder_text,_AX:phone not in[_A5,_BO]and pd.notna(phone),_A3:has_valid_email})
		progress.progress(1.);return reminders
def sanitize_text(text):
	"Sanitize text to ensure it's safe for email sending.";A='This is a payment reminder from RadioSport SchoolSync.'
	if not text:return A
	try:text=text.replace('\x00','');return text.encode(_M,errors='ignore').decode(_M).strip()
	except Exception:return A
def import_data(table_name,file):
	A='skip'
	try:
		if file.name.endswith('.csv'):df=pd.read_csv(file,skipinitialspace=_A,on_bad_lines=A)
		else:df=pd.read_csv(file,sep='\t',skipinitialspace=_A,on_bad_lines=A)
		df.columns=[col.strip().lower()for col in df.columns];conn=sqlite3.connect(DB_NAME);cursor=conn.cursor();cursor.execute(f"PRAGMA table_info({table_name})");existing_columns=[col[1]for col in cursor.fetchall()]
		if not existing_columns:st.error(f"Table {table_name} not found in database");return _B
		valid_columns=[col for col in df.columns if col in existing_columns]
		if not valid_columns:st.error('No matching columns found between file and database table');return _B
		placeholders=_I.join(['?']*len(valid_columns));columns_str=_I.join(valid_columns)
		for(_,row)in df.iterrows():
			if _E in valid_columns and(pd.isna(row.get(_E))or row.get(_E)==''):continue
			values=[row[col]for col in valid_columns];cursor.execute(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})",values)
		conn.commit();conn.close();query_db.cache_clear();return _A
	except Exception as e:st.error(f"Error importing data: {str(e)}");return _B
def delete_record(table_name,record_id):
	try:conn=sqlite3.connect(DB_NAME);cursor=conn.cursor();cursor.execute(f"DELETE FROM {table_name} WHERE id = ?",(record_id,));conn.commit();conn.close();query_db.cache_clear();return _A
	except Exception as e:st.error(f"Error deleting record: {str(e)}");return _B
def update_record(table_name,record_id,updates):
	try:conn=sqlite3.connect(DB_NAME);cursor=conn.cursor();set_clause=_I.join([f"{key} = ?"for key in updates.keys()]);values=list(updates.values())+[record_id];cursor.execute(f"UPDATE {table_name} SET {set_clause} WHERE id = ?",values);conn.commit();conn.close();query_db.cache_clear();return _A
	except Exception as e:st.error(f"Error updating record: {str(e)}");return _B
def format_as_table(results,headers):
	'Format query results as markdown table';B=' |\n';A=' | '
	if not results:return'No data available'
	table='| '+A.join(headers)+B;table+='|'+'|'.join([_c]*len(headers))+'|\n'
	for row in results:table+='| '+A.join(str(x)for x in row)+B
	return table
sections=[_Ar,_B1,_BP,_BQ,_BR,_BS,'Fees',_BT,_BU,_BV,_BW,_BX,_BY,_BZ]
st.markdown('<div class="main-title">RadioSport SchoolSync 🏫</div>',unsafe_allow_html=_A)
st.markdown(f'<div style="font-size: 14px; color: #666;">Version {APP_VERSION}</div>',unsafe_allow_html=_A)
with st.sidebar:
	st.markdown('<div class="sidebar-title">RadioSport SchoolSync</div>',unsafe_allow_html=_A);sections=[_Ar,_B1,_BP,_BQ,_BR,_BS,'Fees',_BT,_BU,_BV,_BW,_BX,_BY,_BZ];selected_section=st.selectbox('Navigate',sections,index=sections.index(st.session_state.current_section),key='navigate_selectbox')
	if selected_section!=st.session_state.current_section:st.session_state.current_section=selected_section;st.rerun()
	ollama_available=bool(get_ollama_models());default_provider_index=0 if ollama_available else 1;ai_provider=st.radio(_CL,[_AH,_Ba],index=default_provider_index,key=_AY,help='Local: Uses Ollama (if available) | Cloud: Uses OpenRouter API');OLLAMA_MODEL=_D;OPENROUTER_MODEL='anthropic/claude-3-haiku'
	@st.cache_data(ttl=3600)
	def get_free_openrouter_models(api_key=_D):
		'Fetch available free models from OpenRouter API, ordered by coding performance';A='Rate limit reached. Using curated model list.';fallback_models=[_s,_BG,_BF,_Ao]
		if not api_key:return fallback_models
		try:
			import time;time.sleep(.5);headers={_CF:f"Bearer {api_key}",_j:_k,_CG:_CH,'X-Title':_CI};response=requests.get('https://openrouter.ai/api/v1/models',headers=headers,timeout=15)
			if response.status_code==429:st.info(A);return fallback_models
			response.raise_for_status();models_data=response.json();free_models=[]
			for model in models_data.get('data',[]):
				if model.get('pricing',{}).get('prompt','0')=='0'or':free'in model.get(_G,'')or model.get(_G)==_Ao:free_models.append(model.get(_G))
			performance_order=[_s,_BG,_BF,_Ao];ordered_models=[]
			for model in performance_order:
				if model in free_models:ordered_models.append(model);free_models.remove(model)
			ordered_models.extend(sorted(free_models));return ordered_models if ordered_models else fallback_models
		except requests.exceptions.RequestException as e:
			if'429'in str(e):st.info(A)
			else:st.info('Could not fetch latest models. Using curated list.')
			return fallback_models
		except Exception as e:return fallback_models
	try:openrouter_api_key=st.secrets[_Bb][_Bc]
	except KeyError:
		if'openrouter_api_key'not in st.session_state:st.session_state.openrouter_api_key=''
		openrouter_api_key=_D
	if ai_provider==_AH:
		st.subheader('Model Selection');col1,col2=st.columns([2,1])
		with col1:
			if st.button('Refresh Models',use_container_width=_A):get_ollama_models_cached.clear();st.session_state.ollama_models=get_ollama_models();st.rerun()
		with col2:auto_refresh=st.checkbox('Auto',help='Auto-refresh models')
		if not st.session_state.get('ollama_models')or auto_refresh:st.session_state.ollama_models=get_ollama_models()
		else:st.session_state.cache_stats[_W]+=1
		ollama_models=st.session_state.ollama_models
		if ollama_models:
			default_model=_AG;default_index=0
			if default_model in ollama_models:default_index=ollama_models.index(default_model)
			OLLAMA_MODEL=st.selectbox('Select a model:',ollama_models,index=default_index,help=f"Available models: {len(ollama_models)}",key='ollama_model_selectbox')
		else:
			OLLAMA_MODEL=_D
			if not ollama_available:st.info('💡 Local AI unavailable. Using Cloud AI provider.')
		with st.expander('🔧 Local Server Configuration',expanded=_B):
			if'ollama_host'not in st.session_state:st.session_state.ollama_host=getattr(globals(),'DEFAULT_OLLAMA_HOST',_CM)
			new_host=st.text_input('Ollama Host URL:',value=st.session_state.ollama_host,help='Default: http://localhost:11434\nRemote: http://192.168.x.x:11434',placeholder=_CM,key='ollama_host_input')
			if new_host!=st.session_state.ollama_host:
				st.session_state.ollama_host=new_host
				if'update_all_ollama_host_references'in globals():update_all_ollama_host_references(new_host)
				if _CN in globals():get_ollama_models_cached.clear()
				st.session_state.ollama_models=[]
				try:
					if _CN in globals():new_models=get_ollama_models_cached()
					else:new_models=get_ollama_models()
					st.session_state.ollama_models=new_models;st.success(f"✅ Connected to {new_host} - Found {len(new_models)} models")
				except Exception as e:st.error(f"❌ Failed to connect to {new_host}: {str(e)}");print(f"DEBUG: Error details: {e}");st.session_state.ollama_models=[]
				st.rerun()
			if st.session_state.ollama_host:st.info(f"Current host: {st.session_state.ollama_host}")
	elif not openrouter_api_key:
		openrouter_api_key=st.text_input(_CO,value=st.session_state.openrouter_api_key,type=_A2,key='openrouter_api_key_input',help='Get your key from https://openrouter.ai/keys or add it to .streamlit/secrets.toml');st.session_state.openrouter_api_key=openrouter_api_key
		if not openrouter_api_key:st.warning('Enter OpenRouter API key to select models');OPENROUTER_MODEL=_D
		else:
			free_models=get_free_openrouter_models(openrouter_api_key);default_index=0
			if _s in free_models:default_index=free_models.index(_s)
			OPENROUTER_MODEL=st.selectbox(_B2,free_models,index=default_index,key=_CP)
	else:
		free_models=get_free_openrouter_models(openrouter_api_key);default_index=0
		if _s in free_models:default_index=free_models.index(_s)
		OPENROUTER_MODEL=st.selectbox(_B2,free_models,index=default_index,key=_CP)
	with st.expander(_CQ):
		try:stats=st.session_state.cache_stats;total_db_requests=stats[_V]+stats[_Z];total_model_requests=stats[_W]+stats[_e];db_hit_rate=stats[_V]/max(total_db_requests,1)*100;model_hit_rate=stats[_W]/max(total_model_requests,1)*100;uptime=datetime.now()-stats[_A1];uptime_str=f"{uptime.days}d {uptime.seconds//3600}h {uptime.seconds%3600//60}m";stats_html=f'''
            <div class="cache-stats">
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Uptime</td><td>{uptime_str}</td></tr>
                    <tr><td>DB Cache Hit Rate</td><td>{db_hit_rate:.1f}%</td></tr>
                    <tr><td>DB Cache Hits</td><td>{stats[_V]}</td></tr>
                    <tr><td>DB Cache Misses</td><td>{stats[_Z]}</td></tr>
                    <tr><td>Model Cache Hit Rate</td><td>{model_hit_rate:.1f}%</td></tr>
                    <tr><td>Model Cache Hits</td><td>{stats[_W]}</td></tr>
                    <tr><td>Model Cache Misses</td><td>{stats[_e]}</td></tr>
                </table>
            </div>
            ''';st.markdown(stats_html,unsafe_allow_html=_A)
		except KeyError as e:st.error(f"Cache stats error: {e}. Resetting stats.");st.session_state.cache_stats={_V:0,_Z:0,_W:0,_e:0,_A1:datetime.now()};st.rerun()
		if st.button(_CR,key='reset_stats_button'):st.session_state.cache_stats={_V:0,_Z:0,_W:0,_e:0,_A1:datetime.now()};query_db.cache_clear();st.rerun()
if st.session_state.current_section==_Ar:
	st.markdown('<div class="section-header">Dashboard</div>',unsafe_allow_html=_A);col1,col2,col3,col4=st.columns(4)
	with col1:results,_=query_db('SELECT COUNT(*) FROM students');st.markdown('<div class="card">Total Students<br><b>{}</b></div>'.format(results[0][0]),unsafe_allow_html=_A)
	with col2:results,_=query_db('SELECT COUNT(*) FROM staff');st.markdown('<div class="card">Total Staff<br><b>{}</b></div>'.format(results[0][0]),unsafe_allow_html=_A)
	with col3:results,_=query_db("SELECT COUNT(*) FROM fees WHERE LOWER(status) = 'unpaid'");st.markdown('<div class="card">Unpaid Fees<br><b>{}</b></div>'.format(results[0][0]),unsafe_allow_html=_A)
	with col4:results,_=query_db("SELECT SUM(amount) FROM fees WHERE LOWER(status) = 'paid'");total_revenue=results[0][0]or 0;formatted_revenue='₡ : {:,.2f}'.format(total_revenue);st.markdown('<div class="card">Total Revenue<br><b>{}</b></div>'.format(formatted_revenue),unsafe_allow_html=_A)
	st.markdown('<div class="section-header">Performance Overview</div>',unsafe_allow_html=_A);col1,col2=st.columns(2)
	with col1:
		st.markdown('<div class="section-header">Fee Status Distribution</div>',unsafe_allow_html=_A);results,columns=query_db("\n            SELECT \n                CASE \n                    WHEN LOWER(status) = 'paid' THEN 'Paid' \n                    WHEN LOWER(status) = 'unpaid' THEN 'Unpaid'\n                    ELSE 'Other'\n                END AS status,\n                COUNT(*) AS student_count\n            FROM fees\n            GROUP BY LOWER(status)\n        ")
		if results:df=pd.DataFrame(results,columns=columns);fig=px.pie(df,names=_P,values='student_count',title=_CS,hole=.3);fig.update_traces(textposition='inside',textinfo='percent+label');st.plotly_chart(fig,use_container_width=_A)
		else:st.info(_CT)
	with col2:
		st.markdown('<div class="section-header">Average Performance per Subject</div>',unsafe_allow_html=_A);results,columns=query_db('SELECT subject, AVG(score) as average_score FROM test_results GROUP BY subject')
		if results:df=pd.DataFrame(results,columns=columns);fig=px.bar(df,x=_d,y='average_score',title='Average Test Scores by Subject',color=_d);fig.update_layout(showlegend=_B);fig.update_yaxes(title=_CU);fig.update_xaxes(title=_AS);st.plotly_chart(fig,use_container_width=_A)
		else:st.info('No test results found')
elif st.session_state.current_section==_B1:
	st.markdown('<div class="section-header">Student Management</div>',unsafe_allow_html=_A)
	with st.expander('Add New Student',expanded=_B):
		with st.form(key='student_form'):
			col1,col2=st.columns(2)
			with col1:
				name=st.text_input(_Ap,key='student_name_input');grade=st.text_input(_AR,key='student_grade_input',help='Enter grade level (e.g., 1R, JHS-2S, KG-B)');room=st.text_input(_AZ,key='student_room_input')
				if grade:teacher=get_teacher_for_student(grade)
				else:teacher='Enter grade first'
				st.text_input(_AI,value=teacher,disabled=_A,key='teacher_display')
			with col2:dob=st.date_input(_Bd,key='student_dob_input');phone=st.text_input(_B3,key='student_phone_input');address=st.text_area(_Be,key='student_address_input');enrollment_date=st.date_input(_Aa,key='student_enrollment_date_input');email=st.text_input(_Bf,key='student_email_input');parent_name=st.text_input(_A6,key='student_parent_input')
			if st.form_submit_button('Add Student'):conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('INSERT INTO students (name, grade, room, dob, phone, address, enrollment_date, email, parent_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',(name,grade,room,dob,phone,address,enrollment_date,email,parent_name));conn.commit();conn.close();query_db.cache_clear();st.success('Student added successfully!')
	st.markdown('<div class="section-header">Import Students</div>',unsafe_allow_html=_A)
	with st.expander(_AJ):
		uploaded_file=st.file_uploader(_AK,type=[_A8,_A9],key='student_import_uploader')
		if uploaded_file is not _D:
			if st.button('Import Students',key='student_import_button'):
				if import_data(_a,uploaded_file):st.success('Students imported successfully!')
	st.markdown('<div class="section-header">Student Records</div>',unsafe_allow_html=_A);results,columns=query_db('SELECT * FROM students LIMIT ?',(MAX_RECORDS,))
	if results:
		df=pd.DataFrame(results,columns=columns);df[_H]=_B
		if _AA in df.columns:df[_AA]=pd.to_datetime(df[_AA],errors='coerce').dt.date
		if _AB in df.columns:df[_AB]=pd.to_datetime(df[_AB],errors='coerce').dt.date
		if _T in df.columns:df[_T]=df[_T].astype(str).replace(_AW,'');df[_T]=df[_T].replace('None','')
		df[_Bg]=df[_T].apply(lambda x:get_teacher_for_student(x)if x else _CA);column_config={_G:st.column_config.NumberColumn('ID',disabled=_A),_H:st.column_config.CheckboxColumn(_p,default=_B),_E:st.column_config.TextColumn(_Ap,help='Student full name',max_chars=100,required=_A),_T:st.column_config.TextColumn(_AR,help='Grade level (e.g., 1R, JHS-2S, KG-B)',max_chars=20,required=_B),'room':st.column_config.TextColumn(_AZ,help='Classroom identifier'),_Bg:st.column_config.TextColumn(_AI,disabled=_A),_R:st.column_config.TextColumn(_B3,help='Contact phone number',max_chars=20),_K:st.column_config.TextColumn(_Bf,help='Email address',max_chars=100),_Bh:st.column_config.TextColumn(_Be,help='Home address',max_chars=200),_O:st.column_config.TextColumn(_A6,help='Parent or guardian name',max_chars=100),_AA:st.column_config.DateColumn(_Bd),_AB:st.column_config.DateColumn(_Aa)};edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config=column_config,key='student_data_editor')
		if st.button(_q,key='save_student_changes'):
			id_to_original={row[_G]:row for(_,row)in df.iterrows()};changes_made=_B
			for(_,edited_row)in edited_df.iterrows():
				student_id=edited_row[_G];original_row=id_to_original.get(student_id)
				if original_row is _D:continue
				updates={}
				for col in df.columns:
					if col not in[_G,_H,_Bg]:
						edited_val=edited_row[col];original_val=original_row[col]
						if col in[_AA,_AB]:
							if pd.isna(edited_val)and pd.isna(original_val):continue
							elif pd.isna(edited_val)and not pd.isna(original_val):updates[col]=_D
							elif not pd.isna(edited_val)and pd.isna(original_val):
								if isinstance(edited_val,date):updates[col]=edited_val.strftime(_AD)
								else:updates[col]=edited_val
							elif not pd.isna(edited_val)and not pd.isna(original_val):
								if isinstance(edited_val,date)and isinstance(original_val,date):
									if edited_val!=original_val:updates[col]=edited_val.strftime(_AD)
								elif edited_val!=original_val:
									if isinstance(edited_val,date):updates[col]=edited_val.strftime(_AD)
									else:updates[col]=edited_val
						else:
							edited_str=str(edited_val)if not pd.isna(edited_val)else'';original_str=str(original_val)if not pd.isna(original_val)else''
							if edited_str!=original_str:updates[col]=edited_str if edited_str else _D
				if updates:
					if update_record(_a,edited_row[_G],updates):changes_made=_A
			if changes_made:st.success('Changes saved successfully!');query_db.cache_clear();st.rerun()
		if st.button(_r,key='delete_students'):
			deleted_count=0
			for(index,row)in edited_df.iterrows():
				if row[_H]:
					if delete_record(_a,row[_G]):deleted_count+=1
			if deleted_count>0:st.success(f"Deleted {deleted_count} students");query_db.cache_clear();st.rerun()
	else:st.info('No student records found')
elif st.session_state.current_section==_BP:
	st.markdown('<div class="section-header">Staff Management</div>',unsafe_allow_html=_A)
	with st.expander('Add New Staff',expanded=_B):
		with st.form(key='staff_form'):
			col1,col2=st.columns(2)
			with col1:name=st.text_input(_Ap,key='staff_name_input');role=st.selectbox('Role',[_AI,_CV,'Support Staff','Upper Primary Staff','Lower Primary Staff','Junior High School Staff'],key='staff_role_selectbox');department=st.text_input('Department',key='staff_department_input')
			with col2:phone=st.text_input(_B3,key='staff_contact_input');hire_date=st.date_input('Hire Date',key='staff_hire_date_input');salary=st.number_input('Salary',min_value=.0,key='staff_salary_input')
			if st.form_submit_button('Add Staff'):conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('INSERT INTO staff (name, role, department, phone, hire_date, salary) VALUES (?, ?, ?, ?, ?, ?)',(name,role,department,phone,hire_date,salary));conn.commit();conn.close();query_db.cache_clear();st.success('Staff added successfully!')
	st.markdown('<div class="section-header">Import Staff</div>',unsafe_allow_html=_A)
	with st.expander(_AJ):
		uploaded_file=st.file_uploader(_AK,type=[_A8,_A9],key='staff_import_uploader')
		if uploaded_file is not _D:
			if st.button('Import Staff',key='staff_import_button'):
				if import_data(_t,uploaded_file):st.success('Staff imported successfully!')
	st.markdown('<div class="section-header">Staff Records</div>',unsafe_allow_html=_A);results,columns=query_db('SELECT * FROM staff LIMIT ?',(MAX_RECORDS,))
	if results:
		df=pd.DataFrame(results,columns=columns);df[_H]=_B;edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_H:st.column_config.CheckboxColumn(_p,help=_Ab,default=_B)})
		if st.button(_q,key='save_staff_changes'):
			for(index,row)in edited_df.iterrows():
				original_row=df.iloc[index]
				if not row.equals(original_row):
					updates={}
					for col in df.columns:
						if col!=_G and col!=_H and row[col]!=original_row[col]:updates[col]=row[col]
					if updates:
						if update_record(_t,row[_G],updates):st.success(f"Updated staff ID {row[_G]}")
		if st.button(_r,key='delete_staff'):
			deleted_count=0
			for(index,row)in edited_df.iterrows():
				if row[_H]:
					if delete_record(_t,row[_G]):deleted_count+=1
			if deleted_count>0:st.success(f"Deleted {deleted_count} staff members");st.rerun()
	else:st.info('No staff records found')
elif st.session_state.current_section==_BR:
	st.markdown('<div class="section-header">Attendance Tracking</div>',unsafe_allow_html=_A)
	with st.expander('Bulk Attendance Management',expanded=_A):
		attendance_date=st.date_input('Select Date',key='bulk_attendance_date');results,_=query_db(_h);student_options={row[1]:row[0]for row in results}
		if student_options:
			existing_attendance={};results,_=query_db('\n                SELECT a.student_id, a.status \n                FROM attendance a \n                WHERE a.date = ?\n            ',(attendance_date,))
			for row in results:existing_attendance[row[0]]=row[1]
			attendance_data=[]
			for(student_name,student_id)in student_options.items():status=existing_attendance.get(student_id,_B4);attendance_data.append({_N:student_id,_E:student_name,_P:status})
			df=pd.DataFrame(attendance_data);edited_df=st.data_editor(df,column_config={_N:_D,_E:st.column_config.TextColumn(_o,disabled=_A),_P:st.column_config.SelectboxColumn(_AC,options=[_B4,'Absent','Late'],required=_A)},hide_index=_A,use_container_width=_A)
			if st.button('Save Attendance',key='save_bulk_attendance'):
				conn=sqlite3.connect(DB_NAME);c=conn.cursor()
				for(_,row)in edited_df.iterrows():
					student_id=row[_N];status=row[_P]
					if status==_B4:c.execute('DELETE FROM attendance WHERE student_id = ? AND date = ?',(student_id,attendance_date))
					else:
						c.execute('SELECT 1 FROM attendance WHERE student_id = ? AND date = ?',(student_id,attendance_date));exists=c.fetchone()
						if exists:c.execute('\n                                UPDATE attendance \n                                SET status = ?\n                                WHERE student_id = ? AND date = ?\n                            ',(status,student_id,attendance_date))
						else:c.execute('\n                                INSERT INTO attendance (student_id, date, status)\n                                VALUES (?, ?, ?)\n                            ',(student_id,attendance_date,status))
				conn.commit();conn.close();st.success('Attendance saved successfully!')
		else:st.info('No students found')
	with st.expander(_CW,expanded=_B):
		with st.form(key='attendance_form'):
			results,_=query_db(_h);student_options={row[1]:row[0]for row in results};student_name=st.selectbox(_o,list(student_options.keys()),key='attendance_student_selectbox');date=st.date_input('Date',key='attendance_date_input');status=st.selectbox(_AC,[_B4,'Absent','Late'],key='attendance_status_selectbox')
			if st.form_submit_button(_CW):student_id=student_options[student_name];conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('INSERT INTO attendance (student_id, date, status) VALUES (?, ?, ?)',(student_id,date,status));conn.commit();conn.close();query_db.cache_clear();st.success('Attendance recorded!')
	st.markdown('<div class="section-header">Import Attendance</div>',unsafe_allow_html=_A)
	with st.expander(_AJ):
		uploaded_file=st.file_uploader(_AK,type=[_A8,_A9],key='attendance_import_uploader')
		if uploaded_file is not _D:
			if st.button('Import Attendance',key='attendance_import_button'):
				if import_data(_u,uploaded_file):st.success('Attendance records imported successfully!')
	st.markdown('<div class="section-header">Attendance Records</div>',unsafe_allow_html=_A);results,columns=query_db('SELECT a.id, s.name, a.date, a.status FROM attendance a JOIN students s ON a.student_id = s.id ORDER BY a.date DESC LIMIT ?',(MAX_RECORDS,));attendance_percentage_query="\n        SELECT \n            s.id AS student_id,\n            s.name,\n            COUNT(a.id) AS total_days,\n            SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) AS present_days,\n            ROUND((SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) * 100.0 / COUNT(a.id)), 2) AS attendance_percentage\n        FROM students s\n        LEFT JOIN attendance a ON s.id = a.student_id\n        GROUP BY s.id\n    ";percentage_results,percentage_columns=query_db(attendance_percentage_query)
	if results:
		df=pd.DataFrame(results,columns=columns);percentage_df=pd.DataFrame(percentage_results,columns=percentage_columns);df=df.merge(percentage_df[[_E,_Bi]],on=_E,how='left');edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_Bi:st.column_config.ProgressColumn('Attendance %',format='%.1f%%',min_value=0,max_value=100)})
		if st.button(_q,key='save_attendance_changes'):
			for(index,row)in edited_df.iterrows():
				original_row=df.iloc[index]
				if not row.equals(original_row):
					updates={}
					for col in df.columns:
						if col!=_G and col!=_Bi and row[col]!=original_row[col]:updates[col]=row[col]
					if updates:
						if update_record(_u,row[_G],updates):st.success(f"Updated attendance ID {row[_G]}")
			st.rerun()
		if st.button(_r,key='delete_attendance'):
			deleted_count=0
			for(index,row)in edited_df.iterrows():
				if _H in row and row[_H]:
					if delete_record(_u,row[_G]):deleted_count+=1
			if deleted_count>0:st.success(f"Deleted {deleted_count} attendance records");st.rerun()
	else:st.info('No attendance records found')
elif st.session_state.current_section==_BQ:
	st.markdown('<div class="section-header">Course Management System</div>',unsafe_allow_html=_A);tab1,tab2,tab3,tab4,tab5=st.tabs(['📅 Terms','📚 Courses','🏫 Classes','👥 Class Members','🎓 Promotions'])
	with tab1:
		st.markdown('### Academic Term Management');col1,col2=st.columns([3,1])
		with col1:
			st.subheader('Existing Terms');results,columns=query_db('SELECT * FROM terms')
			if results:
				df=pd.DataFrame(results,columns=columns);df[_H]=_B;edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_H:st.column_config.CheckboxColumn(_p,help=_Ab,default=_B)})
				if st.button(_q,key='save_term_changes'):
					for(index,row)in edited_df.iterrows():
						original_row=df.iloc[index]
						if not row.equals(original_row):
							updates={}
							for col in df.columns:
								if col!=_G and col!=_H and row[col]!=original_row[col]:updates[col]=row[col]
							if updates:conn=sqlite3.connect(DB_NAME);c=conn.cursor();set_clause=_I.join([f"{col} = ?"for col in updates.keys()]);values=list(updates.values())+[row[_G]];c.execute(f"UPDATE terms SET {set_clause} WHERE id = ?",values);conn.commit();conn.close();st.success(f"Updated term ID {row[_G]}")
				if st.button(_r,key='delete_terms'):
					deleted_count=0
					for(index,row)in edited_df.iterrows():
						if row[_H]:conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('DELETE FROM terms WHERE id = ?',(row[_G],));conn.commit();conn.close();deleted_count+=1
					if deleted_count>0:query_db.cache_clear();st.success(f"Deleted {deleted_count} term(s)");st.rerun()
			else:st.info('No terms defined')
		with col2:
			st.subheader('Add New Term')
			with st.form('term_form'):
				name=st.text_input(_Bj,key=_AL);start_date=st.date_input(_Bk,key='term_start');end_date=st.date_input(_Bl,key='term_end')
				if st.form_submit_button('Add Term'):
					if name and start_date and end_date:
						if start_date<end_date:conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('INSERT INTO terms (name, start_date, end_date) VALUES (?, ?, ?)',(name,start_date,end_date));conn.commit();conn.close();query_db.cache_clear();st.success('Term added successfully!');st.rerun()
						else:st.error('End date must be after start date')
					else:st.error('All fields are required')
	with tab2:
		st.markdown('### Course Management')
		with st.expander('Create Course & Enroll Students',expanded=_B):
			col1,col2=st.columns(2)
			with col1:
				st.subheader('Create New Course')
				with st.form('course_form',clear_on_submit=_A):
					name=st.text_input('Course Name',key='course_name');description=st.text_area(_CX,key='course_description');term_results,_=query_db(_Bm);term_options={row[1]:row[0]for row in term_results}if term_results else{}
					if term_options:term=st.selectbox(_B5,list(term_options.keys()),key='course_term_selectbox')
					else:st.warning(_CY);term=_D
					staff_results,_=query_db("\n                        SELECT id, name \n                        FROM staff \n                        WHERE role IN (\n                            'Teacher', \n                            'Upper Primary Staff', \n                            'Lower Primary Staff', \n                            'Junior High School Staff'\n                        )\n                    ");teacher_options={row[1]:row[0]for row in staff_results}if staff_results else{}
					if teacher_options:teacher=st.selectbox(_AI,list(teacher_options.keys()),key='course_teacher_selectbox')
					else:st.warning("No teachers available. Please add staff with role 'Teacher' first.");teacher=_D
					schedule=st.text_input('Schedule',key='course_schedule_input')
					if st.form_submit_button('Add Course'):
						if teacher and term:term_id=term_options[term];teacher_id=teacher_options[teacher];conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('\n                                INSERT INTO courses (name, description, teacher_id, term_id, schedule)\n                                VALUES (?, ?, ?, ?, ?)\n                            ',(name,description,teacher_id,term_id,schedule));conn.commit();conn.close();st.success('Course added successfully!')
						else:st.error('Please select a teacher and term')
			with col2:
				st.subheader('Enroll Students')
				with st.form('enrollment_form',clear_on_submit=_A):
					course_results,_=query_db('SELECT id, name FROM courses');course_options={row[1]:row[0]for row in course_results}
					if course_options:
						course=st.selectbox('Course',list(course_options.keys()),key='enroll_course_selectbox');student_results,_=query_db(_h);student_options={row[1]:row[0]for row in student_results}
						if student_options:
							student=st.selectbox(_o,list(student_options.keys()),key='enroll_student_selectbox');enrollment_date=st.date_input(_Aa,key='enrollment_date_input');status=st.selectbox(_AC,['Active','Completed'],key='enrollment_status_selectbox')
							if st.form_submit_button('Enroll Student'):course_id=course_options[course];student_id=student_options[student];conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('\n                                    INSERT INTO enrollments (student_id, course_id, enrollment_date, status)\n                                    VALUES (?, ?, ?, ?)\n                                ',(student_id,course_id,enrollment_date,status));conn.commit();conn.close();st.success('Student enrolled successfully!')
						else:st.info('No students available')
					else:st.info('No courses available. Create a course first.')
		st.markdown('#### Existing Courses');staff_results,_=query_db(_CZ);staff_options={row[1]:row[0]for row in staff_results}if staff_results else{};term_results,_=query_db(_Bm);term_options={row[1]:row[0]for row in term_results}if term_results else{};results,columns=query_db('SELECT * FROM courses LIMIT ?',(MAX_RECORDS,))
		if results:
			df=pd.DataFrame(results,columns=columns);df[_H]=_B;edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_H:st.column_config.CheckboxColumn(_p,help=_Ab,default=_B),_Bn:st.column_config.SelectboxColumn(_AI,help='Select teacher',options=list(staff_options.keys()),required=_A),_B6:st.column_config.SelectboxColumn(_B5,help='Select term',options=list(term_options.keys()),required=_A)})
			if st.button('Save Course Changes',key='save_course_changes'):
				id_to_original={row[_G]:row for(_,row)in df.iterrows()}
				for(_,edited_row)in edited_df.iterrows():
					original_row=id_to_original.get(edited_row[_G])
					if original_row is _D:continue
					updates={}
					for col in df.columns:
						if col not in[_G,_H]and edited_row[col]!=original_row[col]:
							if col==_Bn:updates[col]=staff_options.get(edited_row[col],original_row[col])
							elif col==_B6:updates[col]=term_options.get(edited_row[col],original_row[col])
							else:updates[col]=edited_row[col]
					if updates:
						if update_record(_B7,edited_row[_G],updates):st.success(f"Updated course ID {edited_row[_G]}")
			if st.button('Delete Selected Courses',key='delete_courses'):
				deleted_count=0
				for(index,row)in edited_df.iterrows():
					if row[_H]:
						if delete_record(_B7,row[_G]):deleted_count+=1
				if deleted_count>0:st.success(f"Deleted {deleted_count} courses");st.rerun()
		else:st.info('No course records found')
	with tab3:
		st.markdown('### Class Management')
		with st.expander('Create New Class',expanded=_B):
			with st.form('class_form',clear_on_submit=_A):
				class_name=st.text_input(_Ca,key='class_name_input');room=st.text_input(_AZ,key='class_room_input');teacher_results,_=query_db(_CZ)
				if teacher_results:
					teacher_options={row[1]:row[0]for row in teacher_results};term_results,_=query_db(_Bm);term_options={row[1]:row[0]for row in term_results}if term_results else{};col1,col2=st.columns(2)
					with col1:teacher_name=st.selectbox(_CB,options=list(teacher_options.keys()),key='class_teacher_selectbox')
					with col2:
						if term_options:term_name=st.selectbox(_B5,options=list(term_options.keys()),key='class_term_selectbox')
						else:st.warning('No terms available');term_name=_D
					if st.form_submit_button('Create Class'):
						if class_name and teacher_name and term_name:teacher_id=teacher_options[teacher_name];term_id=term_options[term_name];conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute(_Cb,(class_name,room,teacher_id,term_id));conn.commit();conn.close();st.success('Class created successfully!');st.rerun()
						else:st.error('Class Name, Teacher, and Term are required')
				else:st.warning('No staff available. Please add staff first.')
		st.markdown('#### Existing Classes');class_query=_Cc;class_results,class_columns=query_db(class_query)
		if class_results:
			df=pd.DataFrame(class_results,columns=class_columns);df[_H]=_B;edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_Ac:_D,_H:st.column_config.CheckboxColumn(_p,default=_B),_Ad:st.column_config.SelectboxColumn(_AI,options=list(teacher_options.keys())if'teacher_options'in locals()else[]),_AL:st.column_config.SelectboxColumn(_B5,options=list(term_options.keys())if'term_options'in locals()else[])},hide_index=_A)
			if st.button('Save Class Changes',key='save_class_changes'):
				for(index,row)in edited_df.iterrows():
					original_row=df.iloc[index]
					if not row.equals(original_row):
						updates={}
						for col in df.columns:
							if col not in[_Ac,_H]and row[col]!=original_row[col]:updates[col]=row[col]
						if updates:
							if _Ad in updates:updates[_Bn]=teacher_options[updates[_Ad]];del updates[_Ad]
							if _AL in updates:updates[_B6]=term_options[updates[_AL]];del updates[_AL]
							conn=sqlite3.connect(DB_NAME);c=conn.cursor();set_clause=_I.join([f"{col} = ?"for col in updates.keys()]);values=list(updates.values())+[row[_Ac]];c.execute(f"UPDATE classes SET {set_clause} WHERE id = ?",values);conn.commit();conn.close()
				st.success('Class changes saved!')
			if st.button('Delete Selected Classes',key='delete_classes'):
				deleted_count=0
				for(index,row)in edited_df.iterrows():
					if row[_H]:conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('DELETE FROM classes WHERE id = ?',(row[_Ac],));conn.commit();conn.close();deleted_count+=1
				if deleted_count>0:st.success(f"Deleted {deleted_count} classes");st.rerun()
				if deleted_count>0:st.success(f"Deleted {deleted_count} classes");st.rerun()
		else:st.info('No classes defined yet')
	with tab4:
		st.markdown('### Class Members Management');class_query=_Cc;class_results,class_columns=query_db(class_query)
		if class_results:
			class_df=pd.DataFrame(class_results,columns=class_columns);selected_class_name=st.selectbox('Select Class to Manage Members',class_df[_Bo].unique(),key='class_member_select')
			if selected_class_name:
				selected_class_id=class_df[class_df[_Bo]==selected_class_name].iloc[0][_Ac];class_row=class_df[class_df[_Bo]==selected_class_name].iloc[0];class_term_id=class_row[_B6];class_room=class_row['room'];st.info(f"**Class:** {selected_class_name} | **Teacher:** {class_row[_Ad]} | **Term:** {class_row[_AL]} | **Room:** {class_room}");conn=sqlite3.connect(DB_NAME);c=conn.cursor();sync_query="\n                    INSERT OR IGNORE INTO class_members (class_id, student_id, term_id)\n                    SELECT ?, s.id, ?\n                    FROM students s\n                    WHERE TRIM(UPPER(COALESCE(s.room, ''))) = TRIM(UPPER(?))\n                    AND s.id NOT IN (\n                        SELECT student_id \n                        FROM class_members \n                        WHERE term_id = ?\n                    )\n                ";c.execute(sync_query,(selected_class_id,class_term_id,class_room,class_term_id));synced_count=c.rowcount;conn.commit();conn.close()
				if synced_count>0:st.success(f"Auto-synced {synced_count} students from room assignments to class members")
				col1,col2=st.columns([2,1])
				with col1:
					member_query='\n                        SELECT \n                            s.id AS student_id,\n                            s.name AS student_name,\n                            s.id AS student_number,\n                            s.grade,\n                            s.phone,\n                            s.address,\n                            s.parent_name,\n                            s.enrollment_date,\n                            s.email,\n                            s.dob\n                        FROM students s\n                        JOIN class_members cm ON s.id = cm.student_id\n                        WHERE cm.class_id = ? AND cm.term_id = ?\n                        ORDER BY s.name\n                    ';member_results,member_columns=query_db(member_query,(selected_class_id,class_term_id));member_df=pd.DataFrame(member_results,columns=member_columns)if member_results else pd.DataFrame(columns=member_columns);st.subheader(f"Current Members ({len(member_df)} students)")
					if not member_df.empty:
						member_df[_Ae]=_B;display_columns=[_B8,_B9,_T,_R,_Bh,_O,_AB,_K,_AA,_Ae];edited_member_df=st.data_editor(member_df[display_columns],column_config={_B8:st.column_config.TextColumn(_Bp,disabled=_A,width=_AM),_B9:st.column_config.NumberColumn('Student ID',disabled=_A,width=_Bq),_T:st.column_config.NumberColumn(_AR,disabled=_A,width=_Bq),_R:st.column_config.TextColumn(_B3,disabled=_A,width=_AM),_Bh:st.column_config.TextColumn(_Be,disabled=_A,width='large'),_O:st.column_config.TextColumn('Parent Name',disabled=_A,width=_AM),_AB:st.column_config.DateColumn(_Aa,disabled=_A,width=_AM),_K:st.column_config.TextColumn(_Bf,disabled=_A,width=_AM),_AA:st.column_config.DateColumn(_Bd,disabled=_A,width=_AM),_Ae:st.column_config.CheckboxColumn('Remove?',default=_B,width=_Bq)},hide_index=_A,use_container_width=_A,key='class_members_editor')
						if st.button('Remove Selected Students',key='remove_students_button',type=_BA):
							if edited_member_df[_Ae].any():
								conn=sqlite3.connect(DB_NAME);c=conn.cursor();removed_count=0
								try:
									for(_,row)in edited_member_df.iterrows():
										if row[_Ae]:c.execute('DELETE FROM class_members WHERE student_id = ? AND class_id = ? AND term_id = ?',(row[_B9],selected_class_id,class_term_id));c.execute('UPDATE students SET room = NULL WHERE id = ?',(row[_B9],));removed_count+=1
									conn.commit()
									if removed_count>0:st.success(f"Removed {removed_count} students from {selected_class_name}");st.rerun()
									else:st.warning('No students were removed')
								except sqlite3.IntegrityError as e:st.error(f"Error removing students: {str(e)}");conn.rollback()
								finally:conn.close()
							else:st.warning('Please select at least one student to remove')
					else:st.info('No students found in this class')
				with col2:
					st.subheader('Add New Members');available_query='\n                        SELECT DISTINCT s.id, s.name, s.grade, s.phone, s.parent_name\n                        FROM students s\n                        WHERE s.id NOT IN (\n                            SELECT cm.student_id \n                            FROM class_members cm \n                            WHERE cm.term_id = ?\n                        )\n                        ORDER BY s.name\n                    ';available_results,_=query_db(available_query,(class_term_id,))
					if available_results:
						student_display_options=[f"{row[1]} (ID: {row[0]}, Grade: {row[2]})"for row in available_results];student_name_to_id={f"{row[1]} (ID: {row[0]}, Grade: {row[2]})":row[0]for row in available_results};selected_students=st.multiselect('Select Students to Add',student_display_options,key='add_students_multiselect',help=f"Available students not assigned to any class this term")
						if st.button('Add Selected Students',key='add_students_button',type=_X)and selected_students:
							conn=sqlite3.connect(DB_NAME);c=conn.cursor();added_count=0;errors=[]
							for student_display in selected_students:
								student_id=student_name_to_id[student_display];student_name=student_display.split(' (ID:')[0]
								try:c.execute('\n                                        INSERT OR REPLACE INTO class_members (class_id, student_id, term_id)\n                                        VALUES (?, ?, ?)\n                                    ',(selected_class_id,student_id,class_term_id));c.execute('UPDATE students SET room = ? WHERE id = ?',(class_room,student_id));added_count+=1
								except sqlite3.IntegrityError as e:errors.append(f"{student_name}: {str(e)}")
							conn.commit();conn.close()
							if added_count>0:st.success(f"Added {added_count} students to {selected_class_name}")
							if errors:
								for error in errors:st.warning(error)
							if added_count>0:st.rerun()
					else:st.info('All students are already assigned to classes this term')
					st.markdown(_c);st.markdown('**Quick Stats:**');total_students=len(query_db('SELECT id FROM students')[0]);assigned_query='\n                        SELECT COUNT(DISTINCT student_id) \n                        FROM class_members \n                        WHERE term_id = ?\n                    ';assigned_students=query_db(assigned_query,(class_term_id,))[0][0][0];unassigned_students=total_students-assigned_students;st.metric(_Cd,total_students);st.metric('Assigned This Term',assigned_students);st.metric('Unassigned This Term',unassigned_students);class_counts_query='\n                        SELECT c.name, COUNT(cm.student_id) as member_count\n                        FROM classes c\n                        LEFT JOIN class_members cm ON c.id = cm.class_id AND cm.term_id = ?\n                        WHERE c.term_id = ?\n                        GROUP BY c.id, c.name\n                        ORDER BY c.name\n                    ';class_counts=query_db(class_counts_query,(class_term_id,class_term_id))[0]
					if class_counts:
						st.markdown('**Class Sizes This Term:**')
						for(class_name,count)in class_counts:st.write(f"  - {class_name}: {count} students")
		else:st.info('No classes available. Please create classes first in the Classes tab.')
		st.markdown(_c)
		with st.expander('🔄 Bulk Sync from Room Assignments',expanded=_B):
			st.markdown('**Sync All Students from Room Data**');st.markdown("This will add students to class_members table based on their room assignments, only if they're not already assigned.")
			if st.button('Sync All Room Assignments',key='bulk_sync_rooms'):
				conn=sqlite3.connect(DB_NAME);c=conn.cursor();current_term_query='SELECT id FROM terms ORDER BY id DESC LIMIT 1';current_term_result=query_db(current_term_query)
				if current_term_result[0]:
					current_term_id=current_term_result[0][0][0];bulk_sync_query="\n                        INSERT OR IGNORE INTO class_members (class_id, student_id, term_id)\n                        SELECT c.id, s.id, ?\n                        FROM students s\n                        JOIN classes c ON TRIM(UPPER(COALESCE(s.room, ''))) = TRIM(UPPER(c.room))\n                        WHERE c.term_id = ?\n                        AND s.id NOT IN (\n                            SELECT student_id \n                            FROM class_members \n                            WHERE term_id = ?\n                        )\n                    ";c.execute(bulk_sync_query,(current_term_id,current_term_id,current_term_id));synced_count=c.rowcount;conn.commit();conn.close()
					if synced_count>0:st.success(f"Bulk synced {synced_count} students from room assignments to class members");st.rerun()
					else:st.info('No students needed syncing - all room assignments already match class members')
				else:st.error(_Ce)
	with tab5:
		st.markdown('### Student Promotions & Graduation');term_query=_Cf;term_results,_=query_db(term_query)
		if not term_results:st.warning(_Ce)
		else:
			current_term=term_results[0];next_term=term_results[1]if len(term_results)>1 else _D;col1,col2=st.columns([2,1])
			with col1:
				st.subheader('📋 Promotion Rules')
				with st.expander('Configure Promotion Rules',expanded=_B):
					with st.form('promotion_rules_form'):
						from_grade=st.selectbox('From Grade',['1','2','3','4','5','6','JHS1','JHS2','JHS3']);to_grade=st.selectbox('To Grade',['2','3','4','5','6','JHS1','JHS2','JHS3',_Af]);min_score=st.number_input('Minimum Average Score',min_value=.0,max_value=1e2,value=6e1);auto_promote=st.checkbox('Auto-promote at term end',value=_A)
						if st.form_submit_button('Save Rule'):
							try:conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('\n                                    INSERT OR REPLACE INTO promotion_rules \n                                    (from_grade, to_grade, minimum_score, auto_promote)\n                                    VALUES (?, ?, ?, ?)\n                                ',(from_grade,to_grade,min_score,auto_promote));conn.commit();conn.close();st.success('Promotion rule saved!')
							except Exception as e:st.error(f"Error saving rule: {str(e)}")
				try:
					rules_results,rules_columns=query_db('SELECT * FROM promotion_rules')
					if rules_results:rules_df=pd.DataFrame(rules_results,columns=rules_columns);st.subheader('Current Promotion Rules');st.dataframe(rules_df,use_container_width=_A)
				except Exception as e:st.warning(f"Could not load promotion rules: {str(e)}")
				st.subheader('🎓 Students Eligible for Promotion')
				try:
					eligibility_query="\n                    SELECT DISTINCT\n                        s.id,\n                        s.name,\n                        s.grade,\n                        COALESCE(AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END), 0) as avg_score,\n                        COALESCE(pr.minimum_score, 60.0) as minimum_score,\n                        COALESCE(pr.to_grade, 'Review Required') as to_grade,\n                        CASE \n                            WHEN AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END) >= COALESCE(pr.minimum_score, 60.0) THEN 'Eligible'\n                            ELSE 'Needs Review'\n                        END as status,\n                        CASE \n                            WHEN s.grade = 'JHS3' AND AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END) >= COALESCE(pr.minimum_score, 60.0) THEN 'Graduate'\n                            WHEN AVG(CASE WHEN tr.score IS NOT NULL THEN tr.score END) >= COALESCE(pr.minimum_score, 60.0) THEN COALESCE(pr.to_grade, 'Next Grade')\n                            ELSE 'Repeat ' || s.grade\n                        END as recommended_action\n                    FROM students s\n                    LEFT JOIN test_results tr ON s.id = tr.student_id\n                    LEFT JOIN promotion_rules pr ON s.grade = pr.from_grade\n                    WHERE s.status = 'Active' OR s.status IS NULL\n                    GROUP BY s.id, s.name, s.grade, pr.minimum_score, pr.to_grade\n                    ORDER BY s.grade, s.name\n                    ";eligibility_results,eligibility_columns=query_db(eligibility_query)
					if eligibility_results:
						df=pd.DataFrame(eligibility_results,columns=eligibility_columns);df[_Br]=df[_P]=='Eligible';edited_df=st.data_editor(df,column_config={_G:_D,_E:st.column_config.TextColumn(_Bp,disabled=_A),_T:st.column_config.TextColumn('Current Grade',disabled=_A),'avg_score':st.column_config.NumberColumn(_CU,format='%.1f',disabled=_A),'minimum_score':st.column_config.NumberColumn('Required Score',disabled=_A),'to_grade':st.column_config.TextColumn('Next Grade',disabled=_A),_P:st.column_config.TextColumn(_AC,disabled=_A),_Cg:st.column_config.TextColumn('Recommended Action',disabled=_A),_Br:st.column_config.CheckboxColumn('Promote/Graduate?',default=_B)},hide_index=_A,use_container_width=_A,key='promotion_editor')
						if st.button('🚀 Process Selected Promotions',type=_X,key='process_promotions'):
							try:
								conn=sqlite3.connect(DB_NAME);c=conn.cursor();promoted_count=0;graduated_count=0;repeated_count=0
								for(_,row)in edited_df.iterrows():
									if row[_Br]:
										student_id=row[_G];current_grade=row[_T];recommended_action=row[_Cg]
										if recommended_action==_Af:c.execute("UPDATE students SET status = 'Graduate' WHERE id = ?",(student_id,));c.execute(_Bs,(student_id,current_term[0],current_term[0],current_grade,_Af,_Af,datetime.now().date(),'Graduated from JHS3'));graduated_count+=1
										elif not recommended_action.startswith(_Bt):new_grade=recommended_action;c.execute('UPDATE students SET grade = ? WHERE id = ?',(new_grade,student_id));c.execute(_Bs,(student_id,current_term[0],next_term[0]if next_term else current_term[0],current_grade,new_grade,'Promote',datetime.now().date(),f"Promoted from {current_grade} to {new_grade}"));promoted_count+=1
										else:c.execute(_Bs,(student_id,current_term[0],next_term[0]if next_term else current_term[0],current_grade,current_grade,_Bt,datetime.now().date(),f"Repeating {current_grade} - insufficient score"));repeated_count+=1
								conn.commit();conn.close()
								if promoted_count>0:st.success(f"✅ Promoted {promoted_count} students")
								if graduated_count>0:st.success(f"🎓 Graduated {graduated_count} students")
								if repeated_count>0:st.info(f"🔄 {repeated_count} students will repeat their grade")
								if promoted_count>0 or graduated_count>0 or repeated_count>0:st.rerun()
							except Exception as e:st.error(f"Error processing promotions: {str(e)}")
					else:st.info('No students found for promotion evaluation')
				except Exception as e:st.error(f"Error loading student data: {str(e)}")
			with col2:
				st.subheader('📊 Promotion Stats')
				try:
					grade_query="\n                    SELECT grade, COUNT(*) as count\n                    FROM students \n                    WHERE status = 'Active' OR status IS NULL\n                    GROUP BY grade\n                    ORDER BY \n                        CASE \n                            WHEN grade LIKE 'JHS%' THEN 10 + CAST(SUBSTR(grade, 4) AS INTEGER)\n                            ELSE CAST(grade AS INTEGER)\n                        END\n                    ";grade_results,_=query_db(grade_query)
					if grade_results:
						st.markdown('**Current Grade Distribution:**')
						for(grade,count)in grade_results:st.write(f"  - Grade {grade}: {count} students")
				except Exception as e:st.warning(f"Could not load grade stats: {str(e)}")
				st.markdown(_c);st.subheader('📜 Recent Promotions')
				try:
					history_query='\n                    SELECT s.name, ph.from_grade, ph.to_grade, ph.action, ph.promotion_date\n                    FROM promotion_history ph\n                    JOIN students s ON ph.student_id = s.id\n                    ORDER BY ph.promotion_date DESC\n                    LIMIT 10\n                    ';history_results,_=query_db(history_query)
					if history_results:
						for(name,from_grade,to_grade,action,date)in history_results:
							if action==_Af:st.write(f"🎓 **{name}** graduated from {from_grade}")
							elif action=='Promote':st.write(f"⬆️ **{name}** {from_grade} → {to_grade}")
							elif action==_Bt:st.write(f"🔄 **{name}** repeating {from_grade}")
							st.caption(f"   {date}")
					else:st.info('No promotion history yet')
				except Exception as e:st.warning(f"Could not load promotion history: {str(e)}")
elif st.session_state.current_section==_BS:
	st.markdown('<div class="section-header">Test Results</div>',unsafe_allow_html=_A)
	with st.expander(_Ch,expanded=_B):
		with st.form(key='test_results_form'):
			results,_=query_db(_h);student_options={row[1]:row[0]for row in results};student_name=st.selectbox(_o,list(student_options.keys()),key='test_results_student_selectbox');subject=st.text_input(_AS,key='test_results_subject_input');score=st.number_input('Score',min_value=.0,max_value=1e2,key='test_results_score_input');test_date=st.date_input('Test Date',key='test_results_date_input')
			if st.form_submit_button(_Ch):student_id=student_options[student_name];conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('INSERT INTO test_results (student_id, subject, score, test_date) VALUES (?, ?, ?, ?)',(student_id,subject,score,test_date));conn.commit();conn.close();query_db.cache_clear();st.success('Test result recorded!')
	st.markdown('<div class="section-header">Import Test Results</div>',unsafe_allow_html=_A)
	with st.expander(_AJ):
		uploaded_file=st.file_uploader(_AK,type=[_A8,_A9],key='test_import_uploader')
		if uploaded_file is not _D:
			if st.button('Import Test Results',key='test_import_button'):
				if import_data(_v,uploaded_file):st.success('Test results imported successfully!')
	st.markdown('<div class="section-header">Test Result Records</div>',unsafe_allow_html=_A);results,columns=query_db('SELECT * FROM test_results LIMIT ?',(MAX_RECORDS,))
	if results:
		df=pd.DataFrame(results,columns=columns);df[_H]=_B;edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_H:st.column_config.CheckboxColumn(_p,help=_Ab,default=_B)})
		if st.button(_q,key='save_test_changes'):
			for(index,row)in edited_df.iterrows():
				original_row=df.iloc[index]
				if not row.equals(original_row):
					updates={}
					for col in df.columns:
						if col!=_G and col!=_H and row[col]!=original_row[col]:updates[col]=row[col]
					if updates:
						if update_record(_v,row[_G],updates):st.success(f"Updated test result ID {row[_G]}")
		if st.button(_r,key='delete_test_results'):
			deleted_count=0
			for(index,row)in edited_df.iterrows():
				if row[_H]:
					if delete_record(_v,row[_G]):deleted_count+=1
			if deleted_count>0:st.success(f"Deleted {deleted_count} test results");st.rerun()
	else:st.info('No test results records found')
	if results:fig=px.line(df,x=_Ag,y=_i,color=_d,title='Test Scores Over Time');st.plotly_chart(fig,use_container_width=_A)
elif st.session_state.current_section=='Fees':
	st.markdown('<div class="section-header">Fees Management</div>',unsafe_allow_html=_A)
	with st.expander(_Ci,expanded=_B):
		with st.form(key='fees_form'):
			results,_=query_db(_h);student_options={row[1]:row[0]for row in results};student_name=st.selectbox(_o,list(student_options.keys()),key='fees_student_selectbox');amount=st.number_input(_Ah,min_value=.0,key='fees_amount_input');due_date=st.date_input('Due Date',key='fees_due_date_input');status=st.selectbox(_AC,['Paid','Unpaid'],key='fees_status_selectbox');payment_date=st.date_input(_Cj,key='fees_payment_date_input')if status=='Paid'else _D
			if st.form_submit_button(_Ci):student_id=student_options[student_name];conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('INSERT INTO fees (student_id, amount, due_date, status, payment_date) VALUES (?, ?, ?, ?, ?)',(student_id,amount,due_date,status,payment_date));conn.commit();conn.close();query_db.cache_clear();st.success('Fee recorded!')
	st.markdown('<div class="section-header">Import Fees</div>',unsafe_allow_html=_A)
	with st.expander(_AJ):
		uploaded_file=st.file_uploader(_AK,type=[_A8,_A9],key='fees_import_uploader')
		if uploaded_file is not _D:
			if st.button('Import Fees',key='fees_import_button'):
				if import_data(_w,uploaded_file):st.success('Fees imported successfully!')
	st.markdown('<div class="section-header">Fee Records</div>',unsafe_allow_html=_A);results,columns=query_db('SELECT * FROM fees LIMIT ?',(MAX_RECORDS,))
	if results:
		df=pd.DataFrame(results,columns=columns);df[_H]=_B;edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_H:st.column_config.CheckboxColumn(_p,help=_Ab,default=_B)})
		if st.button(_q,key='save_fee_changes'):
			for(index,row)in edited_df.iterrows():
				original_row=df.iloc[index]
				if not row.equals(original_row):
					updates={}
					for col in df.columns:
						if col!=_G and col!=_H and row[col]!=original_row[col]:updates[col]=row[col]
					if updates:
						if update_record(_w,row[_G],updates):st.success(f"Updated fees ID {row[_G]}")
		if st.button(_r,key='delete_fees'):
			deleted_count=0
			for(index,row)in edited_df.iterrows():
				if row[_H]:
					if delete_record(_w,row[_G]):deleted_count+=1
			if deleted_count>0:st.success(f"Deleted {deleted_count} fees");st.rerun()
	else:st.info(_CT)
elif st.session_state.current_section==_BT:
	st.markdown('<div class="section-header">Financial Management</div>',unsafe_allow_html=_A);st.markdown('<div class="section-header">Expenditure Tracking</div>',unsafe_allow_html=_A)
	with st.expander(_Ck,expanded=_B):
		with st.form(key='expenditure_form'):
			col1,col2=st.columns(2)
			with col1:category=st.selectbox(_Bu,['Salaries','Facilities','Supplies','Technology','Transportation','Food','Other'],key='exp_category');description=st.text_area(_CX,key='exp_description')
			with col2:amount=st.number_input(_Ah,min_value=.01,key='exp_amount');date=st.date_input('Date',key='exp_date');vendor=st.text_input('Vendor',key='exp_vendor');notes=st.text_area('Notes',key='exp_notes')
			if st.form_submit_button(_Ck):conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute('INSERT INTO expenditures (category, description, amount, date, vendor, notes) VALUES (?, ?, ?, ?, ?, ?)',(category,description,amount,date,vendor,notes));conn.commit();conn.close();query_db.cache_clear();st.success('Expenditure recorded!')
	st.markdown('<div class="section-header">Import Expenditures</div>',unsafe_allow_html=_A)
	with st.expander(_AJ):
		uploaded_file=st.file_uploader(_AK,type=[_A8,_A9],key='exp_import_uploader')
		if uploaded_file is not _D:
			if st.button('Import Expenditures',key='exp_import_button'):
				if import_data(_x,uploaded_file):st.success('Expenditures imported successfully!')
	st.markdown('<div class="section-header">Expenditure Records</div>',unsafe_allow_html=_A);results,columns=query_db('SELECT * FROM expenditures ORDER BY date DESC LIMIT ?',(MAX_RECORDS,))
	if results:
		df=pd.DataFrame(results,columns=columns);edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g)
		if st.button(_q,key='save_exp_changes'):
			for(index,row)in edited_df.iterrows():
				original_row=df.iloc[index]
				if not row.equals(original_row):
					updates={}
					for col in df.columns:
						if col!=_G and row[col]!=original_row[col]:updates[col]=row[col]
					if updates:
						if update_record(_x,row[_G],updates):st.success(f"Updated expenditure ID {row[_G]}")
			st.rerun()
		if st.button(_r,key='delete_exp'):
			deleted_count=0
			for(index,row)in edited_df.iterrows():
				if _H in row and row[_H]:
					if delete_record(_x,row[_G]):deleted_count+=1
			if deleted_count>0:st.success(f"Deleted {deleted_count} expenditures");st.rerun()
	else:st.info('No expenditure records found')
	if results:st.markdown('<div class="section-header">Expenditure Analysis</div>',unsafe_allow_html=_A);fig=px.pie(df,names='category',values='amount',title=_Cl);st.plotly_chart(fig,use_container_width=_A)
elif st.session_state.current_section==_BU:
	st.markdown('<div class="section-header">Reports</div>',unsafe_allow_html=_A);report_type=st.selectbox('Report Type',[_Cm,_Cn,_Co,_Cp,_Cq],key='report_type_selectbox')
	if report_type==_Cm:
		subjects_query='SELECT DISTINCT subject FROM test_results';subjects=[row[0]for row in query_db(subjects_query)[0]];subject_selects=[]
		for subject in subjects:subject_selects.append(f"ROUND(AVG(CASE WHEN t.subject = '{subject}' THEN t.score ELSE NULL END), 2) AS \"{subject}\"")
		if not subject_selects:subject_selects=['NULL AS no_subjects']
		query=f'''
            SELECT 
                s.name,
                ROUND(AVG(t.score), 2) AS "AVG",
                {_I.join(subject_selects)}
            FROM test_results t
            JOIN students s ON t.student_id = s.id
            GROUP BY s.id
        ''';results,columns=query_db(query);df=pd.DataFrame(results,columns=columns);st.dataframe(df,use_container_width=_A);fig=px.bar(df,x=_E,y='AVG',title='Average Student Performance');st.plotly_chart(fig,use_container_width=_A)
	elif report_type==_Cn:
		st.markdown('<div class="section-header">Individual Student Performance Report</div>',unsafe_allow_html=_A);results,_=query_db(_h);student_options={row[1]:row[0]for row in results}
		if student_options:
			selected_student=st.selectbox(_Bv,list(student_options.keys()),key='student_performance_select')
			if selected_student:
				student_id=student_options[selected_student];student_info=query_db('SELECT name, id, grade, enrollment_date FROM students WHERE id = ?',(student_id,))
				if student_info[0]:name,db_id,grade,enrollment_date=student_info[0][0];st.markdown(f'''
                    <div class="student-report">
                        <h3>{name}</h3>
                        <p><strong>ID No:</strong> {db_id}</p>
                        <p><strong>Grade:</strong> {grade}</p>
                        <p><strong>Enrollment Date:</strong> {enrollment_date}</p>
                    </div>
                    ''',unsafe_allow_html=_A)
				results,columns=query_db('SELECT subject, score, test_date FROM test_results WHERE student_id = ? ORDER BY test_date',(student_id,))
				if results:df=pd.DataFrame(results,columns=columns);st.markdown('<div class="section-header">Performance Over Time</div>',unsafe_allow_html=_A);fig_time=px.line(df,x=_Ag,y=_i,color=_d,title=f"Test Scores Over Time - {selected_student}",markers=_A);st.plotly_chart(fig_time,use_container_width=_A);st.markdown('<div class="section-header">Subject-wise Performance</div>',unsafe_allow_html=_A);subject_avg=df.groupby(_d)[_i].mean().reset_index();fig_subject=px.bar(subject_avg,x=_d,y=_i,title=f"Average Score by Subject - {selected_student}",color=_d);fig_subject.update_layout(showlegend=_B);st.plotly_chart(fig_subject,use_container_width=_A);st.markdown('<div class="section-header">Latest Test Results</div>',unsafe_allow_html=_A);latest_results=df.sort_values(_Ag,ascending=_B).head(10);st.dataframe(latest_results,use_container_width=_A);st.markdown('<div class="section-header">Performance Summary</div>',unsafe_allow_html=_A);col1,col2,col3=st.columns(3);overall_avg=df[_i].mean();best_subject=subject_avg.loc[subject_avg[_i].idxmax(),_d];worst_subject=subject_avg.loc[subject_avg[_i].idxmin(),_d];col1.metric(_Cr,f"{overall_avg:.1f}/100");col2.metric(_Cs,best_subject);col3.metric('Improvement Needed',worst_subject);report_data={_Bp:name,_AR:grade,_Aa:enrollment_date,_Cr:overall_avg,_Cs:best_subject,'Worst Subject':worst_subject};report_df=pd.DataFrame([report_data]);csv=report_df.to_csv(index=_B).encode(_M);st.download_button('📥 Download Performance Summary',data=csv,file_name=f"student_performance_{name.replace(' ','_')}.csv",mime=_AN)
				else:st.info(f"No test results found for {selected_student}")
		else:st.info('No students found in database')
	elif report_type==_Co:results,columns=query_db("SELECT s.name, COUNT(CASE WHEN a.status = 'Present' THEN 1 END) as present_days, COUNT(*) as total_days FROM attendance a JOIN students s ON a.student_id = s.id GROUP BY s.id");df=pd.DataFrame(results,columns=columns);df[_Ct]=df['present_days']/df['total_days']*100;st.dataframe(df,use_container_width=_A);fig=px.bar(df,x=_E,y=_Ct,title='Attendance Rate');st.plotly_chart(fig,use_container_width=_A)
	elif report_type==_Cp:results,columns=query_db(_C9);df=pd.DataFrame(results,columns=columns);st.dataframe(df,use_container_width=_A);fig=px.pie(df,names=_P,values='total',title=_CS);st.plotly_chart(fig,use_container_width=_A)
	elif report_type==_Cq:
		st.markdown('<div class="section-header">Financial Report</div>',unsafe_allow_html=_A);terms_results,_=query_db(_Cf);terms_options={f"{row[1]} ({row[2]} to {row[3]})":row[0]for row in terms_results}
		if terms_options:
			selected_term_label=st.selectbox(_Cu,list(terms_options.keys()));term_id=terms_options[selected_term_label];term_info=[t for t in terms_results if t[0]==term_id][0];term_name=term_info[1];start_date=term_info[2];end_date=term_info[3];st.markdown(f"### Term: {term_name} ({start_date} to {end_date})");revenue_results,_=query_db("\n                SELECT SUM(amount) FROM fees \n                WHERE LOWER(status) = 'paid' \n                AND payment_date >= ? \n                AND payment_date <= ?\n            ",(start_date,end_date));total_revenue=revenue_results[0][0]or 0;expenditure_query='\n                SELECT category, SUM(amount) as total_expenditure \n                FROM expenditures \n                WHERE date BETWEEN ? AND ?\n                GROUP BY category\n            ';expenditure_results,expenditure_columns=query_db(expenditure_query,(start_date,end_date));total_expenditure=sum(row[1]for row in expenditure_results)if expenditure_results else 0;profit=total_revenue-total_expenditure;col1,col2,col3=st.columns(3)
			with col1:st.markdown(f'<div class="financial-card"><div class="financial-label">Total Revenue</div><div class="financial-metric">₡ {total_revenue:,.2f}</div></div>',unsafe_allow_html=_A)
			with col2:st.markdown(f'<div class="financial-card"><div class="financial-label">Total Expenditure</div><div class="financial-metric">₡ {total_expenditure:,.2f}</div></div>',unsafe_allow_html=_A)
			with col3:profit_class='positive-value'if profit>=0 else'negative-value';st.markdown(f'<div class="financial-card"><div class="financial-label">Profit</div><div class="financial-metric {profit_class}">₡ {profit:,.2f}</div></div>',unsafe_allow_html=_A)
			if expenditure_results:st.markdown('### Expenditure Breakdown');exp_df=pd.DataFrame(expenditure_results,columns=expenditure_columns);exp_df.rename(columns={'category':_Bu,'total_expenditure':_Ah},inplace=_A);st.dataframe(exp_df,use_container_width=_A);fig=px.pie(exp_df,names=_Bu,values=_Ah,title=_Cl);st.plotly_chart(fig,use_container_width=_A)
			else:st.info('No expenditures recorded for this term')
			if st.button('📥 Download Financial Report',key='download_financial_report'):
				report_data={_Bj:[term_name],_Bk:[start_date],_Bl:[end_date],'Total Revenue':[total_revenue],'Total Expenditure':[total_expenditure],'Profit':[profit]}
				if expenditure_results:
					for(category,amount)in expenditure_results:report_data[f"Expenditure: {category}"]=[amount]
				report_df=pd.DataFrame(report_data);csv=report_df.to_csv(index=_B).encode(_M);st.download_button('📥 Download CSV Report',data=csv,file_name=f"financial_report_{term_name.replace(' ','_')}_{start_date}_to_{end_date}.csv",mime=_AN)
		else:st.info('No terms defined. Please add terms in the Financials section.')
elif st.session_state.current_section==_BV:
	st.markdown('<div class="section-header">End of Term Report Automation</div>',unsafe_allow_html=_A)
	if _Ay not in st.session_state:st.session_state.gmail_connector=GmailConnector();st.session_state.gmail_connector.load_settings()
	report_scope=st.radio('Report Scope',[_Bw,_Bx,'All'],horizontal=_A);results,_=query_db('SELECT id, name FROM terms ORDER BY end_date DESC');term_options={row[1]:row[0]for row in results}if results else{}
	if not term_options:st.warning(_CY);st.stop()
	selected_term_label=st.selectbox(_Cu,list(term_options.keys()));term_id=term_options[selected_term_label];students_to_process=[]
	if report_scope==_Bw:student_results,_=query_db(_h);student_options={row[1]:row[0]for row in student_results};selected_student=st.selectbox(_Bv,list(student_options.keys()));students_to_process=[(student_options[selected_student],selected_student)]
	elif report_scope==_Bx:
		try:
			class_results,_=query_db("SELECT DISTINCT grade FROM students WHERE grade IS NOT NULL AND grade != '' ORDER BY grade")
			if class_results and len(class_results)>0:
				class_options=[]
				for row in class_results:
					if isinstance(row,(list,tuple))and len(row)>0:class_options.append(str(row[0]).strip())
					else:class_options.append(str(row).strip())
				class_options=list(set([opt for opt in class_options if opt]));class_options.sort()
				if class_options:
					selected_class=st.selectbox('Select Class',class_options);class_students_results,_=query_db('SELECT id, name FROM students WHERE grade = ?',(selected_class,))
					if class_students_results:students_to_process=[(row[0],row[1])for row in class_students_results]
					else:st.warning(f"No students found in class {selected_class}");st.stop()
				else:st.warning('No valid classes/grades found after processing.');st.stop()
			else:st.warning('No classes/grades found in the students table.');st.stop()
		except Exception as e:st.error(f"Error querying classes: {str(e)}");st.stop()
	elif report_scope=='All':all_students_results,_=query_db(_h);students_to_process=[(row[0],row[1])for row in all_students_results]
	def normalize_date(date_str):
		'Convert various date formats to YYYY-MM-DD for comparison'
		if not date_str:return
		try:
			date_formats=[_AD,'%d/%m/%Y',_BN,'%d-%m-%Y','%Y/%m/%d','%d.%m.%Y']
			for fmt in date_formats:
				try:parsed_date=datetime.strptime(str(date_str).strip(),fmt);return parsed_date.strftime(_AD)
				except ValueError:continue
			return
		except Exception:return
	if st.button('Generate Report',key='generate_report_button'):
		if not students_to_process:st.error('No students selected for report generation');st.stop()
		with st.spinner('🔄 Initializing report generation...'):
			progress_bar=st.progress(0);status_container=st.empty();total_students=len(students_to_process);generated_reports={};temporary_files=[]
			try:
				for(i,(student_id,student_name))in enumerate(students_to_process):
					progress=(i+1)/total_students;progress_bar.progress(progress);status_container.markdown(f"**Generating report {i+1}/{total_students}: {student_name}**")
					with st.spinner(f"📝 Processing {student_name}'s report..."):
						graph_path=_D
						try:
							student_data=query_db('SELECT * FROM students WHERE id = ?',(student_id,))[0][0];name=student_data[1];grade=student_data[2];dob=student_data[4];parent_name=student_data[9];teacher_name=get_teacher_for_student(grade);term_info=query_db('SELECT start_date, end_date FROM terms WHERE id = ?',(term_id,))[0][0];term_start=term_info[0];term_end=term_info[1];subjects_query='\n                                SELECT DISTINCT subject \n                                FROM test_results \n                                WHERE student_id = ?\n                            ';all_subjects_results=query_db(subjects_query,(student_id,))[0];subjects=[]
							if all_subjects_results:
								for subject_row in all_subjects_results:
									subject=subject_row[0];subject_test_query='\n                                        SELECT test_date FROM test_results \n                                        WHERE student_id = ? AND subject = ?\n                                    ';subject_dates=query_db(subject_test_query,(student_id,subject))[0]
									for date_row in subject_dates:
										test_date=normalize_date(date_row[0]);normalized_start=normalize_date(term_start);normalized_end=normalize_date(term_end)
										if test_date and normalized_start and normalized_end:
											if normalized_start<=test_date<=normalized_end:subjects.append(subject);break
							table_data=[]
							for subject in subjects:
								student_scores=[];student_scores_query='\n                                    SELECT score, test_date\n                                    FROM test_results\n                                    WHERE student_id = ? AND subject = ?\n                                ';score_results=query_db(student_scores_query,(student_id,subject))[0]
								for score_row in score_results:
									score,test_date=score_row[0],score_row[1];normalized_test_date=normalize_date(test_date);normalized_start=normalize_date(term_start);normalized_end=normalize_date(term_end)
									if normalized_test_date and normalized_start and normalized_end:
										if normalized_start<=normalized_test_date<=normalized_end:student_scores.append(score)
								student_avg=sum(student_scores)/len(student_scores)if student_scores else 0;class_scores=[];class_scores_query='\n                                    SELECT score, test_date\n                                    FROM test_results\n                                    WHERE subject = ?\n                                ';class_score_results=query_db(class_scores_query,(subject,))[0]
								for score_row in class_score_results:
									score,test_date=score_row[0],score_row[1];normalized_test_date=normalize_date(test_date);normalized_start=normalize_date(term_start);normalized_end=normalize_date(term_end)
									if normalized_test_date and normalized_start and normalized_end:
										if normalized_start<=normalized_test_date<=normalized_end:class_scores.append(score)
								class_avg=sum(class_scores)/len(class_scores)if class_scores else 0;all_student_averages={};all_students_query='SELECT DISTINCT student_id FROM test_results WHERE subject = ?';all_students_results=query_db(all_students_query,(subject,))[0]
								for student_row in all_students_results:
									other_student_id=student_row[0];other_scores=[];other_scores_query='\n                                        SELECT score, test_date\n                                        FROM test_results\n                                        WHERE student_id = ? AND subject = ?\n                                    ';other_score_results=query_db(other_scores_query,(other_student_id,subject))[0]
									for score_row in other_score_results:
										score,test_date=score_row[0],score_row[1];normalized_test_date=normalize_date(test_date);normalized_start=normalize_date(term_start);normalized_end=normalize_date(term_end)
										if normalized_test_date and normalized_start and normalized_end:
											if normalized_start<=normalized_test_date<=normalized_end:other_scores.append(score)
									if other_scores:all_student_averages[other_student_id]=sum(other_scores)/len(other_scores)
								position=1
								for other_avg in all_student_averages.values():
									if other_avg>student_avg:position+=1
								if student_avg>=90:remarks='Excellent'
								elif student_avg>=80:remarks='Very Good'
								elif student_avg>=70:remarks='Good'
								elif student_avg>=60:remarks='Satisfactory'
								else:remarks='Needs Improvement'
								table_data.append([subject,f"{student_avg:.1f}",f"{class_avg:.1f}",position,remarks])
							all_student_overall_averages={};all_students_query='SELECT DISTINCT student_id FROM test_results';all_students_results=query_db(all_students_query)[0]
							for student_row in all_students_results:
								other_student_id=student_row[0];other_scores=[];other_scores_query='\n                                    SELECT score, test_date\n                                    FROM test_results\n                                    WHERE student_id = ?\n                                ';other_score_results=query_db(other_scores_query,(other_student_id,))[0]
								for score_row in other_score_results:
									score,test_date=score_row[0],score_row[1];normalized_test_date=normalize_date(test_date);normalized_start=normalize_date(term_start);normalized_end=normalize_date(term_end)
									if normalized_test_date and normalized_start and normalized_end:
										if normalized_start<=normalized_test_date<=normalized_end:other_scores.append(score)
								if other_scores:all_student_overall_averages[other_student_id]=sum(other_scores)/len(other_scores)
							current_student_overall_avg=all_student_overall_averages.get(student_id,0);overall_position=1
							for other_avg in all_student_overall_averages.values():
								if other_avg>current_student_overall_avg:overall_position+=1
							academic_results=[];academic_query='\n                                SELECT subject, score, test_date \n                                FROM test_results \n                                WHERE student_id = ?\n                                ORDER BY test_date\n                            ';all_academic_results=query_db(academic_query,(student_id,))[0]
							for result_row in all_academic_results:
								subject,score,test_date=result_row[0],result_row[1],result_row[2];normalized_test_date=normalize_date(test_date);normalized_start=normalize_date(term_start);normalized_end=normalize_date(term_end)
								if normalized_test_date and normalized_start and normalized_end:
									if normalized_start<=normalized_test_date<=normalized_end:academic_results.append((subject,score,test_date))
							if academic_results and len(academic_results)>0:
								try:
									history_df=pd.DataFrame(academic_results,columns=[_d,_i,_Ag]);fig=px.line(history_df,x=_Ag,y=_i,color=_d,title=f"Performance Over Time - {student_name}",markers=_A,color_discrete_sequence=px.colors.qualitative.Plotly);fig.update_layout(paper_bgcolor='white',plot_bgcolor='white')
									with tempfile.NamedTemporaryFile(delete=_B,suffix='.png')as tmpfile:fig.write_image(tmpfile.name,format='png',width=800,height=600,engine='kaleido',scale=2);graph_path=tmpfile.name;temporary_files.append(graph_path)
								except Exception as e:st.warning(f"Graph generation failed for {student_name}: {str(e)}");graph_path=_D
							principal_name=get_principal_name();prompt=f"""
Generate a brief end-of-term report for student: {name}
Term: {selected_term_label}

Student Information:
- Grade: {grade}
- Class Teacher: {teacher_name}
- Date of Birth: {dob}
- Parent/Guardian: {parent_name}

Academic Performance Summary:
{format_as_table(table_data,[_AS,"Student Score",_CD,"Position","Remarks"])if table_data else"No performance data"}

Performance Trends:
{"Performance graphs included for each subject showing trends over time"if graph_path else"No performance graphs available"}

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
							with st.spinner(f"🤖 Generating AI report content for {student_name}..."):
								report_content=''
								if st.session_state.get(_AY)==_Ai and OLLAMA_MODEL:
									try:
										api_payload={_z:OLLAMA_MODEL,_l:[{_F:_b,_C:_Cv},{_F:_L,_C:prompt}],_A0:_A};response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_j:_k},stream=_A,timeout=300);response.raise_for_status();report_content=''
										for line in response.iter_lines():
											if line:
												data=json.loads(line.decode(_M))
												if _Q in data and _C in data[_Q]:report_content+=data[_Q][_C]
									except Exception as e:report_content=f"Local AI Error: {str(e)}"
								elif st.session_state.get(_AY)==_Aj and OPENROUTER_MODEL and openrouter_api_key:
									try:
										messages=[{_F:_b,_C:_Cv},{_F:_L,_C:prompt}];response=call_openrouter_api(messages,OPENROUTER_MODEL,openrouter_api_key)
										if response:
											report_content=''
											for chunk in response.iter_lines():
												if chunk and chunk!=b'':
													if chunk.startswith(_Ak):
														try:
															data=json.loads(chunk.decode(_M)[5:])
															if _S in data and len(data[_S])>0:delta=data[_S][0].get(_Al,{});report_content+=delta.get(_C,'')
														except json.JSONDecodeError:pass
										else:report_content='Report generation error: API call failed'
									except Exception as e:report_content=f"OpenRouter Error: {str(e)}"
								else:report_content='AI disabled or not configured'
							with st.spinner(f"📄 Creating PDF report for {student_name}..."):student_info_dict={_E:student_data[1],_T:student_data[2],_CC:overall_position};pdf_buffer=generate_report_pdf('End of Term Report',student_name,selected_term_label,student_info_dict,table_data,report_content,graph_path);generated_reports[student_id]={_E:student_name,'pdf':pdf_buffer,_C:report_content}
						except Exception as e:st.error(f"Report generation failed for {student_name}: {str(e)}");continue
						finally:
							if graph_path and os.path.exists(graph_path):
								try:
									os.unlink(graph_path)
									if graph_path in temporary_files:temporary_files.remove(graph_path)
								except Exception as cleanup_error:st.warning(f"Could not clean up temporary file for {student_name}: {cleanup_error}")
			except Exception as e:st.error(f"Report generation process failed: {str(e)}")
			finally:
				for temp_file in temporary_files:
					if os.path.exists(temp_file):
						try:os.unlink(temp_file)
						except Exception as cleanup_error:st.warning(f"Could not clean up temporary file {temp_file}: {cleanup_error}")
				progress_bar.empty();status_container.empty()
			st.session_state.generated_reports=generated_reports;st.session_state.current_term=term_id;st.session_state.current_term_name=selected_term_label
			if generated_reports:st.success(f"✅ Successfully generated {len(generated_reports)} reports!")
			else:st.warning('No reports were generated successfully.')
	if'generated_reports'in st.session_state and st.session_state.generated_reports:
		st.markdown(_c);st.subheader('Generated Reports')
		if report_scope==_Bw:
			for(student_id,report)in st.session_state.generated_reports.items():
				st.markdown(f"### {report[_E]}");st.download_button('📥 Download PDF Report',data=report['pdf'].getvalue(),file_name=f"{report[_E]}_Term_Report_{st.session_state.current_term_name}.pdf",mime=_Cw)
				with st.expander('View Report Content'):st.write(report[_C])
		st.markdown(_c);st.subheader('Email Automation')
		with st.expander('Configure Email Settings',expanded=_A):email_subject=st.text_input('Email Subject',value=f"End of Term Report - {st.session_state.current_term_name}",key='email_subject');email_body=st.text_area('Email Body',height=150,value='Dear Parent/Guardian,\n\nPlease find attached the end of term report for your child.\n\nBest regards,\nSchool Administration',key='email_body');sender_name=st.text_input('Sender Name',value=st.session_state.get(_Av,_AE),key='email_sender_name')
		if st.button('📧 Email Reports',key='email_reports_button',type=_X):
			if not st.session_state.gmail_connector.username or not st.session_state.gmail_connector.password:st.error('Gmail not configured. Please set up email in Ticket Generator section.');st.stop()
			with st.spinner('📧 Connecting to Gmail and sending reports...'):
				success,message=st.session_state.gmail_connector.connect(st.session_state.gmail_connector.username,st.session_state.gmail_connector.password)
				if not success:st.error(f"Gmail connection failed: {message}");st.stop()
				progress_bar=st.progress(0);status_container=st.empty();results=[]
				for(i,(student_id,report))in enumerate(st.session_state.generated_reports.items()):
					progress=(i+1)/len(st.session_state.generated_reports);progress_bar.progress(progress);status_container.markdown(f"**Sending {i+1}/{len(st.session_state.generated_reports)}: {report[_E]}**")
					try:
						student_email=query_db('SELECT email FROM students WHERE id = ?',(student_id,))[0][0][0]
						if not EMAIL_PATTERN.match(student_email):results.append({_By:report[_E],_P:'Skipped',_Q:f"Invalid email: {student_email}"});continue
						personalized_subject=email_subject.replace(_Cx,report[_E]);personalized_body=email_body.replace(_Cx,report[_E]);success,message=st.session_state.gmail_connector.send_email(student_email,personalized_subject,personalized_body,sender_name,report['pdf'].getvalue(),f"{report[_E]}_Report_{st.session_state.current_term_name}.pdf");results.append({_By:report[_E],_P:'Sent'if success else'Failed',_Q:message})
					except Exception as e:results.append({_By:report[_E],_P:'Error',_Q:str(e)})
				progress_bar.empty();status_container.empty();st.success(f"Processed {len(st.session_state.generated_reports)} students");results_df=pd.DataFrame(results);st.dataframe(results_df);csv=results_df.to_csv(index=_B).encode(_M);st.download_button('Download Results Summary',data=csv,file_name=f"report_results_{st.session_state.current_term_name}.csv",mime=_AN,use_container_width=_A)
elif st.session_state.current_section==_BW:
	st.markdown(_Cy,unsafe_allow_html=_A)
	with st.sidebar:
		st.subheader('Chat Controls');col1,col2=st.columns(2)
		with col1:st.session_state.enable_reasoning=st.checkbox('Reasoning',value=st.session_state.get(_Bz,_B),help='Enable reasoning for qwen3* models')
		with col2:clear_chat=st.button('Clear',use_container_width=_A)
	if clear_chat:st.session_state.messages=[];st.rerun()
	for(msg_index,msg)in enumerate(st.session_state.messages):
		with st.chat_message(msg[_F]):
			parts=split_message(msg[_C])
			for(code_index,part)in enumerate(parts):
				if part[_AO]=='text':st.markdown(part[_C],unsafe_allow_html=_A)
				elif part[_AO]==_Am:st.code(part[_Am],language=part[_BB]);ext=part[_BB]if part[_BB]!='text'else _A9;filename=f"code_{msg_index}_{code_index}.{ext}";st.download_button(label='💾 Download',data=part[_Am],file_name=filename,mime=_B_,key=f"download_code_{msg_index}_{code_index}",use_container_width=_B)
	prompt=st.chat_input('Ask the AI Assistant:',key='ai_assistant_chat_input')
	if prompt:
		db_context=get_db_context();system_prompt=f"""
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
		if st.session_state.get(_AY)==_Ai and OLLAMA_MODEL:
			with st.chat_message(_L):st.markdown(prompt,unsafe_allow_html=_A)
			user_query=f"School Administration Query: {prompt}"
			if not st.session_state.get(_Bz,_B)and'qwen3'in OLLAMA_MODEL.lower():user_query+=' /no_think'
			messages_for_api=[{_F:_b,_C:system_prompt}]
			for msg in st.session_state.messages:
				if msg[_C].strip():messages_for_api.append({_F:msg[_F],_C:msg[_C]})
			messages_for_api.append({_F:_L,_C:user_query});api_payload={_z:OLLAMA_MODEL,_l:messages_for_api,_A0:_A}
			with st.spinner(_C0):
				response_placeholder=st.empty();accumulated_response='';reasoning_buffer='';in_thinking_block=_B;sql_query=_D
				try:
					response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_j:_k},stream=_A,timeout=600);response.raise_for_status()
					for line in response.iter_lines():
						if line:
							data=json.loads(line.decode(_M))
							if _Q in data and _C in data[_Q]:
								content_chunk=data[_Q][_C]
								if st.session_state.get(_Bz,_B):
									reasoning_buffer+=content_chunk;processed_content=''
									while _C1 in reasoning_buffer or _C2 in reasoning_buffer:
										if not in_thinking_block and _C1 in reasoning_buffer:
											start_idx=reasoning_buffer.find(_C1)
											if start_idx>0:processed_content+=reasoning_buffer[:start_idx]
											reasoning_buffer=reasoning_buffer[start_idx+7:];in_thinking_block=_A;show_reasoning_window()
										elif in_thinking_block and _C2 in reasoning_buffer:end_idx=reasoning_buffer.find(_C2);think_content=reasoning_buffer[:end_idx].strip();reasoning_buffer=reasoning_buffer[end_idx+8:];in_thinking_block=_B;update_reasoning_window(think_content);hide_reasoning_window()
										else:break
									if in_thinking_block:update_reasoning_window(reasoning_buffer.strip())
									else:processed_content+=reasoning_buffer;reasoning_buffer=''
									if processed_content:accumulated_response+=processed_content;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
								else:accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
					if'SELECT'in accumulated_response.upper():
						match=re.search(_Cz,accumulated_response,re.IGNORECASE|re.DOTALL)
						if match:
							sql_query=match.group(0);results,columns,error=execute_sql(sql_query)
							if error:accumulated_response+=f"\n\n❌ Query Error: {error}";response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
							elif results:
								df=pd.DataFrame(results,columns=columns)
								if not df.empty:accumulated_response+=f"\n\n🔍 Query returned {len(df)} rows:\n";accumulated_response+=df.to_markdown(index=_B);response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A);csv=df.to_csv(index=_B).encode(_M);st.download_button(label=_C_,data=csv,file_name=_D0,mime=_AN,key=f"download_results_{int(time.time())}")
								else:accumulated_response+=_D1;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
							else:accumulated_response+=_D2;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
					st.session_state.messages.append({_F:_L,_C:user_query,_AP:datetime.now().isoformat()});st.session_state.messages.append({_F:_Y,_C:accumulated_response})
				except requests.exceptions.Timeout:error_msg='AI server timed out after 600 seconds.';st.error(error_msg);st.session_state.messages.append({_F:_L,_C:user_query,_AP:datetime.now().isoformat()});st.session_state.messages.append({_F:_Y,_C:error_msg})
				except Exception as e:error_msg=f"Error: {str(e)}";st.error(error_msg);st.session_state.messages.append({_F:_L,_C:user_query,_AP:datetime.now().isoformat()});st.session_state.messages.append({_F:_Y,_C:error_msg})
				finally:hide_reasoning_window()
		elif st.session_state.get(_AY)==_Aj and OPENROUTER_MODEL and openrouter_api_key:
			with st.chat_message(_L):st.markdown(prompt,unsafe_allow_html=_A)
			user_query=f"School Administration Query: {prompt}";messages=[{_F:_b,_C:system_prompt}]
			for msg in st.session_state.messages:
				if msg[_C].strip():messages.append({_F:msg[_F],_C:msg[_C]})
			messages.append({_F:_L,_C:user_query})
			with st.spinner(_C0):
				response_placeholder=st.empty();accumulated_response='';response=call_openrouter_api(messages,OPENROUTER_MODEL,openrouter_api_key)
				if response:
					try:
						for chunk in response.iter_lines():
							if chunk and chunk!=b'':
								if chunk.startswith(_Ak):
									try:
										data=json.loads(chunk.decode(_M)[5:])
										if _S in data and len(data[_S])>0:delta=data[_S][0].get(_Al,{});content_chunk=delta.get(_C,'');accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
									except json.JSONDecodeError:pass
						if'SELECT'in accumulated_response.upper():
							match=re.search(_Cz,accumulated_response,re.IGNORECASE|re.DOTALL)
							if match:
								sql_query=match.group(0);results,columns,error=execute_sql(sql_query)
								if error:accumulated_response+=f"\n\n❌ Query Error: {error}";response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
								elif results:
									df=pd.DataFrame(results,columns=columns)
									if not df.empty:accumulated_response+=f"\n\n🔍 Query returned {len(df)} rows:\n";accumulated_response+=df.to_markdown(index=_B);response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A);csv=df.to_csv(index=_B).encode(_M);st.download_button(label=_C_,data=csv,file_name=_D0,mime=_AN,key=f"download_results_{int(time.time())}")
									else:accumulated_response+=_D1;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
								else:accumulated_response+=_D2;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
						st.session_state.messages.append({_F:_L,_C:user_query,_AP:datetime.now().isoformat()});st.session_state.messages.append({_F:_Y,_C:accumulated_response})
					except Exception as e:error_msg=f"Error processing response: {str(e)}";st.error(error_msg);st.session_state.messages.append({_F:_L,_C:user_query,_AP:datetime.now().isoformat()});st.session_state.messages.append({_F:_Y,_C:error_msg})
				else:error_msg=_C3;st.error(error_msg);st.session_state.messages.append({_F:_L,_C:user_query,_AP:datetime.now().isoformat()})
elif st.session_state.current_section==_BX:
	st.markdown(f'<div class="main-title">Ticket Generator 🍽️</div>',unsafe_allow_html=_A);st.markdown(f'<div style="font-size: 14px; color: #333;">Version {APP_VERSION}</div>',unsafe_allow_html=_A);generator=TicketGenerator();ollama_models=get_ollama_models()
	if _AG in ollama_models:generator.ollama_model=_AG
	elif ollama_models:generator.ollama_model=ollama_models[0]
	with st.sidebar:
		st.markdown('<div class="sidebar-title">Configuration</div>',unsafe_allow_html=_A);data_source=_D3
		with st.expander('📧 Gmail Connector',expanded=_B):
			gmail_enabled=st.checkbox('Enable Gmail Connector',key='gmail_enabled')
			if gmail_enabled:
				credentials_loaded=_B
				try:
					if generator.gmail.load_settings():
						credentials_loaded=_A;st.success(f"✅ Gmail credentials loaded for: {generator.gmail.username}")
						if not generator.gmail.connected:
							success,message=generator.gmail.connect(generator.gmail.username,generator.gmail.password)
							if success:st.session_state[_U]=_A;generator.gmail.connected=_A
							else:st.error(f"Connection failed: {message}")
				except Exception as e:
					try:
						gmail_username=st.secrets.get(_Aw,{}).get(_At,'');gmail_password=st.secrets.get(_Aw,{}).get(_A2,'');sender_name=st.secrets.get(_Aw,{}).get(_Au,_AE)
						if gmail_username and gmail_password:
							generator.gmail.username=gmail_username;generator.gmail.password=gmail_password;st.session_state.gmail_sender_name_input=sender_name;credentials_loaded=_A;st.success(f"✅ Gmail credentials loaded from secrets for: {gmail_username}")
							if not generator.gmail.connected:
								success,message=generator.gmail.connect(gmail_username,gmail_password)
								if success:st.session_state[_U]=_A;generator.gmail.connected=_A
								else:st.error(f"Connection failed: {message}")
					except Exception:pass
				if not credentials_loaded:
					st.warning('⚠️ No saved Gmail credentials found. Please configure manually.');st.markdown('[📖 How to create Gmail App Password](https://support.google.com/accounts/answer/185833)');gmail_username=st.text_input('Gmail Address',placeholder='your-email@gmail.com',key='gmail_username',help='Your full Gmail address. Use an App Password for authentication.');gmail_password=st.text_input('App Password',type=_A2,key='gmail_password',help='16-character app password from Google (not your regular password)');sender_display_name=st.text_input('Display Name',value=_AE,key=_Av,help='Name that appears as sender');col_connect,col_save=st.columns(2)
					with col_connect:
						if st.button('🔗 Connect Gmail',use_container_width=_A,key='connect_gmail'):
							if gmail_username and gmail_password:
								with st.spinner('Connecting to Gmail...'):
									success,message=generator.gmail.connect(gmail_username,gmail_password)
									if success:st.success(message);st.session_state[_U]=_A;generator.gmail.connected=_A
									else:st.error(message);st.session_state[_U]=_B;generator.gmail.connected=_B
							else:st.error('Please enter both email and app password')
					with col_save:
						if st.button('💾 Save Settings',use_container_width=_A,key='save_gmail'):
							if gmail_username and gmail_password:
								generator.gmail.username=gmail_username;generator.gmail.password=gmail_password
								if generator.gmail.save_settings():st.success('Settings saved successfully!');st.rerun()
								else:st.error('Failed to save settings')
							else:st.error('Please enter credentials first')
				if st.session_state.get(_U,_B)and generator.gmail.connected:st.success(f"🟢 Connected as: {generator.gmail.username}")
				else:st.error('🔴 Not connected')
				if st.session_state.get(_U,_B)and generator.gmail.connected:
					st.subheader('📧 Send Test Email');test_email=st.text_input('Test Email Address',placeholder='test@example.com',key='test_email_input')
					if st.button('📤 Send Test Email',use_container_width=_A):
						if test_email:
							sender_name=st.session_state.get(_Av,_AE);success,message=generator.gmail.send_email(test_email,'Test Email from Ticket Generator',f"""This is a test email sent from the Ticket Generator app.

Connection successful!

Best regards,
{sender_name}""",sender_name)
							if success:st.success(f"Test email sent to {test_email}")
							else:st.error(message)
						else:st.error('Please enter a test email address')
		with st.expander('🎟 Ticket Settings',expanded=_B):
			school_name=st.text_input('School Name',value='RadioSport');ticket_date=st.date_input('Issue Date',value=date.today());validity_type=st.selectbox('Validity Type',[_D4,_D5,'Weekly'])
			if validity_type==_D4:valid_date=st.date_input('Valid Date',value=date.today());validity_info={_AV:f"Valid: {valid_date.strftime(_BN)}"}
			elif validity_type==_D5:start_date=st.date_input(_Bk,value=date.today());end_date=st.date_input(_Bl,value=date.today());validity_info={_AV:f"{start_date.strftime(_BC)} - {end_date.strftime(_BC)}"}
			else:week_start=st.date_input('Week Start',value=date.today());week_end=week_start+pd.Timedelta(days=6);validity_info={_AV:f"Week: {week_start.strftime(_BC)} - {week_end.strftime(_BC)}"}
		with st.expander('🤖 AI Settings',expanded=_B):
			ollama_available_ticket=bool(get_ollama_models());default_ticket_provider=0 if ollama_available_ticket else 1;ai_provider=st.radio(_CL,[_AH,_Ba],index=default_ticket_provider,key=_D6,help='Local: Ollama (if available) | Cloud: OpenRouter')
			if ai_provider==_AH:
				if ollama_models:default_index=ollama_models.index(_AG)if _AG in ollama_models else 0;generator.ollama_model=st.selectbox(_B2,ollama_models,index=default_index,key='ticket_ollama_model_select');st.markdown('<div class="ai-status ai-connected">🟢 AI Connected</div>',unsafe_allow_html=_A)
				else:st.markdown('<div class="ai-status ai-disconnected">🔴 AI Disconnected</div>',unsafe_allow_html=_A)
			else:
				auto_loaded_key=''
				try:
					auto_loaded_key=st.secrets.get(_Bb,{}).get(_Bc,'')
					if auto_loaded_key:st.success('✅ OpenRouter API key loaded from secrets')
				except Exception:pass
				openrouter_api_key=st.text_input(_CO,type=_A2,value=auto_loaded_key,key=_D7,help='Get your key from https://openrouter.ai/keys')
				if openrouter_api_key:OPENROUTER_MODEL=st.selectbox(_B2,OPENROUTER_MODELS,index=0,key=_D8);st.markdown('<div class="ai-status ai-connected">🟢 OpenRouter Connected</div>',unsafe_allow_html=_A)
				else:st.markdown('<div class="ai-status ai-disconnected">🔴 Enter API Key</div>',unsafe_allow_html=_A)
			if st.button('🧹 Clear Chat',key='clear_chat_button',help='Clear all chat messages',use_container_width=_A):st.session_state.ticket_messages=[];st.session_state.chat_cleared=_A;st.rerun()
		with st.expander(_CQ):
			try:stats=st.session_state.cache_stats;total_db_requests=stats[_V]+stats[_Z];total_model_requests=stats[_W]+stats[_e];db_hit_rate=stats[_V]/max(total_db_requests,1)*100;model_hit_rate=stats[_W]/max(total_model_requests,1)*100;uptime=datetime.now()-stats[_A1];uptime_str=f"{uptime.days}d {uptime.seconds//3600}h {uptime.seconds%3600//60}m";stats_html=f'''
                <div class="cache-stats">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Uptime</td><td>{uptime_str}</td></tr>
                        <tr><td>DB Cache Hit Rate</td><td>{db_hit_rate:.1f}%</td></tr>
                        <tr><td>DB Cache Hits</td><td>{stats[_V]}</td></tr>
                        <tr><td>DB Cache Misses</td><td>{stats[_Z]}</td></tr>
                        <tr><td>Model Cache Hit Rate</td><td>{model_hit_rate:.1f}%</td></tr>
                        <tr><td>Model Cache Hits</td><td>{stats[_W]}</td></tr>
                        <tr><td>Model Cache Misses</td><td>{stats[_e]}</td></tr>
                    </table>
                </div>
                ''';st.markdown(stats_html,unsafe_allow_html=_A)
			except KeyError as e:st.error(f"Cache stats error: {e}. Resetting stats.");st.session_state.cache_stats={_V:0,_Z:0,_W:0,_e:0,_A1:datetime.now()};st.rerun()
			if st.button(_CR,key='ticket_reset_stats_button'):st.session_state.cache_stats={_V:0,_Z:0,_W:0,_e:0,_A1:datetime.now()};query_db.cache_clear();st.rerun()
	tab1,tab2,tab3=st.tabs(['📊 Data Management','🤖 AI Assistant','🎫 Generate Tickets'])
	with tab1:
		st.markdown(_D9,unsafe_allow_html=_A);col1,col2=st.columns([2,1])
		with col1:
			if data_source==_D3:
				st.markdown('<div class="section-header">Lunch Payment Records</div>',unsafe_allow_html=_A);lunch_query='\n                    SELECT \n                        lp.id,\n                        s.name AS student_name,\n                        s.grade AS class,\n                        s.phone,  -- ADDED\n                        s.email,  -- ADDED\n                        s.parent_name,  -- ADDED\n                        lp.amount,\n                        lp.payment_date,\n                        lp.status,\n                        t.name AS term\n                    FROM lunch_payments lp\n                    JOIN students s ON lp.student_id = s.id\n                    LEFT JOIN terms t ON lp.term_id = t.id\n                ';results,columns=query_db(lunch_query)
				if results:
					df=pd.DataFrame(results,columns=columns);df[_H]=_B;all_students=query_db(_h)[0];student_options={row[1]:row[0]for row in all_students}
					with st.expander('➕ Add New Record',expanded=_B):
						search_term=st.text_input('Search Student',key='student_search');filtered_students=[name for name in student_options.keys()if search_term.lower()in name.lower()]if search_term else list(student_options.keys());selected_student=st.selectbox(_Bv,filtered_students,key='new_student_select');student_details=query_db('SELECT grade FROM students WHERE id = ?',(student_options[selected_student],))[0][0];st.text_input(_Bx,value=student_details[0],disabled=_A,key='auto_class');amount=st.number_input(_Ah,min_value=.0,step=1e2,key='new_amount');status=st.selectbox(_AC,[_n,'unpaid'],key='new_status');payment_date=st.date_input(_Cj,value=date.today(),key='new_payment_date')
						if st.button('Add Record',key='add_lunch_record'):
							conn=sqlite3.connect(DB_NAME);c=conn.cursor();student_id=student_options[selected_student];results,columns=query_db('SELECT id FROM lunch_payments WHERE student_id = ?',(student_id,))
							if results:record_id=results[0][0];c.execute('\n                                    UPDATE lunch_payments \n                                    SET amount = ?, payment_date = ?, status = ?\n                                    WHERE id = ?\n                                ',(amount,payment_date,status,record_id));action='updated'
							else:c.execute('\n                                    INSERT INTO lunch_payments (student_id, amount, payment_date, status)\n                                    VALUES (?, ?, ?, ?)\n                                ',(student_id,amount,payment_date,status));action='added'
							conn.commit();conn.close();st.success(f"Record {action} successfully!");st.rerun()
					edited_df=st.data_editor(df,use_container_width=_A,num_rows=_g,column_config={_H:st.column_config.CheckboxColumn(_p,default=_B),_P:st.column_config.SelectboxColumn(_AC,options=[_n,'unpaid'])});st.session_state.students=df[[_B8,_J,_R,_K,_O]].rename(columns={_B8:_E,_J:_J});st.session_state.payments=df[[_P]].rename(columns={_P:_f});st.session_state.students[_N]=st.session_state.students.index;st.session_state.payments[_N]=st.session_state.students.index;st.session_state.paid_students=st.session_state.students[st.session_state.payments[_f]==_n];st.session_state.students[_R]=st.session_state.students[_R].fillna(_A5);st.session_state.students[_K]=st.session_state.students[_K].fillna(_A4);st.session_state.students[_O]=st.session_state.students[_O].fillna(_A6)
					if st.button(_q,key='save_lunch_changes'):
						conn=sqlite3.connect(DB_NAME);c=conn.cursor();id_to_original={row[_G]:row for(_,row)in df.iterrows()}
						for(_,edited_row)in edited_df.iterrows():
							original_row=id_to_original.get(edited_row[_G])
							if original_row is _D:continue
							updates={}
							for col in df.columns:
								if col not in[_G,_H]and edited_row[col]!=original_row[col]:updates[col]=edited_row[col]
							if updates:set_clause=_I.join([f"{col} = ?"for col in updates.keys()]);values=list(updates.values())+[edited_row[_G]];c.execute(f"UPDATE lunch_payments SET {set_clause} WHERE id = ?",values)
						conn.commit();conn.close();st.success('Changes saved!');st.rerun()
					if st.button(_r,key='delete_lunch_records'):
						deleted_count=0;conn=sqlite3.connect(DB_NAME);c=conn.cursor()
						for(index,row)in edited_df.iterrows():
							if row[_H]:c.execute('DELETE FROM lunch_payments WHERE id = ?',(row[_G],));deleted_count+=1
						conn.commit();conn.close()
						if deleted_count>0:st.success(f"Deleted {deleted_count} records");st.rerun()
				else:st.info('No lunch payment records found')
		with col2:
			st.markdown('<div class="section-header">Summary</div>',unsafe_allow_html=_A)
			if _C4 in st.session_state and not st.session_state.paid_students.empty:
				paid=st.session_state.paid_students;total_students=len(st.session_state.get(_a,[]))
				if total_students>0:payment_rate=len(paid)/total_students*100;summary_data={'Metric':[_Cd,'Paid Students','Payment Rate'],'Value':[str(total_students),str(len(paid)),f"{payment_rate:.1f}%"]};summary_df=pd.DataFrame(summary_data);st.table(summary_df)
				if _J in paid.columns:
					classes=paid[_J].value_counts();st.metric('Classes',len(classes))
					with st.expander('Class Breakdown'):
						for(cls,count)in classes.items():st.write(f"**{cls}:** {count}")
			else:st.info('No student data loaded yet')
	with tab2:
		st.markdown(_Cy,unsafe_allow_html=_A);current_ai_provider=st.session_state.get(_D6,_Ai);openrouter_api_key=st.session_state.get(_D7,'');openrouter_model=st.session_state.get(_D8,'')
		if not ollama_models and current_ai_provider==_AH:
			if not get_ollama_models():st.info('💡 Switching to Cloud AI - Local AI unavailable')
		elif current_ai_provider==_Aj and not openrouter_api_key:st.warning('⚠️ OpenRouter API key required for cloud AI')
		col1,col2=st.columns(2)
		with col1:
			st.subheader('🔍 Data Analysis')
			if st.button('Analyze Data',key='analyze_data_button',type=_X):
				students=st.session_state.get(_a,pd.DataFrame());payments=st.session_state.get(_C5,pd.DataFrame())
				if not students.empty and not payments.empty:
					total=len(students);paid=len(payments[payments[_f]==_n]);unpaid_students=total-paid;payment_rate=paid/total*100 if total>0 else 0;class_info=''
					if _J in students.columns:class_counts=students[_J].value_counts();class_info=f"\n- Classes: {_I.join([f'{cls} ({cnt})'for(cls,cnt)in class_counts.head(5).items()])}"
					prompt=f"""
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
					if current_ai_provider==_Ai and generator.ollama_model:
						api_payload={_z:generator.ollama_model,_l:[{_F:_b,_C:_DA},{_F:_L,_C:prompt}],_A0:_A}
						with st.spinner('🤖 AI analyzing data...'):
							response_placeholder=st.empty();accumulated_response=''
							try:
								response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_j:_k},stream=_A,timeout=60);response.raise_for_status()
								for line in response.iter_lines():
									if line:
										data=json.loads(line.decode(_M))
										if _Q in data and _C in data[_Q]:accumulated_response+=data[_Q][_C];response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
								st.session_state.ticket_messages.append({_F:_Y,_C:accumulated_response})
							except requests.exceptions.Timeout:st.error(_C6)
							except Exception as e:st.error(f"Error: {str(e)}")
					elif current_ai_provider==_Aj and openrouter_api_key and openrouter_model:
						messages=[{_F:_b,_C:_DA},{_F:_L,_C:prompt}]
						with st.spinner('☁️ Cloud AI analyzing data...'):
							response_placeholder=st.empty();accumulated_response=''
							try:
								response=call_openrouter_api(messages,openrouter_model,openrouter_api_key)
								if response:
									for chunk in response.iter_lines():
										if chunk:
											if chunk==b'':continue
											if chunk.startswith(_Ak):
												try:
													data=json.loads(chunk.decode(_M)[5:])
													if _S in data and len(data[_S])>0:delta=data[_S][0].get(_Al,{});content_chunk=delta.get(_C,'');accumulated_response+=content_chunk;response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
												except json.JSONDecodeError:pass
									st.session_state.ticket_messages.append({_F:_Y,_C:accumulated_response})
								else:st.error(_C3)
							except Exception as e:st.error(f"Error: {str(e)}")
					else:st.info(_DB)
				else:st.info(_DC)
		with col2:
			st.subheader('📧 Payment Reminders')
			if st.button('✉️ Generate Reminders',type=_X,key='generate_reminders_button'):
				students=st.session_state.get(_a,pd.DataFrame());payments=st.session_state.get(_C5,pd.DataFrame())
				if not students.empty and not payments.empty:
					paid_ids=payments[payments[_f]==_n][_N];unpaid=students[~students[_N].isin(paid_ids)]
					if not unpaid.empty:
						with st.spinner('✉️ Generating reminders...'):
							sender_info=st.session_state.get('sender_info',{_E:'School Administrator',_A_:_CV,_B0:date.today().strftime(_DD),_Az:(date.today()+pd.Timedelta(days=7)).strftime(_DD)});st.session_state.current_reminders=[]
							for(_,student)in unpaid.iterrows():
								phone=str(student.get(_R,_A5)).strip();email=str(student.get(_K,_A4)).strip().lower();parent_name=str(student.get(_O,_A6)).strip();has_valid_phone=_B
								if phone!=_A5 and phone!=_AW and phone:
									clean_phone=re.sub('\\D','',phone)
									if clean_phone.isdigit()and len(clean_phone)>=7:has_valid_phone=_A
									else:phone=_BO
								has_valid_email=_B
								if email!=_A4 and email!=_AW and email:
									if EMAIL_PATTERN.match(email):has_valid_email=_A
									elif'@'in email and'.'in email.split('@')[-1]:has_valid_email=_A
								prompt=f"""
                            Write the body of a polite payment reminder email (without subject line) for:
                            Student: {student.get(_E,_o)}
                            Class: {student.get(_J,_y)}
                            Parent/Guardian: {parent_name}

                            Create a respectful message about their child's lunch payment.
                            Include that payment is due by {sender_info.get(_Az,"soon")}.
                            Keep it under 100 words, professional but friendly.
                            DO NOT INCLUDE A SUBJECT LINE.

                            End the message with:

                            Best regards,
                            {sender_info[_E]}
                            {sender_info[_A_]}
                            {sender_info[_B0]}

                            For questions, please contact the school office.
                            """;system_prompt=_CJ;reminder_text=f"""Dear {parent_name},

                            This is a reminder that your child {student.get(_E,_o)} from {student.get(_J,_y)} has an outstanding lunch payment.

                            Please submit the payment by {sender_info.get(_Az,"soon")} to ensure uninterrupted meal service.

                            Best regards,
                            {sender_info[_E]}
                            {sender_info[_A_]}
                            {sender_info[_B0]}

                            For questions, please contact the school office."""
								if current_ai_provider==_AH and generator.ollama_model:
									api_payload={_z:generator.ollama_model,_l:[{_F:_b,_C:system_prompt},{_F:_L,_C:prompt}],_A0:_B,'options':{_BM:.5,_CK:300}}
									try:
										response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_j:_k},timeout=60);response.raise_for_status();ai_response=response.json().get(_Q,{}).get(_C,'')
										if ai_response:reminder_text=sanitize_text(ai_response)
									except requests.exceptions.Timeout:pass
									except Exception:pass
								elif current_ai_provider==_Ba:
									if not openrouter_api_key:
										try:openrouter_api_key=st.secrets.get(_Bb,{}).get(_Bc,'')
										except Exception:pass
									if openrouter_api_key and openrouter_model:
										messages=[{_F:_b,_C:system_prompt},{_F:_L,_C:prompt}]
										try:
											response=call_openrouter_api(messages,openrouter_model,openrouter_api_key)
											if response:
												ai_response=''
												for chunk in response.iter_lines():
													if chunk:
														if chunk==b'':continue
														if chunk.startswith(_Ak):
															try:
																data=json.loads(chunk.decode(_M)[5:])
																if _S in data and len(data[_S])>0:delta=data[_S][0].get(_Al,{});ai_response+=delta.get(_C,'')
															except json.JSONDecodeError:pass
												if ai_response:reminder_text=ai_response
										except Exception:pass
								st.session_state.current_reminders.append({_E:student.get(_E),_J:student.get(_J),_O:parent_name,_R:phone,_K:email,_m:reminder_text,_AX:has_valid_phone,_A3:has_valid_email})
							st.session_state.email_status={_AT:{},_AU:_D};st.rerun()
					else:st.success('🎉 All students have paid!')
				else:st.info(_DC)
			if st.session_state.get(_As):
				reminders=st.session_state.current_reminders;st.markdown('### 📝 Payment Reminders');total_reminders=len(reminders);with_phone=sum(1 for r in reminders if r[_AX]);with_email=sum(1 for r in reminders if r[_A3]);col_stats1,col_stats2,col_stats3=st.columns(3);col_stats1.metric('Total Reminders',total_reminders);col_stats2.metric('📱 With Phone',with_phone);col_stats3.metric('📧 With Email',with_email)
				for(i,reminder)in enumerate(reminders):
					with st.expander(f"📧 {reminder[_E]} ({reminder[_J]})",expanded=_B):
						col_contact,col_actions=st.columns([2,1])
						with col_contact:
							st.write(f"**Parent:** {reminder[_O]}")
							if reminder[_AX]:st.write(f"📱 **Phone:** {reminder[_R]}")
							else:st.write('📱 Phone: Not available')
							if reminder[_A3]:st.write(f"📧 **Email:** {reminder[_K]}")
							else:st.write('📧 Email: Not available')
						with col_actions:
							if reminder[_AX]:
								if st.button(f"📱 SMS",key=f"sms_{i}",help='Send SMS reminder'):st.info(f"SMS feature coming soon for {reminder[_R]}")
							email_invalid=not EMAIL_PATTERN.match(reminder[_K])if reminder[_A3]else _A;email_disabled=not st.session_state.get(_U,_B)or email_invalid;email_help='Invalid email format'if email_invalid else'Send email reminder'
							if st.button(f"📧 Email",key=f"email_{i}",help=email_help,disabled=email_disabled):
								if not email_disabled:
									with st.spinner('Sending email...'):success,message=generator.gmail.send_email(reminder[_K],f"Lunch Payment Reminder - {reminder[_E]}",reminder[_m],st.session_state.gmail_sender_name_input);st.session_state.email_status[_AT][i]=success,message;st.rerun()
								else:st.error('Gmail not connected or email invalid')
							if st.button(f"📋 Copy",key=f"copy_{i}",help='Copy message to clipboard'):st.code(reminder[_m],language=_D)
						if i in st.session_state.email_status[_AT]:
							success,message=st.session_state.email_status[_AT][i]
							if success:st.markdown(f'<div class="email-status email-success">{message}</div>',unsafe_allow_html=_A)
							else:st.markdown(f'<div class="email-status email-error">{message}</div>',unsafe_allow_html=_A)
						st.markdown('**Message:**');st.write(reminder[_m]);st.code(reminder[_m],language=_D)
				st.markdown('### 📧 Bulk Actions');col_bulk,col_download=st.columns(2)
				with col_bulk:
					reminders_exist=bool(st.session_state.get(_As,[]));gmail_ready=st.session_state.get(_U,_B)and generator.gmail.connected;send_disabled=not(reminders_exist and gmail_ready)
					if st.button('📧 Send All Reminders',key='send_all_reminders_main',disabled=send_disabled,help='Send all generated reminders via email',use_container_width=_A,type=_X):
						if st.session_state.get(_U,_B)and st.session_state.get(_As):
							with st.spinner(f"Sending {len(st.session_state.current_reminders)} reminder emails..."):
								email_reminders=[r for r in st.session_state.current_reminders if r[_A3]];results=generator.gmail.send_bulk_reminders(email_reminders,st.session_state.gmail_sender_name_input);st.session_state.email_status[_AU]=results;st.success(f"✅ Sent {results[_Ax]} emails successfully")
								if results[_AF]>0:st.error(f"❌ Failed to send {results[_AF]} emails")
								st.rerun()
						else:st.error('Gmail not connected or no reminders generated')
				with col_download:
					if st.button('📄 Download All',key='download_reminders_button',use_container_width=_A):reminder_text='\n\n'+'='*50+'\n\n';reminder_text=reminder_text.join([f"""REMINDER FOR: {r[_E]} ({r[_J]})
Parent: {r[_O]}
Phone: {r[_R]}
Email: {r[_K]}

MESSAGE:
{r[_m]}"""for r in reminders]);st.download_button('📥 Download Reminders',data=reminder_text,file_name=f"payment_reminders_{date.today().strftime(_C7)}.txt",mime=_B_)
				if st.session_state.email_status.get(_AU):results=st.session_state.email_status[_AU];st.success(f"📊 Bulk Send Results: {results[_Ax]} sent, {results[_AF]} failed")
		st.subheader('💬 Chat with AI')
		for msg in st.session_state.ticket_messages:
			with st.chat_message(msg[_F]):
				parts=split_message(msg[_C])
				for part in parts:
					if part[_AO]=='text':st.markdown(part[_C],unsafe_allow_html=_A)
					elif part[_AO]==_Am:st.code(part[_Am],language=part[_BB])
		if(prompt:=st.chat_input('Ask about your lunch program...')):
			st.session_state.ticket_messages.append({_F:_L,_C:prompt});students_df=st.session_state.get(_a,pd.DataFrame());payments_df=st.session_state.get(_C5,pd.DataFrame())
			if not students_df.empty and not payments_df.empty:
				total_students=len(students_df);paid_students_count=len(payments_df[payments_df[_f]==_n]);unpaid_students=total_students-paid_students_count;payment_rate=paid_students_count/total_students*100 if total_students>0 else 0;paid_students_df=st.session_state.get(_C4,pd.DataFrame());paid_names=paid_students_df[_E].tolist()if not paid_students_df.empty else[];paid_ids=payments_df[payments_df[_f]==_n][_N];unpaid_names=students_df[~students_df[_N].isin(paid_ids)][_E].tolist()if not students_df.empty else[];class_info=''
				if _J in students_df.columns:class_counts=students_df[_J].value_counts();paid_by_class=paid_students_df[_J].value_counts()if not paid_students_df.empty else pd.Series();class_breakdown=[f"{cls}: {paid_by_class.get(cls,0)}/{total} paid"for(cls,total)in class_counts.head(10).items()];class_info='\n- '+'\n'.join(class_breakdown)
				prompt_text=f"""
Current Lunch Payment Data:
- Total Students: {total_students}
- Students Who Paid: {paid_students_count}
- Students Still Need to Pay: {unpaid_students}
- Payment Rate: {payment_rate:.1f}%

Students Who Have Paid (Names):
{_I.join(paid_names)if paid_names else"None"}

Students Who Haven't Paid (Names):
{_I.join(unpaid_names)if unpaid_names else"None"}

Class Breakdown:{class_info}

User Question: {prompt}
""";system_prompt='You are analyzing actual school lunch payment data. Use the provided data to answer questions specifically and accurately. Be direct and practical in your responses.'
				if current_ai_provider==_Ai and generator.ollama_model:
					api_payload={_z:generator.ollama_model,_l:[{_F:_b,_C:system_prompt},{_F:_L,_C:prompt_text}],_A0:_A}
					with st.spinner(_C0):
						response_placeholder=st.empty();accumulated_response=''
						try:
							response=requests.post(f"{get_ollama_host()}/api/chat",json=api_payload,headers={_j:_k},stream=_A,timeout=60);response.raise_for_status()
							for line in response.iter_lines():
								if line:
									data=json.loads(line.decode(_M))
									if _Q in data and _C in data[_Q]:accumulated_response+=data[_Q][_C];response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
							st.session_state.ticket_messages.append({_F:_Y,_C:accumulated_response})
						except requests.exceptions.Timeout:st.error(_C6);st.session_state.ticket_messages.append({_F:_Y,_C:_C6})
						except Exception as e:st.error(f"Error: {str(e)}");st.session_state.ticket_messages.append({_F:_Y,_C:f"Error: {str(e)}"})
						st.rerun()
				elif current_ai_provider==_Aj and openrouter_api_key and openrouter_model:
					messages=[{_F:_b,_C:system_prompt},{_F:_L,_C:prompt_text}]
					with st.spinner('☁️ Cloud AI thinking...'):
						response_placeholder=st.empty();accumulated_response=''
						try:
							response=call_openrouter_api(messages,openrouter_model,openrouter_api_key)
							if response:
								for chunk in response.iter_lines():
									if chunk:
										if chunk==b'':continue
										if chunk.startswith(_Ak):
											try:
												data=json.loads(chunk.decode(_M)[5:])
												if _S in data and len(data[_S])>0:delta=data[_S][0].get(_Al,{});accumulated_response+=delta.get(_C,'');response_placeholder.markdown(accumulated_response,unsafe_allow_html=_A)
											except json.JSONDecodeError:pass
								st.session_state.ticket_messages.append({_F:_Y,_C:accumulated_response})
							else:error_msg=_C3;st.error(error_msg);st.session_state.ticket_messages.append({_F:_Y,_C:error_msg})
						except Exception as e:error_msg=f"Error: {str(e)}";st.error(error_msg);st.session_state.ticket_messages.append({_F:_Y,_C:error_msg})
						st.rerun()
				else:st.info(_DB)
			else:st.info('📝 Please load data for specific analysis.')
	with tab3:
		st.markdown('<div class="section-header">Generate Tickets</div>',unsafe_allow_html=_A)
		if _C4 in st.session_state and not st.session_state.paid_students.empty:
			paid_count=len(st.session_state.paid_students);st.success(f"✅ Ready to generate tickets for {paid_count} paid students");st.info(f"📅 {validity_info[_AV]}")
			with st.expander('👀 Preview Student List'):st.dataframe(st.session_state.paid_students[[_E,_J,_N]],use_container_width=_A)
			if st.button('🖨 Generate PDF Tickets',key='generate_pdf_tickets_button',type=_X,use_container_width=_A):
				with st.spinner('🎫 Generating tickets...'):
					try:school_info={_E:school_name};pdf_buffer=generator.generate_pdf(st.session_state.paid_students,school_info,ticket_date,validity_info);st.success('✅ Tickets generated successfully!');filename=f"lunch_tickets_{school_name.replace(' ','_')}_{ticket_date.strftime(_C7)}.pdf";st.download_button('📥 Download PDF Tickets',data=pdf_buffer.getvalue(),file_name=filename,mime=_Cw,key='download_pdf_tickets_button',type=_X,use_container_width=_A);st.info(f"📊 Generated {paid_count} tickets ({(paid_count+14)//15} pages)")
					except Exception as e:st.error(f"❌ Error generating tickets: {str(e)}")
		else:
			st.info('📝 Please load and process student data first to generate tickets')
			with st.expander('ℹ️ How to get started'):st.markdown("\n1. Go to the **Data Management** tab\n2. Load or import your student and payment data\n3. Ensure you have students marked as 'paid'\n4. Click the 'Generate PDF Tickets' button\n")
elif st.session_state.current_section==_BY:
	st.markdown('<div class="section-header">Database Management</div>',unsafe_allow_html=_A);tables=[_a,_t,_u,_v,_w,_x,_AQ,_B7,_DE,'library','behaviour',_DF,_BI,_BJ,_BH];selected_table=st.selectbox('Select Table',tables,key='db_table_select');st.markdown('<div class="section-header">Current Schema</div>',unsafe_allow_html=_A);schema=get_table_schema(selected_table)
	if schema:schema_df=pd.DataFrame(schema,columns=['cid',_E,_AO,'notnull','dflt_value','pk']);st.dataframe(schema_df[[_E,_AO]],use_container_width=_A)
	else:st.info(f"No schema found for table: {selected_table}")
	st.markdown('<div class="section-header">Field Management</div>',unsafe_allow_html=_A)
	with st.expander('Manage Fields',expanded=_B):
		with st.form(key='field_management_form'):
			operation=st.radio('Select Operation',[_C8,_BD,_An],horizontal=_A,key='field_operation')
			if operation==_C8:
				st.subheader('Add New Field');col1,col2=st.columns(2)
				with col1:field_name=st.text_input('Field Name',key='field_name')
				with col2:field_types=['TEXT','INTEGER','REAL','DATE','BOOLEAN'];field_type=st.selectbox('Field Type',field_types,key='field_type')
			elif operation==_BD and schema:
				st.subheader(_BD);col1,col2=st.columns(2)
				with col1:old_name=st.selectbox('Select Field',[col[1]for col in schema],key='old_field_name')
				with col2:new_name=st.text_input('New Name',key='new_field_name')
			elif operation==_An and schema:st.subheader(_An);field_to_delete=st.selectbox('Select Field to Delete',[col[1]for col in schema],key='field_to_delete')
			submit_label=f"{operation.split()[0]} Field";button_type=_X if operation==_An else _BA
			if st.form_submit_button(submit_label,type=button_type):
				if operation==_C8:
					if field_name:
						success,message=add_column_to_table(selected_table,field_name,field_type)
						if success:st.success(message);st.rerun()
						else:st.error(f"Error: {message}")
					else:st.error('Field name is required')
				elif operation==_BD:
					if old_name and new_name:
						success,message=rename_column(selected_table,old_name,new_name)
						if success:st.success(message);st.rerun()
						else:st.error(f"Error: {message}")
					else:st.error('Both fields are required')
				elif operation==_An:
					if field_to_delete:
						success,message=delete_column(selected_table,field_to_delete)
						if success:st.success(message);st.rerun()
						else:st.error(f"Error: {message}")
					else:st.error('Please select a field to delete')
	with st.expander('📤 Export Data',expanded=_B):
		st.subheader('Export Table Data');st.info(f"Export all records from the '{selected_table}' table as CSV")
		if st.button(f"💾 Export {selected_table.capitalize()} Data",key='export_data_button',type=_BA,use_container_width=_A):
			try:
				conn=sqlite3.connect(DB_NAME);df=pd.read_sql_query(f"SELECT * FROM {selected_table}",conn);conn.close()
				if not df.empty:csv=df.to_csv(index=_B).encode(_M);today=date.today().strftime(_C7);filename=f"{selected_table}_export_{today}.csv";st.download_button(label='📥 Download CSV',data=csv,file_name=filename,mime=_AN,key='download_csv_button');st.success(f"✅ Exported {len(df)} records")
				else:st.info(f"No data found in {selected_table} table")
			except Exception as e:st.error(f"Error exporting data: {str(e)}")
	with st.expander(_DG,expanded=_B):
		st.subheader('Import Data into Table');st.info(f"Import data into the '{selected_table}' table from a CSV file");uploaded_file=st.file_uploader(f"Upload CSV file for {selected_table}",type=[_A8],key=f"import_{selected_table}_uploader")
		if uploaded_file is not _D:
			try:
				df=pd.read_csv(uploaded_file);st.success('✅ File uploaded successfully!');st.dataframe(df.head(5));st.subheader('Column Mapping');st.info('Map CSV columns to database columns');schema=get_table_schema(selected_table);db_columns=[col[1]for col in schema]if schema else[]
				if not db_columns:st.error(f"Could not retrieve schema for {selected_table}")
				else:
					mapping={}
					for csv_col in df.columns:
						st.markdown(f"**CSV Column:** `{csv_col}`");selected_db_col=st.selectbox(f"Map to database column:",['(Ignore)']+db_columns,key=f"map_{csv_col}",index=0)
						if selected_db_col!='(Ignore)':mapping[csv_col]=selected_db_col
					if st.button(_DG,key=f"import_{selected_table}_button"):
						if not mapping:st.error('Please map at least one column')
						else:
							with st.spinner('Importing data...'):
								try:mapped_df=df.rename(columns=mapping)[list(mapping.values())];conn=sqlite3.connect(DB_NAME);mapped_df.to_sql(name=selected_table,con=conn,if_exists='append',index=_B);conn.close();st.success(f"✅ Imported {len(df)} records into {selected_table}");query_db.cache_clear();st.rerun()
								except Exception as e:st.error(f"Import error: {str(e)}")
			except Exception as e:st.error(f"Error reading CSV file: {str(e)}")
	st.markdown('<div class="section-header">Backup & Restore</div>',unsafe_allow_html=_A);col1,col2=st.columns(2)
	with col1:
		with st.expander('💾 Backup Database',expanded=_B):
			st.info('Create a complete backup of your entire database');backup_format=st.radio('Backup Format',[_DH,'SQL Script (.sql)'],key='backup_format')
			if st.button('📦 Create Backup',key='create_backup_button',type=_X):
				try:
					timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
					if backup_format==_DH:
						with open(DB_NAME,'rb')as f:backup_data=f.read()
						filename=f"school_backup_{timestamp}.db";mime_type='application/x-sqlite3';st.download_button(label='📥 Download Database Backup',data=backup_data,file_name=filename,mime=mime_type,key='download_db_backup');st.success(f"✅ Database backup ready: {filename}")
					else:
						conn=sqlite3.connect(DB_NAME);sql_dump=io.StringIO();sql_dump.write(f"-- School Database Backup\n");sql_dump.write(f"-- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n");sql_dump.write(f"-- Version: {APP_VERSION}\n\n")
						for line in conn.iterdump():sql_dump.write(f"{line}\n")
						conn.close();backup_data=sql_dump.getvalue();filename=f"school_backup_{timestamp}.sql";st.download_button(label='📥 Download SQL Backup',data=backup_data,file_name=filename,mime=_B_,key='download_sql_backup');st.success(f"✅ SQL backup ready: {filename}")
				except Exception as e:st.error(f"Backup failed: {str(e)}")
			st.markdown(_c);st.markdown('**Current Database Info:**')
			try:
				import os;db_size=os.path.getsize(DB_NAME)/1024;conn=sqlite3.connect(DB_NAME);c=conn.cursor();tables=[_a,_t,_u,_v,_w,_x,_AQ,_B7,_DE,'library','behavior',_DF,_BI,_BJ,_BH];total_records=0
				for table in tables:
					try:count=c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0];total_records+=count
					except:pass
				conn.close();st.metric('Database Size',f"{db_size:.2f} KB");st.metric('Total Records',f"{total_records:,}")
			except Exception as e:st.warning(f"Could not retrieve database info: {str(e)}")
	with col2:
		with st.expander('📂 Restore Database',expanded=_B):
			st.warning('⚠️ Restoring will replace your current database!');restore_file=st.file_uploader('Upload Backup File',type=['db','sql','sqlite','sqlite3'],key='restore_file_uploader',help='Upload a .db or .sql backup file')
			if restore_file is not _D:
				file_size=len(restore_file.getvalue())/1024;st.info(f"📄 File: {restore_file.name} ({file_size:.2f} KB)");restore_mode=st.radio('Restore Mode',[_BE,'Merge with Existing Data'],key='restore_mode',help='Replace: Deletes current database | Merge: Adds to existing data');st.markdown(_c);st.markdown('**⚠️ Confirmation Required**')
				if restore_mode==_BE:confirm_text=st.text_input("Type 'RESTORE DATABASE' to confirm",key='restore_confirm_input',placeholder=_DI);confirm_valid=confirm_text==_DI
				else:confirm_valid=st.checkbox('I understand this will add data to existing records',key='merge_confirm')
				if st.button('🔄 Restore Database',key='restore_button',type=_X,disabled=not confirm_valid):
					try:
						with st.spinner('Restoring database...'):
							if restore_file.name.endswith('.sql'):
								sql_content=restore_file.getvalue().decode(_M)
								if restore_mode==_BE:
									query_db.cache_clear()
									if os.path.exists(DB_NAME):os.remove(DB_NAME)
									conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.executescript(sql_content);conn.commit();conn.close()
								else:
									conn=sqlite3.connect(DB_NAME);c=conn.cursor();merge_sql=[]
									for line in sql_content.split('\n'):
										if not any(keyword in line.upper()for keyword in['DROP TABLE','CREATE TABLE']):merge_sql.append(line)
									c.executescript('\n'.join(merge_sql));conn.commit();conn.close()
							else:
								query_db.cache_clear()
								if restore_mode==_BE:
									if os.path.exists(DB_NAME):os.remove(DB_NAME)
									with open(DB_NAME,'wb')as f:f.write(restore_file.getvalue())
								else:
									backup_path='temp_backup.db'
									with open(backup_path,'wb')as f:f.write(restore_file.getvalue())
									conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute(f"ATTACH DATABASE '{backup_path}' AS backup");tables=c.execute("SELECT name FROM backup.sqlite_master WHERE type='table'").fetchall();merged_count=0
									for(table_name,)in tables:
										try:c.execute(f"INSERT OR IGNORE INTO {table_name} SELECT * FROM backup.{table_name}");merged_count+=c.rowcount
										except sqlite3.Error as e:st.warning(f"Could not merge table {table_name}: {str(e)}")
									c.execute('DETACH DATABASE backup');conn.commit();conn.close();os.remove(backup_path);st.info(f"Merged {merged_count} records")
							st.success('✅ Database restored successfully!');st.balloons();query_db.cache_clear();init_db();time.sleep(1);st.rerun()
					except Exception as e:st.error(f"❌ Restore failed: {str(e)}");st.exception(e)
			else:st.info('👆 Upload a backup file to begin restoration')
	st.markdown(_c)
	with st.expander('📅 Backup Best Practices',expanded=_B):st.markdown('\n        ### Backup Recommendations:\n        \n        1. **Regular Backups**: Create backups at least weekly\n        2. **Before Major Changes**: Always backup before:\n           - Importing large datasets\n           - Database structure changes\n           - End of term operations\n        3. **Multiple Locations**: Store backups in:\n           - Local computer\n           - Cloud storage (Google Drive, Dropbox)\n           - External drive\n        4. **Version Control**: Keep multiple backup versions\n        5. **Test Restores**: Periodically test your backups\n        \n        ### File Formats:\n        - **SQLite (.db)**: Complete binary backup, fastest restore\n        - **SQL Script (.sql)**: Text-based, human-readable, portable\n        \n        ### Restore Modes:\n        - **Replace**: Complete database replacement (recommended for full restore)\n        - **Merge**: Add backup data to existing database (use carefully)\n        ')
	st.markdown(_D9,unsafe_allow_html=_A)
	with st.expander('⚠️ Delete All Data',expanded=_B):
		st.warning('This will permanently delete ALL records from the selected table. This action cannot be undone!');confirm_text=st.text_input(f"Type 'DELETE {selected_table.upper()}' to confirm",key='delete_confirm_input',placeholder=f"DELETE {selected_table.upper()}")
		if st.button('🗑️ Delete All Records',type=_X,disabled=confirm_text!=f"DELETE {selected_table.upper()}",key='delete_all_data_button',help='Confirm by typing the required text above'):
			try:
				conn=sqlite3.connect(DB_NAME);c=conn.cursor();c.execute(f"DELETE FROM {selected_table}");conn.commit()
				if selected_table!=_AQ:c.execute(f"DELETE FROM sqlite_sequence WHERE name = '{selected_table}'");conn.commit()
				conn.close();query_db.cache_clear();st.success(f"✅ All records deleted from {selected_table} table!");st.rerun()
			except Exception as e:st.error(f"Error deleting data: {str(e)}")
elif st.session_state.current_section==_BZ:
	st.markdown('<div class="section-header">Tools</div>',unsafe_allow_html=_A)
	def migrate_room_assignments_to_class_members():
		'\n        Migrate existing room assignments from students table to class_members table\n        ';st.markdown('### 🔄 Data Migration Tool');st.markdown('This tool will migrate students from room-based assignments to the proper class_members table.')
		if st.expander('🔍 Database Diagnostics',expanded=_B):show_database_diagnostics()
		col1,col2=st.columns(2)
		with col1:
			st.subheader('Current Room Assignments');room_stats_query="\n                SELECT \n                    COALESCE(room, 'Unassigned') as room,\n                    COUNT(*) as student_count\n                FROM students \n                GROUP BY room\n                ORDER BY room\n            ";room_stats_result,_=query_db(room_stats_query)
			if room_stats_result:room_stats_df=pd.DataFrame(room_stats_result,columns=[_AZ,_B1]);st.dataframe(room_stats_df)
		with col2:
			st.subheader('Class Members Table');class_members_stats_query='\n                SELECT \n                    COUNT(*) as total_memberships,\n                    COUNT(DISTINCT student_id) as unique_students,\n                    COUNT(DISTINCT class_id) as classes_with_students\n                FROM class_members\n            ';cm_stats_result,_=query_db(class_members_stats_query)
			if cm_stats_result:total_memberships,unique_students,classes_with_students=cm_stats_result[0];st.metric('Total Memberships',total_memberships);st.metric('Unique Students',unique_students);st.metric('Classes with Students',classes_with_students)
		st.markdown(_c);st.subheader('Migration Options');term_results,_=query_db('SELECT id, name FROM terms ORDER BY name')
		if term_results:
			term_options={row[1]:row[0]for row in term_results};migration_term=st.selectbox('Select term for migration',list(term_options.keys()),key='migration_term_select',help='Students will be enrolled in classes for this term');migration_term_id=term_options[migration_term];st.markdown('#### Migration Status');unmigrated_count,rooms_without_classes=check_migration_status(migration_term_id,migration_term);st.markdown(_c);col1,col2,col3=st.columns(3)
			with col1:
				if st.button('🏫 Create Missing Classes',key='create_missing_classes',disabled=rooms_without_classes==0):create_classes_for_rooms(migration_term_id,migration_term)
			with col2:
				if st.button('🔄 Migrate Students',key='migrate_rooms',type=_X,disabled=unmigrated_count==0):migrate_students_by_room(migration_term_id,migration_term)
			with col3:
				if st.button('🧹 Clean Up Room Data',key='cleanup_rooms'):cleanup_room_assignments()
			if rooms_without_classes>0:st.info('💡 Create missing classes first, then migrate students')
			elif unmigrated_count>0:st.success('✅ All classes exist! Ready to migrate students')
			else:st.success('✅ All students are properly assigned to classes')
		else:st.warning('No terms available. Please create terms first.')
	def check_migration_status(term_id,term_name):
		'Check and display migration status';conn=sqlite3.connect(DB_NAME);c=conn.cursor();unmigrated_query="\n            SELECT COUNT(*) \n            FROM students s\n            WHERE s.room IS NOT NULL AND s.room != ''\n            AND s.id NOT IN (\n                SELECT student_id FROM class_members WHERE term_id = ?\n            )\n        ";unmigrated_count=c.execute(unmigrated_query,(term_id,)).fetchone()[0];migrated_query='\n            SELECT COUNT(DISTINCT cm.student_id)\n            FROM class_members cm\n            JOIN students s ON cm.student_id = s.id\n            WHERE cm.term_id = ? AND s.room IS NOT NULL\n        ';migrated_count=c.execute(migrated_query,(term_id,)).fetchone()[0];classes_without_rooms=c.execute("\n            SELECT COUNT(*) FROM classes \n            WHERE term_id = ? AND (room IS NULL OR room = '')\n        ",(term_id,)).fetchone()[0];rooms_needing_classes=c.execute("\n            SELECT COUNT(DISTINCT s.room)\n            FROM students s\n            WHERE s.room IS NOT NULL AND s.room != ''\n            AND s.room NOT IN (\n                SELECT DISTINCT room FROM classes \n                WHERE term_id = ? AND room IS NOT NULL AND room != ''\n            )\n        ",(term_id,)).fetchone()[0];missing_rooms=c.execute("\n            SELECT DISTINCT s.room\n            FROM students s\n            WHERE s.room IS NOT NULL AND s.room != ''\n            AND s.room NOT IN (\n                SELECT DISTINCT room FROM classes \n                WHERE term_id = ? AND room IS NOT NULL AND room != ''\n            )\n        ",(term_id,)).fetchall();conn.close();col1,col2,col3,col4=st.columns(4)
		with col1:st.metric('Students to Migrate',unmigrated_count)
		with col2:st.metric('Already Migrated',migrated_count)
		with col3:st.metric('Classes w/o Rooms',classes_without_rooms)
		with col4:st.metric('Rooms w/o Classes',rooms_needing_classes)
		if missing_rooms:st.info(f"Rooms missing classes: {_I.join([r[0]for r in missing_rooms])}")
		return unmigrated_count,rooms_needing_classes
	def create_classes_for_rooms(term_id,term_name):
		"Create classes for rooms that don't have corresponding classes";conn=sqlite3.connect(DB_NAME);c=conn.cursor();rooms_result=c.execute("\n            SELECT DISTINCT room \n            FROM students \n            WHERE room IS NOT NULL AND room != ''\n        ").fetchall();existing_classes=c.execute('\n            SELECT room FROM classes WHERE term_id = ? AND room IS NOT NULL\n        ',(term_id,)).fetchall();existing_rooms={row[0]for row in existing_classes};all_classes_with_room=c.execute("\n            SELECT room, term_id, name FROM classes \n            WHERE room IS NOT NULL AND room != ''\n        ").fetchall();teacher_result=c.execute(_DJ).fetchone();default_teacher_id=teacher_result[0]if teacher_result else 1;created_count=0;updated_count=0;errors=[]
		for(room,)in rooms_result:
			if room not in existing_rooms:
				existing_in_other_term=[cls for cls in all_classes_with_room if cls[0]==room]
				if existing_in_other_term:
					other_term_class=existing_in_other_term[0];st.warning(f"Room {room} already has a class '{other_term_class[2]}' in term {other_term_class[1]}")
					try:
						c.execute('\n                            UPDATE classes \n                            SET term_id = ? \n                            WHERE room = ? AND term_id = ?\n                        ',(term_id,room,other_term_class[1]))
						if c.rowcount>0:updated_count+=1;st.info(f"Updated existing class for room {room} to {term_name}")
						else:unique_name=f"Class {room} ({term_name})";c.execute(_Cb,(unique_name,room,default_teacher_id,term_id));created_count+=1
					except sqlite3.IntegrityError as e:errors.append(f"Room {room}: {str(e)} - Class may already exist for this term");continue
				else:
					try:class_name=f"Class {room}";c.execute('\n                            INSERT INTO classes (name, room, teacher_id, term_id)\n                            VALUES (?, ?, ?, ?)\n                        ',(class_name,room,default_teacher_id,term_id));created_count+=1
					except sqlite3.IntegrityError as e:errors.append(f"Room {room}: {str(e)}");continue
		conn.commit();conn.close()
		if created_count>0:st.success(f"Created {created_count} new classes for {term_name}")
		if updated_count>0:st.success(f"Updated {updated_count} existing classes to {term_name}")
		if created_count==0 and updated_count==0:st.info('No new classes needed')
		if errors:
			with st.expander('Class Creation Errors',expanded=_B):
				for error in errors:st.warning(error)
			st.markdown('#### Diagnostic Information');st.markdown('**All classes with rooms:**')
			if all_classes_with_room:
				for(room,term_id_val,name)in all_classes_with_room:st.write(f"- Room {room}: '{name}' (Term {term_id_val})")
		if created_count>0 or updated_count>0:st.rerun()
	def migrate_students_by_room(term_id,term_name):
		'Migrate students from room assignments to class_members table';conn=sqlite3.connect(DB_NAME);c=conn.cursor();students_with_rooms=c.execute("\n            SELECT s.id, s.room\n            FROM students s\n            WHERE s.room IS NOT NULL AND s.room != ''\n            AND s.id NOT IN (\n                SELECT student_id FROM class_members WHERE term_id = ?\n            )\n        ",(term_id,)).fetchall()
		if not students_with_rooms:st.info('No students need migration');conn.close();return
		existing_classes=c.execute('\n            SELECT id, room FROM classes \n            WHERE term_id = ? AND room IS NOT NULL\n        ',(term_id,)).fetchall();room_to_class={room:class_id for(class_id,room)in existing_classes};teacher_result=c.execute(_DJ).fetchone();default_teacher_id=teacher_result[0]if teacher_result else 1;unique_rooms=set(room for(_,room)in students_with_rooms);classes_created=0
		for room in unique_rooms:
			if room not in room_to_class:
				try:class_name=f"Class {room}";c.execute('\n                        INSERT INTO classes (name, room, teacher_id, term_id)\n                        VALUES (?, ?, ?, ?)\n                    ',(class_name,room,default_teacher_id,term_id));room_to_class[room]=c.lastrowid;classes_created+=1
				except sqlite3.IntegrityError as e:
					existing_class=c.execute('\n                        SELECT id FROM classes \n                        WHERE room = ? AND term_id = ?\n                    ',(room,term_id)).fetchone()
					if existing_class:room_to_class[room]=existing_class[0]
					else:st.error(f"Failed to create or find class for room {room}: {str(e)}");continue
		migrated_count=0;errors=[]
		for(student_id,room)in students_with_rooms:
			if room in room_to_class:
				try:c.execute('\n                        INSERT INTO class_members (class_id, student_id, term_id)\n                        VALUES (?, ?, ?)\n                    ',(room_to_class[room],student_id,term_id));migrated_count+=1
				except sqlite3.IntegrityError as e:errors.append(f"Student {student_id} (room {room}): {str(e)}")
			else:errors.append(f"No class found for room {room}")
		conn.commit();conn.close();success_msg=f"Migrated {migrated_count} students to {term_name} classes"
		if classes_created>0:success_msg+=f" (created {classes_created} new classes)"
		st.success(success_msg)
		if errors:
			with st.expander(f"Migration Issues ({len(errors)})",expanded=_B):
				for error in errors:st.warning(error)
		if migrated_count>0:st.rerun()
	def cleanup_room_assignments():
		'Clean up room assignments after successful migration';st.warning('⚠️ This will clear all room assignments from the students table!');st.markdown('**Only do this after confirming the migration was successful.**');cm_count_result,_=query_db('SELECT COUNT(*) FROM class_members');cm_count=cm_count_result[0][0]if cm_count_result else 0;st.info(f"Current class memberships: {cm_count}");confirm_cleanup=st.checkbox('I confirm that students are properly assigned in class_members table',key='confirm_cleanup')
		if confirm_cleanup and st.button('Clear Room Assignments',key='clear_rooms',type=_BA):conn=sqlite3.connect(DB_NAME);c=conn.cursor();result=c.execute('UPDATE students SET room = NULL');cleared_count=result.rowcount;conn.commit();conn.close();st.success(f"Cleared room assignments for {cleared_count} students");st.info('Students are now managed through the class_members table only');st.rerun()
	def show_database_diagnostics():
		'Show diagnostic information about the database state';st.subheader('Database Diagnostics');conn=sqlite3.connect(DB_NAME);c=conn.cursor();st.markdown('**Classes by Room and Term:**');all_classes=c.execute("\n            SELECT c.room, c.term_id, t.name as term_name, c.name as class_name, c.id\n            FROM classes c\n            LEFT JOIN terms t ON c.term_id = t.id\n            WHERE c.room IS NOT NULL AND c.room != ''\n            ORDER BY c.room, c.term_id\n        ").fetchall()
		if all_classes:
			classes_df=pd.DataFrame(all_classes,columns=[_AZ,'Term ID',_Bj,_Ca,'Class ID']);st.dataframe(classes_df);room_counts={}
			for(room,term_id,term_name,class_name,class_id)in all_classes:
				if room not in room_counts:room_counts[room]={}
				room_counts[room][term_id]=term_name,class_name
			duplicate_rooms={room:terms for(room,terms)in room_counts.items()if len(terms)>1}
			if duplicate_rooms:
				st.warning('**Rooms with classes in multiple terms:**')
				for(room,terms)in duplicate_rooms.items():term_info=[f"Term {tid} ({tname})"for(tid,(tname,_))in terms.items()];st.write(f"- Room {room}: {_I.join(term_info)}")
		st.markdown('**Students by Room:**');students_by_room=c.execute("\n            SELECT \n                COALESCE(room, 'No Room') as room,\n                COUNT(*) as student_count,\n                GROUP_CONCAT(name, ', ') as students\n            FROM students \n            GROUP BY room\n            ORDER BY room\n        ").fetchall()
		if students_by_room:
			for(room,count,students)in students_by_room:
				with st.expander(f"Room {room} ({count} students)"):st.write(students)
		st.markdown('**Database Schema Info:**');schema_info=c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='classes'").fetchone()
		if schema_info:st.code(schema_info[0],language='sql')
		conn.close()
	migrate_room_assignments_to_class_members()