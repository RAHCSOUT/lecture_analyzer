import os
import gradio as gr
import assemblyai as aai
from groq import Groq
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi
import json
import hashlib
from functools import partial
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import uuid

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
try:
    client = MongoClient(MONGO_URI)
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")


# Initialize API clients
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = 'lecture_analyzer'
USERS_COLLECTION = 'users'
LOGIN_HISTORY_COLLECTION = 'login_history'

# Initialize clients
groq_client = None
youtube = None

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    users_collection = db[USERS_COLLECTION]
    login_history_collection = db[LOGIN_HISTORY_COLLECTION]
    
    # Create unique index on username
    users_collection.create_index('username', unique=True)
    
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")

# Add these new functions
def create_new_user(username: str, password: str, email: str, name: str, role: str = 'student') -> tuple[bool, str]:
    """Create a new user in MongoDB"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    # Validate role
    if role not in ['student', 'instructor']:
        return False, "Role must be either 'student' or 'instructor'"
    
    try:
        # Check if username exists
        if users_collection.find_one({'username': username}):
            return False, "Username already exists"
        
        # Create new user with required fields
        user = {
            'user_id': str(uuid.uuid4()),  # Generate unique user_id
            'username': username,
            'password': hash_password(password),
            'email': email,
            'name': name,
            'role': role,
            'created_at': datetime.utcnow(),
            'last_login': None
        }
        users_collection.insert_one(user)
        return True, "User created successfully"
    except Exception as e:
        return False, f"Error creating user: {str(e)}"
    
def get_user_role(username: str) -> Optional[str]:
    """Get user's role from MongoDB"""
    user = users_collection.find_one({'username': username})
    return user.get('role') if user else None

# 6. Add MongoDB helper functions
def initialize_mongodb():
    """Initialize MongoDB with schema validation"""
    try:
        # Create collection with schema validation
        validator = {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': ['user_id', 'name', 'email', 'role'],
                'properties': {
                    'user_id': {'bsonType': 'string'},
                    'username': {'bsonType': 'string'},
                    'password': {'bsonType': 'string'},
                    'email': {'bsonType': 'string'},
                    'name': {'bsonType': 'string'},
                    'role': {
                        'enum': ['student', 'instructor'],
                        'description': 'must be either student or instructor'
                    },
                    'created_at': {'bsonType': 'date'},
                    'last_login': {'bsonType': ['date', 'null']}
                }
            }
        }
        
        try:
            db.create_collection(USERS_COLLECTION)
        except:
            pass  # Collection might already exist
            
        db.command('collMod', USERS_COLLECTION, validator=validator)
        
        # Create unique index on username
        users_collection.create_index('username', unique=True)
        
        print("MongoDB initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing MongoDB: {str(e)}")
        return False
    
def check_login(username: str, password: str) -> tuple[bool, str]:
   try:
       user = users_collection.find_one({'username': username})
       if user and user['password'] == hash_password(password):
           users_collection.update_one(
               {'username': username},
               {'$set': {'last_login': datetime.utcnow()}}
           )
           
           login_history_collection.insert_one({
               'username': username,
               'timestamp': datetime.utcnow(),
               'success': True,
               'ip_address': gr.request.client.host if hasattr(gr, 'request') else None
           })
           
           return True, "Login successful!"
       
       login_history_collection.insert_one({
           'username': username,
           'timestamp': datetime.utcnow(),
           'success': False,
           'ip_address': gr.request.client.host if hasattr(gr, 'request') else None
       })
       
       return False, "Invalid username or password"
   except Exception as e:
       print(f"Login error: {str(e)}")  # Add logging
       return False, f"Login error: {str(e)}"
   
def initialize_clients():
    """Initialize API clients"""
    global groq_client, youtube
    
    try:
        groq_client = Groq(api_key=groq_api_key)
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        return True
    except Exception as e:
        print(f"Error initializing clients: {str(e)}")
        return False

def extract_video_id(youtube_url):
    query = parse_qs(urlparse(youtube_url).query)
    video_id = query.get("v")
    if video_id:
        return video_id[0]
    elif "youtu.be" in youtube_url:
        return youtube_url.split("/")[-1]
    else:
        raise ValueError("Invalid YouTube URL.")

def get_video_info(video_id):
    """Fetch video information and transcript with timestamps"""
    try:
        # Get video details
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()

        if not response['items']:
            return "No video found", 'en'

        video_info = response['items'][0]['snippet']
        title = video_info['title']
        description = video_info.get('description', 'No description available')
        language_code = video_info.get('defaultLanguage', 'en')

        try:
            # First try with default language
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            except:
                # If default fails, try to get all transcripts and use the first available one
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                if not transcript:
                    transcript = transcript_list.find_manually_created_transcript()
                    if not transcript:
                        transcript = transcript_list[0]
                    transcript = transcript.translate('en')
                transcript_list = transcript.fetch()

            formatted_segments = []
            current_text = []
            current_timestamp = None
            
            for entry in transcript_list:
                text = entry['text'].strip()
                start = entry['start']
                
                hours = int(start // 3600)
                minutes = int((start % 3600) // 60)
                seconds = int(start % 60)
                timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
                
                if not current_timestamp:
                    current_timestamp = timestamp
                
                current_text.append(text)
                
                if text.endswith(('.', '?', '!')):
                    full_text = ' '.join(current_text).strip()
                    if full_text:
                        formatted_segments.append(f"[{current_timestamp}] {full_text}\n\n")
                    current_text = []
                    current_timestamp = None
            
            if current_text:
                full_text = ' '.join(current_text).strip()
                if full_text:
                    formatted_segments.append(f"[{current_timestamp}] {full_text}\n\n")
            
            transcription_text = f"Title: {title}\n\nDescription: {description}\n\nTranscript:\n{''.join(formatted_segments)}"
            return transcription_text, language_code
            
        except Exception as e:
            error_msg = str(e)
            if "Subtitles are disabled for this video" in error_msg:
                msg = "This video does not have available captions/subtitles."
            else:
                msg = f"Error fetching transcript: {error_msg}"
            
            transcription_text = f"Title: {title}\n\nDescription: {description}\n\nTranscript: {msg}"
            return transcription_text, language_code
            
    except Exception as e:
        return f"Error fetching video information: {str(e)}", 'en'

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Helper function to implement retry logic with exponential backoff"""
    for retry_count in range(max_retries):
        try:
            result = func()
            if result:
                return result
            raise Exception("Empty response received")
        except Exception as e:
            if retry_count == max_retries - 1:
                raise e
            
            if "503" in str(e):
                delay = base_delay * (2 ** retry_count) + (random.random() * 0.5)
                delay = min(delay * 2, 30)
            else:
                delay = base_delay * (2 ** retry_count) + (random.random() * 0.1)
            
            print(f"Attempt {retry_count + 1} failed. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
    return None

def generate_analysis(text: str, analysis_type: str = "summary", language_code: str = 'en', max_retries: int = 3) -> str:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        text_to_summarize = " ".join(texts[:4])

        if analysis_type == "summary":
            prompt = f'''Summarize the following text in {language_code}:
            Text: {text_to_summarize}

            Include an INTRODUCTION, BULLET POINTS if applicable, and a CONCLUSION in {language_code}.'''
        elif analysis_type == "quiz":
            prompt = f'''Create a quiz based on this content with 5 multiple choice questions.
            Text: {text_to_summarize}

            Format each question as:
            Q1. [Question]
            a) [Option 1]
            b) [Option 2]
            c) [Option 3]
            d) [Option 4]
            Correct Answer: [letter]
            Explanation: [Why this is correct]'''
        else:  # Q&A
            prompt = f'''Generate 5 important questions and detailed answers about this content:
            Text: {text_to_summarize}

            Format as:
            Q1: [Question]
            A1: [Detailed answer]'''

        def make_api_call():
            if groq_client is None:
                raise Exception("Groq client not initialized")
            completion = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert educational content analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return completion.choices[0].message.content

        response_content = retry_with_backoff(make_api_call, max_retries=max_retries)
        
        if not response_content:
            return "Failed to generate analysis after multiple attempts."
            
        if "Q5" not in response_content and (analysis_type == "quiz" or analysis_type == "qa"):
            response_content += "\n\nNote: The analysis might be incomplete."

        return response_content

    except Exception as e:
        return f"Analysis error: {str(e)}"

def format_time(milliseconds: int) -> str:
    """Format milliseconds into a readable timestamp (HH:MM:SS)"""
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def process_file(audio_file: gr.File | None = None, youtube_url: str | None = None, analysis_type: str = "summary") -> str:
    """Process audio file or YouTube URL and generate analysis"""
    language_code = 'en'
    try:
        if audio_file is not None:
            file_path = audio_file.name
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                auto_chapters=True
            )
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(file_path, config)
            
            if not transcript or not transcript.text:
                return "Transcription error: Unable to transcribe the provided input."
            
            if analysis_type == "summary":
                current_segment = []
                current_start = None
                current_end = None
                formatted_segments = []
                current_text = []
                
                for word in transcript.words:
                    if not current_start:
                        current_start = word.start
                    
                    current_text.append(word.text)
                    current_end = word.end
                    
                    if word.text.rstrip().endswith(('.', '?', '!')):
                        start_time = format_time(current_start)
                        end_time = format_time(current_end)
                        segment_text = ' '.join(current_text).strip()
                        
                        formatted_segments.append(
                            f"[{start_time}] {segment_text}\n\n"
                        )
                        
                        current_text = []
                        current_start = None
                
                if current_text:
                    start_time = format_time(current_start)
                    end_time = format_time(current_end)
                    segment_text = ' '.join(current_text).strip()
                    formatted_segments.append(
                        f"[{start_time}] {segment_text}\n\n"
                    )
                
                transcription_text = ''.join(formatted_segments)
            else:
                transcription_text = transcript.text

        elif youtube_url:
            video_id = extract_video_id(youtube_url)
            transcription_text, language_code = get_video_info(video_id)
            if transcription_text == "Unable to fetch video information":
                return "Transcription error: Unable to fetch video information."
        else:
            return "Please provide either an audio file or YouTube URL"
        
        analysis = generate_analysis(transcription_text, analysis_type, language_code)
        
        if analysis_type == "summary":
            output = f"""Transcription (full text with timestamps):
{transcription_text}

{'='*50}

Analysis:
{analysis}
"""
        else:
            output = f"""Analysis:
{analysis}
"""
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"

# User management
USERS_FILE = "users.json"

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        users = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "admin"
            }
        }
        save_users(users)
        return users

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def create_login_interface():
    with gr.Blocks() as login_interface:
        gr.Markdown("# 🔐 Login")
        
        with gr.Tab("Login"):
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")
            output_message = gr.Textbox(label="Message", interactive=False)
        
        with gr.Tab("Register"):
            new_username = gr.Textbox(label="Username")
            new_password = gr.Textbox(label="Password", type="password")
            confirm_password = gr.Textbox(label="Confirm Password", type="password")
            email = gr.Textbox(label="Email")
            name = gr.Textbox(label="Full Name")
            role = gr.Radio(choices=["student", "instructor"], label="Role", value="student")
            register_button = gr.Button("Register")
            register_message = gr.Textbox(label="Message", interactive=False)
            
            def handle_registration(username, password, confirm, email, name, role):
                if password != confirm:
                    return "Passwords do not match"
                if len(password) < 6:
                    return "Password must be at least 6 characters long"
                if not email or '@' not in email:
                    return "Please provide a valid email address"
                if not name:
                    return "Please provide your full name"
                
                success, message = create_new_user(username, password, email, name, role)
                return message
            
            register_button.click(
                fn=handle_registration,
                inputs=[new_username, new_password, confirm_password, email, name, role],
                outputs=register_message
            )
        
        return login_interface, username, password, login_button, output_message
    
# Create the main application interface
def create_main_interface(username: str):
   with gr.Blocks(title="Upload New Lecture") as main_interface:
       gr.Markdown(f"# 🎓 Upload New Lecture\nWelcome, {username}!")
       logout_btn = gr.Button("Logout", scale=1)
       
       if get_user_role(username) == 'admin':
           with gr.Accordion("User Management", open=False):
               new_username = gr.Textbox(label="New Username")
               new_password = gr.Textbox(label="New Password", type="password")
               new_role = gr.Radio(choices=["user", "admin"], label="Role", value="user")
               create_user_btn = gr.Button("Create User")
               user_message = gr.Textbox(label="Status", interactive=False)
               
               create_user_btn.click(
                   fn=lambda u, p, r: create_new_user(u, p, r)[1],
                   inputs=[new_username, new_password, new_role],
                   outputs=user_message
               )
       
       with gr.Row():
           with gr.Column():
               audio_file = gr.File(label="Upload Audio/Video File", file_types=["audio", "video"])
               youtube_url = gr.Textbox(label="Or Enter YouTube URL", placeholder="https://youtube.com/watch?v=...")
               analysis_type = gr.Radio(choices=["summary", "quiz", "qa"], label="Analysis Type", value="summary")
               submit_btn = gr.Button("Analyze")
           
           with gr.Column():
               output = gr.TextArea(label="Results", lines=40, max_lines=100)
       
       if initialize_clients():
           submit_btn.click(fn=process_file, inputs=[audio_file, youtube_url, analysis_type], outputs=output)
           
       return main_interface, logout_btn
       
# Create the combined application
# Remove the JSON-based user management functions
# Delete these functions:
# - load_users()
# - save_users()
# - check_login() that uses JSON

class AuthApp:
    def __init__(self):
        if not initialize_mongodb():
            raise Exception("Failed to initialize MongoDB")
        
        self.login_interface, self.username, self.password, self.login_button, self.output_message = create_login_interface()
        self.current_user = None

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Group() as login_group:
                login_interface, username, password, login_button, output_message = create_login_interface()
            
            with gr.Group(visible=False) as main_group:
                main_interface, logout_btn = create_main_interface(self.current_user or "Guest")
            
            def auth_and_launch(username, password):
                # Use MongoDB check_login function
                success, message = check_login(username, password)
                if success:
                    self.current_user = username
                    return gr.update(visible=False), gr.update(visible=True), message
                return gr.update(visible=True), gr.update(visible=False), message
            
            login_button.click(
                fn=auth_and_launch,
                inputs=[username, password],
                outputs=[login_group, main_group, output_message]
            )
            
            logout_btn.click(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                outputs=[login_group, main_group]
            )
            
        demo.launch(share=True)

if __name__ == "__main__":
    app = AuthApp()
    app.launch()
