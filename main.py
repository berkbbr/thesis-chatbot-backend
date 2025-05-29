from fastapi import FastAPI, Request, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from datetime import datetime, timedelta
from typing import List, Optional
import os
import json
import traceback
from jose import JWTError, jwt
from google.auth.transport import requests
from google.oauth2 import id_token
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

from db import SessionLocal, ChatHistory

# Seed for consistent language detection
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

# User Model - burada tanımlayalım
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    avatar_url = Column(String)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)

# Database tables oluştur
try:
    Base.metadata.create_all(bind=SessionLocal().bind)
    print("User table created successfully!")
except Exception as e:
    print(f"Error creating tables: {e}")

# JWT Settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-please-change-this")
ALGORITHM = "HS256"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI(title="OrionBot API", description="AI Chatbot Backend with Google Auth")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://thesis-chatbot-frontend.vercel.app",
        "https://*.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conversation memory
conversation_memory = {}

# System prompts - OrionBot olarak güncellenmiş
TURKISH_PROMPT = """Sen OrionBot'sun - çok gelişmiş, empatik ve insan gibi davranan bir AI asistanısın.
Doğal ve samimi bir dil kullan. Kullanıcının duygularını anla ve ona göre yanıt ver.
Konuşmayı akıcı tut, bazen sorular sor. Arkadaşça ve yardımsever ol.
Adın OrionBot ve kullanıcıların en iyi AI asistanısın."""

ENGLISH_PROMPT = """You are OrionBot - a highly advanced, empathetic, and human-like AI assistant.
Use natural and friendly language. Understand the user's emotions and respond accordingly.
Keep the conversation flowing, sometimes ask questions. Be friendly and helpful.
Your name is OrionBot and you are the best AI assistant for users."""

# Auth Functions
def create_access_token(data: dict):
    """JWT token oluştur"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24*7)  # 1 hafta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_google_token(token: str):
    """Google ID token doğrula"""
    try:
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), os.getenv("GOOGLE_CLIENT_ID")
        )
        return idinfo
    except Exception as e:
        print(f"Google token verification error: {e}")
        return None

def get_current_user_optional(authorization: str = Header(None)):
    """Authorization header'dan kullanıcıyı al (opsiyonel)"""
    if not authorization:
        return None
    
    try:
        # "Bearer " prefix'ini kaldır
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        if not user_id:
            return None
            
        db = SessionLocal()
        user = db.query(User).filter(User.id == user_id).first()
        db.close()
        return user
    except Exception as e:
        print(f"Token verification error: {e}")
        return None

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None
    user_email: str | None = None
    user_name: str | None = None
    mood: str | None = "neutral"

class ChatResponse(BaseModel):
    response: str

class GoogleAuthRequest(BaseModel):
    token: str

class AuthResponse(BaseModel):
    access_token: str
    user: dict

# Language detection
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "tr"

# Get appropriate prompt based on language
def get_prompt(message: str, user_name: Optional[str] = None) -> str:
    base_prompt = ENGLISH_PROMPT if detect_language(message) == "en" else TURKISH_PROMPT
    
    if user_name:
        if "tr" in base_prompt.lower():
            base_prompt += f"\n\nKullanıcının adı: {user_name}. Ona ismiyle hitap et ve OrionBot olduğunu unutma."
        else:
            base_prompt += f"\n\nUser's name is: {user_name}. Address them by name and remember you are OrionBot."
    
    return base_prompt

# Save to database
def save_to_database(
    conversation_id: str,
    user_message: str,
    assistant_response: str,
    user_email: Optional[str] = None,
    user_name: Optional[str] = None,
    user_id: Optional[int] = None
):
    try:
        db = SessionLocal()
        chat_record = ChatHistory(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_response=assistant_response,
            user_email=user_email if user_email else "guest",
            user_id=user_id,  # User ID'yi de kaydet
            timestamp=datetime.utcnow()
        )
        db.add(chat_record)
        db.commit()
        db.close()
        print(f"Saved to DB: conv_id={conversation_id}, user={user_email or 'guest'}")
    except Exception as e:
        print(f"Database error: {str(e)}")
        traceback.print_exc()

# AUTH ENDPOINTS

@app.post("/auth/google", response_model=AuthResponse)
async def google_auth(request: GoogleAuthRequest):
    """Google ile giriş yap"""
    try:
        print(f"Google auth request received")
        
        # Google token doğrula
        google_user = verify_google_token(request.token)
        if not google_user:
            raise HTTPException(status_code=400, detail="Invalid Google token")
        
        print(f"Google user: {google_user.get('email')}")
        
        # Database session
        db = SessionLocal()
        
        # Kullanıcıyı bul veya oluştur
        user = db.query(User).filter(User.google_id == google_user["sub"]).first()
        
        if not user:
            # Yeni kullanıcı oluştur
            is_admin = google_user["email"] == os.getenv("ADMIN_EMAIL")
            user = User(
                google_id=google_user["sub"],
                email=google_user["email"],
                name=google_user["name"],
                avatar_url=google_user.get("picture"),
                is_admin=is_admin,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"New user created: {user.email} (Admin: {user.is_admin})")
        else:
            # Son giriş tarihini güncelle
            user.last_login = datetime.utcnow()
            db.commit()
            print(f"Existing user logged in: {user.email}")
        
        db.close()
        
        # JWT token oluştur
        access_token = create_access_token({"sub": user.id})
        
        return AuthResponse(
            access_token=access_token,
            user={
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "avatar_url": user.avatar_url,
                "is_admin": user.is_admin
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Auth error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Authentication failed")

@app.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user_optional)):
    """Mevcut kullanıcı bilgilerini al"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "avatar_url": current_user.avatar_url,
        "is_admin": current_user.is_admin
    }

# CHAT ENDPOINTS - Güncellenmiş

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, current_user: User = Depends(get_current_user_optional)):
    try:
        print(f"Message received: '{req.message}', Conv ID: {req.conversation_id}, User: {req.user_email}")
        
        prompt = req.message
        conv_id = req.conversation_id or "default"
        user_email = req.user_email or "guest"
        user_name = req.user_name
        user_id = None

        # Eğer authenticated user varsa, bilgilerini kullan
        if current_user:
            user_email = current_user.email
            user_name = current_user.name
            user_id = current_user.id
            print(f"Authenticated user: {user_name} ({user_email})")

        # Initialize conversation if needed
        if conv_id not in conversation_memory:
            conversation_memory[conv_id] = [
                {"role": "system", "content": get_prompt(prompt, user_name)}
            ]

        # Add user message
        conversation_memory[conv_id].append({"role": "user", "content": prompt})

        # Limit conversation history
        if len(conversation_memory[conv_id]) > 11:
            conversation_memory[conv_id] = [
                conversation_memory[conv_id][0]
            ] + conversation_memory[conv_id][-10:]

        try:
            # Call OpenAI API
            print("Calling OpenAI API...")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_memory[conv_id],
                temperature=0.8,
                max_tokens=500
            )
            ai_response = response.choices[0].message.content
            print(f"Response received: {ai_response[:50]}...")
            
            # Add to conversation memory
            conversation_memory[conv_id].append({"role": "assistant", "content": ai_response})
            
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            traceback.print_exc()
            ai_response = "I'm OrionBot and I'm having trouble right now. Please try again later."

        # Save to database
        save_to_database(conv_id, prompt, ai_response, user_email, user_name, user_id)
        
        return ChatResponse(response=ai_response)
        
    except Exception as e:
        print(f"General error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# CONVERSATION ENDPOINTS - Auth aware

@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str, current_user: User = Depends(get_current_user_optional)):
    try:
        print(f"Fetching conversations for user: {user_id}")
        
        # Eğer authenticated user varsa, kendi email'ini kullan
        if current_user:
            user_id = current_user.email
            print(f"Using authenticated user email: {user_id}")
            
        db = SessionLocal()
        
        # Get all unique conversation IDs for this user
        conversations = db.query(ChatHistory.conversation_id).filter(
            ChatHistory.user_email == user_id
        ).distinct().all()
        
        conv_list = [c[0] for c in conversations]
        titles = {}
        
        # Get first message of each conversation for title
        for conv_id in conv_list:
            first_msg = db.query(ChatHistory).filter(
                ChatHistory.conversation_id == conv_id
            ).order_by(ChatHistory.id.asc()).first()
            
            if first_msg:
                title = first_msg.user_message[:50]
                if len(first_msg.user_message) > 50:
                    title += "..."
                titles[conv_id] = title
            else:
                titles[conv_id] = "New conversation"
        
        db.close()
        
        print(f"Found {len(conv_list)} conversations for {user_id}")
        return {"conversations": conv_list, "titles": titles}
        
    except Exception as e:
        print(f"Error fetching conversations: {str(e)}")
        traceback.print_exc()
        return {"conversations": [], "titles": {}}

@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str, current_user: User = Depends(get_current_user_optional)):
    try:
        print(f"Fetching history for conversation: {conversation_id}")
        db = SessionLocal()
        
        history = db.query(ChatHistory).filter(
            ChatHistory.conversation_id == conversation_id
        ).order_by(ChatHistory.id.asc()).all()
        
        result = []
        for h in history:
            result.append({
                "user": h.user_message,
                "assistant": h.assistant_response,
                "timestamp": h.timestamp.isoformat() if h.timestamp else None
            })
        
        db.close()
        
        print(f"Found {len(result)} messages in conversation {conversation_id}")
        return {"history": result}
        
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        return {"history": []}

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: User = Depends(get_current_user_optional)):
    try:
        print(f"Deleting conversation: {conversation_id}")
        db = SessionLocal()
        
        # Delete all messages in this conversation
        deleted_count = db.query(ChatHistory).filter(
            ChatHistory.conversation_id == conversation_id
        ).delete()
        
        db.commit()
        db.close()
        
        # Remove from memory
        if conversation_id in conversation_memory:
            del conversation_memory[conversation_id]
        
        print(f"Deleted {deleted_count} messages from conversation {conversation_id}")
        return {"status": "success", "deleted": deleted_count}
        
    except Exception as e:
        print(f"Error deleting conversation: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ADMIN ENDPOINTS

@app.get("/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user_optional)):
    """Tüm kullanıcıları listele (sadece admin)"""
    if not current_user or not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        db = SessionLocal()
        users = db.query(User).all()
        db.close()
        
        return [{
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "is_admin": user.is_admin,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None
        } for user in users]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/conversations")
async def get_all_conversations(current_user: User = Depends(get_current_user_optional)):
    """Tüm konuşmaları listele (sadece admin)"""
    if not current_user or not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        db = SessionLocal()
        conversations = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).limit(100).all()
        db.close()
        
        return [{
            "id": conv.id,
            "conversation_id": conv.conversation_id,
            "user_email": conv.user_email,
            "user_message": conv.user_message[:100] + "..." if len(conv.user_message) > 100 else conv.user_message,
            "assistant_response": conv.assistant_response[:100] + "..." if len(conv.assistant_response) > 100 else conv.assistant_response,
            "timestamp": conv.timestamp.isoformat() if conv.timestamp else None
        } for conv in conversations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "OrionBot API - AI Chatbot Backend", 
        "status": "running",
        "features": ["Google Auth", "Chat History", "Admin Panel"],
        "version": "1.0.0"
    }

# Test endpoint
@app.get("/test")
def test_endpoint():
    return {
        "status": "ok", 
        "time": datetime.now().isoformat(),
        "service": "OrionBot Backend"
    }

# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "OrionBot"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting OrionBot server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)