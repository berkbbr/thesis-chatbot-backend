from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from datetime import datetime
from typing import List, Optional
import os
import json
import traceback

from db import SessionLocal, ChatHistory

# Seed for consistent language detection
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI(title="OrionBot API", description="AI Chatbot Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None
    user_email: str | None = None
    user_name: str | None = None
    mood: str | None = "neutral"

class ChatResponse(BaseModel):
    response: str

# Admin kontrolü için
def verify_admin(email: str):
    admin_email = os.getenv("ADMIN_EMAIL", "berkbuber95@gmail.com")
    if email != admin_email:
        raise HTTPException(status_code=403, detail="Admin access required")
    return True

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
    user_name: Optional[str] = None
):
    try:
        db = SessionLocal()
        chat_record = ChatHistory(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_response=assistant_response,
            user_email=user_email if user_email else "guest",
            timestamp=datetime.utcnow()
        )
        db.add(chat_record)
        db.commit()
        db.close()
        print(f"Saved to DB: conv_id={conversation_id}, user={user_email or 'guest'}")
    except Exception as e:
        print(f"Database error: {str(e)}")
        traceback.print_exc()

# POST /chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        print(f"Message received: '{req.message}', Conv ID: {req.conversation_id}, User: {req.user_email}")
        
        prompt = req.message
        conv_id = req.conversation_id or "default"
        user_email = req.user_email or "guest"
        user_name = req.user_name

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
                model="gpt-4",
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
        save_to_database(conv_id, prompt, ai_response, user_email, user_name)
        
        return ChatResponse(response=ai_response)
        
    except Exception as e:
        print(f"General error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# GET /conversations/{user_id}
@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str):
    try:
        print(f"Fetching conversations for user: {user_id}")
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

# GET /history/{conversation_id}
@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
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

# DELETE /conversations/{conversation_id}
@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
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
# Tüm kullanıcıları listele
@app.get("/admin/users")
async def get_all_users(admin_email: str):
    verify_admin(admin_email)
    try:
        db = SessionLocal()
        users = db.query(ChatHistory.user_email).distinct().all()
        user_list = [user[0] for user in users]
        db.close()
        return {"users": user_list, "total": len(user_list)}
    except Exception as e:
        print(f"Error fetching users: {str(e)}")
        return {"users": [], "total": 0}

# Belirli kullanıcının tüm konuşmalarını getir
@app.get("/admin/user/{user_email}/conversations")
async def get_user_conversations(user_email: str, admin_email: str):
    verify_admin(admin_email)
    try:
        db = SessionLocal()
        conversations = db.query(ChatHistory.conversation_id).filter(
            ChatHistory.user_email == user_email
        ).distinct().all()
        
        conv_list = [c[0] for c in conversations]
        db.close()
        return {"conversations": conv_list, "user": user_email, "total": len(conv_list)}
    except Exception as e:
        print(f"Error fetching user conversations: {str(e)}")
        return {"conversations": [], "user": user_email, "total": 0}

# Tüm mesajları getir (admin için)
@app.get("/admin/all-messages")
async def get_all_messages(admin_email: str, limit: int = 100, offset: int = 0):
    verify_admin(admin_email)
    try:
        db = SessionLocal()
        
        # Toplam mesaj sayısını al
        total_count = db.query(ChatHistory).count()
        
        # Mesajları al (en yeniden eskiye)
        messages = db.query(ChatHistory).order_by(
            ChatHistory.id.desc()
        ).offset(offset).limit(limit).all()
        
        result = []
        for msg in messages:
            result.append({
                "id": msg.id,
                "user_email": msg.user_email,
                "conversation_id": msg.conversation_id,
                "user_message": msg.user_message,
                "assistant_response": msg.assistant_response,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
            })
        
        db.close()
        return {
            "messages": result, 
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
    except Exception as e:
        print(f"Error fetching all messages: {str(e)}")
        return {"messages": [], "total": 0, "limit": limit, "offset": offset, "has_more": False}

# Kullanıcı istatistikleri
@app.get("/admin/stats")
async def get_admin_stats(admin_email: str):
    verify_admin(admin_email)
    try:
        db = SessionLocal()
        
        # Toplam kullanıcı sayısı
        total_users = db.query(ChatHistory.user_email).distinct().count()
        
        # Toplam mesaj sayısı
        total_messages = db.query(ChatHistory).count()
        
        # Toplam konuşma sayısı
        total_conversations = db.query(ChatHistory.conversation_id).distinct().count()
        
        # Bugünkü mesajlar
        today = datetime.utcnow().date()
        today_messages = db.query(ChatHistory).filter(
            ChatHistory.timestamp >= datetime.combine(today, datetime.min.time())
        ).count()
        
        # En aktif kullanıcılar (son 30 gün) - basitleştirilmiş
        from sqlalchemy import func
        from datetime import timedelta
        
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        active_users_query = db.query(
            ChatHistory.user_email,
            func.count(ChatHistory.id).label('message_count')
        ).filter(
            ChatHistory.timestamp >= thirty_days_ago
        ).group_by(ChatHistory.user_email).order_by(
            func.count(ChatHistory.id).desc()
        ).limit(10).all()
        
        active_users = [
            {"email": user[0], "message_count": user[1]} 
            for user in active_users_query
        ]
        
        db.close()
        
        return {
            "total_users": total_users,
            "total_messages": total_messages,
            "total_conversations": total_conversations,
            "today_messages": today_messages,
            "active_users": active_users
        }
    except Exception as e:
        print(f"Error fetching admin stats: {str(e)}")
        traceback.print_exc()
        return {
            "total_users": 0,
            "total_messages": 0,
            "total_conversations": 0,
            "today_messages": 0,
            "active_users": []
        }

# Belirli kullanıcının mesajlarını ara
@app.get("/admin/search")
async def search_messages(admin_email: str, query: str = "", user_email: str = "", limit: int = 50):
    verify_admin(admin_email)
    try:
        db = SessionLocal()
        
        filters = []
        if query:
            filters.append(ChatHistory.user_message.contains(query))
        if user_email:
            filters.append(ChatHistory.user_email == user_email)
        
        if filters:
            messages = db.query(ChatHistory).filter(*filters).order_by(
                ChatHistory.id.desc()
            ).limit(limit).all()
        else:
            messages = db.query(ChatHistory).order_by(
                ChatHistory.id.desc()
            ).limit(limit).all()
        
        result = []
        for msg in messages:
            result.append({
                "id": msg.id,
                "user_email": msg.user_email,
                "conversation_id": msg.conversation_id,
                "user_message": msg.user_message,
                "assistant_response": msg.assistant_response,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
            })
        
        db.close()
        return {"messages": result, "query": query, "user_email": user_email}
        
    except Exception as e:
        print(f"Error searching messages: {str(e)}")
        return {"messages": [], "query": query, "user_email": user_email}

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "OrionBot API - AI Chatbot Backend", 
        "status": "running",
        "features": ["Chat History", "Multi-language Support", "Admin Panel"],
        "version": "1.1.0"
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