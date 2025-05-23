
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Veritabanı bağlantısı (SQLite)
SQLALCHEMY_DATABASE_URL = "sqlite:///./chatlog.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Sohbet tablosu modeli
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True)
    user_message = Column(Text)
    assistant_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_email = Column(String, nullable=True)  # Kullanıcı emaili/id

# Tabloyu veritabanında oluştur
Base.metadata.create_all(bind=engine)