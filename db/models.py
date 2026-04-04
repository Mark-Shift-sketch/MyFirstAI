from sqlalchemy import Column, Integer, String, Text
from db.connection import Base

class DatasetEntry(Base):
    __tablename__ = 'dataset_entries'

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)

class QAPair(Base):
    __tablename__ = 'qa_pairs'
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String(255), unique=True, index=True)
    answer = Column(Text, nullable=False)
