from app.db.database import engine
from app.models.photo import Photo

def create_tables():
    SQLModel.metadata.create_all(bind=engine)
