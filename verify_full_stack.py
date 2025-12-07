import asyncio
import os
import shutil
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import UploadFile
from app.services.photo_service import PhotoService
from app.db.database import DATABASE_URL
from typing import BinaryIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock UploadFile for testing (Simple object matching interface)
class MockUploadFile:
    def __init__(self, filename: str, file: BinaryIO, content_type: str = "image/jpeg"):
        self.filename = filename
        self.content_type = content_type
        self.file = file

    async def read(self, size: int = -1):
        return self.file.read(size)

    async def seek(self, offset: int):
        self.file.seek(offset)

async def verify_full_stack():
    # Setup DB Connection
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        service = PhotoService(db)

        # 1. Prepare Test Image
        test_image_path = "test_image1.jpg"
        if not os.path.exists(test_image_path):
            logger.error(f"Test image {test_image_path} not found")
            return

        with open(test_image_path, "rb") as f:
            mock_file = MockUploadFile(filename="test_upload_gemini.jpg", file=f)

            logger.info("🚀 1. Uploading photo to trigger Gemini Ingestion...")
            photo = await service.save_photo(mock_file, user_description="Integration Test Photo")

            logger.info(f"✅ Photo uploaded with ID: {photo.id}")
            logger.info(f"   Original Filename: {photo.original_filename}")

        # 2. Verify AI Processing
        logger.info("\n🔍 2. Verifying AI Processing in DB...")

        # Reload from DB to ensure persistence
        db.refresh(photo)

        if photo.description:
            logger.info("✅ Gemini Description generated:")
            logger.info(f"   {photo.description[:100]}...")
        else:
            logger.error("❌ NO Description generated!")

        if photo.embedding:
             logger.info(f"✅ Embedding generated (Dimension: {len(photo.embedding)})")
        else:
             logger.error("❌ NO Embedding generated!")

        # 3. Verify Semantic Search
        if photo.embedding:
            logger.info("\n🔎 3. Verifying Semantic Search...")
            query = "Integration Test Photo" # Should match exactly

            search_result = service.search_similar_photos(query_text=query, limit=5)

            found = False
            for res in search_result['results']:
                p = res['photo']
                logger.info(f"   Found match: ID={p.id} | Desc={p.user_description}")
                if p.id == photo.id:
                    found = True

            if found:
                logger.info("✅ SUCCESS: Uploaded photo found via semantic search!")
            else:
                 logger.warning("⚠️ Photo uploaded but NOT found in search results immediately (might be index latency or similarity threshold).")

    except Exception as e:
        logger.error(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(verify_full_stack())
