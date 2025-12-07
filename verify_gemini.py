import os
import logging
from app.services.ai_service import AIService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_integration():
    # Force Gemini model type
    os.environ["AI_MODEL_TYPE"] = "gemini"

    # Check for API key (sanity check)
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in .env.local or export it before running.")
        return

    print("🚀 Initializing AIService with Gemini...")
    try:
        ai_service = AIService()

        # Verify if the correct model was loaded
        # Note: Depending on implementation, we might need to inspect the object
        if "ChatGoogleGenerativeAI" in str(type(ai_service.llm)):
             print("✅ Gemini Model initialized successfully (ChatGoogleGenerativeAI).")
        else:
             print(f"⚠️ Warning: Model initialized is {type(ai_service.llm)}, expected ChatGoogleGenerativeAI.")

        # Test image path (using one present in the repo)
        image_path = "test_image1.jpg"

        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return

        print(f"📸 generating description for {image_path}...")
        description = ai_service._generate_description_gemini(image_path, user_description="Test verifying Gemini integration")

        print("\n✨ Generated Description:")
        print("-" * 50)
        print(description)
        print("-" * 50)

        if description and "Erro" not in description:
            print("✅ Gemini Integration verification PASSED!")
        else:
            print("❌ Gemini Integration verification FAILED (Error in description).")

    except Exception as e:
        print(f"❌ Exception during verification: {e}")

if __name__ == "__main__":
    test_gemini_integration()
