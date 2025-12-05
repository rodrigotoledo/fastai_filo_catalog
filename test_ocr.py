#!/usr/bin/env python3
"""
Test script for OCR functionality and AI provider integrations
"""
import sys
import os
sys.path.append('/app')

from app.services.ai_service import AIService

def test_ai_providers():
    """Test different AI providers"""
    providers = ["local", "openai", "anthropic", "gemini"]

    for provider in providers:
        print(f"\n--- Testing {provider.upper()} provider ---")
        os.environ["AI_MODEL_TYPE"] = provider

        try:
            ai_service = AIService()
            if ai_service.llm:
                print(f"✓ {provider} initialized successfully")
            else:
                print(f"✗ {provider} failed to initialize (returned None)")
        except Exception as e:
            print(f"✗ {provider} error: {str(e)}")

def test_ocr():
    print("Testing OCR functionality...")

    try:
        # Test with default provider
        ai_service = AIService()
        print("✓ AIService initialized")

        # Test OCR with a sample image (using first available image)
        import os
        upload_dir = "uploads"
        if os.path.exists(upload_dir):
            images = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image = os.path.join(upload_dir, images[0])
                print(f"Testing OCR with: {test_image}")
                extracted_data = ai_service._extrair_dados_documento_simplificado(test_image)
                print(f"✓ OCR result: {extracted_data}")
            else:
                print("✗ No images found in uploads directory")
        else:
            print("✗ Uploads directory not found")

        print("\n✓ OCR test completed!")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_providers()
    test_ocr()
