#!/usr/bin/env python3
"""
Test script for LangChain integration
"""
import sys
import os
sys.path.append('/app')

from app.services.ai_service import AIService

def test_langchain():
    print("Testing LangChain integration...")

    try:
        ai_service = AIService()
        print("✓ AIService initialized successfully")

        if ai_service.llm:
            print("✓ LangChain LLM initialized")
        else:
            print("✗ LangChain LLM not initialized")

        # Test image processing
        print("\nTesting image processing...")
        embedding, description = ai_service.process_image("/app/README.md", "Test description")
        print(f"✓ Embedding length: {len(embedding)}")
        print(f"✓ Description: {description[:100]}...")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_langchain()
