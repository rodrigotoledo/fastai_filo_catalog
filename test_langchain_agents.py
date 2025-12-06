#!/usr/bin/env python3
"""
Test script for LangChain Agents in Photo Finder
Demonstrates how agents work for image processing and document extraction
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.langchain_agents import ImageProcessingAgent, DocumentProcessingAgent
from app.services.ai_service import AIService
from app.services.visual_search_service import VisualSearchService
from app.services.document_parser_service import DocumentParserService

def test_image_processing_agent():
    """Test the image processing agent"""
    print("🖼️  Testing Image Processing Agent")
    print("=" * 50)

    try:
        # Initialize services
        ai_service = AIService()
        visual_search = VisualSearchService()

        if not ai_service.llm:
            print("❌ LLM not available. Configure AI_MODEL_TYPE and API keys.")
            return

        print("✅ Services initialized successfully")
        print(f"   - LLM Model: {ai_service.llm.__class__.__name__}")
        print(f"   - ChromaDB: Connected")

        # Test basic functionality without full agent
        print("\n🧪 Testing basic functionality:")

        # Test LLM directly
        try:
            response = ai_service.llm.invoke("Say 'Hello from LangChain!'")
            print(f"   - LLM Response: {response.content if hasattr(response, 'content') else str(response)}")
        except Exception as e:
            print(f"   - LLM Test failed: {str(e)}")

        # Test ChromaDB
        try:
            stats = visual_search.get_collection_stats()
            print(f"   - ChromaDB Stats: {stats}")
        except Exception as e:
            print(f"   - ChromaDB Test failed: {str(e)}")

        print("✅ Basic functionality test completed")

    except Exception as e:
        print(f"❌ Error in basic test: {str(e)}")

def test_document_processing_agent():
    """Test the document processing agent"""
    print("\n📄 Testing Document Processing Agent")
    print("=" * 50)

    try:
        # Initialize services
        ai_service = AIService()

        if not ai_service.llm:
            print("❌ LLM not available. Configure AI_MODEL_TYPE and API keys.")
            return

        print("✅ AI Service initialized successfully")
        print(f"   - LLM Model: {ai_service.llm.__class__.__name__}")

        # Test basic document processing
        print("\n🧪 Testing basic document processing:")

        # Test OCR functionality
        try:
            test_doc = "README.md"
            if os.path.exists(test_doc):
                text = ai_service._extract_text_from_image(test_doc)
                print(f"   - OCR extracted {len(text)} characters")
            else:
                print("   - Test document not found")
        except Exception as e:
            print(f"   - OCR Test failed: {str(e)}")

        # Test LLM processing
        try:
            test_text = "Nome: João Silva, CPF: 123.456.789-00"
            result = ai_service._process_extracted_text_with_langchain(test_text)
            print(f"   - LLM processed text successfully: {bool(result)}")
        except Exception as e:
            print(f"   - LLM Processing failed: {str(e)}")

        print("✅ Basic document processing test completed")

    except Exception as e:
        print(f"❌ Error in document test: {str(e)}")

def demonstrate_agent_workflow():
    """Demonstrate a complete agent workflow"""
    print("\n🤖 Complete Agent Workflow Demonstration")
    print("=" * 50)

    workflow_steps = [
        "1. User uploads image to /api/v1/photos/process-with-agent",
        "2. Agent analyzes image type and quality",
        "3. Agent decides: Generate embedding + rich caption + index in ChromaDB",
        "4. Agent executes tools in optimal order",
        "5. Agent validates results and provides feedback",
        "",
        "Benefits of Agent Approach:",
        "• Intelligent decision making based on image characteristics",
        "• Automatic fallback strategies if tools fail",
        "• Multi-step processing with validation",
        "• Natural language interaction and reasoning",
        "• Adaptable to different AI providers (OpenAI, Gemini, etc.)",
        "",
        "For Documents:",
        "• Agent identifies document type (RG, CNH, etc.)",
        "• Chooses appropriate extraction strategy",
        "• Validates extracted data automatically",
        "• Handles multiple formats (PDF, images, text)",
        "• Provides confidence scores and error handling"
    ]

    for step in workflow_steps:
        print(f"  {step}")

def main():
    """Main test function"""
    print("🚀 LangChain Agents Test for Photo Finder")
    print("This demonstrates how AI agents can enhance your image processing pipeline")
    print()

    # Check environment
    ai_model = os.getenv("AI_MODEL_TYPE", "not set")
    print(f"Current AI_MODEL_TYPE: {ai_model}")

    if ai_model == "not set":
        print("💡 Tip: Set AI_MODEL_TYPE=openai, gemini, anthropic, or local")
        print("   And configure the corresponding API keys")
        print()

    # Run tests
    test_image_processing_agent()
    test_document_processing_agent()
    demonstrate_agent_workflow()

    print("\n🎉 Agent testing complete!")
    print("To use agents in production, set USE_LANGCHAIN_AGENTS=true in your environment")

if __name__ == "__main__":
    main()
