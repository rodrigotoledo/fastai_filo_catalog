#!/usr/bin/env python3
# test_visual_search.py - Test the new ChromaDB-based visual search
import os
import sys
import requests
import time
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_URL = "http://localhost:8000"

def test_visual_search():
    """Test the new visual search functionality"""

    print("🧪 Testing Visual Search Service")
    print("=" * 50)

    # Test 1: Get collection stats
    print("\n1. Getting collection stats...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/photos/visual-search/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Collection stats: {stats}")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

    # Test 2: Add an image to the visual search index
    print("\n2. Adding image to visual search index...")

    # First, let's upload an image via the regular API
    test_image_path = "/tmp/test_cat.jpg"
    if not os.path.exists(test_image_path):
        print("Downloading test image...")
        try:
            # Download a test image
            response = requests.get("https://loremflickr.com/400/300/cat", stream=True)
            if response.status_code == 200:
                with open(test_image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print("✅ Test image downloaded")
            else:
                print("❌ Failed to download test image")
                return
        except Exception as e:
            print(f"❌ Error downloading test image: {e}")
            return

    # Add to visual search
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_cat.jpg', f, 'image/jpeg')}
            data = {'description': 'A cute cat', 'tags': 'cat,animal,pet'}
            response = requests.post(
                f"{BASE_URL}/api/v1/photos/visual-search/add",
                files=files,
                data=data
            )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image added to visual search: {result}")
        else:
            print(f"❌ Failed to add image: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Error adding image: {e}")

    # Test 3: Search by text
    print("\n3. Testing text search...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/photos/visual-search/text",
            params={'q': 'cat', 'limit': 5}
        )

        if response.status_code == 200:
            results = response.json()
            print(f"✅ Text search results: {len(results.get('results', []))} found")
            for i, result in enumerate(results.get('results', [])[:3], 1):
                print(f"   {i}. {result.get('file_name', 'Unknown')} (similarity: {result.get('similarity', 0):.3f})")
        else:
            print(f"❌ Text search failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Error in text search: {e}")

    # Test 4: Reverse image search
    print("\n4. Testing reverse image search...")
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': ('query_cat.jpg', f, 'image/jpeg')}
            response = requests.post(
                f"{BASE_URL}/api/v1/photos/visual-search/image",
                files=files,
                data={'limit': 5}
            )

        if response.status_code == 200:
            results = response.json()
            print(f"✅ Reverse image search results: {len(results.get('results', []))} found")
            for i, result in enumerate(results.get('results', [])[:3], 1):
                print(f"   {i}. {result.get('file_name', 'Unknown')} (similarity: {result.get('similarity', 0):.3f})")
        else:
            print(f"❌ Reverse image search failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Error in reverse image search: {e}")

    # Test 5: Get updated stats
    print("\n5. Getting updated collection stats...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/photos/visual-search/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Updated collection stats: {stats}")
        else:
            print(f"❌ Failed to get updated stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting updated stats: {e}")

    # Cleanup
    if os.path.exists(test_image_path):
        os.unlink(test_image_path)
        print("\n🧹 Cleaned up test image")

    print("\n🎉 Visual search testing completed!")

if __name__ == "__main__":
    # Check if the API is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            test_visual_search()
        else:
            print("❌ API is not responding")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the API is running with: docker compose up")
