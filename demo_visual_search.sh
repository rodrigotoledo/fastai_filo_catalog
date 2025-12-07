#!/bin/bash
# demo_visual_search.sh - Demonstration of the new ChromaDB-based visual search

echo "🎨 Photo Finder - Visual Search Demo"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8000"

# Function to check if API is running
check_api() {
    echo -e "${BLUE}Checking if API is running...${NC}"
    if curl -s "$BASE_URL/docs" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is running${NC}"
        return 0
    else
        echo -e "${RED}❌ API is not running${NC}"
        echo -e "${YELLOW}Please start the services with: docker compose up -d${NC}"
        return 1
    fi
}

# Function to show processing stats (since visual search is integrated)
show_stats() {
    echo -e "\n${BLUE}📊 Photo Processing Stats:${NC}"
    response=$(curl -s "$BASE_URL/api/v1/photos/processing/stats")
    if [ $? -eq 0 ]; then
        echo "$response" | python3 -m json.tool
    else
        echo -e "${RED}Failed to get stats${NC}"
    fi
}

# Function to add a sample image
add_sample_image() {
    echo -e "\n${BLUE}🖼️ Adding sample image to visual search...${NC}"

    # Download a sample image
    echo "Downloading sample image..."
    curl -s -L "https://loremflickr.com/640/480/cat" -o /tmp/demo_cat.jpg

    if [ ! -f /tmp/demo_cat.jpg ]; then
        echo -e "${RED}Failed to download sample image${NC}"
        return 1
    fi

    # Upload image (which automatically adds to visual search)
    response=$(curl -s -X POST \
        -F "files=@/tmp/demo_cat.jpg" \
        -F "description=A beautiful cat in a garden" \
        "$BASE_URL/api/v1/photos/upload")

    if echo "$response" | grep -q "filename"; then
        echo -e "${GREEN}✅ Image uploaded and added to visual search successfully!${NC}"
        echo "$response" | python3 -m json.tool
    else
        echo -e "${RED}❌ Failed to upload image${NC}"
        echo "$response"
    fi

    # Cleanup
    rm -f /tmp/demo_cat.jpg
}

# Function to search by text
search_by_text() {
    local query="$1"
    echo -e "\n${BLUE}🔍 Searching for: '$query'${NC}"

    response=$(curl -s "$BASE_URL/api/v1/photos/search/text?q=$query&page=1&page_size=5")

    if echo "$response" | grep -q "results"; then
        echo -e "${GREEN}✅ Search completed!${NC}"
        echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Query: {data.get(\"query\", \"Unknown\")}')
print(f'Total results: {data.get(\"total_found\", 0)}')
for i, result in enumerate(data.get('results', [])[:3], 1):
    print(f'{i}. {result.get(\"file_name\", \"Unknown\")} (similarity: {result.get(\"similarity\", 0):.3f})')
    print(f'   Caption: {result.get(\"caption\", \"\")[:100]}...')
        "
    else
        echo -e "${RED}❌ Search failed${NC}"
        echo "$response"
    fi
}

# Function to demonstrate reverse image search
reverse_image_search() {
    echo -e "\n${BLUE}🔄 Testing reverse image search...${NC}"

    # Download another cat image for testing
    echo "Downloading query image..."
    curl -s -L "https://loremflickr.com/320/240/cat" -o /tmp/query_cat.jpg

    if [ ! -f /tmp/query_cat.jpg ]; then
        echo -e "${RED}Failed to download query image${NC}"
        return 1
    fi

    response=$(curl -s -X POST \
        -F "file=@/tmp/query_cat.jpg" \
        -F "page=1" \
        -F "page_size=5" \
        "$BASE_URL/api/v1/photos/search/image")

    if echo "$response" | grep -q "results"; then
        echo -e "${GREEN}✅ Reverse image search completed!${NC}"
        echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Total results: {len(data.get(\"results\", []))}')
for i, result in enumerate(data.get('results', [])[:3], 1):
    photo = result.get('photo', {})
    print(f'{i}. {photo.get(\"original_filename\", \"Unknown\")} (similarity: {result.get(\"similarity_score\", 0):.3f})')
        "
    else
        echo -e "${RED}❌ Reverse image search failed${NC}"
        echo "$response"
    fi

    # Cleanup
    rm -f /tmp/query_cat.jpg
}

# Main demo flow
main() {
    if ! check_api; then
        exit 1
    fi

    echo -e "\n${YELLOW}🚀 Starting Visual Search Demo${NC}"
    echo -e "${YELLOW}================================${NC}"

    # Clear any existing visual search data
    echo -e "\n${BLUE}🧹 Clearing visual search index...${NC}"
    response=$(curl -s -X DELETE "$BASE_URL/api/v1/photos/clear")
    if echo "$response" | grep -q "cleared"; then
        echo -e "${GREEN}✅ Index cleared${NC}"
    else
        echo -e "${YELLOW}⚠️ Could not clear index (might be empty)${NC}"
    fi

    # Show initial stats

    # Add a sample image
    add_sample_image

    # Show updated stats
    show_stats

    # Test text search
    search_by_text "cat"
    search_by_text "animal with fur"
    search_by_text "garden outdoors"

    # Test reverse image search
    reverse_image_search

    echo -e "\n${GREEN}🎉 Demo completed!${NC}"
    echo -e "\n${BLUE}Try these commands to explore more:${NC}"
    echo "• curl '$BASE_URL/api/v1/photos/search/text?q=your+query'"
    echo "• curl -X POST -F 'file=@image.jpg' '$BASE_URL/api/v1/photos/search/image'"
    echo "• docker compose exec app python test_visual_search.py"
}

# Run main function
main "$@"
