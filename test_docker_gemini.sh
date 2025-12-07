#!/bin/bash

# Load variables from .env.local if it exists
if [ -f .env.local ]; then
    echo "Loading environment from .env.local..."
    export $(grep -v '^#' .env.local | xargs)
fi

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ Error: GOOGLE_API_KEY not found in .env.local"
    echo "Please add GOOGLE_API_KEY=your_key to .env.local"
    exit 1
fi

echo "🚀 Running verification inside Docker container..."
echo "Container: photo_finder_app"
echo "Passing AI_MODEL_TYPE=gemini and API Key..."

# Run the script inside the container with explicit env vars
docker exec -e GOOGLE_API_KEY="$GOOGLE_API_KEY" -e AI_MODEL_TYPE="gemini" photo_finder_app python verify_full_stack.py
