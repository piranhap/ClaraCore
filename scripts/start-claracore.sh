#!/bin/bash
# Quick Start Script for ClaraCore using Docker
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}ClaraCore Docker Quick Start${NC}"
echo -e "${BLUE}============================${NC}"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker to use this script.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start the Docker daemon.${NC}"
    exit 1
fi

# Detect OS and choose compose file
OS="$(uname)"
COMPOSE_FILE=""
if [[ "$OS" == "Linux" ]]; then
    # Simple check for nvidia-smi for CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected. Using CUDA compose file.${NC}"
        COMPOSE_FILE="docker-cuda/docker-compose.yml"
    else
        echo -e "${YELLOW}ⓘ No NVIDIA GPU detected. Using CPU compose file.${NC}"
        COMPOSE_FILE="docker-cpu/docker-compose.yml"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    echo -e "${GREEN}✓ macOS detected. Using CPU compose file (for now).${NC}"
    COMPOSE_FILE="docker-cpu/docker-compose.yml"
else
    echo -e "${RED}❌ Unsupported operating system for this Docker script: $OS${NC}"
    exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo -e "${RED}❌ Compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Starting ClaraCore using Docker...${NC}"
docker compose -f "$COMPOSE_FILE" up -d --build

# Wait for container to be healthy
echo ""
echo -e "${BLUE}Waiting for ClaraCore container to initialize...${NC}"
sleep 5

# Check if accessible
MAX_ATTEMPTS=20
ATTEMPT=0
RUNNING=false
PORT="5890" # Match the port in docker-compose

echo -n "  Checking"

while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
    if command -v curl >/dev/null 2>&1; then
        if curl -s -f "http://localhost:$PORT/" >/dev/null 2>&1; then
            RUNNING=true
            break
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -q -O /dev/null "http://localhost:$PORT/" 2>/dev/null; then
            RUNNING=true
            break
        fi
    fi
    
    echo -n "."
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done

echo ""
echo ""

if [[ "$RUNNING" == true ]]; then
    echo -e "${GREEN}✅ ClaraCore is running in Docker and accessible!${NC}"
    echo ""
    echo -e "${BLUE}┌─────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│  Open your browser and visit:          │${NC}"
    echo -e "${BLUE}│                                         │${NC}"
    echo -e "${CYAN}│  http://localhost:$PORT/ui/              │${NC}"
    echo -e "${BLUE}│                                         │${NC}"
    echo -e "${BLUE}└─────────────────────────────────────────┘${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ ClaraCore container may still be starting up...${NC}"
    echo ""
    echo -e "${CYAN}Try accessing the web interface in a moment:${NC}"
    echo -e "  ${CYAN}http://localhost:$PORT/ui/${NC}"
    echo ""
    echo -e "${CYAN}To check container status:${NC}"
    echo -e "  ${CYAN}docker compose -f $COMPOSE_FILE ps${NC}"
    echo ""
    echo -e "${CYAN}To view container logs:${NC}"
    echo -e "  ${CYAN}docker compose -f $COMPOSE_FILE logs -f${NC}"
    echo ""
fi