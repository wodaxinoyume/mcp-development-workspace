#!/usr/bin/env python3
"""
Auto Tool Caller API

REST API for Metropolitan Museum of Art Intelligent Assistant

Usage:
1. Install dependencies: pip install fastapi uvicorn mcp-agent[anthropic]
2. Configure API key in mcp_agent.secrets.yaml
3. Run: python api_main.py
4. Access API docs at: http://localhost:8000/docs
"""

import asyncio
import os
import sys
import base64
import re
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 直接引用本地 mcp-agent 源码
import sys
from pathlib import Path

# 添加本地 mcp-agent 源码路径到 Python 路径
local_mcp_agent_path = (Path(__file__).parent / ".." / ".." / "mcp-agent" / "src").resolve()
sys.path.insert(0, str(local_mcp_agent_path))

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

# Global variables for app and agent
app_instance: Optional[MCPApp] = None
agent_instance: Optional[Agent] = None
llm_instance: Optional[AnthropicAugmentedLLM] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str
    success: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    agent_initialized: bool


class AutoToolCallerAPI:
    """Auto Tool Caller API - REST API wrapper for MCP Agent"""
    
    def __init__(self):
        self.app = MCPApp(name="auto_tool_caller_api")
        self.agent: Optional[Agent] = None
        self.llm: Optional[AnthropicAugmentedLLM] = None
        self.app_context = None
    
    async def initialize(self):
        """Initialize Agent and LLM"""
        global app_instance, agent_instance, llm_instance
        
        async with self.app.run() as agent_app:
            self.app_context = agent_app
            app_instance = self.app
            
            logger = agent_app.logger
            
            # Create intelligent Agent that can access Metropolitan Museum MCP server
            self.agent = Agent(
                name="met_museum_assistant",
                instruction="""
                You are an intelligent assistant for the Metropolitan Museum of Art, specialized in helping users search and learn about artwork information.
                You can use the following tools:
                - Metropolitan Museum search tools: Search for artworks, artists, exhibitions, etc.
                - Get detailed artwork information: Including descriptions, historical background, creation dates, etc.
                - Browse museum collections: Explore artworks from different periods, styles, and cultures

                Please intelligently select and call appropriate tools based on user needs to complete tasks. If user needs require multiple steps, please execute them in order.""",
                server_names=["met-museum"]  # Specify available MCP servers
            )
            
            async with self.agent:
                # Attach LLM to Agent
                self.llm = await self.agent.attach_llm(AnthropicAugmentedLLM)
                
                # Set global instances
                agent_instance = self.agent
                llm_instance = self.llm
                
                logger.info("Auto Tool Caller API initialized successfully")
    
    async def process_message(self, message: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Process user message and return response"""
        if not self.llm:
            raise HTTPException(status_code=500, detail="LLM not initialized")
        
        try:
            result = await self.llm.generate_str_with_tool_results(                
                message=message,
                request_params=RequestParams(
                    maxTokens=max_tokens,
                    temperature=temperature
                )
            )
            return result
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


# Global API instance
api_instance: Optional[AutoToolCallerAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global api_instance
    
    # Startup
    logger.info("Starting Auto Tool Caller API...")
    
    # Check configuration files
    config_file = "mcp_agent.config.yaml"
    secrets_file = "mcp_agent.secrets.yaml"
    
    if not os.path.exists(config_file):
        logger.error(f"Configuration file {config_file} does not exist")
        raise Exception(f"Configuration file {config_file} does not exist")
    
    if not os.path.exists(secrets_file):
        logger.error(f"Secrets file {secrets_file} does not exist")
        raise Exception(f"Secrets file {secrets_file} does not exist")
    
    # Check if API key is configured
    with open(secrets_file, 'r') as f:
        content = f.read()
        if "your-anthropic-api-key-here" in content:
            logger.error("Please configure your Anthropic API key in mcp_agent.secrets.yaml")
            raise Exception("Please configure your Anthropic API key in mcp_agent.secrets.yaml")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize API
    api_instance = AutoToolCallerAPI()
    await api_instance.initialize()
    
    logger.info("Auto Tool Caller API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Auto Tool Caller API...")


# Create FastAPI app
app = FastAPI(
    title="Auto Tool Caller API",
    description="REST API for Metropolitan Museum of Art Intelligent Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Auto Tool Caller API",
        "description": "REST API for Metropolitan Museum of Art Intelligent Assistant",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global agent_instance, llm_instance
    
    return HealthResponse(
        status="healthy" if agent_instance and llm_instance else "unhealthy",
        message="Auto Tool Caller API is running" if agent_instance and llm_instance else "API not initialized",
        agent_initialized=bool(agent_instance and llm_instance)
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - Process user message and return AI response"""
    global api_instance
    
    if not api_instance:
        raise HTTPException(status_code=500, detail="API not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Process the message
        response = await api_instance.process_message(
            message=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response,
            session_id=request.session_id or "default",
            success=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        return ChatResponse(
            response="",
            session_id=request.session_id or "default",
            success=False,
            error=str(e)
        )


@app.get("/capabilities")
async def get_capabilities():
    """Get available capabilities"""
    return {
        "capabilities": [
            "Search for artworks and artists",
            "Get detailed information about artworks",
            "Browse museum collections",
            "Explore art from different periods and styles",
            "Learn about art history and cultural background"
        ],
        "tools": [
            "Metropolitan Museum search tools",
            "Artwork information retrieval",
            "Collection browsing",
            "Art history analysis"
        ]
    }


@app.get("/status")
async def get_status():
    """Get API status and configuration"""
    global agent_instance, llm_instance
    
    return {
        "api_status": "running",
        "agent_initialized": bool(agent_instance),
        "llm_initialized": bool(llm_instance),
        "mcp_servers": ["met-museum"],
        "version": "1.0.0"
    }


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
