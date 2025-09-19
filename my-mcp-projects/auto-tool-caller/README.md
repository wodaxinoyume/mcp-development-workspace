# Auto Tool Caller API

REST API for Metropolitan Museum of Art Intelligent Assistant

## Files

- `main.py` - FastAPI backend server
- `index.html` - Frontend web interface

## Quick Start

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn mcp-agent[anthropic]
   ```

2. Configure API key in `mcp_agent.secrets.yaml`

3. Start the server:
   ```bash
   python main.py
   ```

4. Open `index.html` in your browser

## API Endpoints

- `POST /chat` - Send message to AI assistant
- `GET /health` - Health check
- `GET /docs` - API documentation

## Usage

### Frontend
Open `index.html` in your browser and start chatting with the AI assistant.

### API Call
```javascript
fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: "Search for Van Gogh paintings"})
})
.then(response => response.json())
.then(data => console.log(data.response));
```