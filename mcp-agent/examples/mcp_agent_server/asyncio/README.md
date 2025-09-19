# MCP Agent Server Example (Asyncio)

This example is an mcp-agent application that is exposed as an MCP server, aka the "MCP Agent Server".

The MCP Agent Server exposes agentic workflows as MCP tools.

It shows how to build, run, and connect to an MCP server using the asyncio execution engine.

https://github.com/user-attachments/assets/f651af86-222d-4df0-8241-616414df66e4

## Concepts Demonstrated

- Creating workflows with the `Workflow` base class
- Registering workflows with an `MCPApp`
- Exposing workflows as MCP tools using `create_mcp_server_for_app`, optionally using custom FastMCP settings
- Preferred: Declaring MCP tools with `@app.tool` and `@app.async_tool`
- Connecting to an MCP server using `gen_client`
- Running workflows remotely and monitoring their status

## Preferred: Define tools with decorators

You can declare tools directly from plain Python functions using `@app.tool` (sync) and `@app.async_tool` (async). This is the simplest and recommended way to expose agent logic.

```python
from mcp_agent.app import MCPApp
from typing import Optional

app = MCPApp(name="basic_agent_server")

# Synchronous tool – returns the final result to the caller
@app.tool
async def grade_story(story: str, app_ctx: Optional[Context] = None) -> str:
    """
    Grade a student's short story and return a structured report.
    """
    # ... implement using your agents/LLMs ...
    return "Report..."

# Asynchronous tool – starts a workflow and returns IDs to poll later
@app.async_tool(name="grade_story_async")
async def grade_story_async(story: str, app_ctx: Optional[Context] = None) -> str:
    """
    Start grading the story asynchronously.

    This tool starts the workflow and returns 'workflow_id' and 'run_id'. Use the
    generic 'workflows-get_status' tool with the returned IDs to retrieve status/results.
    """
    # ... implement using your agents/LLMs ...
    return "(async run)"
```

What gets exposed:

- Sync tools appear as `<tool_name>` and return the final result (no status polling needed).
- Async tools appear as `<tool_name>` and return `{"workflow_id","run_id"}`; use `workflows-get_status` to query status.

These decorator-based tools are registered automatically when you call `create_mcp_server_for_app(app)`.

## Components in this Example

1. **BasicAgentWorkflow**: A simple workflow that demonstrates basic agent functionality:

   - Connects to external servers (fetch, filesystem)
   - Uses LLMs (Anthropic Claude) to process input
   - Supports multi-turn conversations
   - Demonstrates model preference configuration

2. **ParallelWorkflow**: A more complex workflow that shows parallel agent execution:
   - Uses multiple specialized agents (proofreader, fact checker, style enforcer)
   - Processes content using a fan-in/fan-out pattern
   - Aggregates results into a final report

## Available Endpoints

The MCP agent server exposes the following tools:

- `workflows-list` - Lists available workflows and their parameter schemas
- `workflows-get_status` - Get status for a running workflow by `run_id` (and optional `workflow_id`)
- `workflows-cancel` - Cancel a running workflow

If you use the preferred decorator approach:

- Sync tool: `grade_story` (returns final result)
- Async tool: `grade_story_async` (returns `workflow_id/run_id`; poll with `workflows-get_status`)

The workflow-based endpoints (e.g., `workflows-<Workflow>-run`) are still available when you define explicit workflow classes.

## Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) package manager
- API keys for Anthropic and OpenAI

## Configuration

Before running the example, you'll need to configure the necessary paths and API keys.

### API Keys

1. Copy the example secrets file:

```
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
```

2. Edit `mcp_agent.secrets.yaml` to add your API keys:

```
anthropic:
  api_key: "your-anthropic-api-key"
openai:
  api_key: "your-openai-api-key"
```

## How to Run

### Using the Client Script

The simplest way to run the example is using the provided client script:

```
# Make sure you're in the mcp_agent_server/asyncio directory
uv run client.py
```

This will:

1. Start the basic_agent_server.py as a subprocess
2. Connect to the server
3. Run the BasicAgentWorkflow
4. Monitor and display the workflow status

### Running the Server and Client Separately

You can also run the server and client separately:

1. In one terminal, start the server:

```
uv run basic_agent_server.py

# Optionally, run with the example custom FastMCP settings
uv run basic_agent_server.py --custom-fastmcp-settings
```

2. In another terminal, run the client:

```
uv run client.py

# Optionally, run with the example custom FastMCP settings
uv run client.py --custom-fastmcp-settings
```

## Receiving Server Logs in the Client

The server advertises the `logging` capability (via `logging/setLevel`) and forwards its structured logs upstream using `notifications/message`. To receive these logs in a client session, pass a `logging_callback` when constructing the client session and set the desired level:

```python
from datetime import timedelta
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.types import LoggingMessageNotificationParams
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

async def on_server_log(params: LoggingMessageNotificationParams) -> None:
    print(f"[SERVER LOG] [{params.level.upper()}] [{params.logger}] {params.data}")

def make_session(read_stream: MemoryObjectReceiveStream,
                 write_stream: MemoryObjectSendStream,
                 read_timeout_seconds: timedelta | None) -> ClientSession:
    return MCPAgentClientSession(
        read_stream=read_stream,
        write_stream=write_stream,
        read_timeout_seconds=read_timeout_seconds,
        logging_callback=on_server_log,
    )

# Later, when connecting via gen_client(..., client_session_factory=make_session)
# you can request the minimum server log level:
# await server.set_logging_level("info")
```

The example client (`client.py`) demonstrates this end-to-end: it registers a logging callback and calls `set_logging_level("info")` so logs from the server appear in the client's console.

## MCP Clients

Since the mcp-agent app is exposed as an MCP server, it can be used in any MCP client just
like any other MCP server.

### MCP Inspector

You can inspect and test the server using [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```
npx @modelcontextprotocol/inspector \
  uv \
  --directory /path/to/mcp-agent/examples/mcp_agent_server/asyncio \
  run \
  basic_agent_server.py
```

This will launch the MCP Inspector UI where you can:

- See all available tools
- Test workflow execution
- View request/response details

### Claude Desktop

To use this server with Claude Desktop:

1. Locate your Claude Desktop configuration file (usually in `~/.claude-desktop/config.json`)

2. Add a new server configuration:

```json
"basic-agent-server": {
  "command": "/path/to/uv",
  "args": [
    "--directory",
    "/path/to/mcp-agent/examples/mcp_agent_server/asyncio",
    "run",
    "basic_agent_server.py"
  ]
}
```

3. Restart Claude Desktop, and you'll see the server available in the tool drawer

4. (**claude desktop workaround**) Update `mcp_agent.config.yaml` file with the full paths to npx/uvx on your system:

Find the full paths to `uvx` and `npx` on your system:

```
which uvx
which npx
```

Update the `mcp_agent.config.yaml` file with these paths:

```yaml
mcp:
  servers:
    fetch:
      command: "/full/path/to/uvx" # Replace with your path
      args: ["mcp-server-fetch"]
    filesystem:
      command: "/full/path/to/npx" # Replace with your path
      args: ["-y", "@modelcontextprotocol/server-filesystem"]
```

## Code Structure

- `basic_agent_server.py` - Defines the workflows and creates the MCP server
- `client.py` - Example client that connects to the server and runs workflows
- `mcp_agent.config.yaml` - Configuration for MCP servers and execution engine
- `mcp_agent.secrets.yaml` - Contains API keys (not included in repository)
- `short_story.md` - Sample content for testing the ParallelWorkflow

## Understanding the Workflow System

### Workflow Definition

Workflows are defined by subclassing the `Workflow` base class and implementing the `run` method:

```python
@app.workflow
class BasicAgentWorkflow(Workflow[str]):
    @app.workflow_run
    async def run(self, input: str) -> WorkflowResult[str]:
        # Workflow implementation...
        return WorkflowResult(value=result)
```

### Server Creation

The server is created using the `create_mcp_server_for_app` function:

```python
mcp_server = create_mcp_server_for_app(agent_app)
await mcp_server.run_stdio_async()
```

Similarly, you can launch the server over SSE, Websocket or Streamable HTTP transports.

### Client Connection

The client connects to the server using the `gen_client` function:

```python
async with gen_client("basic_agent_server", context.server_registry) as server:
    # Call server tools
    workflows_response = await server.call_tool("workflows-list", {})
    run_result = await server.call_tool(
        "workflows-BasicAgentWorkflow-run",
        arguments={"run_parameters": {"input": "..."}}
    )
```
