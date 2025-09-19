import asyncio
import json
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncGenerator, Optional

import mcp.types as types
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl, BaseModel

from mcp_agent.cli.utils.ux import (
    console,
    print_error,
    print_success,
)

DEFAULT_CLIENT_INFO = types.Implementation(name="mcp", version="0.1.0")


class Workflow(BaseModel):
    """An workflow definition that the server is capable of running."""

    name: str
    """A human-readable name for this resource."""

    description: Optional[str | None] = None
    """A description of what this resource represents."""

    capabilities: Optional[list[str]] = []
    """A list of capabilities that this workflow provides. E.g. 'run', 'resume', 'cancel', 'get_status'."""

    tool_endpoints: Optional[list[str]] = []
    """A list of tool endpoints that this workflow can call. E.g. 'workflows-{name}-run'."""

    run_parameters: Optional[dict[str, Any]] = {}


class ListWorkflowsResult(BaseModel):
    """Processed server response to a workflows-list request from the client."""

    workflows: list[Workflow]


class WorkflowRunState(BaseModel):
    """The current state of a workflow run."""

    status: str
    """The current status of the workflow run, e.g. 'running', 'completed', 'failed'."""

    metadata: dict
    """Metadata associated with the workflow run state."""

    updated_at: float
    """The time when the workflow run state was last updated."""

    error: Optional[str] = None
    """An error message if the workflow run failed, otherwise None."""


class WorkflowRunResult(BaseModel):
    """The result of a workflow run."""

    value: str
    """The value returned by the workflow run, if any."""

    metadata: dict
    """Metadata associated with the workflow run result."""

    start_time: Optional[float] = None
    """The time when the workflow run started."""

    end_time: Optional[float] = None
    """The time when the workflow run ended, if applicable."""


class WorkflowRunTemporal(BaseModel):
    """Temporal-specific metadata for a workflow run."""

    id: str
    """Identifier for this workflow instance."""

    workflow_id: str
    """Identifier for the workflow instance being run."""

    run_id: str
    """Identifier for this specific run of the workflow instance."""

    status: str
    """The temporal status of this workflow run."""

    error: Optional[str] = None
    """An error message if the workflow run failed."""

    start_time: Optional[float] = None
    """The time when the workflow run started."""

    close_time: Optional[float] = None
    """The time when the workflow run completed."""


class WorkflowRun(BaseModel):
    """An execution instance of a workflow definition."""

    id: str
    """A unique identifier for this run of the workflow."""

    name: str
    """The name/type for the Workflow Definition being run."""

    status: str
    """The temporal status for this run of the workflow."""

    running: bool
    """Whether this run of the workflow is currently running."""

    state: Optional[WorkflowRunState] = None
    """The current state of the workflow run."""

    result: Optional[WorkflowRunResult] = None
    """The result of the workflow run, if it has completed."""

    completed: Optional[bool] = False
    """Whether this run of the workflow has completed."""

    error: Optional[str] = None
    """An error message if the workflow run failed."""

    temporal: Optional[WorkflowRunTemporal] = None
    """The temporal state of this workflow run, if applicable."""


class ListWorkflowRunsResult(BaseModel):
    """Processed server response to a workflows-runs-list request from the client."""

    workflow_runs: list[WorkflowRun]


class MCPClientSession(ClientSession):
    """MCP Client Session with additional support for mcp-agent functionality."""

    async def list_workflows(self) -> ListWorkflowsResult:
        """Send a workflows-list request."""
        workflows_response = await self.call_tool("workflows-list", {})
        if workflows_response.isError:
            error_message = (
                workflows_response.content[0].text
                if len(workflows_response.content) > 0
                and workflows_response.content[0].type == "text"
                else "Error listing workflows"
            )
            raise Exception(error_message)

        workflows = []
        for item in workflows_response.content:
            if isinstance(item, types.TextContent):
                # Assuming the content is a JSON string representing a Workflow item dict
                try:
                    workflow_data = json.loads(item.text)
                    for value in workflow_data.values():
                        workflows.append(
                            Workflow(
                                **value,
                            )
                        )
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid workflow data: {e}")

        return ListWorkflowsResult(workflows=workflows)

    async def list_workflow_runs(self) -> ListWorkflowRunsResult:
        """Send a workflows-runs-list request."""
        runs_response = await self.call_tool("workflows-runs-list", {})
        if runs_response.isError:
            error_message = (
                runs_response.content[0].text
                if len(runs_response.content) > 0
                and runs_response.content[0].type == "text"
                else "Error listing workflow runs"
            )
            raise Exception(error_message)

        runs = []
        for item in runs_response.content:
            if isinstance(item, types.TextContent):
                # Assuming the content is a JSON string representing a WorkflowRun item dict
                try:
                    run_data = json.loads(item.text)
                    runs.append(WorkflowRun(**run_data))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid workflow run data: {e}")

        return ListWorkflowRunsResult(workflow_runs=runs)


class TransportType(Enum):
    """Transport types for MCP client-server communication."""

    SSE = "SSE"
    STREAMABLE_HTTP = "STREAMABLE_HTTP"


class MCPClient:
    """MCP Client for interacting with the MCP App server."""

    def __init__(
        self,
        server_url: AnyUrl,
        api_key: str | None = None,
        transport_type: TransportType = TransportType.STREAMABLE_HTTP,
    ) -> None:
        self._api_key = api_key
        self.server_url = server_url
        self.transport_type = transport_type

    def _create_client(self):
        kwargs = {
            "url": str(self.server_url),
            "headers": {
                "Authorization": (f"Bearer {self._api_key}" if self._api_key else None),
            },
        }
        if self.transport_type == TransportType.STREAMABLE_HTTP:
            kwargs = {
                **kwargs,
                "terminate_on_close": True,
            }
            return streamablehttp_client(
                **kwargs,
            )
        else:  # SSE
            return sse_client(**kwargs)

    @asynccontextmanager
    async def client_session(self) -> AsyncGenerator[MCPClientSession, None]:
        """Async context manager to create and yield a ClientSession connected to the MCP server."""
        async with self._create_client() as client:
            # Support both 2-tuple and 3-tuple
            if isinstance(client, tuple):
                if len(client) == 2:
                    read_stream, write_stream = client
                elif len(client) == 3:
                    read_stream, write_stream, _ = client
                else:
                    raise ValueError(
                        f"Unexpected tuple length from _create_client: {len(client)}"
                    )
            else:
                # Assume single duplex stream
                read_stream = write_stream = client
            async with MCPClientSession(read_stream, write_stream) as session:
                console.print("Initializing MCPClientSession")
                await session.initialize()
                yield session


@asynccontextmanager
async def mcp_connection_session(server_url: str, api_key: str):
    status = console.status(
        "[cyan]Connecting to MCP server with sse...",
        spinner="dots",
    )
    try:
        status.start()
        mcp_client = MCPClient(
            server_url=AnyUrl(server_url + "/sse"),
            api_key=api_key,
            transport_type=TransportType.SSE,
        )
        async with mcp_client.client_session() as session:
            await asyncio.wait_for(session.send_ping(), timeout=10)
            print_success(f"Connected to MCP server at {server_url} using sse.")
            status.stop()
            yield session

    except Exception as e:
        status.stop()
        if isinstance(e, asyncio.TimeoutError):
            print_error(
                f"Connection to MCP server at {server_url} timed out using SSE. Please check the server URL and your network connection.",
                e,
            )
        else:
            print_error(
                f"Error connecting to MCP server using SSE at {server_url}: {str(e)}",
            )

        raise e
