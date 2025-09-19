"""MCP Agent Cloud CLI entry point."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from importlib.metadata import version as metadata_version

import click
import typer
from rich.console import Console
from rich.panel import Panel
from typer.core import TyperGroup

from mcp_agent.cli.cloud.commands import (
    configure_app,
    deploy_config,
    login,
    logout,
    whoami,
)
from mcp_agent.cli.cloud.commands.logger import tail_logs
from mcp_agent.cli.cloud.commands.app import (
    delete_app,
    get_app_status,
    list_app_workflows,
)
from mcp_agent.cli.cloud.commands.apps import list_apps
from mcp_agent.cli.cloud.commands.workflow import get_workflow_status
from mcp_agent.cli.cloud.commands.servers import (
    list_servers,
    describe_server,
    delete_server,
)
from mcp_agent.cli.exceptions import CLIError
from mcp_agent.cli.utils.ux import print_error

# Setup file logging
LOG_DIR = Path.home() / ".mcp-agent" / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = LOG_DIR / "mcp-agent.log"

# Configure separate file logging without console output
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure logging - only sending to file, not to console
logging.basicConfig(level=logging.INFO, handlers=[file_handler])


class HelpfulTyperGroup(TyperGroup):
    """Typer group that shows help before usage errors for better UX."""

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            click.echo(ctx.get_help())

            console = Console(stderr=True)
            error_panel = Panel(
                str(e),
                title="Error",
                title_align="left",
                border_style="red",
                expand=True,
            )
            console.print(error_panel)
            ctx.exit(2)

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except CLIError as e:
            # Handle CLIError cleanly - show error message and exit
            logging.error(f"CLI error: {str(e)}")
            print_error(str(e))
            ctx.exit(e.exit_code)


# Root typer for `mcp-agent` CLI commands
app = typer.Typer(
    help="MCP Agent Cloud CLI for deployment and management",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)

# Simply wrap the function with typer to preserve its signature
app.command(
    name="configure",
    help="Configure an MCP app with the required params (e.g. user secrets).",
)(configure_app)


# Deployment command
app.command(
    name="deploy",
    help="""
Deploy an MCP agent using the specified configuration.

An MCP App is deployed from bundling the code at the specified config directory.\n\n

This directory must contain an 'mcp_agent.config.yaml' at its root.\n\n

If secrets are required (i.e. `no_secrets` is not set), a secrets file named 'mcp_agent.secrets.yaml' must also be present.\n
The secrets file is processed to replace secret tags with secret handles before deployment and that transformed 
file is included in the deployment bundle in place of the original secrets file.
""".strip(),
)(deploy_config)


# Sub-typer for `mcp-agent apps` commands
app_cmd_apps = typer.Typer(
    help="Management commands for multiple MCP Apps",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)
app_cmd_apps.command(name="list")(list_apps)
app.add_typer(app_cmd_apps, name="apps", help="Manage MCP Apps")

# Sub-typer for `mcp-agent app` commands
app_cmd_app = typer.Typer(
    help="Management commands for an MCP App",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)
app_cmd_app.command(name="delete")(delete_app)
app_cmd_app.command(name="status")(get_app_status)
app_cmd_app.command(name="workflows")(list_app_workflows)
app.add_typer(app_cmd_app, name="app", help="Manage an MCP App")

# Sub-typer for `mcp-agent workflow` commands
app_cmd_workflow = typer.Typer(
    help="Management commands for MCP Workflows",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)
app_cmd_workflow.command(name="status")(get_workflow_status)
app.add_typer(app_cmd_workflow, name="workflow", help="Manage MCP Workflows")

# Sub-typer for `mcp-agent servers` commands
app_cmd_servers = typer.Typer(
    help="Management commands for MCP Servers",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)
app_cmd_servers.command(name="list")(list_servers)
app_cmd_servers.command(name="describe")(describe_server)
app_cmd_servers.command(name="delete")(delete_server)
app.add_typer(app_cmd_servers, name="servers", help="Manage MCP Servers")

# Alias for servers - apps should behave identically
app.add_typer(app_cmd_servers, name="apps", help="Manage MCP Apps (alias for servers)")

# Sub-typer for `mcp-agent cloud` commands
app_cmd_cloud = typer.Typer(
    help="Cloud operations and management",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)
# Sub-typer for `mcp-agent cloud auth` commands
app_cmd_cloud_auth = typer.Typer(
    help="Cloud authentication commands",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)
# Register auth commands under cloud auth
app_cmd_cloud_auth.command(
    name="login",
    help="""
Authenticate to MCP Agent Cloud API.\n\n
Direct to the api keys page for obtaining credentials, routing through login.
""".strip(),
)(login)
app_cmd_cloud_auth.command(name="whoami", help="Print current identity and org(s).")(
    whoami
)
app_cmd_cloud_auth.command(name="logout", help="Clear credentials.")(logout)
# Sub-typer for `mcp-agent cloud logger` commands
app_cmd_cloud_logger = typer.Typer(
    help="Log configuration and streaming commands",
    no_args_is_help=True,
    cls=HelpfulTyperGroup,
)
# Register logger commands under cloud logger
app_cmd_cloud_logger.command(
    name="tail",
    help="Retrieve and stream logs from deployed MCP apps",
)(tail_logs)

# Add sub-typers to cloud
app_cmd_cloud.add_typer(app_cmd_cloud_auth, name="auth", help="Authentication commands")
app_cmd_cloud.add_typer(
    app_cmd_cloud_logger, name="logger", help="Logging and observability"
)
app_cmd_cloud.add_typer(
    app_cmd_servers, name="servers", help="Server management commands"
)
app_cmd_cloud.add_typer(
    app_cmd_servers, name="apps", help="App management commands (alias for servers)"
)
# Register cloud commands
app.add_typer(app_cmd_cloud, name="cloud", help="Cloud operations and management")
# Top-level auth commands that map to cloud auth commands
app.command(
    name="login",
    help="""
Authenticate to MCP Agent Cloud API.\n\n
Direct to the api keys page for obtaining credentials, routing through login.
""".strip(),
)(login)
app.command(name="whoami", help="Print current identity and org(s).")(whoami)
app.command(name="logout", help="Clear credentials.")(logout)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version and exit", is_flag=True
    ),
) -> None:
    """MCP Agent Cloud CLI."""
    if version:
        v = metadata_version("mcp-agent")
        typer.echo(f"MCP Agent Cloud CLI version: {v}")
        raise typer.Exit()


def run() -> None:
    """Run the CLI application."""
    try:
        app()
    except Exception as e:
        # Unexpected errors - log full exception and show clean error to user
        logging.exception("Unhandled exception in CLI")
        print_error(f"An unexpected error occurred: {str(e)}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    run()
