from typing import Any, Dict

from temporalio import activity

from mcp_agent.mcp.client_proxy import (
    log_via_proxy,
    ask_via_proxy,
    notify_via_proxy,
    request_via_proxy,
)
from mcp_agent.core.context_dependent import ContextDependent


class SystemActivities(ContextDependent):
    """Activities used by Temporal workflows to interact with the MCPApp gateway."""

    @activity.defn(name="mcp_forward_log")
    async def forward_log(
        self,
        execution_id: str,
        level: str,
        namespace: str,
        message: str,
        data: Dict[str, Any] | None = None,
    ) -> bool:
        registry = self.context.server_registry
        gateway_url = getattr(self.context, "gateway_url", None)
        gateway_token = getattr(self.context, "gateway_token", None)
        return await log_via_proxy(
            registry,
            execution_id=execution_id,
            level=level,
            namespace=namespace,
            message=message,
            data=data or {},
            gateway_url=gateway_url,
            gateway_token=gateway_token,
        )

    @activity.defn(name="mcp_request_user_input")
    async def request_user_input(
        self,
        session_id: str,
        workflow_id: str,
        execution_id: str,
        prompt: str,
        signal_name: str = "human_input",
    ) -> Dict[str, Any]:
        # Reuse proxy ask API; returns {result} or {error}
        registry = self.context.server_registry
        gateway_url = getattr(self.context, "gateway_url", None)
        gateway_token = getattr(self.context, "gateway_token", None)
        return await ask_via_proxy(
            registry,
            execution_id=execution_id,
            prompt=prompt,
            metadata={
                "session_id": session_id,
                "workflow_id": workflow_id,
                "signal_name": signal_name,
            },
            gateway_url=gateway_url,
            gateway_token=gateway_token,
        )

    @activity.defn(name="mcp_relay_notify")
    async def relay_notify(
        self, execution_id: str, method: str, params: Dict[str, Any] | None = None
    ) -> bool:
        registry = self.context.server_registry
        gateway_url = getattr(self.context, "gateway_url", None)
        gateway_token = getattr(self.context, "gateway_token", None)
        return await notify_via_proxy(
            registry,
            execution_id=execution_id,
            method=method,
            params=params or {},
            gateway_url=gateway_url,
            gateway_token=gateway_token,
        )

    @activity.defn(name="mcp_relay_request")
    async def relay_request(
        self, execution_id: str, method: str, params: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        registry = self.context.server_registry
        gateway_url = getattr(self.context, "gateway_url", None)
        gateway_token = getattr(self.context, "gateway_token", None)
        return await request_via_proxy(
            registry,
            execution_id=execution_id,
            method=method,
            params=params or {},
            gateway_url=gateway_url,
            gateway_token=gateway_token,
        )
