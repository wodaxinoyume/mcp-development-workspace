# Orchestrator workflow example

This example shows an Orchestrator workflow which dynamically plans across a number of agents to accomplish a multi-step task.

It parallelizes the task executions where possible, and continues execution until the objective is attained.

This particular example is a student assignment grader, which requires:

- Finding the student's assignment in a short_story.md on disk (using MCP filesystem server)
- Using proofreader, fact checker and style enforcer agents to evaluate the quality of the report
- The style enforcer requires reading style guidelines from the APA website using the MCP fetch server.
- Writing the graded report to disk (using MCP filesystem server)

<img width="1650" alt="Image" src="https://github.com/user-attachments/assets/12263f81-f2f8-41e2-a758-13d764f782a1" />

---

![Orchestrator workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75)

## `1` App set up

First, clone the repo and navigate to the workflow orchestrator worker example:

```bash
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/workflows/workflow_orchestrator_worker
```

Install `uv` (if you don’t have it):

```bash
pip install uv
```

Sync `mcp-agent` project dependencies:

```bash
uv sync
```

Install requirements specific to this example:

```bash
uv pip install -r requirements.txt
```

## `2` Set up environment variables

Copy and configure your secrets and env variables:

```bash
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
```

Then open `mcp_agent.secrets.yaml` and add your api key for your preferred LLM.

## (Optional) Configure tracing

In `mcp_agent.config.yaml`, you can set `otel` to `enabled` to enable OpenTelemetry tracing for the workflow.
You can [run Jaeger locally](https://www.jaegertracing.io/docs/2.5/getting-started/) to view the traces in the Jaeger UI.

## `3` Run locally

Run your MCP Agent app:

```bash
uv run main.py
```
