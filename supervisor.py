from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.types import Command

from agents import critic_agent, planner_agent, researcher_agent
from config import SUPERVISOR_PROMPT, get_settings
from schemas import CritiqueResult, ResearchPlan
from tools import save_report

settings = get_settings()


def _extract_text_from_state(state: Any) -> str:
    if isinstance(state, dict):
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "\n".join(parts)
        structured = state.get("structured_response")
        if structured is not None:
            if hasattr(structured, "model_dump_json"):
                return structured.model_dump_json(indent=2)
            return str(structured)

    if isinstance(state, AIMessage):
        return str(state.content)

    return str(state)


def _safe_filename_from_request(request: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", request.lower()).strip("_")
    if not slug:
        slug = "research_report"
    slug = slug[:60]
    return f"{slug}.md"


def _dedupe_queries(queries: list[str], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(q.strip())
        if len(result) >= limit:
            break
    return result


def _planner_run(request: str) -> ResearchPlan:
    print(f"\n[Supervisor → Planner]\n🔧 plan({request!r})")
    result = planner_agent.invoke({"messages": [{"role": "user", "content": request}]})
    plan_obj: ResearchPlan = result["structured_response"]
    print(f"📎 {json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2)}")
    return plan_obj


def _research_run(request: str) -> str:
    print(f"\n[Supervisor → Researcher]\n🔧 research({request!r})")
    result = researcher_agent.invoke({"messages": [{"role": "user", "content": request}]})
    findings = _extract_text_from_state(result)
    preview = findings.replace("\n", " ")[:300]
    suffix = "..." if len(findings) > 300 else ""
    print(f"📎 {preview}{suffix}")
    return findings


def _critique_run(*, original_request: str, plan_json: str, findings: str) -> CritiqueResult:
    try:
        plan_data = json.loads(plan_json)
        plan_obj = ResearchPlan.model_validate(plan_data)
    except Exception:
        plan_obj = ResearchPlan(
            goal="Review findings quality",
            search_queries=[],
            sources_to_check=["knowledge_base", "web"],
            output_format="structured critique",
        )

    critique_input = f"""
Original user request:
{original_request}

Approved research plan:
{json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2)}

Current findings:
{findings}
""".strip()

    preview = findings[:180]
    print(f"\n[Supervisor → Critic]\n🔧 critique({preview!r}{'...' if len(findings) > 180 else ''})")
    result = critic_agent.invoke({"messages": [{"role": "user", "content": critique_input}]})
    critique_obj: CritiqueResult = result["structured_response"]
    print(f"📎 {json.dumps(critique_obj.model_dump(), ensure_ascii=False, indent=2)}")
    return critique_obj


@tool("plan")
def plan(request: str) -> str:
    """Create a structured research plan as JSON for the supervisor."""
    plan_obj = _planner_run(request)
    return json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2)


@tool("research")
def research(request: str) -> str:
    """Execute research for the given plan or revision request and return findings text."""
    return _research_run(request)


@tool("critique")
def critique(original_request: str, plan_json: str, findings: str) -> str:
    """Review research findings against the original request and approved plan, then return a structured critique as JSON."""
    critique_obj = _critique_run(
        original_request=original_request,
        plan_json=plan_json,
        findings=findings,
    )
    return json.dumps(critique_obj.model_dump(), ensure_ascii=False, indent=2)


@tool("save_report")
def save_report_tool(filename: str, content: str) -> str:
    """Save the final approved markdown report to disk and return the saved file path."""
    return save_report(filename=filename, content=content)


model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

checkpointer = InMemorySaver()

supervisor_agent = create_agent(
    model=model,
    tools=[plan, research, critique, save_report_tool],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_report": {"allowed_decisions": ["approve", "edit", "reject"]}
            },
            description_prefix="Report save pending approval",
        )
    ],
    checkpointer=checkpointer,
)


def run_supervisor(user_request: str, thread_id: str):
    return supervisor_agent.invoke(
        {"messages": [{"role": "user", "content": user_request}]},
        config={"configurable": {"thread_id": thread_id}},
        version="v2",
    )


def resume_supervisor(command: Command, thread_id: str):
    return supervisor_agent.invoke(
        command,
        config={"configurable": {"thread_id": thread_id}},
        version="v2",
    )


def revise_report_with_feedback(report: dict[str, Any], feedback: str) -> dict[str, Any]:
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You revise markdown research reports. Keep the report structure, "
                    "apply the user's feedback, and return only the revised markdown."
                )
            ),
            HumanMessage(
                content=(
                    f"User feedback:\n{feedback}\n\n"
                    f"Current filename:\n{report['filename']}\n\n"
                    f"Current report:\n{report['content']}"
                )
            ),
        ]
    )
    updated = dict(report)
    updated["content"] = response.content if isinstance(response.content, str) else str(response.content)
    return updated


def new_thread_id() -> str:
    return str(uuid.uuid4())
