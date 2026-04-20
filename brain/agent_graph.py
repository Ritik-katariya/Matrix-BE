"""
brain/agent_graph.py

LangGraph multi-agent pipeline.

Graph flow:
  ┌──────────┐     ┌──────────────┐     ┌───────────────┐
  │  intake  │────▶│ intent_router│────▶│ domain_agent  │
  └──────────┘     └──────────────┘     │  (one of N)   │
                                        └───────┬───────┘
                                                │
                                        ┌───────▼───────┐
                                        │  response_gen │
                                        └───────────────┘

Agents defined here (Phase 1):
  • ConversationAgent  — general chat, Hinglish awareness
  • TaskAgent          — reminders, timers, notes
  • KnowledgeAgent     — factual Q&A (searches + reasoning)

Phase 2+ will add: EmailAgent, CalendarAgent, CodeAgent, etc.
Adding a new agent = one new node + one new intent label. Nothing else changes.
"""
from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from brain.llm_router import TaskPriority, get_response, stream_response
from brain.prompts import INTENT_ROUTER_PROMPT, SYSTEM_PROMPT
from core.logger import get_logger

logger = get_logger("brain.graph")


# ── Graph state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str                      # detected by router node
    priority: str                    # "critical" | "standard" | "offline"
    language: str                    # detected language from STT
    response_tokens: list[str]       # accumulated streaming tokens


# ── Node: intake ───────────────────────────────────────────────────────────────

async def intake_node(state: AgentState) -> AgentState:
    """
    Normalize the latest user message.
    Sets default values for downstream nodes.
    """
    logger.debug("intake_node", msg_count=len(state["messages"]))
    return {
        **state,
        "intent": state.get("intent", "unknown"),
        "priority": state.get("priority", "standard"),
        "language": state.get("language", "en"),
        "response_tokens": [],
    }


# ── Node: intent_router ────────────────────────────────────────────────────────

async def intent_router_node(state: AgentState) -> AgentState:
    """
    Fast intent classification using a lightweight call.
    Uses Ollama (local) for speed — no round-trip to cloud.
    Returns one of: conversation | task | knowledge | critical
    """
    last_msg = state["messages"][-1].content if state["messages"] else ""
    prompt = INTENT_ROUTER_PROMPT.format(user_message=last_msg)

    # Use Ollama for routing (local, fast, ~100ms)
    result = await get_response(
        [SystemMessage(content=prompt)],
        priority=TaskPriority.OFFLINE,
    )

    # Parse JSON intent
    try:
        data = json.loads(result.strip())
        intent = data.get("intent", "conversation")
        priority = data.get("priority", "standard")
    except json.JSONDecodeError:
        intent = "conversation"
        priority = "standard"

    logger.info("Intent classified", intent=intent, priority=priority)
    return {**state, "intent": intent, "priority": priority}


# ── Node: conversation_agent ───────────────────────────────────────────────────

async def conversation_agent_node(state: AgentState) -> AgentState:
    """
    General chat. Handles Hinglish natively.
    Uses NVIDIA NIM (smart, free) by default.
    """
    system = SystemMessage(content=SYSTEM_PROMPT.format(language=state["language"]))
    messages = [system] + state["messages"]

    priority = TaskPriority(state["priority"])
    tokens: list[str] = []
    async for token in stream_response(messages, priority=priority):
        tokens.append(token)

    response = "".join(tokens)
    logger.debug("conversation_agent done", chars=len(response))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "response_tokens": tokens,
    }


# ── Node: task_agent ───────────────────────────────────────────────────────────

async def task_agent_node(state: AgentState) -> AgentState:
    """
    Handles: reminders, timers, notes, alarms.
    Phase 1: parses task and confirms. Phase 2: integrates with calendar/notifications.
    """
    system = SystemMessage(content=(
        "You are a task management assistant. "
        "Extract: task_type (reminder/timer/note), content, time (if any). "
        "Respond in the same language the user used. "
        "Keep response short and confirm what you understood."
    ))
    messages = [system] + state["messages"]
    tokens: list[str] = []
    async for token in stream_response(messages, priority=TaskPriority.STANDARD):
        tokens.append(token)

    response = "".join(tokens)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "response_tokens": tokens,
    }


# ── Node: knowledge_agent ──────────────────────────────────────────────────────

async def knowledge_agent_node(state: AgentState) -> AgentState:
    """
    Factual Q&A. Uses NVIDIA NIM for reasoning quality.
    Phase 2: will add web search tool.
    """
    system = SystemMessage(content=(
        "You are a knowledgeable assistant. "
        "Answer factually and concisely. "
        "If you're uncertain, say so honestly. "
        "Respond in the same language/script the user used."
    ))
    messages = [system] + state["messages"]
    tokens: list[str] = []
    async for token in stream_response(messages, priority=TaskPriority.STANDARD):
        tokens.append(token)

    response = "".join(tokens)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "response_tokens": tokens,
    }


# ── Routing function ───────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """LangGraph conditional edge — returns the next node name."""
    intent = state.get("intent", "conversation")
    route_map = {
        "conversation": "conversation_agent",
        "task":         "task_agent",
        "knowledge":    "knowledge_agent",
        "critical":     "conversation_agent",  # Phase 2: dedicated critical agent
    }
    return route_map.get(intent, "conversation_agent")


# ── Build graph ────────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Compile and return the LangGraph StateGraph.
    Called once at startup; the compiled graph is reused for every request.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("intake",              intake_node)
    builder.add_node("intent_router",       intent_router_node)
    builder.add_node("conversation_agent",  conversation_agent_node)
    builder.add_node("task_agent",          task_agent_node)
    builder.add_node("knowledge_agent",     knowledge_agent_node)

    # Edges
    builder.set_entry_point("intake")
    builder.add_edge("intake", "intent_router")
    builder.add_conditional_edges("intent_router", route_by_intent)

    # All domain agents → END
    for node in ("conversation_agent", "task_agent", "knowledge_agent"):
        builder.add_edge(node, END)

    graph = builder.compile()
    logger.info("Agent graph compiled", nodes=list(builder.nodes))
    return graph


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph
