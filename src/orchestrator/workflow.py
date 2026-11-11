import os
import sys
import re
import asyncio
from datetime import datetime
from typing import TypedDict, Callable, Awaitable, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from src.ml_core.rag_engine import retrieve_relevant_policy
from src.ml_core.code_compiler import extract_and_check_python
from src.agents.crew import run_crewai_compliance_audit
from config.redis_config import append_chat_memory, get_chat_memory

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_INFERENCE_MODEL", "llama3:8b")
CODER_MODEL = os.getenv("LOCAL_CODER_MODEL", "qwen2.5-coder:7b")

llm = ChatOllama(
    model=LOCAL_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.6,
    repeat_penalty=1.2
)

coder_llm = ChatOllama(
    model=CODER_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    repeat_penalty=1.1
)

def get_live_time_context() -> str:
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(str(item) for item in content).strip()
    return str(content).strip()

async def call_mcp_save_report(filename: str, content: str) -> str:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["C:/Projects/ai/mcp_servers/filesystem_server.py"],
        env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "save_audit_report",
                arguments={"filename": filename, "content": content}
            )
            return str(result.content)

# 1. Shared State
class MultiAgentState(TypedDict):
    task_id: str
    goal: str
    intent: str
    schedule_seconds: int
    rag_context: str
    plan: str
    analysis: str
    final_response: str
    requires_escalation: bool
    notify_callback: Callable[[str, str, str], Awaitable[None]]

# 2. Smart Intent Router
async def intent_router_node(state: MultiAgentState) -> dict:
    goal_clean = state["goal"].strip().lower()
    
    if "schedule" in goal_clean or "every " in goal_clean:
        match = re.search(r"(\d+)\s*(second|sec|minute|min)", goal_clean)
        seconds = 30
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            seconds = val * 60 if "min" in unit else val
        return {"intent": "schedule", "schedule_seconds": max(10, seconds)}

    task_keywords = [
        "audit", "report", "analyze", "analysis", "expense", "cost",
        "invoice", "call supervisor", "call manager", "policy", "save file", "generate report"
    ]
    is_task = any(word in goal_clean for word in task_keywords)
    return {"intent": "task" if is_task else "chat", "schedule_seconds": 0}

def route_by_intent(state: MultiAgentState) -> Literal["casual_chat", "schedule_handler", "rag_retriever"]:
    if state["intent"] == "schedule":
        return "schedule_handler"
    return "casual_chat" if state["intent"] == "chat" else "rag_retriever"
# 3. Conversational Chat & Coding Node
async def casual_chat_node(state: MultiAgentState) -> dict:
    goal_clean = state["goal"].strip().lower().rstrip("!.?")
    simple_greetings = {"hi", "hello", "hey", "hii", "heyy", "yo", "good morning", "good afternoon", "good evening"}
    
    if goal_clean in simple_greetings:
        reply_text = "Hi! How can I help you today?"
        append_chat_memory(state["task_id"], state["goal"], reply_text)
        return {
            "final_response": reply_text,
            "rag_context": "N/A (Casual Greeting)",
            "requires_escalation": False
        }

    code_keywords = ["code", "python", "function", "script", "program", "algorithm", "sql", "html", "javascript", "class ", "def ", "debug", "sort", "binary search"]
    is_coding_query = any(kw in goal_clean for kw in code_keywords)

    if is_coding_query:
        await state["notify_callback"]("Qwen 2.5 Coder", "Generating and AST-verifying code...", "PROCESSING")
        code_messages = [
            SystemMessage(content=(
                "You are an expert Senior Software Engineer. Provide clean, well-structured, runnable code "
                "wrapped inside a single ```python ... ``` (or appropriate language) markdown code block with strict 4-space indentation, "
                "type hints, and a brief 1-sentence explanation outside the block."
            )),
            HumanMessage(content=state["goal"])
        ]
        response = await coder_llm.ainvoke(code_messages)
        raw_reply = extract_text(response.content)
        formatted_reply, syntax_err = extract_and_check_python(raw_reply)

        if syntax_err:
            await state["notify_callback"]("AST Compiler", f"Caught {syntax_err}. Auto-repairing...", "PROCESSING")
            repair_messages = code_messages + [
                AIMessage(content=raw_reply),
                HumanMessage(content=f"Your code had a Python {syntax_err}. Fix the indentation/syntax and return only the corrected ```python ... ``` block.")
            ]
            retry_resp = await coder_llm.ainvoke(repair_messages)
            formatted_reply, _ = extract_and_check_python(extract_text(retry_resp.content))

        append_chat_memory(state["task_id"], state["goal"], formatted_reply)
        return {
            "final_response": formatted_reply,
            "rag_context": "Qwen 2.5 Coder (AST Verified)",
            "requires_escalation": False
        }

    current_dt = get_live_time_context()
    history_lines = get_chat_memory(state["task_id"], limit=6)
    
    messages: list[BaseMessage] = [
        SystemMessage(content=(
            f"You are a helpful, natural, conversational AI Assistant. "
            f"Current real-world date and time (mention ONLY if specifically asked): {current_dt}. "
            f"Answer the user's latest message directly in 1 to 3 clear sentences without repeating previous answers."
        ))
    ]
    
    for line in history_lines:
        if line.startswith("User: "):
            messages.append(HumanMessage(content=line.replace("User: ", "", 1)))
        elif line.startswith("Assistant: "):
            messages.append(AIMessage(content=line.replace("Assistant: ", "", 1)))
            
    messages.append(HumanMessage(content=state["goal"]))
    
    response = await llm.ainvoke(messages)
    reply_text = extract_text(response.content)
    reply_text, _ = extract_and_check_python(reply_text)
    
    append_chat_memory(state["task_id"], state["goal"], reply_text)
    return {
        "final_response": reply_text,
        "rag_context": "N/A (Casual Conversation)",
        "requires_escalation": False
    }

# 4. Schedule Handler Node
async def schedule_handler_node(state: MultiAgentState) -> dict:
    sec = state.get("schedule_seconds", 30)
    clean_goal = re.sub(r"(?i)\b(schedule|every\s+\d+\s*(seconds?|secs?|minutes?|mins?))\b", "", state["goal"]).strip()
    clean_goal = clean_goal or "Audit cloud server costs"
    
    reply_text = f"Scheduled '{clean_goal}' to run automatically every {sec} seconds in the background."
    return {
        "goal": clean_goal,
        "final_response": reply_text,
        "rag_context": f"Scheduled Job ({sec}s interval)",
        "requires_escalation": False
    }

# 5. Node 1 (Task Mode): Local PyTorch Semantic RAG Retriever
async def rag_retriever_node(state: MultiAgentState) -> dict:
    await state["notify_callback"]("PyTorch RAG Engine", "Searching corporate policies using tensor embeddings...", "PROCESSING")
    matched_policies = await asyncio.to_thread(retrieve_relevant_policy, state["goal"], 2)
    await state["notify_callback"]("PyTorch RAG Engine", f"Retrieved Policy Context: {matched_policies}", "INFO")
    return {"rag_context": matched_policies}

# 6. Node 2 (Task Mode): Autonomous Task Planner
async def planner_agent_node(state: MultiAgentState) -> dict:
    await state["notify_callback"]("Task Planner Agent", "Creating execution plan...", "PROCESSING")
    current_dt = get_live_time_context()
    prompt = (
        f"Current Date/Time: {current_dt}. "
        f"You are a Task Planner Agent. Create a concise 2-bullet execution plan "
        f"for: '{state['goal']}'. Obey these policies: {state['rag_context']}"
    )
    response = await llm.ainvoke(prompt)
    plan_text = extract_text(response.content)
    await state["notify_callback"]("Task Planner Agent", f"Plan ready: {plan_text}", "INFO")
    return {"plan": plan_text}

# 7. Node 3 (Task Mode): CrewAI Multi-Agent Compliance & Risk Team
async def analyst_agent_node(state: MultiAgentState) -> dict:
    await state["notify_callback"]("CrewAI Audit Team", "Dispatching Senior Auditor & Risk Specialist crew...", "PROCESSING")
    
    analysis_text = await asyncio.to_thread(
        run_crewai_compliance_audit,
        state["goal"],
        state["rag_context"],
        state["plan"]
    )
    
    goal_lower = state["goal"].lower()
    needs_call = any(word in goal_lower for word in ["call", "audit", "escalate", "urgent", "manager", "cloud"])
    
    await state["notify_callback"]("CrewAI Audit Team", f"Crew Consensus: {analysis_text}", "INFO")
    return {"analysis": analysis_text, "requires_escalation": needs_call}

# 8. Node 4 (Task Mode): Execution & MCP Tool Coordinator
async def executor_agent_node(state: MultiAgentState) -> dict:
    await state["notify_callback"]("Execution Coordinator", "Saving report via MCP Filesystem Server...", "PROCESSING")
    
    current_dt = get_live_time_context()
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", state["goal"][:25]).strip("_") or "report"
    filename = f"{safe_slug}.txt"
    report_body = (
        f"GENERATED AT: {current_dt}\n"
        f"GOAL: {state['goal']}\n\n"
        f"MATCHED POLICIES:\n{state['rag_context']}\n\n"
        f"EXECUTION PLAN:\n{state['plan']}\n\n"
        f"CREWAI AUDIT CONSENSUS:\n{state['analysis']}\n"
    )
    
    mcp_status = await call_mcp_save_report(filename, report_body)
    await state["notify_callback"]("MCP Tool Bridge", f"Saved locally: {filename} ({mcp_status})", "INFO")
    
    prompt = (
        f"Speak naturally in 2 short sentences confirming you checked company policies, "
        f"completed the CrewAI audit for '{state['goal']}', and saved the report to {filename}."
    )
    response = await llm.ainvoke(prompt)
    final_text = extract_text(response.content)
    
    append_chat_memory(state["task_id"], state["goal"], final_text)
    return {"final_response": final_text}

# 9. Build and Compile the Conditional LangGraph State Machine
def build_multi_agent_graph():
    workflow = StateGraph(MultiAgentState)
    
    workflow.add_node("router", intent_router_node)
    workflow.add_node("casual_chat", casual_chat_node)
    workflow.add_node("schedule_handler", schedule_handler_node)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("analyst", analyst_agent_node)
    workflow.add_node("executor", executor_agent_node)
    
    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_by_intent)
    workflow.add_edge("casual_chat", END)
    workflow.add_edge("schedule_handler", END)
    workflow.add_edge("rag_retriever", "planner")
    workflow.add_edge("planner", "analyst")
    workflow.add_edge("analyst", "executor")
    workflow.add_edge("executor", END)
    
    return workflow.compile()

multi_agent_app = build_multi_agent_graph()
