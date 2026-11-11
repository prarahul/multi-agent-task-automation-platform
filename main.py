import json
import asyncio
import urllib.parse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.orchestrator.workflow import multi_agent_app, MultiAgentState
from src.orchestrator.scheduler import register_scheduled_task, list_scheduled_jobs, cancel_all_jobs
from config.database import save_task_record, fetch_recent_records
from config.database import save_task_record, fetch_recent_records, fetch_chat_sessions, fetch_session_messages


load_dotenv()

app = FastAPI(title="Multi-Agent Task Automation Platform Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutomationGoal(BaseModel):
    task_id: str
    goal: str

@app.get("/")
async def root_status_check():
    return {
        "status": "ONLINE", 
        "platform": "Multi-Agent Task Automation Engine",
        "version": "1.0.0"
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/dashboard")
async def serve_admin_dashboard():
    return FileResponse("src/api/index.html")

@app.get("/api/v1/tasks/history")
async def get_task_history():
    records = await asyncio.to_thread(fetch_recent_records, 10)
    return {"total": len(records), "records": records}

@app.get("/api/v1/tasks/scheduled")
async def get_scheduled_tasks():
    return {"jobs": list_scheduled_jobs()}

@app.post("/api/v1/tasks/scheduled/stop")
async def stop_all_scheduled_tasks():
    stopped = cancel_all_jobs()
    return {"stopped_jobs": stopped, "message": "All scheduled background tasks cancelled."}

@app.post("/api/v1/tasks/dispatch")
async def ingest_platform_request(payload: AutomationGoal):
    return {
        "status": "QUEUED", 
        "task_id": payload.task_id, 
        "message": "Automation goal ingested safely."
    }

@app.websocket("/api/v1/ws/telemetry/{task_id}")
async def stream_live_telemetry_to_ui(websocket: WebSocket, task_id: str):
    await websocket.accept()
    
    async def send_agent_update(agent_name: str, message: str, status: str = "INFO"):
        try:
            await websocket.send_json({
                "task_id": task_id,
                "status": status,
                "agent": agent_name,
                "message": message
            })
        except Exception:
            pass

    async def execute_pipeline(goal_text: str):
        initial_state: MultiAgentState = {
            "task_id": task_id,
            "goal": goal_text,
            "intent": "",
            "schedule_seconds": 0,
            "rag_context": "",
            "plan": "",
            "analysis": "",
            "final_response": "",
            "requires_escalation": False,
            "notify_callback": send_agent_update
        }
        
        final_state = await multi_agent_app.ainvoke(initial_state)
        
        if final_state.get("intent") == "schedule":
            sec = final_state.get("schedule_seconds", 30)
            scheduled_goal = final_state["goal"]
            job = register_scheduled_task(scheduled_goal, sec, execute_pipeline)
            await send_agent_update(
                "Task Scheduler",
                f"Job #{job['job_id']} registered: '{scheduled_goal}' will run every {sec}s.",
                "INFO"
            )

        record_id = await asyncio.to_thread(
            save_task_record,
            task_id,
            goal_text,
            final_state["rag_context"],
            final_state["final_response"],
            final_state.get("requires_escalation", False)
        )
        await send_agent_update(
            "Database Persistence", 
            f"Audit record #{record_id} committed to relational database.", 
            "INFO"
        )
        
        await send_agent_update(
            "AI Assistant", 
            final_state["final_response"], 
            "SPEAKING"
        )
        
        if final_state.get("requires_escalation"):
            target_phone = "919999999999"
            prefilled_text = urllib.parse.quote(f"Escalation review regarding Task #{task_id}")
            await websocket.send_json({
                "task_id": task_id,
                "status": "REQUIRES_CALL",
                "agent": "Compliance Monitor",
                "message": "Escalation criteria met. Manual supervisor call recommended.",
                "phone_number": target_phone,
                "actionable_url": f"https://wa.me{target_phone}?text={prefilled_text}",
                "ui_type": "CALL_ACTION_CARD"
            })

    try:
        await send_agent_update(
            "System Orchestrator", 
            f"Multi-Agent LangGraph + CrewAI + Scheduler runtime ready for session: {task_id}", 
            "CONNECTED"
        )
        
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            if parsed_data.get("action") == "START_TASK":
                goal = parsed_data.get("goal", "")
                await execute_pipeline(goal)
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print(f"[WEBSOCKET] Connection closed for task session frame: {task_id}")
@app.get("/api/v1/sessions")
async def get_all_sessions():
    sessions = await asyncio.to_thread(fetch_chat_sessions, 15)
    return {"sessions": sessions}

@app.get("/api/v1/sessions/{session_id}")
async def get_session_detail(session_id: str):
    messages = await asyncio.to_thread(fetch_session_messages, session_id)
    return {"session_id": session_id, "messages": messages}
