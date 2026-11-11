from datetime import datetime, timezone
from sqlalchemy import create_engine, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column

# Uses SQLite locally by default so it runs zero-config; swap to PostgreSQL URL in production
DB_URL = "sqlite:///C:/Projects/ai/platform_audit_logs.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TaskExecutionRecord(Base):
    __tablename__ = "task_execution_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    rag_context: Mapped[str] = mapped_column(Text, default="")
    final_summary: Mapped[str] = mapped_column(Text, default="")
    escalated: Mapped[str] = mapped_column(String(10), default="NO")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

# Automatically create the database tables on startup
Base.metadata.create_all(bind=engine)

def save_task_record(session_id: str, goal: str, rag_context: str, final_summary: str, escalated: bool) -> int:
    """Commits a completed multi-agent execution record to the relational database."""
    db = SessionLocal()
    try:
        record = TaskExecutionRecord(
            session_id=session_id,
            goal=goal,
            rag_context=rag_context,
            final_summary=final_summary,
            escalated="YES" if escalated else "NO"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()

def fetch_recent_records(limit: int = 10) -> list[dict]:
    """Retrieves recent task history for dashboard auditing."""
    db = SessionLocal()
    try:
        rows = db.query(TaskExecutionRecord).order_by(TaskExecutionRecord.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "goal": r.goal,
                "escalated": r.escalated,
                "created_at": r.created_at.isoformat() if r.created_at is not None else ""
            }
            for r in rows
        ]
    finally:
        db.close()
def fetch_chat_sessions(limit: int = 15) -> list[dict]:
    """Returns unique chat sessions ordered by most recent activity."""
    db = SessionLocal()
    try:
        rows = db.query(TaskExecutionRecord).order_by(TaskExecutionRecord.id.desc()).limit(100).all()
        seen = set()
        sessions = []
        for r in rows:
            if r.session_id and r.session_id not in seen:
                seen.add(r.session_id)
                sessions.append({
                    "session_id": r.session_id,
                    "title": r.goal[:40] + ("..." if len(r.goal) > 40 else ""),
                    "escalated": r.escalated
                })
                if len(sessions) >= limit:
                    break
        return sessions
    finally:
        db.close()

def fetch_session_messages(session_id: str) -> list[dict]:
    """Returns all chronological messages for a specific session_id."""
    db = SessionLocal()
    try:
        rows = (
            db.query(TaskExecutionRecord)
            .filter(TaskExecutionRecord.session_id == session_id)
            .order_by(TaskExecutionRecord.id.asc())
            .all()
        )
        return [
            {
                "id": r.id,
                "goal": r.goal,
                "rag_context": r.rag_context,
                "final_summary": r.final_summary,
                "escalated": r.escalated
            }
            for r in rows
        ]
    finally:
        db.close()
