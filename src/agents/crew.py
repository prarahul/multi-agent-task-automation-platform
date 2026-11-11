import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_INFERENCE_MODEL", "llama3:8b")

# Configure CrewAI to use your local Ollama inference engine
local_llm = LLM(
    model=f"ollama/{LOCAL_MODEL}",
    base_url=OLLAMA_BASE_URL
)

def run_crewai_compliance_audit(goal: str, rag_context: str, plan: str) -> str:
    """Executes a collaborative CrewAI two-agent compliance and risk audit."""
    auditor = Agent(
        role="Senior Forensic Compliance Auditor",
        goal="Verify operational tasks against strict corporate policies and identify financial or data security risks.",
        backstory="An expert enterprise auditor specialized in local-first compliance and expense verification.",
        llm=local_llm,
        verbose=False,
        allow_delegation=False
    )

    reviewer = Agent(
        role="Lead Risk Mitigation Specialist",
        goal="Synthesize audit findings into a crisp, single-sentence executive risk assessment.",
        backstory="A concise risk officer who translates technical audits into clear operational guidance.",
        llm=local_llm,
        verbose=False,
        allow_delegation=False
    )

    audit_task = Task(
        description=(
            f"Evaluate this goal: '{goal}' and execution plan: '{plan}' "
            f"against these retrieved company policies: '{rag_context}'."
        ),
        expected_output="A brief 2-point compliance verification check.",
        agent=auditor
    )

    summary_task = Task(
        description="Summarize the auditor's compliance findings in 1 clear, direct sentence.",
        expected_output="One concise sentence summarizing compliance findings.",
        agent=reviewer
    )

    crew = Crew(
        agents=[auditor, reviewer],
        tasks=[audit_task, summary_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    return str(result).strip()
