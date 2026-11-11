import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Define and create the isolated sandbox directory
SANDBOX_DIR = Path("C:/Projects/ai/mcp_sandbox_workspace").resolve()
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the FastMCP server
mcp = FastMCP("mcp-local-filesystem")

@mcp.tool()
def save_audit_report(filename: str, content: str) -> str:
    """Saves a text or markdown report safely inside the local sandbox workspace."""
    safe_name = os.path.basename(filename)
    target_path = SANDBOX_DIR / safe_name
    target_path.write_text(content, encoding="utf-8")
    return f"SUCCESS: Report saved to {target_path}"

@mcp.tool()
def list_workspace_files() -> str:
    """Lists all files currently saved in the sandbox workspace."""
    files = [f.name for f in SANDBOX_DIR.iterdir() if f.is_file()]
    return f"Workspace files: {', '.join(files) if files else 'No files found.'}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
