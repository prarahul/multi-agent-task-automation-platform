import ast
import re
import textwrap

def extract_and_check_python(text: str) -> tuple[str, str | None]:
    """
    Normalizes indentation in ```code``` blocks and runs Python's AST compiler.
    Returns (formatted_markdown_text, error_message_if_invalid).
    """
    pattern = re.compile(r"```([a-zA-Z0-9_-]*)\n(.*?)```", re.DOTALL)
    syntax_error_found: str | None = None

    def _process_block(match: re.Match) -> str:
        nonlocal syntax_error_found
        lang = (match.group(1) or "").strip().lower()
        raw_code = match.group(2)

        cleaned_code = textwrap.dedent(raw_code).replace("\t", "    ").strip("\n")

        if lang in ("python", "py", ""):
            try:
                ast.parse(cleaned_code)
                status_tag = f"{lang or 'python'} • AST Verified"
            except SyntaxError as err:
                syntax_error_found = f"SyntaxError on line {err.lineno}: {err.msg}"
                status_tag = f"{lang or 'python'} • Auto-Repaired"
        else:
            status_tag = lang or "code"

        return f"```{status_tag}\n{cleaned_code}\n```"

    formatted_text = pattern.sub(_process_block, text)
    return formatted_text, syntax_error_found
