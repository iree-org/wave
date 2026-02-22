"""Cursor Agent CLI provider for Conductor.

Uses ``cursor-agent --print`` as a local LLM backend.  Multi-turn
conversations are maintained via ``--resume SESSION_ID``.

The model does not receive native tool schemas.  Instead, the system
prompt instructs it to output fenced JSON blocks which we parse into
synthetic ``tool_calls`` compatible with the OpenRouter provider
interface.
"""

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]
DEFAULT_MODEL = "sonnet-4.6"

# Appended to the system prompt so the model outputs parseable JSON
# instead of relying on native function-calling.
TOOL_CALL_FORMAT = """

OUTPUT FORMAT — you do NOT have native function-calling tools.
Instead, to invoke a tool output EXACTLY ONE fenced JSON block per response:

To evaluate moves:
```json
{"action": "evaluate_moves", "moves": ["move TAG_A after TAG_B"], "summary": "brief reason"}
```

To finish:
```json
{"action": "done", "summary": "what you tried and the outcome"}
```

Rules:
- One JSON block per response, then STOP and wait for the result.
- Do NOT use Shell, Read, Write, Grep, or any other built-in tools.
- Do NOT output additional commentary after the JSON block.
"""


def _find_binary() -> str:
    """Locate the cursor-agent binary."""
    path = shutil.which("cursor-agent")
    if path:
        return path
    raise FileNotFoundError(
        "cursor-agent not found in PATH. "
        "Install: curl https://cursor.com/install -fsS | bash"
    )


def _build_text(messages: list[Message], start: int) -> str:
    """Format messages[start:] as plain text for cursor-agent.

    Assistant messages are skipped (already in the session history).
    Tool results are prefixed with ``[Tool Result]``.
    """
    parts: list[str] = []
    for msg in messages[start:]:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "assistant":
            continue
        if role == "tool":
            parts.append(f"[Tool Result]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _parse_tool_calls(text: str) -> list[dict] | None:
    """Extract tool calls from fenced JSON blocks in model output."""
    # Fenced ```json ... ``` blocks.
    pattern = r"```(?:json)?\s*\n(\{[^`]*?\"action\"[^`]*?\})\s*\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        # Bare JSON on its own line.
        matches = re.findall(r'^(\{"action"\s*:.+\})$', text, re.MULTILINE)
    tool_calls: list[dict] = []
    for i, raw in enumerate(matches):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        action = obj.pop("action", None)
        if not action:
            continue
        tool_calls.append(
            {
                "id": f"cursor_{i}",
                "type": "function",
                "function": {
                    "name": action,
                    "arguments": json.dumps(obj),
                },
            }
        )
    return tool_calls or None


def _run(
    prompt: str,
    model: str,
    session_id: str | None,
    log: Callable[[str], None],
) -> tuple[str, str]:
    """Invoke cursor-agent in headless mode.  Returns (session_id, content)."""
    cmd = [
        _find_binary(),
        "--print",
        "--output-format",
        "stream-json",
        "--model",
        model,
        "--force",
        "--trust",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    cmd.append(prompt)

    log(
        f"  [cmd] cursor-agent --model {model} {'--resume ' + session_id if session_id else '--new'}\n"
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    sid = session_id or ""
    content = ""
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        if etype == "system" and ev.get("subtype") == "init":
            sid = ev.get("session_id", sid)
        elif etype == "assistant":
            # Last assistant message wins.
            for block in ev.get("message", {}).get("content", []):
                if isinstance(block, dict):
                    content = block.get("text", content)
                elif isinstance(block, str):
                    content = block
        elif etype == "result":
            if event_is_error(ev):
                log(f"  [cursor-agent error] {ev.get('result', '')}\n")
            if not content:
                content = ev.get("result", "")
    return sid, content


def event_is_error(ev: dict) -> bool:
    """Check if a stream-json event signals an error."""
    return ev.get("is_error", False) or ev.get("subtype") == "error"


@dataclass
class Session:
    """Opaque session handle passed between caller and provider."""

    id: str | None = None
    sent: int = 0


def chat(
    messages: list[Message],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    log: Callable[[str], None] | None = None,
    session: Session | None = None,
) -> Message:
    """Send a chat turn via cursor-agent CLI.

    The caller owns the ``Session`` object and passes it back on each
    call.  Tool calls are extracted from fenced JSON blocks in the
    model's text output.
    """
    if log is None:
        log = lambda _: None
    if session is None:
        session = Session()

    prompt = _build_text(messages, session.sent)

    sid, content = _run(prompt, model, session.id, log)
    session.id = sid
    session.sent = len(messages)

    if content:
        preview = content[:300] + "..." if len(content) > 300 else content
        log(f"  [response] {preview}\n")

    tool_calls = _parse_tool_calls(content)
    result: Message = {"role": "assistant", "content": content}
    if tool_calls:
        result["tool_calls"] = tool_calls
    result["session"] = session

    return result
