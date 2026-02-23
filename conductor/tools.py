"""Tool registry for LLM function calling."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Param:
    """Single tool parameter specification."""

    name: str
    description: str
    type: str = "string"
    required: bool = True
    items: dict[str, str] | None = None


@dataclass
class ToolDef:
    """Tool definition with schema and handler."""

    name: str
    description: str
    params: list[Param] = field(default_factory=list)
    func: Callable[..., str] | None = None

    def to_api(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        properties: dict[str, Any] = {}
        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.items is not None:
                prop["items"] = p.items
            properties[p.name] = prop
        required = [p.name for p in self.params if p.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Central registry for tool definitions and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def add(
        self,
        name: str,
        description: str,
        params: list[Param],
        func: Callable[..., str],
    ) -> None:
        """Register a tool with its definition and handler."""
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            params=params,
            func=func,
        )

    def definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions in OpenAI API format."""
        return [tool.to_api() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str) -> str:
        """Execute a tool by name with JSON-encoded arguments."""
        name = name.strip()
        tool = self._tools.get(name)
        if tool is None or tool.func is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            kwargs: dict[str, Any] = json.loads(arguments)
            return tool.func(**kwargs)
        except Exception as e:
            return json.dumps({"error": str(e)})
