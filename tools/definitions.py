"""OpenAI-compatible JSON schemas and strict argument validation for agent tools."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command from the current task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1, "description": "Shell command to run."},
                    "timeout": {"type": "number", "minimum": 0.001, "maximum": 300, "default": 30},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file from the task workspace using 1-based inclusive line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "description": "Workspace-relative file path."},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text to a workspace-relative file, creating parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "description": "Workspace-relative file path."},
                    "content": {"type": "string", "description": "Complete file contents."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search a workspace file or directory with a Python regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "minLength": 1, "description": "Workspace-relative file or directory."},
                },
                "required": ["pattern", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Run an allowed Git subcommand in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1, "description": "Git subcommand and arguments."},
                    "timeout": {"type": "number", "minimum": 0.001, "maximum": 300, "default": 30},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
)

_SCHEMAS_BY_NAME = {schema["function"]["name"]: schema["function"]["parameters"] for schema in TOOL_SCHEMAS}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return a copy of the five standard OpenAI-style tool definitions."""
    return deepcopy(list(TOOL_SCHEMAS))


def validate_tool_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """Validate one tool's JSON arguments and return the same argument mapping."""
    if name not in _SCHEMAS_BY_NAME:
        raise ValueError(f"unknown tool: {name}")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")

    schema = _SCHEMAS_BY_NAME[name]
    properties = schema["properties"]
    required = schema["required"]
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(f"missing required argument(s) for {name}: {', '.join(missing)}")

    if not schema.get("additionalProperties", True):
        extra = sorted(set(arguments) - set(properties))
        if extra:
            raise ValueError(f"unexpected argument(s) for {name}: {', '.join(extra)}")

    for key, value in arguments.items():
        prop = properties[key]
        expected = prop["type"]
        if expected == "string":
            valid_type = isinstance(value, str)
        elif expected == "integer":
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        elif expected == "number":
            valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            raise ValueError(f"unsupported schema type {expected!r} for {name}.{key}")
        if not valid_type:
            raise ValueError(f"argument {key!r} for {name} must be {expected}")
        if expected == "number" and not math.isfinite(value):
            raise ValueError(f"argument {key!r} for {name} must be finite")
        if "minLength" in prop and len(value) < prop["minLength"]:
            raise ValueError(f"argument {key!r} for {name} must not be empty")
        if "minimum" in prop and value < prop["minimum"]:
            raise ValueError(f"argument {key!r} for {name} must be >= {prop['minimum']}")
        if "maximum" in prop and value > prop["maximum"]:
            raise ValueError(f"argument {key!r} for {name} must be <= {prop['maximum']}")

    if name in {"shell", "git"} and not arguments["command"].strip():
        raise ValueError(f"argument 'command' for {name} must not be blank")
    if name in {"read_file", "write_file", "grep"}:
        path = arguments["path"]
        if Path(path).is_absolute() or "\x00" in path:
            raise ValueError("tool paths must be workspace-relative and contain no NUL bytes")
        if ".." in Path(path).parts:
            raise ValueError("tool paths must not traverse to a parent directory")

    if name == "read_file" and "end_line" in arguments:
        start_line = arguments.get("start_line", 1)
        if arguments["end_line"] < start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
    return arguments
