"""Schema definitions for planner and worker structured outputs."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft7Validator


PLAN_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PlanV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "version",
        "plan_id",
        "language",
        "task_summary",
        "tool_contract",
        "slots",
        "output_skeleton",
        "router_notes",
        "questions_to_user",
    ],
    "properties": {
        "version": {
            "type": "string",
            "enum": ["plan.v1"],
        },
        "plan_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "description": "Unique plan id (uuid or timestamp).",
        },
        "language": {
            "type": "string",
            "enum": ["zh", "en"],
            "description": "Planner decides based on user_input. Default zh.",
        },
        "task_summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "description": "One-sentence summary of the user's goal.",
        },
        "tool_contract": {
            "type": "object",
            "additionalProperties": False,
            "required": ["enabled", "mode", "notes"],
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether planner expects tool execution in router.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["mock", "real"],
                    "description": "Execution mode for future tool contract plumbing.",
                },
                "notes": {
                    "type": "string",
                    "minLength": 0,
                    "maxLength": 2000,
                },
            },
        },
        "slots": {
            "type": "array",
            "minItems": 1,
            "maxItems": 256,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "slot_id",
                    "title",
                    "worker_brief",
                    "expected_output_schema_hint",
                    "tool_requests",
                    "depends_on",
                    "budget",
                    "risk",
                ],
                "properties": {
                    "slot_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 16,
                        "description": "e.g., S1, S2, ... Router uses the order in this array.",
                    },
                    "title": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                    },
                    "worker_brief": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                        "description": "Instruction for the worker. Must mention slot position context.",
                    },
                    "expected_output_schema_hint": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 300,
                        "description": "One-line hint about what fields matter most in SlotOutput.",
                    },
                    "tool_requests": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 32,
                        "description": "Explicit tool contract requests for this slot.",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["call_id", "tool_name", "args", "bind_var", "required"],
                            "properties": {
                                "call_id": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 64,
                                },
                                "tool_name": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 128,
                                },
                                "args": {
                                    "type": "object",
                                    "description": "Tool invocation args in JSON object form.",
                                },
                                "bind_var": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 64,
                                    "description": "Variable name the router injects into slot context.",
                                },
                                "required": {
                                    "type": "boolean",
                                    "description": "Whether missing tool output blocks realistic completion.",
                                },
                            },
                        },
                    },
                    "depends_on": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 32,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 16,
                        },
                        "description": "Slot ids that must complete before this slot runs.",
                    },
                    "budget": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["max_tokens", "priority"],
                        "properties": {
                            "max_tokens": {
                                "type": "integer",
                                "minimum": 32,
                                "maximum": 32000,
                            },
                            "priority": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                    },
                    "risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
            },
        },
        "output_skeleton": {
            "type": "string",
            "minLength": 1,
            "maxLength": 8000,
            "description": "Composable output skeleton such as '{{S1}} ... {{S2}}'.",
        },
        "router_notes": {
            "type": "string",
            "minLength": 0,
            "maxLength": 2000,
            "description": "Optional notes for the router. Tools/function-calling are NOT used in MVP.",
        },
        "questions_to_user": {
            "type": "array",
            "minItems": 0,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "question", "why", "default_assumption"],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 16},
                    "question": {"type": "string", "minLength": 1, "maxLength": 300},
                    "why": {"type": "string", "minLength": 1, "maxLength": 300},
                    "default_assumption": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 300,
                    },
                },
            },
        },
    },
}


SLOT_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SlotOutputV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "version",
        "status",
        "slot_id",
        "title",
        "answer",
        "assumptions",
        "open_questions",
        "evidence_used",
        "unsupported_claims",
        "needs_tools",
        "confidence",
        "notes_for_router",
    ],
    "properties": {
        "version": {
            "type": "string",
            "enum": ["slot_output.v1"],
        },
        "status": {
            "type": "string",
            "enum": ["ok", "error"],
            "description": "Use 'error' if structured generation failed after retries; still fill required fields.",
        },
        "slot_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 16,
        },
        "title": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
        },
        "answer": {
            "type": "string",
            "minLength": 0,
            "maxLength": 8000,
            "description": "Markdown is allowed as a JSON string.",
        },
        "assumptions": {
            "type": "array",
            "minItems": 0,
            "maxItems": 10,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
        "open_questions": {
            "type": "array",
            "minItems": 0,
            "maxItems": 10,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
        "evidence_used": {
            "type": "array",
            "minItems": 0,
            "maxItems": 32,
            "items": {"type": "string", "minLength": 1, "maxLength": 64},
            "description": "Bind vars consumed from slot_context.injected.",
        },
        "unsupported_claims": {
            "type": "array",
            "minItems": 0,
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
            "description": "Claims produced without injected evidence support.",
        },
        "needs_tools": {
            "type": "array",
            "minItems": 0,
            "maxItems": 32,
            "items": {"type": "string", "minLength": 1, "maxLength": 64},
            "description": "Bind vars still needed for evidence-grounded completion.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "notes_for_router": {
            "type": "string",
            "minLength": 0,
            "maxLength": 2000,
        },
    },
}


_PLAN_VALIDATOR = Draft7Validator(PLAN_JSON_SCHEMA)
_SLOT_OUTPUT_VALIDATOR = Draft7Validator(SLOT_OUTPUT_JSON_SCHEMA)


def planner_text_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "json_schema": {
                "name": "plan_v1",
                "strict": True,
                "schema": PLAN_JSON_SCHEMA,
            },
        }
    }


def worker_text_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "json_schema": {
                "name": "slot_output_v1",
                "strict": True,
                "schema": SLOT_OUTPUT_JSON_SCHEMA,
            },
        }
    }


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(_PLAN_VALIDATOR.iter_errors(plan), key=lambda err: list(err.path))
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.path) or "<root>"
        raise ValueError(f"Plan schema validation failed at {path}: {error.message}")
    return plan


def validate_slot_output(slot_output: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(_SLOT_OUTPUT_VALIDATOR.iter_errors(slot_output), key=lambda err: list(err.path))
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.path) or "<root>"
        raise ValueError(f"SlotOutput schema validation failed at {path}: {error.message}")
    return slot_output


def make_error_slot_output(slot_id: str, title: str, reason: str) -> dict[str, Any]:
    return {
        "version": "slot_output.v1",
        "status": "error",
        "slot_id": slot_id,
        "title": title,
        "answer": "",
        "assumptions": [],
        "open_questions": [],
        "evidence_used": [],
        "unsupported_claims": [],
        "needs_tools": [],
        "confidence": 0.0,
        "notes_for_router": reason,
    }
