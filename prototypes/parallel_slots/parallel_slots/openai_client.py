"""OpenAI wrapper layer for planner/worker calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
import time
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .config import Settings
from .schemas import (
    make_error_slot_output,
    planner_text_format,
    validate_plan,
    validate_slot_output,
    worker_text_format,
)


_PLANNER_SYSTEM_PROMPT = """You are the Planner role in a Parallel Slots prototype.
Return ONLY one JSON object that strictly follows schema plan.v1.
Constraints:
- Set version to plan.v1.
- Decide language as zh or en (default zh unless user input is clearly English).
- Provide 1-10 slots.
- Each slot.worker_brief must mention slot position context in the full plan.
- router_notes can suggest execution ordering/concurrency.
- questions_to_user can be empty; router will continue execution regardless.
- No function calling and no tool usage.
"""


_WORKER_SYSTEM_PROMPT = """You are a Worker role in a Parallel Slots prototype.
Return ONLY one JSON object that strictly follows schema slot_output.v1.
Constraints:
- Set version to slot_output.v1.
- Use status=ok when successful.
- Keep slot_id/title aligned with current slot context.
- answer should focus on this slot only while respecting full-plan context.
- No function calling and no tool usage.
"""


_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class PlannerExecutionError(RuntimeError):
    """Raised when planner output is unavailable after retries."""

    def __init__(self, message: str, metrics: list["ApiCallMetric"] | None = None):
        super().__init__(message)
        self.metrics = metrics or []


class StructuredOutputError(RuntimeError):
    """Raised when structured output parsing/validation fails."""

    def __init__(self, message: str, metrics: list["ApiCallMetric"] | None = None):
        super().__init__(message)
        self.metrics = metrics or []


class OpenAICallError(RuntimeError):
    """Raised when OpenAI API call fails after retries."""

    def __init__(self, message: str, metrics: list["ApiCallMetric"] | None = None):
        super().__init__(message)
        self.metrics = metrics or []


@dataclass
class ApiCallMetric:
    operation: str
    model: str
    attempt: int
    duration_ms: float
    success: bool
    parse_ok: bool | None
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAIParallelSlotsClient:
    """OpenAI client wrapper used by planner/router/worker orchestration."""

    def __init__(self, settings: Settings):
        self._settings = settings

        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        if settings.openai_org_id:
            client_kwargs["organization"] = settings.openai_org_id
        if settings.openai_project_id:
            client_kwargs["project"] = settings.openai_project_id

        self._client = OpenAI(**client_kwargs)

    def call_planner(self, user_input: str) -> dict[str, Any]:
        plan, _ = self.call_planner_with_metrics(user_input)
        return plan

    def call_planner_with_metrics(self, user_input: str) -> tuple[dict[str, Any], list[ApiCallMetric]]:
        input_messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Generate plan.v1 for the following user input.\n"
                    "Do not add any non-JSON text.\n\n"
                    f"user_input:\n{user_input}"
                ),
            },
        ]

        try:
            plan, metrics = self._call_structured_json(
                operation="planner",
                model=self._settings.planner_model,
                temperature=self._settings.planner_temperature,
                input_messages=input_messages,
                text_format=planner_text_format(),
                validator=validate_plan,
                structured_retry_count=1,
            )
            return plan, metrics
        except StructuredOutputError as exc:
            raise PlannerExecutionError(str(exc), metrics=exc.metrics) from exc

    def call_worker(
        self,
        user_input: str,
        plan: dict[str, Any],
        slot: dict[str, Any],
        slot_index: int,
        slot_total: int,
    ) -> dict[str, Any]:
        slot_output, _ = self.call_worker_with_metrics(user_input, plan, slot, slot_index, slot_total)
        return slot_output

    def call_worker_with_metrics(
        self,
        user_input: str,
        plan: dict[str, Any],
        slot: dict[str, Any],
        slot_index: int,
        slot_total: int,
    ) -> tuple[dict[str, Any], list[ApiCallMetric]]:
        slot_overview = [
            {
                "slot_id": slot_item["slot_id"],
                "title": slot_item["title"],
            }
            for slot_item in plan["slots"]
        ]

        worker_context = {
            "user_input": user_input,
            "plan": {
                "plan_id": plan["plan_id"],
                "language": plan["language"],
                "task_summary": plan["task_summary"],
                "router_notes": plan.get("router_notes", ""),
                "slots_overview": slot_overview,
            },
            "current_slot": {
                "slot_id": slot["slot_id"],
                "slot_index": slot_index,
                "slot_total": slot_total,
                "title": slot["title"],
                "worker_brief": slot["worker_brief"],
                "expected_output_schema_hint": slot["expected_output_schema_hint"],
            },
        }

        input_messages = [
            {"role": "system", "content": _WORKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Produce slot_output.v1 for the following context.\n"
                    "Do not add any non-JSON text.\n\n"
                    + json.dumps(worker_context, ensure_ascii=False, indent=2)
                ),
            },
        ]

        try:
            slot_output, metrics = self._call_structured_json(
                operation=f"worker:{slot['slot_id']}",
                model=self._settings.worker_model,
                temperature=self._settings.worker_temperature,
                input_messages=input_messages,
                text_format=worker_text_format(),
                validator=validate_slot_output,
                structured_retry_count=1,
            )
            slot_output["slot_id"] = slot["slot_id"]
            slot_output["title"] = slot["title"]
            return validate_slot_output(slot_output), metrics
        except StructuredOutputError as exc:
            error_result = make_error_slot_output(
                slot_id=slot["slot_id"],
                title=slot["title"],
                reason=f"Structured generation failed: {exc}",
            )
            return error_result, exc.metrics

    def _call_structured_json(
        self,
        *,
        operation: str,
        model: str,
        temperature: float,
        input_messages: list[dict[str, Any]],
        text_format: dict[str, Any],
        validator: Callable[[dict[str, Any]], dict[str, Any]],
        structured_retry_count: int,
    ) -> tuple[dict[str, Any], list[ApiCallMetric]]:
        metrics: list[ApiCallMetric] = []

        for structured_attempt in range(1, structured_retry_count + 2):
            try:
                response, request_metrics = self._responses_create_with_retries(
                    operation=operation,
                    model=model,
                    temperature=temperature,
                    input_messages=input_messages,
                    text_format=text_format,
                )
            except OpenAICallError as exc:
                metrics.extend(exc.metrics)
                raise StructuredOutputError(
                    f"{operation} failed during API call phase: {exc}",
                    metrics=metrics,
                ) from exc

            metrics.extend(request_metrics)
            try:
                payload = self._parse_structured_json(response)
                validator(payload)
                if metrics:
                    metrics[-1].parse_ok = True
                return payload, metrics
            except Exception as exc:
                if metrics:
                    metrics[-1].parse_ok = False
                    metrics[-1].error = f"structured_output_error: {exc}"
                if structured_attempt <= structured_retry_count:
                    continue
                raise StructuredOutputError(
                    f"{operation} structured parsing failed after {structured_retry_count + 1} attempts: {exc}",
                    metrics=metrics,
                ) from exc

        raise StructuredOutputError(f"{operation} unknown structured output failure", metrics=metrics)

    def _responses_create_with_retries(
        self,
        *,
        operation: str,
        model: str,
        temperature: float,
        input_messages: list[dict[str, Any]],
        text_format: dict[str, Any],
    ) -> tuple[Any, list[ApiCallMetric]]:
        metrics: list[ApiCallMetric] = []

        for attempt in range(1, self._settings.max_retries + 2):
            start_time = time.perf_counter()
            try:
                response = self._client.responses.create(
                    model=model,
                    temperature=temperature,
                    input=input_messages,
                    text=text_format,
                )
                duration_ms = (time.perf_counter() - start_time) * 1000
                usage = self._extract_usage(response)
                metrics.append(
                    ApiCallMetric(
                        operation=operation,
                        model=model,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        success=True,
                        parse_ok=None,
                        error=None,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        total_tokens=usage.get("total_tokens"),
                    )
                )
                return response, metrics
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000
                retryable = self._is_retryable_error(exc)
                metrics.append(
                    ApiCallMetric(
                        operation=operation,
                        model=model,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        success=False,
                        parse_ok=None,
                        error=f"{type(exc).__name__}: {exc}",
                        input_tokens=None,
                        output_tokens=None,
                        total_tokens=None,
                    )
                )

                if retryable and attempt <= self._settings.max_retries:
                    self._sleep_with_backoff(attempt)
                    continue

                raise OpenAICallError(
                    f"{operation} API call failed at attempt {attempt}: {exc}",
                    metrics=metrics,
                ) from exc

        raise OpenAICallError(f"{operation} exhausted retries", metrics=metrics)

    def _sleep_with_backoff(self, attempt: int) -> None:
        base_delay_s = self._settings.retry_base_delay_ms / 1000.0
        exponential_delay = base_delay_s * (2 ** (attempt - 1))
        jitter = random.uniform(0.0, exponential_delay * 0.2)
        time.sleep(exponential_delay + jitter)

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            return status_code in _RETRYABLE_STATUS_CODES
        return False

    @staticmethod
    def _parse_structured_json(response: Any) -> dict[str, Any]:
        text = getattr(response, "output_text", None)
        if text:
            return json.loads(text)

        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            extracted = OpenAIParallelSlotsClient._extract_text_from_dump(dumped)
            if extracted is not None:
                return json.loads(extracted)

        raise ValueError("Responses API payload did not contain output_text")

    @staticmethod
    def _extract_text_from_dump(payload: dict[str, Any]) -> str | None:
        output_items = payload.get("output", [])
        for item in output_items:
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    return content["text"]
        return None

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }

        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif isinstance(usage, dict):
            usage_dict = usage
        else:
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }

        input_tokens = usage_dict.get("input_tokens")
        output_tokens = usage_dict.get("output_tokens")
        total_tokens = usage_dict.get("total_tokens")

        if input_tokens is None:
            input_tokens = usage_dict.get("prompt_tokens")
        if output_tokens is None:
            output_tokens = usage_dict.get("completion_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
