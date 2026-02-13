# Parallel Slots Prototype

一个用于验证“Planner -> Router -> Worker 并行 slot 生成”架构端到端可运行性与效率的最小原型。

## Architecture

### Planner (浅层规划)
- 输入：`user_input`。
- 输出：严格结构化的 `plan.v1` JSON（Structured Outputs, strict=true）。
- 核心字段：`plan_id`、`language`、`task_summary`、`tool_contract`、`slots[]`、`output_skeleton`、`router_notes`、`questions_to_user`。
- 每个 `slots[]` 必含：`tool_requests[]`（`call_id/tool_name/args/bind_var/required`）、`depends_on[]`、`budget`、`risk`。

### Router (确定性调度)
- 调用 Planner 获取完整 Plan。
- 按 `depends_on` 进行依赖批次调度；同一批次并发调用 Worker（并发度受 `MAX_CONCURRENCY` 控制）。
- 不做 merge / 去重融合，只按 slot 顺序拼接结果，便于观察原始并行输出。
- 为每个 slot 注入 `slot_context.injected`（统一 `evidence_pack.v1`：`items[{source_type,source_id,title,snippet,meta}]`）与 `slot_context.missing_required`（required 工具缺失列表）。
- 内置语义健康检查闸门：识别退化输出并自动重试 1 次，仍失败则将该 slot 置为 `status=error` 并保留原始输出片段。
- 将产物写入 `runs/<run_id>/`。

### Worker (深层填空)
- 输入包含：完整 Plan 概览、当前 slot 上下文（`slot_id/slot_index/slot_total`）、`slot_context` 注入区、用户原始输入。
- 输出：严格结构化 `slot_output.v1` JSON（Structured Outputs, strict=true）。
- 证据依赖信号字段：`evidence_used[]`、`unsupported_claims[]`、`needs_tools[]`。
- 若结构化输出失败（自动重试后仍失败），返回 `status="error"` 的合法 SlotOutput，Router 继续执行其他 slots。

### Baseline
- 不走 Planner/slots，只调用一次 worker model 输出普通回答。
- 用于与 slots 模式对比耗时与 token。

### MVP hard constraints
- 禁止 function calling / 实际 tools 执行（仅保留可执行工具合约字段与注入接口）。
- Planner 与 Worker 均使用 JSON Schema 严格结构化输出。
- Router 只做顺序拼接，不做语义融合。

## Configuration

1. 进入目录并安装依赖：

```powershell
cd prototypes/parallel_slots
python -m pip install -e .[dev]
```

2. 复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

3. 编辑 `.env`（不要提交）：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（可空）
- `OPENAI_ORG_ID`（可空）
- `OPENAI_PROJECT_ID`（可空）
- `PLANNER_MODEL`
- `WORKER_MODEL`
- `PLANNER_TEMPERATURE`
- `WORKER_TEMPERATURE`
- `MAX_CONCURRENCY`
- `MAX_RETRIES`
- `RETRY_BASE_DELAY_MS`

## Run

### Slots 模式

```powershell
python -m parallel_slots.cli --mode slots --input "请给我一个并行 slot 原型的验证计划"
```

或：

```powershell
python -m parallel_slots.cli --mode slots --input-file .\input.txt
```

### Baseline 模式

```powershell
python -m parallel_slots.cli --mode baseline --input "请给我一个并行 slot 原型的验证计划"
```

### Repeat 基准

```powershell
python -m parallel_slots.cli --mode slots --input "同一个问题" --repeat 5
python -m parallel_slots.cli --mode baseline --input "同一个问题" --repeat 5
```

## Outputs

### Slots run

`runs/slots_<timestamp>_<id>/`
- `plan.json`
- `slot_S1.json`, `slot_S2.json`, ...
- `final.json`（`plan + slots + metrics`）
- `final.txt`（按 slot 顺序拼接的人类可读文本）
- `metrics.json`（API 调用耗时、tokens、总耗时、并发度等）

### Baseline run

`runs/baseline_<timestamp>_<id>/`
- `baseline_answer.txt`
- `baseline_final.json`
- `baseline_metrics.json`

### Repeat summary

`runs/repeat_summary_<mode>_<timestamp>.json`
- 包含每次 run 指标与均值/方差（耗时、tokens）。
