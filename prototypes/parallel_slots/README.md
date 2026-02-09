# Parallel Slots Prototype

## Setup

1. Ensure Python 3.10+ is available.
2. Create local runtime config:

```powershell
Copy-Item .env.example .env
```

3. Fill values in `.env` (`OPENAI_API_KEY`, models, and runtime knobs).

Configuration is loaded from `.env` plus process environment variables. Existing environment variables take precedence.
