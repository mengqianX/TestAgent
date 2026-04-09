# Toast Prompt Versioning

## Directory layout

- `current.json`: the active prompt used by runtime when `VGA_TOAST_PROMPT_VERSION=current`.
- `archive/*.json`: immutable historical prompt versions for A/B comparison.

## Prompt file schema

Each JSON file must contain:

- `id`: unique prompt identifier, e.g. `toast_v2026.04.07_r00_baseline`
- `system_prompt`: system instruction text
- `user_prompt_template`: user prompt template with placeholders

Supported placeholders in `user_prompt_template`:

- `{task_intent}`
- `{candidate_timestamp_sec:.2f}`
- `{keywords_text}`
- `{preprocess_summary}`
- `{preprocess_structured_json}`

## Runtime switching

Set env var before running:

```bash
export VGA_TOAST_PROMPT_VERSION=current
```

Or switch to an archived version by id:

```bash
export VGA_TOAST_PROMPT_VERSION=toast_v2026.04.07_r00_baseline
```
