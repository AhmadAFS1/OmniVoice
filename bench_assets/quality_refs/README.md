# Quality Reference Set

This directory holds stable multilingual API requests and generated WAV files
used to validate that inference changes do not obviously degrade audio quality.

The source prompts live in:

- [reference_requests.json](/workspace/OmniVoice/bench_assets/quality_refs/reference_requests.json)

Generate the reference WAVs with:

```bash
./.venv/bin/python scripts/generate_quality_reference_set.py \
  --base-url http://127.0.0.1:8002
```

This writes generated assets to:

- `bench_assets/quality_refs/generated/*.wav`
- `bench_assets/quality_refs/generated/*.headers.txt`
- `bench_assets/quality_refs/generated/*.txt`
- `bench_assets/quality_refs/generated/index.json`

Recommended quality baseline settings:

- `mode=design`
- `num_step=16`
- `guidance_scale=2.0`

Suggested workflow after each optimization phase:

1. regenerate the reference set
2. listen to a few anchor languages such as English, Chinese, Arabic, and Hindi
3. compare against the previously generated WAVs before moving to the next phase
