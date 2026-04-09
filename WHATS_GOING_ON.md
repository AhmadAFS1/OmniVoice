# What's Going On In OmniVoice

This document is a repo-wide walkthrough meant to answer a simple question:

What is this codebase doing, where does each part live, and how do the pieces fit together?

Scope:

- This guide is based on the tracked project files in `/OmniVoice`.
- It focuses on the actual repo, not `.git`, `.venv`, or `__pycache__` directories.
- It includes source files, docs, example scripts, config files, generated maps, and empty package markers.

## The Big Picture

OmniVoice is a Python-first multilingual text-to-speech project with four main layers:

1. Inference layer
   `omnivoice/models/omnivoice.py` is the real engine. It loads the model, prepares prompts, runs iterative token generation, handles long-text chunking, and decodes audio.

2. User entry points
   `omnivoice/cli/` provides the public command-line programs:
   `omnivoice-demo`, `omnivoice-api`, `omnivoice-infer`, `omnivoice-infer-batch`, and the training entry point.

3. Training/data layer
   `omnivoice/data/` and `omnivoice/training/` define how tokenized datasets are read, packed, collated, and fed into distributed training.

4. Preprocessing and evaluation layer
   `omnivoice/scripts/` builds datasets and token shards.
   `omnivoice/eval/` computes WER, speaker similarity, and UTMOS on benchmark sets.

In other words:

- If you want to synthesize speech, the center of gravity is `OmniVoice.generate(...)`.
- If you want a UI, use `omnivoice/cli/demo.py`.
- If you want HTTP access, use `omnivoice/cli/api_server.py`.
- If you want repeatable large-scale batch runs, use `omnivoice/cli/infer_batch.py`.
- If you want to train or fine-tune, the stack is `examples/` -> `training/config.py` -> `training/builder.py` -> `training/trainer.py`.
- If you want to build data, the stack is `scripts/jsonl_to_webdataset.py` and `scripts/extract_audio_tokens*.py`.

## How The Main Workflows Fit Together

### 1. Local demo flow

`omnivoice-demo`

-> loads `OmniVoice.from_pretrained(...)`

-> Gradio UI collects text, language, reference audio, optional instruct, and generation options

-> UI builds `OmniVoiceGenerationConfig`

-> clone mode uses `create_voice_clone_prompt(...)`

-> calls `model.generate(...)`

-> converts the output waveform to `int16` for browser playback

### 2. Single inference flow

`omnivoice-infer`

-> loads one model instance

-> forwards CLI args directly into `model.generate(...)`

-> writes one WAV file with `torchaudio.save(...)`

### 3. HTTP API flow

`omnivoice-api`

-> loads one shared model instance at startup

-> exposes `/health`, `/languages`, and `/generate`

-> validates mode-specific request inputs

-> serializes uploaded reference audio to a temp file for clone mode

-> guards generation behind a lock so concurrent requests do not step on shared GPU state

-> returns WAV bytes directly in the HTTP response

### 4. Batch inference flow

`omnivoice-infer-batch`

-> reads JSONL with `read_test_list(...)`

-> estimates reference + target durations

-> groups work into duration-aware or fixed-size batches

-> spawns one or more workers per GPU

-> each worker loads `OmniVoice.from_pretrained(...)` once

-> batched `model.generate(...)` runs in parallel

-> saves WAVs and reports aggregate real-time factor

### 5. Training flow

`accelerate launch -m omnivoice.cli.train`

-> `TrainingConfig.from_json(...)`

-> `build_model_and_tokenizer(...)`

-> `prepare_data_manifests_from_json(...)`

-> `WebDatasetReader` + `PackingIterableDataset` + `PackingDataCollator`

-> `OmniTrainer.train()`

-> `OmniVoice.forward(...)` computes masked-token loss

-> checkpoints and TensorBoard logs are written during training

### 6. Data preparation flow

raw JSONL or existing WebDataset

-> optional `denoise_audio.py`

-> optional `extract_audio_tokens_add_noise.py`

-> or plain `extract_audio_tokens.py`

-> writes token tar shards + paired JSONL metadata + `data.lst`

-> training config points at those `data.lst` manifests

### 7. Evaluation flow

generated WAVs from `omnivoice.cli.infer_batch`

-> WER scripts in `omnivoice/eval/wer/`

-> speaker similarity in `omnivoice/eval/speaker_similarity/sim.py`

-> MOS prediction in `omnivoice/eval/mos/utmos.py`

## Core Architectural Ideas

### OmniVoice is an audio-token language model wrapper around an LLM backbone

The model combines:

- a Hugging Face text model backend (`self.llm`)
- separate audio token embeddings (`self.audio_embeddings`)
- a linear audio prediction head (`self.audio_heads`)
- an audio tokenizer (`HiggsAudioV2TokenizerModel`) for encoding/decoding waveforms

During training:

- text and style tokens are part of the input
- audio tokens are masked
- the model predicts masked audio tokens

During inference:

- the target audio region is initialized entirely with mask tokens
- the model repeatedly fills them in over `num_step` decoding rounds
- classifier-free guidance compares conditional and unconditional passes
- the final tokens are decoded back to waveform audio

### The repo supports three inference modes with one API

All of these go through `OmniVoice.generate(...)`:

- voice cloning: supply reference audio and optionally reference text
- voice design: supply `instruct`
- auto voice: supply neither and let the model pick a voice

### Long-text support is built into the model layer

Long-form generation is not a separate feature bolted on in the UI.
The model code estimates output duration, decides whether chunking is needed, splits text with punctuation-aware logic, generates chunk by chunk, and stitches chunks with cross-fades.

## Root Files

### `README.md`

This is the main public-facing project overview. It explains installation, quick start, Python API usage, CLI tools, supported features like voice cloning and voice design, and points readers to the training/evaluation docs. It is the best high-level starting point for a new developer.

### `pyproject.toml`

This defines OmniVoice as a Python package named `omnivoice` with Python `>=3.10`. Important points:

- main dependencies include `torch`, `torchaudio`, `transformers`, `accelerate`, `pydub`, `gradio`, `numpy`, and `soundfile`
- optional `eval` extras pull in evaluation-only packages such as `jiwer`, `librosa`, `s3prl`, `funasr`, `zhconv`, `zhon`, and `unidecode`
- console scripts expose:
  - `omnivoice-api`
  - `omnivoice-infer`
  - `omnivoice-infer-batch`
  - `omnivoice-demo`
- `uv` is configured to pin PyTorch and route Linux/Windows installs to the CUDA 12.8 wheel index

### `uv.lock`

This is the dependency lockfile generated by `uv`. It is not handwritten application logic. It records exact package resolutions, hashes, platform markers, and alternate wheel sources so installs are reproducible across machines.

### `LICENSE`

The project uses the Apache License 2.0. Some individual source files also note copied or adapted code from external projects, especially in evaluation utilities and model definitions.

## `docs/`

### `docs/data_preparation.md`

This explains the main dataset format used for training. The repo expects custom WebDataset-style shards:

- tar files contain tokenized audio arrays
- paired JSONL files contain metadata
- `data.lst` links shard tar files to shard metadata and records counts/duration

It also documents the token extraction pipeline and the JSON shape expected from raw manifests.

### `docs/data_preparation_advanced.md`

This extends the basic data prep story with two optional stages:

- denoising with Sidon
- prompt noise/RIR augmentation during token extraction

This doc maps directly to `denoise_audio.py` and `extract_audio_tokens_add_noise.py`.

### `docs/evaluation.md`

This is the evaluation overview. It explains supported test sets, required extras, and the three main metrics:

- WER or CER for intelligibility
- speaker similarity
- UTMOS for predicted MOS

### `docs/generation-parameters.md`

This is the parameter reference for `OmniVoice.generate(...)` and `OmniVoiceGenerationConfig`. It is the human-readable companion to the generation config dataclass in `omnivoice/models/omnivoice.py`.

### `docs/languages.md`

This is the human-friendly supported language table. It lists 646 languages and includes:

- display name
- OmniVoice language ID
- ISO 639-3 code
- training hours

It is documentation, not code, but it is important because language handling is a major feature of the project.

### `docs/lang_id_name_map.tsv`

This is the machine-readable language mapping source. It drives:

- the generated language map in `omnivoice/utils/lang_map.py`
- evaluation language conversions in `minimax.py` and `fleurs.py`

It includes language ID, language name, ISO 639-3 code, and training duration.

### `docs/training.md`

This is the short guide for launching training with `accelerate`, resuming from checkpoints, and initializing from pretrained weights. It complements the example shell scripts and JSON configs.

### `docs/voice-design.md`

This is the reference for voice-design instructions. It documents the valid attribute vocabulary for:

- gender
- age
- pitch
- whisper style
- English accents
- Chinese dialects

It matches the normalization logic in `omnivoice/utils/voice_design.py` and `_resolve_instruct(...)` in the model.

## `examples/`

### `examples/README.md`

This is the workflow guide for training from scratch, fine-tuning, and evaluation. It tells users which example script maps to which use case and what data is expected.

### `examples/run_emilia.sh`

This is the end-to-end example for training on Emilia. It is structured by stages:

- stage 0 checks raw data and manifests
- stage 1 tokenizes train/dev splits into WebDataset shards
- stage 2 launches distributed training

It is a practical script, not a general library entry point.

### `examples/run_finetune.sh`

This is the fine-tuning version of the pipeline. It tokenizes one train JSONL and one optional dev JSONL, then launches `omnivoice.cli.train`.

### `examples/run_eval.sh`

This is the benchmark orchestrator. It:

- downloads test sets and evaluation models
- runs `omnivoice.cli.infer_batch`
- runs speaker similarity, WER, and UTMOS evaluation scripts

It also includes special handling for the Emilia checkpoint and a note that FLEURS evaluation needs a separate environment for omnilingual-asr.

### `examples/config/data_config_emilia.json`

Training/dev manifest configuration for Emilia English and Chinese shards. This is consumed by `prepare_data_manifests_from_json(...)`.

### `examples/config/data_config_finetune.json`

Minimal train/dev manifest config for the fine-tuning pipeline.

### `examples/config/ds_config_zero2.json`

DeepSpeed ZeRO stage-2 configuration template. The main trainer supports DeepSpeed through Accelerate, although the default examples do not require it.

### `examples/config/train_config_emilia.json`

Training hyperparameters for the Emilia recipe. Notable choices:

- no language or instruct conditioning ratios
- no init checkpoint
- 300k steps
- bf16 mixed precision

### `examples/config/train_config_finetune.json`

Fine-tuning hyperparameters. It differs from the Emilia config mainly by:

- `init_from_checkpoint = "k2-fsa/OmniVoice"`
- fewer steps
- lower learning rate
- stronger language conditioning ratio

### `examples/config/train_config_multilingual.json`

A more feature-complete multilingual training config with language, pinyin, and instruct usage turned on. This reads like the full training recipe for the general multilingual model rather than a narrow fine-tune.

## `omnivoice/`

### `omnivoice/__init__.py`

This is the public package surface. It:

- suppresses a few noisy warnings
- tries to expose package version metadata
- re-exports `OmniVoice`, `OmniVoiceConfig`, and `OmniVoiceGenerationConfig`

If someone writes `from omnivoice import OmniVoice`, this file makes that work.

## `omnivoice/cli/`

### `omnivoice/cli/__init__.py`

Empty package marker. It only makes `omnivoice.cli` importable.

### `omnivoice/cli/demo.py`

This is the Gradio UI server. It is intentionally a thin wrapper over the model API.

What it does:

- auto-detects device
- loads the model once
- defines the supported language dropdown from `LANG_NAMES`
- builds a Gradio Blocks UI with two tabs:
  - voice cloning
  - voice design
- collects generation settings into `OmniVoiceGenerationConfig`
- clone mode calls `model.create_voice_clone_prompt(...)`
- both tabs call `model.generate(...)`
- converts the result to `int16` NumPy audio for Gradio playback

What it does not do:

- it does not implement model logic itself
- it is not the HTTP API layer; that now lives in `omnivoice/cli/api_server.py`

### `omnivoice/cli/api_server.py`

This is the FastAPI wrapper around the existing OmniVoice inference API.

What it does:

- loads a single shared model instance during app creation
- exposes:
  - `GET /health`
  - `GET /languages`
  - `POST /generate`
- accepts three modes:
  - `auto`
  - `design`
  - `clone`
- validates mode-specific fields before generation
- converts uploaded reference audio into a temporary file path for clone mode
- builds `OmniVoiceGenerationConfig`
- calls the same `model.generate(...)` path as the UI and CLI tools
- returns raw WAV bytes via FastAPI
- uses a `Lock` to serialize generation requests against one shared model instance

This file is important because it turns the library into an app-facing HTTP service without reimplementing synthesis logic.

### `omnivoice/cli/infer.py`

This is the simple one-shot CLI. It is the most direct mapping from CLI flags to `model.generate(...)`. Good for scripting or sanity checks.

### `omnivoice/cli/infer_batch.py`

This is the large-scale batch inference path. Important responsibilities:

- reads a JSONL test list with `read_test_list(...)`
- estimates total generation duration with `RuleDurationEstimator`
- clusters samples by total duration or fixed batch size
- spawns worker processes per GPU
- each worker loads a full OmniVoice instance only once
- performs batched generation
- saves generated WAVs
- reports total synthesis time, audio duration, and average RTF

This file is important for both production-style batch generation and benchmark pipelines.

### `omnivoice/cli/train.py`

This is the training entry point. It is intentionally small:

- parse paths
- load `TrainingConfig`
- build model/tokenizer
- build dataloaders
- instantiate `OmniTrainer`
- call `trainer.train()`

The real training logic lives in `omnivoice/training/`.

## `omnivoice/data/`

### `omnivoice/data/__init__.py`

Empty package marker.

### `omnivoice/data/batching.py`

This file defines iterable dataset wrappers that make streaming variable-length data trainable efficiently.

`StreamLengthGroupDataset`

- used mainly in preprocessing
- groups samples into duration buckets
- yields groups whose combined duration stays under a target batch duration

`PackingIterableDataset`

- used for training
- processes raw samples one by one
- keeps packing them until `batch_tokens` would be exceeded
- yields batches based on real token count, not fixed sample count

### `omnivoice/data/collator.py`

This builds the final packed tensor batch for training. It:

- concatenates processed samples
- pads to `batch_tokens`
- produces `input_ids`, `labels`, `audio_mask`, `position_ids`, and `document_ids`

`document_ids` matters because the model uses block masking so attention stays inside each packed sample.

### `omnivoice/data/dataset.py`

This is the core dataset IO file.

Important pieces:

`prepare_data_manifests_from_json(...)`

- reads the training data config JSON
- expands train/dev manifest lists
- supports repeat factors for balancing datasets

`webdataset_manifest_reader(...)`

- reads the repo's custom `data.lst` format

`SampleDecoder`

- decodes either audio tokens (`.npy`) or raw audio (`wav`/`flac`/`mp3`)
- loads the paired JSONL label record for the current shard

`LabelDataset`

- in-memory lookup for shard metadata by sample ID

`WebDatasetReader`

- the main iterable reader for tokenized shard data
- supports shuffle by epoch
- feeds training/eval loaders

`JsonlDatasetReader`

- reads raw JSONL data directly from `audio_path`
- used heavily by preprocessing scripts

`MuxWebDatasetReader` and `LazyIteratorMultiplexer`

- combine multiple readers with weighted random sampling
- useful when mixing multilingual datasets

### `omnivoice/data/processor.py`

This file turns a raw sample into a masked training example.

`OmniVoiceSampleProcessor`

- injects style prompt tokens:
  - optional denoise tag
  - language tag
  - instruct tag
- optionally swaps in pinyin text
- chooses prompt ratio and mask ratio
- can drop conditioning entirely for unconditional training
- masks the target audio token region
- builds the final `input_ids`, `labels`, and `audio_mask`

`OmniVoiceSimpleSampleProcessor`

- a reduced reference implementation
- kept for understanding the core training logic
- not the main processor used in training

This file is one of the key places to understand how the training objective is constructed.

## `omnivoice/models/`

### `omnivoice/models/__init__.py`

Empty package marker.

### `omnivoice/models/omnivoice.py`

This is the heart of the repo.

Important data structures:

`VoiceClonePrompt`

- reusable prompt bundle for cloning
- contains reference audio tokens, reference text, and reference RMS

`OmniVoiceGenerationConfig`

- generation knobs such as steps, guidance scale, temperatures, chunk sizes, and pre/post-processing flags

`GenerationTask`

- normalized per-batch inference payload used internally by generation

`OmniVoiceConfig`

- Hugging Face config wrapper for the OmniVoice model
- stores audio vocabulary size, mask token, codebook count, codebook weights, and nested LLM config

`OmniVoice`

- subclasses `PreTrainedModel`
- owns the LLM backbone, audio embeddings, and audio prediction head

Key training-side behavior:

- `_prepare_embed_inputs(...)` merges text embeddings and audio token embeddings
- `forward(...)` runs the LLM and projects hidden states into audio-token logits
- loss is computed only over masked audio token positions
- codebook layers are weighted using normalized `audio_codebook_weights`

Key inference-side behavior:

`from_pretrained(...)`

- loads pretrained model weights
- in inference mode, also loads:
  - text tokenizer
  - Higgs audio tokenizer
  - audio feature extractor
  - duration estimator
  - optional ASR model

`load_asr_model(...)` and `transcribe(...)`

- provide optional Whisper-based reference transcription

`generate(...)`

- main public synthesis API
- normalizes inputs through `_preprocess_all(...)`
- splits batch into short vs long items
- short items use `_generate_iterative(...)`
- long items use `_generate_chunked(...)`
- final outputs are decoded and post-processed

`create_voice_clone_prompt(...)`

- loads or resamples reference audio
- adjusts RMS if the reference is very quiet
- optionally trims long audio and removes silence
- auto-transcribes if `ref_text` is missing
- encodes reference audio into audio tokens

`_preprocess_all(...)`

- normalizes single values vs lists
- resolves language names to IDs
- validates and normalizes voice-design instructions
- creates voice clone prompts if only raw reference audio was passed
- estimates target token counts
- handles duration overrides vs speed factors

`_prepare_inference_inputs(...)`

- builds the actual model input sequence for one item:
  - style prompt
  - text prompt
  - optional reference audio tokens
  - masked target region

`_generate_iterative(...)`

- the actual decoding loop
- builds conditional and unconditional batches for classifier-free guidance
- predicts logits
- scores candidate tokens
- selects the next positions to unmask
- iterates for `num_step` rounds

`_generate_chunked(...)`

- handles long-form generation
- splits text by punctuation-aware rules
- batches chunk generation across items when possible
- uses the first generated chunk as reference in no-reference long-form mode

`_decode_and_post_process(...)` and `_post_process_audio(...)`

- decode token tensors back to waveforms
- optionally remove long silences
- normalize against reference RMS
- add fades and padding
- cross-fade chunk boundaries for long outputs

Standalone helpers at the bottom:

- `_resolve_language(...)`
- `_resolve_instruct(...)`
- `_tokenize_with_nonverbal_tags(...)`
- `_combine_text(...)`
- sampling helpers for top-k, Gumbel noise, and timestep scheduling

If someone on the team wants to understand OmniVoice at the deepest level, this is the file to study first.

## `omnivoice/scripts/`

### `omnivoice/scripts/__init__.py`

Empty package marker.

### `omnivoice/scripts/jsonl_to_webdataset.py`

This converts raw JSONL audio manifests into the repo's custom WebDataset format with:

- FLAC tar shards
- paired shard JSONL metadata
- `data.lst`
- `errors.jsonl`

It uses a process pool plus per-process thread pools to load, resample, and pack audio efficiently.

### `omnivoice/scripts/extract_audio_tokens.py`

This is the main tokenization pipeline for training data.

What it does:

- accepts either raw JSONL or existing `data.lst`
- streams samples through `JsonlDatasetReader` or `WebDatasetReader`
- filters by audio duration
- uses the Higgs audio tokenizer to encode waveforms into 8 codebook token streams
- writes `.npy` token payloads into tar shards
- writes cleaned metadata to paired JSONL shard files
- builds a new `data.lst`

This is the bridge from raw audio to trainable token data.

### `omnivoice/scripts/extract_audio_tokens_add_noise.py`

This is the augmentation-aware version of token extraction.

In addition to the base tokenization pipeline, it can:

- sample environmental noise from a noise WebDataset
- sample room impulse responses from an RIR WebDataset
- apply augmentation only to the front part of the waveform
- record `clean_start_token_idx` in metadata so training knows where clean tokens begin

That metadata is later used by `OmniVoiceSampleProcessor` for prompt-denoising behavior.

### `omnivoice/scripts/denoise_audio.py`

This is the most infrastructure-heavy preprocessing script.

What it does:

- denoises audio with TorchScripted Sidon components
- supports raw JSONL input or existing `data.lst`
- dynamically groups samples by duration
- uses a subprocess-based GPU worker pool so CUDA device selection happens before PyTorch initializes CUDA state
- writes denoised FLAC tar shards + paired JSONL metadata + `data.lst`

Important internal pieces:

- `extract_seamless_m4t_features(...)`: Torch-only feature extractor
- `SpeechDenoisingProcessor`: loads Sidon TorchScript modules and runs denoising
- `StreamLengthGroupDataset`: reused to build efficient variable-length batches
- `_GPUWorker` and `GPUWorkerPool`: robust multi-worker GPU orchestration

This file exists because denoising large audio corpora efficiently is a systems problem, not just a model call.

## `omnivoice/training/`

### `omnivoice/training/__init__.py`

Empty package marker.

### `omnivoice/training/builder.py`

This constructs the trainable system from config.

`build_model_and_tokenizer(...)`

- loads tokenizer from either the base LLM or init checkpoint
- adds OmniVoice-specific special tokens
- either loads a pretrained OmniVoice checkpoint or builds a new OmniVoice model around a base LLM
- resizes token embeddings if needed

`build_dataloaders(...)`

- creates `OmniVoiceSampleProcessor`
- loads manifest lists from the data config
- builds `WebDatasetReader`
- wraps it with `PackingIterableDataset`
- uses `PackingDataCollator`
- returns train and optional eval `DataLoader`s

### `omnivoice/training/checkpoint.py`

This file handles checkpointing and logging.

`TrainLogger`

- owns the progress bar
- pushes metrics to Accelerate trackers
- formats concise console logging

`save_checkpoint(...)`

- saves optimizer/scheduler/RNG state through Accelerate
- saves the model in Hugging Face format
- saves the tokenizer
- rotates old checkpoints if configured

`load_checkpoint(...)`

- restores state and infers the step number from the checkpoint directory name

### `omnivoice/training/config.py`

This is the training hyperparameter dataclass. It defines:

- model/audio token settings
- prompt masking ratios
- language/pinyin/instruct ratios
- optimizer and scheduler values
- mixed precision and DeepSpeed options
- logging, eval, and save intervals

This file is the schema behind the example JSON configs.

### `omnivoice/training/trainer.py`

This is the actual training loop wrapper around Hugging Face Accelerate.

Responsibilities:

- initialize Accelerator
- optionally configure DeepSpeed
- set up optimizer and LR scheduler
- prepare model/optimizer/scheduler
- resume from checkpoint if requested
- iterate endlessly over the streaming dataloader
- accumulate gradients
- clip gradients
- step optimizer and scheduler
- periodically log, evaluate, and save

This file is intentionally generic. Most model-specific training behavior is pushed into the model forward pass and sample processor.

## `omnivoice/utils/`

### `omnivoice/utils/__init__.py`

Empty package marker.

### `omnivoice/utils/audio.py`

Shared audio helpers used mostly during inference and prompt preparation.

It provides:

- `load_audio(...)`
- silence removal helpers
- tensor <-> `AudioSegment` conversion
- fade and pad
- trimming long reference audio at silence boundaries
- cross-fading chunk outputs

### `omnivoice/utils/common.py`

Small shared helpers:

- `str2bool(...)` for CLI flags
- `fix_random_seed(...)` for deterministic seeding

### `omnivoice/utils/data_utils.py`

This reads the JSONL test-list format used by batch inference and evaluation. It is small but central because many tools rely on that exact schema.

### `omnivoice/utils/duration.py`

This file estimates speech duration from text using heuristic script-aware weights across many writing systems. It is used during generation to predict target token lengths and decide when chunking is necessary.

### `omnivoice/utils/lang_map.py`

This is the generated Python language map derived from `docs/lang_id_name_map.tsv`.

It provides:

- `LANG_NAME_TO_ID`
- `LANG_NAMES`
- `LANG_IDS`
- `lang_display_name(...)`

This is used by the model and the Gradio UI to resolve and display languages.

### `omnivoice/utils/text.py`

This handles text-side utilities for inference:

- `chunk_text_punctuation(...)` splits long text by punctuation while avoiding common abbreviation mistakes
- `add_punctuation(...)` appends sentence-ending punctuation when missing

This file is especially important for stable long-form generation.

### `omnivoice/utils/voice_design.py`

This defines the allowed voice-design attribute vocabularies in English and Chinese, plus normalization maps and mutual-exclusion sets. The validation logic in `_resolve_instruct(...)` depends on these constants.

## `omnivoice/eval/`

### `omnivoice/eval/__init__.py`

Tiny package initializer that suppresses some warning noise during evaluation.

### `omnivoice/eval/utils.py`

Shared audio loading helper for evaluation. It uses `soundfile` and `librosa`, supports truncation, and returns either NumPy arrays or tensors.

## `omnivoice/eval/models/`

### `omnivoice/eval/models/ecapa_tdnn_wavlm.py`

This is the speaker embedding model used for SIM-o evaluation.

At a high level:

- loads WavLM features, optionally from a local checkpoint
- learns weighted combinations of hidden layers
- applies an ECAPA-TDNN-style network
- outputs speaker embeddings for cosine similarity

This is model-definition code, not business logic.

### `omnivoice/eval/models/utmos.py`

This contains the UTMOS strong learner network definition used for predicted MOS scoring. It includes:

- a wav2vec2-style feature extractor
- a transformer encoder
- a BLSTM
- a projection head to score utterances

Again, this is mostly architecture code used by the evaluation script.

## `omnivoice/eval/mos/`

### `omnivoice/eval/mos/utmos.py`

This is the executable MOS evaluator. It:

- loads the UTMOS model weights
- spawns GPU workers
- scores each generated WAV
- aggregates overall and per-language averages

## `omnivoice/eval/speaker_similarity/`

### `omnivoice/eval/speaker_similarity/sim.py`

This computes SIM-o by comparing reference and generated speaker embeddings.

It:

- loads the ECAPA-TDNN + WavLM speaker model
- reads reference/generated wav pairs from the test list
- computes cosine similarity
- aggregates average and per-language SIM-o

## `omnivoice/eval/wer/`

### `omnivoice/eval/wer/common.py`

Shared WER utilities:

- `process_one(...)` runs normalization and computes jiwer stats
- `log_metrics(...)` prints weighted error summaries

### `omnivoice/eval/wer/text_norm_omni.py`

Normalization helper copied from Meta's omnilingual-asr tooling. It lowercases, strips punctuation, applies mapping rules, optionally removes numbers, and normalizes whitespace.

### `omnivoice/eval/wer/norm_config_module.py`

Large normalization config table, also copied from omnilingual-asr. It defines language-specific punctuation, deletion, mapping, and digit rules, and loads `punctuations.lst`.

### `omnivoice/eval/wer/punctuations.lst`

Resource file loaded by `norm_config_module.py`. It is a tab-separated punctuation inventory used to extend the normalization rules.

### `omnivoice/eval/wer/hubert.py`

English LibriSpeech-PC evaluator using a Hubert ASR pipeline. It parallelizes transcription across GPUs, then computes weighted WER.

### `omnivoice/eval/wer/seedtts.py`

Seed-TTS evaluator:

- Whisper for English
- Paraformer for Chinese

It computes both average-of-item WER and weighted WER.

### `omnivoice/eval/wer/sensevoice.py`

Cantonese evaluation path using SenseVoiceSmall. It filters the test list down to `language_id == "yue"` and computes CER-style metrics through the shared WER machinery.

### `omnivoice/eval/wer/minimax.py`

Multilingual evaluation script for the MiniMax set.

What makes it different:

- maps OmniVoice language IDs through `docs/lang_id_name_map.tsv`
- routes Chinese to Paraformer and other languages to Whisper
- applies language-aware normalization rules
- aggregates per-language and macro-average WER/CER

### `omnivoice/eval/wer/fleurs.py`

FLEURS multilingual evaluator using omnilingual-asr. It also:

- maps language IDs between OmniVoice and omnilingual-asr conventions
- contains extra spacing cleanup for CJK text
- reports per-language, macro-average, and threshold-count statistics

This script is the reason `examples/run_eval.sh` recommends a separate environment for FLEURS evaluation.

## Empty Package Marker Files

The following files are intentionally empty and exist only so Python treats the directories as packages:

- `omnivoice/cli/__init__.py`
- `omnivoice/data/__init__.py`
- `omnivoice/models/__init__.py`
- `omnivoice/scripts/__init__.py`
- `omnivoice/training/__init__.py`
- `omnivoice/utils/__init__.py`

They are not missing logic.

## Files That Are Generated, Copied, Or Mostly Data

These are important, but they are not where the repo's handwritten control flow lives:

- `uv.lock`
  Generated dependency lockfile.

- `docs/lang_id_name_map.tsv`
  Source data table for languages.

- `omnivoice/utils/lang_map.py`
  Generated Python mapping from the TSV.

- `omnivoice/eval/wer/text_norm_omni.py`
  Copied normalization code from omnilingual-asr.

- `omnivoice/eval/wer/norm_config_module.py`
  Copied normalization config module.

- `omnivoice/eval/models/ecapa_tdnn_wavlm.py`
  Speaker model architecture definition adapted from prior work.

- `omnivoice/eval/models/utmos.py`
  UTMOS model architecture implementation.

## If You Want To Modify Something, Start Here

### Add or change synthesis behavior

Start with:

- `omnivoice/models/omnivoice.py`
- `omnivoice/utils/audio.py`
- `omnivoice/utils/text.py`
- `omnivoice/utils/voice_design.py`

### Change the UI

Start with:

- `omnivoice/cli/demo.py`

### Extend the API server

The current workspace already includes a FastAPI server in:

- `omnivoice/cli/api_server.py`

Its real dependencies are still the same core model methods:

- `OmniVoice.from_pretrained(...)`
- `create_voice_clone_prompt(...)`
- `generate(...)`

### Change training behavior

Start with:

- `omnivoice/data/processor.py`
- `omnivoice/training/builder.py`
- `omnivoice/training/trainer.py`
- `omnivoice/models/omnivoice.py`

### Change dataset preprocessing

Start with:

- `omnivoice/scripts/jsonl_to_webdataset.py`
- `omnivoice/scripts/extract_audio_tokens.py`
- `omnivoice/scripts/extract_audio_tokens_add_noise.py`
- `omnivoice/scripts/denoise_audio.py`

### Change benchmark logic

Start with:

- `omnivoice/eval/wer/`
- `omnivoice/eval/speaker_similarity/sim.py`
- `omnivoice/eval/mos/utmos.py`

## Practical Mental Model For The Repo

If you only remember one model of the codebase, use this:

- `omnivoice/models/omnivoice.py` is the engine
- `omnivoice/cli/` is how users touch the engine
- `omnivoice/data/` and `omnivoice/training/` are how the engine learns
- `omnivoice/scripts/` are how raw data becomes trainable data
- `omnivoice/eval/` is how the team measures whether the engine is good
- `docs/` and `examples/` are the human-readable operating manual

That is what is going on in this repo.
