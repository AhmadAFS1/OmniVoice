# Throughput Metrics

This document records the throughput and latency measurements collected during
the API-server testing session in this chat.

Important scope note:

- All measurements below are from `mode=design`
- These are not voice-cloning benchmarks
- These are single-request measurements on the current API server
- The current server serializes inference with a global generate lock, so these
  runs do not reflect dynamic batching or concurrent serving performance

Relevant implementation detail:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:355)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:551)

## Test Request

Most measurements in this document used:

```text
mode=design
text=This is a benchmark request for comparing RTX 3080 and RTX 3090 inference.
instruct=female, low pitch, british accent
guidance_scale=2.0
```

The later repeated benchmark additionally fixed:

```text
duration=6.2
postprocess_output=false
```

## High-Level Findings

- The current API server is effectively one-request-at-a-time.
- The shell loops used in testing were sequential.
- The server itself also serializes `model.generate(...)` with a lock.
- In the 20-run fixed-duration benchmark, RTX 3080 and RTX 3090 were nearly tied.
- Under this current single-request path, the 3090 does not clearly separate
  from the 3080.
- This likely means the current benchmark is not saturating the GPU strongly
  enough to expose the 3090's larger headroom.

## Phase 1: Early One-Off Runs

These runs were taken before the system was fully warmed up, so they are useful
as context but should not be treated as the cleanest comparison.

| GPU | Steps | Latency (ms) | Audio Duration (s) | RTF |
|---|---:|---:|---:|---:|
| RTX 3090 | 16 | 1624.62 | 5.950 | 0.2730 |
| RTX 3090 | 32 | 1035.52 | 5.580 | 0.1856 |
| RTX 3080 | 16 | 1369.36 | 6.090 | 0.2249 |
| RTX 3080 | 32 | 1071.84 | 5.730 | 0.1871 |

Notes:

- These results showed obvious warmup effects.
- `32-step` appearing faster than `16-step` in some early samples is a strong
  sign that startup or kernel warmup cost was still dominating the first run.

## Phase 2: Warmed One-Off Runs

These runs were collected after warming the servers more.

| GPU | Steps | Latency (ms) | Audio Duration (s) | RTF |
|---|---:|---:|---:|---:|
| RTX 3080 | 16 | 589.73 | 6.120 | 0.0964 |
| RTX 3080 | 32 | 1109.90 | 5.180 | 0.2143 |
| RTX 3090 | 16 | 684.53 | 5.260 | 0.1301 |
| RTX 3090 | 32 | 1032.26 | 5.470 | 0.1887 |

Notes:

- These were much more believable than the first cold-sensitive samples.
- The one-off warmed numbers still had varying audio durations, so they were
  not yet perfect apples-to-apples comparisons.

## Phase 3: 20-Run Fixed-Duration Benchmark

This was the cleanest benchmark collected so far.

Conditions:

- `mode=design`
- `duration=6.2`
- `postprocess_output=false`
- sequential request loop
- current API server with serialization lock

### Aggregate Summary

| GPU | Steps | Runs | Mean Latency (ms) | Median Latency (ms) | Min (ms) | Max (ms) | Latency StdDev (ms) | Mean RTF | Median RTF | Min RTF | Max RTF | RTF StdDev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 3080 | 16 | 20 | 540.50 | 535.60 | 526.67 | 581.32 | 15.07 | 0.0872 | 0.0864 | 0.0849 | 0.0938 | 0.0024 |
| RTX 3080 | 32 | 20 | 1053.73 | 1047.37 | 1031.26 | 1116.26 | 23.54 | 0.1700 | 0.1689 | 0.1663 | 0.1800 | 0.0038 |
| RTX 3090 | 16 | 20 | 540.60 | 547.13 | 520.58 | 565.99 | 12.83 | 0.0872 | 0.0882 | 0.0840 | 0.0913 | 0.0021 |
| RTX 3090 | 32 | 20 | 1060.01 | 1065.86 | 1022.62 | 1105.74 | 22.84 | 0.1710 | 0.1719 | 0.1649 | 0.1783 | 0.0037 |

### Interpretation

- `16-step`: effectively tied
- `32-step`: effectively tied
- The differences are small enough that they look like noise-level variation
  under this current serving path
- The test currently reflects single-request latency much more than true
  throughput under queue pressure

Approximate throughput under this specific benchmark shape:

- `16-step`: about `1.85 requests/sec`
- `32-step`: about `0.95 requests/sec`

These are rough single-server, single-request-path numbers inferred from mean
latency, not a batched concurrency benchmark.

## Raw CSV: RTX 3080

```csv
host,steps,run,latency_ms,audio_duration_s,rtf
e9b27791c166,16,1,581.32,6.200,0.0938
e9b27791c166,32,1,1048.17,6.200,0.1691
e9b27791c166,16,2,535.65,6.200,0.0864
e9b27791c166,32,2,1046.57,6.200,0.1688
e9b27791c166,16,3,532.10,6.200,0.0858
e9b27791c166,32,3,1043.38,6.200,0.1683
e9b27791c166,16,4,536.00,6.200,0.0865
e9b27791c166,32,4,1042.74,6.200,0.1682
e9b27791c166,16,5,538.71,6.200,0.0869
e9b27791c166,32,5,1049.28,6.200,0.1692
e9b27791c166,16,6,535.55,6.200,0.0864
e9b27791c166,32,6,1064.91,6.200,0.1718
e9b27791c166,16,7,550.59,6.200,0.0888
e9b27791c166,32,7,1044.64,6.200,0.1685
e9b27791c166,16,8,536.95,6.200,0.0866
e9b27791c166,32,8,1100.91,6.200,0.1776
e9b27791c166,16,9,533.44,6.200,0.0860
e9b27791c166,32,9,1048.78,6.200,0.1692
e9b27791c166,16,10,535.89,6.200,0.0864
e9b27791c166,32,10,1040.27,6.200,0.1678
e9b27791c166,16,11,531.63,6.200,0.0857
e9b27791c166,32,11,1116.26,6.200,0.1800
e9b27791c166,16,12,534.87,6.200,0.0863
e9b27791c166,32,12,1049.25,6.200,0.1692
e9b27791c166,16,13,535.02,6.200,0.0863
e9b27791c166,32,13,1031.31,6.200,0.1663
e9b27791c166,16,14,580.48,6.200,0.0936
e9b27791c166,32,14,1049.68,6.200,0.1693
e9b27791c166,16,15,527.89,6.200,0.0851
e9b27791c166,32,15,1036.65,6.200,0.1672
e9b27791c166,16,16,526.67,6.200,0.0849
e9b27791c166,32,16,1102.79,6.200,0.1779
e9b27791c166,16,17,557.29,6.200,0.0899
e9b27791c166,32,17,1037.59,6.200,0.1674
e9b27791c166,16,18,537.01,6.200,0.0866
e9b27791c166,32,18,1040.17,6.200,0.1678
e9b27791c166,16,19,530.03,6.200,0.0855
e9b27791c166,32,19,1050.03,6.200,0.1694
e9b27791c166,16,20,532.99,6.200,0.0860
e9b27791c166,32,20,1031.26,6.200,0.1663
```

## Raw CSV: RTX 3090

```csv
host,steps,run,latency_ms,audio_duration_s,rtf
7d7877499228,16,1,565.99,6.200,0.0913
7d7877499228,32,1,1068.42,6.200,0.1723
7d7877499228,16,2,539.10,6.200,0.0870
7d7877499228,32,2,1073.69,6.200,0.1732
7d7877499228,16,3,547.60,6.200,0.0883
7d7877499228,32,3,1081.02,6.200,0.1744
7d7877499228,16,4,548.29,6.200,0.0884
7d7877499228,32,4,1063.35,6.200,0.1715
7d7877499228,16,5,535.11,6.200,0.0863
7d7877499228,32,5,1070.68,6.200,0.1727
7d7877499228,16,6,541.45,6.200,0.0873
7d7877499228,32,6,1073.87,6.200,0.1732
7d7877499228,16,7,547.75,6.200,0.0883
7d7877499228,32,7,1065.08,6.200,0.1718
7d7877499228,16,8,553.27,6.200,0.0892
7d7877499228,32,8,1053.74,6.200,0.1700
7d7877499228,16,9,548.90,6.200,0.0885
7d7877499228,32,9,1073.83,6.200,0.1732
7d7877499228,16,10,521.16,6.200,0.0841
7d7877499228,32,10,1057.31,6.200,0.1705
7d7877499228,16,11,547.92,6.200,0.0884
7d7877499228,32,11,1064.51,6.200,0.1717
7d7877499228,16,12,521.93,6.200,0.0842
7d7877499228,32,12,1028.18,6.200,0.1658
7d7877499228,16,13,550.82,6.200,0.0888
7d7877499228,32,13,1077.56,6.200,0.1738
7d7877499228,16,14,547.75,6.200,0.0883
7d7877499228,32,14,1066.63,6.200,0.1720
7d7877499228,16,15,536.98,6.200,0.0866
7d7877499228,32,15,1081.08,6.200,0.1744
7d7877499228,16,16,548.82,6.200,0.0885
7d7877499228,32,16,1105.74,6.200,0.1783
7d7877499228,16,17,546.67,6.200,0.0882
7d7877499228,32,17,1023.19,6.200,0.1650
7d7877499228,16,18,520.58,6.200,0.0840
7d7877499228,32,18,1026.41,6.200,0.1656
7d7877499228,16,19,520.98,6.200,0.0840
7d7877499228,32,19,1022.62,6.200,0.1649
7d7877499228,16,20,520.97,6.200,0.0840
7d7877499228,32,20,1023.30,6.200,0.1650
```

## Current Conclusion

For the current API server implementation and current benchmark shape:

- the two GPUs are effectively tied
- the current server path is serialized
- batching and concurrent request scheduling are the next major optimizations
- voice-cloning throughput still needs its own dedicated benchmark pass
