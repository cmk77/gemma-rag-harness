#!/usr/bin/env bash
# serving/vllm_launch.sh
#
# Gemma 4 E4B + MTP(Multi-Token Prediction) 추측 디코딩으로 서빙.
# 단일 GPU(RTX 6000 Ada 48GB) 기준.
#
# MTP란: 작은 초안 모델(assistant, ~0.15GB)이 여러 토큰을 미리 예측하고
#        본 모델(E4B)이 한 번에 검증 → 품질 손실 없이 생성 속도 향상.
#        초안 모델은 본 모델의 KV 캐시를 공유해 추가 VRAM이 거의 없다.
#
# 두 모델 모두 Apache-2.0 공개(gated 아님) → 토큰 없이 받을 수 있다.
#
# 사용: bash serving/vllm_launch.sh
#   더 큰 모델로 바꾸려면(품질↑ 속도↓):
#     MODEL=google/gemma-4-12B-it DRAFTER=google/gemma-4-12B-it-assistant bash serving/vllm_launch.sh
#     MODEL=google/gemma-4-26B-A4B-it DRAFTER=google/gemma-4-26B-A4B-it-assistant ...
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
DRAFTER="${DRAFTER:-google/gemma-4-E4B-it-assistant}"   # MTP 초안 모델
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"             # 컨텍스트 상한. BASE_K=10 + 재검색(문서 15개)
                                       # 기준으로 4096은 부족(실측 4097 초과 사고) → 8192
GPU_UTIL="${GPU_UTIL:-0.40}"           # E4B는 작아서 0.40이면 넉넉
NUM_SPEC="${NUM_SPEC:-4}"              # 초안이 미리 예측할 토큰 수
TP="${TP:-1}"

echo "[vllm] model=$MODEL drafter=$DRAFTER max_len=$MAX_LEN util=$GPU_UTIL spec_tokens=$NUM_SPEC port=$PORT"

# MTP 추측 디코딩 설정 (Gemma 4 assistant는 method=mtp 로 처리)
SPEC_CONFIG="{\"method\":\"mtp\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$NUM_SPEC}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TP" \
  --enable-prefix-caching \
  --served-model-name "$MODEL" \
  --speculative-config "$SPEC_CONFIG"

# ── 기동 확인 (다른 셸에서) ───────────────────────────────
#   curl http://localhost:8000/v1/models
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"google/gemma-4-E4B-it",
#          "messages":[{"role":"user","content":"안녕"}]}'
#
# 로그에서 SpeculativeConfig(method='mtp', ...) 가 보이면 MTP 정상 활성화.
