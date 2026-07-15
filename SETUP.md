# SETUP.md — WSL에서 처음부터 실행하기

Windows 11 + WSL2(Ubuntu) + Python venv 환경 기준 단계별 가이드.
프로젝트 위치: `/home/mozi/gemma-rag-harness` (= `\\wsl.localhost\Ubuntu\home\mozi\gemma-rag-harness`)

**하드웨어 기준**: RTX 6000 Ada 48GB · 128GB RAM · i9-14900K
→ Gemma 4 E4B + MTP(추측 디코딩)를 vLLM으로 서빙한다. E4B는 가벼워 VRAM 여유가 크고,
  MTP로 챗봇 수준 응답 속도가 나온다. **vLLM 단일 경로**를 표준으로 한다.

---

## 전체 그림

이 프로젝트를 돌리려면 3개의 서비스가 필요하다:

1. **Elasticsearch** — 문서를 검색 (Docker, nori 포함)
2. **vLLM 서빙** — Gemma 4 E4B + MTP가 답변을 생성 (GPU)
3. **하네스 본체** — 위 둘을 오케스트레이션 (이 Python 프로젝트, venv)

터미널을 여러 개 쓰게 된다: ES(Docker, 백그라운드), vLLM(한 터미널 점유),
하네스 실행(또 다른 터미널). vLLM은 포그라운드로 떠서 로그를 보여주므로
전용 터미널 하나를 차지한다.

---

## 0단계: WSL·GPU 기본 확인

Windows 터미널(PowerShell)에서:

```powershell
wsl --status          # WSL2인지 확인
wsl                   # Ubuntu 진입
```

Ubuntu 안에서 GPU가 보이는지 확인 (Windows에 최신 NVIDIA 드라이버 필요):

```bash
nvidia-smi
```

**RTX 6000 Ada**와 약 **49140MiB** 메모리가 보이면 준비 완료다.
여기서 GPU가 안 보이면 vLLM이 동작하지 않으므로, Windows NVIDIA 드라이버를
최신으로 업데이트한 뒤 `wsl --shutdown`으로 WSL을 재시작한다.

```bash
python3 --version     # 3.11+ 권장
```

---

## 1단계: 프로젝트 압축 풀기

`gemma-rag-harness.tar.gz`를 Windows 다운로드 폴더에서 WSL 홈으로 복사해 푼다.
WSL에서 Windows 파일은 `/mnt/c/...`로 접근한다.

```bash
cd ~
cp /mnt/c/Users/<윈도우사용자명>/Downloads/gemma-rag-harness.tar.gz ~/
tar -xzf gemma-rag-harness.tar.gz
cd gemma-rag-harness
ls   # CLAUDE.md README.md SETUP.md corpus/ docs/ eval/ harness/ ... 가 보이면 OK
```

---

> **집 노트북(Anaconda) 세팅?** conda 기반 절차는 [SETUP_HOME.md](SETUP_HOME.md)를 보세요.

## 2단계: Python 가상환경(venv) + 의존성

```bash
cd ~/gemma-rag-harness

python3 -m venv .venv
source .venv/bin/activate         # 프롬프트 앞에 (.venv) 표시 확인

pip install --upgrade pip
pip install -e ".[dev]"           # 핵심 의존성 + 개발도구
```

이 시점에 **모델·ES 없이 테스트가 통과하는지** 먼저 확인한다:

```bash
pytest              # 42개 전부 통과해야 정상
ruff check .        # 린트 통과
```

> 테스트는 스텁을 써서 모델/ES 없이 돈다. 여기까지 통과하면 코드 배선은 정상이다.
> **주의**: 앞으로 Python 명령(5·6·7단계)을 실행할 땐 항상 이 venv가 활성화돼
> 있어야 한다. 새 터미널을 열면 `source ~/gemma-rag-harness/.venv/bin/activate`로 다시 켠다.

---

## 3단계: Elasticsearch 기동 (Docker, nori 포함)

WSL에 Docker가 없으면 Docker Desktop(Windows)을 설치하고 설정에서 WSL2
integration을 켠다.

이 프로젝트는 한국어 코퍼스를 색인하므로 **nori 형태소 분석기**가 필요하다.
기본 ES 이미지에는 nori가 없어서, nori를 포함한 커스텀 이미지를 만들어 쓴다.

### nori 포함 이미지 빌드

```bash
cd ~/gemma-rag-harness

# Dockerfile 작성 (두 줄)
echo 'FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.0' > Dockerfile.es
echo 'RUN bin/elasticsearch-plugin install --batch analysis-nori' >> Dockerfile.es
cat Dockerfile.es   # 두 줄이 제대로 들어갔는지 확인

# 이미지 빌드 (nori 플러그인 다운로드·설치, 1~3분)
docker build -t es-nori -f Dockerfile.es .
# 마지막에 naming to docker.io/library/es-nori:latest 가 나오면 성공
```

### nori 이미지로 컨테이너 기동

```bash
docker run -d --name es \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  es-nori
```

### 3가지 확인

```bash
# (1) 컨테이너 살아있는지 — STATUS가 Up 이어야
docker ps

# (2) ES 응답 — "cluster_name":"docker-cluster" 가 나와야
curl http://localhost:9200

# (3) nori 작동 — 형태소로 쪼개져 나와야 (핵심 확인)
curl -X POST "http://localhost:9200/_analyze" \
  -H "Content-Type: application/json" \
  -d '{"analyzer":"nori","text":"한국어 형태소 분석 테스트"}'
```

(3)에서 "한국", "어", "형태", "소", "분석", "테스트" 같은 토큰이 나오면 nori 정상이다.

> 컨테이너가 `Exited`거나 안 보이면 `docker logs es`로 로그를 본다.
> WSL 메모리 부족이면 ES가 시작하다 죽을 수 있는데, 그 경우 ES_JAVA_OPTS를 줄이거나
> WSL 할당 메모리를 늘린다(.wslconfig).

---

## 4단계: 모델 서빙 — vLLM으로 Gemma 4 E4B + MTP (GPU)

Gemma 4 E4B(8B)에 MTP(Multi-Token Prediction) 초안 모델을 붙여 서빙한다.
MTP는 작은 초안 모델이 여러 토큰을 미리 예측하고 본 모델이 한 번에 검증하는
추측 디코딩으로, **품질 손실 없이 생성 속도를 높인다**. 챗봇 수준 응답이 나온다.

### 4-1. 모델 접근 — Gemma 4는 공개 모델 (토큰 불필요)

Gemma 4는 **Apache-2.0 라이선스 공개 모델**이라, Gemma 3와 달리 라이선스 동의나
토큰이 필요 없다. 바로 다운로드된다. (초안 모델도 공개)

- 본 모델: https://huggingface.co/google/gemma-4-E4B-it
- 초안 모델: https://huggingface.co/google/gemma-4-E4B-it-assistant

### 4-2. vLLM 설치

vLLM은 CUDA 의존이 커서 별도 extra로 분리해 뒀다. **Gemma 4 MTP는 vLLM 0.24+**
에서 지원한다:

```bash
cd ~/gemma-rag-harness
source .venv/bin/activate          # venv 활성화 확인
pip install -e ".[serving]"        # vLLM 설치 (용량 큼, 수 분 소요)
```

### 4-3. vLLM 서빙 기동

**이 명령은 전용 터미널 하나를 계속 점유한다**(포그라운드로 로그 출력).
새 터미널을 열어 venv 활성화 후 실행:

```bash
cd ~/gemma-rag-harness
source .venv/bin/activate

# E4B + MTP 서빙 (토큰 불필요, 공개 모델)
bash serving/vllm_launch.sh

# 더 큰 모델로 품질을 올리려면(속도는 느려짐):
#   MODEL=google/gemma-4-12B-it DRAFTER=google/gemma-4-12B-it-assistant bash serving/vllm_launch.sh
```

첫 실행 시 E4B 가중치(~15GB)와 초안 모델(~0.15GB)을 HuggingFace에서 받는다.
`~/.cache/huggingface`에 캐시되며 다음부터는 재사용한다.

기동 로그에서 **`SpeculativeConfig(method='mtp', ...)`** 가 보이면 MTP가 정상
활성화된 것이다. `Application startup complete` + `Uvicorn running on ...8000`이
나오면 서빙 준비 완료다.

**다른 터미널에서 서빙 확인:**

```bash
# 등록된 모델 확인
curl http://localhost:8000/v1/models

# 실제 생성 테스트 (MTP로 빠르게 응답)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"google/gemma-4-E4B-it",
       "messages":[{"role":"user","content":"한 문장으로 자기소개 해줘"}]}'

# GPU 점유 확인
nvidia-smi
```

> 두 번째 다운로드부터, 그리고 네트워크를 아예 안 쓰려면 `export HF_HUB_OFFLINE=1`을
> 앞에 붙이면 로컬 캐시만으로 뜬다(모델을 이미 받은 경우).

### 4-4. .env 설정

하네스가 이 엔드포인트를 쓰도록 설정한다:

```bash
cp .env.example .env
```

`.env` 내용 (기본값이 이미 E4B라 대부분 그대로):

```
LLM_BACKEND=vllm
LLM_MODEL=google/gemma-4-E4B-it
VLLM_URL=http://localhost:8000/v1
LLM_API_KEY=EMPTY
ES_URL=http://localhost:9200
ES_INDEX=gemma_rag
```

> `LLM_MODEL`은 vLLM에 뜬 모델명(`--served-model-name`)과 정확히 일치해야 한다.
> 12B로 서빙했다면 여기도 google/gemma-4-12B-it로 맞춘다.
> Gemma 4는 공개 모델이라 토큰 줄(HF_TOKEN)은 없어도 된다.

---

## 5단계: 코퍼스 색인

ES가 떠 있고(3단계) venv가 활성화된 상태에서, 검색할 문서를 색인한다.
(이 단계는 vLLM 없이도 된다 — 임베딩 모델만 쓴다.)

```bash
# 새 터미널 (vLLM 터미널은 그대로 두고)
cd ~/gemma-rag-harness
source .venv/bin/activate

# corpus/ 의 문서를 ES에 색인
# 첫 실행 시 bge-m3 임베딩 모델(약 2GB)을 자동 다운로드
python -m scripts.index_corpus

# 출력 예:
#   [index] 1개 문서 색인 시작...
#     - es_rag_guide.md
#   [index] 완료: N개 청크 색인됨
```

> `corpus/es_rag_guide.md`는 샘플 한국어 문서다. 실제로는 여기에 원하는 문서를 넣는다.

---

## 6단계: 질의 실행

vLLM(4단계) + ES(3단계) + 색인(5단계)이 모두 준비되면:

```bash
cd ~/gemma-rag-harness
source .venv/bin/activate

# 단일 질문
python -m scripts.ask "RRF의 rank_constant는 무슨 역할인가?"

# 대화형 모드
python -m scripts.ask
```

출력에 답변 + 실행 경로(`router → retrieve → generate → verify`) +
종료 사유(`stop_reason`) + 근거 문서가 표시된다. VERIFY가 근거 부족으로 판정하면
재검색 루프가 도는 것도 추적에서 볼 수 있다.

---

## 7단계: 평가 실행

```bash
cd ~/gemma-rag-harness
source .venv/bin/activate

# 배선만 점검 (모델 불필요)
python -m eval.run_regression --goldenset eval/goldenset.jsonl --dry-run

# 실제 평가 (vLLM + ES 필요) — 검색·생성 품질 측정
python -m eval.run_regression --goldenset eval/goldenset.jsonl

# 이번 결과를 베이스라인으로 저장 → 이후 회귀 감지에 사용
mkdir -p eval/reports
cp eval/reports/latest.json eval/reports/baseline.json
```

이후 코드를 바꾸고 다시 평가하면 베이스라인 대비 점수 하락을 회귀로 감지한다:

```bash
python -m eval.run_regression \
  --goldenset eval/goldenset.jsonl \
  --baseline eval/reports/baseline.json
# 핵심 지표 하락 시 exit 1
```

---

## 8단계: Git 초기화 (포트폴리오용)

```bash
cd ~/gemma-rag-harness
git init
git add .
git commit -m "feat: initial gemma-rag-harness scaffold"

git remote add origin https://github.com/<계정>/gemma-rag-harness.git
git branch -M main
git push -u origin main
```

`.gitignore`가 모델 가중치·venv·캐시를 제외하므로 안전하다.

---

## 실행 순서 요약 (매번 켤 때)

재부팅 후 다시 돌릴 때의 순서:

```bash
# 1) ES 컨테이너 시작 (한 번 만들었으면 start로 재사용)
docker start es
curl http://localhost:9200                    # 확인

# 2) vLLM 서빙 (전용 터미널, E4B+MTP, 토큰 불필요)
cd ~/gemma-rag-harness && source .venv/bin/activate
bash serving/vllm_launch.sh                    # 이 터미널 점유

# 3) 질의 (또 다른 터미널)
cd ~/gemma-rag-harness && source .venv/bin/activate
python -m scripts.ask "질문..."
```

---

## 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `pytest`가 모듈 못 찾음 | `pip install -e .` 안 함 / venv 미활성화 |
| 색인 시 connection refused | ES 안 뜸. `docker ps`, `curl localhost:9200` 확인 |
| 질의 시 connection refused | vLLM 안 뜸. `curl localhost:8000/v1/models` 확인 |
| `nvidia-smi` GPU 안 보임 | Windows NVIDIA 드라이버 업데이트 후 `wsl --shutdown` |
| vLLM OOM (메모리 부족) | `QUANT=fp8`, `MAX_LEN` 축소, `GPU_UTIL` 하향 |
| MTP method='draft_model'로 뜸 | vLLM이 0.24 미만. Gemma 4 MTP 지원 버전으로 업그레이드 |
| 모델명 불일치 | `.env` LLM_MODEL = vLLM `--served-model-name` |
| nori analyzer 에러 | 3단계 nori 이미지로 빌드했는지 확인 |
| ES 컨테이너 Exited | `docker logs es`. WSL 메모리 부족이면 .wslconfig 조정 |

---

## 최단 실행 순서 (전체, 처음 1회)

```bash
# 프로젝트
cd ~ && tar -xzf gemma-rag-harness.tar.gz && cd gemma-rag-harness
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                          # 배선 확인

# ES + nori
echo 'FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.0' > Dockerfile.es
echo 'RUN bin/elasticsearch-plugin install --batch analysis-nori' >> Dockerfile.es
docker build -t es-nori -f Dockerfile.es .
docker run -d --name es -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" es-nori

# vLLM (전용 터미널, E4B+MTP, 토큰 불필요)
pip install -e ".[serving]"
bash serving/vllm_launch.sh

# 색인 + 질의 (또 다른 터미널)
cp .env.example .env
python -m scripts.index_corpus
python -m scripts.ask "RRF의 rank_constant는?"
```
