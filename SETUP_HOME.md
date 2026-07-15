# SETUP_HOME.md — 집 노트북 세팅 가이드 (WSL2 + Anaconda)

회사(WSL2 + venv)와 동일한 개발·테스트 환경을 집 노트북에 만드는 절차.
가상환경만 **venv 대신 Anaconda(conda)** 를 쓴다. 나머지 구조는 SETUP.md와 같으며,
이 문서는 그동안 실제로 겪은 함정(버전 충돌, CUDA, VRAM)의 회피법을 포함한다.

> 상세한 배경 설명이 필요하면 각 단계에서 [SETUP.md](SETUP.md)의 해당 단계를 참고.

---

## 0단계: 먼저 GPU 유형 확인 — 경로가 갈린다

Windows 설정 → 시스템 → 정보(또는 `nvidia-smi`)에서 그래픽 카드를 확인한다.

| 내 GPU | 경로 | 서빙 방식 |
|---|---|---|
| **NVIDIA (VRAM 16GB+)** | **A** | vLLM + E4B + MTP (회사와 동일) — 아래 전체 단계 수행 |
| **NVIDIA (VRAM 12GB 이하)** | **A'** | vLLM + fp8 양자화 또는 E2B — A와 같되 10단계에서 조정 |
| **NVIDIA 없음 (Intel/AMD 내장)** | **B** | **Ollama(CPU) + 작은 Gemma** — 6단계(CUDA)와 vLLM 설치를 건너뛰고 10-B단계로 |

> 내장 그래픽의 "16GB"는 시스템 RAM 공유 메모리이지 CUDA용 VRAM이 아니다.
> NVIDIA가 없으면 vLLM·MTP는 불가하며, 경로 B로 간다.

**경로 B에서 되는 것 / 안 되는 것:**
- ✅ 하네스 코드 개발·디버깅, ES+nori 색인, 42개 유닛테스트, 프롬프트 튜닝, 문서 추출
- ✅ RAG 질의 동작 확인 (작은 Gemma, CPU라 답변당 수십 초 — 개발용으론 충분)
- ❌ MTP·E4B 속도/품질, 성능 계측 — 이건 워크스테이션에서만. **집에서 잰 골든셋
  점수를 워크스테이션 baseline과 비교하지 말 것** (모델이 달라 무의미)

| 공통 요구 | |
|---|---|
| Windows 10 21H2+ / 11, RAM 16GB+ (경로 B는 32GB 권장) | 디스크 여유 20GB+ |

---

## 1단계: WSL2 + Ubuntu 설치

PowerShell(관리자)에서:

```powershell
wsl --install -d Ubuntu-24.04
```

재부팅 후 Ubuntu 초기 사용자/비밀번호 설정. 이미 설치돼 있으면 건너뜀.

GPU가 WSL에서 보이는지 확인 (Ubuntu 터미널):
```bash
nvidia-smi   # GPU 목록이 나오면 정상 (Windows 드라이버를 그대로 사용)
```

---

## 2단계: Docker Desktop (Elasticsearch용)

1. Windows에 Docker Desktop 설치
2. Settings → Resources → **WSL Integration** → Ubuntu 켜기
3. Ubuntu 터미널에서 확인: `docker ps` (에러 없이 빈 목록이면 OK)

---

## 3단계: Anaconda 설치 (WSL 안에)

Ubuntu 터미널에서:

```bash
cd ~

# Miniconda(아나콘다 최소판) 설치 스크립트 다운로드 — 가볍고 동일하게 동작
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치 (라이선스 yes → 경로 기본값 → conda init yes)
bash Miniconda3-latest-Linux-x86_64.sh

# 셸 재시작 후 확인
source ~/.bashrc
conda --version
```

> 전체 Anaconda 배포판을 원하면 https://www.anaconda.com/download 에서
> Linux용 설치 스크립트 URL을 받아 같은 방식으로 설치하면 된다. conda 명령은 동일.

---

## 4단계: conda 가상환경 생성

```bash
# python 3.12 환경 생성 (회사 venv와 동일 버전)
conda create -n gemma-rag python=3.12 -y

# 활성화 — 앞으로 모든 작업은 이 환경에서
conda activate gemma-rag
```

프롬프트 앞에 `(gemma-rag)`가 붙으면 활성화된 것.
**회사의 `source .venv/bin/activate` 자리에 항상 `conda activate gemma-rag`를 쓴다.**
그 외 모든 명령은 SETUP.md와 동일하다.

---

## 5단계: 프로젝트 배치 + 의존성 설치

```bash
# 압축 해제 (다운로드 경로는 실제에 맞게)
cd ~
tar -xzf /mnt/c/Users/<사용자명>/Downloads/gemma-rag-harness.tar.gz
cd gemma-rag-harness

# 의존성 설치 — 반드시 이 순서로
pip install -e ".[dev]"        # ① 본체 + 개발도구 (sentence-transformers 3.x 핀)
pip install -e ".[serving]"    # ② vLLM (용량 큼, 수 분) — ⚠ 경로 B(NVIDIA 없음)는 건너뜀
```

**② 설치 끝에 빨간 경고가 뜨는 것이 정상이다:**
```
ERROR: pip's dependency resolver ... sentence-transformers 3.4.x requires
transformers<5.0.0, but you have transformers 5.x which is incompatible.
```
vLLM이 transformers 5.x를 요구하고 sentence-transformers는 4 미만을 요구해
pip이 충돌을 경고하지만, **이 조합(sentence-transformers 3.4.x + transformers 5.12.x)
으로 bge-m3 임베딩과 vLLM 서빙이 모두 정상 동작함을 실측으로 확인했다.**
경고를 무시하고 진행한다. (절대 transformers를 4.x로 내리지 말 것 — vLLM이 안 뜬다)

설치 검증:
```bash
python -m pytest tests/ -q     # 42 passed 나와야 함
python -m models.backends      # gemma-4-E4B-it @ http://localhost:8000/v1 나와야 함
```

---

## 6단계: CUDA Toolkit 13 설치 (경로 A 전용 — 경로 B는 건너뜀)

vLLM의 FlashInfer가 커널을 JIT 컴파일할 때 시스템 nvcc를 쓴다.
PyTorch(CUDA 13.0 빌드)와 메이저 버전이 같은 **CUDA Toolkit 13**을 설치한다.

**⚠ 함정 주의:**
- `sudo apt install nvidia-cuda-toolkit`(우분투 기본 저장소) **금지** — 구버전 12.0이
  깔려서 `no kernel image` 에러가 난다 (회사에서 실제로 겪음).
- `cuda` / `cuda-drivers` 메타패키지 **금지** — WSL에 Linux 드라이버를 깔려고
  시도해서 망가진다. WSL은 Windows 드라이버를 쓰므로 **Toolkit만** 설치한다.

```bash
# NVIDIA의 WSL 전용 저장소 등록
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Toolkit만 설치 (13-0이 없으면: apt-cache search cuda-toolkit-13 으로 확인)
sudo apt-get install -y cuda-toolkit-13-0

# PATH 등록
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version    # release 13.x 확인
```

---

## 7단계: Elasticsearch + nori (Docker)

```bash
cd ~/gemma-rag-harness

# nori 포함 이미지 빌드 (Dockerfile.es가 프로젝트에 포함돼 있음)
docker build -t es-nori -f Dockerfile.es .

# 컨테이너 기동
docker run -d --name es \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  es-nori

# 확인
curl http://localhost:9200          # cluster_name 나오면 OK
```

이후 재부팅 시엔 `docker start es`만 하면 된다.

---

## 8단계: 코퍼스 데이터 이전 (중요!)

**실제 ExampleCorp 문서(corpus/*.md, raw/)는 회사 PC에만 있다.** 압축 파일의 corpus/에는
샘플 1개만 들어 있으므로, ExampleCorp 질의를 하려면 회사에서 데이터를 가져와야 한다.

회사 WSL에서:
```bash
cd ~/gemma-rag-harness
tar -czf /mnt/c/Users/<사용자명>/Desktop/corpus_data.tar.gz corpus/ raw/
# → USB나 클라우드로 집에 전달
```

집 WSL에서:
```bash
cd ~/gemma-rag-harness
tar -xzf /mnt/c/Users/<사용자명>/Downloads/corpus_data.tar.gz
ls corpus/ | wc -l    # 33개 안팎 나와야 (ExampleCorp 문서들)
```

> raw/(원본 PDF·HTML)까지 가져오면 나중에 재추출도 가능하다.
> corpus/만 있어도 색인·질의는 된다.

---

## 9단계: .env 생성 + 색인

```bash
cd ~/gemma-rag-harness

# 로컬 설정 파일 생성 (.env는 압축에 안 들어있음 — 머신마다 만드는 것)
cp .env.example .env
cat .env    # LLM_MODEL=google/gemma-4-E4B-it, VLLM_URL=...:8000 확인

# 색인 — 첫 실행 시 bge-m3 임베딩 모델(~4.3GB) 자동 다운로드
python -m scripts.index_corpus
# "[index] 완료: N개 청크 색인됨" 나오면 성공
```

> `.env`는 점(.)으로 시작하는 숨김 파일이라 Windows 탐색기에서 안 보일 수 있다.
> `ls -la`로 확인하거나 터미널에서 `cat .env`로 내용을 본다.

---

## 10단계: 모델 서빙 — 경로 A (NVIDIA: E4B + MTP)

**전용 터미널**을 하나 열어서 (conda 활성화 후):

```bash
conda activate gemma-rag
cd ~/gemma-rag-harness

bash serving/vllm_launch.sh
```

- 첫 실행 시 E4B(~15GB) + MTP 초안 모델(~0.15GB)을 자동 다운로드한다.
  **Gemma 4는 Apache-2.0 공개 모델이라 HF 토큰이 필요 없다.**
- 로그에서 `SpeculativeConfig(method='mtp', ...)` → MTP 정상 활성화
- `Application startup complete` → 서빙 준비 완료
- 이 터미널은 계속 점유된다 (Ctrl+C 하면 서버가 꺼짐)

GPU VRAM이 16~24GB라면 GPU_UTIL을 올려서:
```bash
GPU_UTIL=0.85 bash serving/vllm_launch.sh
```

---

## 10-B단계: 모델 서빙 — 경로 B (NVIDIA 없음: Ollama CPU)

vLLM 대신 Ollama로 작은 Gemma를 CPU에서 돌린다. 하네스는 OpenAI 호환 URL만
바라보므로 **.env 설정 두 줄 외에 코드 변경이 전혀 없다.**

```bash
# Ollama 설치 (WSL 안)
curl -fsSL https://ollama.com/install.sh | sh

# 서버 기동 (전용 터미널 점유, 또는 백그라운드로 이미 떠 있으면 생략)
ollama serve
```

새 터미널에서 모델 받기 — 작은 Gemma를 쓴다:

```bash
# Gemma 4 소형이 Ollama 라이브러리에 있으면 그걸 우선 시도
ollama pull gemma4:e4b   # 없다는 에러가 나면 ↓
ollama pull gemma3:4b    # 확실히 존재하는 대안 (품질 준수, CPU에서 감당 가능)
# 더 빠른 응답이 필요하면: ollama pull gemma3:1b

# 동작 확인
ollama run gemma3:4b "한 문장으로 자기소개"
```

**.env를 Ollama로 변경** (이 두 줄이 경로 B의 전부):

```bash
cd ~/gemma-rag-harness && nano .env
```
```
LLM_MODEL=gemma3:4b                      # ollama pull 한 모델명과 정확히 일치
VLLM_URL=http://localhost:11434/v1       # Ollama의 OpenAI 호환 엔드포인트
```

확인:
```bash
python -m models.backends    # gemma3:4b @ http://localhost:11434/v1 나와야 함
```

> CPU 추론이라 GENERATE에 수십 초 걸릴 수 있다(정상). 개발·기능 확인용이며,
> 속도·품질 측정은 워크스테이션(경로 A)에서 한다.
> Intel Arc 내장 GPU 가속(IPEX-LLM 등)도 존재하지만 WSL에서 설정이 까다로워
> 권장하지 않는다. 필요해지면 그때 별도로 다룬다.

---

## 11단계: 질의 (동작 확인)

새 터미널에서:

```bash
conda activate gemma-rag
cd ~/gemma-rag-harness

python -m scripts.ask          # 대화형 모드 (권장 — 두 번째 질문부터 빠름)
# 질문> ExampleCorp은 어떤 회사인가요?
```

실행 경로(`ROUTER → RETRIEVE → GENERATE → VERIFY`)와 근거 기반 답변이 나오면 완성.

평가(골든셋 15문항 채점):
```bash
python -m eval.run_regression --goldenset eval/goldenset.jsonl
```

---

## 매번 켜는 순서 (요약)

**경로 A (NVIDIA):**
```bash
# 터미널 1 — 서빙
conda activate gemma-rag && cd ~/gemma-rag-harness
bash serving/vllm_launch.sh

# 터미널 2 — ES + 질의
docker start es
conda activate gemma-rag && cd ~/gemma-rag-harness
python -m scripts.ask
```

**경로 B (Ollama):**
```bash
# 터미널 1 — 서빙 (백그라운드로 이미 떠 있으면 생략)
ollama serve

# 터미널 2 — ES + 질의
docker start es
conda activate gemma-rag && cd ~/gemma-rag-harness
python -m scripts.ask
```

---

## 자주 겪는 문제 (이 프로젝트에서 실제 겪은 것들)

| 증상 | 원인 · 해결 |
|---|---|
| `Pooling.__init__() missing 'embedding_dimension'` | sentence-transformers 5.x가 깔림. `pip install "sentence-transformers>=3.0,<4.0"` 로 내림 |
| `ImportError: Transformers v4 ... removed in vLLM` | transformers가 4.x로 내려감. `pip install --upgrade "transformers>=5.5.3"` |
| `no kernel image is available` | 시스템 nvcc가 구버전(12.0). 6단계대로 CUDA Toolkit 13 설치 |
| `Available KV cache memory: -X GiB` | VRAM 부족. `GPU_UTIL` 하향, `MAX_LEN=2048` 축소, 또는 `--quantization fp8` |
| 질의가 `Connection refused` | vLLM(8000)이 안 떠 있거나, 터미널에 `HEAVY_URL` 등 옛 환경변수 잔재. `unset HEAVY_URL HEAVY_MODEL LIGHT_URL LIGHT_MODEL` 후 `python -m models.backends`로 8000 확인 |
| `python: command not found` | conda 환경 비활성. `conda activate gemma-rag` |
| 색인이 tokenizer.json 등에서 멈춤 | HF 익명 rate limit. 잠시 후 재시도하거나 `export HF_TOKEN=...` |
| 색인 프로세스가 안 끝남/충돌 | 중복 실행. `pkill -9 -f index_corpus` 후 하나만 재실행 |
