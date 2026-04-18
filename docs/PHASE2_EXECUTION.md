# Phase 2 실행 플레이북

> 워크스테이션에서 Phase 2 (DINOv2 baseline 측정 + supervised 학습)을 처음 돌리는
> 사용자가 그대로 따라가도록 작성된 명령어 교본.
> 측정 프로토콜의 근거는 `docs/ROADMAP.md` §2.6 참조.

---

## 0. 환경 가정

| 항목 | 가정 |
|------|------|
| 워크스테이션 OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3090 / 4090 (VRAM 24GB) |
| CUDA | 12.x |
| Python | 3.10+ (3.11 권장) |
| 디스크 여유 | 데이터셋 + 체크포인트용 100GB+ |
| 인터넷 | 첫 실행 시 DINOv2 가중치 다운로드 (~85MB) |

위와 다르면 6장 트러블슈팅 참조.

---

## 1. 사전 준비

### 1.1 저장소 동기화

로컬(Mac) → 워크스테이션:
```bash
# 워크스테이션에서
git clone <원격> ~/camera_project   # 원격 저장소가 있는 경우
# 또는 로컬에서 rsync (git 저장소 미사용 시)
rsync -av --exclude='.venv' --exclude='outputs' --exclude='data' \
  ~/development/camera_project/ user@workstation:~/camera_project/
```

### 1.2 가상환경 + 의존성

```bash
cd ~/camera_project
uv venv --python 3.11 .venv         # uv 미설치 시 `python3.11 -m venv .venv`
source .venv/bin/activate
uv pip install -e ".[dev,logging]"  # logging extra → tensorboard 포함
```

### 1.3 데이터 위치

```bash
mkdir -p data
ln -s /path/to/rugd       data/rugd
ln -s /path/to/rellis3d   data/rellis3d
```

데이터셋 다운로드 안내가 필요하면:
```bash
python scripts/download_datasets.py --dataset rugd
python scripts/download_datasets.py --dataset rellis3d
```

---

## 2. Pre-flight 체크

순서대로 실행. 하나라도 실패하면 다음 단계로 진행 금지.

### 2.1 단위 테스트

```bash
pytest tests/
```
**기대**: `32 passed, 1 skipped` (DINOv2 통합 테스트는 `RUN_SLOW=1`로 활성화 가능)

### 2.2 GPU smoke test

DINOv2 가중치 다운로드 + 1 epoch 합성 데이터 학습:
```bash
python scripts/smoke_test.py --accelerator gpu --epochs 1
```
**기대**: `[smoke] OK — training pipeline runs end-to-end`

### 2.3 데이터셋 무결성

```bash
python scripts/verify_dataset.py --dataset rugd --split train
python scripts/verify_dataset.py --dataset rellis3d --split train
```
**기대**: 샘플 수 출력 + 통합 클래스 분포 출력 + 에러 없음

### 2.4 데이터셋 시각화

```bash
python scripts/visualize_samples.py --dataset rugd --n 8 --out outputs/sanity/rugd
python scripts/visualize_samples.py --dataset rellis3d --n 8 --out outputs/sanity/rellis3d
```
**기대**: 각 폴더에 8장 triptych PNG. 라벨 색상이 의도대로(예: 하늘=하늘색,
의자/사람=노랑) 보여야 함. 어긋나면 `configs/datasets/<name>.yaml`의 `to_unified`
매핑 점검.

---

## 3. TensorBoard 띄우기 (실시간 모니터링)

### 3.1 워크스테이션에서

```bash
# 별도 터미널에서 (학습 시작 전 미리)
tensorboard --logdir=outputs/runs --port=6006 --bind_all
```

### 3.2 로컬(Mac)에서 SSH 포트 포워딩

```bash
ssh -L 6006:localhost:6006 user@workstation
```
포워딩 활성화 후 로컬 브라우저에서 → http://localhost:6006

### 3.3 어떤 panel이 보여야 하는가

학습이 시작되면 TensorBoard에 다음이 자동 생성:

**SCALARS 탭**
- `train/loss_step`, `train/loss_epoch`
- `val/loss`
- `val/mIoU`, `val/trav_IoU` (binary 주행가능성)
- `val/iou/<class>` — 6개 클래스 각각의 IoU
- `lr-AdamW`

**IMAGES 탭** (PredictionVizCallback이 켜져 있을 때, 기본 ON)
- `predictions/sample_0` ~ `predictions/sample_3`
- 각 타일은 가로로 `[입력 이미지 | GT 마스크 | 예측 마스크]` 결합
- 매 epoch마다 갱신 → 슬라이더로 epoch 이동하며 학습 진행 관찰

> **시각화 콜백 설정**: `configs/train/default.yaml`의 `viz` 블록
> `viz.enabled=false`로 끄거나, `n_samples`/`every_n_epochs` 조정 가능.

---

## 4. Baseline 실행 (ROADMAP §2.6 매핑)

### 4.1 B0 — 무학습 sanity check

학습 없이 random-init linear head로 evaluation. 파이프라인 동작 확인.
```bash
python scripts/eval.py --baseline_b0 --model dinov2_linear \
  --dataset rugd --split val --out outputs/eval/b0_rugd_val
```
**기대**: `mIoU ≈ 1/K` (랜덤 추측). 0이면 데이터/타xonomy/메트릭 어디서 끊겼는지 점검.
**산출물**: `outputs/eval/b0_rugd_val/metrics.json` + 예측/GT triptych PNG

RELLIS-3D에도 동일하게:
```bash
python scripts/eval.py --baseline_b0 --model dinov2_linear \
  --dataset rellis3d --split val --out outputs/eval/b0_rellis_val
```

### 4.2 B1 — Linear probe (RUGD only)

```bash
python scripts/train.py --model dinov2_linear \
  data.datasets=[rugd] \
  trainer.max_epochs=30 \
  data.batch_size=16
```
**예상 시간**: RTX 4090 기준 약 1~2시간 (RUGD train ~7000장, ViT-S/14 frozen)
**모니터링 포인트**:
- `val/mIoU`가 epoch별 단조 상승하는지 (TB SCALARS)
- `predictions/sample_*`이 점점 GT에 가까워지는지 (TB IMAGES)
- `train/loss`가 감소하는지

**산출물**:
- `outputs/runs/dinov2_linear_<timestamp>/checkpoints/<best>.ckpt`
- `outputs/runs/dinov2_linear_<timestamp>/resolved_config.yaml` (재현용)

### 4.3 B1' — Linear probe (RUGD + RELLIS 합본)

도메인 다양성이 mIoU에 어떻게 영향 주는지 확인:
```bash
python scripts/train.py --model dinov2_linear \
  data.datasets=[rugd,rellis3d] \
  trainer.max_epochs=30 \
  data.batch_size=16
```

### 4.4 B2 — DPT head

DPT decoder는 메모리 더 사용 → batch 절반:
```bash
python scripts/train.py --model dinov2_dpt \
  data.datasets=[rugd] \
  trainer.max_epochs=30 \
  data.batch_size=8
```
**비교 포인트**: B1 대비 mIoU 향상이 head 복잡도 증가의 가치를 정당화하는지

---

## 5. Best checkpoint 평가

각 학습이 끝나면 best ckpt로 정량 평가:

```bash
RUN=outputs/runs/dinov2_linear_<timestamp>
BEST=$(ls $RUN/checkpoints/*.ckpt | grep -v last | sort | tail -1)

# 학습에 사용한 데이터셋과 사용하지 않은 데이터셋 모두 평가
python scripts/eval.py --ckpt $BEST --dataset rugd --split val \
  --out outputs/eval/$(basename $RUN)_rugd_val
python scripts/eval.py --ckpt $BEST --dataset rellis3d --split val \
  --out outputs/eval/$(basename $RUN)_rellis_val
```

각 폴더의 `metrics.json` 안에 mIoU, per-class IoU, traversability binary IoU 모두 기록됨.

---

## 6. 결과 정리 — `reports/phase2_baseline.md`

다음 표 형태로 정리:

```
| Baseline | RUGD val mIoU | RELLIS val mIoU | RUGD trav_IoU | RELLIS trav_IoU | latency (ms/img) |
|----------|---------------|------------------|----------------|------------------|-------------------|
| B0 random| ...           | ...              | ...            | ...              | ...               |
| B1 RUGD  | ...           | ...              | ...            | ...              | ...               |
| B1' both | ...           | ...              | ...            | ...              | ...               |
| B2 DPT   | ...           | ...              | ...            | ...              | ...               |
```

추가:
- per-class IoU 표 (특히 grass↔non_traversable, smooth↔non_traversable 혼동)
- 정성 실패 케이스 50장 발췌 (그림자, 물, 키 큰 풀, 야간) — `outputs/eval/*/pred_*.png`에서 선별

이 보고서가 작성되면 ROADMAP §2.7의 baseline 채택 여부 결정 가능.

---

## 7. 트러블슈팅

| 증상 | 원인 후보 | 대응 |
|------|-----------|------|
| `CUDA out of memory` | batch 너무 큼 | `data.batch_size=8` (절반), 안 되면 `data.crop_h=420 data.crop_w=420` (14의 배수) |
| `val/mIoU=NaN` | 한 클래스가 val에 한 픽셀도 없음 | 데이터 split 점검, NaN 평균은 정상이지만 학습 모니터링은 nanmean 사용 중이므로 무시 가능 |
| DataLoader 느림 (GPU 사용률 낮음) | `num_workers` 부족 | `data.num_workers=8` 또는 `16` |
| `train/loss` 감소 안 함 | lr 너무 큼 / BN 비호환 | `optim.lr=1e-4`, 또는 `model.head_kwargs.use_bn=false` |
| HF 다운로드 실패 (오프라인) | 네트워크 차단 | `HF_HUB_OFFLINE=1` + 미리 캐시한 `~/.cache/huggingface/` 복사 |
| TensorBoard에 IMAGES 탭 안 보임 | viz 콜백 비활성 | `configs/train/default.yaml`의 `viz.enabled: true` 확인. CSV/wandb 로거에서는 자동 무시 |
| `lightning` 임포트 실패 | venv 활성화 안 됨 | `source .venv/bin/activate` 후 `which python` 확인 |
| `RUGDDataset: No samples found` | 경로 오타 / split 파일 없음 | `ls data/rugd/images/`, `ls data/rugd/splits/train.txt` |

---

## 8. Phase 2 종료 체크리스트

- [ ] `outputs/eval/b0_*/metrics.json` 존재 (B0)
- [ ] `outputs/eval/<run>_rugd_val/metrics.json` 존재 (B1)
- [ ] `outputs/eval/<run>_rellis_val/metrics.json` 존재 (B1, cross-domain)
- [ ] `outputs/eval/<run>_dpt_*/metrics.json` 존재 (B2)
- [ ] `reports/phase2_baseline.md` 작성 (위 4종 결과 표 + 정성 분석)
- [ ] 추론 latency 측정 (워크스테이션, 518×518, batch=1)
- [ ] 실패 케이스 50장 추출

전부 ✅이면 Phase 3 진입 가능.

---

## 부록 A. 자주 쓰는 OmegaConf override 예시

```bash
# epoch 줄여서 빠른 검증
python scripts/train.py --model dinov2_linear trainer.max_epochs=3

# 큰 모델로 전환
python scripts/train.py --model dinov2_linear model.backbone_variant=base

# 학습률 변경
python scripts/train.py --model dinov2_linear optim.lr=5e-4

# Dice loss 추가
python scripts/train.py --model dinov2_linear model.dice_weight=0.5

# viz 끄기 (속도 ↑)
python scripts/train.py --model dinov2_linear viz.enabled=false

# 다른 logger
python scripts/train.py --model dinov2_linear logger.type=csv
```

---

## 부록 B. 워크스테이션 vs 로컬 차이

| 작업 | 워크스테이션 | 로컬 (Mac) |
|------|--------------|------------|
| 코드 작성/수정 | ❌ (sync 받기만) | ✅ |
| 단위 테스트 | ✅ | ✅ (CPU) |
| smoke test | ✅ (gpu) | ✅ (`--accelerator cpu`) |
| 본격 학습 | ✅ | ❌ |
| TensorBoard 호스팅 | ✅ | — (포워딩으로 보기) |
| 평가 (eval.py) | ✅ | ✅ (느림) |

