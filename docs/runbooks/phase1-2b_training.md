# Runbook — Phase 1-2b 2차 게이트 (GPU 서버)

1차 승인된 Phase 1-2b 코드를 GPU 서버에서 실행해 **CODa training split 베이스라인 mAP** 를 확정하기 위한 핸드오프 문서. 이 문서의 모든 단계는 GPU 서버에서 진행하며, 결과는 `docs/progress/phase1-2b_detection.md` 의 "검증 로그" · "사용자 검토 결과 (2차)" 섹션에 append 한 뒤 단일 PR 로 묶어 제출한다.

**Origin**: 2026-04-24 1차 리뷰 승인 세션 (`docs/progress/phase1-2b_detection.md`). 관련 ADR: `docs/decisions/20260424_detr-head-library.md`.

**작성 원칙**: 각 단계는 **실행 커맨드** · **성공 기준** · **실패 시 대응** 3개 축으로 기술. 새 Claude Code 세션이 이 문서만 보고 자율 실행할 수 있도록 쓴다.

---

## 0. 사전 체크 (1회)

- [ ] HuggingFace 계정 로그인 + [DINOv3 모델 페이지](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) 에서 license accept 완료
- [ ] HF access token 발급 — `Read` 권한이면 충분. 토큰을 `HF_TOKEN` 환경변수로 export
- [ ] CODa 데이터 접근 확보 — [UT AMRL CODa](https://amrl.cs.utexas.edu/coda/) 에서 최소 `2d_rect/cam0`, `3d_bbox/os1`, `calibrations`, `metadata` 트리. 전체 training split 은 수백 GB 이므로 디스크 여유 확인
- [ ] GPU — 최소 24GB VRAM 권장 (batch_size=2 @ 1024² 기준). 12-16GB 라면 `detection.training.batch_size=1` + `detection.image_size=768` 로 내려서 실행

**한 개라도 미충족이면 이 문서 실행 불가.** 사전 해소 후 1. 단계로.

---

## 1. 환경 구성

```bash
# 저장소 동기화
git clone https://github.com/marsberry/SCVTerraVision.git
cd SCVTerraVision
git checkout objectdetection   # 1차 승인 세션에서 작업한 브랜치

# Python 3.10+ venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch — CUDA 버전에 맞는 휠로 별도 설치 (pyproject 는 torch 를 pin 하지 않음)
# CUDA 12.1 예시:
pip install torch==2.3.* torchvision==0.18.* --index-url https://download.pytorch.org/whl/cu121

# 나머지 프로젝트 의존성
pip install -e .[dev]

# HF 토큰 환경 변수 (세션 지속시간만 유효)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# (선택) W&B 로그인
export WANDB_API_KEY=xxx
```

**성공 기준**:
```bash
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
# → cuda: True  devices: ≥1

python3 -m pytest -q
# → 52 passed, 8 skipped (1차 세션과 동일해야 함; 이미 설치된 pycocotools 때문에
#    test_det_metrics 는 3개가 SKIPPED→PASSED 로 바뀔 수 있음)
```

**실패 시**: torch CUDA 미감지 → 드라이버 · CUDA 버전 · 설치한 휠 버전 확인. transformers ImportError → `pip install transformers>=4.56` 재확인.

---

## 2. Pre-flight — DINOv3 + DETR smoke

실데이터 없이 **모델 빌드 · 1 forward** 만 먼저 확인. 이 단계에서 터지는 에러는 보통 HF 버전 · 토큰 · CUDA 조합 이슈이며, 전체 학습을 돌리기 전에 잡아야 시간이 안 든다.

```bash
# 백본 smoke (이미 1차에서 스킵된 3개 케이스)
RUN_DINO_SMOKE=1 python3 -m pytest tests/test_backbone.py -v -k forward_shape

# DETR 어댑터 smoke (HF DeformableDetr 인스턴스화 + forward)
RUN_DINO_SMOKE=1 python3 -m pytest tests/test_detr_head.py -v -k "load_builds or forward_returns"
```

**성공 기준**: `test_forward_shape_224_square`, `test_forward_shape_1024_square`, `test_forward_rejects_non_multiple_input`, `test_load_builds_hf_deformabledetr`, `test_forward_returns_detr_output_with_loss` 전부 PASSED.

**실패 시**:
- `401 Unauthorized` on DINOv3 fetch → license accept 재확인, `HF_TOKEN` 재발급
- HF internal path mismatch (`model.model.backbone.conv_encoder` 등) → transformers 버전이 4.56 대와 멀어졌을 가능성. `pip install 'transformers==4.56.*'` 로 pin
- OOM on 1024² forward → `detection.image_size=512` 로 낮춰 재시도 (smoke 한정)

---

## 3. CODa → COCO 변환 (subset 먼저)

Full training split 변환 · 학습을 바로 돌리지 말고, **3-5개 시퀀스로 loop sanity 먼저 검증**. 이 단계는 주로 data path · taxonomy · 투영 수치에 문제 없는지 확인하는 용도.

```bash
# 예: metadata/ 에 있는 시퀀스 이름 중 3개 선택 (CODa 실제 시퀀스 이름은 0, 1, 2 ...)
python3 -m training.datasets.coda_to_coco \
    --coda-root /mnt/data/coda \
    --split training \
    --output data/annotations/coda_training_subset.json \
    --sequences 0 1 2
```

**성공 기준**: 표준출력에 `wrote ... | images=N annotations=M dropped(taxonomy=..., occlusion=..., projection=...)` 출력. N > 0, M > 0. `dropped_by_projection` 이 annotations 의 50% 를 넘으면 calibration · 좌표변환 의심 (이슈 보고 후 stop).

**실패 시**:
- `ValueError: unknown CODa class ...` → CODa 가 새 클래스 추가. `configs/dataset/coda_taxonomy.yaml` 에 `coda_to_operational` 또는 `coda_dropped` 로 명시 추가
- 시퀀스 이름이 `0 1 2` 로 안 맞으면 `ls /mnt/data/coda/metadata/` 로 확인 후 교체

---

## 4. Dry-run 학습 (subset, 3 epoch)

```bash
# subset 으로 annotations 만 가리키고, 나머지 config 은 기본값 + epochs 만 3 으로
python3 -m training.train_detection \
    detection=dinov3_detr_base \
    dataset.annotations_path=data/annotations/coda_training_subset.json \
    detection.training.epochs=3 \
    detection.training.save_every_n_epochs=1 \
    logging.wandb.enabled=false
```

**성공 기준**:
- 로그에 `epoch 0 step 0 loss ...` 가 반드시 찍힘 (첫 forward+backward 통과)
- epoch 0 → 2 동안 loss 가 **감소 추세** (단조 감소일 필요는 없음, 평균이 내려가면 OK)
- `freeze_backbone_epochs=2` 기본값 때문에 epoch 2 진입 시 로그에 `epoch 2: unfreezing backbone` 가 찍힘
- 종료 시 `outputs/<date>/<time>/checkpoints/dinov3_detr_base/epoch_003.pt` 가 생성됨

**실패 시**:
- 첫 step 에서 OOM → batch_size 또는 image_size 축소 (위 0 단계 참조)
- loss NaN → `detection.training.lr=1e-4` 로 낮추고 재시도. 그래도 터지면 이슈 보고
- 로그에 unfreeze 메시지가 안 뜨면 `set_backbone_frozen` 연결 버그 가능성 — 이슈 보고

---

## 5. Full 학습 (CODa training split 전체, 50 epoch)

subset dry-run 녹색이면 진행. 소요 시간은 GPU 에 따라 12-48h.

```bash
# 전체 training split 변환 (이미 subset 으로 만들었다면 덮어쓰기)
python3 -m training.datasets.coda_to_coco \
    --coda-root /mnt/data/coda \
    --split training \
    --output data/annotations/coda_training_coco.json

# validation split 도 함께 (평가용)
python3 -m training.datasets.coda_to_coco \
    --coda-root /mnt/data/coda \
    --split validation \
    --output data/annotations/coda_validation_coco.json

# 본 학습
python3 -m training.train_detection \
    detection=dinov3_detr_base \
    dataset.annotations_path=data/annotations/coda_training_coco.json \
    logging.wandb.enabled=true \
    logging.wandb.project=terravision
```

**성공 기준**:
- 학습이 50 epoch 완주 (중간 checkpoint 는 `save_every_n_epochs=5`)
- loss 가 10 epoch 이내에 유의미하게 내려감 (초기 대비 30% 이상 감소를 거친 기준으로 봄)
- VRAM peak 사용량 · epoch 당 시간을 기록해둘 것 (나중에 Phase 1-5 최적화 기준선)

**실패 시**:
- disk full → subset 으로 재시도, 또는 `save_every_n_epochs` 를 10 으로 키움
- 40 epoch 이상 돌리는데 loss 정체 → LR 스케줄 의심. 이 경우는 2차 리뷰 대상이니 중단 후 보고
- W&B 로그인 실패 → `logging.wandb.enabled=false` 로 계속 (W&B 없이도 학습은 정상)

---

## 6. 평가 — CODa validation mAP

학습 루프 자체는 평가를 수행하지 않음 (Phase 1-2b scope 에서 제외). 학습 후 별도 스크립트로 돌림.

평가 스크립트는 아직 미작성 → 이 세션에서 짧은 one-off `scripts/eval_detection.py` 를 만들어도 되고, 아래 snippet 을 노트북에서 실행해도 됨:

```python
# 대략적 템플릿. 본 runbook 을 따라오는 세션이 실제 코드로 작성.
from pathlib import Path
import torch, json
from models.detection import DetrHeadConfig, DinoV3DeformableDetr
from models.backbone.dinov3_backbone import DinoV3BackboneConfig
from training.datasets.coco_loader import CocoDetectionDataset
from training.train_detection import make_collate_fn
from evaluation import compute_coco_map

CKPT = "outputs/.../checkpoints/dinov3_detr_base/epoch_050.pt"
VAL_GT = "data/annotations/coda_validation_coco.json"
IMG_ROOT = "/mnt/data/coda/2d_rect/cam0"

# 1) 모델 로드 + state_dict
wrapper = DinoV3DeformableDetr()
model = wrapper.load()
state = torch.load(CKPT, map_location="cuda")
model.load_state_dict(state["model_state"])
model.eval().cuda()

# 2) val DataLoader
dataset = CocoDetectionDataset(VAL_GT, IMG_ROOT)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, num_workers=4,
    collate_fn=make_collate_fn(1024),
)

# 3) inference → COCO predictions list
predictions = []
with torch.inference_mode():
    for batch in loader:
        out = model(
            pixel_values=batch["pixel_values"].cuda(),
            pixel_mask=batch["pixel_mask"].cuda(),
        )
        # logits: (B, Q, num_labels+1) pred_boxes: (B, Q, 4) cxcywh [0,1]
        # 변환: top-K boxes per image → COCO xywh pixel space
        # (letterbox 역변환 포함 — train_detection.preprocess_sample 의 역연산)
        ...  # 구현 필요

# 4) mAP
result = compute_coco_map(predictions, VAL_GT)
Path("outputs/eval_phase1-2b.json").write_text(json.dumps(result, indent=2))
print(result["mAP"], result["AP50"])
```

**성공 기준**: `outputs/eval_phase1-2b.json` 에 `{mAP, AP50, AP75, per_class}` dict 기록. mAP 숫자는 본 runbook 의 합격선을 규정하지 않음 — 베이스라인이므로 "수치 확정 = 성공".

**작성 시 주의**: inference post-processing (letterbox 역변환, top-K 선택, score threshold) 은 HF `DeformableDetrImageProcessor.post_process_object_detection` 가 제공. 가급적 그걸 사용하고, CocoDetectionDataset item 의 `orig_size` 와 letterbox scale 조합으로 원본 이미지 좌표로 되돌릴 것.

---

## 7. 검증 로그 append → `phase1-2b_detection.md`

위 단계의 산출물을 모아 `docs/progress/phase1-2b_detection.md` 의 **검증 로그** 섹션 끝에 append:

```markdown
### 2차 게이트 실행 (2026-XX-XX, GPU 서버: <hostname/spec>)

- 환경: Python 3.10 · torch 2.3.x CUDA 12.1 · transformers 4.56.x · GPU <model, VRAM>
- Pre-flight smoke: `RUN_DINO_SMOKE=1 pytest tests/test_backbone.py tests/test_detr_head.py` → N passed (기존 52 + gated 5 = 57)
- Subset dry-run: 3 sequences · 3 epoch · 초기 loss <X.XX> → epoch 2 loss <Y.YY>. unfreeze 정상.
- Full training: CODa training split 전체 (images=<N>, annotations=<M>) · 50 epoch · <HH:MM:SS> · peak VRAM <N>GB
- W&B run: <URL or "offline">
- Validation mAP (CODa validation split):
  - mAP@[0.5:0.95] = <X.XX>
  - AP50 = <X.XX>
  - AP75 = <X.XX>
  - per-class: pedestrian=<X>, bicycle=<X>, ..., fire_hydrant=<X>
- Checkpoint: `outputs/<path>/epoch_050.pt` (sha256: <hash>)
```

그리고 **사용자 검토 결과 (2차)** 섹션에 날짜 + 간단한 소회 (이슈 · 다음 단계 제안) 를 남긴다.

---

## 8. PR 묶기

- 1차 세션에서 작업한 Phase 1-2b 코드 + 본 runbook + 2차 결과로 채운 `phase1-2b_detection.md` 까지 **하나의 PR** 로 제출 (사용자 방침, 2026-04-24 결정)
- 커밋 순서: (a) 1차 코드 (models/detection, configs/detection, training, evaluation, tests), (b) runbook, (c) 2차 수치로 업데이트된 phase1-2b + progress/README + PLAN
- PR 제목 예: `Phase 1-2b — DINOv3 + HF Deformable DETR 헤드 베이스라인`
- PR 본문: 1차 아키텍처 요약 + 2차 수치 · 트레이드오프 · 다음 단계 (Phase 1-2c 파인튠 계획)

---

## 9. 세션 핸드오프 메모

다음 GPU 서버 Claude Code 세션이 이 문서를 열었을 때 바로 시작할 수 있도록:

1. 이 runbook 의 `0.` 섹션부터 순서대로 **체크박스 단위로** 진행할 것
2. 각 단계의 **성공 기준** 을 반드시 확인한 뒤 다음 단계로 이동 (건너뛰지 말 것)
3. 실패 시 스스로 고치려 하기 전에 원인을 `phase1-2b_detection.md` 에 단락으로 기록 (이후 세션의 디버깅 근거)
4. 본 원래 세션 (2026-04-24 1차 승인) 의 전제 조건을 바꾸는 변경 (예: transformers 버전 pin, backbone 교체, 헤드 라이브러리 교체) 은 **새 ADR 작성 후** 진행
5. 2차 수치까지 채운 뒤 단일 PR 로 묶고, 머지는 2차 리뷰 승인 이후에만
