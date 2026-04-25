# Phase 1-2b — DETR 헤드 학습

- **상태**: ✅ 1차 승인 완료 (2026-04-24) · ⏳ 2차 게이트 대기 (GPU 환경 실학습 mAP)
- **시작일**: 2026-04-24
- **완료일**: —
- **담당 PR**: (작성 후 링크)
- **관련 ADR**: `docs/decisions/20260424_detr-head-library.md` (**확정 2026-04-24**)

## 목표

DINOv3 백본(Phase 1-2a) 위에 DETR-style detection 헤드를 올려 공개 데이터 베이스라인 mAP 를 뽑는다. 데이터는 **CODa training split** 단독 (primary 데이터셋 원칙, ADR 20260422). 산출 지표는 COCO mAP@50:95 · mAP@50 · 클래스별 AP.

본 단계가 "환경 설정이 선행되어야 의미 있는" 단계이므로 진행을 두 축으로 분리:

- **코드·설정 (이 단계 scope)**: wrapper · 학습 루프 · Hydra config · 평가 코드 · 스모크 테스트.
- **실학습 실행·수치 확정 (후속 세션)**: GPU 환경 + HF_TOKEN + CODa 실데이터 갖춰진 사용자 장비에서 1회 실행, 결과를 본 문서의 "검증 로그" 에 append.

## 확정된 결정 (2026-04-24 승인)

1. **DETR 헤드 라이브러리**: **HF Transformers `DeformableDetrForObjectDetection`** (플랜의 mmdet 대신). ADR `20260424_detr-head-library.md` 확정.
2. **학습 해상도**: **1024×1024** square crop 으로 시작. rectangular (1024×1216) 실험은 후속.
3. **Multi-scale feature**: **single-scale 로 시작** (DINOv3 ViT-B/16 stride-16). 수치가 나쁘면 1-2c 에서 간단 pyramid 추가.
4. **베이스라인 데이터 규모**: **CODa training split 전체**.
5. **승인 게이트**: 2단계 — (a) 코드 · 설정 · 스모크 테스트 green → 1차 리뷰, (b) 실학습 1회 mAP 수치 append → 2차 리뷰.

## 체크리스트

- [x] `models/detection/__init__.py`
- [x] `models/detection/detr_head.py`
  - [x] `DinoV3DeformableDetr` — `DinoV3Backbone` feature → HF DeformableDetr decoder 어댑터
  - [x] `DetrHeadConfig` dataclass (num_labels, num_queries, d_model, num_encoder/decoder_layers, aux_loss)
  - [x] forward → `DeformableDetrObjectDetectionOutput` (loss, loss_dict, logits, pred_boxes)
  - [x] lazy transformers import (Phase 1-2a 패턴 재사용)
- [x] `configs/detection/dinov3_detr_base.yaml` — Hydra 서브그룹
  - [x] defaults: `[/backbone: dinov3_vitb16, /dataset: coda, _self_]`
  - [x] num_labels=16, num_queries=300, d_model=256, encoder_layers=6, decoder_layers=6
- [x] `training/train_detection.py` — Hydra entrypoint (순수 PyTorch 루프, HF Trainer 미사용)
  - [x] optimizer + LR scheduler (AdamW head / AdamW low-LR backbone, cosine + warmup)
  - [x] backbone freeze → epoch N 부터 unfreeze (`freeze_backbone_epochs`)
  - [x] W&B 연결 (base.yaml 의 `logging.wandb` 플래그 기준)
  - [x] checkpoint 저장 `paths.output_root/checkpoints/dinov3_detr_base/`
- [x] `evaluation/__init__.py`, `evaluation/det_metrics.py` — pycocotools 기반 mAP
  - [x] `compute_coco_map(predictions, coco_gt_path)` → dict(mAP, AP50, AP75, AR_*, per_class)
- [x] `tests/test_detr_head.py` — lazy-import 스모크
  - [x] config 기본값 · 검증 규칙 (num_labels / d_model / heads 나눗셈 / multi-scale reject)
  - [x] transformers 미설치 환경에서 ImportError guidance
  - [x] Hydra YAML → DetrHeadConfig 왕복 + defaults 그룹 존재 확인
  - [x] (gated) `RUN_DINO_SMOKE=1` 에서 HF DeformableDetr 빌드 + forward 1회 (loss_dict key, logits/pred_boxes shape)
- [x] `tests/test_det_metrics.py` — 합성 pred / gt → 예상 mAP
  - [x] 입력 스키마 검증 + pycocotools 미설치 ImportError + GT 파일 부재 FileNotFoundError
  - [x] (auto-gated) 완벽 예측 → mAP=1.0, 빈 예측 → mAP=0.0, 박스 shift → AP50 유지 · AP75 하락
- [x] 전체 pytest green — `52 passed, 8 skipped`
- [ ] **(후속)** 실학습 1회 실행 + mAP 수치 기록 (GPU 환경에서 본 문서 검증 로그에 append)

## 산출물

- `models/detection/{__init__.py, detr_head.py}`
- `configs/detection/dinov3_detr_base.yaml`
- `training/__init__.py`, `training/train_detection.py`
- `evaluation/{__init__.py, det_metrics.py}`
- `tests/{test_detr_head.py, test_det_metrics.py}`

## 검증 로그

- `python3 -m pytest -q` → `52 passed, 8 skipped in 0.22s` (시스템 pytest 6.2.5, Python 3.10.12). ✅
  - Phase 1-1: 7 · Phase 1-1b: 11 · Phase 1-2a: 18 (3 gated) · Phase 1-2b detection: 16 (2 gated) · Phase 1-2b metrics: 8 (3 auto-gated on pycocotools 미설치) = 60 cases, non-gated 전부 녹색.
- `python3 -c "from models.detection import DetrHeadConfig, DinoV3DeformableDetr"` — torch / transformers 없이 import 성공. ✅
- `python3 -c "import training.train_detection"` — Hydra 데코레이터를 `_cli()` 내부에서만 생성하므로 torch / hydra 없이 모듈 collect 성공. ✅
- `python3 -c "from evaluation import compute_coco_map"` — pycocotools 없이 import 성공. ✅
- 실모델 · 실데이터 smoke 는 후속: (a) `RUN_DINO_SMOKE=1 HF_TOKEN=... pytest tests/test_detr_head.py -k "load_builds or forward_returns"`, (b) `python -m training.train_detection detection=dinov3_detr_base` — 두 경로 모두 GPU 환경에서 실행.

### 2차 게이트 실행 — 중간 진행 (2026-04-25, GPU 서버: RTX 4090 × GPU3)

1차 승인 후 다른 PC 에서 본 GPU 서버로 핸드오프. runbook §1 ~ §4 + eval 스크립트 작성까지만 수행. Full 50-epoch 학습은 다음 세션.

**환경**:
- Python 3.10.20 · torch 2.3.1+cu121 · transformers 4.56.2 · pycocotools 2.0.11 · hydra 1.3.2 · wandb 0.26.1
- GPU: RTX 4090 24GB (4장 중 GPU3 단일 사용, `CUDA_VISIBLE_DEVICES=3`)
- CODa 경로: `/home/marsberry/dataset/coda-devkit/data/CODa_full` (22 시퀀스, 174GB)
- venv: 기존 .venv 는 pip 부재로 재생성 불가 → miniconda3 `terravision` 환경 (python 3.10) 로 교체

**runbook §1 `pytest -q`**: 51 passed, **1 failed** (latent 버그 노출), 8 skipped.
- 실패: `test_det_metrics::test_offset_prediction_drops_ap_at_high_iou` — IoU 계산이 주석과 불일치 (실제 0.324, 주석은 "≈0.52"). 1차 세션은 pycocotools 미설치로 auto-skip 되어 노출 안 됐음. Phase 1-2b scope 에 영향 없음. 수정은 후속 PR.

**runbook §2 gated smoke** (`RUN_DINO_SMOKE=1`): 5 건 전부 PASSED (224²/1024²/non-multiple backbone + DETR load_builds + DETR forward_returns).
- 주의: DETR 2건은 transformers 4.56.2 의 `verify_backbone_config_arguments` 강화로 처음엔 실패. `DeformableDetrConfig(..., backbone=None, use_pretrained_backbone=False)` 를 `_build_detr_config` 에 추가 (surgical fix) — HF 의 ResNet 경로는 어차피 post-construction 에서 DINOv3 shim 으로 교체되므로 동작 동일.

**CODa → COCO subset 변환** (runbook §3, 시퀀스 0·1·2):
- training split: images=1984, annotations=11683, dropped(taxonomy=30281, occlusion=27157, projection=47)
- validation split: images=424, annotations=2465, dropped(taxonomy=5538, occlusion=6070, projection=18)
- `dropped_by_projection` 비율 < 1% (runbook 50% 기준 여유)
- 변환기에 실데이터 호환 4건 surgical fix 필요했음 (1차 세션엔 실CODa 접근 없어서 미노출):
  1. `_IMAGE_TEMPLATE`: `.jpg` → `.png` (CODa 실제 파일 확장자)
  2. `frames_for_split`: `ObjectTracking.{split}` 은 프레임 int 가 아닌 bbox JSON 경로 문자열 리스트 (`"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_{frame}.json"`). basename 에서 frame 추출 + 정렬 + 중복 제거.
  3. `annotations_from_frame`: 프레임 bbox JSON key 는 `"3dbbox"` (코드는 `"3dannotations"` 가정). 둘 다 허용으로 변경.
  4. `annotations_from_frame`: `isOccluded` 는 `entry.labelAttributes.isOccluded` 아래 중첩 (코드는 top-level 가정). 중첩 우선, top-level fallback.

**runbook §4 Dry-run 학습** (subset, 3 epoch, batch_size=2 @ 1024²):
- Hydra 호출 형태: `+detection=dinov3_detr_base +dataset=coda +backbone=dinov3_vitb16 dataset.coda_root=... dataset.annotations_path=...`. detection subgroup 의 `- /dataset: coda` 가 top-level 이 아닌 `detection.dataset` 으로 중첩 배치되는 Hydra 동작 때문에 top-level 도 CLI 로 별도 주입 필요 (scaffold 측은 추후 base.yaml defaults 재구성 검토 필요).
- 소요: 16:21 (GPU3, pre-unfreeze 4 steps/s, post-unfreeze 2 steps/s)
- Loss: epoch 0 step 0 = 934.79 → step 500 = 15.46 → epoch 1 평균 ~14 → epoch 2 평균 ~12 (단조감소 아니지만 추세 하락). 경고 주의: `save_every_n_epochs=1` + `epochs=3` 이면 cosine LR 이 epoch 3 종료 시점에 이미 0 도달하므로 epoch_003.pt 는 수렴 전 상태.
- "epoch 2: unfreezing backbone" 로그 정상 ✓
- Checkpoints: `outputs/checkpoints/dinov3_detr_base/epoch_{001,002,003}.pt` (각 467MB)
- W&B run: [super-dragon-1 (a8kb0ekx)](https://wandb.ai/j-soobin-daegu/terravision/runs/a8kb0ekx) — online 동기화, train/loss + train/lr + epoch 로깅 확인

**신규 — `scripts/eval_detection.py`** (runbook §5/§6 대응 정식 CLI):
- CLI: `--checkpoint / --val-annotations / --images-root / --output / --image-size / --batch-size / --score-threshold / --top-k`
- 재사용: `DinoV3DeformableDetr` / `CocoDetectionDataset` / `make_collate_fn` / `compute_coco_map` / HF `DeformableDetrImageProcessor.post_process_object_detection`
- Letterbox 역변환: HF post-process 에 target_sizes=(image_size, image_size) 로 letterbox-frame 좌표 받은 후 `scale = image_size / max(orig_h, orig_w)` 로 나눠 원본 pixel 좌표 복원 + image 경계 clamp.
- subset ckpt (epoch_003.pt) 실행 결과 — end-to-end 통과 확인용 (수치 자체는 full 학습에서 교체):
  - mAP = 0.0029, AP50 = 0.0133, AP75 = 0.0003 (3-epoch subset + cosine LR 조기 소진 이라 당연히 낮음)
  - per-class: pedestrian=0.0056, scooter=0.0145, pole=0.0058, bicycle=0.0056, vehicle=0.0, traffic_light=0.0, ...
  - 결과: `outputs/eval_phase1-2b_subset.json`

**후속 (다음 세션)**:
- runbook §5 Full 학습 — CODa training split 전체 (22 시퀀스) · 50 epoch. W&B online 유지. 예상 소요 12~48h.
- GPU 할당: **GPU0 + GPU3** (2026-04-25 사용자 지정). 현재 `train_detection.py` 는 단일-GPU 루프이므로 두 장을 한 학습에 쓰려면 DDP (`torchrun --nproc_per_node=2`) 도입 — Phase 1-2b scope 밖이라 사전에 ADR 확정 필요. 단일 장으로 충분하다고 판단되면 GPU3 단독 권장.
- runbook §6 Full validation 평가 — full validation split 변환 + `scripts/eval_detection.py` 로 베이스라인 mAP 확정.
- Full 수치 나오면 "사용자 검토 결과 (2차)" 섹션 작성 + runbook §8 PR 묶기.

### 2차 게이트 실행 — Full 학습 1차 시도 + 결함 발견 (2026-04-25 오후)

**Full 학습 1차 시도** (batch_size=4, GPU3, 8.5h 진행 후 중단):
- `coda_to_coco --split training` 전체 변환: images=19,511, annotations=215,615, dropped_proj=6,212 (<3%)
- `coda_to_coco --split validation` 전체 변환: images=4,176, annotations=45,729, dropped_proj=1,286
- W&B run: [feasible-dream-4 (y7hydwdd)](https://wandb.ai/j-soobin-daegu/terravision/runs/y7hydwdd)
- step rate: 2.6 steps/s (frozen) → 1.1 steps/s (post-unfreeze)
- VRAM 실측: 3.7 GB (frozen) → 14.7 GB (post-unfreeze)
- Loss: step 0 = 472 → epoch 7 step 37,150 = 4.85
- 저장된 체크포인트: epoch_005.pt (mAP=0.2104, AP50=0.4838 — daemon 평가 결과), epoch_010.pt

**Eval daemon 도입** (`scripts/eval_daemon.py`):
- 별도 GPU0 에서 polling 으로 새 ckpt 평가 → W&B 별도 run 으로 로깅
- W&B 로그가 안 보이는 문제 해결: `define_metric("epoch")` + `define_metric("val/*", step_metric="epoch")` 추가. 첫 시도 (`run.log(..., step=epoch)`) 는 wandb 0.26 의 step 의미 변경으로 history 가 서버에 송신 안 됨 — `epoch` 필드를 dict 에 포함하고 step 인자 제거하는 형태로 정정.
- backfill 로직: 데몬 재시작 시 `eval_epoch_*.json` 들 자동 재로깅.

**🚨 발견된 critical 결함 — `train_detection.build_optimizer`**:
- `if not p.requires_grad: continue` 필터 때문에 빌드 시점 frozen 인 backbone params 가 optimizer 에 전혀 추가되지 않음
- `set_backbone_frozen(False)` 가 epoch 2 에서 `requires_grad=True` 로 바꿔도 optimizer 는 backbone 을 모르므로 **AdamW step 이 backbone 가중치를 update 하지 않음**
- 결과: 1차 시도는 사실상 50 epoch 내내 frozen DINOv3 + head-only fine-tune 으로 학습 (의도와 다름)
- 체크포인트 검증: epoch_005.pt 의 `optimizer_state["state"]` 가 268 param tensor / 10.9M params 만 추적 (head + input_proj). DINOv3 backbone 388 tensor / 85.7M params 는 부재.
- 수정: `build_optimizer` 의 `requires_grad` 필터 제거 (frozen 상태에서도 backbone group 에 추가; AdamW 가 grad=None 인 step 을 자동 스킵하므로 안전).

**Latent test 버그 동시 fix**:
- `tests/test_det_metrics.py::test_offset_prediction_drops_ap_at_high_iou`: shift=15 가정 (IoU 주석 0.52 라고 적힘) — 실제 IoU=0.32 라 AP50=0 → AP50=1.0 단언 실패. shift=7 (IoU=0.587) 로 수정.
- `tests/test_coda_to_coco.py::test_convert_coda_split_end_to_end`: 변환기 fix 후 `.png` 사용 → 테스트 fixture 의 `.jpg` assertion 도 `.png` 로 갱신.

**검증**: 변경 후 `pytest -q` → **52 passed, 8 skipped** (1차 승인 상태와 동일 수준 회복). gated `RUN_DINO_SMOKE=1` smoke 5건도 모두 PASSED.

**다음 단계 (2026-04-26 이후)**: 단일 GPU 브랜치(`objectdetection`)에서 fix 적용 후 커밋. DDP 시도는 별도 브랜치로 분기 — 시간 안 맞으면 단일 GPU 브랜치로 롤백 가능.

**이 세션에서 변경된 파일**:
- `models/detection/detr_head.py` — `_build_detr_config` 에 `backbone=None, use_pretrained_backbone=False` (transformers 4.56 호환)
- `training/datasets/coda_to_coco.py` — `.png` 확장자, `frames_for_split` path 파싱, `3dbbox` key 허용, `labelAttributes.isOccluded` 중첩 파싱
- `training/train_detection.py` — `class_labels` 1-indexed → 0-indexed 변환, `build_optimizer` 가 frozen backbone 도 추적
- `scripts/eval_detection.py` — 평가 CLI 신규 작성 (label+1 역변환 포함)
- `scripts/eval_daemon.py` — 신규 polling 데몬 (define_metric + backfill)
- `tests/test_det_metrics.py`, `tests/test_coda_to_coco.py` — fixture 갱신
- `data/annotations/coda_{training,validation,training_subset,validation_subset}_coco.json` — 변환 결과
- (학습 산출물 — 디렉토리 .gitignore 처리됨): `outputs/checkpoints/dinov3_detr_base{,_full}/`, `outputs/eval_phase1-2b_*.json`, `wandb/run-*/`

## 검토 요청 노트 (1차 — 코드부분)

- **승인 범위 명확화 요청**: 본 단계 PR 의 승인 기준을 다음과 같이 제안.
  - **1차 승인 (이 PR)**: 코드 구조 · Hydra config · lazy-import · 60 pytest green. **아키텍처 · 인터페이스 수준 리뷰**.
  - **2차 승인 (후속 세션)**: GPU 환경에서 `RUN_DINO_SMOKE=1` 단발 smoke + CODa training split 1회 학습 + mAP 수치 append.
- **HF 내부 결합 지점**: `DinoV3DeformableDetr.load()` 은 HF 의 `model.model.backbone.conv_encoder` 와 `model.model.input_proj` 를 직접 교체한다. transformers 버전 업그레이드 시 이 두 경로가 깨지면 알림이 필요. 최소한 Phase 1-2c 까지는 `transformers==4.56.x` 로 pin 하는 것을 권장.
- **PyTorch fallback for deformable attention**: `disable_custom_kernels=true` 로 두었으므로 첫 학습 속도는 custom CUDA op 대비 느림. 학습 완료 후 Phase 1-2c 에서 custom op 빌드 검토.
- **DINOv3 backbone LR**: base LR 2e-4 / backbone LR 2e-5 로 10× 낮춤. 플랜 "첫 2 epoch 동결 → 이후 low LR 파인튠" 과 일치. 실학습에서 loss 추이 보고 재조정 가능.
- **Jetson 배포와의 결합도**: HF DeformableDetr 경로를 택했으므로 Phase 3 TRT export 는 표준 PyTorch→ONNX→TRT. deformable attention ONNX 지원은 HF 쪽에서 제공 — export 단계에서 다시 검증.

## 사용자 검토 결과

- **2026-04-24 — 1차 리뷰 승인**: 코드 구조 · Hydra config · lazy-import · 52 pytest green (8 gated skip) · HF DeformableDetr 어댑터 구조 수용. 2차 게이트 (GPU 환경 `RUN_DINO_SMOKE=1` smoke + CODa training split 1회 학습 + mAP 수치) 는 후속 세션에서 진행 후 본 섹션에 append. 최종 승인 전까지는 Phase 1-2c 착수 보류.
- **2026-04-XX — 2차 리뷰**: (후속 — GPU 환경에서 실학습 실행 후 수치 append)
