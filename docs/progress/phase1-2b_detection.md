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
