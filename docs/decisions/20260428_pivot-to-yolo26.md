# ADR 2026-04-28 — Detection 라인 전환: DINOv3+Deformable DETR → Ultralytics YOLO26 fine-tune

- **상태**: 확정 (사용자 승인 2026-04-28)
- **일자**: 2026-04-28
- **관련 phase**: 1-3 (YOLO26 baseline, 신설). 1-2c (DETR 캠퍼스 fine-tune) 폐기.
- **관련 ADR**:
  - `docs/decisions/20260424_detr-head-library.md` — Phase 1-2b DETR 헤드 결정. 본 ADR 로 detection 라인은 종료(코드/체크포인트는 보존).
  - `docs/decisions/20260422_dinov3-backbone.md` — DINOv3 백본 결정. detection 에서는 사용 종료, **Phase 2 Traversability Segmentation 에서 재활용 가능성 유지**.
  - `docs/decisions/20260422_coda-primary-dataset.md` — CODa primary 원칙은 유지.
- **관련 파일**:
  - 신규: `training/datasets/coda_to_yolo.py`, `training/train_yolo.py`, `scripts/eval_yolo.py`, `scripts/verify_yolo_dataset.py`
  - 신규 config: `configs/dataset/coda_yolo.yaml`, `configs/dataset/coda_yolo_taxonomy.yaml`, `configs/detection/yolo26_{n,s,m,l}.yaml`
  - 보존: `outputs/checkpoints/dinov3_detr_base_full/*`, `configs/detection/dinov3_detr_base.yaml`, `training/train_detection.py`, `models/{backbone,detection}/*`

## 배경

Phase 1-2b 에서 DINOv3 ViT-B/16 + HF Deformable DETR (imgsz=1024, single-scale) 50-epoch DDP 학습을 완료해 베이스라인을 확보했다.

| 지표 | epoch_050 | 비고 |
|------|-----------|------|
| mAP@[.50:.95] | **0.623** | 21h DDP GPU0-2 |
| AP50 | **0.925** | |
| AP75 | 0.700 | |
| AP_small | **0.35** | AP_large 0.74 의 절반 미만 — 약점 |
| AP_medium | ~0.55 | |
| AP_large | 0.74 | |

per-class 약점은 motorcycle (0.518) / cone (0.505) / traffic_light (0.470) / fire_hydrant (0.455) 으로, 작은 객체·희소 클래스에 집중. Phase 1-2c (자체 캠퍼스 데이터로 fine-tune) 는 열린 질문 5건 해소를 대기 중이었다.

이 시점에 사용자는 다음 두 가지 관찰을 근거로 detection 라인을 YOLO 기반으로 전환할 것을 결정했다:

1. **AP_small 격차** 가 캠퍼스 데이터 추가만으로 메워지기 어려움. multi-scale·해상도 상향 같은 아키텍처 변경이 별도로 필요한 상황.
2. **학습/배포 비용** — DINOv3 ViT-B/16 86M backbone 단독으로 21h × 3 GPU. Phase 3 Jetson Orin 온보드 배포 목표(≥ 15 FPS @ 1280×720) 와 충돌하기 시작.

YOLO26 (Ultralytics, 2026-01-14 릴리스) 은 두 문제에 동시에 답을 줄 수 있는 후보다.

## 옵션

### A. 현 라인 유지 — 1-2c 진행 (DETR 캠퍼스 fine-tune)

| 장점 | 단점 |
|---|---|
| 베이스라인이 이미 확보됨 (mAP=0.623) | AP_small 격차 해소가 데이터 추가만으로 어려움 |
| HF 생태계 일관성, 라이선스 Apache-2.0 | imgsz 상향(1280/1536) → 학습 비용 추가 폭증 |
| Phase 2 segmentation 분기 시 백본 공유 | Edge 배포 경로 복잡 (ViT는 ONNX 변환은 되나 TRT 효율 떨어짐) |

### B. YOLO26 fine-tune 라인 신설 (제안)

| 장점 | 단점 |
|---|---|
| **STAL (Small-Target-Aware Label Assignment)** 가 AP_small 약점 직접 공략 | **AGPL-3.0** 라이선스 — 상용 배포 시 소스 공개 의무 |
| n/s/m/l/x 5종으로 edge → server 까지 동일 코드베이스 | DINOv3 backbone 미사용 → Phase 1-2b 자산 직접 재사용 못함 (백본 가중치) |
| NMS-free + ProgLoss + MuSGD optimizer (안정 학습) | 91-class 통합 head 학습 시 COCO 70개 클래스에서 catastrophic forgetting 위험 |
| nano 학습 시간 ~1-2h × 1 GPU (실험 회전 ~10×) | Ultralytics 자체 wandb 통합과 기존 `_wandb_init()` 중복 — 비활성화 필요 |
| TRT/ONNX/CoreML 변환 1-line, Jetson 배포 검증 풍부 | YOLO26 nightly 안정성 (릴리스 후 약 4개월) |

### C. 두 라인 병행 (DETR 1-2c + YOLO 신설)

| 장점 | 단점 |
|---|---|
| 보험 — 한쪽이 막혀도 다른 쪽 진행 | GPU/시간 비용 가장 큼 |
| 직접 비교 가능 | 진행 파일 / 체크포인트 / W&B 운영 부담 ↑ |

## 결정 — 옵션 B (YOLO26 fine-tune 라인 신설, DETR 라인 종료)

### 핵심 결정 사항

| ID | 결정 | 비고 |
|----|------|------|
| **D1** | **91-class 통합 head** (COCO80 0..79 + CoDA 신규 11 80..90) | 추론 시 운영 클래스만 후처리 필터. CoDA 신규 11 = scooter, tree, pole, sign, bollard, cone, barrier, bike_rack, trash_can, service_vehicle, golf_cart |
| **D2** | n → s → m → l 4단계 순차 (x 제외) | sub-phase 1-3a/b/c/d, 게이트 통과 시 다음 진행 |
| **D3** | Detection only (RUGD seg 별도 phase) | YOLO26-seg 도입은 Phase 2 결정 시점으로 미룸 |
| **D4** | **Phase 1-2c 폐기** (DETR 캠퍼스 fine-tune) | progress 파일에 ⛔ Closed 마커. 1-2b ckpt 보존 |
| **D5** | **Vehicle dispatch** — CoDA 'vehicle'(op id 5) → 라벨 raw name 기준 분배 | Car→2(car), Bus→5(bus), Truck/Pickup Truck/Delivery Truck→7(truck), Service Vehicle/Utility Vehicle→89(service_vehicle), Golf Cart→90(golf_cart) |
| **D6** | 기본 imgsz=640 | 1-3a 결과 AP_small 미달 시 1024 sweep |
| **D7** | optimizer=auto (YOLO26 default MuSGD) | AdamW fallback |
| **D8** | pretrained `yolo26{n,s,m,l}.pt` + `freeze=10` | 91-class head는 ultralytics 자동 reset, backbone 부분 동결로 catastrophic forgetting 완화 |
| **D9** | W&B 충돌 회피 | Ultralytics `SETTINGS.update({"wandb": False})` 후 기존 `_wandb_init()` 재사용 + `on_fit_epoch_end` 콜백에서 직접 push |
| **D10** | Phase 번호 = **1-3 재배치** | YOLO26 baseline = 1-3, 기존 Tracking 1-3→1-4, BEV 1-4→1-5, Integration 1-5→1-6 |

### 게이트 체계 (Phase 1-3)

| Sub | 모델 | 시간 | 통과 조건 |
|-----|------|-----|-----------|
| 1-3a | nano | ~1-2h | mAP ≥ 0.50, **AP_small ≥ 0.40** (DETR 0.35 + 0.05), COCO80 mini-val 회귀 ≤ -0.05. 미달 시 imgsz→1024 sweep 분기 |
| 1-3b | small | ~3h | mAP ≥ 0.58, AP_small ≥ 0.45 |
| 1-3c | medium | ~6h | mAP ≥ 0.62 (DETR parity), AP_small ≥ 0.50 |
| 1-3d | large | ~12-15h | mAP ≥ 0.65, AP_small ≥ 0.52 → **YOLO26 baseline 확정** |

### 평가 메트릭 호환성

`evaluation/det_metrics.compute_coco_map` 을 그대로 재사용해 12-stat 출력 + per-class AP 를 산출. `scripts/eval_yolo.py --mode coda16` 은 91-class 예측을 CoDA 16-class 운영 ID 로 reverse-map 한 뒤 동일 메트릭으로 점수화 → Phase 1-2b 와 직접 비교 가능.

## 리스크 및 대응

### R1. **AGPL-3.0 라이선스 (High severity)**

Ultralytics 는 AGPL-3.0 으로 배포된다. 본 저장소가 ultralytics 를 import 하는 순간 파생 저작물의 소스 공개 의무가 따라붙는다.

- **대응**:
  - 본 ADR 에 명시. README 의 License 섹션에도 추가.
  - 학술/연구 범위에서는 무관. **상용 배포 전에 반드시 재검토**.
  - 상용 배포 시점에 (a) Ultralytics 상용 라이선스 구매 또는 (b) RT-DETR (Apache-2.0) 같은 대체로 head 교체 옵션 평가.
- **추가 라이선스 누적**: 학습 데이터(CODa CC BY-NC-SA 4.0 + UT NCA) 는 그 자체로 상용 금지이므로 라이선스 충돌은 학습 결과물에 한해서는 즉각적이지 않다. 다만 distillation/ablation 으로 데이터 라이선스가 끊긴 후 모델만 배포하는 시나리오에서 AGPL-3.0 의 부담이 단독으로 남는다.

### R2. **Catastrophic forgetting (Med severity)**

91-class head 학습 중 COCO 70개 클래스의 prior 가 손상될 수 있음. 대응:
- `freeze=10` (cfg) 로 backbone 초기 stage 동결.
- 매 5 epoch 마다 COCO80 mini-val 회귀 측정 (`scripts/eval_yolo.py --mode coco80_regression`). 1-3a 게이트에 회귀 임계 -0.05 포함.
- 회귀가 임계를 넘으면 `freeze` 값을 키우거나 lr0 을 낮춰 recover.

### R3. **STAL 효과 미달성 (High severity, 본 ADR 채택의 핵심 가정)**

본 ADR 채택의 핵심 가정인 "STAL 이 AP_small 을 끌어올린다" 가 빗나갈 가능성. 대응:
- 1-3a 미달 시 즉시 imgsz 640 → 1024 재학습.
- 1-3d 까지 가도 AP_small ≥ 0.52 미달이면 multi-scale (`rect=False`, `multi_scale=True`) + copy_paste 강도 상향.
- 그래도 미달이면 본 ADR 재검토 — DINOv3+DETR 라인 부활 또는 RT-DETR 평가.

### R4. **Vehicle dispatch 라벨 누락 (Med severity)**

`coda_raw_to_yolo` 에 등재되지 않은 raw name 이 CODa 업데이트로 들어오면 변환기가 ValueError 로 abort. 이는 의도된 동작 — silent loss 방지. CoDA 가 새 vehicle subtype 을 추가하면 dispatch 테이블 업데이트가 필요.

### R5. **W&B 충돌 (Low severity)**

Ultralytics 자체 W&B integration 과 기존 `_wandb_init()` 가 동시 활성화되면 동일 run 에 두 채널이 쓰여 step 카운트가 어긋남. `D9` 로 disable. 1-3a 첫 epoch 후 로그 단일 소스 검증 필요.

### R6. **GPU 메모리 (Low severity)**

large 단계 batch=8 이 24GB 카드에서 빠듯. DDP 2-GPU 또는 imgsz 유지 + accumulate=2 옵션 보유.

### R7. **YOLO26 nightly 안정성 (Low severity)**

릴리스 후 약 4개월 (2026-04-28 시점). `pyproject.toml` 에 `ultralytics>=8.3.50` floor 핀.

## 채택하지 않은 대안

- **A (DETR 라인 유지)**: Phase 1-2c 진행 → AP_small 격차 해소가 데이터 추가만으로 어렵고 학습 비용이 누적적으로 증가. edge 배포 비용까지 고려하면 라인 자체를 바꾸는 편이 합리적.
- **C (병행)**: 두 라인 모두 운영하기에는 GPU/문서 비용이 크고, 본 시점은 비교가 아니라 결정의 단계.

## 후속 작업

- Phase 1-2c 진행 파일에 ⛔ Closed 마커.
- 신규 진행 파일 `docs/progress/phase1-3_yolo26.md` 생성, sub 1-3a~d 체크리스트 골격.
- 기존 1-3 (Tracking) 이후 phase 파일 rename: 1-3→1-4, 1-4→1-5, 1-5→1-6.
- `docs/PLAN.md`, `docs/progress/README.md`, top-level `README.md` 갱신.
- `pyproject.toml` 에 `ultralytics>=8.3.50` 추가, `pip check` + `numpy>=1.26` 호환 검증.
- 1-3d 통과 시 본 ADR 의 § "결정" 섹션에 final baseline 메트릭(mAP/AP50/AP_small) stamp.
