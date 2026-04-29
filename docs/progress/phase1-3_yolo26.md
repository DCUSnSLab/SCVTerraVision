# Phase 1-3 — YOLO26 fine-tune baseline (Detection)

- **상태**: ⏳ 착수 대기 (구현 끝, 학습 미실행)
- **시작일**: 2026-04-28 (코드 작성)
- **완료일**: —
- **담당 PR**: (작성 후 링크)
- **선행 단계**: Phase 1-2b 🟢 승인완료 (DINOv3+DETR 베이스라인 mAP=0.623, AP50=0.925)
- **승계**: Phase 1-2c (DETR 캠퍼스 fine-tune) ⛔ Closed (ADR 20260428 로 대체)
- **관련 ADR**: [`docs/decisions/20260428_pivot-to-yolo26.md`](../decisions/20260428_pivot-to-yolo26.md)

## 목표

YOLO26 사전학습 가중치(`yolo26{n,s,m,l}.pt`)를 91-class (COCO80 + CoDA 신규 11) 통합 head 로 fine-tune 해 새 detection 베이스라인을 확정한다. 핵심 KPI:

- **AP_small 0.35 → ≥ 0.52** (DETR 베이스라인 대비 +0.17, STAL 기대 효과)
- **mAP ≥ 0.65** (DETR 0.623 대비 +0.027 이상)
- **COCO80 mini-val 회귀 ≤ -0.05** (catastrophic forgetting 한계)

## 1-2b 베이스라인 요약 (재진입 시 비교 기준)

| 지표 | DETR epoch_050 | YOLO 1-3 목표 | 게이트 |
|------|---------------|--------------|-------|
| mAP@[.50:.95] | 0.623 | ≥ 0.65 (large) | 1-3d |
| AP50 | 0.925 | ≥ 0.92 | — |
| AP_small | 0.35 | ≥ 0.52 | 1-3d |
| AP_medium | ~0.55 | ≥ 0.60 | — |
| AP_large | 0.74 | ≥ 0.74 | — |

## Sub-phase 게이트

| Sub | 모델 | 입력 | 시간(추정) | 통과 조건 |
|-----|------|------|-----------|-----------|
| 1-3a | nano | A 변환 완료, configs/detection/yolo26_n.yaml | 1-2h × 1GPU | mAP ≥ 0.50 **AND** AP_small ≥ 0.40 **AND** COCO80 회귀 ≤ -0.05. 미달 시 imgsz 1024 sweep 분기 |
| 1-3b | small | 1-3a 통과 | ~3h | mAP ≥ 0.58, AP_small ≥ 0.45 |
| 1-3c | medium | 1-3b 통과 | ~6h | mAP ≥ 0.62 (DETR parity), AP_small ≥ 0.50 |
| 1-3d | large | 1-3c 통과 | ~12-15h (DDP 2GPU) | mAP ≥ 0.65, AP_small ≥ 0.52 → **YOLO26 baseline 확정** |

## 데이터 파이프라인 (1-3a 진입 게이트 — 학습 전 필수)

- [x] `python -m training.datasets.coda_to_yolo --split training` (2026-04-28, 26s)
  - 출력: 19,511 images / 215,615 annotations (Phase 1-2b 수치와 일치) ✓
- [x] `python -m training.datasets.coda_to_yolo --split validation` (2026-04-28, 7s)
  - 출력: 4,176 images / 45,729 annotations ✓
- [x] `python -m scripts.verify_yolo_dataset --sample-n 50` (2026-04-28)
  - cross-check 0 errors, validate 0 errors, 클래스 분포 정상
  - 주의: golf_cart(90) 0건 — CoDA에 'Golf Cart' raw 라벨이 없거나 모두 occlusion/area 필터에서 drop됨. 학습 신호 없음 → 1-3d 통과 후 taxonomy 정리 검토.
  - bus(5) 11건, motorcycle(3) 147건, fire_hydrant(10) 387건 등 long-tail 다수.
- [x] tiny smoke: `+detection=yolo26_n +dataset=coda_yolo training.epochs=1 imgsz=320 batch=32 workers=0` (2026-04-28, 약 30분)
  - **첫 실행은 ~16분 cache scan + 12분 1-epoch + 2min val** (workers=0 이슈)
  - 결과 (1 epoch만의 sanity 수치, 성능 목표 아님): mAP50=0.0236, mAP50-95=0.00827, P=0.757, R=0.024
  - 전체 파이프라인 무결: weights yolo26n.pt 자동 다운로드, 91-class head 자동 reset (606/708 transferred), freeze=10 적용, best.pt 5.5MB 저장, val_batch_pred.jpg 시각화 정상.
  - **Hot-fix**: 체크포인트 경로가 `runs/detect/outputs/checkpoints/yolo26_n_smoke/` 로 떨어짐 (ultralytics SETTINGS.runs_dir prepended). `_resolve_project_dir(cfg)` 추가로 absolute path 강제 — 다음 실행부터 `outputs/checkpoints/<name>/` 직행. (2026-04-28 commit)

## 학습 체크리스트

### 1-3a (nano) — 1차 시도 (2026-04-28, 게이트 미달)

- [x] 4-GPU DDP 학습 (100 epoch, batch=64, imgsz=640, MuSGD, freeze=10) — 4.010h
- [x] W&B online stream 확인 (run id: `yolo26_n_phase1-3a_20260428_143335`)
- [x] best.pt 저장: `outputs/checkpoints/yolo26_n_phase1-3a/weights/best.pt` (5.6MB, epoch 100, mAP50-95=0.4889 by ultralytics val)
- [x] eval_yolo --mode coda16 (pycocotools, DETR 호환 메트릭)
  - 결과 JSON: `outputs/eval_phase1-3a_nano_coda16.json`

#### DETR baseline 비교 (Phase 1-2b epoch_050 vs YOLO26-n best.pt)

| 지표 | DETR (ViT-B, imgsz=1024) | YOLO26-n (imgsz=640) | Δ |
|------|----:|----:|----:|
| mAP@[.50:.95] | 0.623 | **0.450** | -0.173 |
| AP50 | 0.925 | 0.738 | -0.187 |
| AP75 | 0.700 | 0.482 | -0.218 |
| **AP_small** | 0.350 | **0.134** | **-0.216** |
| AP_medium | 0.550 | 0.369 | -0.181 |
| AP_large | 0.740 | 0.582 | -0.158 |

#### 게이트 판정 (mAP ≥ 0.50, AP_small ≥ 0.40, COCO80 회귀 ≤ -0.05)

- mAP = 0.450 (-0.05 미달)
- **AP_small = 0.134 (-0.266 대폭 미달, 핵심 실패 지점)**
- COCO80 회귀: 미평가 (외부 mini-val 없음)
- **결론: 게이트 미달 → ADR plan의 1차 fallback (imgsz 1024 sweep) 검토**

#### 원인 분석

- **모델 크기 격차**: DETR ViT-B 86M vs YOLO26-n 2.6M (33×). nano capacity 한계.
- **입력 해상도**: DETR 1024 vs YOLO 640. AP_small 격차의 주요 원인 가능성.
- STAL/NMS-free 등 YOLO26 혁신은 capacity 충분 시 빛남. nano 단독으론 부족.

#### 보존된 산출물

- best.pt (학습 weights), last.pt
- results.csv, BoxPR_curve / BoxF1_curve / confusion_matrix 시각화
- val_batch*_pred.jpg / labels.jpg 시각화
- W&B online run (계정 j-soobin-daegu)

### 1-3a (nano) — 2차 시도 (imgsz=1024 fallback, 부분 통과)

- [x] 4-GPU DDP 학습 (100 epoch, batch=32, imgsz=1024, freeze=10) — 6.001h
  - GPU 사용 ~5GB/24GB (AMP 효과로 batch=32에서도 매우 여유)
  - W&B run: `yolo26_n_phase1-3a_1024_20260428_191411` (j-soobin-daegu)
- [x] best.pt 저장: `outputs/checkpoints/yolo26_n_phase1-3a_1024/weights/best.pt` (5.7MB, epoch 100, mAP50-95=0.5478 by ultralytics val)
- [x] eval_yolo --mode coda16: `outputs/eval_phase1-3a_nano_1024_coda16.json`

#### DETR baseline / 1-3a 1차 / 1-3a 2차 비교

| 지표 | DETR (1-2b) | 1-3a 1차 (640) | **1-3a 2차 (1024)** | Δ 1차→2차 | Δ vs DETR |
|------|----:|----:|----:|----:|----:|
| mAP@[.50:.95] | 0.623 | 0.450 | **0.505** | **+0.055** | -0.118 |
| AP50 | 0.925 | 0.738 | **0.784** | +0.046 | -0.141 |
| AP75 | 0.700 | 0.482 | **0.562** | +0.080 | -0.138 |
| **AP_small** | 0.350 | 0.134 | **0.241** | **+0.107 (+80%)** | -0.109 |
| AP_medium | 0.550 | 0.369 | **0.433** | +0.064 | -0.117 |
| AP_large | 0.740 | 0.582 | **0.640** | +0.058 | -0.100 |
| AR_100 | — | 0.647 | **0.706** | +0.059 | — |

#### 게이트 판정 (mAP ≥ 0.50, AP_small ≥ 0.40)

- mAP = **0.505** → ✅ **통과**
- AP_small = 0.241 → ❌ 미달 (그러나 1차 대비 +80% 개선)
- **결론: 부분 통과 — AP_small은 nano 단독으로 한계, small/medium 단계에서 추가 capacity 필요**

#### imgsz=1024 효과 정리

- 모든 metric 일관되게 +0.04~+0.11 끌어올림. AP_small 효과 가장 큼 (+80%).
- nano + imgsz=1024 천장은 mAP@[.50:.95] ≈ 0.505, AP_small ≈ 0.24 부근.
- close_mosaic=10 효과는 미미했음 (epoch 91→100 +0.0035 정도, 1차의 큰 점프 +0.035와 다른 양상).

### 1-3a (nano) — 3차 시도 (Pseudo-labeling, 사용자 issue 진단 후)

**Trigger**: 사용자가 1차/2차 결과로 다음 두 이슈 발견 (2026-04-29):
1. 재학습 후 기존 yolo의 정확한 bbox 가 살짝 어긋남 (차량 등)
2. 기존 COCO80 클래스 검출 성능 저하 (catastrophic forgetting)

**진단 (사실 확인 후)**:
- 사용자 가설 (16:9 letterbox 부재) 은 사실 X. CoDA = 1224×1024, yolo26 사전학습 = 640 정사각형, ultralytics 가 letterbox 자동.
- 진짜 원인: (a) catastrophic forgetting, (b) **CoDA 라벨 자체가 LiDAR 3D→2D 투영이라 시각적 박스보다 약간 어긋남** — 그걸 학습하니 추론도 어긋남.

**해결 — Pseudo-labeling (D11~D17, 2026-04-29 결정)**:

| ID | 결정 |
|----|------|
| D11 | 사전학습 yolo26n.pt 추론 결과를 학습 라벨로 사용 |
| D12 | Vehicle 8 raw subtype (Car/Truck/Bus/Pickup/Delivery/Service Vehicle/Utility Vehicle/Golf Cart) 전면 pseudo 대체. service_vehicle(89)/golf_cart(90) 학습 신호 0 → deprecated |
| D13 | COCO80 6 overlap (Pedestrian/Bike/Motorcycle/Traffic Light/Fire Hydrant/Bench) pseudo 대체 |
| D14 | CoDA-only 9 (scooter/tree/pole/sign/bollard/cone/barrier/bike_rack/trash_can) 만 CoDA 라벨 유지 |
| D15 | Pseudo 추론 모델 = yolo26n.pt |
| D16 | conf=0.25, iou=0.7, imgsz=1024 |
| D17 | small 진입 보류 — 본 시도 게이트 통과 후 결정 |

- [x] taxonomy 수정 (`coda_yolo_taxonomy.yaml` D12/D13 14 raw 클래스 → coda_dropped 이동)
- [x] `training/datasets/pseudo_label_coda.py` 신규 (yolo26n inference)
- [x] `training/datasets/merge_pseudo_with_coda.py` 신규 (cls 0..79 pseudo + cls 80..88 coda merge)
- [x] `tests/test_merge_pseudo_with_coda.py` 7 케이스 (D12/D13/D14 검증) + 기존 `test_coda_to_yolo.py` v2 동작에 맞게 갱신 — pytest 79 passed, 8 skipped
- [x] `configs/dataset/coda_yolo_v2.yaml` 신규
- [x] Pseudo-label 생성 (단일 GPU, 15분 32초): train 19,511 / val 4,176
- [x] Merge: train 152,095 pseudo + 93,802 coda-only = 245,897 box (v1 대비 +14%); val 32,240 + 20,093 = 52,333. 폐기 박스 train 121,813 / val 25,636.
- [x] verify_yolo_dataset pass — COCO80 30+ 클래스 신규 학습 신호 (person 47.58%, car 5.65%, backpack 2.27% 등). CoDA-only 분포 변동 없음. service_vehicle/golf_cart = 0.

#### 학습 + 평가 (완료)
- [x] 4-GPU DDP 학습 완료 (100 epoch, batch=32, imgsz=1024, 5.915h, best ep=97 mAP50-95=0.4387)
  - 1차 시도 실패 (image symlink 으로 v1 cache 사용) → swap fix (v1 labels을 v2 내용으로 교체) 후 재시작
- [x] eval_yolo --mode coda16: `outputs/eval_phase1-3a_nano_pseudo_coda16.json`
- [x] 시각적 비교 sample 생성: `outputs/compare/{2nd_v1_1024, 3rd_pseudo_1024}/*.jpg` (8장)

#### 1차 / 2차 / 3차 / DETR 비교

| 지표 | DETR | 1차 (640) | 2차 (1024) | **3차 (1024+pseudo)** |
|------|----:|----:|----:|----:|
| mAP@[.50:.95] | 0.623 | 0.450 | **0.505** ✅ | 0.298 |
| AP50 | 0.925 | 0.738 | 0.784 | 0.492 |
| AP75 | 0.700 | 0.482 | 0.562 | 0.320 |
| AP_small | 0.350 | 0.134 | 0.241 | 0.098 |
| AP_medium | 0.550 | 0.369 | 0.433 | 0.245 |
| AP_large | 0.740 | 0.582 | 0.640 | 0.389 |

#### ⚠️ 메트릭 vs 실제 정확도 괴리 (중요 진단)

3차 mAP가 압도적으로 낮은 원인 — eval_yolo coda16 모드는 **CoDA val GT (LiDAR 3D→2D 투영 박스)** 로 평가. 그런데 우리 pseudo-labeled 모델은 **시각적 박스로 학습** 됨 → 같은 객체를 잘 detect해도 GT(LiDAR) vs 추론(시각) IoU 가 0.5~0.75 즈음에 집중되어 mAP@[.5:.95] 큰 손실.

즉 사용자 우려 #1 (시각적 bbox 정확도) 은 **정성적으로 해결됐을 가능성 큼**. 하지만 현재 metric 으로는 측정 불가. **시각적 inspection 필수**:
- `outputs/compare/2nd_v1_1024/*.jpg` (8장) — 2차 결과
- `outputs/compare/3rd_pseudo_1024/*.jpg` (8장) — 3차 결과
- 같은 8장 이미지에 두 모델 추론. bbox 가 차량/사람에 잘 fit 하는지 직접 비교.

#### 게이트 재판정

| 게이트 | 임계 | 1차 결과 | 2차 결과 | 3차 결과 |
|--------|------|---------|---------|---------|
| mAP@[.50:.95] ≥ 0.50 | 0.50 | 0.450 ❌ | 0.505 ✅ | 0.298 ❌ (메트릭 불일치 가능) |
| AP_small ≥ 0.40 | 0.40 | 0.134 ❌ | 0.241 ❌ | 0.098 ❌ |

- 2차가 quantitative 게이트(mAP 통과) 측면에서 nano 최고 결과
- 3차는 정성적(시각적 박스 정확도) 측면에서 더 좋을 가능성 (시각 검증 필요)

#### 다음 단계 후보 (사용자 결정 대기)

| 옵션 | 내용 | 예상 시간 |
|------|------|---------|
| **A** | 2차 (yolo26_n_phase1-3a_1024) 채택 + 1-3b small + v1 (CoDA-only 매핑) | DDP 4-GPU ~6-8h |
| **B** | 3차 (pseudo) 채택 + 1-3b small + v2 (pseudo) — small capacity 로 91-class 학습 개선 기대 | DDP 4-GPU ~6-8h |
| C | hybrid — small + v2 (pseudo) 로 시도, 시각적 검증 후 최종 결정 | 같음 |
| D | 1-3a sweep 추가 (lr0/AdamW) — ROI 낮음 | 다중 |

### 1-3b (small) — Pseudo 라벨 + 1024 (D17 결정에 따라 진행)

사용자가 3차 nano 결과를 시각적으로 평가하여 pseudo-labeling 노선이 더 정확하다고 판단 → small 단계로 escalate.

- [x] 4-GPU DDP 학습 (80 epoch, batch=16, imgsz=1024) — 7.152h
  - GPU 사용 ~3.7GB/24GB (매우 여유)
  - W&B run: `yolo26_s_phase1-3b_pseudo_20260429_110153`
- [x] best.pt 저장 (epoch 54, ultralytics val mAP50-95=0.5003)
- [x] eval_yolo --mode coda16: `outputs/eval_phase1-3b_small_pseudo_coda16.json`
- [x] 시각 비교 sample (`outputs/compare/small_v2_1024/*.jpg` 8장)

#### 1차 / 2차 / 3차 nano / 1-3b small 비교 (eval_yolo coda16)

| 지표 | DETR | 1차 nano | 2차 nano | 3차 nano | **small** |
|------|----:|----:|----:|----:|----:|
| mAP@[.5:.95] | 0.623 | 0.450 | **0.505** | 0.298 | 0.354 |
| AP50 | 0.925 | 0.738 | 0.784 | 0.492 | 0.537 |
| AP75 | 0.700 | 0.482 | 0.562 | 0.320 | 0.394 |
| AP_small | 0.350 | 0.134 | 0.241 | 0.098 | **0.165** |
| AP_medium | 0.550 | 0.369 | 0.433 | 0.245 | 0.309 |
| AP_large | 0.740 | 0.582 | 0.640 | 0.389 | 0.449 |

#### 핵심 관찰

- Small이 3차 nano 대비 모든 metric +0.05~+0.07 향상 (capacity 효과 확인)
- 그러나 **v2 pseudo 라인은 2차 nano (v1) metric 보다 항상 손해**: CoDA val GT (LiDAR 투영) vs 추론 (시각 박스) IoU 격차로 mAP 측정에서 -0.10~-0.15 차감
- 1-3b 게이트 (mAP ≥ 0.58, AP_small ≥ 0.45) **미달** — quantitative

#### 정성적 평가 (사용자 시각 비교)

- 3차 nano 결과가 2차 nano 대비 시각적 박스 정확도 우수 — 사용자 확인됨
- small (1-3b) 은 더 큰 capacity로 추가 향상 기대
- 시각 비교 자료: `outputs/compare/{2nd_v1_1024, 3rd_pseudo_1024, small_v2_1024}/*.jpg` (각 8장 동일 이미지)

#### 다음 단계 후보 (사용자 결정 대기)

| 옵션 | 내용 | 시간 |
|------|------|------|
| **A** | medium + v2 pseudo 직행 — capacity ↑로 metric+정성 모두 향상 | DDP 4-GPU ~15-20h |
| **B** | BDD100K 합치기 (시각 박스 풍부, 변환기 이미 보유) — pseudo 보강 | 변환 ~30분 + 학습 6h |
| **C** | small 그대로 production 베이스라인 채택, 1-4 (Tracking) 진입 | 즉시 |
| D | CoDA GT 시각 박스로 자체 라벨링 — 비용 큼 | days |

### 1-3b (small)

- [ ] (1-3a 통과 후) 학습 실행 + 평가
- [ ] 게이트 판정 → 1-3c

### 1-3c (medium)

- [ ] (1-3b 통과 후) 학습 + 평가
- [ ] DETR parity 도달 시 ADR 본문 업데이트

### 1-3d (large) — final baseline

- [ ] (1-3c 통과 후) 학습 + 평가
- [ ] ADR `20260428_pivot-to-yolo26.md` 의 결정 섹션에 final 메트릭 stamp
- [ ] `docs/progress/README.md` 1-3 → 🟢, 1-4 (Tracking) ⏳ 활성화

## 산출물 (예상)

- `outputs/checkpoints/yolo26_{n,s,m,l}/weights/{best,last}.pt`
- `outputs/eval_phase1-3{a,b,c,d}_<model>_coda16.json`
- `outputs/eval_phase1-3{a,b,c,d}_<model>_coco80.json`
- `outputs/verify_yolo/{train,val}/*.jpg` (sanity 시각화)
- W&B runs: `terravision/yolo26_{n,s,m,l}` 4개

## 위험 및 대응 (ADR 와 동기화)

- **STAL 효과 미달성** (R3): 1-3a 미달 → imgsz 1024 sweep, 그래도 미달 → multi-scale + copy_paste 강화
- **Catastrophic forgetting** (R2): 5-epoch 단위 COCO80 회귀 측정, freeze 값 / lr0 조정
- **AGPL-3.0** (R1): 상용 배포 직전 재검토 — 학습/연구 범위에서는 비차단

## 사용자 검토 결과

(1-3a/b/c/d 각 sub 통과 시점에 채움)
