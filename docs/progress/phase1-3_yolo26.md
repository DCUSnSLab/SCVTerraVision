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

### 1-3a (nano) — 2차 시도 (imgsz=1024 fallback, ADR plan 분기)

- [ ] `python -m training.train_yolo +detection=yolo26_n +dataset=coda_yolo "detection.training.device='0,1,2,3'" detection.training.imgsz=1024 detection.training.batch=32 detection.training.workers=8 detection.training.exp_name=yolo26_n_phase1-3a_1024 logging.wandb.enabled=true`
  - 사유: AP_small=0.134 → 0.40 도달 위한 해상도 ↑. 동일 nano로 변수 분리.
  - batch는 1024 해상도 메모리 영향으로 64→32 축소.
- [ ] eval_yolo --mode coda16 → 새 비교 표
- [ ] 게이트 재판정. 통과 시 1-3b 착수, 미달 시 small + imgsz=1024 escalate (옵션 C).

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
