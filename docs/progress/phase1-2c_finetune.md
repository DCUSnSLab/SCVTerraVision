# Phase 1-2c — 캠퍼스 데이터 파인튠

- **상태**: ⏳ 착수 대기 (열린 질문 해소 후 진행)
- **시작일**: —
- **완료일**: —
- **담당 PR**: (작성 후 링크)
- **선행 단계**: Phase 1-2b 🟢 승인완료 (2026-04-26, mAP=0.623, AP50=0.925 @ epoch_050)

## 목표

Phase 1-2b 의 CODa-only 베이스라인 (`outputs/checkpoints/dinov3_detr_base_full/epoch_050.pt`) 을 출발점으로, **자체 촬영 캠퍼스 데이터** 로 파인튠해 실제 운용 환경에서의 성능을 끌어올린다. CODa primary 원칙 (ADR 20260422) 은 유지 — 캠퍼스 데이터는 fine-tune supplement 로 합쳐진다.

부수적으로 1-2b 베이스라인에서 드러난 **AP_small=0.35 vs AP_large=0.74** 의 ~2× 격차 대응안을 본 단계 또는 별도 단계에서 다룰지 결정한다 (열린 질문 §3).

## 1-2b 결과 요약 (재진입 시 참고)

| 지표 | epoch_050 | 비고 |
|------|-----------|------|
| mAP@[.50:.95] | 0.623 | DDP GPU0-2, 21h |
| AP50 | 0.925 | |
| AP75 | 0.700 | |
| AP_small | 0.35 | **하위** — 본 단계에서 대응 검토 |
| AP_medium | ~0.55 | |
| AP_large | 0.74 | |

per-class 약점: motorcycle 0.518 / cone 0.505 / traffic_light 0.470 / fire_hydrant 0.455 (소형 + 데이터 희소). 캠퍼스 데이터에서는 보행자·자전거·볼라드·벤치 비중이 높을 것으로 예상 — 캠퍼스 분포가 CODa 약점 클래스를 보강해 줄지 데이터 수집 후 통계 확인.

## 열린 질문 (착수 전 해소 필요)

각 질문의 결론은 본 문서 또는 신규 ADR 에 기록한 뒤 체크리스트가 확정된다.

1. **데이터 수집 일정·플랫폼·센서 사양** — PLAN 의 열린 질문 #3 본격화 시점.
   - 카메라 = CODa 와 동일 intrinsic? 아니면 캠퍼스 로봇 탑재용 별도 캘리브?
   - 시간대·날씨·실내/실외 분포 목표
   - 한 번에 수집할 분량 (frames / sequences)
2. **라벨링 도구·체계** — PLAN 은 CVAT 명시. 16-class 운영 택소노미 (CODa 와 동일) 유지 vs 캠퍼스 특화 클래스 (볼라드·벤치·자전거 거치대 등은 이미 포함됨, 추가 후보는?) 도입.
3. **AP_small 격차 대응 시점** — 본 단계에 포함 vs 별도 단계 (예: 1-2d) 분리.
   - 후보 A: 학습 해상도 1024 → 1280/1536 상향 (single-scale 유지, 가장 단순)
   - 후보 B: multi-scale pyramid (ViT stride-16 위 FPN 또는 ViTDet 스타일 simple feature pyramid) — 아키텍처 변경 → 별도 ADR 필요
   - 후보 C: small-object augmentation (mosaic, copy-paste) — 학습 루프 변경
   - 후보 D: 본 단계에서는 데이터만 다루고 구조 변경은 1-2d 로 분리
4. **fine-tune 절차**
   - 출발점: 1-2b 의 `epoch_050.pt` 모델 state 만 재사용 (optimizer state 폐기, scheduler 새로 시작)
   - LR: 베이스 대비 낮춤 (예: head 5e-5 / backbone 5e-6) — 사전학습 대비 1/4 가 일반적
   - freeze 정책: backbone 2 epoch freeze 재적용 vs 처음부터 unfreeze (이미 CODa 로 적응돼 있음)
   - 데이터 mix: campus-only fine-tune vs CODa+campus 혼합 (catastrophic forgetting 방지). 권장 시작점: 혼합 + sample weighting
5. **평가 분리** — campus validation set 정의 (예: 별도 시퀀스 hold-out) + CODa validation 도 동시에 평가해 회귀 감지

## 체크리스트 (잠정 — 열린 질문 §1~5 해소 후 확정)

### 데이터 (열린 질문 §1·§2 해소 후)

- [ ] 캠퍼스 데이터 수집 계획서 (시퀀스 수 · 시간대 · 분포) → 본 문서에 append
- [ ] 카메라 캘리브레이션 (Phase 1-4a 와 공유 가능 — 사전 수행 시 본 단계 단축)
- [ ] CVAT 인스턴스 셋업 + 라벨러 가이드 (16-class 정의 + edge case 룰)
- [ ] 라벨링 + 1차 QA
- [ ] CVAT export → COCO 변환 스크립트 (`training/datasets/cvat_to_coco.py` 또는 기존 변환기 확장)
- [ ] 데이터셋 통계: 클래스 분포·박스 크기 분포·시간대 분포 → CODa 와 비교 리포트

### 학습 (열린 질문 §3·§4 해소 후)

- [ ] 새 Hydra config: `configs/dataset/campus.yaml` + `configs/detection/dinov3_detr_finetune.yaml` (LR · freeze · checkpoint resume 명시)
- [ ] `training/train_detection.py` 에 ckpt resume 옵션 추가 (현재는 처음부터 학습만 가정 — model_state 만 로드, optimizer/scheduler 는 새로 빌드하는 경로 필요)
- [ ] dry-run (subset, 3-5 epoch) — 캠퍼스 데이터 forward 통과 + loss 추이 확인
- [ ] full fine-tune 실행 (epoch 수는 dry-run 후 확정, 예상 10-30 epoch)
- [ ] (열린 질문 §3 결정에 따라) 해상도 상향 또는 multi-scale 도입 분기

### 평가 (열린 질문 §5 해소 후)

- [ ] campus validation 평가 → mAP / AP50 / per-class
- [ ] CODa validation 회귀 평가 (1-2b 베이스라인 대비 성능 변화 기록)
- [ ] 시각화: bbox overlay 샘플 (양호 / 실패 케이스 각 N장)
- [ ] W&B run 링크 본 문서에 기록

## 산출물 (예상)

- `data/raw/campus/` (수집 원본 — .gitignore)
- `data/annotations/campus_{training,validation}_coco.json`
- `configs/dataset/campus.yaml`, `configs/detection/dinov3_detr_finetune.yaml`
- `training/datasets/cvat_to_coco.py` (또는 기존 변환기 확장)
- `tests/test_cvat_to_coco.py`
- `outputs/checkpoints/dinov3_detr_finetune/epoch_*.pt`
- `outputs/eval_phase1-2c_*.json`

## 검토 요청 노트 (착수 시 채움)

- 열린 질문 §1~5 의 결정 결과
- multi-scale 도입 시 ADR (`docs/decisions/YYYYMMDD_multiscale-feature.md`) 추가 필요 여부
- fine-tune 데이터 mix 비율 (campus-only vs CODa+campus) 결정 근거
- 회귀 평가에서 CODa mAP 가 어느 정도 떨어지면 수용/거부할지 임계값

## 사용자 검토 결과

(2차 머지 후 본 단계 착수 시 채움)
