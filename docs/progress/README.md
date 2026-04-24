# 진행 상황 대시보드

본 문서는 TerraVision 개발의 단계별 상태를 한눈에 보기 위한 인덱스이다. 세부 체크리스트·검증 로그·사용자 검토 결과는 각 `phase*.md` 에서 관리한다.

## 상태 범례

- ⏳ 진행중
- ✅ 완료·검토대기 (사용자 검토 대기 상태)
- 🟢 승인완료 (머지 및 다음 단계 진입 가능)
- ⛔ 블록
- ⚪ 예정 (아직 시작 전)

## 대시보드

| # | 단계 | 상태 | 파일 | 비고 |
|---|------|-----|------|------|
| 0 | 기반 인프라 | 🟢 승인완료 (2026-04-22) | [phase0.md](phase0.md) | Phase 0 PR 범위 |
| 1-1 | 데이터 로더 (COCO / BDD100K) | 🟢 승인완료 (2026-04-22) | [phase1-1_data_loader.md](phase1-1_data_loader.md) | COCO 단일 스키마 + BDD100K → auxiliary 로 downgrade |
| 1-1b | CODa 어댑터 (primary) | 🟢 승인완료 (2026-04-24) | [phase1-1b_coda_adapter.md](phase1-1b_coda_adapter.md) | 3D→2D 투영 + 16-class 택소노미 확정 · 18 pytest green · ADR: 20260422_coda-primary-dataset |
| 1-2a | DINOv3 백본 래퍼 | 🟢 승인완료 (2026-04-24) | [phase1-2a_backbone.md](phase1-2a_backbone.md) | HF AutoModel lazy-load + patch grid · 36 pytest green (3 gated skip) |
| 1-2b | DETR head 학습 | ✅ 1차 승인 (2026-04-24) · ⏳ 2차 (GPU 실학습 mAP) | [phase1-2b_detection.md](phase1-2b_detection.md) | 52 pytest green (8 gated skip) · HF DeformableDetr 어댑터 + CODa Hydra config + train/eval 루프 · 2차 smoke: `RUN_DINO_SMOKE=1 HF_TOKEN=... pytest` + `python -m training.train_detection` |
| 1-2c | 캠퍼스 데이터 파인튠 | ⚪ 예정 | — | CODa primary + 자체 촬영 |
| 1-3 | Tracking (ByteTrack) | ⚪ 예정 | — | boxmot 사용 |
| 1-4 | BEV projection | ⚪ 예정 | — | 1-4a 캘리브 → 1-4b IPM → 1-4c 제어 인터페이스 |
| 1-5 | 통합 · 최적화 | ⚪ 예정 | — | ≥15 FPS @ 1280×720 |
| 2 | Traversability Segmentation | ⚪ 예정 | — | 착수 시 재계획 |
| 3 | 멀티 카메라 · 온보드 | ⚪ 예정 | — | Fisheye + Jetson |

## 갱신 규칙

1. 단계가 `in progress → 완료·검토대기` 로 바뀌면 해당 행의 상태와 `파일` 링크를 즉시 갱신한다.
2. 사용자 검토에서 승인되면 상태를 🟢 로 바꾸고, 다음 단계 행의 상태를 ⏳ 로 전환 + 새 `phase*.md` 를 생성한다.
3. 중요 기술 결정은 `docs/decisions/` 에 ADR 을 추가하고, 여기 `비고` 칼럼에 파일명 1줄 참조만 남긴다.
