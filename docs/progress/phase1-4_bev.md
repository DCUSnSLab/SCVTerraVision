# Phase 1-4 — BEV Projection Baseline

- **상태**: ⏳ 다음 세션 착수 예정 (plan 승인 2026-04-29)
- **시작일**: —
- **완료일**: —
- **선행 단계**: [Phase 1-3](phase1-3_yolo26.md) 🟢 승인완료 (2026-04-29) — production ckpt `yolo26_s_phase1-3b_pseudo/weights/best.pt`
- **재배치 사유**: 사용자 결정 2026-04-29 — Tracking 보다 BEV 를 먼저 진행. 원래 PLAN.md 의 1-5=BEV → **1-4=BEV (본 phase)**, 1-5=Tracking, 1-6=Integration 으로 재번호.
- **관련 ADR**: [`docs/decisions/20260429_bev-projection-baseline.md`](../decisions/20260429_bev-projection-baseline.md) (다음 세션에서 작성)
- **Plan 파일** (외부): `~/.claude/plans/dinov2-dert-polished-sedgewick.md`

## 목표

YOLO26-s detection bbox 를 카메라 calibration 기반 IPM 으로 **base_link 좌표계의 ground (x_m, y_m)** 으로 변환한다. detection-only 베이스라인 (track_id, vx, vy 는 1-5 에서 합침). CoDA 3D LiDAR cuboid GT 로 정량 평가.

## 결정 사항 (plan 으로 확정 D1~D7)

| ID | 결정 |
|----|------|
| D1 | Ground frame = base_link, z=0 |
| D2 | 좌표 규약 = REP-103 (x=forward, y=left, z=up) |
| D3 | Distortion = `cv2.undistortPoints` point-level |
| D4 | Foot point = bbox 하단 중앙 (x+w/2, y+h) |
| D5 | 출력 = 미터 (x_m, y_m) in base_link |
| D6 | per-image calib lookup = COCO val JSON `sequence` 필드 → `calibrations/{seq}/*.yaml`. LRU cache |
| D7 | `ground_height_m` 파라미터 (기본 0.0). 1 시퀀스 calib z 로 sanity 후 ADR stamp |

## 다음 세션 실행 순서 (게이트)

| Step | 산출물 | 검증 |
|------|--------|------|
| S1 | ADR + PLAN.md/dashboard/README rename | 사용자 검토 (이번 세션에서 일부 완료) |
| S2 | preprocessing/bev/{calibration,foot_point,ipm,project}.py + 테스트 3종 | pytest green + 기존 회귀 없음 |
| S3 | inference/detect_and_project.py + configs/pipeline/coda_cam0_bev.yaml | seq 0 첫 10 프레임 smoke run |
| S4 | scripts/visualize_bev.py | 5장 샘플 시각화 — BEV 위치 육안 sanity |
| S5 | scripts/eval_bev.py — 전체 CoDA val | overall MAE/RMSE + 거리·클래스·시퀀스 bin + 산점도 |
| S6 | 본 progress 파일 결과 stamp | 사용자 승인 → 1-5 (Tracking) 진입 |

## 신규 파일 (다음 세션)

```
preprocessing/bev/{calibration,foot_point,ipm,project}.py
inference/detect_and_project.py
configs/pipeline/coda_cam0_bev.yaml
scripts/{eval_bev,visualize_bev}.py
tests/test_{ipm,calibration_set,foot_point}.py
docs/decisions/20260429_bev-projection-baseline.md
```

## 재사용 대상

- `training/datasets/coda_to_coco.py` — `load_camera_calibration`, `_yaml_matrix`, `_BBOX_TEMPLATE`, `get_3dbbox_corners`, `project_corners_to_image`
- `scripts/eval_yolo.py` — `_run_predictions_on_coco` 패턴, `_load_yolo_to_operational` (91→16 op-id)

## 사용자 검토 결과

(다음 세션 S6 통과 후 채움)
