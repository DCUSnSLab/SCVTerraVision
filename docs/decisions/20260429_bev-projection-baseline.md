# ADR 2026-04-29 — Phase 1-4 BEV Projection Baseline

- **상태**: 확정 (사용자 승인 2026-04-29 plan 단계). 구현 stamp 는 1-4 완료 시 추가
- **일자**: 2026-04-29
- **관련 phase**: 1-4 (BEV projection baseline, 신설)
- **관련 ADR**: [`20260428_pivot-to-yolo26.md`](20260428_pivot-to-yolo26.md) (1-3 detection)
- **관련 파일** (다음 세션 신규):
  - `preprocessing/bev/{calibration,foot_point,ipm,project}.py`
  - `inference/detect_and_project.py`
  - `configs/pipeline/coda_cam0_bev.yaml`
  - `scripts/{eval_bev,visualize_bev}.py`
  - `tests/test_{ipm,calibration_set,foot_point}.py`

## 배경

Phase 1-3 detection 학습 단계 일단락. Production checkpoint = `outputs/checkpoints/yolo26_s_phase1-3b_pseudo/weights/best.pt` (5.6MB). 다음으로 BEV projection 베이스라인 구축.

원래 PLAN 은 1-4=Tracking → 1-5=BEV 였으나 사용자 결정으로 **1-4=BEV, 1-5=Tracking** 재배치. Tracking 없이 detection-only 로 ground 좌표 출력 후, 1-5 에서 track_id 합침.

## 결정 사항

### D1. Ground frame = base_link, z=0
LiDAR(os1) 가 아닌 base_link 의 ground 평면 사용. CoDA 의 `calib_os1_to_base.yaml` 의 `extrinsic_matrix` (4×4) 적용. 로봇 ground clearance 자동 보정 의도.

### D2. 좌표 규약 = REP-103
x=forward, y=left, z=up. 1 시퀀스 cam 위치 → base_link 변환 후 print 로 sanity check 한 결과를 본 ADR 에 stamp 예정.

### D3. Distortion = point-level `cv2.undistortPoints`
이미지 전체 undistort 안 함. CoDA `2d_rect/cam0/` 이 이미 rectified 일 가능성 → dist=[0,...] 시 no-op. 합성 plumb_bob 으로 round-trip 단위 테스트로 보장.

### D4. Foot point = bbox 하단 중앙 (x+w/2, y+h)
보행자 발끝, 차량 바닥 등 클래스별 contact 차이는 베이스라인 한도 내. 클래스별 MAE 메트릭으로 정량화.

### D5. 출력 단위·정렬 = 미터 (x_m, y_m) in base_link
forward = x, left = y. JSON schema 에서 `frame: "base_link"` 명시. 1-5 통합 시 `track_id, vx, vy` 추가.

### D6. per-image calibration lookup
CoDA val JSON 의 `sequence` 필드 → `calibrations/{seq}/{calib_cam0_intrinsics, calib_os1_to_cam0, calib_os1_to_base}.yaml`. `functools.lru_cache(maxsize=32)` 로 시퀀스별 1회 로드.

### D7. ground_height_m 파라미터 (기본 0.0)
base_link 가 robot 중심에 있으면 ground 가 z<0 일 수 있음. 1 시퀀스 calib z 성분으로 sanity 후 본 ADR 에 최종값 stamp.

## 평가 메트릭

CoDA 3D cuboid GT (os1 frame) → base_link 변환 → ground 투영 (mode: cuboid center xy 또는 bottom-face center xy 두 모드 병기).

매칭: predicted 2D bbox ↔ GT cuboid 8-corner projected 2D bbox, IoU≥0.5 greedy.

집계:
- MAE / RMSE (overall, per-axis)
- 거리 bin: [0,5), [5,15), [15,30), [30,∞)
- 클래스 bin: 91→16 op-id (기존 `_load_yolo_to_operational` 재사용)
- 시퀀스 bin (calib 변동 sanity)

## 위험 및 대응

| 위험 | 베이스라인 대응 |
|------|----------------|
| `2d_rect/cam0` rectified, dist=0 | `cv2.undistortPoints` 가 dist=0 시 no-op, 코드 변경 없음 |
| 좌표 규약 mismatch | 1 시퀀스 cam 위치 print sanity, 어긋나면 axis swap |
| Foot point ≠ 실제 contact | 베이스라인 수용, 클래스별 MAE 로 정량화 |
| Ground 비평탄 (경사·계단) | 평지 가정, 시퀀스별 RMSE 차이로 정량화 |
| GT cuboid center vs ground contact | 두 모드 (a center / b bottom-face) 병기, `gt_mode` 필드 |
| base_link z=0 가 ground 가 아닐 가능성 | `ground_height_m` 파라미터, 1 시퀀스 sanity 후 stamp |

## 비범위 (1-4 에서 안 함)

- Tracking (1-5): `track_id, vx, vy` schema 자리만 (null)
- ROS2 / TCP 직렬화 (1-6)
- Ground depth estimation / multi-plane (Phase 2 traversability 와 결합 시)
- Multi-camera (Phase 3)

## Final stamp (1-4 완료 시)

(아직 미작성. 다음 세션 S6 통과 후:
- 좌표 규약 sanity 결과
- ground_height_m 최종값
- overall MAE/RMSE
- 클래스별 / 거리별 표 요약
)
