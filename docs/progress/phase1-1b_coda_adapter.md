# Phase 1-1b — CODa 어댑터

- **상태**: 🟢 승인완료 (2026-04-24)
- **시작일**: 2026-04-22
- **완료일**: 2026-04-22
- **담당 PR**: (작성 후 링크)
- **관련 ADR**: `docs/decisions/20260422_coda-primary-dataset.md`

## 목표

CODa (UT Campus Object Dataset) 를 Detection 파인튠의 primary 코퍼스로 쓰기 위한 어댑터를 완성한다. 범위:

- CODa 의 3D bbox 주석과 cam0 rectified 이미지를 짝지어 **COCO 포맷 2D detection dataset** 으로 변환.
- 3D → 2D 투영은 devkit (`helpers/geometry.py`) 과 같은 수학 (cv2.projectPoints + distortion) 으로 자체 구현. devkit 설치 없이도 동작하게 한다.
- CODa 원본 59 클래스 → 운영 택소노미 (~15–20 클래스) 매핑을 YAML 로 외부화.
- Occlusion · FoV · 최소 면적 필터를 기본값으로 적용.
- Phase 1-1 에서 정의한 `CocoDetectionDataset` 과 그대로 호환되는 JSON 산출.

Phase 1-1 의 BDD 컨버터는 유지하되 auxiliary 로만 쓴다. 본 단계에서 BDD 코드는 건드리지 않는다.

## 체크리스트

- [x] CODa 59 클래스 → 운영 택소노미 매핑 YAML 초안 작성 + **사용자 확인 완료** (2026-04-22)
- [x] `configs/dataset/coda.yaml` — Hydra 서브그룹 (경로 규칙, 필터 기본값, 택소노미 파일 참조)
- [x] `configs/dataset/coda_taxonomy.yaml` — 매핑 테이블 (CODa 원본명 → 운영 class_id + name)
- [x] `training/datasets/coda_to_coco.py`
  - [x] CODa calibration YAML 파서 (`calib_cam0_intrinsics.yaml`, `calib_os1_to_cam0.yaml`) → K, dist_coeffs, T_lidar_to_cam
  - [x] `get_3dbbox_corners(bbox_dict)` — (cX, cY, cZ, h, l, w, r, p, y) → 8 코너 (Rz·Ry·Rx 순)
  - [x] `project_corners_to_image(corners, calib)` → 2D 픽셀 좌표 + in-front 마스크
  - [x] `corners_to_xywh(image_points, in_front, image_size)` — clip + visibility 검사 후 xywh
  - [x] 필터: 최소 면적, 가시 코너 수, `isOccluded` 수용 집합
  - [x] split 파싱 — 시퀀스 metadata JSON 의 `ObjectTracking.{training,validation,testing}`
  - [x] CLI: `--coda-root`, `--split`, `--output`, `--taxonomy`, `--sequences`, `--cameras`, `--min-area`, `--min-visible-corners`, `--allow-occlusion`
- [x] `tests/test_coda_to_coco.py`
  - [x] 합성 calibration + 합성 3D bbox → 예상 2D xywh (회귀)
  - [x] Occlusion 필터 동작 확인 (`annotations_from_frame`, `convert_coda_split` 둘 다)
  - [x] FoV 밖 객체 제거 확인 (뒤쪽 + 좌/우 off-screen 두 케이스)
  - [x] 택소노미 매핑에서 drop 대상 클래스 skip 확인
  - [x] Empty split 처리
- [x] `python3 -m pytest -q` 전체 그린 (Phase 1-1 7건 + Phase 1-1b 11건 = 18 passed)

## 산출물

- `configs/dataset/coda.yaml`, `configs/dataset/coda_taxonomy.yaml`
- `training/datasets/coda_to_coco.py`
- `tests/test_coda_to_coco.py`
- (실데이터 없이도 단위 테스트만으로 검증되도록 설계)

## 검증 로그

- `python3 -m pytest -q` → `18 passed in 0.20s` (시스템 pytest 6.2.5, Python 3.10.12). ✅
  - `tests/test_bdd100k_to_coco.py` (Phase 1-1): 3 케이스
  - `tests/test_coco_loader.py` (Phase 1-1): 4 케이스
  - `tests/test_coda_to_coco.py` (Phase 1-1b): 11 케이스
    - `test_get_corners_at_origin_no_rotation` — (l, w, h) → 반길이 축 정렬 검증
    - `test_projection_identity_calibration_matches_pinhole` — box (0,0,5) · 단위 치수 → 예상 xywh ≈ (584.44, 304.44, 111.11, 111.11) 핀홀 식과 일치
    - `test_corners_to_xywh_drops_box_behind_camera` — Z_cam < 0 전 케이스 drop
    - `test_corners_to_xywh_drops_box_outside_fov` — X=15m 옆으로 이동, 전 코너 이미지 밖 → drop
    - `test_annotations_from_frame_applies_taxonomy_and_occlusion_filters` — taxonomy + occlusion 동시 필터 검증
    - `test_annotations_from_frame_raises_on_unknown_class` — 미등록 CODa 클래스 만나면 ValueError (silent drop 방지)
    - `test_load_taxonomy_real_yaml` — 출하된 `coda_taxonomy.yaml` 로드 + 16 클래스 + 매핑 일관성
    - `test_load_camera_calibration_roundtrip` — YAML → CameraCalibration 왕복
    - `test_convert_coda_split_empty_training` — empty split → images/annotations 모두 빈 COCO
    - `test_convert_coda_split_end_to_end` — 2 프레임 합성 CODa → file_name 규칙 · category id · 중복 ann_id 없음
    - `test_convert_coda_split_respects_allow_occlusion` — CLI 옵션이 드라이버로 전달되는 경로 검증
- `python3 -m training.datasets.coda_to_coco --help` → argparse usage 정상 출력. ✅
- 실데이터 검증은 Phase 1-2b 착수 시점에 devkit `vis_annos_rgb.py` 출력과 우리 컨버터 출력을 1개 시퀀스로 시각 비교 예정 (본 단계 scope 밖, 검토 요청 노트 참조).

## 검토 요청 노트

- **택소노미 매핑**: Phase 1-1b 의 핵심 의사결정. CODa 의 59 클래스 중 campus mobility 운행에 의미 있는 것만 남기고 나머지는 drop/merge. 초안은 별도 섹션(아래 "택소노미 초안 v1") 에서 제시.
- **3D→2D 투영 검증**: 실제 CODa 시퀀스 1건을 받아 devkit 의 `vis_annos_rgb.py` 출력과 우리 컨버터의 `debug_draw.py` 출력을 시각 비교하는 단계가 필요. 본 단계에서는 단위 테스트 까지만 마치고, 실데이터 검증은 Phase 1-2b 착수 시점에 같이 본다.
- **Occlusion 기본값**: `{None, Light, Medium}` 까지만 학습셋에 포함, `{Heavy, Full, Unknown}` 은 drop 제안. 데이터 볼륨 손실을 수치로 확인하고 조정.
- **FoV 필터**: 8 코너 중 **최소 2개가 이미지 내부** 인 객체만 통과를 제안. 너무 엄격하면 부분 가림 객체 손실, 너무 느슨하면 허위 박스.
- **이미지 경로 규칙**: 실데이터는 `data/raw/coda/` 에 풀어둘 예정. 컨버터는 CODa 원본 디렉토리 구조(`2d_rect/cam0/{SEQ}/*.jpg`, `3d_bbox/os1/*.json`, `calibrations/{SEQ}/`) 를 그대로 입력으로 받고, 출력 COCO JSON 의 `file_name` 은 `{SEQ}/{frame}.jpg` 같은 시퀀스 prefix 를 유지.

## 택소노미 v1 — CODa 59 → 운영 집합 (2026-04-22 확정)

의도: campus mobility robot 의 driving-path 관점에서 (a) 충돌 회피 대상, (b) 경로 유도 단서, (c) 빈도 있는 정적 장애물 만 남기고, 실내 디스펜서·가구·희귀 카테고리는 drop.

| 운영 id | 운영 class | CODa 원본 (id) | 드랍 / 병합 사유 |
|---|---|---|---|
| 1 | pedestrian | Pedestrian (1) | |
| 2 | bicycle | Bike (2) | |
| 3 | motorcycle | Motorcycle (3) | |
| 4 | scooter | Scooter (6), Segway (55), Skateboard (57) | 모두 sub-pedestrian 속도 wheeled-board |
| 5 | vehicle | Car (0), Truck (5), Bus (56), Pickup Truck (40), Delivery Truck (41), Service Vehicle (42), Utility Vehicle (43), Golf Cart (4) | 캠퍼스 로봇 관점에서 서브타입 구분 이득 낮음 |
| 6 | tree | Tree (7) | |
| 7 | pole | Pole (18) | |
| 8 | traffic_light | Traffic Light (10) | |
| 9 | sign | Traffic Sign (8), Informational Sign (19), Room Label (29), Wall Sign (49), Floor Sign (50) | 시각 특성 유사, Phase 1 단계에선 통합 |
| 10 | bollard | Bollard (12) | 캠퍼스 특화 핵심 |
| 11 | cone | Cone (23) | |
| 12 | barrier | Construction Barrier (13), Fence (21), Railing (22), Stanchion (30), Traffic Arm (48) | 선형 정적 장애물 그룹 |
| 13 | bike_rack | Bike Rack (11) | |
| 14 | bench | Bench (25) | |
| 15 | trash_can | Trash Can (27), Dumpster (53) | |
| 16 | fire_hydrant | Fire Hydrant (16) | |

### 드랍 (배경으로 처리, detection label 미부여)

| CODa 원본 | id | 드랍 사유 |
|---|---|---|
| Canopy | 9 | 이동체가 올려다보는 구조물, 지상 충돌 대상 아님 |
| Parking Kiosk | 14 | 희귀, barrier 로 섞으면 false-positive 유발 |
| Mailbox | 15 | 희귀 |
| Freestanding Plant | 17 | Phase 2 Traversability 로 처리 (지면 질감) |
| Door | 20 | 실내 중심, 구별 ROI 낮음 |
| Chair | 24 | 실내 가구 |
| Table | 26 | 실내 가구 |
| Newspaper Dispenser | 28 | 희귀 |
| Sanitizer Dispenser | 31 | 실내 |
| Condiment Dispenser | 32 | 실내 |
| Vending Machine | 33 | 실내 |
| Emergency Aid Kit | 34 | 실내 |
| Fire Extinguisher | 35 | 실내 |
| Computer | 36 | 실내 |
| Television | 37 | 실내 |
| Other | 38 | 의미 불명확 |
| Horse | 39 | 단일 시퀀스 노이즈 |
| Fire Alarm | 44 | 실내 |
| ATM | 45 | 희귀 |
| Cart | 46 | 동적인지 정적인지 컨텍스트 의존 (Phase 1-2c 에서 재검토 가능) |
| Couch | 47 | 실내 |
| Door Switch | 51 | 실내 |
| Emergency Phone | 52 | 희귀 |
| Vacuum Cleaner | 54 | 희귀, 단일 시퀀스 |
| Water Fountain | 58 | 희귀, 충돌 회피 관점에선 barrier 로 처리해도 무방 — 더 실데이터 빈도 보고 판단 |

### BDD100K → 운영 택소노미 보조 매핑 (auxiliary)

| BDD 클래스 | → 운영 id |
|---|---|
| pedestrian | 1 pedestrian |
| rider | 1 pedestrian (별도 rider 클래스 미채택 — 빈도 낮음) |
| bicycle | 2 bicycle |
| motorcycle | 3 motorcycle |
| car / truck / bus | 5 vehicle |
| train | drop (캠퍼스 해당 없음) |
| traffic light | 8 traffic_light |
| traffic sign | 9 sign |

## 사용자 검토 결과

- **2026-04-22 — 택소노미 v1 확정**: 제시한 16 클래스 매핑 그대로 승인. scooter 병합, vehicle 8종 통합, sign 5종 통합 모두 유지. Cart/Water Fountain drop 유지 (빈도 데이터 확보 후 Phase 1-2c 에서 재검토). 이 결정으로 `configs/dataset/coda_taxonomy.yaml` 이 YAML 형태로 외부화되어 컨버터·파인튠에서 공용.
- **2026-04-24 — 단계 승인 완료**: 18 pytest 녹색 · CLI 동작 · 택소노미 확정 상태로 사용자 승인. 실데이터 투입 전 컨버터 출력 시각 검증(Phase 1-2b 시작점)은 후속 작업으로 이월. Phase 1-2a (DINOv3 백본 래퍼) 진입.
