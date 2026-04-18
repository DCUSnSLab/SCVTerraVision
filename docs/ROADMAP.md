# Camera Perception 시스템 - 전체 로드맵

> 캠퍼스/험지 자율주행 이동체용 카메라 기반 perception 시스템.
> 본 문서는 Phase 1~5 및 cross-cutting 주제(ROS2, calibration, safety, 데이터 수집)를 다룬다.

---

## 1. 프로젝트 목표

자율주행 이동체가 **맵에 없는 실시간 지형**에서도 안전하게 주행하기 위해, 카메라 입력만으로:

1. **Traversability**: 픽셀 단위 "갈 수 있는 영역" 판단
2. **Object detection**: 사람·차량·지형지물 등 주변 객체 인식
3. **BEV 출력**: 다중 카메라를 융합한 bird's-eye view → 행동 예측·경로 제어의 입력

**개발 환경**: 워크스테이션 GPU (학습/실험)
**배포 환경**: NVIDIA Jetson (Orin 가정)
**핵심 접근**: DINOv2 foundation model 기반 + (a) supervised seg head, (b) WVN-style self-supervised 두 방식 비교

---

## 2. 시스템 아키텍처 (최종 형태)

```
            ┌─────────────────────────────────────────────────┐
            │           6× Cameras (front2 + side2 + rear1+1) │
            └────────────────────┬────────────────────────────┘
                                 │ ROS2 image_raw topics
                                 ▼
                    ┌────────────────────────────┐
                    │   Per-camera Inference     │  DINOv2 + heads
                    │   (TensorRT FP16 on Jetson)│  - traversability seg
                    └────────────────────┬───────┘  - object detection
                                         │
                    ┌────────────────────▼───────┐
                    │   Multi-cam Fusion / BEV   │  IPM → LSS
                    │   (uncertainty-aware)      │
                    └────────────────────┬───────┘
                                         │ /perception/bev_grid
                                         │ /perception/objects
                                         ▼
                    ┌────────────────────────────┐
                    │  Behavior Planner / MPC    │  (downstream)
                    └────────────────────────────┘
```

---

## 3. Phase 로드맵 요약

| Phase | 목표 | 상태 | 핵심 산출물 |
|-------|------|------|-------------|
| **1** | 데이터 파이프라인 | ✅ 완료 | RUGD/RELLIS-3D 로더, 통합 taxonomy(6-class), augmentation, viz |
| **2** | 단일 카메라 PoC (모델) | 진행 예정 | DINOv2 backbone + seg head, WVN-style 비교, 학습/평가 루프 |
| **3** | Object detection 통합 | | OWL-ViT / YOLO-World 또는 DINOv2 detection head |
| **4** | 멀티 카메라 + BEV | | 6-cam calibration, IPM 베이스라인, LSS 확장 |
| **5** | Edge 배포 (Jetson) | | ONNX/TensorRT export, FP16/INT8, ROS2 노드 패키징 |

---

## 4. Phase 상세

### Phase 1 — 데이터 파이프라인 ✅

**완료된 항목** (구현 상세는 `/Users/soobinjeon/.claude/plans/buzzing-splashing-karp.md` 참조)

- 통합 taxonomy v1 (6-class + ignore): traversable_smooth, traversable_grass, non_traversable_terrain, obstacle_static, obstacle_dynamic, sky
- RUGD (RGB 라벨), RELLIS-3D (id 라벨) 로더
- Albumentations 기반 transforms (DINOv2 14배수 정렬, ImageNet norm)
- Visualization (triptych, class histogram)
- 검증/시각화 스크립트
- 15개 단위 테스트 통과

---

### Phase 2 — 단일 카메라 PoC (모델 학습)

**목표**: 1대의 카메라로 unified taxonomy의 6-class semantic segmentation을 수행하는 모델을, **두 가지 접근**으로 구현해 비교.

#### 2.1 접근 A — Supervised: DINOv2 + linear/DPT head

- **Backbone**: DINOv2 ViT-S/14 또는 ViT-B/14 (frozen 우선, 후에 fine-tune 비교)
  - HuggingFace `facebook/dinov2-small` 또는 PyTorch Hub `dinov2_vits14`
- **Head 옵션**:
  - Linear probe (간단, 빠름) — feature → 1×1 conv → upsample
  - DPT-style decoder (Dense Prediction Transformer) — 다중 layer feature 융합
- **Loss**: weighted cross-entropy (class imbalance 완화) + 옵션으로 Dice loss
- **Metric**: per-class IoU, mIoU, traversability binary IoU

#### 2.2 접근 B — Self-supervised: WVN(Wild Visual Navigation) 스타일

- **아이디어**: 차량이 실제로 지나간 경로(궤적)를 카메라 영상에 투영 → positive 샘플로 활용
- **구성**:
  - Foot-print supervision: pose + camera extrinsic으로 차량 발자국을 이미지 좌표로 변환
  - Confidence MLP를 DINOv2 feature 위에 학습 (양성 샘플만 있어도 학습 가능한 anomaly-style loss)
- **데이터**: 자체 수집 데이터 필요 (Phase 4-5에서 본격화) — 초기엔 RELLIS-3D의 pose+image로 시뮬레이션
- **참고**: Frey et al. "Fast Traversability Estimation for Wild Visual Navigation" (2023)

#### 2.3 비교 실험 설계

| 항목 | 접근 A | 접근 B |
|------|--------|--------|
| 데이터 | RUGD + RELLIS-3D (라벨 있음) | 자체 데이터 (라벨 불필요) |
| 출력 | 6-class semantic | binary traversability + confidence |
| 비교 metric | mIoU, FPS, 라벨 비용 | 자체 데이터에서 위양성률, 미지 지형 일반화 |

#### 2.4 추가될 디렉토리/파일

```
src/camera_perception/
├── models/
│   ├── backbones/dinov2.py          # HF or torch.hub wrapper, freeze 옵션
│   ├── heads/{linear,dpt,wvn}.py
│   └── losses.py                    # weighted CE, Dice, WVN confidence loss
├── training/
│   ├── trainer.py                   # PyTorch Lightning 또는 minimal loop
│   ├── metrics.py                   # mIoU, confusion matrix
│   └── callbacks.py                 # viz logging (wandb/tensorboard)
├── inference/
│   └── predictor.py                 # 단일 이미지/배치 추론, 결과 후처리
configs/
├── model/{dinov2_linear,dinov2_dpt,wvn}.yaml
└── train/default.yaml               # optimizer, schedule, batch size 등
scripts/
├── train.py
└── eval.py
```

#### 2.5 실험 추적

- 기본 logger는 **TensorBoard** (`configs/train/default.yaml`의 `logger.type`).
  wandb/csv도 지원 — 필요시 override.
- 매 epoch마다 validation 샘플 N개의 `image | gt | pred` 타일을 자동 업로드:
  `PredictionVizCallback` (`src/camera_perception/training/callbacks.py`).
  비활성화는 `viz.enabled=false`. 실행/포트 포워딩 절차는
  `docs/PHASE2_EXECUTION.md` §3 참조.

#### 2.6 Baseline 측정 프로토콜

**왜 필요한가**: RUGD/RELLIS-3D + DINOv2 + 통합 6-class 세팅을 직접 평가한
공개 논문이 없음. 따라서 "기대 성능"을 외부 수치로 가정하지 않고 **본 프로젝트의
첫 번째 실험을 baseline 측정 자체로 정의**한다.

**참조 가능한 인접 수치 (직접 비교는 불가, 감 잡기용)**

| 출처 | 세팅 | 수치 |
|------|------|------|
| RELLIS-3D 원 논문 (Jiang et al., 2021) | RELLIS image, 20-class fine-grained | HRNetV2+OCR mIoU 52.92% |
| OFFSEG (Viswanath et al., 2021) | RUGD/RELLIS reduced-class | reduced-class에서 mIoU 향상 보고 (정확 수치는 논문 참조) |
| DINOv2 원 논문 (Oquab et al., 2023) | ADE20K (도메인 다름) | ViT-L linear+multiscale mIoU 60.2% |
| TARTS (2025) | ORFD/RTSD binary traversability | IoU 94.1% / 94.5% (binary, training-free) |

위 수치는 **목표가 아닌 참조 환경**. 우리 세팅(통합 6-class, RUGD+RELLIS 합본)의
실제 기대치는 아래 측정 결과로 확정한다.

**측정 단계 (Phase 2 진입 직후 가장 먼저 실행)**

1. **B0 — 무학습 sanity check**: DINOv2 frozen + 무학습 linear head (random init).
   목적: 파이프라인 동작 확인, "최악의 경우" 하한 확보
2. **B1 — Linear probe**: DINOv2 frozen + linear head를 RUGD train으로 학습.
   가장 단순한 학습형 baseline. 측정: 각 데이터셋 val의 per-class IoU + mIoU
3. **B2 — DPT head**: DINOv2 frozen + DPT decoder. B1 대비 향상 폭이
   head 복잡도 추가의 가치를 정당화하는지 판단
4. **B3 — Backbone fine-tune (선택)**: 시간이 허락하면 ViT-S/14 마지막 N개
   block만 unfreeze해서 향상 측정

**측정 항목 (모든 baseline 공통)**

- Per-class IoU + mIoU (RUGD val, RELLIS val, 합본 val 각각)
- Confusion matrix (특히 grass↔non_traversable, smooth↔non_traversable 혼동률)
- 추론 latency (워크스테이션 GPU, 518×518)
- 정성적 실패 케이스 50장 추출 (그림자, 물, 키 큰 풀 등 모호한 영역)

**산출물**: `reports/phase2_baseline.md`에 위 표/수치/실패 케이스 정리.
이 문서가 확정되면 아래 2.7의 완료 기준 mIoU 수치를 측정값 기반으로 갱신한다.

#### 2.7 완료 기준

- 2.6의 baseline 측정(B0~B2) 완료 및 `reports/phase2_baseline.md` 작성
- B1 또는 B2 중 더 높은 mIoU를 baseline으로 채택, 접근 A/B 비교 보고서가
  이 baseline을 기준으로 작성됨 (`reports/phase2_comparison.md`)
- 워크스테이션에서 추론 ≥ 30 FPS @ 518×518 (RTX 기준)

> 참고: 초기 문서에 있던 "RUGD val mIoU ≥ 0.55"는 외부 근거 없는 추정치였으므로
> 제거. 측정 후 실제 수치를 기반으로 목표를 재설정한다.

---

### Phase 3 — Object Detection 통합

**목표**: 사람/차량/자전거 등 동적 객체의 **bounding box + class** 검출을 traversability와 함께 제공.

#### 3.1 접근 옵션

| 옵션 | 장점 | 단점 |
|------|------|------|
| **OWL-ViT v2 / OWLv2** | open-vocabulary, 라벨 없는 클래스도 검출 | 무거움, edge에 부적합 |
| **YOLO-World** | open-vocabulary + 빠름 | 학습 복잡도 |
| **YOLOv9/v10** + custom 학습 | 가볍고 빠름, 잘 검증됨 | closed-set, 데이터 필요 |
| **DINOv2 + DETR head** | backbone 공유 (효율) | 구현 복잡 |

**권장 기본**: YOLOv10-s (closed-set, COCO + 자체 험지 데이터로 fine-tune) + 향후 OWLv2를 long-tail 클래스 보완용으로.

**Backbone 공유 여부**: 초기엔 별도 모델로 단순하게 분리 운영. backbone 공유는 Phase 5에서 latency 최적화 시 검토.

#### 3.2 출력 통합

```python
PerceptionOutput {
    seg_mask: int8[H, W]              # unified 6-class
    seg_uncertainty: float[H, W]      # 옵션
    detections: list[Detection]       # bbox + class + score + track_id
    timestamp: float
    frame_id: str
}
```

#### 3.3 추가될 디렉토리

```
src/camera_perception/
├── detection/
│   ├── models/{yolo,owlv2}.py
│   ├── postprocess.py               # NMS, score threshold
│   └── tracker.py                   # ByteTrack 또는 SORT (Phase 4와 통합)
configs/detection/
└── yolov10_offroad.yaml
```

#### 3.4 완료 기준

- COCO person/vehicle subset에서 mAP ≥ 0.4
- 통합 PerceptionOutput 데이터 클래스 정의 + 테스트
- 단일 카메라에서 seg + det 동시 추론 데모 영상

---

### Phase 4 — 멀티 카메라 + BEV

**목표**: 6대 카메라의 입력을 융합해 차량 좌표계의 BEV 그리드(예: 30m × 30m, 0.1m/cell)를 생성.

#### 4.1 카메라 배치 가정

| ID | 위치 | 시야각 | 용도 |
|----|------|--------|------|
| `cam_front_left` | 전방 좌 | wide | 주행 영역 + 객체 |
| `cam_front_right` | 전방 우 | wide | 주행 영역 + 객체 |
| `cam_side_left` | 좌측면 | wide | 사각 보완 |
| `cam_side_right` | 우측면 | wide | 사각 보완 |
| `cam_rear` | 후방 | wide | 후진/추월 감지 |
| `cam_aux` | 추가 (예: 상부 광각) | ultrawide | 전체 커버리지 |

#### 4.2 Calibration

- **Intrinsic**: ChArUco 보드로 OpenCV `calibrateCamera`. fish-eye 렌즈인 경우 `cv2.fisheye.calibrate`. 결과는 `configs/cameras/<cam_id>.yaml`에 저장.
- **Extrinsic** (cam → base_link): hand-eye calibration 또는 ChArUco 보드를 차량 좌표계 기준 위치에 두고 PnP. 검증은 lidar↔camera 또는 known landmark 재투영 오차.
- **Synchronization**: 하드웨어 트리거가 가능하면 사용, 아니면 ROS2 `message_filters.ApproximateTimeSynchronizer`. 초기 단계에선 software sync로 시작.

`configs/cameras/<cam_id>.yaml` 예시:
```yaml
camera_id: cam_front_left
model: pinhole              # or fisheye
image_size: [1280, 720]
K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
distortion: [k1, k2, p1, p2, k3]
T_base_cam: [[r11..], ..., [0, 0, 0, 1]]   # 4×4 SE(3)
```

#### 4.3 BEV 생성: 두 단계로 진입

**4.3.1 베이스라인 — IPM (Inverse Perspective Mapping)**
- 평지 가정으로 ground plane에 perspective transform
- 빠르고 단순, calibration 정확도 검증에도 유용
- 한계: 비평지·키 큰 객체에서 왜곡

**4.3.2 발전형 — LSS (Lift-Splat-Shoot) 계열**
- 픽셀별 depth 분포 예측 → 3D frustum으로 lift → BEV 그리드에 splat
- 다중 카메라 자연스럽게 융합
- 참고: Philion & Fidler "Lift, Splat, Shoot" (2020), BEVFormer/BEVDet 후속

**4.4 BEV 출력 포맷**
```
bev_grid: float32[C, H_bev, W_bev]
  channels:
    0: traversability_score (0~1)
    1~6: class probabilities (unified 6-class projected)
    7: object occupancy (detections projected)
    8: confidence / uncertainty
```

#### 4.5 추가될 디렉토리

```
src/camera_perception/
├── calibration/
│   ├── intrinsic.py             # ChArUco → K, distortion
│   ├── extrinsic.py             # cam → base_link
│   └── io.py                    # yaml load/save
├── bev/
│   ├── ipm.py                   # baseline
│   ├── lss/                     # Lift-Splat-Shoot 모듈
│   └── grid.py                  # BEV grid 정의, projection 유틸
└── fusion/
    ├── multi_cam.py             # 6-cam 시간 동기화 + projection
    └── temporal.py              # 짧은 history aggregation (옵션)
configs/bev/
├── grid_30m.yaml
└── lss_baseline.yaml
```

#### 4.6 완료 기준

- 6대 calibration yaml 파일 + 재투영 오차 < 2 px (intrinsic), 0.1 m (extrinsic)
- IPM 베이스라인이 정적인 BEV 영상에서 차선/도로 경계가 시각적으로 일치
- LSS 베이스라인이 RUGD/자체 데이터에서 traversability를 BEV에 정확히 투영
- 단일 frame 처리 시간 < 100 ms (워크스테이션)

---

### Phase 5 — Jetson 배포 (Edge optimization)

**목표**: 학습된 모델을 Jetson Orin에서 실시간(>15 FPS) 추론 가능하게 패키징.

#### 5.1 변환 파이프라인

```
PyTorch (.pt) → ONNX (.onnx) → TensorRT (.engine)
                                  ├─ FP16 (1차 목표)
                                  └─ INT8 (calibration dataset 필요)
```

- DINOv2는 ONNX export 시 attention shape 처리 주의 (dynamic shape 권장)
- LSS의 grid sampling 등 일부 op는 TensorRT plugin 필요할 수 있음

#### 5.2 ROS2 노드 패키징

- ament_python 패키지로 구성
- 노드 구조:
  - `perception_inference_node` (per-camera 또는 batched)
  - `bev_fusion_node`
  - `perception_diagnostics_node` (FPS, GPU usage, 모델 헬스)

#### 5.3 추가될 디렉토리

```
src/camera_perception/
└── export/
    ├── to_onnx.py
    ├── to_tensorrt.py
    └── int8_calibration.py
ros2_ws/                            # 별도 ROS2 workspace
└── src/camera_perception_ros/
    ├── package.xml
    ├── setup.py
    ├── camera_perception_ros/
    │   ├── inference_node.py
    │   ├── bev_node.py
    │   └── msgs/                   # custom msg가 필요하면
    └── launch/
        └── perception.launch.py
```

#### 5.4 완료 기준

- Jetson Orin AGX에서 6-cam BEV ≥ 15 FPS (FP16 기준)
- ROS2 launch 한 번으로 전체 perception 파이프라인 기동
- bag 파일 재생 → BEV 출력 검증 가능

---

## 5. Cross-cutting 주제

### 5.1 ROS2 통합

**도입 시점**: Phase 5에서 본격 패키징하지만, **메시지 인터페이스는 Phase 3 종료 시점에 동결**해야 downstream(planner) 팀이 병렬 작업 가능.

**입력 토픽** (perception이 subscribe)
| 토픽 | 메시지 타입 | QoS | 용도 |
|------|-------------|-----|------|
| `/cam_front_left/image_raw` | `sensor_msgs/Image` | best_effort, depth=1 | 카메라 raw 이미지 |
| `/cam_front_left/camera_info` | `sensor_msgs/CameraInfo` | reliable | intrinsic |
| (각 카메라마다 위와 동일) | | | |
| `/tf` | `tf2_msgs/TFMessage` | | 차량 pose, cam extrinsic |

**출력 토픽** (perception이 publish)
| 토픽 | 메시지 타입 | 주기 | 용도 |
|------|-------------|------|------|
| `/perception/seg/cam_front_left` | `sensor_msgs/Image` (mono8, class id) | 카메라 frame rate | per-cam seg |
| `/perception/detections` | `vision_msgs/Detection2DArray` | 카메라 frame rate | object detection |
| `/perception/bev_grid` | `nav_msgs/OccupancyGrid` (또는 custom multi-channel) | 10 Hz | BEV traversability |
| `/perception/bev_objects` | `vision_msgs/Detection3DArray` | 10 Hz | BEV 객체 |
| `/perception/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 1 Hz | 헬스체크 |

**Custom msg 검토**
- `OccupancyGrid`는 단일 채널 — multi-channel BEV (traversability + class prob + uncertainty)가 필요하면 custom msg 정의 (`PerceptionBEVGrid.msg`).
- `Detection2DArray`로 충분한지 (track_id, 3D bbox 등 필요?)

**구현 힌트**:
- ROS2 humble 또는 jazzy 기준 (Jetson L4T 호환 확인)
- `cv_bridge`로 sensor_msgs/Image ↔ numpy
- 토픽 인터페이스 정의는 `ros2_ws/src/camera_perception_msgs/msg/*.msg`에 별도 패키지로

---

### 5.2 Calibration

#### 5.2.1 Intrinsic
- **도구**: OpenCV `cv2.calibrateCamera` (pinhole) 또는 `cv2.fisheye.calibrate`
- **타겟**: ChArUco 보드 (5×7, 30 mm 정도). 30~50장 다양한 각도 촬영
- **저장 포맷**: `configs/cameras/<cam_id>.yaml` (위 4.2 예시)
- **재투영 오차 목표**: < 0.5 px (좋음), < 1.0 px (수용)

#### 5.2.2 Extrinsic (cam → base_link)
- **방법**: 차량 좌표계 기준 알려진 위치에 ChArUco 보드 설치 → PnP로 cam pose 추정
- **검증**: 알려진 3D 점(예: 차량 휠 중심)이 영상에 정확히 투영되는지

#### 5.2.3 멀티캠 일관성
- 인접 카메라 간 overlap 영역에서 동일 랜드마크의 BEV 좌표 일치 확인
- 불일치 시 extrinsic 재보정

#### 5.2.4 자동화 도구
- `scripts/calibrate_intrinsic.py` (ChArUco 이미지 → yaml)
- `scripts/calibrate_extrinsic.py` (보드 사진 → cam→base_link yaml)
- `scripts/verify_calibration.py` (재투영 오차 계산, 시각적 overlay)

---

### 5.3 안전·페일세이프

#### 5.3.1 불확실성 추정

| 종류 | 방법 | 비용 |
|------|------|------|
| Aleatoric (data noise) | Softmax entropy, predictive variance | 무료 |
| Epistemic (model uncertainty) | MC Dropout, Deep Ensembles, Evidential Deep Learning | 중~고 |
| OOD (out-of-distribution) | DINOv2 feature distance, ODIN, Mahalanobis | 저~중 |

**권장 시작점**: per-pixel softmax entropy + DINOv2 feature 기반 OOD score. 비싸지 않고 정성적으로 의미 있는 신호.

#### 5.3.2 Fail-safe 정책

| 트리거 | 행동 |
|--------|------|
| 카메라 토픽 timeout (>500 ms) | `/perception/diagnostics`에 ERROR, planner는 안전 정지 |
| 모델 추론 실패 / NaN | 해당 카메라 출력을 ignore로 마킹, 나머지로 BEV 생성 |
| 평균 BEV uncertainty가 임계 초과 | planner에 "low confidence" 플래그 전달 → 속도 제한 |
| OOD 영역 비율 > 30% | 동일하게 low confidence + 정지 후보 |

#### 5.3.3 모니터링

- `diagnostic_msgs/DiagnosticArray`로 publish: per-camera FPS, GPU 메모리, 추론 latency, OOD 비율
- 로그: rosbag + wandb (학습 시)

---

### 5.4 데이터 수집·라벨링 (자체 캠퍼스 데이터)

**도입 시점**: Phase 2 후반 ~ Phase 3 시작. 공개 데이터로 baseline을 잡은 뒤 도메인 적응을 위해.

#### 5.4.1 수집 파이프라인

- **하드웨어**: 차량에 모든 카메라 + IMU + (가능하면) GPS·LiDAR 장착
- **수집 방식**: ROS2 `ros2 bag record` 사용. 토픽: 모든 image_raw + camera_info + tf + odom
- **수집 시나리오**:
  - 다양한 시간대/날씨 (맑음, 흐림, 비, 야간)
  - 캠퍼스 주요 경로 (포장도로, 산책로, 잔디밭, 계단 옆 등)
  - 험지(가능한 범위에서): 진흙, 자갈, 비탈, 키 큰 풀
  - 동적 객체: 보행자, 자전거, 차량
- **메타데이터**: 각 bag에 `metadata.yaml` 첨부 (날씨, 시간, 위치, 카메라 캘리브 버전)

#### 5.4.2 라벨링 전략

| 단계 | 방법 | 비용 | 품질 |
|------|------|------|------|
| 1. 자가 라벨링 (auto-label) | 학습된 RUGD 모델로 pseudo-label → 사람이 수정 | 저 | 중 |
| 2. SAM / SAM2 + 클릭 보조 | 사람이 영역 클릭 → SAM이 마스크 생성 → unified class 할당 | 중 | 고 |
| 3. WVN-style 자가지도 | 차량 주행 trajectory를 positive로 사용 (라벨 불필요) | 저 (수집 자동) | 특정 task만 |

**권장 진행 순서**:
1. 초기 batch (~500장) → SAM2 + 사람 (Phase 2 후반)
2. 학습 → pseudo-label → 사람 수정 (active learning loop)
3. 병행하여 WVN-style 데이터(주행 영상 + pose)는 자동 축적

#### 5.4.3 라벨링 도구 후보

- **CVAT** (open-source, SAM 통합 있음) — 권장
- **Label Studio** (유연하지만 segmentation 워크플로우는 CVAT가 더 매끄러움)
- **Roboflow** (편하지만 클라우드 의존)

#### 5.4.4 데이터 관리

- **저장**: 외장 SSD/NAS. raw bag은 압축 후 보관. 추출 이미지는 `data/campus_v1/`에 RUGD 형식으로 정렬
- **버전**: `data/campus_v1`, `data/campus_v2` 식으로 스냅샷
- **DVC** (Data Version Control) 도입 검토 (Phase 3에서)

---

## 6. 기술 스택

| 영역 | 라이브러리 | 비고 |
|------|------------|------|
| ML 프레임워크 | PyTorch ≥ 2.2 | DINOv2 native |
| Foundation model | DINOv2 (HF / torch.hub) | ViT-S/14, ViT-B/14 |
| Augmentation | Albumentations | mask 동기 변환 |
| Detection | Ultralytics YOLO, transformers (OWLv2) | 옵션 |
| 학습 루프 | PyTorch Lightning 또는 자체 trainer | Phase 2에서 결정 |
| 실험 추적 | wandb 또는 tensorboard | 둘 중 결정 |
| Calibration | OpenCV | ChArUco |
| BEV (LSS) | 자체 구현 + 참조 BEVFormer/BEVDet | |
| Edge | TensorRT, ONNX Runtime | Jetson L4T 버전 호환 확인 |
| ROS | ROS2 humble/jazzy | Jetson L4T 매트릭스 확인 |
| 데이터 라벨링 | CVAT (SAM2 통합) | self-host 가능 |
| Config 관리 | OmegaConf / Hydra | |

---

## 7. 주요 위험·미결 사항

| 항목 | 영향 | 완화 |
|------|------|------|
| Jetson에서 DINOv2 ViT-B 추론 속도 | 실시간성 미달 가능 | ViT-S/14 우선, distillation 검토 |
| WVN-style은 자체 데이터 필수 | Phase 2 비교 실험 지연 | RELLIS-3D pose+image로 시뮬레이션 |
| 멀티캠 동기화 (HW trigger 없음 경우) | BEV 시간 정렬 오차 | software sync + 보간, 필요시 HW 도입 |
| 험지에서 calibration drift | extrinsic 정확도 저하 | 정기 재보정, online calibration 검토 |
| 라벨링 비용 | 진척 둔화 | SAM2 보조 + WVN-style 병행 |
| ROS2 msg 인터페이스 후속 변경 | downstream 영향 | Phase 3 종료 시점에 동결 |
| 통합 6-class + DINOv2 baseline 미공개 | 기대치 가늠 어려움 | Phase 2 첫 실험을 baseline 측정으로 정의 (2.6 참조) |

---

## 8. 참고 문헌·리소스

- DINOv2: Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision" (2023)
- WVN: Frey et al. "Fast Traversability Estimation for Wild Visual Navigation" (RSS 2023)
- DPT: Ranftl et al. "Vision Transformers for Dense Prediction" (2021)
- LSS: Philion & Fidler "Lift, Splat, Shoot" (ECCV 2020)
- BEVFormer: Li et al. (ECCV 2022)
- OWLv2: Minderer et al. "Scaling Open-Vocabulary Object Detection" (NeurIPS 2023)
- RUGD dataset: http://rugd.vision/
- RELLIS-3D: https://github.com/unmannedlab/RELLIS-3D
- SAM 2: https://github.com/facebookresearch/sam2

---

## 9. 진행 상황 트래킹

> 각 Phase 완료 시 본 문서 상단의 표와 해당 섹션의 "완료 기준" 체크박스 갱신.
> 상세 의사결정·실험 결과는 `reports/phase<N>_*.md`로 별도 기록.
