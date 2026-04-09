# 🤖 TerraVision

**Foundation Model 기반 모빌리티 로봇 인지 시스템**

다양한 주행 환경(캠퍼스, 도심, 험지, 농경지)에서 자율주행 로봇이 주변 환경을 이해하고 주행 가능 경로를 판단하기 위한 카메라 기반 인지 모델입니다. DINOv2 등 비전 파운데이션 모델을 활용하여 범용적이고 강건한 인지 성능을 목표로 합니다.

---

## 📌 프로젝트 목적

모빌리티 로봇의 자율주행에 필요한 **시각 인지 파이프라인**을 구축합니다.

- **Object Detection** — 주행 경로 상의 객체를 탐지하고 분류
- **Freespace Segmentation** — 주행 가능 영역과 불가 영역을 픽셀 단위로 판별
- **멀티 환경 대응** — 단일 모델로 캠퍼스, 도심, 험지, 농경지 등 다양한 환경에서 동작

---

## 🌍 대상 주행 환경 및 인지 객체

### 공통 (전 환경)
사람(보행자), 차량/로봇, 자전거/킥보드, 동물, 낙하물/장애물

### 캠퍼스
벤치, 볼라드, 자전거 거치대, 계단/경사로, 건물 출입문, 표지판, 화단/조경물

### 도심
승용차/버스/트럭, 신호등, 횡단보도, 가드레일, 전봇대, 공사 구조물(바리케이드/콘)

### 험지
바위, 웅덩이, 경사면, 쓰러진 나무/나뭇가지, 도랑/수로, 불규칙 지면(자갈/진흙)

### 농경지
밭고랑, 비닐하우스, 관개 시설, 농기계, 작물 열(row), 울타리/경계선

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                   Camera Inputs                     │
│  ┌───────────┐  ┌───────────────────────────────┐   │
│  │  Front    │  │   4x Fisheye (360° Surround)  │   │
│  │  Camera   │  │   FL  /  FR  /  RL  /  RR     │   │
│  └─────┬─────┘  └──────────────┬────────────────┘   │
│        │                       │                    │
│        └───────────┬───────────┘                    │
│                    ▼                                │
│        ┌───────────────────────┐                    │
│        │  Image Preprocessing  │                    │
│        │  (Undistort / Stitch) │                    │
│        └───────────┬───────────┘                    │
│                    ▼                                │
│        ┌───────────────────────┐                    │
│        │   Foundation Model    │                    │
│        │   Backbone (DINOv2)   │                    │
│        └─────┬─────────┬──────┘                    │
│              │         │                           │
│         ┌────▼───┐ ┌───▼────────┐                  │
│         │  Det   │ │  Seg Head  │                  │
│         │  Head  │ │ (Freespace)│                  │
│         └────┬───┘ └───┬────────┘                  │
│              │         │                           │
│              ▼         ▼                           │
│        ┌───────────────────────┐                    │
│        │   Fusion & Decision   │                    │
│        │  (Navigation Output)  │                    │
│        └───────────────────────┘                    │
└─────────────────────────────────────────────────────┘
```

---

## 📷 카메라 구성

| 카메라 | 용도 | 비고 |
|--------|------|------|
| 전면 카메라 | 전방 정밀 인지 (원거리 객체 탐지) | 일반 렌즈 |
| Fisheye × 4 | 360° 서라운드 뷰 구성 | 전좌/전우/후좌/후우 |

- Fisheye 왜곡 보정 후 합성 또는 BEV(Bird's Eye View) 변환
- 전면 카메라는 원거리 정밀 탐지에 활용

---

## 🔬 기술 스택 (계획)

| 구분 | 기술 |
|------|------|
| Backbone | DINOv2 / Grounding DINO |
| Detection | DINO-DETR / Co-DETR 기반 |
| Segmentation | Mask2Former / SegFormer 기반 |
| Framework | PyTorch |
| 추론 최적화 | TensorRT / ONNX Runtime |
| 데이터 관리 | CVAT / Label Studio |
| 실험 관리 | Weights & Biases / MLflow |

---

## 📂 프로젝트 구조 (예정)

```
terravision/
├── configs/                # 학습/추론 설정 파일
├── data/
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   └── annotations/       # 라벨 데이터
├── models/
│   ├── backbone/          # 파운데이션 모델 관련
│   ├── detection/         # Object Detection Head
│   └── segmentation/      # Freespace Segmentation Head
├── preprocessing/
│   ├── undistort/         # Fisheye 왜곡 보정
│   ├── stitching/         # 멀티 카메라 합성
│   └── bev/               # Bird's Eye View 변환
├── training/              # 학습 스크립트
├── inference/             # 추론 파이프라인
├── evaluation/            # 평가 메트릭 및 스크립트
├── utils/                 # 유틸리티 함수
├── notebooks/             # 실험 노트북
├── docs/                  # 문서
└── tests/                 # 테스트 코드
```

---

## 🗺️ 로드맵

### Phase 1 — 기반 구축
- [ ] 프로젝트 환경 설정 및 개발 인프라 구축
- [ ] 카메라 캘리브레이션 및 전처리 파이프라인 개발
- [ ] 공개 데이터셋 조사 및 선정 (nuScenes, Cityscapes, RUGD 등)
- [ ] DINOv2 백본 통합 및 기본 Detection/Segmentation 구현

### Phase 2 — 환경별 모델 개발
- [ ] 캠퍼스/도심 환경 Object Detection 학습 및 평가
- [ ] 험지/농경지 환경 Object Detection 학습 및 평가
- [ ] Freespace Segmentation 모델 개발
- [ ] 멀티태스크 학습 구조 실험

### Phase 3 — 통합 및 최적화
- [ ] 멀티 카메라 합성 파이프라인 통합
- [ ] 모델 경량화 및 추론 속도 최적화 (TensorRT)
- [ ] 온보드 하드웨어 배포 테스트
- [ ] 실환경 주행 테스트 및 피드백 반영

---

## 🤝 Contributing

프로젝트 기여 방법은 추후 업데이트 예정입니다.

---

## 📄 License

TBD

---

## 📚 참고 자료

- [DINOv2 (Meta AI)](https://github.com/facebookresearch/dinov2)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [nuScenes Dataset](https://www.nuscenes.org/)
- [RUGD (Robot Unstructured Ground Driving)](http://rugd.vision/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
