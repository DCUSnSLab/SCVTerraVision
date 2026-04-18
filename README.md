# Camera Perception for Off-road Autonomous Mobility

캠퍼스/험지 자율주행 이동체용 카메라 기반 perception 시스템.

## 목표

- **Traversability segmentation**: 픽셀 단위로 "갈 수 있는 영역" 판단
- **Object detection**: 사람/차량/지형지물 인식
- **BEV 출력**: 다중 카메라 융합 → bird's eye view (행동 예측/경로 제어 입력)

## 개발 단계

| Phase | 내용 |
|-------|------|
| 1 | 데이터 파이프라인 (현재) |
| 2 | DINOv2 기반 단일 카메라 PoC |
| 3 | Object detection 통합 |
| 4 | 멀티 카메라 + BEV |
| 5 | Jetson 배포 (TensorRT) |

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 데이터

`data/` 디렉토리 (gitignore)에 데이터셋 위치. 심볼릭 링크 권장.

```
data/
├── rugd/
│   ├── images/
│   └── labels/
└── rellis3d/
    ├── images/
    └── labels/
```

데이터셋 출처 안내: `python scripts/download_datasets.py --help`

## 빠른 검증

```bash
python scripts/verify_dataset.py --dataset rugd --root data/rugd
python scripts/visualize_samples.py --dataset rugd --root data/rugd --n 8
pytest
```

## 프로젝트 구조

자세한 설계는 `/Users/soobinjeon/.claude/plans/buzzing-splashing-karp.md` 참조.
