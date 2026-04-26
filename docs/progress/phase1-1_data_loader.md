# Phase 1-1 — 데이터 로더

- **상태**: 🟢 승인완료 (2026-04-22, 단 데이터 전략 변경 — 아래 "사용자 검토 결과" 참조)
- **시작일**: 2026-04-22
- **완료일**: 2026-04-22
- **담당 PR**: (작성 후 링크)

## 목표

Phase 1 Detection 학습·평가가 밟을 **공통 데이터 인터페이스**를 정의한다. 내부 포맷은 COCO JSON 단일화이고, BDD100K 를 그 포맷으로 변환하는 스모크 경로까지를 본 단계의 마감선으로 삼는다. 학습 루프 · 증강 · 샘플링은 Phase 1-2 에서 다룬다.

## 체크리스트

- [x] `training/__init__.py`, `training/datasets/__init__.py`
- [x] `training/datasets/bdd100k_to_coco.py` — 변환 함수 + `python -m ...` CLI
- [x] `training/datasets/coco_loader.py` — torch 없이 동작하는 인덱서 + DataLoader 호환 Dataset 클래스
- [x] `configs/dataset/coco.yaml` — Hydra subgroup 스켈레톤 (경로 규칙만)
- [x] `tests/test_bdd100k_to_coco.py` — 합성 BDD JSON → COCO 구조 검증
- [x] `tests/test_coco_loader.py` — 합성 COCO + PNG → 항목 구조 검증
- [x] `pytest -q` 녹색 (2 파일, 4 케이스 이상)

## 산출물

- `training/datasets/{bdd100k_to_coco.py, coco_loader.py}`
- `configs/dataset/coco.yaml`
- `tests/{test_bdd100k_to_coco.py, test_coco_loader.py}`

## 검증 로그

- `python3 -m pytest -q` → `7 passed in 0.06s` (시스템 pytest 6.2.5). ✅
  - `tests/test_bdd100k_to_coco.py`: 3 케이스 (인메모리 변환 / 파일 왕복 / 잘못된 입력 rejection)
  - `tests/test_coco_loader.py`: 4 케이스 (인덱스 그룹핑 / 누락 키 rejection / 항목 shape·dtype / 주석 0건 이미지)
- `python3 -m training.datasets.bdd100k_to_coco --help` → argparse usage 정상 출력. ✅
- `configs/dataset/coco.yaml` YAML 파싱 성공, Hydra interpolation 문자열(`${paths.data_root}`) 그대로 보존. ✅
- torch · pycocotools 미설치 환경에서 전 경로 통과 확인. Phase 1-2b 에서 torch 설치 후 DataLoader 와의 실제 연동을 재검증할 예정.

## 검토 요청 노트

- **경로 규칙**: 내부 표준 = `data/raw/<source>/{images/...}` + `data/annotations/<source>_<split>_coco.json`. BDD100K 원본을 어디에 풀어둘지(`data/raw/bdd100k/`)만 정해지면 변환 CLI 는 그대로 동작.
- **카테고리 매핑**: BDD100K detection 10 클래스를 1-indexed 로 그대로 옮겼다. 캠퍼스 자체 라벨은 10 이후 ID 를 사용할지(동일 id 공간에 append) 별개 데이터셋으로 유지할지 Phase 1-2c 에서 결정.
- **이미지 크기**: BDD100K 는 공식 1280×720. 변환 CLI 기본값도 그렇게 두었고, 다른 크기 데이터가 섞이면 `--image-width/--image-height` 로 override 하거나 변환 전 이미지에서 읽도록 확장. 지금은 일괄 인자로 받는 단순형.
- **torch / pycocotools 의존**: 본 단계 코드는 torch · pycocotools 없이 동작한다. torch 는 실제 학습 루프(Phase 1-2b)에서 처음 import 한다. 경로 전체가 깔끔한지 리뷰 요청.
- **캘리브 데이터**: BDD100K 에는 intrinsic 이 없다. BEV(1-4) 단계에서는 자체 촬영 체커보드 데이터를 쓸 예정 — 1-1 범위 밖.

## 사용자 검토 결과

- **2026-04-22 — 승인 + 데이터 전략 변경**
  - Phase 1-1 산출물(`coco_loader.py` + `bdd100k_to_coco.py` + 7 tests)은 그대로 머지 승인.
  - 단, BDD100K 의 **역할은 primary → auxiliary 로 downgrade**. 캠퍼스 소형 모빌리티 로봇 배치 조건(카메라 높이·클래스 분포·속도 스케일·환경 클러터) 과 BDD100K(차량 시점 도심 주행) 간 플랫폼 불일치가 파인튠 품질을 해친다는 판단.
  - **CODa** (UT Austin Campus Object Dataset) 를 **primary 로 채택**. 상세 근거는 `docs/decisions/20260422_coda-primary-dataset.md` (ADR).
  - 개발팀이 대학 연구실이고 연구용 한정이므로 CODa 의 CC-BY-NC-SA + UT Non-Commercial Agreement 는 조건 그대로 수용 — 상용 트랙 분리 불필요.
  - 후속 작업 = **Phase 1-1b (CODa 어댑터)** 에서 처리. BDD100K 컨버터는 제거하지 않고 auxiliary 로 유지.
