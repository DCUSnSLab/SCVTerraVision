# Phase 0 — 기반 인프라

- **상태**: 🟢 승인완료
- **시작일**: 2026-04-22
- **완료일**: 2026-04-22
- **담당 PR**: (작성 후 링크)

## 목표

저장소를 실제 개발 착수가 가능한 상태로 올리는 일회성 부트스트랩. Phase 1 이후 작업이 이 인프라 위에서 돌아갈 수 있도록 의존성 정의 · 린트 · 테스트 수집 · Hydra 골격 · 문서/진행 추적 체계를 배치한다. 모델/학습/추론 코드는 본 단계에서 **구현하지 않는다**.

## 체크리스트

- [x] `pyproject.toml` — 런타임 + dev 의존성, ruff/black 설정
- [x] `.pre-commit-config.yaml` — ruff + black + 기본 hygiene hook
- [x] `pytest.ini` — `testpaths=tests`
- [x] `.gitignore` — Python 표준 + data/ 대용량 + `.env` (HF_TOKEN)
- [x] `configs/base.yaml` — Hydra root 스켈레톤 (subgroup 자리는 주석만)
- [x] `docs/PLAN.md` — 승인 플랜의 in-repo 사본
- [x] `docs/progress/README.md` — 단계 대시보드
- [x] `docs/progress/phase0.md` — 본 문서
- [x] `docs/decisions/20260422_dinov3-backbone.md` — ADR
- [x] `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/annotations/.gitkeep`
- [x] `tests/__init__.py` — 빈 골격
- [x] `models/segmentation/` → `models/traversability/` 디렉토리 교체 (빈 디렉토리)
- [x] `README.md` 수정 — Traversability 용어 병기, 디렉토리 트리 갱신, License 섹션 MIT, 기술 스택/로드맵 현행화
- [x] `pytest -q` 수집 에러 없이 종료 (테스트 0건 허용)

## 산출물

- `pyproject.toml`, `.pre-commit-config.yaml`, `pytest.ini`, `.gitignore`
- `configs/base.yaml`
- `docs/PLAN.md`, `docs/progress/README.md`, `docs/progress/phase0.md`
- `docs/decisions/20260422_dinov3-backbone.md`
- `models/traversability/.gitkeep`
- `data/{raw,processed,annotations}/.gitkeep`
- `tests/__init__.py`
- `README.md` (수정)

## 검증 로그

- `python3 -m pytest -q` → `no tests ran in 0.00s`, exit code 5 (pytest 의 "no tests collected" 코드, 수집 에러 아님). 시스템 pytest 6.2.5 로 실행. ✅
- `python3 -c "import yaml; yaml.safe_load(open('configs/base.yaml').read())"` → 파싱 성공, keys = `['defaults', 'project', 'paths', 'logging', 'hydra']`. ✅
- `python3 -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml').read())"` → 파싱 성공, repos 3 개 (ruff / black / pre-commit-hooks). ✅
- `pyproject.toml` 파싱 — 현 시스템 Python 3.10 에는 `tomllib` 이 없어 라이브 검증 생략. PEP 621 문법 준수, 실제 검증은 `pip install -e .[dev]` 시도 시 수행. ⚪
- `pre-commit run --all-files` — pre-commit 이 아직 시스템에 설치되지 않아 Phase 0 세션에서는 수행하지 않음. 승인 후 머지 전 확인 권장. ⚪
- `python -c "import torch; print(torch.cuda.is_available())"` — PyTorch 설치는 Phase 0 범위 밖. Phase 1-2a 착수 시 실행. ⚪

## 검토 요청 노트

- **의존성 버전**: `transformers>=4.56` 은 DINOv3 지원 최소치. 다른 하한은 합리적인 2026-01 시점 안정판 기준이며, 필요 시 상향/하향 조정 가능.
- **PyTorch 미락**: CUDA 버전에 따라 각자 설치하도록 `pyproject.toml` 에서 의도적으로 제외. 팀 표준 CUDA 버전이 정해지면 `docs/progress/phase0.md` 말미에 "권장 설치 커맨드" 를 추가할지 결정 필요.
- **디렉토리 네이밍**: `models/segmentation/` 제거 후 `models/traversability/` 로 교체. 기존 참조는 Phase 0 에서 아직 없음 (그린필드).
- **W&B 단일화**: MLflow 는 플랜에 따라 제거. `configs/base.yaml.logging.wandb` 만 존재.

## 사용자 검토 결과

- 2026-04-22: 사용자 승인 — "검토 완료, 다음 진행". Phase 1-1 착수 지시.
