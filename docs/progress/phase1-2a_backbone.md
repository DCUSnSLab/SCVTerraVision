# Phase 1-2a — DINOv3 백본 래퍼

- **상태**: ✅ 완료·검토대기
- **시작일**: 2026-04-24
- **완료일**: 2026-04-24
- **담당 PR**: (작성 후 링크)
- **관련 ADR**: `docs/decisions/20260422_dinov3-backbone.md`

## 목표

Detection 학습·추론 파이프라인이 사용할 **DINOv3 ViT-B/16 백본 래퍼**를 완성한다. 범위:

- `AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")` 을 감싸 DETR 스타일 헤드가 기대하는 **(B, C, H_patch, W_patch)** feature map 을 돌려주는 얇은 wrapper 를 제공.
- torch · transformers 를 **lazy-import** 해 두 패키지가 없어도 모듈 import · 단위 테스트 · 설정 로드가 통과하도록 한다 (실제 forward 는 Phase 1-2b 학습 환경에서 처음 실행).
- gated 접근 대응: HF_TOKEN 없거나 transformers 미설치 환경에서는 실모델 로딩 테스트를 `pytest.skip` 처리.
- Hydra 서브그룹 `configs/backbone/dinov3_vitb16.yaml` 로 model_id · patch_size · hidden_dim · freeze · dtype · output_layer 를 외부화.
- patch grid math 는 torch 없이 테스트 — 224×224 → 14×14, 1024×1024 → 64×64, 비배수 입력은 ValueError.

Phase 1-2b (DETR 헤드 학습) · Phase 1-2c (캠퍼스 파인튠) 진입 전 backbone forward 가 독립적으로 검증되어 있어야 upstream 디버깅 부담이 줄어든다는 판단.

## 체크리스트

- [x] `models/__init__.py`, `models/backbone/__init__.py` — 패키지 진입점
- [x] `models/backbone/dinov3_backbone.py`
  - [x] `DinoV3BackboneConfig` dataclass (model_id, patch_size, hidden_dim, freeze, dtype, output_layer) + `__post_init__` 검증
  - [x] `DinoV3Backbone` wrapper — lazy load, freeze 옵션, forward → (B, C, Hp, Wp)
  - [x] `patch_grid_shape(H, W)` / `num_patches(H, W)` — 배수·양수 검증
  - [x] transformers / torch 는 메소드 내부에서 지연 import, 없으면 ImportError with guidance
  - [x] register token 개수 변동 대응: `last_hidden_state[:, -Hp*Wp:, :]` slice
- [x] `configs/backbone/dinov3_vitb16.yaml` — Hydra subgroup
- [x] `configs/base.yaml` 주석에 backbone · dataset subgroup 명시
- [x] `tests/test_backbone.py`
  - [x] config 기본값 검증 (+ 잘못된 patch_size/hidden_dim/dtype rejection)
  - [x] `patch_grid_shape` 정상 케이스 (224, 1024, 480×640, 720×1280, 1024×1216, 16×16)
  - [x] `patch_grid_shape` 비배수 입력 · nonpositive 입력 → ValueError
  - [x] custom patch_size (DINOv2 fallback) 경로 검증
  - [x] Hydra YAML 로 config 재구성 왕복
  - [x] torch 없는 환경에서 `load()` → ImportError (guidance 메시지 포함)
  - [x] (gated) `RUN_DINO_SMOKE=1` 없으면 skip — 실모델 forward shape 224, 1024, 비배수 rejection
- [x] `python3 -m pytest -q` 전체 그린 (Phase 1-1 + 1-1b + 1-2a 비-gated = 36 passed, 3 skipped)

## 산출물

- `models/backbone/dinov3_backbone.py`
- `configs/backbone/dinov3_vitb16.yaml`
- `tests/test_backbone.py`
- `configs/base.yaml` (subgroup 주석 업데이트)

## 검증 로그

- `python3 -m pytest -q` → `36 passed, 3 skipped in 0.20s` (시스템 pytest 6.2.5, Python 3.10.12). ✅
  - Phase 1-1: 7 케이스, Phase 1-1b: 11 케이스, Phase 1-2a: 18 케이스 (3 케이스는 `RUN_DINO_SMOKE` 게이트)
- `python3 -c "from models.backbone import DinoV3Backbone, DinoV3BackboneConfig"` — torch 미설치 환경에서도 import 성공. ✅
- `python3 -c "from models.backbone import DinoV3Backbone; DinoV3Backbone().load()"` — `ImportError: DinoV3Backbone.load() requires torch and transformers.` (의도된 경로). ✅
- 실모델 forward smoke (`RUN_DINO_SMOKE=1 HF_TOKEN=... pytest tests/test_backbone.py -k forward_shape`) 는 torch + transformers + gated 가중치 접근이 모두 갖춰진 환경(Phase 1-2b 시작점)에서 단발 실행 예정.

## 검토 요청 노트

- **lazy import 선택 근거**: 현재 개발 환경에 torch/transformers 가 없고, Phase 1-2a 의 스코프는 "wrapper 구조·shape 논리 확정" 이다. 실제 forward smoke 는 torch 설치된 환경(Phase 1-2b 직전) 에서 `RUN_DINO_SMOKE=1 HF_TOKEN=... pytest` 로 단발 검증한다.
- **freeze 정책**: 설정 파일 기본값은 `freeze: true` (플랜의 "첫 2 epoch 동결" 정책에 맞춤). DETR 학습 루프 (Phase 1-2b) 가 epoch 기준으로 unfreeze 하도록 훅 포인트만 열어둔다. 현재 래퍼는 `freeze=False` 옵션도 지원.
- **output_layer / register tokens**: DINOv3 는 CLS + K register token 을 앞쪽에 둔다. patch token 만 slice 해서 spatial grid 로 reshape. HF 구현의 register token 개수가 checkpoint 별로 달라질 가능성에 대비해 N_tokens 에서 뒤쪽 `Hp * Wp` 만 slice 하는 방식을 택함 — config 에서 해당 가정을 명시.
- **dtype 기본값**: `float32`. bfloat16 로 바꾸면 메모리 절감되지만 Jetson 추론 단계(Phase 3) 와 혼동 방지를 위해 학습은 fp32 로 시작. 필요시 `dtype=bfloat16` 오버라이드.
- **DINOv2 폴백**: ADR 20260422_dinov3-backbone 에 명시된 폴백 경로. 현재 Phase 1-2a 코드는 `model_id` 만 바꾸면 DINOv2 ViT-B/14 로도 동작 가능하지만 patch_size (14) · hidden_dim 값 차이가 있어 config override 가 필요. 별도 YAML `configs/backbone/dinov2_vitb14.yaml` 은 실제 폴백 상황에서 추가 예정.
- **notebook 은 이번 단계에서 작성하지 않음**: 플랜의 "(노트북 포함)" 은 feature map 시각화를 가정했지만, 실데이터·실모델 확보 후(Phase 1-2b 시작점) 만들어야 의미가 있다. 현재는 shape 검증 목적의 노트북이 테스트와 중복이 되므로 생략.

## 사용자 검토 결과

(검토 후 기입 — 승인 시 Phase 1-2b 진입, 관련 YAML 은 `configs/detection/dinov3_detr_base.yaml` 로 확장)
