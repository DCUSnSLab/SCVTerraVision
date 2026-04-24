# ADR 2026-04-24 — DETR 헤드 라이브러리: mmdetection 대신 HF Transformers Deformable DETR 를 Phase 1-2b 베이스라인으로 채택

- **상태**: 확정 (사용자 승인 2026-04-24)
- **일자**: 2026-04-24
- **관련 phase**: 1-2b (DETR 헤드 학습), 1-2c (파인튠)
- **관련 ADR**: `docs/decisions/20260422_dinov3-backbone.md`
- **관련 파일(예정)**: `models/detection/detr_head.py`, `configs/detection/dinov3_detr_base.yaml`, `training/train_detection.py`

## 배경

승인된 PLAN 은 DETR 헤드로 **mmdetection 의 DINO-DETR** 를 쓰도록 기술했다. 한편 Phase 0 의 `pyproject.toml` 은 `mmdetection`, `mmengine`, `mmcv` 를 **의존성에 포함하지 않았고**, `transformers>=4.56` 만 들어 있다. 즉 플랜의 문구와 실제 설치된 인프라 사이에 공백이 있다. Phase 1-2b 코드 작성 전에 이 공백을 메우는 결정이 필요하다.

세 가지 경로를 검토했다.

## 옵션

### A. 플랜대로 mmdetection DINO-DETR 도입

| 장점 | 단점 |
|---|---|
| DINO-DETR 구현 품질이 입증됨 (COCO SOTA 기반) | `mmcv-full` 이 CUDA·PyTorch 버전 조합에 매우 민감 — 빌드 실패 흔함 |
| 학계 표준, 벤치마크 재현 용이 | 의존 그래프가 크고 (`mmengine` + `mmcv` + `mmdet`) lock-in 심함 |
| Deformable attention + denoising 등 최신 기법 내장 | 백본 교체가 mmdet 내부 convention (`MMDET_BACKBONES` registry) 에 맞춰야 함 — DINOv3 ViT wrapper 를 다시 포팅 |
| — | Jetson 온보드 배포(Phase 3) 때 mmdet 의존이 남으면 TRT 변환 경로가 복잡 |

### B. HF Transformers `DeformableDetrForObjectDetection` 채택 (제안)

| 장점 | 단점 |
|---|---|
| 이미 설치된 `transformers>=4.56` 만으로 동작 — 추가 의존 없음 | DINO 특유의 denoising training · CDN · query selection 부재 |
| 백본이 같은 HF 생태계 — `AutoModel` 와 통합 자연스러움 | DINO-DETR 대비 벤치마크 수치 조금 낮음 (COCO 2017 Deformable DETR ≈ 46 AP vs DINO-DETR 49 AP, ViT-B 기준) |
| ONNX / TRT 변환 경로가 HF + 표준 PyTorch 로 단순 | |
| Hungarian matcher · set loss 내장, 학습 루프 단순 | |
| 라이선스 Apache-2.0 | |

### C. 직접 구현 (Hungarian matcher + set loss + decoder)

- 플랜 명시적으로 "자체 구현 없음" 이라 배제. 단 참조용.

## 제안 — 옵션 B (HF Deformable DETR)

Phase 1-2b 베이스라인은 **HF `DeformableDetrForObjectDetection`** 을 커스텀 백본(`DinoV3Backbone`) 과 결합해서 쓴다. 근거:

1. **인프라 충격 최소**. mmdet 설치에 시간 쓸 필요 없고, Phase 0·1-2a 와 스택이 일관.
2. **백본-헤드 일관성**. DINOv3 백본도 HF, 헤드도 HF → `AutoModel` / `PreTrainedModel` / Trainer 등 통일된 API.
3. **Deformable DETR 은 오픈 벤치마크 기준으로 충분**. Phase 1-2b 목표는 "수용 가능한 베이스라인" 이지 SOTA 재현이 아니다. 46 AP 수준이면 CODa · BDD 서브셋 학습 결과 해석에 무리 없다.
4. **DINO 특유 최적화가 필요해지면 Phase 1-2c 에서 mmdet 또는 HF Grounding-DINO 경로로 upgrade 가능**. 이때는 베이스라인 수치가 이미 있으니 delta 로만 판단.

### 수반되는 PLAN 편차

- 기존 PLAN 의 "mmdetection DINO-DETR" 문구는 본 ADR 로 대체한다.
- `models/detection/detr_head.py` = `DinoV3Backbone` + `DeformableDetrDecoder` 어댑터로 구현.
- `training/train_detection.py` 는 plain PyTorch + HF Trainer **없이** 자체 루프 (Hydra entrypoint) 로 짠다 — HF Trainer 는 분산/로깅 덩어리가 커서 교육·디버깅 투명성이 떨어짐.

## 리스크

1. **Deformable attention CUDA extension**: HF 의 deformable attention 은 기본적으로 순수 PyTorch fallback 을 사용하지만, 성능을 위해 커스텀 CUDA op 을 컴파일하는 경로도 있음. 처음엔 PyTorch fallback 으로 충분.
2. **Query 개수·encoder layer 조정**: DeformableDetr 기본값은 COCO 80-class 를 가정. 우리 운영 택소노미는 16 class 라 query 수(300) 는 그대로 두고 `num_labels=16` 으로 맞춤.
3. **Multi-scale features**: Deformable DETR 은 multi-scale feature 를 기대. DINOv3 ViT-B 는 single-scale (stride 16). 여러 해상도를 만들려면 FPN-like 추가 모듈이 필요. Phase 1-2b 에서는 single-scale 로 시작하고 (HF `disable_custom_kernels=True` 경로에서 가능), 필요하면 1-2c 에서 간단한 pyramid (stride 8/16/32) 를 추가.

## 대안 (채택하지 않음)

- **A (mmdet)**: 인프라 투자 대비 현 시점 이득 불분명. Phase 1-2c 에서 데이터·베이스라인 정리 후 재검토.
- **C (자체 구현)**: 플랜 원칙 위배. 교육 목적이라도 검증된 구현 위에서 실험하는 편이 효율적.

## 후속 결정 (사용자 확정 후)

- `pyproject.toml` 은 그대로 (transformers 만으로 충족).
- `configs/detection/dinov3_detr_base.yaml` 스키마 — query 수, encoder/decoder layer, aux_loss, weight_dict 등 기본값.
- Phase 1-2b 베이스라인 데이터셋 — CODa training split 단독 vs CODa + BDD100K 합성. 기본값: **CODa training split 단독** (ADR 20260422_coda-primary-dataset 의 primary 원칙 그대로).
- 학습 해상도 — 1024 (patch multiple, CODa native 와 근접) vs 800 (COCO 관행). 기본값: **1024×1024 center-crop 또는 1024×1216 (CODa 비율)**.
