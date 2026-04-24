# ADR 2026-04-22 — Detection backbone 채택: DINOv3 ViT-B/16

- **상태**: 확정 (Phase 0 승인 플랜 기준)
- **일자**: 2026-04-22
- **관련 phase**: 1-2 Detection
- **관련 파일(예정)**: `models/backbone/dinov3_backbone.py`, `configs/detection/dinov3_detr_base.yaml`

## 결정

Phase 1 Detection 모델의 백본으로 **DINOv3 ViT-B/16** (`facebook/dinov3-vitb16-pretrain-lvd1689m`) 을 채택한다. 로딩 경로는 Hugging Face `transformers.AutoModel.from_pretrained(...)` 이다. Head 는 `mmdetection` 의 DINO-DETR 구현을 베이스로 하고 백본만 교체한다.

## 근거

- Foundation-model 수준의 일반화 성능이 캠퍼스 · 도심 · 험지 · 농경지 혼합 환경에 유리.
- ViT-B/16 (86M) 은 Jetson Orin 급 온보드 추론을 염두에 둔 파라미터 규모 — ViT-L/H 는 FPS 목표 15+ 달성이 어렵다고 판단.
- `AutoModel.from_pretrained` 경로는 `torch.hub` 와 달리 repo clone 이 불필요하고 캐시/오프라인 로딩이 깔끔하다.
- DETR head 자체 재구현은 공수 대비 이득이 낮음 — 성숙한 `mmdetection` DINO-DETR 을 재사용.

## 리스크

1. **라이선스 (gated)**: DINOv3 는 Apache-2.0 이 아니라 자체 **DINOv3 License** 로 배포되며, HF 에서 gated access 로 제공된다. 다운로드에 `HF_TOKEN` 과 라이선스 동의가 필요하고, **상용 배포 시 조건 재검토가 필요**하다.
   - 완화: `docs/progress/phase1-2_detection.md` 착수 시 라이선스 조건을 명시하고, 폴백(아래)을 `configs/detection/` 에서 선택 가능하도록 설계.
2. **HF 토큰 관리**: 팀 공용 계정 vs 개인 토큰, CI 에서의 캐시/미러링 전략 미정.
   - 완화: Phase 1-2a 착수 시점에 합의. 당장은 개인 토큰 + 로컬 캐시로 시작 가능.
3. **패치 크기 제약**: ViT/16 이므로 입력 해상도가 16 의 배수여야 한다. (예: 1024×1024 → 64×64 token grid.)
   - 완화: 전처리 단계에서 크기를 명시적으로 16 배수로 clamp.
4. **mmdetection 의존**: 무거운 프레임워크를 끌고 들어오는 비용.
   - 완화: Phase 1-2b 진입 시 실제 학습 루프 범위에서만 import, 추론 경로에는 별도 경량 래퍼를 두도록 설계.

## 대안 (채택하지 않음)

- **DINOv2 ViT-B/14 (Apache-2.0)**: 라이선스 제약 없음. 상용 배포 시 폴백 후보로 유지한다. `configs/detection/dinov2_detr_base.yaml` 을 Phase 1-2 에서 함께 제공해 한 줄 스위치로 교체 가능하게 만든다.
- **Grounding DINO**: open-vocab 장점은 크지만, 제어부에 고정 클래스 id 를 내려주는 현재 파이프라인과는 궁합이 떨어져 보류.
- **YOLO-X / RT-DETR 의 자체 백본**: FPS 는 유리하나 멀티환경 일반화가 foundation-model 대비 약함 — Phase 3 경량화 단계에서 재고 가능.

## 후속 결정 (Phase 1-2 착수 시)

- `HF_TOKEN` 관리 방법 확정 (개인 vs 팀 공용, CI 캐시 전략).
- mmdetection 버전 pin.
- 입력 해상도 기본값 (1024 vs 1280) 결정 — 15 FPS 목표 달성 가능 범위에서 최댓값 선택.
