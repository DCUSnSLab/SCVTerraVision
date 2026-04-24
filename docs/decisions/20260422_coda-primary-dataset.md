# ADR 2026-04-22 — CODa 를 Detection 파인튠 primary 데이터셋으로 채택, BDD100K 를 auxiliary 로 downgrade

- **상태**: 확정 (2026-04-22, 사용자 승인)
- **일자**: 2026-04-22
- **관련 phase**: 1-1b (CODa 어댑터), 1-2b/1-2c (학습·파인튠)
- **관련 파일(예정)**: `training/datasets/coda_to_coco.py`, `configs/dataset/coda.yaml`, `configs/dataset/coda_taxonomy.yaml`

## 결정

Detection 파인튠의 **primary 학습·평가 코퍼스** 를 **CODa (UT Campus Object Dataset, UT AMRL, 2023)** 로 한다. 앞선 계획에서 primary 였던 **BDD100K 는 auxiliary 로 내려간다** — 컨버터·로더는 그대로 유지하되, 도심 도메인 증강 ablation 용으로만 선택적으로 투입한다.

CODa 사용을 위해 3D 박스를 2D 로 투영하는 어댑터(`coda_to_coco.py`)를 Phase 1-1b 에서 구현한다. 운영 택소노미는 CODa 원본 59 클래스 → trim 후 약 15~20 클래스 수준으로 축소한다. 구체 매핑은 Phase 1-1b 의 첫 checkbox 에서 확정한다.

## 근거 (네 축)

| 축 | BDD100K | CODa | 판단 |
|---|---|---|---|
| 카메라 플랫폼·시점 | 승용차 대시보드, 1.2–1.5 m, 수평 전방 | Husky 급 소형 로봇, ~0.5–0.8 m 시점 | 우리 타겟(소형 모빌리티 로봇)과 CODa 가 일치 |
| 클래스 빈도 분포 | 차량 지배적, 보행자는 작은 bbox | 보행자·자전거·볼라드·벤치·식수대 등 campus-native 비중 높음 | 파인튠 손실이 실전 빈도와 일치 |
| 속도·스케일 분포 | 상대속도 30–100+ km/h, 작은 원거리 bbox 다수 | 보행 속도, 큰 근거리 bbox 다수 | 모델 prior 가 배치 조건에 맞게 형성됨 |
| 환경 클러터 | 정돈된 차로·차선·신호 | 인도·건물 근접·잔디·벤치·보도 | Phase 2 Traversability 와 동일 도메인 → 백본 공유 이득 극대화 |

또한 DINOv3 백본이 이미 foundation-model 수준의 일반 표현을 가지고 있어 파인튠 단계에서는 **데이터 "량" 보다 "배치 조건 일치도"** 가 성능에 결정적이다. 적은 양이라도 도메인 일치 데이터를 우선한다는 원칙에 따라 CODa 를 primary 로 한다.

## 라이선스

CODa 는 **CC BY-NC-SA 4.0** 와 별도의 **"UT Campus Object Dataset License Agreement for Non-Commercial Use (Oct 2023)"** 을 함께 따른다. 주요 의미:

- Non-Commercial: 상용 배포 금지.
- Share-Alike: 본 데이터에서 파생된 데이터를 재배포할 경우 같은 조건.
- Attribution: 논문/문서에 인용 필요 (Zhang et al., "Towards Robust Robot 3D Perception in Urban Environments: The UT Campus Object Dataset", 2023, arXiv:2309.13549).

현재 개발팀은 **대학 연구실 소속이고 연구 목적으로만 활용**할 계획이므로 NC 조건을 그대로 수용한다. 따라서 DINOv3 (자체 비상용 조건) 과 CODa (NC-SA) 가 중첩된 상태의 가중치를 그대로 쓴다 — 상용 트랙 분리(Track-B)는 현재 불필요.

이 전제가 향후 변경되면(스핀오프·기술이전 등으로 상용화 논의가 열리면) 본 ADR 을 재검토하고 새 ADR 로 Track 분리 전략을 확정한다. 이 조건을 `configs/dataset/coda.yaml` 의 최상단 주석에도 명시한다.

## BDD100K 의 역할 재정의

- 컨버터(`training/datasets/bdd100k_to_coco.py`)와 스모크 테스트는 유지.
- Phase 1-2b 베이스라인 수치(CODa primary) 가 나온 뒤, 도심 일반화 개선 ablation 에서 **CODa + BDD100K subset 합성 학습** 을 시도할지 판단.
- BDD 의 10 클래스는 CODa 기반 운영 택소노미 위로 매핑되는 슬림 서브셋만 들어옴 — Phase 1-1b 의 택소노미 결정 후 보조 매핑 테이블을 같은 YAML 에 추가.

## 리스크

1. **2D 박스 없음** — 3D → 2D 투영이 필수. CODa devkit (`helpers/geometry.py::project_3dto2d_bbox`) 이 이미 `cv2.projectPoints` + distortion 처리 포함. Phase 1-1b 에서 동일 math 재구현. 회전 객체에서 axis-aligned corner projection 은 박스가 과대 추정될 수 있어, `isOccluded ∈ {None, Light, Medium}` 필터와 최소 면적 필터를 컨버터 기본값으로 적용.
2. **택소노미 mismatch** — CODa 59 vs 우리 운영 집합. 과잉 세분(실내 가구·디스펜서 등) 을 적절히 collapse/drop 하지 않으면 rare-class 롱테일이 학습을 해침. Phase 1-1b 에서 매핑 YAML 로 외부화.
3. **도메인 편중** — UT Austin 단일 캠퍼스 환경. 우리 캠퍼스(대구가톨릭대) 와의 건축·표지·포장 차이는 자체 촬영 + CVAT 라벨링으로 보강(1-2c).
4. **데이터 다운로드** — Texas Data Repository 폼 게이팅. 자동화 어려움 → 사람이 한 번 받아 팀 공용 경로(`data/raw/coda/`) 에 배치. `.gitignore` 이미 적용됨.

## 대안 (채택하지 않음)

- **BDD 기본 유지 + CODa 캠퍼스 파인튠만 추가**: 플랫폼 불일치 문제가 backbone feature 사용부터 전파됨. 파인튠 loss 가 BDD 의 차량 중심 분포로 편향 — 원복 비용이 큼.
- **CODa + BDD 동시 primary**: 샘플 균형 조절이 어렵고 평가 지표가 흐려짐. 먼저 CODa 단독 베이스라인을 고정한 뒤 BDD 합성은 ablation 으로 시도(위의 auxiliary 역할).
- **CODa 만 사용, BDD 제거**: 컨버터를 이미 구현했고 도시 일반화 ablation 자산으로서 보존 가치가 있음. 유지가 합리적.

## 후속 결정 (Phase 1-1b 착수 시)

- CODa 59 → 운영 택소노미 매핑 YAML 의 구체 내용. 초안은 Phase 1-1b 진행 파일에서 제시하고 사용자 확인 후 확정.
- `isOccluded` / 최소 면적 / FoV 필터 기본값.
- 우리 캠퍼스 자체 촬영의 클래스 추가 정책 — CODa 택소노미 뒤에 append vs 기존 클래스에 병합.
