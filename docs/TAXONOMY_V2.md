# Unified Taxonomy v2

> 18 classes + ignore (255). Supersedes v1 (6 classes). Covers 캠퍼스 · 도심 ·
> 험지 · 농경지 4개 환경. v1 rollup으로 하위 호환.

## Design rationale

**문제** (v1의 한계)
- v1은 `traversable_smooth`, `traversable_grass`, `non_traversable_terrain`,
  `obstacle_static`, `obstacle_dynamic`, `sky` 6개로만 분할 — off-road 편향
- 도심에서 중요한 포장/비포장 구분, 신호등/표지판 구분 없음
- 농경지 전용 요소(밭고랑, 비닐하우스) 자리 없음
- `obstacle_dynamic`이 person·rider·bicycle·vehicle·animal을 통째로 묶어 planner의
  행동 예측에 정보 손실

**해결**
v2는 계층형으로 설계. 각 Tier 1 그룹 아래에 세부 클래스가 있고, Tier 1이 v1에
대응되므로 inference-time에 rollup이 가능.

## Class table (18 classes)

| id | 이름 | 그룹 | 설명 | 예시 |
|----|------|------|------|------|
| 0 | road_paved | TRAVERSABLE | 포장 주행면 | 아스팔트, 콘크리트, 보도, 횡단보도 |
| 1 | path_unpaved | TRAVERSABLE | 비포장 주행 가능면 | 흙길, 자갈, 임도, 캠퍼스 흙 산책로 |
| 2 | grass_low | TRAVERSABLE | 예초된 잔디 | 캠퍼스·공원 잔디 |
| 3 | furrow | TRAVERSABLE | 밭고랑 (농경지 전용, 현 데이터셋에 없음) | |
| 4 | water | NON_TRAVERSABLE_TERRAIN | 물·웅덩이 | 연못, 도랑, 수로 |
| 5 | mud | NON_TRAVERSABLE_TERRAIN | 진흙 | 비 온 뒤 off-road, 논 |
| 6 | rough_terrain | NON_TRAVERSABLE_TERRAIN | 거친 주행 불가 지형 | 키 큰 풀, 덤불, 바위밭, 돌무더기 |
| 7 | building_wall | OBSTACLE_STATIC | 대형 구조물 | 건물 외벽, 담장 |
| 8 | vertical_pole | OBSTACLE_STATIC | 얇은 수직 장애물 | 전봇대, 신호등, 표지판, 가로등 |
| 9 | tree | OBSTACLE_STATIC | 나무·큰 식생 | 가로수, 산림, 과수 |
| 10 | barrier_fence | OBSTACLE_STATIC | 경계·장애선 | 펜스, 가드레일, **연석(curb)**, 볼라드 |
| 11 | polytunnel | OBSTACLE_STATIC | 비닐하우스 (농경지 전용, 현 데이터셋에 없음) | |
| 12 | person | OBSTACLE_DYNAMIC | 보행자 | 걷거나 서 있는 사람 |
| 13 | rider | OBSTACLE_DYNAMIC | 탑승자 | 자전거·오토바이·킥보드 탄 사람 |
| 14 | bicycle | OBSTACLE_DYNAMIC | 자전거 객체 | 탑승 여부 무관, 자전거 자체 |
| 15 | vehicle | OBSTACLE_DYNAMIC | 차량·기계 | 승용차/트럭/버스/오토바이/로봇/농기계 |
| 16 | animal | OBSTACLE_DYNAMIC | 동물 | 개, 고양이, 사슴, 소 |
| 17 | sky | CONTEXT | 하늘 | |
| 255 | ignore | — | void, 애매 영역, dataset에 없는 클래스 | |

## Hierarchical groups

```
TRAVERSABLE              {road_paved, path_unpaved, grass_low, furrow}
NON_TRAVERSABLE_TERRAIN  {water, mud, rough_terrain}
OBSTACLE_STATIC          {building_wall, vertical_pole, tree, barrier_fence, polytunnel}
OBSTACLE_DYNAMIC         {person, rider, bicycle, vehicle, animal}
CONTEXT                  {sky}
```

Group은 yaml의 `groups:` 필드로 정의되며, Python 쪽에서
`taxonomy.class_ids_in_group("TRAVERSABLE")`으로 id 리스트를 얻음.
`BinaryTraversabilityIoU` 메트릭이 이를 사용해 v1/v2 taxonomy 모두에서 자동 동작.

## v1 rollup

```yaml
rollup_to_v1:
  road_paved:      traversable_smooth
  path_unpaved:    traversable_smooth
  grass_low:       traversable_grass
  furrow:          traversable_smooth
  water:           non_traversable_terrain
  mud:             non_traversable_terrain
  rough_terrain:   non_traversable_terrain
  building_wall:   obstacle_static
  vertical_pole:   obstacle_static
  tree:            obstacle_static
  barrier_fence:   obstacle_static
  polytunnel:      obstacle_static
  person:          obstacle_dynamic
  rider:           obstacle_dynamic
  bicycle:         obstacle_dynamic
  vehicle:         obstacle_dynamic
  animal:          obstacle_dynamic
  sky:             sky
```

사용법:
```python
from camera_perception.data.taxonomy import UnifiedTaxonomy

v2 = UnifiedTaxonomy.load("configs/taxonomy/traversability_v2.yaml")
v1 = UnifiedTaxonomy.load("configs/taxonomy/traversability_v1.yaml")
lut = v2.rollup_lut(v1)   # (v2_ids,) -> v1_ids

v1_mask = lut[v2_pred_mask]  # broadcast-friendly
```

## Per-dataset class coverage

공개 데이터셋은 v2의 모든 18 클래스를 라벨하지 않는다. 각 데이터셋 config는
`ignore_if_absent:` 필드로 해당 데이터셋이 라벨하지 **않는** 클래스를 명시하고,
학습 시 loss 코드가 이 클래스들의 logit을 `-inf`로 마스킹해 gradient를 차단한다.

### 현재 지원 데이터셋

| 클래스 | RUGD | RELLIS-3D |
|--------|:----:|:---------:|
| road_paved | ✅ (asphalt/concrete) | ✅ (asphalt/concrete) |
| path_unpaved | ✅ (dirt/gravel/mulch/sand) | ✅ (dirt) |
| grass_low | ✅ (grass) | ✅ (grass) |
| furrow | ❌ | ❌ |
| water | ✅ (water) | ✅ (water/puddle) |
| mud | 🟡 (dirt, ambiguous → ignore) | ✅ (mud) |
| rough_terrain | ✅ (bush/rock-bed/rock) | ✅ (bush/rubble) |
| building_wall | ✅ (building/container) | ✅ (building/object) |
| vertical_pole | ✅ (pole/sign) | ✅ (pole) |
| tree | ✅ (tree) | ✅ (tree) |
| barrier_fence | ✅ (fence/log/bridge/picnic-table) | ✅ (fence/barrier/log) |
| polytunnel | ❌ | ❌ |
| person | ✅ | ✅ |
| rider | ❌ (RUGD labels person regardless) | ❌ |
| bicycle | ✅ | ❌ |
| vehicle | ✅ | ✅ |
| animal | ❌ | ❌ |
| sky | ✅ | ✅ |

### Phase B 확장 예정

Cityscapes, BDD100K 추가 시 도심 커버리지 확보. 그 두 데이터셋은 반대로
off-road 클래스(water/mud/rough_terrain/path_unpaved/grass_low/furrow/polytunnel/animal)를
`ignore_if_absent`로 선언하게 됨.

## 학습 시 per-dataset ignore 동작

1. DataModule이 각 샘플을 반환할 때 `present_classes: BoolTensor[num_classes]`를
   함께 반환 (True = 해당 샘플의 출처 데이터셋이 이 클래스를 라벨함)
2. `LitSegmenter._step`이 batch에서 `present_classes`를 꺼내 loss 함수에 전달
3. Loss 함수가 absent 클래스의 logit을 `-inf`로 채운 뒤 softmax 계산
   - CrossEntropy: absent 클래스에 대한 gradient = 0
   - Dice: absent 클래스는 per-class dice 평균에서 제외
4. 결과: 모델은 "이 샘플의 출처 데이터셋이 라벨하지 않는 클래스"를 예측으로
   내놓지 않으며, 해당 클래스의 head 파라미터도 이 샘플로부터 잘못된 신호를 받지 않음

## 구현 참고

- 정의: `configs/taxonomy/traversability_v2.yaml`
- 로드: `src/camera_perception/data/taxonomy.py` (`UnifiedTaxonomy`, `DatasetMapping`)
- 샘플 반환: `src/camera_perception/data/datasets/base.py` (`present_classes` 추가됨)
- Loss 마스킹: `src/camera_perception/models/losses.py` (`_mask_logits`)
- 메트릭: `src/camera_perception/training/metrics.py` (`_traversable_class_ids`)
- 이전 버전: `configs/taxonomy/traversability_v1.yaml` (rollup 대상으로 유지)
