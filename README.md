# AI광고 vs 인간광고 분류 모델

CLIP ViT-L/14와 ConvNeXt-Tiny CNN의 고정 임베딩 앙상블로 AI 생성 광고와 인간 제작 광고를 분류하고 AI 티 점수(0~100)를 출력합니다.

---

## 성능

| 평가 방식 | 정확도 |
|---|---|
| CLIP 학습 세트 5-Fold CV | **86.9% ± 3.0%** |
| 앙상블 검증 정확도 | **90.3%** |
| 독립 테스트 세트 정확도 | **80.6%** (83/103) |

```
              precision  recall  f1-score  support
    인간광고       0.78    0.87      0.82       52
      AI광고       0.84    0.75      0.79       51
    accuracy                         0.81      103
```

혼동행렬: `[[45, 7], [13, 38]]`

---

## 최종 데이터셋

- 기존 B 학습 데이터 고유 이미지: 인간 247장, AI 275장
- A에서 B와 겹치지 않는 신규 이미지: 인간 100장, AI 63장
- 최종 중복 제거 데이터: **총 685장** (인간 347장, AI 338장)
- 분할: 학습 479장, 검증 103장, 테스트 103장
- 라벨과 광고 유형(사람 모델·제품·캐릭터·기존 광고)을 함께 계층화해 분할
- 정리 경로: `datasets/organized_combined/`
- `캐릭터` 폴더는 인간 제작 광고로 분류
- SHA-256 완전 중복과 dHash 유사 중복 제거
- 이미지 원본은 로컬에만 보관하며 GitHub 저장소에는 포함하지 않음

---

## 최종 모델 구조

- **임베딩**: CLIP ViT-L/14와 ConvNeXt-Tiny CNN (사전학습 가중치 고정)
- **입력 방식**: 원본과 좌우 반전 이미지의 정규화 임베딩 평균
- **분류기**: 모델별 Logistic Regression, 최종 확률 가중치 CLIP 0.4 + CNN 0.6
- **선택 방법**: C는 학습 세트 5-Fold CV로 선택하고 앙상블 구성·가중치는 검증 세트에서만 선택
- **비교 결과**: DINOv2는 학습 CV 75.2%로 제외. 단일 CLIP 테스트 78.6%에서 앙상블 80.6%로 개선
- **AI 티 점수**: 두 모델의 AI 클래스 확률 가중평균 × 100. 절대적인 사람 평가 점수가 아니라 학습 데이터 기준 유사도

---

## 파일 구조

```
CNN_AIM/
├── dataset_utils.py # 중복 제거·계층화 분할
├── train_advanced_ensemble.py # CLIP/DINOv2/CNN 비교·앙상블 학습
├── predict_ensemble.py # 최종 앙상블 및 AI 티 점수 추론
├── organize_datasets.py # A/B 중복 제거 및 데이터 정리
├── artifacts/advanced_ensemble/metrics.json # 최종 평가 결과
├── requirements.txt     # 패키지 목록
└── README.md
```

---

## 실행 방법

### 로컬 실행

```bash
pip install -r requirements.txt

# 학습 및 비교
python train_advanced_ensemble.py \
  --data-dir "/path/to/organized_dataset" \
  --output-dir artifacts/advanced_ensemble

# AI 티 점수 추론
python predict_ensemble.py "/path/to/image.jpg"
```

---

## 의존성

```
torch >= 2.0.0
torchvision >= 0.15.0
Pillow >= 9.0.0
scikit-learn >= 1.2.0
numpy >= 1.23.0
```

## 데이터 폴더 규칙

`dataset_utils.py`는 하위 폴더 중 이름이 `AI`, `AI 광고`, `AI광고`,
`인간`, `인간 광고`, `인간광고`, `캐릭터`인 폴더만 라벨로 사용합니다.
이 데이터셋에서 `캐릭터`는 인간 제작으로 분류하며, 파일 해시가 같은 중복 이미지는 제거합니다.
