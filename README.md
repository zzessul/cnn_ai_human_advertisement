# CNN_AIM — AI광고 vs 인간광고 분류 모델

EfficientNet-B0 기반 Transfer Learning으로 AI 생성 광고와 인간 제작 광고를 분류하는 CNN 모델입니다.

---

## 성능

| 평가 방식 | 정확도 |
|---|---|
| 검증 세트 최고 정확도 | **72.2%** |
| 독립 테스트 세트 정확도 | **69.9%** (51/73) |

```
              precision  recall  f1-score  support
    인간광고       0.72    0.68      0.70       38
      AI광고       0.68    0.71      0.69       35
    accuracy                         0.70       73
```

혼동행렬: `[[26, 12], [10, 25]]`

---

## 최종 데이터셋

- 기존 B 학습 데이터 고유 이미지: 인간 153장, AI 149장
- A에서 B와 겹치지 않는 신규 이미지: 인간 101장, AI 79장
- 최종 중복 제거 데이터: **총 482장** (인간 254장, AI 228장)
- 정리 경로: `datasets/organized_combined/`
- `캐릭터` 폴더는 인간 제작 광고로 분류
- SHA-256 완전 중복과 dHash 유사 중복 제거
- 이미지 원본은 로컬에만 보관하며 GitHub 저장소에는 포함하지 않음

---

## 모델 구조

- **Backbone**: EfficientNet-B0 (ImageNet 사전학습)
- **Classifier**: Dropout(0.4) → Linear(1280→256) → SiLU → Dropout(0.2) → Linear(256→2)
- **학습 전략**: 2단계 학습
  1. **Phase 1** (10 epoch): Backbone 고정, Head만 학습 (LR=1e-3)
  2. **Phase 2** (최대 25 epoch): 전체 모델 fine-tune (LR=3e-5)
- **Early stopping**: 검증 정확도가 7 epoch 동안 개선되지 않으면 종료

---

## 파일 구조

```
CNN_AIM/
├── CNN_AIM_Colab.ipynb  # 학습 노트북 (Colab용)
├── train.py             # 로컬 학습 스크립트
├── predict.py           # 단일 이미지 추론 스크립트
├── train_user_data.py   # 정리된 데이터 학습 스크립트
├── predict_user_model.py # 재학습 모델 추론 스크립트
├── organize_datasets.py # A/B 중복 제거 및 데이터 정리
├── artifacts/user_data/metrics.json # 최종 평가 결과
├── requirements.txt     # 패키지 목록
└── README.md
```

---

## 실행 방법

### Google Colab (권장)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zzessul/cnn_ai_human_advertisement/blob/main/CNN_AIM_Colab.ipynb)

1. 위 버튼 클릭
2. 런타임 → 런타임 유형 변경 → **T4 GPU** 설정
3. 셀 순서대로 실행 (약 15~20분 소요)

### 로컬 실행

```bash
pip install -r requirements.txt

# 학습
python train.py

# 단일 이미지 추론
python predict.py 광고_최종/AI광고/AI_01_가방.png
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

## 별도 데이터 폴더로 학습

`train_user_data.py`는 하위 폴더 중 이름이 `AI`, `AI 광고`, `AI광고`,
`인간`, `인간 광고`, `인간광고`, `캐릭터`인 폴더만 라벨로 사용합니다.
이 데이터셋에서 `캐릭터`는 인간 제작으로 분류하며, 파일 해시가 같은 중복 이미지는 제거합니다.

```bash
python train_user_data.py \
  --data-dir "/path/to/cnn 학습 데이터" \
  --output-dir artifacts/user_data

python predict_user_model.py "/path/to/image.jpg" \
  --model artifacts/user_data/best_model.pth
```

결과 폴더에는 모델(`best_model.pth`), 데이터 점검 결과(`data_audit.json`),
테스트 지표(`metrics.json`), 재현 가능한 테스트 목록(`test_manifest.json`)이 저장됩니다.
