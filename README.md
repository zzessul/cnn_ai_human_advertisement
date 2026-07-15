# CNN_AIM — AI광고 vs 인간광고 분류 모델

EfficientNet-B0 기반 Transfer Learning으로 AI 생성 광고와 인간 제작 광고를 분류하는 CNN 모델입니다.

---

## 성능

| 평가 방식 | 정확도 |
|---|---|
| 전체 데이터셋 TTA 정확도 | **95.0%** |
| 5-Fold CV 평균 (Fold 4) | 83.3% |
| 5-Fold CV 평균 (Fold 5) | 80.8% |
| 5-Fold CV 전체 평균 | 73.3% |

```
              precision  recall  f1-score  support
    인간광고       0.96    0.95      0.95      115
      AI광고       0.95    0.96      0.96      124
    accuracy                         0.95      239
```

---

## 데이터셋

- 총 239장 (AI광고 124장, 인간광고 115장)
- 경로: `광고_최종/AI광고/`, `광고_최종/인간광고/`

---

## 모델 구조

- **Backbone**: EfficientNet-B0 (ImageNet 사전학습)
- **Classifier**: Dropout(0.5) → Linear(1280→256) → SiLU → Dropout(0.25) → Linear(256→2)
- **학습 전략**: 3단계 학습
  1. **Phase 1** (10 epoch): Backbone 고정, Head만 학습 (LR=1e-3)
  2. **Phase 2** (30 epoch): 마지막 4블록 + Head fine-tune, Mixup (LR=5e-4)
  3. **Phase 3** (20 epoch): 전체 모델 fine-tune, Mixup (LR=5e-5)
- **추론**: TTA 5종 앙상블 (원본, 좌우반전, CenterCrop, ColorJitter, 회전)

---

## 파일 구조

```
CNN_AIM/
├── 광고_최종/
│   ├── AI광고/        # AI 생성 광고 이미지 (124장)
│   └── 인간광고/      # 인간 제작 광고 이미지 (115장)
├── CNN_AIM_Colab.ipynb  # 학습 노트북 (Colab용)
├── train.py             # 로컬 학습 스크립트
├── predict.py           # 단일 이미지 추론 스크립트
├── best_model.pth       # 학습된 모델 가중치 (95% acc)
├── requirements.txt     # 패키지 목록
└── README.md
```

---

## 실행 방법

### Google Colab (권장)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/editpanda-dev/CNN_AIM/blob/main/CNN_AIM_Colab.ipynb)

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

## A-B 데이터 통합 재학습 결과

- 기존 B 학습 데이터 고유 이미지: 인간 153장, AI 149장
- A에서 B와 겹치지 않는 신규 이미지: 인간 101장, AI 79장
- 최종 중복 제거 데이터: 인간 254장, AI 228장, 총 482장
- 검증 최고 정확도: 72.2%
- 독립 테스트 정확도: 69.9% (51/73)
- 혼동행렬: `[[26, 12], [10, 25]]`

정리된 이미지는 `datasets/organized_combined/`에 있으며,
`organize_datasets.py`로 SHA-256 완전 중복과 dHash 유사 중복을 제거할 수 있습니다.
