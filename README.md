# SpanNER-kr

한국어 개체명 인식(Named Entity Recognition, NER)을 위한 SpanNER 기반 모델 구현

## 소개
이 프로젝트는 한국어 텍스트에서 개체명을 인식하기 위한 SpanNER 모델을 구현한 것입니다. KLUE 데이터셋을 기반으로 학습되었으며, 높은 정확도의 개체명 인식을 목표로 합니다.

## 프로젝트 구조
```
.
├── config/             # 모델 설정 파일
├── corpus/            # 학습 및 평가용 말뭉치 데이터
├── document/         # 문서 및 가이드
├── error_check/      # 오류 검사 관련 파일
├── klue/             # KLUE 데이터셋 관련 유틸리티
├── model/            # 학습된 모델 저장
├── pattern_check/    # 패턴 검사 관련 파일
├── utils/            # 유틸리티 함수들
├── run_ner.py        # 메인 실행 파일
├── test_ner.py       # 테스트 실행 파일
├── ner_datasets.py   # 데이터셋 처리
├── ner_def.py        # 상수 및 정의
└── ner_utils.py      # NER 관련 유틸리티 함수
```

## 주요 기능
- SpanNER 기반의 한국어 개체명 인식
- KLUE 데이터셋 지원
- 다양한 평가 메트릭 제공 (Entity F1, Character F1)
- TensorBoard를 통한 학습 모니터링

## 사용 방법
1. 학습 실행:
```bash
python run_ner.py
```

2. 테스트 실행:
```bash
python test_ner.py
```

## 의존성
- PyTorch
- Transformers
- NumPy
- tqdm
- attrdict

## 라이선스
Private Repository - All Rights Reserved
