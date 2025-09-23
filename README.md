# End-to-End Portfolio Optimization with Deep Learning

## 프로젝트 개요

이 프로젝트는 **TCN(Temporal Convolutional Networks) 기반의 딥러닝 포트폴리오 최적화 시스템**입니다. 다양한 손실 함수(Sharpe 비율, KL Divergence, VaR)를 통해 포트폴리오의 위험 대비 수익을 최적화하며, CAPM 모델과 결합하여 실제 금융 시장에서 사용 가능한 솔루션을 제공합니다.

## 주요 특징

- **딥러닝 기반**: TCN을 활용한 시계열 데이터 처리
- **다양한 최적화 목표**: Sharpe 비율, KL Divergence, VaR 손실 함수 지원
- **실무 지향**: CAPM 모델과 베타 추정을 통한 실제 포트폴리오 관리
- **유연한 구조**: 모듈화된 설계로 새로운 모델 쉽게 추가 가능
- **포괄적 평가**: Sharpe 비율, MDD, 누적 수익률 등 다양한 성과 지표

## 프로젝트 구조

```
end-to-end-portfolio/
├── main.ipynb                 # 메인 실행 노트북
├── train.py                   # 모델 훈련 스크립트
├── test.py                    # 모델 테스트 및 평가
├── dataloader.py              # 데이터 로딩 및 전처리
├── trainer.py                 # 훈련 루프 관리
│
├── data/                      # 금융 데이터
│   ├── *_daily_technical_data.csv    # 개별 주식 데이터
│   └── F-F_Research_Data_Factors*.CSV # Fama-French 팩터 데이터
│
├── src/
│   ├── models/               # 핵심 모델들
│   │   ├── base_model.py     # 기본 포트폴리오 최적화 클래스
│   │   ├── sharpe_model.py   # Sharpe 비율 최적화
│   │   ├── kl_model.py       # KL Divergence 손실 포함
│   │   ├── var_model.py      # VaR 손실 포함
│   │   └── components/       # 모델 구성 요소
│   │       ├── tcn_encoder.py     # TCN 인코더
│   │       ├── beta_estimator.py  # 베타 추정기
│   │       └── optimizer.py       # 포트폴리오 QP 최적화
│   │
│   └── utils/
│       └── financial_calculations.py # 금융 계산 함수들
│
├── checkpoint/               # 모델 체크포인트
└── result/                  # 결과 저장
```

## 데이터

### 주식 데이터
프로젝트는 **16개 주요 미국 주식**의 일일 기술적 데이터를 사용합니다:

**기술주**: AAPL, MSFT, NVDA, INTC, ORCL, IBM  
**금융주**: JPM, BAC, MS  
**소비재**: PG, JNJ, WMT, COST  
**기타**: UNH, XOM, F

### 시장 팩터
- **Fama-French 리서치 데이터**: 시장 위험 프리미엄, 무위험 수익률

## 모델 아키텍처

### 1. Base Model (`BasePortfolioOptimizer`)
모든 포트폴리오 최적화 모델의 기본 클래스:
- **TCN 기반 베타 추정**: 시계열 패턴 학습
- **CAPM 기대수익률 계산**: 베타와 시장 팩터 활용
- **QP 최적화**: 제약 조건하의 포트폴리오 가중치 계산

### 2. Sharpe Model (`sharpe_model.py`)
```python
Loss = -E[Sharpe Ratio]
```
- **목표**: Sharpe 비율 최대화
- **특징**: 단순하고 직관적인 위험 대비 수익 최적화

### 3. KL Model (`kl_model.py`)
```python
Loss = -E[Sharpe Ratio] + γ × KL_Divergence
```
- **목표**: Sharpe 비율 + 분포 일치
- **특징**: 예측 분포와 역사적 분포의 차이 최소화

### 4. VaR Model (`var_model.py`)
```python
Loss = -E[Sharpe Ratio] + γ × KL_Divergence + η × VaR
```
- **목표**: 종합적 위험 관리
- **특징**: VaR을 통한 극한 손실 제어


### 주요 파라미터
- `--model`: 사용할 모델 (`model_sharpe`, `model_kl`, `model_var`)
- `--epochs`: 훈련 에포크 수 (기본값: 30)
- `--learning_rate`: 학습률 (기본값: 1e-4)
- `--risk_aversion`: 위험 회피도 (기본값: 4)
- `--lookback`: 과거 데이터 길이 (기본값: 252일)
- `--pred_horizon`: 예측 기간 (기본값: 21일)


## 성과 지표

프로젝트는 다음과 같은 포괄적인 성과 평가를 제공합니다:

- **Sharpe 비율**: 위험 대비 수익 측정
- **누적 수익률**: 전체 투자 성과
- **최대 낙폭(MDD)**: 최대 손실 폭
- **변동성**: 포트폴리오 위험도
- **베타**: 시장 민감도

## 기술적 세부사항

### TCN (Temporal Convolutional Networks)
- **인과적 컨볼루션**: 미래 정보 누출 방지
- **잔차 연결**: 깊은 네트워크에서 그래디언트 소실 방지
- **확장 컨볼루션**: 장기 의존성 포착

### 포트폴리오 최적화
- **제약 조건**: 가중치 합 = 1, 공매도 금지 (선택적)
- **QP 최적화**: 효율적인 2차 계획법 사용

## 실험 및 결과

### 백테스팅 설정
- **훈련 기간**: 2000-2009
- **검증 기간**: 2010-2014
- **테스트 기간**: 2015-2024
- **리밸런싱**: 월별

### 벤치마크 비교
- **시장 지수 (S&P 500)**
- **전통적 평균-분산 최적화**

## 확장 가능성

### 새로운 모델 추가
1. `src/models/`에 새 모델 파일 생성
2. `BasePortfolioOptimizer` 상속
3. `_calculate_loss` 메서드 구현
4. `train.py`에 모델 등록

### 새로운 손실 함수
```python
def _calculate_loss(self, performance_metrics, expected_returns, 
                   asset_data, weights, cov_matrix, **kwargs):
    # 커스텀 손실 함수 구현
    return {'loss': custom_loss}
```

## 참고 문헌

- **TCN**: Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
- **CAPM**: Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk.
- **포트폴리오 이론**: Markowitz, H. (1952). Portfolio selection.



