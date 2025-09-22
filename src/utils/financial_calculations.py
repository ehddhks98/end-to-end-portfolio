import torch
import torch.nn.functional as F
import numpy as np


def adjust_data_dim(asset_data, common_data, future_returns, future_common_data):
    """
    데이터로더에서 오는 다양한 형태의 데이터 차원을 모델의 배치 처리에 맞게 조정
    """
    # 1. 자산 데이터 차원 조정 (asset_data)
    if asset_data.dim() == 4 and asset_data.shape[-1] == 1:
        # 일반적인 배치 형태: (batch, assets, seq, 1) -> (batch, assets, seq)
        asset_data = asset_data.squeeze(-1)
    elif asset_data.dim() == 3 and asset_data.shape[2] == 1:
        # 단일 샘플 (추론 시): (assets, seq, 1) -> (1, assets, seq)
        asset_data = asset_data.squeeze(-1).unsqueeze(0)
    
    # 2. 공통 데이터 차원 조정 (common_data)
    if common_data.dim() == 2:
        # 단일 샘플: (seq, features) -> (1, seq, features)
        common_data = common_data.unsqueeze(0)

    # 3. 미래 수익률 데이터 차원 조정 (future_returns)
    # 실현 샤프 비율 계산을 위해 원본 형태는 보존해야 함
    original_future_returns = future_returns.clone()
    
    if future_returns.dim() == 2:
        # 단일 샘플: (pred_horizon, assets) -> (1, pred_horizon, assets)
        original_future_returns = future_returns.unsqueeze(0)
        # 훈련 시에는 평균값을 사용
        future_returns = future_returns.mean(dim=0).unsqueeze(0)
    elif future_returns.dim() == 3:
        # 배치: (batch, pred_horizon, assets) -> (batch, assets)
        future_returns = future_returns.mean(dim=1)
        
    if future_common_data.dim() == 2:
        future_common_data = future_common_data.unsqueeze(0)

    return asset_data, common_data, future_returns, original_future_returns, future_common_data


def capm_expected_returns(betas, common_data):
    """
    CAPM을 사용하여 기대 수익률을 계산
    Fama-French 데이터를 직접 사용
    E(R_i) = R_f + β_i * (E(R_m) - R_f)
    """
    if common_data is None or common_data.shape[-1] < 2:
        raise ValueError("common_data에 Mkt-RF와 RF 데이터가 필요합니다.")
    
    # Fama-French 데이터에서 직접 추출하여 lookback 기간의 평균값 사용
    market_premium = common_data[:, :, 0].mean(dim=1, keepdim=True)  # Mkt-RF -> (batch, 1)
    risk_free_rate = common_data[:, :, 1].mean(dim=1, keepdim=True)  # RF -> (batch, 1)
    
    # CAPM 공식 적용
    expected_returns = risk_free_rate + betas * market_premium
    return expected_returns


def cal_cov_matrix_capm(betas, returns_data, common_data):
    """
    (최종 수정) 단일 지수 모형을 사용하여 공분산 행렬을 계산
    TCN 모델은 베타만 예측하므로, CAPM 이론에 따라 알파는 0으로 가정
    Σ = ββ^T * Var(R_m - R_f) + D(Var(ε_i))
    
    Args:
        betas (torch.Tensor): 모델이 추정한 베타 벡터 (batch, num_assets)
        returns_data (torch.Tensor): 자산의 '총' 수익률 데이터 (batch, seq, assets)
        common_data (torch.Tensor): 시장 데이터 (batch, seq, features). 0번: Mkt-RF, 1번: RF
        
    Returns:
        torch.Tensor: 단일 지수 모형 기반 공분산 행렬 (batch, assets, assets)
    """
    # --- 1. 체계적 위험 계산 ---
    market_excess_returns = common_data[:, :, 0] # Mkt-RF, shape: (batch, seq)
    market_variance = market_excess_returns.var(dim=1, unbiased=True, keepdim=True) # (batch, 1)

    beta_outer = torch.bmm(betas.unsqueeze(2), betas.unsqueeze(1))
    systematic_cov = beta_outer * market_variance.unsqueeze(-1)

    # --- 2. 비체계적 위험 계산 ---
    
    # 2a. 초과 수익률 계산
    asset_excess_returns = returns_data - common_data[:, :, 1].unsqueeze(-1) # (batch, seq, assets)
    
    # 2b. 잔차(residuals) 계산 (알파=0 가정): ε = r_i - β * r_m
    # 브로드캐스팅을 위해 차원 조정:
    # betas: (batch, assets) -> (batch, 1, assets)
    # market_excess_returns: (batch, seq) -> (batch, seq, 1)
    predicted_asset_excess = betas.unsqueeze(1) * market_excess_returns.unsqueeze(2)
    residuals = asset_excess_returns - predicted_asset_excess # (batch, seq, assets)
    
    # 2c. 잔차의 분산을 비체계적 위험 분산으로 사용
    idiosyncratic_variances = torch.var(residuals, dim=1, unbiased=True)
    
    # 2d. 대각행렬 D 생성
    D = torch.diag_embed(idiosyncratic_variances)

    # --- 3. 최종 공분산 행렬 결합 ---
    cov_matrix_sim = systematic_cov + D
    
    return cov_matrix_sim
    

def cal_cov_matrix(returns_data):
    """  
    Args:
        returns_data (torch.Tensor): 수익률 데이터 (batch_size, time_steps, num_assets)
    
    Returns:
        torch.Tensor: 축소된 공분산 행렬 (batch_size, num_assets, num_assets)
    """
    batch_size, T, N = returns_data.shape
    
    # 1. 데이터 중심화
    mean_returns = returns_data.mean(dim=1, keepdim=True)
    centered_returns = returns_data - mean_returns
    
    # 2. 샘플 공분산 행렬 (S)
    # transpose의 차원 인덱스가 1, 2인 것을 확인
    S = torch.bmm(centered_returns.transpose(1, 2), centered_returns) / (T - 1)
    
    # 3. 구조화된 추정기 (F) - Shrinkage Target
    prior_variances = torch.diagonal(S, dim1=-2, dim2=-1)
    F_diag = prior_variances.mean(dim=-1, keepdim=True).expand(-1, N)
    F = torch.diag_embed(F_diag)
    
    # 4. 축소 강도(Shrinkage Intensity) 파라미터 계산
    # d^2 = ||S - F||^2
    d2 = (S - F).pow(2).sum(dim=[1, 2])
    
    # b^2 = 1/(T-1)^2 * sum_{t=1 to T} || (x_t * x_t^T) - S ||^2
    # x_t는 중심화된 수익률 벡터
    
    # (batch, T, N, 1) @ (batch, T, 1, N) -> (batch, T, N, N)
    xxt = centered_returns.unsqueeze(3) @ centered_returns.unsqueeze(2)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    # S를 T 차원에 맞게 확장
    S_expanded = S.unsqueeze(1)
    
    # b^2의 분자 부분
    # 분모를 T^2에서 (T-1)^2으로 변경하여 S와 일관성 유지
    b2_numerator = (xxt - S_expanded).pow(2).sum(dim=[2, 3]).sum(dim=1)
    b2 = b2_numerator / (T**2)

    # 5. 축소 계수(Shrinkage Coefficient) 계산
    shrinkage_coef = b2 / torch.clamp(d2, min=1e-10)
    shrinkage_coef = torch.clamp(shrinkage_coef, 0.0, 1.0)
    
    # 6. 최종 축소된 공분산 행렬 계산
    delta = shrinkage_coef.view(-1, 1, 1)
    shrunk_cov = delta * F + (1 - delta) * S
    
    # 수치적 오차 교정을 위한 대칭성 보장
    return (shrunk_cov + shrunk_cov.transpose(1, 2)) / 2


def cal_cov_matrix_simple(returns_data):
    """
    Ledoit-Wolf 축소 없이, PyTorch 기본 기능을 사용하여 미분 가능하게 구현
    Args:
        returns_data (torch.Tensor): 수익률 데이터 (batch_size, time_steps, num_assets)    
    Returns:
        torch.Tensor: 샘플 공분산 행렬 (batch_size, num_assets, num_assets)
    """
    batch_size, T, N = returns_data.shape
    
    # 데이터 중심화 (각 자산의 시간축 평균을 뺌)
    mean_returns = returns_data.mean(dim=1, keepdim=True)
    centered_returns = returns_data - mean_returns
    
    # 샘플 공분산 행렬 계산
    # 공식: Cov(X) = (X_centered^T * X_centered) / (n - 1)
    # torch.bmm은 배치 행렬 곱셈을 수행
    # centered_returns.transpose(1, 2) -> (batch, num_assets, time_steps)
    # centered_returns                  -> (batch, time_steps, num_assets)
    # 결과 -> (batch, num_assets, num_assets)
    cov_matrix = torch.bmm(centered_returns.transpose(1, 2), centered_returns) / (T - 1)
    
    return cov_matrix
    

def cal_realized_sharpe_ratio(weights, future_returns, future_common_data):
    """
    실제 미래 수익률 데이터를 사용하여 실현된 샤프 비율을 계산
    """
    # 각 날짜별 실제 포트폴리오 수익률
    daily_portfolio_returns = torch.sum(weights.unsqueeze(1) * future_returns, dim=2)
    daily_rf_rate = future_common_data[:,:,1]
    
    # 포트폴리오 수익률 통계 (일간)
    daily_excess_portfolio_returns = daily_portfolio_returns - daily_rf_rate

    mean_excess_pf_return = daily_excess_portfolio_returns.mean(dim=1)
    std_pf_return = daily_portfolio_returns.std(dim=1)
    
    # 연간화
    annualization_factor = 252
    portfolio_return_annual = mean_excess_pf_return * annualization_factor
    portfolio_std_annual = std_pf_return * (annualization_factor ** 0.5)    
    
    # 실현 샤프 비율
    realized_sharpe = portfolio_return_annual/ torch.clamp(portfolio_std_annual, min=1e-8)
    
    return realized_sharpe, daily_portfolio_returns


def validate_portfolio_weights(weights, tolerance=1e-4):
    """
    포트폴리오 가중치의 합이 1이고, 유한한 값인지 검증
    """
    if not torch.isfinite(weights).all():
        print("경고: 가중치에 NaN 또는 Inf가 포함되어 있습니다.")
        return False
        
    weights_sum = weights.sum(dim=1)
    if not (torch.abs(weights_sum - 1.0) <= tolerance).all():
        print(f"경고: 가중치의 합이 1이 아닙니다. 합: {weights_sum}")
        return False
        
    return True


def ols_beta_estimation(asset_returns, market_returns, risk_free_rate):
    """
    Args:
        asset_returns (torch.Tensor): 자산 수익률 (batch_size, seq_len, num_assets)
        market_returns (torch.Tensor): 시장 수익률 (Mkt-RF가 아닌 실제 시장 인덱스 수익률) (batch_size, seq_len)
        risk_free_rate (torch.Tensor): 무위험 이자율 (batch_size, seq_len)
    
    Returns:
        torch.Tensor: 각 자산의 베타 (batch_size, num_assets)
    """
    # 입력 차원 확장
    market_returns = market_returns.unsqueeze(-1) # (batch, seq, 1)
    risk_free_rate = risk_free_rate.unsqueeze(-1) # (batch, seq, 1)

    # 초과 수익률 계산
    y_excess = asset_returns - risk_free_rate
    x_excess = market_returns - risk_free_rate

    # 회귀분석을 위한 설계 행렬 X = [x_excess, 1] (상수항 추가)
    X = torch.cat([x_excess, torch.ones_like(x_excess)], dim=2)

    # torch.linalg.lstsq를 사용하여 (X * solution = y)를 만족하는 solution을 찾음    
    try:
        solution = torch.linalg.lstsq(X, y_excess).solution
    except torch.linalg.LinAlgError:
        # 특이 행렬 오류 발생 시 의사 역행렬(pseudo-inverse) 사용
        X_pinv = torch.linalg.pinv(X)
        solution = torch.bmm(X_pinv.transpose(1, 2), y_excess)

    # solution은 (batch, 2, num_assets) 형태
    # solution[:, 0, :]는 기울기(베타)
    # solution[:, 1, :]는 절편(알파)
    betas_ols = solution[:, 0, :]
    
    return betas_ols


def cal_sp500_sharpe_ratio(future_common_data):
    """
    Args:
        future_common_data (torch.Tensor): 
        미래 기간의 공통 데이터. 
        (batch_size, pred_horizon, num_features) 형태.
        인덱스 0: Mkt-RF, 인덱스 1: RF

    Returns:
        torch.Tensor: 각 배치 샘플에 대한 S&P 500의 샤프 지수. (batch_size,) 형태.
    """
    # 'Mkt-RF'는 시장의 '초과 수익률'이므로 그대로 사용
    # shape: (batch_size, pred_horizon)
    daily_market_excess_returns = future_common_data[:, :, 0]
    risk_free_rate = future_common_data[:,:,1]
    daily_market_return= daily_market_excess_returns+risk_free_rate
    
    # 일별 초과 수익률의 평균과 표준편차 계산
    mean_excess_market_return = daily_market_excess_returns.mean(dim=1)
    std_excess_market_return = daily_market_return.std(dim=1)
    
    # 연간화
    annualization_factor = 252
    market_return_annual = mean_excess_market_return * annualization_factor
    market_std_annual = std_excess_market_return * (annualization_factor ** 0.5)    
    
    # S&P 500 실현 샤프 지수
    sp500_sharpe = market_return_annual / torch.clamp(market_std_annual, min=1e-8)
    
    return sp500_sharpe
    

def cal_mdd(daily_portfolio_returns):
    """
    Args:
        daily_portfolio_returns (torch.Tensor): (batch_size, horizon) 형태의 일별 수익률 텐서.

    Returns:
        torch.Tensor: 각 배치 샘플에 대한 MDD 값 (양수). (batch_size,) 형태.
    """
    device = daily_portfolio_returns.device
    # 1. 포트폴리오 가치 계산 (초기값 1)
    # (1 + r_t)를 누적으로 곱하기 위해 1을 추가
    cumulative_returns = torch.cumprod(1 + daily_portfolio_returns, dim=1)    
    
    initial_value = torch.ones(cumulative_returns.shape[0], 1, device=device)
    portfolio_values = torch.cat([initial_value, cumulative_returns], dim=1)

    # 2. 기간 내 최고점(Running Peak) 계산
    running_peak = torch.cummax(portfolio_values, dim=1).values

    # 3. 각 시점의 낙폭(Drawdown) 계산
    # 낙폭 = (현재 가치 - 최고점) / 최고점    
    epsilon = 1e-9
    drawdown = (portfolio_values - running_peak) / (running_peak + epsilon)

    # 4. 최대 낙폭(MDD) 찾기
    # drawdown 텐서에서 가장 작은 값(가장 큰 손실)을 확인
    # .min()은 값과 인덱스를 함께 반환하므로 .values로 값만 추출
    max_drawdown = torch.min(drawdown, dim=1).values    
    
    return -max_drawdown
