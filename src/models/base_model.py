import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.models.components.beta_estimator import BetaEstimator
from src.models.components.optimizer import PortfolioQPOptimizer
from src.utils.financial_calculations import (
    adjust_data_dim, capm_expected_returns, cal_cov_matrix, 
    cal_cov_matrix_capm, cal_realized_sharpe_ratio, cal_sp500_sharpe_ratio,
    cal_mdd, validate_portfolio_weights, ols_beta_estimation
)


class BasePortfolioOptimizer(nn.Module, ABC):
    """포트폴리오 최적화 모델의 베이스 클래스"""
    
    def __init__(self, num_assets, hidden_size, num_channels, kernel_size, risk_aversion, dropout):
        super().__init__()
        self.num_assets = num_assets
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.risk_aversion = risk_aversion
        self.cov_regularization = 1e-6
        
        # 공통 컴포넌트들
        self.beta_estimator = BetaEstimator(num_assets, hidden_size, num_channels, kernel_size, dropout)
        self.qp_optimizer = PortfolioQPOptimizer(risk_aversion)
    
    def _prepare_data(self, asset_data, common_data, future_returns, future_common_data):
        """데이터 전처리"""
        return adjust_data_dim(asset_data, common_data, future_returns, future_common_data)
    
    def _estimate_parameters(self, asset_data, common_data, static_mkt_rf=None):
        """모델 파라미터 추정"""
        # 베타 및 기대 수익률 예측
        if static_mkt_rf is not None:
            common_data_capm = common_data.clone()
            common_data_capm[:, :, 0] = static_mkt_rf
        else:
            common_data_capm = common_data

        betas = self.beta_estimator(asset_data.transpose(1, 2), common_data)
        expected_returns = capm_expected_returns(betas, common_data)
        
        # OLS 기반 벤치마크
        market_excess_for_ols = common_data[:, :, 0]
        risk_free_rate_for_ols = common_data[:, :, 1]
        market_returns_for_ols = market_excess_for_ols + risk_free_rate_for_ols
        
        beta_ols = ols_beta_estimation(asset_data, market_returns_for_ols, risk_free_rate_for_ols)
        return_ols = capm_expected_returns(beta_ols, common_data)
        
        return betas, expected_returns, beta_ols, return_ols
    
    def _calculate_covariance(self, asset_data, common_data, betas, static_cov=None):
        """공분산 행렬 계산 - 서브클래스에서 오버라이드 가능"""
        if static_cov is not None:
            batch_size = asset_data.shape[0]
            return static_cov.unsqueeze(0).expand(batch_size, -1, -1)
        
        return cal_cov_matrix(asset_data)
    
    def _optimize_portfolio(self, expected_returns, return_ols, cov_matrix, common_data, allow_short_selling=True):
        """포트폴리오 최적화"""
        current_rf = common_data[:, :, 1].mean(dim=1).unsqueeze(-1)
        weights = self.qp_optimizer.optimize(expected_returns, cov_matrix, current_rf, allow_short_selling)
        weights_ols = self.qp_optimizer.optimize(return_ols, cov_matrix, current_rf, allow_short_selling)
        
        validate_portfolio_weights(weights)  
        validate_portfolio_weights(weights_ols)
        
        return weights, weights_ols, current_rf
    
    def _calculate_performance_metrics(self, weights, weights_ols, original_future_returns, future_common_data_adj):
        """성능 지표 계산"""
        realized_sharpe, daily_portfolio_returns = cal_realized_sharpe_ratio(
            weights, original_future_returns, future_common_data_adj
        )
        ols_sharpe, ols_portfolio_returns = cal_realized_sharpe_ratio(
            weights_ols, original_future_returns, future_common_data_adj
        )
        sp500_sharpe = cal_sp500_sharpe_ratio(future_common_data_adj)
        
        model_mdd = cal_mdd(daily_portfolio_returns)
        ols_mdd = cal_mdd(ols_portfolio_returns)
        
        return {
            'realized_sharpe': realized_sharpe,
            'daily_portfolio_returns': daily_portfolio_returns,
            'ols_sharpe': ols_sharpe,
            'ols_portfolio_returns': ols_portfolio_returns,
            'sp500_sharpe': sp500_sharpe,
            'model_mdd': model_mdd,
            'ols_mdd': ols_mdd
        }
    
    @abstractmethod
    def _calculate_loss(self, performance_metrics, expected_returns, asset_data, weights, cov_matrix, **kwargs):
        """손실 함수 계산 - 서브클래스에서 구현해야 함"""
        pass
    
    def forward(self, asset_data, common_data, future_returns, future_common_data, 
                allow_short_selling: bool = True, static_cov=None, static_mkt_rf=None, **kwargs):
        """순전파"""
        # 1. 데이터 전처리
        asset_data, common_data, _, original_future_returns, future_common_data_adj = \
            self._prepare_data(asset_data, common_data, future_returns, future_common_data)
        
        # 2. 모델 파라미터 추정
        betas, expected_returns, beta_ols, return_ols = self._estimate_parameters(asset_data, common_data, static_mkt_rf)
        
        # 3. 공분산 행렬 계산
        cov_matrix = self._calculate_covariance(asset_data, common_data, betas, static_cov)
        
        # 4. 포트폴리오 최적화
        weights, weights_ols, current_rf = self._optimize_portfolio(
            expected_returns, return_ols, cov_matrix, common_data, allow_short_selling
        )
        
        # 5. 성능 지표 계산
        performance_metrics = self._calculate_performance_metrics(
            weights, weights_ols, original_future_returns, future_common_data_adj
        )
        
        # 6. 손실 계산 (서브클래스에서 구현)
        loss_info = self._calculate_loss(
            performance_metrics, expected_returns, asset_data, weights, cov_matrix, **kwargs
        )
        
        # 위험 프리미엄 분석을 위한 RF 값
        current_rf_scalar = common_data[:, :, 1].mean(dim=1)
        
        # 결과 반환
        result = {
            'betas': betas,
            'expected_returns': expected_returns,
            'weights': weights,
            'cov_matrix': cov_matrix,
            'realized_sharpe_ratio': performance_metrics['realized_sharpe'],
            'realized_return': performance_metrics['daily_portfolio_returns'],
            'ols_sharpe_ratio': performance_metrics['ols_sharpe'],
            'ols_return': performance_metrics['ols_portfolio_returns'],
            'sp500_sharpe_ratio': performance_metrics['sp500_sharpe'],
            'max_drawdown': performance_metrics['model_mdd'],
            'ols_drawdown': performance_metrics['ols_mdd'],
            'ols_weights': weights_ols,
            'ols_expected_returns': return_ols,
            'risk_free_rate': current_rf_scalar,
            'ols_beta': beta_ols
        }
        
        # 손실 정보 추가
        result.update(loss_info)
        
        return result
