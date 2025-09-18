import torch
from src.models.base_model import BasePortfolioOptimizer
from src.utils.financial_calculations import cal_cov_matrix_capm


class PortfolioOptimizer(BasePortfolioOptimizer):
    """Sharpe 비율 최대화에 특화된 포트폴리오 최적화 모델"""
    
    def _calculate_covariance(self, asset_data, common_data, betas):
        """CAPM 기반 공분산 행렬 계산"""
        return cal_cov_matrix_capm(betas, asset_data, common_data)
    
    def _calculate_loss(self, performance_metrics, expected_returns, asset_data, weights, cov_matrix, **kwargs):
        """Sharpe 비율 기반 손실 계산"""
        realized_sharpe = performance_metrics['realized_sharpe']
        loss = -realized_sharpe.mean()
        
        return {
            'loss': loss
        }
