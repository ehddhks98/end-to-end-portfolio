import torch
from src.models.base_model import BasePortfolioOptimizer
from src.utils.financial_calculations import cal_cov_matrix


class PortfolioOptimizer(BasePortfolioOptimizer):
    """KL Divergence 손실을 포함한 포트폴리오 최적화 모델"""
    
    def cal_kl_divergence_loss(self, predicted_mu, historical_asset_returns):
        """
        예측 기대수익률 분포와 과거 수익률 분포 간의 KL Divergence를 계산
        
        Args:
            predicted_mu (torch.Tensor): 모델이 예측한 기대수익률 벡터 (batch, assets)
            historical_asset_returns (torch.Tensor): 과거 자산 수익률 데이터 (batch, seq, assets)
        """
        # P 분포 (과거 데이터)의 파라미터 계산
        mu_p = historical_asset_returns.mean(dim=1)
        sigma_p = cal_cov_matrix(historical_asset_returns)

        # Q 분포 (예측)의 파라미터
        mu_q = predicted_mu

        try:
            inv_sigma_p = torch.linalg.inv(sigma_p)
        except RuntimeError:
            identity = torch.eye(self.num_assets, device=sigma_p.device).unsqueeze(0)
            inv_sigma_p = torch.linalg.inv(sigma_p + identity * self.cov_regularization)
            
        diff = mu_q - mu_p
        temp = torch.bmm(diff.unsqueeze(1), inv_sigma_p)
        kl_div = 0.5 * torch.bmm(temp, diff.unsqueeze(-1))
        return kl_div.mean()
    
    def _calculate_loss(self, performance_metrics, expected_returns, asset_data, weights, cov_matrix, gamma=0.1, **kwargs):
        """Sharpe 비율 + KL Divergence 손실 계산"""
        realized_sharpe = performance_metrics['realized_sharpe']
        
        # 손실 계산
        loss_sharpe = -realized_sharpe.mean()
        loss_kl = self.cal_kl_divergence_loss(expected_returns, asset_data)
        total_loss = loss_sharpe + gamma * loss_kl
        
        return {
            'loss_sharpe': loss_sharpe,
            'loss_kl': loss_kl,
            'loss': total_loss
        }
