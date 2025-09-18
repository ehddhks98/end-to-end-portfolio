import torch
from src.models.base_model import BasePortfolioOptimizer
from src.utils.financial_calculations import cal_cov_matrix


class PortfolioOptimizer(BasePortfolioOptimizer):
    """VaR 손실을 포함한 포트폴리오 최적화 모델"""
    
    def cal_kl_divergence_loss(self, predicted_mu, historical_asset_returns):
        """
        예측 기대수익률 분포와 과거 수익률 분포 간의 KL Divergence를 계산
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

    def cal_portfolio_var(self, weights, expected_returns, cov_matrix, confidence_level=0.95):
        """
        포트폴리오의 파라메트릭 VaR(Value at Risk)를 계산.
        """
        z_score_map = {0.95: 1.64485, 0.99: 2.32635}
        if confidence_level not in z_score_map:
            raise ValueError("지원되는 신뢰수준은 0.95 또는 0.99 입니다.")
        z_score = z_score_map[confidence_level]

        portfolio_return = torch.sum(weights * expected_returns, dim=1)
        portfolio_variance = torch.bmm(weights.unsqueeze(1), cov_matrix).bmm(weights.unsqueeze(-1)).squeeze()
        portfolio_std = torch.sqrt(torch.clamp(portfolio_variance, min=1e-8))

        var = z_score * portfolio_std - portfolio_return
        
        return var
    
    def _calculate_loss(self, performance_metrics, expected_returns, asset_data, weights, cov_matrix, 
                      gamma=0.1, eta=0.1, var_confidence_level=0.95, **kwargs):
        """Sharpe 비율 + KL Divergence + VaR 손실 계산"""
        realized_sharpe = performance_metrics['realized_sharpe']
        
        # 손실 계산 (세 가지 요소를 모두 사용)
        # 손실 1: 샤프 비율 최대화 (음수 샤프 비율)
        loss_sharpe = -realized_sharpe.mean()
        
        # 손실 2: 예측과 과거의 분포 차이 최소화
        loss_kl = self.cal_kl_divergence_loss(expected_returns, asset_data)
        
        # 손실 3: 예측된 포트폴리오의 잠재적 손실(VaR) 최소화
        portfolio_var = self.cal_portfolio_var(weights, expected_returns, cov_matrix, var_confidence_level)
        loss_var = portfolio_var.mean()

        # 최종 복합 손실 (가중치 gamma, eta를 사용하여 합산)
        total_loss = loss_sharpe + gamma * loss_kl + eta * loss_var
        
        return {
            'portfolio_var': portfolio_var,
            'loss_sharpe': loss_sharpe,
            'loss_kl': loss_kl,
            'loss_var': loss_var,
            'loss': total_loss
        }
