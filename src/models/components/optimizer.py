import torch
from qpth.qp import QPFunction


class PortfolioQPOptimizer:
    """포트폴리오 최적화를 위한 QP 솔버"""
    
    def __init__(self, risk_aversion: float = 4.0):
        self.risk_aversion = risk_aversion
    
    def optimize(self, expected_returns, cov_matrix, risk_free_rate, allow_short_selling: bool = True):
        """
        포트폴리오 최적화를 수행합니다.
        
        Args:
            expected_returns (torch.Tensor): 각 자산의 기대 수익률 (batch_size, num_assets)
            cov_matrix (torch.Tensor): 자산 공분산 행렬 (batch_size, num_assets, num_assets)
            risk_free_rate (torch.Tensor or float): 무위험 수익률 (batch_size, 1) 또는 스칼라
            allow_short_selling (bool): 공매도 허용 여부

        Returns:
            torch.Tensor: 최적의 포트폴리오 가중치 (batch_size, num_assets)
        """
        batch_size = expected_returns.shape[0]
        n = expected_returns.shape[1]
        device = expected_returns.device

        # --- QP 문제 목적 함수 설정 ---
        # 우리가 원하는 목적 함수 (최대화): (w^T mu - r_f) - lambda * (w^T Sigma w)
        # qpth는 최소화 (1/2) w^T Q w + p^T w 를 풀므로, 목적 함수를 변환:
        # min (lambda * w^T Sigma w - (w^T mu - r_f))
        # min ( (1/2) w^T (2 * lambda * Sigma) w - (w^T mu - r_f * e) )

        # Q matrix: 2 * lambda * cov_matrix
        # qpth는 Q에 1/2이 곱해진 형태로 Q를 입력받으므로, Q = 2 * lambda * Sigma
        Q = 2 * self.risk_aversion * cov_matrix 
        
        # p matrix: -(mu - r_f * e) => 마이너스 초과 수익률 벡터
        # 무위험 수익률을 배치 사이즈에 맞게 확장 (만약 스칼라로 들어온다면)
        if isinstance(risk_free_rate, (float, int)):
            r_f_tensor = torch.full((batch_size, 1), float(risk_free_rate), device=device)
        else:
            r_f_tensor = risk_free_rate.view(batch_size, 1) # (batch_size, 1) 형태로 확보

        # 각 자산의 초과 수익률 벡터 (mu - r_f * e)
        excess_returns_vector = expected_returns - r_f_tensor 

        # Linear term p: -(초과 수익률 벡터)
        p = -excess_returns_vector

        # --- QP 문제 제약 조건 설정 ---
        # s.t.   G w <= h
        #        A w = b

        # Equality constraint: sum of weights = 1 (A w = b)
        A = torch.ones(batch_size, 1, n, device=device) # (batch_size, 1, n)
        b = torch.ones(batch_size, 1, device=device) # (batch_size, 1)

        # Inequality constraints: w_i >= 0 (G w <= h)
        if allow_short_selling:
            # 공매도 허용 시: 효과 없는 제약 조건
            # 예: 0 * w <= 0 (항상 참)
            G = torch.zeros(batch_size, 1, n, device=device)
            h = torch.zeros(batch_size, 1, device=device)
        else:
            # 공매도 금지: w_i >= 0  => -w_i <= 0
            G = -torch.eye(n, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            h = torch.zeros(batch_size, n, device=device)

        # QPFunction을 사용하여 최적화 문제 해결
        weights = QPFunction(verbose=False)(Q, p, G, h, A, b)
        
        return weights
