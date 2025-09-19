import torch
import torch.nn as nn
from src.models.components.tcn_encoder import TCNEncoder


class BetaEstimator(nn.Module):
    def __init__(self, num_assets, hidden_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_assets = num_assets
        
        self.asset_encoder = TCNEncoder(
            input_size=1,
            hidden_size=hidden_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )       

        self.market_encoder = TCNEncoder(
            input_size=1,
            hidden_size=hidden_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        self.asset_embeddings = nn.Embedding(num_assets, hidden_size)
        self.beta_hidden1 = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        self.beta_hidden2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5)
        )
        
        self.beta_output = nn.Linear(hidden_size // 2, 1)
        # 베타 초기화 - 평균 1.0 근처에서 시작
        nn.init.normal_(self.beta_output.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.beta_output.bias, 1.0)  # 베타 1.0에서 시작

    def forward(self, asset_data, common_data):        
        if asset_data.dim() == 4:
            asset_data = asset_data.squeeze(-1)
        elif asset_data.dim() == 3 and asset_data.shape[2] == 1:
            asset_data = asset_data.squeeze(-1).unsqueeze(0)
        
        if common_data.dim() == 2:
            common_data = common_data.unsqueeze(0)
            
        batch_size, num_assets, seq_len = asset_data.shape
        device = asset_data.device
        
        # 시장 수익률 추출 및 인코딩
        market_returns = common_data[:, :, 0].unsqueeze(-1)  # (batch, seq_len, 1)
        market_context = self.market_encoder(market_returns)  # (batch, hidden_size)
        
        # 배치 처리를 위한 자산 데이터 재구성
        asset_returns_flat = asset_data.transpose(1, 2).reshape(-1, seq_len).unsqueeze(-1)       
        asset_contexts_flat = self.asset_encoder(asset_returns_flat)  # (batch*assets, hidden_size)        
        # 개별 자산으로 다시 재구성
        asset_contexts = asset_contexts_flat.view(batch_size, num_assets, -1)  # (batch, assets, hidden_size)
        
        # 모든 자산에 대한 자산 임베딩 한 번에 생성
        asset_ids = torch.arange(num_assets, device=device).unsqueeze(0).expand(batch_size, -1)
        asset_embeddings = self.asset_embeddings(asset_ids)  # (batch, assets, hidden_size)
        
        # 시장 컨텍스트를 자산 차원에 맞게 확장
        market_context_expanded = market_context.unsqueeze(1).expand(-1, num_assets, -1)
        
        # 모든 정보 소스 결합
        combined = torch.cat([
            asset_contexts,          # (batch, assets, hidden_size)
            market_context_expanded, # (batch, assets, hidden_size)
            asset_embeddings         # (batch, assets, hidden_size)
        ], dim=2)  # (batch, assets, hidden_size * 3)
        
        # MLP 처리를 위한 평탄화 및 모든 베타 한 번에 예측
        combined_flat = combined.view(-1, combined.size(-1))  # (batch*assets, hidden_size*3)
        
        # 베타 예측
        hidden1 = self.beta_hidden1(combined_flat)  # (batch*assets, hidden_size)
        hidden2 = self.beta_hidden2(hidden1)        # (batch*assets, hidden_size//2)
        betas_flat = self.beta_output(hidden2)      # (batch*assets, 1)
        
        # 원래 배치 형태로 재구성
        betas = betas_flat.view(batch_size, num_assets)  # (batch, assets)
        
        return betas
