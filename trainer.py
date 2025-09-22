import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from pathlib import Path
import seaborn as sns

class PortfolioTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, num_epochs,
                learning_rate=1e-3, device='cuda', model_name=None, save_epochs=None, static_cov=None, static_mkt_rf=None):
        """
        Args:
            model: PortfolioOptimizer 모델
            train_loader, val_loader, test_loader: 데이터 로더들
            learning_rate: 학습률
            device: 계산 디바이스
            model_name: 모델 이름 (디렉토리 생성용)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs        
        self.save_epochs = set(save_epochs) if save_epochs is not None else set()
        self.static_cov = static_cov.to(self.device) if static_cov is not None else None
        self.static_mkt_rf = static_mkt_rf.to(self.device) if static_mkt_rf is not None else None
        
        # 디렉토리 생성
        if model_name:
            trainer_dir = Path.cwd() # 현재 작업 디렉토리를 기준으로 설정
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self.checkpoint_dir = trainer_dir / 'checkpoint' / f"{timestamp}_{model_name.lower()}"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.save_path = self.checkpoint_dir
            print(f"저장 경로: {self.checkpoint_dir}")
        
        # 옵티마이저 및 스케줄러
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.7, patience=15, min_lr=1e-6) # 샤프지수 최대화를 위해 mode='max'
        
        # 훈련 기록 (OLS 샤프 추가)
        self.train_losses = []
        self.val_losses = []
        self.train_sharpe = []
        self.val_sharpe = []
        self.train_ols_sharpe = [] 
        self.val_ols_sharpe = []   
        self.train_sp500_sharpe = []
        self.val_sp500_sharpe = []
        self.train_returns = []
        self.val_returns = []
        self.train_mdd = []
        self.val_mdd = []
        
    def format_weights_display(self, weights):        
        avg_weights = weights.mean(dim=0).detach().cpu().numpy()
        weights_str = '[' + ', '.join([f'{w:.3f}' for w in avg_weights]) + ']'
        return weights_str
    
    def train_epoch(self):        
        self.model.train()
        epoch_loss, epoch_sharpe, epoch_ols_sharpe, epoch_sp500_sharpe, epoch_returns, epoch_mdd = 0, 0, 0, 0, 0, 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="훈련 중")
        
        for batch_idx, batch in enumerate(progress_bar):
            asset_data = batch['asset_data'].to(self.device)
            common_data = batch['common_data'].to(self.device)
            future_returns = batch['future_returns'].to(self.device)
            future_common_data = batch['future_common_data'].to(self.device)
            
            self.optimizer.zero_grad()
            results = self.model(asset_data, common_data, future_returns, future_common_data,static_cov=self.static_cov,
                static_mkt_rf=self.static_mkt_rf)
            
            loss = results['loss']
            realized_sharpe = results['realized_sharpe_ratio']
            ols_sharpe = results['ols_sharpe_ratio'] 
            weights = results['weights']
            realized_returns = results['realized_return']
            max_drawdown = results['max_drawdown']
            sp500_sharpe = results['sp500_sharpe_ratio']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_sharpe += realized_sharpe.mean().item()
            epoch_ols_sharpe += ols_sharpe.mean().item()
            epoch_sp500_sharpe += sp500_sharpe.mean().item()
            epoch_returns += realized_returns.mean().item()
            epoch_mdd += max_drawdown.mean().item() 
            num_batches += 1
            
            weights_display = self.format_weights_display(weights)
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'RealSharpe': f'{realized_sharpe.mean().item():.4f}',
                'OLS_Sharpe': f'{ols_sharpe.mean().item():.4f}', 
                'SP500_Sharpe': f'{sp500_sharpe.mean().item():.4f}', 
                'Return': f'{realized_returns.mean().item():.4f}',
                'MDD': f'{max_drawdown.mean().item():.4f}', 
                'Weights': weights_display,
            })
            
        return (epoch_loss / num_batches, 
                epoch_sharpe / num_batches, 
                epoch_ols_sharpe / num_batches,
                epoch_sp500_sharpe / num_batches,
                epoch_returns / num_batches,
                epoch_mdd / num_batches)
    
    def validate(self):
        """검증"""
        self.model.eval()
        epoch_loss, epoch_sharpe, epoch_ols_sharpe, epoch_sp500_sharpe, epoch_returns, epoch_mdd = 0, 0, 0, 0, 0, 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="검증 중")
            
            for batch_idx, batch in enumerate(progress_bar):
                asset_data = batch['asset_data'].to(self.device)
                common_data = batch['common_data'].to(self.device)
                future_returns = batch['future_returns'].to(self.device)
                future_common_data = batch['future_common_data'].to(self.device)

                results = self.model(asset_data, common_data, future_returns, future_common_data, static_cov=self.static_cov,
                    static_mkt_rf=self.static_mkt_rf)
                
                loss = results['loss']
                realized_sharpe = results['realized_sharpe_ratio']
                ols_sharpe = results['ols_sharpe_ratio'] 
                sp500_sharpe = results['sp500_sharpe_ratio']
                weights = results['weights']
                realized_returns = results['realized_return']
                max_drawdown = results['max_drawdown']
                
                epoch_loss += loss.item()
                epoch_sharpe += realized_sharpe.mean().item()
                epoch_ols_sharpe += ols_sharpe.mean().item() 
                epoch_sp500_sharpe += sp500_sharpe.mean().item()
                epoch_returns += realized_returns.mean().item()
                epoch_mdd += max_drawdown.mean().item() 
                num_batches += 1
                
                weights_display = self.format_weights_display(weights)
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'RealSharpe': f'{realized_sharpe.mean().item():.4f}',
                    'OLS_Sharpe': f'{ols_sharpe.mean().item():.4f}',
                    'SP500_Sharpe': f'{sp500_sharpe.mean().item():.4f}', 
                    'MDD': f'{max_drawdown.mean().item():.4f}', 
                    'Return': f'{realized_returns.mean().item():.4f}',
                    'Weights': weights_display
                })

        return (epoch_loss / num_batches, 
                epoch_sharpe / num_batches, 
                epoch_ols_sharpe / num_batches,
                epoch_sp500_sharpe / num_batches,
                epoch_returns / num_batches,
                epoch_mdd / num_batches)
    
    def train(self):
        """전체 훈련 과정"""
        if not hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir = Path.cwd() / 'checkpoints'
            self.checkpoint_dir.mkdir(exist_ok=True)
        
        best_val_sharpe = float('-inf') # 샤프지수는 높을수록 좋으므로 -inf에서 시작
        
        print(f"모델 저장 경로: {self.checkpoint_dir}")
        print(f"포트폴리오 최적화 훈련 시작: {self.num_epochs} 에포크")
        
        for epoch in range(self.num_epochs):
            print(f"\n에포크 {epoch+1}/{self.num_epochs} {'-'*50}")
            
            # 훈련 (OLS 샤프 반환값 추가)
            train_loss, train_sharpe, train_ols_sharpe, train_sp500_sharpe, train_returns, train_mdd = self.train_epoch()
            
            # 검증 (OLS 샤프 반환값 추가)
            val_loss, val_sharpe, val_ols_sharpe, val_sp500_sharpe, val_returns, val_mdd = self.validate()
            
            # 학습률 조정 (검증 샤프 지수 기준)
            self.scheduler.step(val_sharpe)
            
            # 기록 저장 (OLS 샤프 추가)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_sharpe.append(train_sharpe)
            self.val_sharpe.append(val_sharpe)
            self.train_ols_sharpe.append(train_ols_sharpe)
            self.val_ols_sharpe.append(val_ols_sharpe)
            self.train_sp500_sharpe.append(train_sp500_sharpe)
            self.val_sp500_sharpe.append(val_sp500_sharpe)
            self.train_returns.append(train_returns)
            self.val_returns.append(val_returns)
            self.train_mdd.append(train_mdd)
            self.val_mdd.append(val_mdd)
            
            # 결과 출력 (OLS 샤프 비교 구문 추가)
            print(f"  훈련 - 손실: {train_loss:.4f} | 모델 샤프: {train_sharpe:.4f} |  OLS 샤프: {train_ols_sharpe:.4f} | SP500샤프: {train_sp500_sharpe:.4f} | MDD: {train_mdd:.4f} |수익률: {train_returns:.4f}")
            print(f"  검증 - 손실: {val_loss:.4f} | 모델 샤프: {val_sharpe:.4f} | OLS 샤프: {val_ols_sharpe:.4f} | SP500샤프: {val_sp500_sharpe:.4f} | MDD: {val_mdd:.4f} |  수익률: {val_returns:.4f}")
            print(f"  현재 학습률: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 최고 모델 저장 (최고 검증 샤프 지수 기준)
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                model_save_path = self.checkpoint_dir / 'best_model.pth'
                
                try:
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"  >> 새로운 최고 모델 저장 (Val Sharpe: {best_val_sharpe:.4f})")
                except Exception as e:
                    print(f"  >> 모델 저장 실패: {e}")

            if (epoch + 1) in self.save_epochs:
                epoch_model_path = self.checkpoint_dir / f'epoch_{epoch+1}_model.pth'
                torch.save(self.model.state_dict(), epoch_model_path)
                print(f"  >> 지정된 에폭 {epoch+1} 모델 저장 완료.")
                
        # 훈련 완료 후 그래프 저장
        self.plot_training_history(self.checkpoint_dir)
        
        return {k: getattr(self, k) for k in ['train_losses', 'val_losses', 'train_sharpe', 'val_sharpe', 'train_ols_sharpe', 'val_ols_sharpe', 'train_sp500_sharpe', 'val_sp500_sharpe',  'train_returns', 'val_returns', 'train_mdd', 'val_mdd']}

    def test(self):        
        self.model.eval()    
        
        all_results_list = {          
            'loss': [],            
            'betas': [], 
            'weights': [], 
            'realized_sharpe_ratio': [],
            'ols_sharpe_ratio': [], 
            'realized_return': [], 
            'expected_returns': [], 
            'sp500_sharpe_ratio':[], 
            'cov_matrices': [], 
            'max_drawdown': [], 
            'ols_drawdown': [], 
            'cov_matrix': [], 
            'ols_weights': [], 
            'ols_expected_returns': [], 
            'risk_free_rate': []
        }
    
        with torch.no_grad():
            # 테스트 데이터 로더를 순회
            progress_bar = tqdm(self.test_loader, desc="테스트 중 (결과 수집)")
    
            for batch in progress_bar:
                # 데이터를 디바이스로 이동
                asset_data = batch['asset_data'].to(self.device)
                common_data = batch['common_data'].to(self.device)
                future_returns = batch['future_returns'].to(self.device)
                future_common_data = batch['future_common_data'].to(self.device)
    
                # 모델 순전파
                results = self.model(asset_data, common_data, future_returns, future_common_data, static_cov=self.static_cov,
                    static_mkt_rf=self.static_mkt_rf)
    
                # 루프 내에서는 평균을 계산하지 않고, 모든 결과를 리스트에 그대로 수집
                # 이렇게 해야 마지막 배치의 크기가 달라도 정확한 전체 평균을 구할 수 있음
                for key in all_results_list.keys():
                    if key in results:
                        # loss는 스칼라 값이므로 .item()으로 추출
                        if key == 'loss':
                            all_results_list[key].append(results[key].item())
                        # 나머지는 텐서이므로 .cpu()로 옮겨서 리스트에 추가
                        else:
                            all_results_list[key].append(results[key].cpu())
    
        # --- 루프 종료 후, 수집된 전체 결과를 처리 ---
    
        # 텐서 리스트들을 하나의 큰 텐서로 결합 (concatenation)
        final_results = {}
        for key, val_list in all_results_list.items():
            if val_list: # 리스트가 비어있지 않은 경우
                if isinstance(val_list[0], torch.Tensor):
                    final_results[key] = torch.cat(val_list, dim=0)
                else: # loss 같은 스칼라 값들의 리스트
                    final_results[key] = torch.tensor(val_list)    
        
        # 결과가 없을 경우를 대비하여 float('nan')을 기본값으로 설정
        avg_loss = final_results['loss'].mean().item() if 'loss' in final_results and len(final_results['loss']) > 0 else float('nan')
        avg_model_sharpe = final_results['realized_sharpe_ratio'].mean().item() if 'realized_sharpe_ratio' in final_results and len(final_results['realized_sharpe_ratio']) > 0 else float('nan')
        avg_ols_sharpe = final_results['ols_sharpe_ratio'].mean().item() if 'ols_sharpe_ratio' in final_results and len(final_results['ols_sharpe_ratio']) > 0 else float('nan')
        avg_returns = final_results['realized_return'].mean().item() if 'realized_return' in final_results and len(final_results['realized_return']) > 0 else float('nan')
        avg_model_mdd = final_results['max_drawdown'].mean().item() if 'max_drawdown' in final_results and len(final_results['max_drawdown'])> 0 else float('nan')
        avg_ols_mdd = final_results['ols_drawdown'].mean().item() if 'ols_drawdown' in final_results and len(final_results['ols_drawdown'])> 0 else float('nan')
        avg_sp500_sharpe = final_results['sp500_sharpe_ratio'].mean().item() if 'sp500_sharpe_ratio' in final_results and len(final_results['sp500_sharpe_ratio'])> 0 else float('nan')
        
        return {
            'avg_loss': avg_loss,
            'avg_model_sharpe': avg_model_sharpe,
            'avg_ols_sharpe': avg_ols_sharpe,
            'avg_returns': avg_returns,
            'avg_model_mdd': avg_model_mdd,
            'avg_ols_mdd': avg_ols_mdd,
            'avg_sp500_sharpe': avg_sp500_sharpe, 
            'results': final_results  
        }
    
    def plot_training_history(self, save_path):
        """(수정됨) 훈련 히스토리 플롯 (손실, 샤프 비율, MDD, 수익률)"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Training and Validation History', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # --- [0] 손실 그래프 ---
        axes[0].plot(self.train_losses, label='Train Loss', color='royalblue', linewidth=2)
        axes[0].plot(self.val_losses, label='Validation Loss', color='tomato', linewidth=2)
        axes[0].set_title('Loss', fontsize=14)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # --- [1] 샤프 비율 그래프 (모델 vs OLS vs S&P 500) ---
        # 모델 성능
        axes[1].plot(self.train_sharpe, label='Model Sharpe (Train)', color='royalblue', linewidth=2, zorder=5)
        axes[1].plot(self.val_sharpe, label='Model Sharpe (Val)', color='tomato', linestyle='-', linewidth=2, zorder=5)
        
        # OLS 벤치마크
        axes[1].plot(self.val_ols_sharpe, label='OLS Sharpe (Val)', color='lightcoral', linestyle='--', linewidth=2, zorder=4)
        
        # S&P 500 벤치마크
        axes[1].plot(self.val_sp500_sharpe, label='S&P 500 Sharpe (Val)', color='gray', linestyle=':', linewidth=2.5, zorder=3)
    
        axes[1].set_title('Sharpe Ratio: Model vs. Benchmarks', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Realized Sharpe Ratio', fontsize=12)
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.7) # 0 기준선       

        # --- [2] MDD 그래프 ---
        axes[2].plot(self.train_mdd, label='Model MDD (Train)', color='green', linewidth=2)
        axes[2].plot(self.val_mdd, label='Model MDD (Val)', color='orange', linestyle='-', linewidth=2)
        axes[2].set_title('Maximum Drawdown (MDD)', fontsize=14)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('MDD (Lower is better)', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # --- [3] 실현 수익률 그래프 ---
        axes[3].plot(self.train_returns, label='Model Returns (Train)', color='royalblue', linewidth=2)
        axes[3].plot(self.val_returns, label='Model Returns (Val)', color='tomato', linewidth=2)
        axes[3].set_title('Realized Returns', fontsize=14)
        axes[3].set_xlabel('Epoch', fontsize=12)
        axes[3].set_ylabel('Mean Daily Return', fontsize=12)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = save_path / 'training_history_full.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"훈련 과정 전체 그래프 저장 완료: {plot_path}")

    def plot_performance_analysis_test(self, test_results, save_path):
        """(수정됨) 테스트 결과 기반 성능 분석 그래프 (S&P 500 비교 추가)"""
        
        print("\n=== 테스트 결과 성능 분석 시각화 시작 ===")
        results_data = test_results['results']
        
        # 샤프 비율 분포 (모델 vs OLS vs S&P 500)
        if all(k in results_data for k in ['realized_sharpe_ratio', 'ols_sharpe_ratio', 'sp500_sharpe_ratio']):
            try:
                model_sharpe = results_data['realized_sharpe_ratio'].cpu().numpy()
                ols_sharpe = results_data['ols_sharpe_ratio'].cpu().numpy()
                sp500_sharpe = results_data['sp500_sharpe_ratio'].cpu().numpy()
                
                plt.figure(figsize=(12, 7))
                sns.histplot(model_sharpe, kde=True, bins=30, label=f'Model (Mean: {np.mean(model_sharpe):.2f})', color='royalblue', stat='density')
                sns.histplot(ols_sharpe, kde=True, bins=30, label=f'OLS (Mean: {np.mean(ols_sharpe):.2f})', color='tomato', alpha=0.7, stat='density')
                sns.histplot(sp500_sharpe, kde=True, bins=30, label=f'S&P 500 (Mean: {np.mean(sp500_sharpe):.2f})', color='green', alpha=0.6, stat='density')
                
                plt.title('Distribution of Realized Sharpe Ratios on Test Set', fontsize=16)
                plt.xlabel('Sharpe Ratio', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                
                plot_path = save_path / 'test_1_sharpe_ratio_distribution.png'
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"샤프 비율 분포 그래프 저장 완료: {plot_path}")
            except Exception as e:
                print(f"샤프 비율 분포 그래프 생성 실패: {e}")

        # MDD 분포 (모델 vs OLS)
        if 'max_drawdown' in results_data and 'ols_drawdown' in results_data:
            try:
                model_mdd = results_data['max_drawdown'].cpu().numpy()
                ols_mdd = results_data['ols_drawdown'].cpu().numpy()
    
                plt.figure(figsize=(12, 7))
                sns.histplot(model_mdd, kde=True, bins=30, label=f'Model MDD (Mean: {np.mean(model_mdd):.3f})', color='darkcyan', stat='density')
                sns.histplot(ols_mdd, kde=True, bins=30, label=f'OLS MDD (Mean: {np.mean(ols_mdd):.3f})', color='orange', alpha=0.7, stat='density')
                
                plt.title('Distribution of Max Drawdown (MDD) on Test Set', fontsize=16)
                plt.xlabel('Max Drawdown (Lower is better)', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                
                plot_path = save_path / 'test_2_mdd_distribution.png'
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"최대 낙폭(MDD) 분포 그래프 저장 완료: {plot_path}")
            except Exception as e:
                print(f"최대 낙폭(MDD) 분포 그래프 생성 실패: {e}")

    

    def plot_portfolio_rationale(self, test_results, asset_names, sample_indices, save_path):
        """
        테스트 결과의 특정 샘플에 대한 포트폴리오 구성 근거를 시각화합니다.
        (기대 수익률 | 공분산 | 최종 가중치)
        """
        print("\n=== 포트폴리오 구성 근거 시각화 시작 ===")
        results_data = test_results['results']
        
        # 필요한 모든 데이터가 있는지 확인
        required_keys = ['expected_returns', 'ols_expected_returns', 'cov_matrix', 'weights', 'ols_weights']
        if not all(k in results_data for k in required_keys):
            print("시각화에 필요한 데이터(기대수익률, 공분산, 가중치)가 부족합니다.")
            return

        for idx in sample_indices:
            try:
                # 1. 특정 샘플(idx)의 데이터 추출
                model_returns = results_data['expected_returns'][idx].cpu().numpy()
                ols_returns = results_data['ols_expected_returns'][idx].cpu().numpy()
                cov_matrix = results_data['cov_matrix'][idx].cpu().numpy()
                model_weights = results_data['weights'][idx].cpu().numpy()
                ols_weights = results_data['ols_weights'][idx].cpu().numpy()

                # 2. 1x3 형태의 서브플롯 생성
                fig, axes = plt.subplots(1, 3, figsize=(24, 7))
                fig.suptitle(f'Portfolio Rationale for Test Sample #{idx}', fontsize=18, fontweight='bold')
                
                n_assets = len(asset_names)
                x = np.arange(n_assets)  # the label locations
                width = 0.35  # the width of the bars

                # --- [0] 기대 수익률 비교 ---
                ax1 = axes[0]
                rects1 = ax1.bar(x - width/2, model_returns, width, label='Model', color='royalblue')
                rects2 = ax1.bar(x + width/2, ols_returns, width, label='OLS', color='tomato')
                ax1.set_title('1. Expected Returns', fontsize=14)
                ax1.set_ylabel('Expected Daily Return', fontsize=12)
                ax1.set_xticks(x)
                ax1.set_xticklabels(asset_names, rotation=45, ha="right")
                ax1.legend()
                ax1.grid(axis='y', linestyle='--', alpha=0.7)

                # --- [1] 공분산 행렬 히트맵 ---
                ax2 = axes[1]
                sns.heatmap(cov_matrix, annot=False, fmt=".2e", cmap='coolwarm',
                            xticklabels=asset_names, yticklabels=asset_names, ax=ax2)
                ax2.set_title('2. Covariance Matrix', fontsize=14)
                ax2.tick_params(axis='x', rotation=45)
                ax2.tick_params(axis='y', rotation=0)

                # --- [2] 최종 가중치 비교 ---
                ax3 = axes[2]
                rects3 = ax3.bar(x - width/2, model_weights, width, label='Model', color='royalblue')
                rects4 = ax3.bar(x + width/2, ols_weights, width, label='OLS', color='tomato')
                ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1) # 0 기준선
                ax3.set_title('3. Final Portfolio Weights', fontsize=14)
                ax3.set_ylabel('Weight', fontsize=12)
                ax3.set_xticks(x)
                ax3.set_xticklabels(asset_names, rotation=45, ha="right")
                ax3.legend()
                ax3.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plot_path = save_path / f'test_rationale_sample_{idx}.png'
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"구성 근거 대시보드 저장 완료: {plot_path}")
            except Exception as e:
                print(f"구성 근거 대시보드 생성 실패 (Sample #{idx}): {e}")

    def plot_strategic_bias_analysis(self, test_results, asset_names, save_path):
        """
        테스트셋 전체 평균 데이터로 모델의 전략적 편향을 분석
        (평균 위험 프리미엄 | 평균 공분산 | 평균 상관관계 | 평균 가중치)
        """
        print("\n=== 모델 전략적 편향 분석 시각화 시작 ===")
        results_data = test_results['results']
        
        required_keys = ['expected_returns', 'ols_expected_returns', 'cov_matrix', 'weights', 'ols_weights', 'risk_free_rate']
        if not all(k in results_data for k in required_keys):
            print("전략 분석에 필요한 데이터가 부족합니다.")
            return

        try:
            # 1. 평균값 계산 (위험 프리미엄, 공분산, 가중치)
            rf_rate_expanded = results_data['risk_free_rate'].unsqueeze(1)
            model_risk_premium = results_data['expected_returns'] - rf_rate_expanded
            ols_risk_premium = results_data['ols_expected_returns'] - rf_rate_expanded
            avg_model_risk_premium = model_risk_premium.mean(dim=0).cpu().numpy()
            avg_ols_risk_premium = ols_risk_premium.mean(dim=0).cpu().numpy()
            
            avg_cov_matrix = results_data['cov_matrix'].mean(dim=0).cpu().numpy()
            avg_model_weights = results_data['weights'].mean(dim=0).cpu().numpy()
            avg_ols_weights = results_data['ols_weights'].mean(dim=0).cpu().numpy()
            
            # 평균 공분산으로부터 평균 상관관계 계산
            variances = np.diag(avg_cov_matrix)
            stds = np.sqrt(variances)
            stds[stds == 0] = 1e-8
            outer_product_stds = np.outer(stds, stds)
            avg_corr_matrix = avg_cov_matrix / outer_product_stds
            
            # ▼▼▼ [수정] 1x3 -> 1x4 서브플롯으로 변경 ▼▼▼
            fig, axes = plt.subplots(2, 2, figsize=(18, 16)) # 가로 크기 확장
            fig.suptitle('Strategic Bias Analysis (Averaged over Test Set)', fontsize=20, fontweight='bold')
            
            n_assets = len(asset_names)
            x = np.arange(n_assets)
            width = 0.35

            # --- [0] 평균 위험 프리미엄 비교 (수익률 관점) ---
            ax1 = axes[0,0]
            ax1.bar(x - width/2, avg_model_risk_premium, width, label='Model View', color='royalblue')
            ax1.bar(x + width/2, avg_ols_risk_premium, width, label='OLS View', color='tomato')
            ax1.set_title('1. Average Risk Premium', fontsize=16)
            ax1.set_ylabel('Average Daily Risk Premium', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(asset_names, rotation=45, ha="right")
            ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
            ax1.legend()
            ax1.grid(axis='y', linestyle='--', alpha=0.7)

            # --- [1] 평균 공분산 행렬 (위험 크기 관점) --- (새로 추가)
            ax2 = axes[0,1]
            sns.heatmap(avg_cov_matrix, annot=False, cmap='coolwarm',
                        xticklabels=asset_names, yticklabels=asset_names, ax=ax2,
                        cbar_kws={'label': 'Covariance'})
            ax2.set_title('2. Average Covariance Matrix', fontsize=16)
            ax2.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='y', rotation=0)

            # --- [2] 평균 상관관계 행렬 (위험 방향/강도 관점) ---
            ax3 = axes[1,0]
            sns.heatmap(avg_corr_matrix, annot=False, cmap='coolwarm',
                        xticklabels=asset_names, yticklabels=asset_names, ax=ax3, vmin=-1, vmax=1,
                        cbar_kws={'label': 'Correlation'})
            ax3.set_title('3. Average Correlation Matrix', fontsize=16)
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='y', rotation=0)

            # --- [3] 평균 포트폴리오 가중치 (최종 결론) ---
            ax4 = axes[1,1]
            ax4.bar(x - width/2, avg_model_weights, width, label='Model Allocation', color='royalblue')
            ax4.bar(x + width/2, avg_ols_weights, width, label='OLS Allocation', color='tomato')
            ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1)
            ax4.set_title('4. Average Portfolio Weights', fontsize=16)
            ax4.set_ylabel('Average Weight', fontsize=12)
            ax4.set_xticks(x)
            ax4.set_xticklabels(asset_names, rotation=45, ha="right")
            ax4.legend()
            ax4.grid(axis='y', linestyle='--', alpha=0.7)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.94])
            plot_path = save_path / 'test_strategic_bias_analysis_full.png' # 파일 이름 변경
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"전략적 편향 전체 분석 대시보드 저장 완료: {plot_path}")
        except Exception as e:
            print(f"전략적 편향 분석 대시보드 생성 실패: {e}")