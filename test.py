import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import time
import importlib
from dataloader import create_dataloaders
from trainer import PortfolioTrainer
import json
import random


def set_seed(seed):
    """재현성을 위해 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def parse_arguments():    
    parser = argparse.ArgumentParser(description='TCN 기반 적응형 포트폴리오 최적화 테스트')
    
    # 스크립트 위치 기준으로 기본 데이터 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'data')

    # 데이터 관련 argument
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='데이터 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lookback', type=int, default=252, help='과거 데이터 길이 (일)')
    parser.add_argument('--pred_horizon', type=int, default=21, help='예측 기간 (일)')
    parser.add_argument('--normalize', action='store_true', default=False, help='데이터 정규화 활성화')
    parser.add_argument('--rebalancing_frequency', type=str, default='monthly', choices=['daily', 'monthly'])
    parser.add_argument('--train_end_date', type=str, default=None, help='훈련 데이터 종료 날짜 (YYYY-MM-DD), 기본값: None')
    parser.add_argument('--val_end_date', type=str, default=None, help='검증 데이터 종료 날짜 (YYYY-MM-DD), 기본값: None')
    
    # 모델 관련 argument
    parser.add_argument('--model', type=str, default='model_sharpe')
    parser.add_argument('--model_path', type=str, default='/home/dongwan/Project/TCN_Portfolio/checkpoint/20250807_042417_model_var/best_model.pth', help='best_model.pth 파일 경로')
    parser.add_argument('--hidden_size', type=int, default=64, help='TCN의 은닉 상태 크기')
    parser.add_argument('--num_channels', type=int, nargs='+', default=[8, 16, 32], help='TCN의 각 블록별 채널 수 리스트')
    parser.add_argument('--kernel_size', type=int, default=3, help='TCN 컨볼루션 커널 크기')
    parser.add_argument('--dropout', type=float, default=0.5, help='드롭아웃 비율')
    parser.add_argument('--risk_aversion', type=int, default=4, help='위험 회피도')
    
    # 평가 관련 argument
    parser.add_argument('--device', type=str, default='cuda', help='평가 디바이스 (cuda, cpu)')
    parser.add_argument('--epochs', type=int, default=1, help='에포크 수 (테스트용)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='학습률 (테스트용)')
    parser.add_argument('--seed', type=int, default=42, help= '시드 번호')
    return parser.parse_args()

def get_model_class(model_name):    
    try:
        # train.py와 동일하게, 모델 별명과 실제 import 경로를 매핑합니다.
        if model_name == 'model_var':
            from src.models.var_model import PortfolioOptimizer
        elif model_name == 'model_sharpe':
            from src.models.sharpe_model import PortfolioOptimizer
        elif model_name == 'model_kl':
            from src.models.kl_model import PortfolioOptimizer
        else:
            # 지원하지 않는 모델 이름일 경우 에러 발생
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
        
        return PortfolioOptimizer # 매핑된 클래스를 반환
        
    except (ImportError, AttributeError, ValueError) as e:
        # ImportError: 파일을 찾지 못할 때
        # AttributeError: 파일 안에 PortfolioOptimizer 클래스가 없을 때
        # ValueError: if/elif 문에서 정의되지 않은 모델일 때
        raise ImportError(f"모델 '{model_name}'을 로드하는 데 실패했습니다: {e}")


def main():
    try:
        args = parse_arguments()

        print("=" * 60)
        print("TCN 기반 포트폴리오 최적화 모델 테스트")
        print("=" * 60)
        
        # 모델 경로 확인
        model_path = Path(args.model_path)
        if model_path.is_dir():
            model_path = model_path / 'best_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        print(f"모델 경로: {model_path}")

        # config.json 파일 경로 추정 (모델과 같은 폴더에 있어야 함)
        config_path = model_path.parent / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일(config.json)을 모델과 같은 경로에서 찾을 수 없습니다: {config_path}")
            
        print(f"모델 경로: {model_path}")
        print(f"설정 파일 경로: {config_path}")
        
        # 저장된 설정(config.json) 파일 불러오기
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("\n=== 훈련 시 사용된 설정 불러오기 ===")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 체크포인트의 부모 폴더 이름(e.g., 'model_var_20250807_042417')
        checkpoint_folder_name = f"{model_path.parent.name}_{model_path.stem}"
        
        # 결과 저장 폴더 경로를 재구성 및 체크포인트의 이름을 그대로 사용
        results_save_path = Path.cwd() / 'result' / f"Test_{checkpoint_folder_name}"
        results_save_path.mkdir(parents=True, exist_ok=True)
        print(f"\n테스트 결과 저장 경로: {results_save_path}")
        
        # 디바이스 설정
        device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        if args.device == 'cuda' and device == 'cpu':
            print("CUDA가 사용 불가능하여 CPU로 전환합니다.")
        print(f"사용 디바이스: {device}")

        # 데이터 로더 생성
        print("\n=== 데이터 로딩 ===")
        _, _, test_loader, static_cov, static_mkt_rf = create_dataloaders(
            data_dir=args.data_dir, batch_size=args.batch_size, lookback=args.lookback,
            pred_horizon=args.pred_horizon, normalize=args.normalize,
            rebalancing_frequency=args.rebalancing_frequency,
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,
            pin_memory=(device == 'cuda'),
            train_end_date=config.get('train_end_date', None),
            val_end_date=config.get('val_end_date', None),
            conditional_cov=config.get('conditional_cov', False),
            conditional_mkt_rf=config.get('conditional_mkt_rf', False)

        )
        
        num_assets = test_loader.dataset.num_assets
        print(f"감지된 자산 개수: {num_assets}")

        # 모델 초기화
        model_class = get_model_class(config['model'])
        print(f"사용 모델: {config['model']}")
        model = model_class(
            num_assets=num_assets, 
            hidden_size=config['hidden_size'],
            num_channels=config['num_channels'], 
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            risk_aversion=config['risk_aversion']            
        ).to(device)

        # 모델 체크포인트 로드
        print("\n=== 모델 체크포인트 로드 ===")
        checkpoint = torch.load(model_path, map_location=device)
        
        # state_dict가 딕셔너리 안에 있는지, 아니면 그 자체인지 확인하여 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint: print(f"  > 훈련된 에포크: {checkpoint['epoch']}")
            if 'val_sharpe' in checkpoint: print(f"  > 최종 검증 샤프 (모델): {checkpoint.get('val_sharpe', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"모델 로드 완료: {model_path}")
        
        config['tested_checkpoint_path'] = str(model_path.resolve())
        test_config_save_path = results_save_path / 'config.json'
        with open(test_config_save_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"\n 테스트에 사용된 설정 파일 저장 완료: {test_config_save_path}")

        # Trainer 초기화
        trainer = PortfolioTrainer(
            model=model, train_loader=None, val_loader=None, test_loader=test_loader,
            num_epochs=1, learning_rate=0, device=device, model_name=None,
            static_cov=static_cov,
            static_mkt_rf=static_mkt_rf
        )
        asset_names = trainer.test_loader.dataset.asset_names

        # 테스트 실행
        print(f"\n=== 테스트 시작 (총 배치 수: {len(test_loader)}) ===")
        test_results = trainer.test()
        
        # 테스트 성능 요약 (OLS 비교 추가)
        print("\n=== 테스트 성능 요약 ===")
        performance_summary = {
            'Average Loss': test_results.get('avg_loss', float('nan')),
            'Average Sharpe (Model)': test_results.get('avg_model_sharpe', float('nan')),
            'Average Sharpe (OLS)': test_results.get('avg_ols_sharpe', float('nan')),
            'Average Sharpe (SP500)': test_results.get('avg_sp500_sharpe', float('nan')),
            'Average MDD (Model)': test_results.get('avg_model_mdd'), # MDD 요약 추가
            'Average MDD (OLS)': test_results.get('avg_ols_mdd'), 
            'Average Returns (Model)': test_results.get('avg_returns', float('nan'))
        }
        summary_stats = pd.DataFrame([performance_summary]).T
        summary_stats.columns = ['Value']
        summary_stats['Value'] = summary_stats['Value'].map('{:,.4f}'.format)
        print(summary_stats)

        # 결과 저장 및 시각화
        print("\n=== 결과 저장 및 시각화 ===")
        try:
            results_data = test_results['results']
            model_sharpe_np = results_data['realized_sharpe_ratio'].cpu().numpy()
            ols_sharpe_np = results_data['ols_sharpe_ratio'].cpu().numpy()
            sp500_shrpe_np = results_data['sp500_sharpe_ratio'].cpu().numpy()
            weights_np = results_data['weights'].cpu().numpy()
            returns_np = results_data['realized_return'].cpu().numpy()

            # 데이터프레임 생성
            summary_df = pd.DataFrame({
                'model_realized_sharpe': model_sharpe_np,
                'ols_realized_sharpe': ols_sharpe_np,
                'sp500_realized_sharpe': sp500_shrpe_np, 
                'model_max_drawdown': results_data['max_drawdown'].cpu().numpy(),
                'ols_max_drawdown': results_data['ols_drawdown'].cpu().numpy()
            })
            weights_df = pd.DataFrame(weights_np, columns=[f'weight_{i+1}' for i in range(weights_np.shape[1])])
            returns_df = pd.DataFrame(returns_np, columns=[f'return_day_{i+1}' for i in range(returns_np.shape[1])])

            # CSV 파일로 저장
            summary_df.to_csv(results_save_path / 'test_performance_summary.csv', index=False)
            weights_df.to_csv(results_save_path / 'test_portfolio_weights.csv', index=False)
            returns_df.to_csv(results_save_path / 'test_daily_returns.csv', index=False)

            print(f"테스트 요약 저장 완료: {results_save_path / 'test_performance_summary.csv'}")
            print(f"포트폴리오 가중치 저장 완료: {results_save_path / 'test_portfolio_weights.csv'}")
            print(f"일별 실현 수익률 저장 완료: {results_save_path / 'test_daily_returns.csv'}")

            # 성능 분석 시각화 함수 호출 
            trainer.plot_performance_analysis_test(test_results, results_save_path)
            sample_indices_to_plot = [0, 10, 20]
            """
            trainer.plot_portfolio_rationale(
                test_results=test_results,
                asset_names=asset_names, 
                sample_indices=sample_indices_to_plot,
                save_path=results_save_path
            )
            """
            trainer.plot_strategic_bias_analysis(
                test_results=test_results,
                asset_names=asset_names,
                save_path=results_save_path
            )         

        except Exception as e:
            print(f"결과 저장 또는 시각화 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n모든 테스트 절차 완료. 결과는 다음 경로에 저장되었습니다:\n{results_save_path}")
        
    except Exception as e:
        print(f"\n테스트 실행 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()