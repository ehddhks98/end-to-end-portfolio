import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용하지 않도록 설정
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import time
import importlib
from dataloader import create_dataloaders
from trainer import PortfolioTrainer
import json
import pandas as pd
import random


def set_seed(seed):
    """재현성을 위해 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='TCN 기반 적응형 포트폴리오 최적화 훈련')    
    
    try:
        script_dir = Path(__file__).parent
    except NameError:
        script_dir = Path.cwd()
        
    default_data_dir = script_dir / 'data'
    
    # 데이터 관련 인수
    parser.add_argument('--data_dir', type=str, default=str(default_data_dir), help='데이터 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lookback', type=int, default=252, help='과거 데이터 길이 (일)')
    parser.add_argument('--pred_horizon', type=int, default=21, help='예측 기간 (일)')
    parser.add_argument('--normalize', action='store_true', default=False, help='데이터 정규화 활성화')
    parser.add_argument('--rebalancing_frequency', type=str, default='monthly', choices=['daily', 'monthly'])
    parser.add_argument('--train_end_date', type=str, default=None, help='훈련 데이터 종료 날짜 (YYYY-MM-DD), 기본값: None')
    parser.add_argument('--val_end_date', type=str, default=None, help='검증 데이터 종료 날짜 (YYYY-MM-DD), 기본값: None')

    # 모델 관련 인수
    parser.add_argument('--model', type=str, default='model_sharpe', help='사용할 모델 파일 이름')
    parser.add_argument('--hidden_size', type=int, default=4, help='TCN의 은닉 상태 크기')
    parser.add_argument('--num_channels', type=int, nargs='+', default=[8, 16, 16, 32, 32, 64], help='TCN의 각 블록별 채널 수 리스트')
    parser.add_argument('--kernel_size', type=int, default=3, help='TCN 컨볼루션 커널 크기')
    parser.add_argument('--dropout', type=float, default=0.2, help='드롭아웃 비율')
    parser.add_argument('--risk_aversion', type=int, default=4, help='위험 회피도')
    parser.add_argument('--conditional_cov', action='store_true', help='훈련 세트 전체의 정적 공분산 행렬을 사용합니다.')
    parser.add_argument('--conditional_mkt_rf', action='store_true', help='훈련 세트 전체의 정적 Mkt-RF 평균을 사용합니다.')
    
    # 훈련 관련 인수
    parser.add_argument('--epochs', type=int, default=30, help='총 훈련 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='초기 학습률')
    parser.add_argument('--device', type=str, default='cuda', help='훈련 디바이스 (cuda, cpu)')
    parser.add_argument('--save_epochs', type=int, nargs='*', default=[10, 30],  help='저장할 특정 에폭 번호 리스트')
    parser.add_argument('--seed', type=int, default=42, help= '시드 번호')
    return parser.parse_args()

def get_model_class(model_name):    
    try:
        if model_name == 'model_var':
            from src.models.var_model import PortfolioOptimizer
        elif model_name == 'model_sharpe':
            from src.models.sharpe_model import PortfolioOptimizer
        elif model_name == 'model_kl':
            from src.models.kl_model import PortfolioOptimizer
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        return PortfolioOptimizer
        
    except (ImportError, AttributeError) as e:
        raise ImportError(f"모델 '{model_name}' 또는 그 안의 'PortfolioOptimizer' 클래스를 찾을 수 없습니다: {e}")


def main():
    """메인 훈련 함수"""
    try:
        # 인수 파싱
        args = parse_arguments()
        set_seed(args.seed)
        print("=" * 60)
        print(f"{args.model.upper()} 기반 적응형 포트폴리오 최적화 훈련 시작")
        print("=" * 60)
        
        # 데이터 로더 생성
        train_loader, val_loader, test_loader, static_cov, static_mkt_rf = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            lookback=args.lookback,
            pred_horizon=args.pred_horizon,
            normalize=args.normalize,
            rebalancing_frequency=args.rebalancing_frequency,
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,
            pin_memory=torch.cuda.is_available(),
            train_end_date=args.train_end_date,
            val_end_date=args.val_end_date,
            conditional_cov=args.conditional_cov,
            conditional_mkt_rf=args.conditional_mkt_rf
        )
        
        num_assets = train_loader.dataset.num_assets
        print(f"자산 개수: {num_assets}")

        # 모델 클래스 동적 import
        model_class = get_model_class(args.model)
        print(f"사용할 모델: {args.model}")

        # 디바이스 설정
        device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        if args.device == 'cuda' and device == 'cpu':
            print("CUDA가 사용 불가능하여 CPU로 전환합니다.")
        print(f"사용할 디바이스: {device}")

        # 모델 초기화
        model = model_class(
            num_assets=num_assets,
            hidden_size=args.hidden_size,
            num_channels=args.num_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            risk_aversion=args.risk_aversion
        )
        
        # 트레이너 초기화
        trainer = PortfolioTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            model_name=args.model,
            save_epochs=args.save_epochs,
            static_cov=static_cov,
            static_mkt_rf=static_mkt_rf,
        )
        config_path = trainer.checkpoint_dir / 'config.json'
        args_dict = vars(args)
        with open(config_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
        
        print(f"설정 파일 저장 완료: {config_path}")

        # 훈련 시작
        print(f"\n=== 훈련 시작 (에포크: {args.epochs}) ===")
        # trainer.train()은 모든 기록이 담긴 딕셔너리를 반환
        history = trainer.train()
        
        # === 훈련 요약 (수정된 부분) ===
        print("\n" + "=" * 60)
        print("=== 훈련 최종 요약 ===")
        print("=" * 60)

        if history['val_sharpe']:
            # 최고 검증 샤프 지수를 기록한 에포크 찾기
            best_epoch_idx = np.argmax(history['val_sharpe'])
            best_model_sharpe = history['val_sharpe'][best_epoch_idx]
            best_ols_sharpe_at_that_time = history['val_ols_sharpe'][best_epoch_idx]
            
            print(f"최고 성과 에포크: {best_epoch_idx + 1}")
            print(f"  - 모델 최고 검증 샤프 비율: {best_model_sharpe:.4f}")
            print(f"  - 당시 OLS 벤치마크 샤프 비율: {best_ols_sharpe_at_that_time:.4f}")
            
            # 모델과 OLS 벤치마크의 성능 차이 계산
            performance_gain = best_model_sharpe - best_ols_sharpe_at_that_time
            print(f"  - OLS 대비 성능 향상: {performance_gain:+.4f}")

        # 최종 에포크 결과 출력
        print("\n최종 에포크 결과:")
        print(f"  - 최종 훈련 손실: {history['train_losses'][-1]:.4f}")
        print(f"  - 최종 검증 손실: {history['val_losses'][-1]:.4f}")
        print(f"  - 최종 검증 샤프 (모델): {history['val_sharpe'][-1]:.4f}")
        print(f"  - 최종 검증 샤프 (OLS):  {history['val_ols_sharpe'][-1]:.4f}")

        try:            
            history_df = pd.DataFrame(history)            
            history_df.insert(0, 'epoch', range(1, len(history_df) + 1))
            history_path = trainer.checkpoint_dir / 'training_history.csv'
            history_df.to_csv(history_path, index=False)        
            print(f"\n훈련 히스토리 저장 완료: {history_path}")
            
        except Exception as e:
            print(f"\n훈련 히스토리 저장 실패: {e}")
        
        print(f"\n훈련 완료! 결과가 {trainer.checkpoint_dir}에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n훈련 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()