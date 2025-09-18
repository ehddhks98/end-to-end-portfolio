import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinancialDataset(Dataset):
    """
    TCN 기반 포트폴리오 최적화를 위한 금융 데이터셋
    
    데이터 구조:
    - 자산 데이터: (data_dir 내의 *_daily_technical_data.csv 파일에 해당하는 모든 종목)의 일일 수익률
    - 시장 데이터: Fama-French 3요인 모델 (Mkt-RF, RF)
    - 시계열 슬라이딩 윈도우 방식으로 과거 데이터와 미래 수익률 생성
    """
    
    def __init__(self, data_dir, lookback=252, pred_horizon=21, split='train', normalize=False, rebalancing_frequency='monthly', train_end_date=None, val_end_date=None, conditional_cov=True, conditional_mkt_rf=True):
        """
        Args:
            data_dir (str): 데이터 디렉토리 경로
            lookback (int): 과거 데이터 길이 (일) - 기본값 252일 (1년)
            pred_horizon (int): 예측 기간 (일) - 기본값 21일 (1개월)
            split (str): 데이터 분할 ('train', 'val', 'test')
            normalize (bool): 데이터 정규화 여부
            rebalancing_frequency (str): 리밸런싱 주기 ('daily' 또는 'monthly')
        """
        self.data_dir = Path(data_dir)
        self.lookback = lookback
        self.pred_horizon = pred_horizon
        self.split = split
        self.normalize = normalize
        self.rebalancing_frequency = rebalancing_frequency        
        self.train_end_date = train_end_date
        self.val_end_date = val_end_date
        
        self.conditional_cov = conditional_cov
        self.conditional_mkt_rf = conditional_mkt_rf
        self.static_cov_matrix = None
        self.static_mean_mkt_rf = None

        self.asset_names = [] 
        self.num_assets = 0   
        
        # 정규화를 위한 스케일러
        self.asset_scaler = StandardScaler() if normalize else None
        self.market_scaler = StandardScaler() if normalize else None
        
        # 데이터 로드 및 전처리
        self._load_and_preprocess_data() # 이 함수 호출 시 self.asset_names와 self.num_assets가 채워짐
        
        # 데이터 분할
        self._create_data_split()
        
        if split == 'train':  # 훈련 데이터에서만 상세 정보 출력
            self._print_dataset_info()
            self._print_rebalancing_info()
    
    def _load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        if self.split == 'train':
            print(f"\n=== {self.split.upper()} 데이터 로딩 시작 ===")
        
        # 데이터 디렉토리 존재 확인
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")
        
        # 자산 데이터 로드 (asset_name을 동적으로 결정)
        asset_dfs, asset_names_found = self._load_asset_data()
        self.asset_names = asset_names_found
        self.num_assets = len(self.asset_names)

        if not self.asset_names:
            raise ValueError(f"데이터 디렉토리 '{self.data_dir}'에서 유효한 자산 데이터 파일(*_daily_technical_data.csv)을 찾을 수 없습니다.")
        
        # Fama-French 데이터 로드
        ff_data = self._load_fama_french_data()
        
        # 데이터 병합 및 정리
        self._merge_and_clean_data(asset_dfs, ff_data)
        
        # 데이터 품질 검증
        self._validate_data_quality()
        
        if self.split == 'train':
            print(f"{self.split.upper()} 데이터 로딩 완료")
    
    def _load_asset_data(self):
        """자산 데이터 로드"""
        
        # 데이터 디렉토리에서 *_daily_technical_data.csv 패턴의 파일 찾기
        asset_file_paths = list(self.data_dir.glob('*_daily_technical_data.csv'))
        
        if not asset_file_paths:
            return [], [] # 파일이 없으면 빈 리스트 반환
            
        asset_dfs = []
        found_asset_names = []
        
        for file_path in asset_file_paths:
            try:
                # 파일 이름에서 자산명 추출 (예: 'AAPL_daily_technical_data.csv' -> 'AAPL')
                name = file_path.stem.replace('_daily_technical_data', '')
                
                # CSV 파일 로드
                df = pd.read_csv(file_path)
                
                # 필수 컬럼 존재 확인
                required_cols = ['Date', 'Close']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    warnings.warn(f"'{name}' 데이터에 필수 컬럼이 없습니다: {missing_cols}. 이 자산은 제외됩니다.")
                    continue # 필수 컬럼 없으면 해당 자산 제외
                
                # 날짜 형식 변환
                df['Date'] = pd.to_datetime(df['Date'])
                
                # 수익률 계산
                # 첫 번째 수익률은 NaN이 되므로, 나중에 dropna() 시 제거됨
                df['Returns'] = df['Close'].pct_change() 
                
                # 데이터 정렬
                df = df.sort_values('Date').reset_index(drop=True)
                
                asset_dfs.append(df)
                found_asset_names.append(name)
                
                if self.split == 'train':
                    print(f"  {name}: {len(df)} 행, 기간: {df['Date'].min()} ~ {df['Date'].max()}")
                    
            except Exception as e:
                warnings.warn(f"'{file_path.name}' 데이터 로딩 중 오류 발생: {e}. 이 자산은 제외됩니다.")
                continue # 오류 발생 시 해당 자산 제외
        
        # 자산 이름을 알파벳 순으로 정렬
        sorted_assets = sorted(zip(found_asset_names, asset_dfs), key=lambda x: x[0])
        sorted_asset_names = [name for name, df in sorted_assets]
        sorted_asset_dfs = [df for name, df in sorted_assets]

        return sorted_asset_dfs, sorted_asset_names
    
    def _load_fama_french_data(self):
        ff_path = self.data_dir / 'F-F_Research_Data_Factors_daily.CSV'
        
        if not ff_path.exists():
            raise FileNotFoundError(f"Fama-French 데이터 파일을 찾을 수 없습니다: {ff_path}")
        
        try:
            # CSV 파일 로드
            ff_data = pd.read_csv(ff_path)
            
            # 필수 컬럼 존재 확인
            required_cols = ['Date', 'Mkt-RF', 'RF']
            missing_cols = [col for col in required_cols if col not in ff_data.columns]
            if missing_cols:
                raise ValueError(f"Fama-French 데이터에 필수 컬럼이 없습니다: {missing_cols}")
            
            # 날짜 형식 변환 (YYYYMMDD -> datetime)
            ff_data['Date'] = pd.to_datetime(ff_data['Date'].astype(str), format='%Y%m%d')
            
            # 퍼센트를 소수로 변환
            ff_data['Mkt-RF'] = ff_data['Mkt-RF'] / 100
            ff_data['RF'] = ff_data['RF'] / 100
            
            # 데이터 정렬
            ff_data = ff_data.sort_values('Date').reset_index(drop=True)
            
            if self.split == 'train':
                print(f"  Fama-French: {len(ff_data)} 행, 기간: {ff_data['Date'].min()} ~ {ff_data['Date'].max()}")
            
            return ff_data
            
        except Exception as e:
            raise RuntimeError(f"Fama-French 데이터 로딩 중 오류 발생: {e}")
    
    def _merge_and_clean_data(self, asset_dfs, ff_data):
        """데이터 병합 및 정리"""
        # 공통 날짜 찾기
        date_sets = [set(df['Date']) for df in asset_dfs]
        date_sets.append(set(ff_data['Date']))
        common_dates = sorted(list(set.intersection(*date_sets)))
        
        if len(common_dates) == 0:
            raise ValueError("자산 데이터와 Fama-French 데이터에 공통 날짜가 없습니다.")
        
        if self.split == 'train':
            print(f"  공통 날짜: {len(common_dates)}개 ({common_dates[0]} ~ {common_dates[-1]})")
        
        # 자산 수익률 데이터 정리
        asset_returns_list = [] # 리스트 이름을 변경하여 혼동 방지
        for df in asset_dfs:
            # 공통 날짜로 필터링 및 정렬
            filtered_df = df[df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
            
            # 수익률 추출 (NaN 제거)
            # pct_change()의 첫 번째 값은 NaN이므로, dropna()를 통해 해당 행이 제거됨
            # 모든 df에서 NaN이 제거된 후, 각 df의 길이가 동일한지 확인해야 함
            returns_series = filtered_df['Returns'].dropna() 
            if len(returns_series) != len(common_dates) - 1: # NaN 제거 후 길이가 (공통날짜 수 - 1)이어야 함
                warnings.warn(f"자산 데이터 '{self.asset_names[asset_dfs.index(df)]}'의 수익률 데이터 길이가 공통 날짜와 일치하지 않습니다. 데이터 품질을 확인하세요.")
            asset_returns_list.append(returns_series.values) # .values로 numpy 배열로 변환
        
        # 모든 자산의 수익률 길이가 동일한지 다시 확인
        if not asset_returns_list:
            raise ValueError("처리할 유효한 자산 수익률 데이터가 없습니다.")
            
        min_common_len = min(len(arr) for arr in asset_returns_list)
        if min_common_len == 0:
            raise ValueError("모든 자산의 수익률 데이터 길이가 0입니다. 데이터 로드/전처리를 확인하세요.")
        
        # 시장 데이터 정리
        ff_filtered = ff_data[ff_data['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

        # Fama-French 데이터도 asset_returns와 동일한 기간에 맞춰야 함
        # ff_data['Date']의 첫 번째 날짜는 수익률 계산에 사용되지 않으므로, ff_filtered도 첫 행을 제거
        market_data_aligned = ff_filtered[['Mkt-RF', 'RF']].iloc[1:].values # 첫 행 제거
        
        # 최종 데이터 길이 맞추기: asset_returns_list와 market_data_aligned 중 가장 짧은 길이에 맞춤
        min_len_overall = min(min_common_len, len(market_data_aligned))
        
        if min_len_overall < self.lookback + self.pred_horizon + 1: # 1은 수익률 계산으로 인한 첫 행 제외분
            raise ValueError(f"데이터가 부족합니다. 최소 {self.lookback + self.pred_horizon + 1}일 필요, 현재: {min_len_overall}일")
        
        # 최종 데이터 배열 생성
        self.asset_data = np.array([returns[:min_len_overall] for returns in asset_returns_list]).T  # (time, num_assets)
        self.market_data = market_data_aligned[:min_len_overall]  # (time, 2)
        
        # 날짜 배열 (asset_data와 market_data에 맞춰 첫 날짜를 제외)
        self.dates = [d for i, d in enumerate(common_dates) if i > 0][:min_len_overall]
        
        if self.split == 'train':
            print(f"  최종 데이터 형태: 자산 {self.asset_data.shape}, 시장 {self.market_data.shape}")
            print(f"  최종 데이터 기간: {self.dates[0]} ~ {self.dates[-1]}")
    
    def _validate_data_quality(self):
        """데이터 품질 검증"""
        # NaN 값 확인
        asset_nan_count = np.isnan(self.asset_data).sum()
        market_nan_count = np.isnan(self.market_data).sum()
        
        if asset_nan_count > 0 and self.split == 'train':
            print(f"자산 데이터에 NaN 값 {asset_nan_count}개 발견. 학습에 문제 발생 가능.")
        
        if market_nan_count > 0 and self.split == 'train':
            print(f"시장 데이터에 NaN 값 {market_nan_count}개 발견. 학습에 문제 발생 가능.")
        
        # 이상치 확인 (절댓값이 0.5보다 큰 일일 수익률)
        extreme_returns = np.abs(self.asset_data) > 0.5
        if extreme_returns.any() and self.split == 'train':
            extreme_count = extreme_returns.sum()
            print(f"극한 수익률 (|return| > 50%) {extreme_count}개 발견. 데이터 이상치 확인 필요.")
        
        # 데이터 통계 출력 (훈련 세트에서만)
        if self.split == 'train':
            print(f"  자산 수익률 통계:")
            for i, name in enumerate(self.asset_names):
                returns = self.asset_data[:, i]
                print(f"    {name}: 평균={returns.mean():.4f}, 표준편차={returns.std():.4f}, "
                f"최소={returns.min():.4f}, 최대={returns.max():.4f}")
    
    def _create_data_split(self):
        """데이터 분할 생성"""
        total_days = len(self.dates)
        if self.train_end_date and self.val_end_date:
            print("\n=== 날짜 기준으로 데이터 분할 ===")
            train_end_dt = pd.to_datetime(self.train_end_date)
            val_end_dt = pd.to_datetime(self.val_end_date)
            
            # self.dates에서 지정된 날짜 또는 그 이전의 가장 마지막 거래일 인덱스를 찾음
            train_end_idx = np.searchsorted(self.dates, train_end_dt, side='right') - 1
            val_end_idx = np.searchsorted(self.dates, val_end_dt, side='right') - 1

            if train_end_idx < 0: raise ValueError(f"Train end date '{self.train_end_date}'가 데이터 기간 이전에 있습니다.")
            if val_end_idx <= train_end_idx: raise ValueError("Validation end date는 Train end date 이후여야 합니다.")

            print(f"  훈련 종료 날짜: {self.train_end_date} (실제 데이터: {self.dates[train_end_idx].strftime('%Y-%m-%d')})")
            print(f"  검증 종료 날짜: {self.val_end_date} (실제 데이터: {self.dates[val_end_idx].strftime('%Y-%m-%d')})")
        else:
            print("\n=== 비율 기준으로 데이터 분할 (70/15/15) ===")
            train_end_idx = int(total_days * 0.70)
            val_end_idx = int(total_days * 0.85)

        if self.split == 'train':
            # 정적 공분산 행렬 계산 (선택 사항)
            if not self.conditional_cov:
                train_asset_data = self.asset_data[:train_end_idx]
                # 행렬 곱셈을 이용한 효율적인 공분산 계산
                mean_returns = train_asset_data.mean(axis=0, keepdims=True)
                centered_returns = train_asset_data - mean_returns
                T = centered_returns.shape[0]
                cov_matrix_np = (centered_returns.T @ centered_returns) / (T - 1)
                self.static_cov = torch.FloatTensor(cov_matrix_np)                

            # 정적 Mkt-RF 평균 계산 (선택 사항)
            if not self.conditional_mkt_rf:
                train_market_data = self.market_data[:train_end_idx]
                mean_mkt_rf = train_market_data[:, 0].mean() # Mkt-RF는 첫 번째 열
                self.static_mkt_rf = torch.FloatTensor([mean_mkt_rf])

        # 각 분할에서 유효한 인덱스 범위 계산
        # 인덱스는 lookback부터 시작하여, 미래 예측 기간을 제외한 곳까지
        if self.split == 'train':
            start_offset = self.lookback # 과거 데이터를 위한 오프셋
            end_limit = train_end_idx - self.pred_horizon # 미래 예측 기간을 위한 제한
            self.valid_indices = list(range(start_offset, end_limit))
        elif self.split == 'val':
            start_offset = train_end_idx + self.lookback # 훈련 끝 이후 lookback만큼 건너뛰기
            end_limit = val_end_idx - self.pred_horizon
            self.valid_indices = list(range(start_offset, end_limit))
        else:  # test
            start_offset = val_end_idx + self.lookback # 검증 끝 이후 lookback만큼 건너뛰기
            end_limit = total_days - self.pred_horizon
            self.valid_indices = list(range(start_offset, end_limit, self.pred_horizon)) # 리밸런싱 시 pred_horizon 간격으로 샘플링
        
        # 유효한 인덱스 생성
        if not self.valid_indices or max(self.valid_indices) >= total_days - self.pred_horizon: # 인덱스 범위 다시 확인
            warnings.warn(f"{self.split.upper()} 데이터셋에 유효한 샘플이 부족하거나 없습니다. 시작/종료 인덱스를 확인하세요.")
            self.valid_indices = [] # 유효한 샘플이 없으면 빈 리스트로 설정

        # 정규화 (훈련 데이터로만 학습)
        if self.normalize and self.split == 'train':
            self._fit_scalers()
        elif self.normalize and self.split in ['val', 'test']:
            # 검증/테스트 데이터는 변환만 수행 (훈련에서 학습된 스케일러 사용)
            pass
    
    def _fit_scalers(self):
        """정규화 스케일러 학습 (훈련 데이터만)"""
        if self.asset_scaler is not None:
            # 훈련 데이터 전체 기간을 사용하여 스케일러 학습 (lookback 고려 없이)
            train_asset_data_for_scaler = self.asset_data[:int(len(self.dates) * 0.70)]
            self.asset_scaler.fit(train_asset_data_for_scaler)
        
        if self.market_scaler is not None:
            train_market_data_for_scaler = self.market_data[:int(len(self.dates) * 0.70)]
            self.market_scaler.fit(train_market_data_for_scaler)
    
    def _print_dataset_info(self):
        """데이터셋 정보 출력"""
        print(f"\n=== 데이터셋 정보 ===")
        print(f"전체 기간: {self.dates[0]} ~ {self.dates[-1]} ({len(self.dates)}일)")
        print(f"자산 수: {self.num_assets} ({', '.join(self.asset_names)})")
        print(f"시계열 길이: {self.lookback}일")
        print(f"예측 기간: {self.pred_horizon}일")
        print(f"정규화: {'Yes' if self.normalize else 'No'}")
        
        # 분할별 샘플 수 계산 (간단한 계산으로 대체)
        total_days = len(self.dates)
        train_end_idx = int(total_days * 0.70)
        val_end_idx = int(total_days * 0.85)

        train_samples = max(0, train_end_idx - self.lookback - self.pred_horizon)
        val_samples = max(0, val_end_idx - train_end_idx - self.lookback - self.pred_horizon)
        test_samples = max(0, total_days - val_end_idx - self.lookback - self.pred_horizon)
        
        print(f"데이터 분할:")
        print(f"  훈련: {train_samples:,}개 샘플")
        print(f"  검증: {val_samples:,}개 샘플") 
        print(f"  테스트: {test_samples:,}개 샘플")
        print(f"  현재 분할({self.split}): {len(self.valid_indices):,}개 샘플")
    
    def _print_rebalancing_info(self):
        """리밸런싱 정보 출력"""
        print(f"\n=== 리밸런싱 설정 정보 ===")
        if self.rebalancing_frequency == 'monthly':
            print(f"월별 리밸런싱 설정:")
            print(f"  - 예측 기간: {self.pred_horizon}일 (약 {self.pred_horizon/21:.1f}개월)")
            print(f"  - 한 달 거래일: 약 21일")
            print(f"  - 연간화 기준: 12개월")
            if self.pred_horizon < 21:
                print(f" 예측 기간이 한 달(21일)보다 짧습니다. 스케일링을 통해 월간 수익률로 변환됩니다.")
            elif self.pred_horizon >= 21:
                n_months = self.pred_horizon // 21
                print(f"  ✓ 예측 기간에서 {n_months}개의 완전한 월간 데이터를 계산할 수 있습니다.")
        else:
            print(f"일별 리밸런싱 설정:")
            print(f"  - 예측 기간: {self.pred_horizon}일")
            print(f"  - 연간화 기준: 252거래일")
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        데이터 샘플 반환
        """
        if idx >= len(self.valid_indices):
            raise IndexError(f"인덱스 {idx}가 데이터셋 크기 {len(self.valid_indices)}를 초과합니다.")
        
        t = self.valid_indices[idx] 
        
        asset_hist = self.asset_data[t-self.lookback:t]
        market_hist = self.market_data[t-self.lookback:t]
        
        future_rets = self.asset_data[t:t+self.pred_horizon]
        future_common_rets = self.market_data[t:t+self.pred_horizon]
        
        if self.normalize:
            if self.asset_scaler is not None:
                asset_hist = self.asset_scaler.transform(asset_hist)
            if self.market_scaler is not None:
                market_hist = self.market_scaler.transform(market_hist)
        
        asset_data = torch.FloatTensor(asset_hist).unsqueeze(-1)   

        common_data = torch.FloatTensor(market_hist)
        future_returns = torch.FloatTensor(future_rets)
        future_common_data = torch.FloatTensor(future_common_rets)
        
        if torch.isnan(asset_data).any() or torch.isnan(common_data).any() or torch.isnan(future_returns).any():
            warnings.warn(f"샘플 {idx} (인덱스 {t})에서 NaN 값 발견. 모델 학습에 영향을 줄 수 있습니다.")
        
        return {
            'asset_data': asset_data,
            'common_data': common_data, 
            'future_returns': future_returns,
            'future_common_data': future_common_data
        }


def create_dataloaders(data_dir, batch_size=32, lookback=252, pred_horizon=21, normalize=False, 
                num_workers=4, pin_memory=True, rebalancing_frequency='monthly', device='cuda', train_end_date=None, val_end_date=None, conditional_cov=True, conditional_mkt_rf=True):
    """
    훈련/검증/테스트 데이터 로더 생성
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        batch_size (int): 배치 크기
        lookback (int): 과거 데이터 길이 (일)
        pred_horizon (int): 예측 기간 (일)
        normalize (bool): 데이터 정규화 여부
        num_workers (int): 데이터 로딩 워커 수
        pin_memory (bool): GPU 메모리 고정 여부
        rebalancing_frequency (str): 리밸런싱 주기 ('daily' 또는 'monthly')
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"데이터 로더 생성 중...")
    
    try:
        # 데이터셋 생성
        train_dataset = FinancialDataset(
            data_dir=data_dir,
            lookback=lookback,
            pred_horizon=pred_horizon,
            split='train',
            normalize=normalize,
            rebalancing_frequency=rebalancing_frequency,
            train_end_date=train_end_date, 
            val_end_date=val_end_date,
            conditional_cov=conditional_cov,
            conditional_mkt_rf=conditional_mkt_rf
        )
        
        if not conditional_cov:
            static_cov = train_dataset.static_cov

        if not conditional_mkt_rf:
            static_mkt_rf = train_dataset.static_mkt_rf
            
        val_dataset = FinancialDataset(
            data_dir=data_dir,
            lookback=lookback,
            pred_horizon=pred_horizon,
            split='val',
            normalize=normalize,
            rebalancing_frequency=rebalancing_frequency,
            train_end_date=train_end_date, 
            val_end_date=val_end_date
        )
        
        test_dataset = FinancialDataset(
            data_dir=data_dir,
            lookback=lookback,
            pred_horizon=pred_horizon,
            split='test',
            normalize=normalize,
            rebalancing_frequency=rebalancing_frequency,
            train_end_date=train_end_date, 
            val_end_date=val_end_date
        )
        val_dataset.static_cov = static_cov
        val_dataset.static_mkt_rf = static_mkt_rf
        test_dataset.static_cov = static_cov
        test_dataset.static_mkt_rf = static_mkt_rf

        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # 훈련은 셔플
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False  # 마지막 배치도 사용
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 검증/테스트는 순서 유지
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        # 데이터 형태 검증
        sample_batch = next(iter(train_loader))
        
        # 로더 통계 (간소화)
        print(f"데이터 로더 생성 완료:")
        print(f"  훈련: {len(train_dataset):,} 샘플")
        print(f"  검증: {len(val_dataset):,} 샘플")
        print(f"  테스트: {len(test_dataset):,} 샘플")
        print(f"  데이터 형태: {sample_batch['asset_data'].shape}, {sample_batch['common_data'].shape}, {sample_batch['future_returns'].shape}")
        
        return train_loader, val_loader, test_loader, static_cov, static_mkt_rf
        
    except Exception as e:
        print(f"데이터 로더 생성 실패: {e}")
        raise


def test_dataloaders(data_dir='data'):
    """데이터 로더 테스트 함수"""
    print("데이터 로더 테스트 시작...")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            lookback=20,
            pred_horizon=5
        )
        
        # 각 로더에서 샘플 추출 테스트
        for name, loader in [('훈련', train_loader), ('검증', val_loader), ('테스트', test_loader)]:
            print(f"\n{name} 로더 테스트:")
            batch = next(iter(loader))
            
            print(f"  자산 데이터: {batch['asset_data'].shape}")
            print(f"  시장 데이터: {batch['common_data'].shape}")
            print(f"  미래 수익률: {batch['future_returns'].shape}")
            print(f"  NaN 확인: 자산={torch.isnan(batch['asset_data']).sum()}, "
                f"시장={torch.isnan(batch['common_data']).sum()}, "
                f"미래={torch.isnan(batch['future_returns']).sum()}")
        
        print("\n모든 테스트 통과!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        raise


if __name__ == "__main__":
    # 테스트 실행
    test_dataloaders()