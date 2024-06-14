# 신민석 20239469

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# 데이터 로딩 : 저는 종목코드를 컬럼으로 사용하기 위하여 필요없는 행은 전처리한 데이터를 활용하였습니다
data = pd.read_excel("Quant/processed_quant_mid.xlsx", sheet_name=None)

# 각 Sheet별 데이터 프레임 생성 및 인덱스 설정
stock_df = data['Sheet1'].set_index('Date')
four_factors_df = data['Sheet2'].set_index('Date')

#  상장폐지 수익률 -100% 처리를 위한 데이터 전처리
#  stock 과 4팩터 데이터의 각 열의 마지막 유효값 인덱스 확인 및 마지막 유효값 다음 값은 0으로 처리
last_valid_index_stock = stock_df.apply(lambda col: col.last_valid_index())
last_valid_index_market = four_factors_df.apply(lambda col: col.last_valid_index())

for column in stock_df.columns:
    last_valid_date = stock_df[column].last_valid_index()
    next_row_index = stock_df.index.get_loc(last_valid_date) + 1 if last_valid_date is not None else None
    if next_row_index is not None and next_row_index < len(stock_df):
        stock_df.at[stock_df.index[next_row_index], column] = 0

for column in four_factors_df.columns:
    last_valid_date = four_factors_df[column].last_valid_index()
    next_row_index = four_factors_df.index.get_loc(last_valid_date) + 1 if last_valid_date is not None else None
    if next_row_index is not None and next_row_index < len(four_factors_df):
        four_factors_df.at[four_factors_df.index[next_row_index], column] = 0


# 월별 수익률 계산
stock_returns = stock_df.pct_change()
four_factors_returns = four_factors_df.pct_change()

# 과거 N 개월 데이터를 사용하여 롤링 윈도우 방식으로 베타 구하기 
N = 12

market_mean_N_month = four_factors_returns['I.001'].rolling(window=N).mean()
market_var_N_month = four_factors_returns['I.001'].rolling(window=N).var()
stock_returns_N_month = stock_returns.rolling(window=N).mean()

def calculate_beta(stock_returns, market_returns, market_mean_N_month, market_var_N_month):
    covariance = (stock_returns - stock_returns.rolling(window=N).mean()) * (market_returns - market_mean_N_month)
    covariance = covariance.rolling(window=12).sum()
    beta = covariance / (12*market_var_N_month)
    return beta

betas = stock_returns.apply(lambda x: calculate_beta(x, four_factors_returns['I.001'], market_mean_N_month, market_var_N_month))

# 위에서 산출한 베타값을 기준으로 각 행별 베타 10분위로 나누기   
def assign_beta_decile_levels(betas):
    if betas.isna().all():  # 모든 값이 NaN인 경우, 변경 없이 반환
        return betas
    # 음수인 베타값을 제외하고 나머지 값에 대해 분위수를 할당
    betas_non_negative = betas.dropna()[betas.dropna() >= 0]
    if betas_non_negative.empty:
        return pd.Series(index=betas.index)  # 모든 값이 음수인 경우, 빈 Series를 반환
    labels = [f"D{i+1}" for i in range(10)] # D1은 저베타, D10이 고베타
    return pd.qcut(betas_non_negative, 10, labels=labels, duplicates='drop')
deciles = betas.apply(assign_beta_decile_levels, axis=1)

# 베타 각 분위수별로 평균 수익률 구하기
decile_returns = pd.DataFrame(index=stock_returns.index, columns=[f"D{i+1}" for i in range(10)]) # 저장할 데이터 프레임 선언
for date, row in deciles.iterrows():
    for decile in decile_returns.columns:
        selected_stocks = row[row == decile].index
        if not selected_stocks.empty:
            decile_returns.at[date, decile] = stock_returns.loc[date, selected_stocks].mean()
decile_returns = decile_returns.astype(float)

# 저베타 전략을 4팩터에 대한 회기 분석 수행 및 알파 구하기
factors = four_factors_returns[['I.001', '3FM.2B3.SMB', '3FM.2B3.HML', '3FM.2M3.MOM']].astype(float)
factors = sm.add_constant(factors)
# 베타 분위수별 회귀 분석 수행을 위한 함수 선언
def regress_and_get_alpha(returns, factors):
    model = sm.OLS(returns, factors, missing='drop').fit()
    return model.params[0]  # alpha is the intercept
# 베타 분위수별 회귀 분석 수행 후 알파 저장
decile_alphas = {}
for decile in decile_returns.columns:
    decile_alphas[decile] = regress_and_get_alpha(decile_returns[decile], factors)
decile_alphas_df = pd.DataFrame(list(decile_alphas.items()), columns=['Decile', 'Alpha'])


### BAB 전략 ###

#BAB 전략 수식 구현
z_i = betas.rank(axis=1)
z_bar = z_i.mean(axis=1)
deviations = z_i.subtract(z_bar, axis=0)
k = 2 / deviations.abs().sum(axis=1)
positive_deviations = deviations.clip(lower=0)
negative_deviations = deviations.clip(upper=0).abs()
w_H = positive_deviations.multiply(k, axis=0)
w_L = negative_deviations.multiply(k, axis=0)
r_f = 0 # 무위험 이자율 0으로 설정
# BAB 전략에 따른 levered, delevered weight
beta_L = (betas * w_L).sum(axis=1) / w_L.sum(axis=1)
beta_H = (betas * w_H).sum(axis=1) / w_H.sum(axis=1)
beta_data = pd.DataFrame({'beta_L': beta_L, 'beta_H': beta_H})
# BAB 수익률
return_L = (stock_returns.shift(-1) * w_L).sum(axis=1) / w_L.sum(axis=1) 
return_H = (stock_returns.shift(-1) * w_H).sum(axis=1) / w_H.sum(axis=1)
BAB_returns = (return_L - r_f) / beta_L - (return_H - r_f) / beta_H
cumulative_log_returns = np.log1p(BAB_returns).cumsum()
#4팩터 회귀 분석 수행
four_factors = four_factors_returns[['I.001', '3FM.2B3.SMB', '3FM.2B3.HML', '3FM.2M3.MOM']].astype(float)
four_factors_with_constant = sm.add_constant(four_factors)
BAB_regression_model = sm.OLS(BAB_returns, four_factors_with_constant, missing='drop').fit()


### 결과 출력 ###

# 저베타 전략  베타 분위수별 평균 수익률 및 알파 차트
plt.figure(1)
plt.bar(decile_returns.mean().index,decile_returns.mean().values)
plt.title('Beta decile returns')
plt.figure(2)
plt.bar(decile_alphas_df['Decile'], decile_alphas_df['Alpha'])
plt.title("Beta decile alpha")

#BAB 전략 수익률 
plt.figure(3)
plt.title('BaB Returns')
plt.plot(BAB_returns.index, BAB_returns.values)

# beta_L과 beta_H 비중 차트
plt.figure(4)
plt.title('Weight')
plt.plot(beta_data.index, beta_data['beta_H'])
plt.plot(beta_data.index, beta_data['beta_L']) 
plt.legend(["beta_H", "beta_L"])

# BAB 전략 로그 누적 수익률 차트
plt.figure(5)
plt.title('Cumulative returns')
plt.plot(cumulative_log_returns.index, cumulative_log_returns.values)
plt.show()

#BAB전략 4팩터 회귀 분석 결과 출력
print(BAB_regression_model.summary())