# 신민석 20239469

import pandas as pd
import numpy as np

# 데이터 로딩 : 저는 종목코드를 컬럼으로 사용하기 위하여 필요없는 행은 전처리한 데이터를 활용하였습니다
data = pd.read_excel("Quant/processed_quant_mid.xlsx")
data.set_index('Date', inplace=True)

#  상장폐지 수익률 -100% 처리를 위한 데이터 전처리
#  데이터 각 열의 마지막 유효값 인덱스 확인 및 마지막 유효값 다음 값은 0으로 처리
last_valid_index = data.apply(lambda col: col.last_valid_index())
for column in data.columns:
    last_valid_date = data[column].last_valid_index()
    next_row_index = data.index.get_loc(last_valid_date) + 1 if last_valid_date is not None else None
    if next_row_index is not None and next_row_index < len(data):
        data.at[data.index[next_row_index], column] = 0

# 월별 수익률
monthly_return = data.pct_change()

# J, K 기간 리스트 생성
J = [3,6,9,12] 
K = [3,6,9,12]

# 각 종목 과거 J개월의 수익률
lookback_returns = {}
for period in J:
    lookback_returns[period] = (1 + monthly_return).rolling(window=period).apply(np.prod, raw=True) - 1


# 상위 30%, 하위 30% 종목 포트폴리오 구성 및  Holding period 수익률 구하기 위한 함수 
def calculate_future_returns(data, past_returns, holding_period):
    future_returns = data.rolling(window=holding_period).mean()
    top_30_returns = future_returns[past_returns >= past_returns.quantile(0.7)].mean(axis=1)
    bottom_30_returns = future_returns[past_returns <= past_returns.quantile(0.3)].mean(axis=1)
    return top_30_returns, bottom_30_returns

# J,K조합별 상위 30% 포트폴리오 수익률 저장할 딕셔너리 생성
buy_portfolio_returns = {}
# 하위 30% 포트폴리오 수익률 저장할 딕셔너리 생성
sell_portfolio_returns = {}
# 포트폴리오 평균 수익률을 데이터 프레임으로 변환하기 위한 리스트 생성
results = []

# J,K 조합에 따른 평균 수익률 계산 및 buy, sell, buy-sell 포트폴리오별로 result리스트에 저장
for j in J:
    for k in K:
        buy_portfolio_returns[(j, k)] = [] 
        sell_portfolio_returns[(j, k)] = []
        top_30_returns, bottom_30_returns = calculate_future_returns(monthly_return, lookback_returns[j], k)
        buy_portfolio_returns[(j, k)].append(top_30_returns)
        sell_portfolio_returns[(j, k)].append(bottom_30_returns)

        buy_returns_df = pd.DataFrame(buy_portfolio_returns[(j, k)]).T.mean(axis=1)
        sell_returns_df = pd.DataFrame(sell_portfolio_returns[(j, k)]).T.mean(axis=1)
        buy_avg_return = buy_returns_df.mean()
        sell_avg_return = sell_returns_df.mean()
        buy_minus_sell_avg_return = buy_avg_return - sell_avg_return

        if buy_avg_return:
            results.append({'J': j, 'K': k, 'Portfolio': 'buy', 'Return': buy_avg_return})
        if sell_avg_return:
            results.append({'J': j, 'K': k, 'Portfolio': 'sell', 'Return': sell_avg_return})
        if buy_minus_sell_avg_return : 
            results.append({'J': j, 'K': k, 'Portfolio': 'buy-sell', 'Return': buy_minus_sell_avg_return})
    

#results를  데이터프레임으로 변환 및 최종 결과 출력
df_results = pd.DataFrame(results)
print(df_results.pivot_table(index = ['J', 'Portfolio' ],  columns='K'))
