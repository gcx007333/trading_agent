from openbb import obb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 手动计算RSI函数
def calculate_rsi(prices, window=14):
    """手动计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 1. 获取公司概况
print("=== 公司概况 ===")
try:
    profile = obb.equity.profile("TSLA")
    print(profile.to_df())
except Exception as e:
    print(f"获取公司概况出错: {e}")

# 2. 获取历史股价数据
print("\n=== 股价数据 ===")
try:
    historical_data = obb.equity.price.historical("TSLA", start_date="2024-01-01", provider="yfinance")
    df = historical_data.to_df()
    print("数据列名:", df.columns.tolist())
    print(df.tail())
    
    # 绘制价格走势
    plt.figure(figsize=(10, 5))
    if 'date' in df.columns:
        plt.plot(df['date'], df['close'])
    elif 'datetime' in df.columns:
        plt.plot(df['datetime'], df['close'])
    else:
        plt.plot(df.index, df['close'])
    
    plt.title('TSLA Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tsla_price.png')
    plt.show()
    
except Exception as e:
    print(f"获取股价数据出错: {e}")

# 3. 使用手动计算的RSI
print("\n=== 技术指标 (RSI - 手动计算) ===")
try:
    rsi_manual = calculate_rsi(df['close'], window=14)
    print("手动计算的RSI:")
    print(rsi_manual.tail())
    
    # 绘制RSI图表
    plt.figure(figsize=(10, 5))
    if 'date' in df.columns:
        plt.plot(df['date'], rsi_manual)
    elif 'datetime' in df.columns:
        plt.plot(df['datetime'], rsi_manual)
    else:
        plt.plot(df.index, rsi_manual)
    
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    plt.title('TSLA RSI Indicator')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tsla_rsi.png')
    plt.show()
    
except Exception as e:
    print(f"计算RSI出错: {e}")

# 3. 获取关键财务指标（如市盈率）
tsla_ratios = obb.equity.fundamental.ratios("TSLA")
print(tsla_ratios.to_df().head())