import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import numpy as np
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="劉總裁選股系統", layout="wide")

# --- 核心邏輯：爬取股票清單 ---
@st.cache_data(ttl=86400)
def get_taiwan_stock_list():
    """從證交所 ISIN 網站抓取上市股票清單"""
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    res = requests.get(url)
    df = pd.read_html(res.text)[0]
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    # 過濾出股票 (格式為 "代碼 名稱")
    df = df[df['有價證券代號及名稱'].str.contains('  ')]
    stock_list = df['有價證券代號及名稱'].tolist()
    # 提取產業分類
    industries = sorted(list(set(df['產業別'].dropna().tolist())))
    return df, industries

# --- 核心邏輯：技術指標計算 ---
def calculate_indicators(df):
    if len(df) < 35:  # 確保有足夠數據計算 MACD (26+9)
        return None
    
    # 均線
    df['SMA15'] = df['Close'].rolling(window=15).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # 布林通道
    std = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + 2 * std
    df['Lower'] = df['SMA20'] - 2 * std
    
    # MACD 計算
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

# --- 核心邏輯：篩選策略 ---
def check_strategies(df, strategies):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    results = []
    
    # 策略八：MACD 緩步爬升 (這是您 Notebook 的核心)
    if "S8" in strategies:
        cond = (last['MACD'] > prev['MACD'] > prev2['MACD']) and \
               (last['MACD'] > last['Signal']) and \
               (last['Hist'] > 0)
        if cond: results.append("MACD 緩步爬升")
        
    # 策略三：布林通道突破
    if "S3" in strategies:
        if last['Close'] > last['Upper'] and prev['Close'] <= prev['Upper']:
            results.append("突破布林上軌")

    # 可在此處繼續添加 S1~S7 的邏輯...
    
    return results

# --- 下載與處理單一股票 ---
def process_stock(stock_str, strategies, min_p, max_p):
    try:
        symbol = stock_str.split('  ')[0] + ".TW"
        data = yf.download(symbol, period="2y", interval="1wk", progress=False)
        if data.empty or len(data) < 35: return None
        
        # 股價區間檢查
        current_price = data['Close'].iloc[-1]
        if not (min_p <= current_price <= max_p): return None
        
        df = calculate_indicators(data)
        matches = check_strategies(df, strategies)
        
        if matches:
            return {"symbol": stock_str, "df": df, "matches": matches, "price": current_price}
    except:
        return None
    return None

# --- UI 介面 ---
st.sidebar.title("🔍 選股條件設定")
raw_df, industry_list = get_taiwan_stock_list()

selected_industry = st.sidebar.selectbox("選擇產業", ["全部"] + industry_list)
price_range = st.sidebar.slider("股價區間", 0, 1000, (10, 500))

st.sidebar.subheader("篩選策略")
s8 = st.sidebar.checkbox("策略八：MACD 緩步爬升", value=True)
s3 = st.sidebar.checkbox("策略三：布林通道突破", value=False)

strategies = []
if s8: strategies.append("S8")
if s3: strategies.append("S3")

if st.sidebar.button("開始掃描台股"):
    # 準備目標清單
    target_df = raw_df if selected_industry == "全部" else raw_df[raw_df['產業別'] == selected_industry]
    stock_targets = target_df['有價證券代號及名稱'].tolist()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    # 使用 ThreadPoolExecutor 加速下載
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_stock, s, strategies, price_range[0], price_range[1]) for s in stock_targets]
        for i, future in enumerate(futures):
            res = future.result()
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(stock_targets))
            status_text.text(f"正在掃描: {i+1}/{len(stock_targets)}")

    st.success(f"掃描完成！找到 {len(results)} 支符合條件的股票。")
    
    # 顯示結果
    for item in results:
        with st.expander(f"📈 {item['symbol']} - 價格: {item['price']:.2f} (符合: {', '.join(item['matches'])})"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(item['df'].tail(5)[['Close', 'MACD', 'Signal', 'Hist']])
            with col2:
                # 繪製 K 線與 MACD
                df_plot = item['df'].tail(40)
                mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
                s  = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc)
                
                # MACD 子圖
                addplots = [
                    mpf.make_addplot(df_plot['MACD'], panel=1, color='fuchsia', secondary_y=False),
                    mpf.make_addplot(df_plot['Signal'], panel=1, color='b', secondary_y=False),
                    mpf.make_addplot(df_plot['Hist'], type='bar', panel=1, color='gray', secondary_y=False)
                ]
                
                fig, axlist = mpf.plot(df_plot, type='candle', style=s, addplot=addplots, 
                                      volume=True, returnfig=True, figsize=(10, 6), panel_ratios=(2,1))
                st.pyplot(fig)

else:
    st.info("請在左側設定條件並點擊「開始掃描」")
