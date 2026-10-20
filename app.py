import json
import urllib.request

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="台股 ETF 儀表板",
    page_icon="📈",
    layout="wide",
)


ETF_CODES = [
    "0050",
    "0056",
    "006208",
    "00878",
    "00915",
    "00919",
    "00938",
    "00952",
]

API_URL = (
    "https://openapi.twse.com.tw/"
    "v1/exchangeReport/STOCK_DAY_ALL"
)


def 轉成數字(資料):
    文字 = str(資料 or "").strip().replace(",", "")
    
    if not 文字 or 文字 == "--" or 文字 == "---":
        return None
        
    try:
        return float(文字)
    except ValueError:
        return None


@st.cache_data(ttl=1800)
def 取得證交所資料():
    要求 = urllib.request.Request(
        API_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    
    with urllib.request.urlopen(要求, timeout=30) as 回應:
        內容 = 回應.read().decode("utf-8")
        return json.loads(內容)


def 建立ETF資料表(全部資料):
    資料索引 = {
        str(商品.get("Code", "")).strip(): 商品
        for 商品 in 全部資料
    }
    
    報表資料 = []
    
    for ETF代號 in ETF_CODES:
        商品 = 資料索引.get(ETF代號)
        
        if 商品 is None:
            報表資料.append({
                "ETF代號": ETF代號,
                "ETF名稱": "查無資料",
                "收盤價": None,
                "漲跌價差": None,
                "漲跌幅(%)": None,
            })
            continue
            
        收盤價 = 轉成數字(商品.get("ClosingPrice"))
        漲跌價差 = 轉成數字(商品.get("Change"))
        
        昨收價 = None
        漲跌幅 = None
        
        if 收盤價 is not None and 漲跌價差 is not None:
            昨收價 = 收盤價 - 漲跌價差
            
            if 昨收價 != 0:
                漲跌幅 = 漲跌價差 / 昨收價 * 100
                
        報表資料.append({
            "ETF代號": ETF代號,
            "ETF名稱": 商品.get("Name", ""),
            "收盤價": 收盤價,
            "漲跌價差": 漲跌價差,
            "漲跌幅(%)": 漲跌幅,
        })
        
    return pd.DataFrame(報表資料)


st.title("📈 台股 ETF 每日收盤儀表板")
st.caption(
    "資料來源：臺灣證券交易所。"
    "資料僅供資訊參考，不構成投資建議。"
)

if st.button("重新整理資料"):
    st.cache_data.clear()


try:
    全部資料 = 取得證交所資料()
    ETF資料 = 建立ETF資料表(全部資料)
    
    可計算漲跌幅 = ETF資料["漲跌幅(%)"].dropna()
    
    上漲檔數 = int((可計算漲跌幅 > 0).sum())
    下跌檔數 = int((可計算漲跌幅 < 0).sum())
    平盤檔數 = int((可計算漲跌幅 == 0).sum())
    
    欄位1, 欄位2, 欄位3, 欄位4 = st.columns(4)
    
    欄位1.metric("ETF 檔數", len(ETF資料))
    欄位2.metric("上漲", 上漲檔數)
    欄位3.metric("下跌", 下跌檔數)
    欄位4.metric("平盤", 平盤檔數)
    
    搜尋文字 = st.text_input(
        "搜尋 ETF 代號或名稱",
        placeholder="例如：0050 或 台灣50",
    ).strip()
    
    顯示資料 = ETF資料.copy()
    
    if 搜尋文字:
        篩選條件 = (
            顯示資料["ETF代號"]
            .astype(str)
            .str.contains(搜尋文字, case=False, na=False)
            |
            顯示資料["ETF名稱"]
            .astype(str)
            .str.contains(搜尋文字, case=False, na=False)
        )
        
        顯示資料 = 顯示資料[篩選條件]
        
    st.dataframe(
        顯示資料,
        width="stretch",
        hide_index=True,
        column_config={
            "ETF代號": st.column_config.TextColumn("ETF代號"),
            "ETF名稱": st.column_config.TextColumn("ETF名稱"),
            "收盤價": st.column_config.NumberColumn(
                "收盤價",
                format="%.2f",
            ),
            "漲跌價差": st.column_config.NumberColumn(
                "漲跌價差",
                format="%+.2f",
            ),
            "漲跌幅(%)": st.column_config.NumberColumn(
                "漲跌幅(%)",
                format="%+.2f%%",
            ),
        },
    )
    
except Exception as 錯誤:
    st.error("目前無法取得證交所資料。")
    st.code(str(錯誤))
