import json
import urllib.parse
import urllib.request
from datetime import date

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="台股 ETF 歷史收盤價",
    page_icon="📈",
    layout="wide",
)


# 可自行新增或刪除 ETF
ETF_LIST = {
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "006208": "富邦台50",
    "00878": "國泰永續高股息",
    "00915": "凱基優選高股息30",
    "00919": "群益台灣精選高息",
    "00938": "凱基優選30",
    "00952": "凱基台灣AI50",
}


def 轉成數字(資料):
    """將證交所回傳的文字轉成數字。"""
    
    文字 = (
        str(資料 or "")
        .strip()
        .replace(",", "")
        .replace("+", "")
    )
        
    if not 文字 or 文字 in ["--", "---"]:
        return None

        try:
        return float(文字)
    except ValueError:
        return None


def 西元轉民國日期(選擇日期):
    """例如 2026-10-20 轉成 115/10/20。"""
    
    民國年 = 選擇日期.year - 1911
    
    return (
        f"{民國年:03d}/"
        f"{選擇日期.month:02d}/"
        f"{選擇日期.day:02d}"
    )


@st.cache_data(ttl=3600)
def 取得單檔ETF月份資料(ETF代號, 查詢月份):
    """取得一檔 ETF 指定月份的每日行情。"""
    
    參數 = urllib.parse.urlencode({
        "date": 查詢月份,
        "stockNo": ETF代號,
        "response": "json",
    })
    
    網址 = (
        "https://www.twse.com.tw/"
        "rwd/zh/afterTrading/STOCK_DAY?"
        + 參數
    )
    
    要求 = urllib.request.Request(
        網址,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    
    with urllib.request.urlopen(要求, timeout=30) as 回應:
        內容 = 回應.read().decode("utf-8")
        return json.loads(內容)


def 查詢指定日期ETF(ETF代號, ETF名稱, 選擇日期):
    """找出指定 ETF 在所選日期的行情。"""
    
    查詢月份 = 選擇日期.strftime("%Y%m01")
    目標民國日期 = 西元轉民國日期(選擇日期)
    
    回傳資料 = 取得單檔ETF月份資料(
        ETF代號,
        查詢月份,
    )
    
    if 回傳資料.get("stat") != "OK":
        return None
        
    每日資料 = 回傳資料.get("data", [])
    
    for 每日行情 in 每日資料:
        if not 每日行情:
            continue
            
        if 每日行情[0] != 目標民國日期:
            continue
            
        # 證交所欄位順序：
        # 日期、成交股數、成交金額、開盤價、最高價、
        # 最低價、收盤價、漲跌價差、成交筆數
        
        收盤價 = 轉成數字(每日行情[6])
        漲跌價差 = 轉成數字(每日行情[7])
        
        昨收價 = None
        漲跌幅 = None
        
        if 收盤價 is not None and 漲跌價差 is not None:
            昨收價 = 收盤價 - 漲跌價差
            if 昨收價 != 0:
                漲跌幅 = 漲跌價差 / 昨收價 * 100
        return {
            "日期": 選擇日期.strftime("%Y-%m-%d"),
            "ETF代號": ETF代號,
            "ETF名稱": ETF名稱,
            "收盤價": 收盤價,
            "漲跌價差": 漲跌價差,
            "漲跌幅(%)": 漲跌幅,
            "成交股數": 轉成數字(每日行情[1]),
        }
    return None
st.title("📈 台股 ETF 歷史收盤價查詢")
st.caption(
    "資料來源：臺灣證券交易所。"
    "資料僅供資訊參考，不構成投資建議。"
)
# 日期選擇器
選擇日期 = st.date_input(
    "選擇查詢日期",
    value=date.today(),
    min_value=date(2010, 1, 4),
    max_value=date.today(),
    format="YYYY/MM/DD",
)
# ETF 選擇器
選擇ETF = st.multiselect(
    "選擇 ETF",
    options=list(ETF_LIST.keys()),
    default=list(ETF_LIST.keys()),
    format_func=lambda 代號: (
        f"{代號} {ETF_LIST[代號]}"
    ),
)
if st.button("查詢收盤價", type="primary"):
    if not 選擇ETF:
        st.warning("請至少選擇一檔 ETF。")
    else:
        with st.spinner("正在查詢證交所資料……"):
            查詢結果 = []
            for ETF代號 in 選擇ETF:
                try:
                    單檔結果 = 查詢指定日期ETF(
                        ETF代號,
                        ETF_LIST[ETF代號],
                        選擇日期,
                    )
                    if 單檔結果 is not None:
                        查詢結果.append(單檔結果)
                except Exception as 錯誤:
                    st.warning(
                        f"{ETF代號} 查詢失敗：{錯誤}"
                    )
        if 查詢結果:
            資料表 = pd.DataFrame(查詢結果)
            st.success(
                f"查詢日期："
                f"{選擇日期.strftime('%Y-%m-%d')}"
            )
            上漲檔數 = int(
                (資料表["漲跌幅(%)"] > 0).sum()
            )
            下跌檔數 = int(
                (資料表["漲跌幅(%)"] < 0).sum()
            )
            平盤檔數 = int(
                (資料表["漲跌幅(%)"] == 0).sum()
            )
            欄位1, 欄位2, 欄位3, 欄位4 = st.columns(4)
            欄位1.metric("查詢檔數", len(資料表))
            欄位2.metric("上漲", 上漲檔數)
            欄位3.metric("下跌", 下跌檔數)
            欄位4.metric("平盤", 平盤檔數)
            st.dataframe(
                資料表,
                width="stretch",
                hide_index=True,
                column_config={
                    "日期": st.column_config.DateColumn(
                        "日期",
                        format="YYYY-MM-DD",
                    ),
                    "ETF代號":
                        st.column_config.TextColumn(
                            "ETF代號"
                        ),
                    "ETF名稱":
                        st.column_config.TextColumn(
                            "ETF名稱"
                        ),
                    "收盤價":
                        st.column_config.NumberColumn(
                            "收盤價",
                            format="%.2f",
                        ),
                    "漲跌價差":
                        st.column_config.NumberColumn(
                            "漲跌價差",
                            format="%+.2f",
                        ),
                    "漲跌幅(%)":
                        st.column_config.NumberColumn(
                            "漲跌幅(%)",
                            format="%+.2f%%",
                        ),
                    "成交股數":
                        st.column_config.NumberColumn(
                            "成交股數",
                            format="%,.0f",
                        ),
                },
            )
            # 下載查詢結果
            CSV資料 = 資料表.to_csv(
                index=False,
                encoding="utf-8-sig",
            )
            st.download_button(
                "下載查詢結果 CSV",
                data=CSV資料,
                file_name=(
                    "ETF收盤價_"
                    f"{選擇日期:%Y%m%d}.csv"
                 ),
                mime="text/csv",
            )
        else:
            st.info(
                "所選日期查不到交易資料。"
                "該日可能是假日、週末，"
                "或 ETF 當時尚未上市。"
            )
