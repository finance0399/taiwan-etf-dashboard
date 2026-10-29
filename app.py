import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import date, datetime, timedelta


# ============================================================
# 基本設定
# ============================================================

st.set_page_config(
    page_title="台灣 ETF 及共同基金績效比較",
    page_icon="📊",
    layout="wide",
)

st.title("📊 台灣 ETF 及共同基金績效比較")

st.caption(
    "比較 ETF 與共同基金於指定日期的淨值／收盤價及 Daily、MTD、YTD、1Y、2Y、3Y 績效。"
)


# ============================================================
# 商品設定
# ============================================================

PRODUCTS = {
    "00981A": {
        "name": "00981A 統一台股增長主動式ETF",
        "type": "ETF",
        "code": "00981A",
    },
    "A09012": {
        "name": "統一奔騰基金",
        "type": "FUND",
        "code": "A09012",
    },
    "00407A": {
        "name": "00407A 凱基台灣主動式ETF",
        "type": "ETF",
        "code": "00407A",
    },
    "A41209": {
        "name": "凱基台灣精五門基金",
        "type": "FUND",
        "code": "A41209",
    },
    "009816": {
        "name": "009816 凱基台灣TOP 50 ETF",
        "type": "ETF",
        "code": "009816",
    },
    "0050": {
        "name": "0050 元大台灣50",
        "type": "ETF",
        "code": "0050",
    },
}


# ============================================================
# MoneyDJ 查詢網址
# ============================================================

MONEYDJ_URL = "https://www.moneydj.com/funddj/ya/navfastpage.djhtm"


# ============================================================
# 共用工具
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return np.nan

        text = str(value).strip().replace(",", "")

        if text in ["", "--", "---", "None", "nan", "NaN"]:
            return np.nan

        return float(text)

    except Exception:
        return np.nan


def clean_date(value):
    try:
        return pd.to_datetime(value).normalize()
    except Exception:
        return pd.NaT


def pct_change(current, previous):
    if pd.isna(current) or pd.isna(previous):
        return np.nan

    if previous == 0:
        return np.nan

    return (current / previous - 1) * 100


def format_number(value):
    if pd.isna(value):
        return "-"

    return f"{value:,.4f}".rstrip("0").rstrip(".")


def format_pct(value):
    if pd.isna(value):
        return "-"

    return f"{value:+.2f}%"


# ============================================================
# 台灣證交所 ETF 歷史資料
# ============================================================

@st.cache_data(ttl=3600)
def fetch_twse_month(stock_code, year, month):
    """
    從 TWSE STOCK_DAY API 取得單月交易資料。
    """

    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    params = {
        "response": "json",
        "date": f"{year}{month:02d}01",
        "stockNo": stock_code,
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=12,
        )

        response.raise_for_status()

        data = response.json()

        if "data" not in data:
            return pd.DataFrame(columns=["date", "nav"])

        records = []

        for row in data["data"]:
            if len(row) < 7:
                continue

            roc_date = str(row[0])

            try:
                roc_year, mm, dd = roc_date.split("/")
                year_ad = int(roc_year) + 1911

                trade_date = pd.Timestamp(
                    year=year_ad,
                    month=int(mm),
                    day=int(dd),
                )

                close_price = safe_float(row[6])

                if pd.notna(close_price):
                    records.append(
                        {
                            "date": trade_date,
                            "nav": close_price,
                        }
                    )

            except Exception:
                continue

        if not records:
            return pd.DataFrame(columns=["date", "nav"])

        df = pd.DataFrame(records)

        df["date"] = pd.to_datetime(df["date"])
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

        df = df.dropna(subset=["date", "nav"])
        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

        return df

    except Exception:
        return pd.DataFrame(columns=["date", "nav"])


def month_iterator(start_date, end_date):
    current = pd.Timestamp(
        year=start_date.year,
        month=start_date.month,
        day=1,
    )

    end_month = pd.Timestamp(
        year=end_date.year,
        month=end_date.month,
        day=1,
    )

    while current <= end_month:
        yield current.year, current.month
        current = current + pd.offsets.MonthBegin(1)


@st.cache_data(ttl=3600)
def fetch_etf_history(stock_code, end_date):
    """
    為了計算 3Y，往前抓約 3 年 + 緩衝期間。
    """

    end_date = pd.Timestamp(end_date)

    start_date = end_date - pd.DateOffset(
        years=3,
        months=2,
    )

    frames = []

    for year, month in month_iterator(start_date, end_date):
        temp = fetch_twse_month(
            stock_code,
            year,
            month,
        )

        if not temp.empty:
            frames.append(temp)

    if not frames:
        return pd.DataFrame(columns=["date", "nav"])

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result["date"] = pd.to_datetime(result["date"])
    result["nav"] = pd.to_numeric(
        result["nav"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["date", "nav"]
    )

    result = result[
        result["date"] <= end_date
    ]

    result = result.sort_values("date")

    result = result.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return result.reset_index(drop=True)


# ============================================================
# 共同基金 CSV
# ============================================================

LOCAL_FUND_FILE = Path("fund_nav_history.csv")


@st.cache_data(ttl=300)
def load_fund_history():
    """
    支援常見 CSV 欄位格式：

    date,code,name,nav

    或：

    日期,基金代碼,基金名稱,淨值
    """

    if not LOCAL_FUND_FILE.exists():
        return pd.DataFrame(
            columns=["date", "code", "name", "nav"]
        )

    try:
        data = pd.read_csv(
            LOCAL_FUND_FILE,
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:
        try:
            data = pd.read_csv(
                LOCAL_FUND_FILE,
                encoding="big5",
            )

        except Exception:
            return pd.DataFrame(
                columns=["date", "code", "name", "nav"]
            )

    except Exception:
        return pd.DataFrame(
            columns=["date", "code", "name", "nav"]
        )

    # 移除欄位名稱前後空白
    data.columns = [
        str(col).strip()
        for col in data.columns
    ]

    rename_map = {
        "日期": "date",
        "Date": "date",
        "DATE": "date",

        "基金代碼": "code",
        "代碼": "code",
        "Code": "code",
        "CODE": "code",

        "基金名稱": "name",
        "名稱": "name",
        "Name": "name",
        "NAME": "name",

        "淨值": "nav",
        "NAV": "nav",
        "Nav": "nav",
    }

    data = data.rename(
        columns=rename_map
    )

    required = ["date", "code", "nav"]

    for col in required:
        if col not in data.columns:
            return pd.DataFrame(
                columns=["date", "code", "name", "nav"]
            )

    if "name" not in data.columns:
        data["name"] = ""

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data["code"] = (
        data["code"]
        .astype(str)
        .str.strip()
    )

    data["name"] = (
        data["name"]
        .astype(str)
        .str.strip()
    )

    data["nav"] = pd.to_numeric(
        data["nav"]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    data = data.dropna(
        subset=["date", "nav"]
    )

    data = data.sort_values(
        ["code", "date"]
    )

    data = data.drop_duplicates(
        subset=["code", "date"],
        keep="last",
    )

    return data.reset_index(drop=True)


def get_fund_history(code, fund_name):
    data = load_fund_history()

    if data.empty:
        return pd.DataFrame(
            columns=["date", "nav"]
        )

    # 優先基金代碼
    temp = data[
        data["code"] == code
    ].copy()

    # 如果代碼找不到，再嘗試基金名稱
    if temp.empty and "name" in data.columns:
        temp = data[
            data["name"].str.contains(
                fund_name,
                na=False,
                regex=False,
            )
        ].copy()

    if temp.empty:
        return pd.DataFrame(
            columns=["date", "nav"]
        )

    temp = temp[
        ["date", "nav"]
    ].copy()

    temp = temp.sort_values("date")

    return temp.reset_index(drop=True)


# ============================================================
# 找指定日期以前最近一筆資料
# ============================================================

def get_value_on_or_before(history, target_date):
    if history.empty:
        return None, np.nan

    target_date = pd.Timestamp(target_date)

    temp = history[
        history["date"] <= target_date
    ].copy()

    if temp.empty:
        return None, np.nan

    row = temp.iloc[-1]

    return row["date"], row["nav"]


def get_previous_value(history, current_date):
    temp = history[
        history["date"] < current_date
    ]

    if temp.empty:
        return None, np.nan

    row = temp.iloc[-1]

    return row["date"], row["nav"]


# ============================================================
# 績效計算
# ============================================================

def calculate_performance(history, selected_date):
    if history.empty:
        return None

    history = history.copy()

    history["date"] = pd.to_datetime(
        history["date"]
    )

    history["nav"] = pd.to_numeric(
        history["nav"],
        errors="coerce",
    )

    history = history.dropna(
        subset=["date", "nav"]
    )

    history = history.sort_values("date")

    selected_date = pd.Timestamp(
        selected_date
    )

    current_date, current_nav = (
        get_value_on_or_before(
            history,
            selected_date,
        )
    )

    if current_date is None:
        return None

    # Daily
    previous_date, previous_nav = (
        get_previous_value(
            history,
            current_date,
        )
    )

    daily = pct_change(
        current_nav,
        previous_nav,
    )

    # MTD
    month_start = current_date.replace(
        day=1
    )

    mtd_base_date = (
        month_start - timedelta(days=1)
    )

    _, month_base_nav = get_value_on_or_before(
        history,
        mtd_base_date,
    )

    mtd = pct_change(
        current_nav,
        month_base_nav,
    )

    # YTD
    year_start = pd.Timestamp(
        year=current_date.year,
        month=1,
        day=1,
    )

    ytd_base_date = (
        year_start - timedelta(days=1)
    )

    _, year_base_nav = get_value_on_or_before(
        history,
        ytd_base_date,
    )

    ytd = pct_change(
        current_nav,
        year_base_nav,
    )

    # 1Y / 2Y / 3Y
    one_year_date = (
        current_date - pd.DateOffset(years=1)
    )

    two_year_date = (
        current_date - pd.DateOffset(years=2)
    )

    three_year_date = (
        current_date - pd.DateOffset(years=3)
    )

    _, one_year_nav = get_value_on_or_before(
        history,
        one_year_date,
    )

    _, two_year_nav = get_value_on_or_before(
        history,
        two_year_date,
    )

    _, three_year_nav = get_value_on_or_before(
        history,
        three_year_date,
    )

    one_year = pct_change(
        current_nav,
        one_year_nav,
    )

    two_year = pct_change(
        current_nav,
        two_year_nav,
    )

    three_year = pct_change(
        current_nav,
        three_year_nav,
    )

    return {
        "date": current_date,
        "nav": current_nav,
        "Daily": daily,
        "MTD": mtd,
        "YTD": ytd,
        "1Y": one_year,
        "2Y": two_year,
        "3Y": three_year,
    }


# ============================================================
# 日期選擇
# ============================================================

today = date.today()

selected_date = st.date_input(
    "📅 選擇查詢日期",
    value=today,
    max_value=today,
)

st.caption(
    "若所選日期為假日或當日尚無資料，系統會採用該日期以前最近一個有資料的交易日。"
)


# ============================================================
# 建立結果
# ============================================================

results = []

progress = st.progress(0)

product_items = list(PRODUCTS.items())

for index, (product_code, info) in enumerate(product_items):

    if info["type"] == "ETF":

        history = fetch_etf_history(
            info["code"],
            selected_date,
        )

        data_source = "TWSE"

    else:

        fund_short_name = info["name"]

        history = get_fund_history(
            product_code,
            fund_short_name,
        )

        data_source = "fund_nav_history.csv"

    performance = calculate_performance(
        history,
        selected_date,
    )

    if performance is None:

        row = {
            "基金名稱": info["name"],
            "淨值日期": "-",
            "最新淨值": np.nan,
            "Daily": np.nan,
            "MTD": np.nan,
            "YTD": np.nan,
            "1Y": np.nan,
            "2Y": np.nan,
            "3Y": np.nan,
            "資料來源": data_source,
        }

    else:

        row = {
            "基金名稱": info["name"],
            "淨值日期": performance["date"].strftime(
                "%Y/%m/%d"
            ),
            "最新淨值": performance["nav"],
            "Daily": performance["Daily"],
            "MTD": performance["MTD"],
            "YTD": performance["YTD"],
            "1Y": performance["1Y"],
            "2Y": performance["2Y"],
            "3Y": performance["3Y"],
            "資料來源": data_source,
        }

    results.append(row)

    progress.progress(
        (index + 1) / len(product_items)
    )

progress.empty()


# ============================================================
# 結果 DataFrame
# ============================================================

result_df = pd.DataFrame(results)


# ============================================================
# 上方資訊
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "查詢日期",
        pd.Timestamp(selected_date).strftime(
            "%Y/%m/%d"
        ),
    )

with col2:

    valid_dates = result_df[
        result_df["淨值日期"] != "-"
    ]["淨值日期"]

    if len(valid_dates) > 0:
        latest_data_date = max(valid_dates)
    else:
        latest_data_date = "-"

    st.metric(
        "目前最新資料日期",
        latest_data_date,
    )

with col3:
    st.metric(
        "比較商品數",
        len(result_df),
    )


# ============================================================
# 表格格式
# ============================================================

display_df = result_df.copy()

display_df["最新淨值"] = (
    display_df["最新淨值"]
    .apply(format_number)
)

performance_columns = [
    "Daily",
    "MTD",
    "YTD",
    "1Y",
    "2Y",
    "3Y",
]

for column in performance_columns:
    display_df[column] = (
        display_df[column]
        .apply(format_pct)
    )


# ============================================================
# 顏色
# ============================================================

def performance_color(value):
    if not isinstance(value, str):
        return ""

    if value == "-":
        return ""

    try:
        number = float(
            value
            .replace("%", "")
            .replace("+", "")
        )

        # 台灣市場習慣：
        # 上漲紅色
        # 下跌綠色
        if number > 0:
            return (
                "color: #d62728; "
                "font-weight: bold;"
            )

        if number < 0:
            return (
                "color: #138a36; "
                "font-weight: bold;"
            )

    except Exception:
        pass

    return ""


styled_df = (
    display_df.style
    .map(
        performance_color,
        subset=performance_columns,
    )
)


# ============================================================
# 顯示主表
# ============================================================

st.subheader("📈 績效比較")

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 找不到共同基金資料時顯示提醒
# ============================================================

missing_funds = []

for _, row in result_df.iterrows():

    fund_name = row["基金名稱"]

    if (
        fund_name in [
            "統一奔騰基金",
            "凱基台灣精五門基金",
        ]
        and row["淨值日期"] == "-"
    ):
        missing_funds.append(
            fund_name
        )


if missing_funds:
    st.warning(
        "⚠️ 以下共同基金目前在 "
        "fund_nav_history.csv 找不到可使用的歷史淨值資料："
        + "、".join(missing_funds)
    )


# ============================================================
# MoneyDJ
# ============================================================

st.divider()

st.subheader("🔎 共同基金淨值查詢")

st.write(
    "若共同基金尚未建立歷史資料，可先至 MoneyDJ "
    "確認最新淨值，再將資料加入 fund_nav_history.csv。"
)

st.link_button(
    "開啟 MoneyDJ 基金淨值查詢",
    MONEYDJ_URL,
)


# ============================================================
# CSV 格式說明
# ============================================================

with st.expander(
    "📁 fund_nav_history.csv 應該怎麼放資料？"
):

    st.write(
        "請確認 fund_nav_history.csv "
        "和 app.py 放在 GitHub Repository 的同一層。"
    )

    example_df = pd.DataFrame(
        [
            {
                "date": "2026-10-27",
                "code": "A09012",
                "name": "統一奔騰基金",
                "nav": 100.12,
            },
            {
                "date": "2026-10-28",
                "code": "A09012",
                "name": "統一奔騰基金",
                "nav": 101.25,
            },
            {
                "date": "2026-10-27",
                "code": "A41209",
                "name": "凱基台灣精五門基金",
                "nav": 80.15,
            },
            {
                "date": "2026-10-28",
                "code": "A41209",
                "name": "凱基台灣精五門基金",
                "nav": 81.30,
            },
        ]
    )

    st.dataframe(
        example_df,
        use_container_width=True,
        hide_index=True,
    )

    st.code(
        """date,code,name,nav
2026-10-27,A09012,統一奔騰基金,100.12
2026-10-28,A09012,統一奔騰基金,101.25
2026-10-27,A41209,凱基台灣精五門基金,80.15
2026-10-28,A41209,凱基台灣精五門基金,81.30"""
    )


# ============================================================
# 計算方式說明
# ============================================================

with st.expander(
    "ℹ️ 績效計算方式"
):

    st.markdown(
        """
**Daily**

目前淨值 ÷ 前一筆淨值 − 1

**MTD**

目前淨值 ÷ 上月底淨值 − 1

**YTD**

目前淨值 ÷ 去年底淨值 − 1

**1Y**

目前淨值 ÷ 一年前最近可取得淨值 − 1

**2Y**

目前淨值 ÷ 兩年前最近可取得淨值 − 1

**3Y**

目前淨值 ÷ 三年前最近可取得淨值 − 1
"""
    )


# ============================================================
# 資料說明
# ============================================================

st.divider()

st.caption(
    "資料僅供績效比較與研究用途。ETF 使用市場交易資料；"
    "共同基金資料以 fund_nav_history.csv 為準。"
)

st.caption(
    "共同基金實際淨值仍應以各基金公司正式公告資料為準。"
)
