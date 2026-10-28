import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# 1. Streamlit 網頁基本設定
# ============================================================

st.set_page_config(
    page_title="台灣基金績效比較",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. 商品基本資料
# ============================================================

PRODUCTS = {
    "00981A": {
        "name": "統一台股增長主動式ETF基金",
        "type": "ETF",
        "group": "統一系列",
        "ticker": "00981A.TW",
    },
    "A09012": {
        "name": "統一奔騰基金",
        "type": "共同基金",
        "group": "統一系列",
        "ticker": None,
    },
    "00407A": {
        "name": "凱基台灣主動式ETF基金",
        "type": "ETF",
        "group": "凱基系列",
        "ticker": "00407A.TW",
    },
    "A41209": {
        "name": "凱基台灣精五門基金",
        "type": "共同基金",
        "group": "凱基系列",
        "ticker": None,
    },
    "009816": {
        "name": "凱基台灣TOP 50 ETF基金",
        "type": "ETF",
        "group": "凱基系列",
        "ticker": "009816.TW",
    },
    "0050": {
        "name": "元大台灣50",
        "type": "ETF",
        "group": "市場基準",
        "ticker": "0050.TW",
    },
}

DISPLAY_ORDER = [
    "00981A",
    "A09012",
    "00407A",
    "A41209",
    "009816",
    "0050",
]

LOCAL_FUND_FILE = Path("fund_nav_history.csv")


# ============================================================
# 3. 自訂 CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .dashboard-header {
        padding: 22px 26px;
        margin-bottom: 18px;
        border-radius: 16px;
        color: white;
        background: linear-gradient(
            135deg,
            #0b3c5d 0%,
            #176b87 55%,
            #64ccc5 100%
        );
        box-shadow: 0 8px 20px rgba(11, 60, 93, 0.16);
    }

    .dashboard-header h1 {
        margin: 0;
        padding: 0;
        font-size: 32px;
        font-weight: 750;
    }

    .dashboard-header p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 15px;
        opacity: 0.92;
    }

        .info-card {
        min-height: 116px;
        padding: 18px;
        border: 1px solid #e4e9ef;
        border-radius: 14px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(40, 54, 70, 0.06);
    }

    .info-label {
        margin-bottom: 7px;
        color: #667085;
        font-size: 14px;
        font-weight: 600;
    }

    .info-value {
        color: #172b4d;
        font-size: 25px;
        font-weight: 750;
    }

    .info-note {
        margin-top: 6px;
        color: #7b8794;
        font-size: 12px;
    }

    .section-title {
        margin-top: 18px;
        margin-bottom: 10px;
        color: #172b4d;
        font-size: 21px;
        font-weight: 750;
    }

    .positive-value {
        color: #d32f2f;
        font-weight: 700;
    }

    .negative-value {
        color: #138a50;
        font-weight: 700;
    }

    .neutral-value {
        color: #667085;
        font-weight: 600;
    }

        div[data-testid="stMetric"] {
        padding: 14px;
        border: 1px solid #e4e9ef;
        border-radius: 12px;
        background-color: white;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e4e9ef;
        border-radius: 12px;
        overflow: hidden;
    }

    .data-note {
        padding: 13px 16px;
        border-left: 4px solid #176b87;
        border-radius: 8px;
        background-color: #eef8fb;
        color: #344054;
        font-size: 14px;
        line-height: 1.7;
    }

    .footer-note {
        margin-top: 28px;
        padding-top: 14px;
        border-top: 1px solid #dfe4ea;
        color: #7b8794;
        font-size: 12px;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. 共用函式
# ============================================================

def remove_timezone(index):
    """
    移除日期索引的時區資訊，避免不同資料來源合併時發生錯誤。
    """
    date_index = pd.DatetimeIndex(index)

    if date_index.tz is not None:
        date_index = date_index.tz_localize(None)

    return date_index.normalize()


def clean_price_series(series):
    """
    清理價格或淨值資料。
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    cleaned = pd.to_numeric(series, errors="coerce").dropna()

    if cleaned.empty:
        return pd.Series(dtype=float)

    cleaned.index = remove_timezone(cleaned.index)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()

    return cleaned


def safe_return(latest_value, base_value):
    """
    計算報酬率。
    """
    if latest_value is None or base_value is None:
        return None

    if pd.isna(latest_value) or pd.isna(base_value):
        return None

    if float(base_value) == 0:
        return None

    return (float(latest_value) / float(base_value) - 1) * 100


def get_value_on_or_before(series, target_date):
    """
    取得指定日期或指定日期之前最近一筆有效資料。
    """
    if series is None or len(series) == 0:
        return None

    target_timestamp = pd.Timestamp(target_date).normalize()
    available = series.loc[series.index <= target_timestamp]

    if available.empty:
        return None

    return float(available.iloc[-1])


def calculate_performance(series):
    """
    計算最新淨值、Daily、MTD、YTD、1Y、2Y、3Y。
    """
    series = clean_price_series(series)

    empty_result = {
        "latest": None,
        "date": None,
        "Daily": None,
        "MTD": None,
        "YTD": None,
        "1Y": None,
        "2Y": None,
        "3Y": None,
    }

    if series.empty:
        return empty_result

    latest_value = float(series.iloc[-1])
    latest_date = pd.Timestamp(series.index[-1]).normalize()

    previous_value = None
    if len(series) >= 2:
        previous_value = float(series.iloc[-2])

    month_start = latest_date.replace(day=1)
    previous_month_end = month_start - pd.Timedelta(days=1)

    year_start = pd.Timestamp(
        year=latest_date.year,
        month=1,
        day=1,
    )
    previous_year_end = year_start - pd.Timedelta(days=1)

    one_year_date = latest_date - pd.DateOffset(years=1)
    two_year_date = latest_date - pd.DateOffset(years=2)
    three_year_date = latest_date - pd.DateOffset(years=3)

    month_base = get_value_on_or_before(series, previous_month_end)
    year_base = get_value_on_or_before(series, previous_year_end)
    one_year_base = get_value_on_or_before(series, one_year_date)
    two_year_base = get_value_on_or_before(series, two_year_date)
    three_year_base = get_value_on_or_before(series, three_year_date)

    return {
        "latest": latest_value,
        "date": latest_date,
        "Daily": safe_return(latest_value, previous_value),
        "MTD": safe_return(latest_value, month_base),
        "YTD": safe_return(latest_value, year_base),
        "1Y": safe_return(latest_value, one_year_base),
        "2Y": safe_return(latest_value, two_year_base),
        "3Y": safe_return(latest_value, three_year_base),
    }


def format_nav(value):
    """
    格式化淨值或ETF價格。
    """
    if value is None or pd.isna(value):
        return "無資料"

    return f"{float(value):,.2f}"


def format_return(value):
    """
    格式化報酬率。
    """
    if value is None or pd.isna(value):
        return "-"

    value = float(value)

    if value > 0:
        return f"+{value:.2f}%"

    return f"{value:.2f}%"


def performance_style(value):
    """
    DataFrame 報酬率顏色。
    台股慣例：上漲紅色，下跌綠色。
    """
    if not isinstance(value, str):
        return ""

    if value.startswith("+"):
        return (
            "color: #d32f2f;"
            "font-weight: 700;"
            "background-color: #fff5f5;"
        )

    if value.startswith("-") and value != "-":
        return (
            "color: #138a50;"
            "font-weight: 700;"
            "background-color: #f1fbf5;"
        )

    return "color: #667085; font-weight: 600;"


# ============================================================
# 5. ETF 資料下載
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_etf_data(ticker, period="5y"):
    """
    使用 Yahoo Finance 下載 ETF 歷史價格。

    auto_adjust=False：
    顯示實際收盤價時使用 Close。

    績效計算：
    優先使用 Adj Close，將配息與除權息影響納入；
    若沒有 Adj Close，則改用 Close。
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(-1):
                data = data.xs(ticker, axis=1, level=-1)
            else:
                data.columns = data.columns.get_level_values(0)

        data.index = remove_timezone(data.index)
        data = data[~data.index.duplicated(keep="last")]
        data = data.sort_index()

        return data

    except Exception:
        return pd.DataFrame()


def get_etf_series(ticker):
    """
    取得 ETF 顯示價格與績效序列。
    """
    data = download_etf_data(ticker)

    if data.empty:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    close_series = pd.Series(dtype=float)
    performance_series = pd.Series(dtype=float)

    if "Close" in data.columns:
        close_series = clean_price_series(data["Close"])

    if "Adj Close" in data.columns:
        performance_series = clean_price_series(data["Adj Close"])

    if performance_series.empty:
        performance_series = close_series.copy()

    return close_series, performance_series


# ============================================================
# 6. 共同基金 CSV 資料
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_local_fund_data(file_modified_time=None):
    """
    讀取共同基金淨值 CSV。
    """
    empty_data = pd.DataFrame(
        columns=["date", "code", "nav"]
    )

    if not LOCAL_FUND_FILE.exists():
        return empty_data

    try:
        data = pd.read_csv(
            LOCAL_FUND_FILE,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            LOCAL_FUND_FILE,
            encoding="big5",
        )
    except Exception:
        return empty_data

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    lower_column_map = {
        str(column).strip().lower(): column
        for column in data.columns
    }

    date_candidates = [
        "date",
        "日期",
        "淨值日期",
        "資料日期",
    ]

    code_candidates = [
        "code",
        "基金代碼",
        "代碼",
    ]

    nav_candidates = [
        "nav",
        "淨值",
        "基金淨值",
        "value",
    ]

    date_column = None
    code_column = None
    nav_column = None

    for candidate in date_candidates:
        if candidate.lower() in lower_column_map:
            date_column = lower_column_map[candidate.lower()]
            break

    for candidate in code_candidates:
        if candidate.lower() in lower_column_map:
            code_column = lower_column_map[candidate.lower()]
            break

    for candidate in nav_candidates:
        if candidate.lower() in lower_column_map:
            nav_column = lower_column_map[candidate.lower()]
            break

    if date_column is None:
        return empty_data

    if code_column is not None and nav_column is not None:
        result = data[
            [date_column, code_column, nav_column]
        ].copy()

        result.columns = ["date", "code", "nav"]

    else:
        value_columns = [
            column
            for column in data.columns
            if column != date_column
        ]

        result = data.melt(
            id_vars=[date_column],
            value_vars=value_columns,
            var_name="code",
            value_name="nav",
        )

        result = result.rename(
            columns={date_column: "date"}
        )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["code"] = (
        result["code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result["nav"] = (
        result["nav"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    result["nav"] = pd.to_numeric(
        result["nav"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["date", "code", "nav"]
    )

    result = result[
        result["code"].isin(["A09012", "A41209"])
    ]

    result = result.sort_values(
        ["code", "date"]
    )

    result = result.drop_duplicates(
        subset=["code", "date"],
        keep="last",
    )

    return result


def get_fund_series(fund_data, code):
    """
    從共同基金長表取得指定基金淨值序列。
    """
    if fund_data.empty:
        return pd.Series(dtype=float)

    selected = fund_data.loc[
        fund_data["code"] == code,
        ["date", "nav"],
    ].copy()

    if selected.empty:
        return pd.Series(dtype=float)

    selected = selected.set_index("date")

    return clean_price_series(selected["nav"])


# ============================================================
# 7. 建立全部商品資料
# ============================================================

def build_dashboard_data():
    """
    建立績效表格與歷史序列。
    """
    modified_time = None

   if LOCAL_FUND_FILE.exists():
        modified_time = LOCAL_FUND_FILE.stat().st_mtime

    fund_data = load_local_fund_data(modified_time)
    result_rows = []
    history_series = {}
    source_status = []

    for code in DISPLAY_ORDER:
        product = PRODUCTS[code]

        if product["type"] == "ETF":
            display_series, performance_series = get_etf_series(
                product["ticker"]
            )

            display_statistics = calculate_performance(
                display_series
            )

            performance_statistics = calculate_performance(
                performance_series
            )

            statistics = performance_statistics.copy()
            statistics["latest"] = display_statistics["latest"]
            statistics["date"] = display_statistics["date"]

            history_series[code] = performance_series

            if display_series.empty:
                status = "ETF資料讀取失敗"
            else:
                status = "Yahoo Finance"

        else:
            fund_series = get_fund_series(
                fund_data,
                code,
            )

          statistics = calculate_performance(
                fund_series
            )

            history_series[code] = fund_series

            if fund_series.empty:
                status = "等待 fund_nav_history.csv"
            else:
                status = "fund_nav_history.csv"

        result_rows.append(
            {
                "代號": code,
                "基金名稱": product["name"],
                "類型": product["type"],
                "系列": product["group"],
                "最新淨值": statistics["latest"],
                "資料日期": statistics["date"],
                "Daily": statistics["Daily"],
                "MTD": statistics["MTD"],
                "YTD": statistics["YTD"],
                "1Y": statistics["1Y"],
                "2Y": statistics["2Y"],
                "3Y": statistics["3Y"],
            }
        )

        source_status.append(
            {
                "代號": code,
                "基金名稱": product["name"],
                "資料來源": status,
            }
        )

    result_df = pd.DataFrame(result_rows)
    source_df = pd.DataFrame(source_status)

    return result_df, history_series, source_df


# ============================================================
# 8. 走勢圖函式
# ============================================================

def get_period_start_date(series, period_name):
    """
    依使用者選擇的期間決定走勢圖開始日期。
    """
    if series is None or series.empty:
        return None

end_date = pd.Timestamp(series.index[-1]).normalize()

    if period_name == "1M":
        return end_date - pd.DateOffset(months=1)

    if period_name == "3M":
        return end_date - pd.DateOffset(months=3)

    if period_name == "6M":
        return end_date - pd.DateOffset(months=6)

    if period_name == "YTD":
        return pd.Timestamp(
            year=end_date.year,
            month=1,
            day=1,
        )

    if period_name == "1Y":
        return end_date - pd.DateOffset(years=1)

    if period_name == "2Y":
        return end_date - pd.DateOffset(years=2)

    if period_name == "3Y":
        return end_date - pd.DateOffset(years=3)

    return pd.Timestamp(series.index[0]).normalize()


def normalize_series(series, period_name):
    """
    將走勢標準化為起始值100，方便比較不同價格尺度。
    """
    series = clean_price_series(series)

    if series.empty:
        return pd.Series(dtype=float)

    start_date = get_period_start_date(
        series,
        period_name,
    )

    selected = series.loc[
        series.index >= start_date
    ].copy()

 if selected.empty:
        return pd.Series(dtype=float)

    first_value = selected.iloc[0]

    if pd.isna(first_value) or first_value == 0:
        return pd.Series(dtype=float)

    return selected / first_value * 100


def build_performance_chart(
    selected_codes,
    history_series,
    period_name,
):
    """
    建立標準化績效走勢圖。
    """
    color_map = {
        "00981A": "#e53935",
        "A09012": "#fb8c00",
        "00407A": "#1565c0",
        "A41209": "#00acc1",
        "009816": "#7b1fa2",
        "0050": "#546e7a",
    }

    figure = go.Figure()
    effective_count = 0

    for code in selected_codes:
        series = history_series.get(
            code,
            pd.Series(dtype=float),
        )

        normalized = normalize_series(
            series,
            period_name,
        )

        if normalized.empty:
            continue

        product_name = PRODUCTS[code]["name"]

        figure.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=f"{code} {product_name}",
                line={
                    "width": 2.5,
                    "color": color_map.get(code),
                },
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "標準化績效：%{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

     effective_count += 1

    figure.update_layout(
        height=530,
        margin={
            "l": 20,
            "r": 20,
            "t": 40,
            "b": 20,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        xaxis={
            "title": "",
            "showgrid": False,
            "rangeslider": {
                "visible": False
            },
        },
        yaxis={
            "title": "標準化績效（起始值＝100）",
            "showgrid": True,
            "gridcolor": "#e9edf2",
            "zeroline": False,
        },
    )

    figure.add_hline(
        y=100,
        line_width=1,
        line_dash="dash",
        line_color="#98a2b3",
    )

    return figure, effective_count


# ============================================================
# 9. 側邊欄
# ============================================================

with st.sidebar:
    st.markdown("## 🔎 篩選條件")

    group_options = [
        "全部",
        "統一系列",
        "凱基系列",
        "市場基準",
    ]

    selected_group = st.selectbox(
        "基金系列",
        group_options,
        index=0,
    )

   type_options = [
        "全部",
        "ETF",
        "共同基金",
    ]

    selected_type = st.selectbox(
        "商品類型",
        type_options,
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📈 走勢比較")

    default_chart_codes = [
        "00981A",
        "A09012",
        "00407A",
        "A41209",
        "009816",
        "0050",
    ]

    chart_codes = st.multiselect(
        "選擇比較商品",
        options=DISPLAY_ORDER,
        default=default_chart_codes,
        format_func=lambda code: (
            f"{code}｜{PRODUCTS[code]['name']}"
        ),
    )

    chart_period = st.radio(
        "比較期間",
        options=[
            "1M",
            "3M",
            "6M",
            "YTD",
            "1Y",
            "2Y",
            "3Y",
            "全部",
        ],
        index=3,
        horizontal=True,
    )

    st.markdown("---")

    if st.button(
        "🔄 重新整理資料",
        use_container_width=True,
        type="primary",
    ):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "ETF資料暫存1小時。按下重新整理後，"
        "會重新向資料來源取得資料。"
    )


# ============================================================
# 10. 網頁標題
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <h1>📊 台灣基金績效比較 Dashboard</h1>
        <p>
            比較主動式ETF、共同基金及0050市場基準，
            快速查看 Daily、MTD、YTD 與中長期績效。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 11. 載入資料
# ============================================================

with st.spinner("正在更新基金與ETF資料，請稍候..."):
    dashboard_df, history_data, source_df = (
        build_dashboard_data()
    )


# ============================================================
# 12. 套用篩選
# ============================================================

filtered_df = dashboard_df.copy()

if selected_group != "全部":
    filtered_df = filtered_df.loc[
        filtered_df["系列"] == selected_group
    ]

if selected_type != "全部":
    filtered_df = filtered_df.loc[
        filtered_df["類型"] == selected_type
    ]

filtered_df = filtered_df.copy()

order_map = {
    code: position
    for position, code in enumerate(DISPLAY_ORDER)
}

filtered_df["_排序"] = (
    filtered_df["代號"]
    .map(order_map)
)

filtered_df = (
    filtered_df
    .sort_values("_排序")
    .drop(columns=["_排序"])
)


# ============================================================
# 13. 摘要資訊卡
# ============================================================

valid_dates = dashboard_df["資料日期"].dropna()

if valid_dates.empty:
    latest_data_date = "尚無資料"
else:
    latest_data_date = (
        pd.to_datetime(valid_dates)
        .max()
        .strftime("%Y/%m/%d")
    )

available_products = int(
    dashboard_df["最新淨值"].notna().sum()
)

valid_daily = dashboard_df.dropna(
    subset=["Daily"]
)

best_daily_name = "尚無資料"
best_daily_return = None

if not valid_daily.empty:
    best_daily_index = valid_daily["Daily"].idxmax()
    best_daily_row = valid_daily.loc[best_daily_index]

    best_daily_name = (
        f"{best_daily_row['代號']} "
        f"{best_daily_row['基金名稱']}"
    )

    best_daily_return = best_daily_row["Daily"]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">最新資料日期</div>
            <div class="info-value">{latest_data_date}</div>
            <div class="info-note">
                各商品可能因交易日或資料來源而略有差異
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">已有資料商品</div>
            <div class="info-value">
                {available_products}／{len(DISPLAY_ORDER)}
            </div>
            <div class="info-note">
                ETF與共同基金合計
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    best_daily_text = format_return(
        best_daily_return
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">單日表現最佳</div>
            <div class="info-value"
                 style="font-size: 18px;">
                {best_daily_name}
            </div>
            <div class="info-note">
                Daily：{best_daily_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 14. 績效比較表
# ============================================================

st.markdown(
    '<div class="section-title">基金績效總覽</div>',
    unsafe_allow_html=True,
)

display_df = filtered_df[
    [
        "代號",
        "基金名稱",
        "最新淨值",
        "Daily",
        "MTD",
        "YTD",
        "1Y",
        "2Y",
        "3Y",
    ]
].copy()

display_df["最新淨值"] = (
    display_df["最新淨值"]
    .apply(format_nav)
)

return_columns = [
    "Daily",
    "MTD",
    "YTD",
    "1Y",
    "2Y",
    "3Y",
]

for column in return_columns:
    display_df[column] = (
        display_df[column]
        .apply(format_return)
    )

styled_display_df = (
    display_df.style
    .map(
        performance_style,
        subset=return_columns,
    )
    .set_properties(
        subset=["代號"],
        **{
            "font-weight": "700",
            "color": "#176b87",
        },
    )
       .set_properties(
        subset=["基金名稱"],
        **{
            "font-weight": "600",
            "color": "#172b4d",
        },
    )
)

st.dataframe(
    styled_display_df,
    use_container_width=True,
    hide_index=True,
    height=(len(display_df) * 42) + 42,
    column_config={
        "代號": st.column_config.TextColumn(
            "代號",
            width="small",
        ),
        "基金名稱": st.column_config.TextColumn(
            "基金名稱",
            width="large",
        ),
        "最新淨值": st.column_config.TextColumn(
            "最新淨值",
            width="small",
        ),
        "Daily": st.column_config.TextColumn(
            "Daily",
            help="相較前一筆交易日資料的報酬率",
        ),
        "MTD": st.column_config.TextColumn(
            "MTD",
            help="相較上月底的報酬率",
        ),
        "YTD": st.column_config.TextColumn(
            "YTD",
            help="相較前一年度最後一筆資料的報酬率",
        ),
        "1Y": st.column_config.TextColumn(
            "1Y",
            help="近一年報酬率",
        ),
        "2Y": st.column_config.TextColumn(
            "2Y",
            help="近兩年累積報酬率",
        ),
        "3Y": st.column_config.TextColumn(
            "3Y",
            help="近三年累積報酬率",
        ),
    },
)

st.markdown(
    """
    <div class="data-note">
        <strong>閱讀方式：</strong>
        紅色代表正報酬，綠色代表負報酬。
        「-」表示商品成立期間不足，或現有歷史資料不足以計算該期間績效。
        ETF績效優先以調整後收盤價計算，以降低除息造成的績效失真。
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 15. 下載績效資料
# ============================================================

download_df = filtered_df[
    [
        "代號",
        "基金名稱",
        "類型",
        "系列",
        "最新淨值",
        "資料日期",
        "Daily",
        "MTD",
        "YTD",
        "1Y",
        "2Y",
        "3Y",
    ]
].copy()

if not download_df.empty:
    download_df["資料日期"] = pd.to_datetime(
        download_df["資料日期"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

csv_data = download_df.to_csv(
    index=False,
    encoding="utf-8-sig",
)

st.download_button(
    label="⬇️ 下載目前績效表 CSV",
    data=csv_data,
    file_name=(
        f"fund_performance_"
        f"{datetime.now().strftime('%Y%m%d')}.csv"
    ),
    mime="text/csv",
)


# ============================================================
# 16. 績效走勢圖
# ============================================================

st.markdown(
    '<div class="section-title">績效走勢比較</div>',
    unsafe_allow_html=True,
)

if not chart_codes:
    st.info("請在左側至少選擇一檔商品。")

else:
    chart, chart_count = build_performance_chart(
        selected_codes=chart_codes,
        history_series=history_data,
        period_name=chart_period,
    )

    if chart_count == 0:
        st.warning(
            "目前選擇的商品沒有足夠歷史資料可繪製走勢圖。"
        )

    else:
        st.plotly_chart(
            chart,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
            },
        )

        st.caption(
            "走勢圖將各商品於所選期間的第一筆有效資料設定為100，"
            "用來比較相對績效，不代表實際淨值或市價。"
        )


# ============================================================
# 17. 各基金詳細資料
# ============================================================

st.markdown(
    '<div class="section-title">各基金詳細資料</div>',
    unsafe_allow_html=True,
)

detail_code = st.selectbox(
    "選擇基金",
    options=DISPLAY_ORDER,
    format_func=lambda code: (
        f"{code}｜{PRODUCTS[code]['name']}"
    ),
)

detail_row = dashboard_df.loc[
    dashboard_df["代號"] == detail_code
].iloc[0]

detail_col1, detail_col2, detail_col3, detail_col4 = (
    st.columns(4)
)

detail_col1.metric(
    "最新淨值",
    format_nav(detail_row["最新淨值"]),
)

detail_col2.metric(
    "Daily",
    format_return(detail_row["Daily"]),
)

detail_col3.metric(
    "MTD",
    format_return(detail_row["MTD"]),
)

detail_col4.metric(
    "YTD",
    format_return(detail_row["YTD"]),
)

detail_col5, detail_col6, detail_col7 = st.columns(3)

detail_col5.metric(
    "1Y",
    format_return(detail_row["1Y"]),
)

detail_col6.metric(
    "2Y",
    format_return(detail_row["2Y"]),
)

detail_col7.metric(
    "3Y",
    format_return(detail_row["3Y"]),
)

detail_date = detail_row["資料日期"]

if pd.isna(detail_date):
    detail_date_text = "無資料"
else:
    detail_date_text = pd.Timestamp(
        detail_date
    ).strftime("%Y/%m/%d")

st.caption(
    f"商品類型：{detail_row['類型']}｜"
    f"基金系列：{detail_row['系列']}｜"
    f"資料日期：{detail_date_text}"
)


# ============================================================
# 18. 資料來源狀態
# ============================================================

with st.expander("查看資料來源與讀取狀態"):
    st.dataframe(
        source_df,
        use_container_width=True,
        hide_index=True,
    )

    if not LOCAL_FUND_FILE.exists():
        st.warning(
            "目前 GitHub 專案內找不到 "
            "fund_nav_history.csv，"
            "因此 A09012 與 A41209 暫時不會顯示淨值。"
        )


# ============================================================
# 19. 計算說明
# ============================================================

with st.expander("報酬率計算方式"):
    st.markdown(
        """
        - **Daily**：最新值 ÷ 前一筆交易日數值 − 1
        - **MTD**：最新值 ÷ 上月底最後一筆有效數值 − 1
        - **YTD**：最新值 ÷ 前一年度最後一筆有效數值 − 1
        - **1Y**：最新值 ÷ 一年前最近一筆有效數值 − 1
        - **2Y**：最新值 ÷ 兩年前最近一筆有效數值 − 1
        - **3Y**：最新值 ÷ 三年前最近一筆有效數值 − 1

        ETF績效優先使用調整後收盤價計算；表格中的「最新淨值」
        則顯示ETF實際收盤價或共同基金最新淨值。
        """
    )


# ============================================================
# 20. 頁尾
# ============================================================

st.markdown(
    f"""
    <div class="footer-note">
        最後更新時間：
        {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}
        <br>
        本網站資訊僅供資料整理與績效比較，不構成任何投資建議。
        實際淨值、成交價格與報酬率仍應以證券交易所、
        基金公司及基金公開資訊來源為準。
    </div>
    """,
    unsafe_allow_html=True,
)
