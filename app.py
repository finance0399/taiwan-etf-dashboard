from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# 1. 網頁基本設定
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
# 3. 網頁樣式
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }

    .block-container {
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
            #0b3c5d,
            #176b87,
            #64ccc5
        );
    }

    .dashboard-header h1 {
        margin: 0;
        font-size: 32px;
    }

    .dashboard-header p {
        margin-top: 8px;
        margin-bottom: 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e4e9ef;
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. 資料處理函式
# ============================================================

def clean_series(series):
    """清理價格或淨值序列。"""

    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    result = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if result.empty:
        return pd.Series(dtype=float)

    date_index = pd.DatetimeIndex(result.index)

    if date_index.tz is not None:
        date_index = date_index.tz_localize(None)

    result.index = date_index.normalize()

    result = result[
        ~result.index.duplicated(keep="last")
    ]

    return result.sort_index()


def get_value_on_or_before(series, target_date):
    """取得指定日期或之前最近一筆有效資料。"""

    if series is None or series.empty:
        return None

    target_date = pd.Timestamp(
        target_date
    ).normalize()

    available = series.loc[
        series.index <= target_date
    ]

    if available.empty:
        return None

    return float(available.iloc[-1])


def calculate_return(latest_value, base_value):
    """計算報酬率。"""

    if latest_value is None or base_value is None:
        return None

    if pd.isna(latest_value) or pd.isna(base_value):
        return None

    if float(base_value) == 0:
        return None

    return (
        float(latest_value) / float(base_value) - 1
    ) * 100


def calculate_performance(series):
    """計算最新值與各期間報酬率。"""

    series = clean_series(series)

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
    latest_date = pd.Timestamp(series.index[-1])

    previous_value = None

    if len(series) >= 2:
        previous_value = float(series.iloc[-2])

    month_start = latest_date.replace(day=1)

    previous_month_end = (
        month_start - pd.Timedelta(days=1)
    )

    year_start = pd.Timestamp(
        year=latest_date.year,
        month=1,
        day=1,
    )

    previous_year_end = (
        year_start - pd.Timedelta(days=1)
    )

    month_base = get_value_on_or_before(
        series,
        previous_month_end,
    )

    year_base = get_value_on_or_before(
        series,
        previous_year_end,
    )

    one_year_base = get_value_on_or_before(
        series,
        latest_date - pd.DateOffset(years=1),
    )

    two_year_base = get_value_on_or_before(
        series,
        latest_date - pd.DateOffset(years=2),
    )

    three_year_base = get_value_on_or_before(
        series,
        latest_date - pd.DateOffset(years=3),
    )

    return {
        "latest": latest_value,
        "date": latest_date,
        "Daily": calculate_return(
            latest_value,
            previous_value,
        ),
        "MTD": calculate_return(
            latest_value,
            month_base,
        ),
        "YTD": calculate_return(
            latest_value,
            year_base,
        ),
        "1Y": calculate_return(
            latest_value,
            one_year_base,
        ),
        "2Y": calculate_return(
            latest_value,
            two_year_base,
        ),
        "3Y": calculate_return(
            latest_value,
            three_year_base,
        ),
    }


def format_nav(value):
    """格式化最新淨值或價格。"""

    if value is None or pd.isna(value):
        return "無資料"

    return f"{float(value):,.2f}"


def format_return(value):
    """格式化報酬率。"""

    if value is None or pd.isna(value):
        return "-"

    value = float(value)

    if value > 0:
        return f"+{value:.2f}%"

    return f"{value:.2f}%"


def performance_style(value):
    """正報酬顯示紅色，負報酬顯示綠色。"""

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
# 5. ETF 資料
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_etf_data(ticker):
    """從 Yahoo Finance 下載 ETF 歷史價格。"""

    try:
        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = (
                data.columns.get_level_values(0)
            )

        date_index = pd.DatetimeIndex(data.index)

        if date_index.tz is not None:
            date_index = date_index.tz_localize(None)

        data.index = date_index.normalize()

        data = data[
            ~data.index.duplicated(keep="last")
        ]

        return data.sort_index()

    except Exception:
        return pd.DataFrame()


def get_etf_series(ticker):
    """取得 ETF 收盤價與績效計算序列。"""

    data = download_etf_data(ticker)

    if data.empty:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    close_series = pd.Series(dtype=float)
    performance_series = pd.Series(dtype=float)

    if "Close" in data.columns:
        close_series = clean_series(
            data["Close"]
        )

    if "Adj Close" in data.columns:
        performance_series = clean_series(
            data["Adj Close"]
        )

    if performance_series.empty:
        performance_series = close_series.copy()

    return close_series, performance_series


# ============================================================
# 6. 共同基金 CSV 資料
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_fund_data(file_modified_time=None):
    """讀取 fund_nav_history.csv。"""

    del file_modified_time

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
        str(column).strip().lower()
        for column in data.columns
    ]

    required_columns = {
        "date",
        "code",
        "nav",
    }

    if not required_columns.issubset(data.columns):
        return empty_data

    result = data[
        ["date", "code", "nav"]
    ].copy()

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
        result["code"].isin(
            ["A09012", "A41209"]
        )
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
    """取得指定共同基金的淨值序列。"""

    if fund_data.empty:
        return pd.Series(dtype=float)

    selected = fund_data.loc[
        fund_data["code"] == code,
        ["date", "nav"],
    ].copy()

    if selected.empty:
        return pd.Series(dtype=float)

    selected = selected.set_index("date")

    return clean_series(selected["nav"])


# ============================================================
# 7. 建立儀表板資料
# ============================================================

def build_dashboard_data():
    """建立所有商品的績效資料。"""

    modified_time = None

    if LOCAL_FUND_FILE.exists():
        modified_time = (
            LOCAL_FUND_FILE.stat().st_mtime
        )

    fund_data = load_fund_data(modified_time)

    result_rows = []
    history_series = {}
    source_status = []

    for code in DISPLAY_ORDER:
        product = PRODUCTS[code]

        if product["type"] == "ETF":
            display_series, performance_series = (
                get_etf_series(product["ticker"])
            )

            display_statistics = (
                calculate_performance(display_series)
            )

            statistics = calculate_performance(
                performance_series
            )

            statistics["latest"] = (
                display_statistics["latest"]
            )

            statistics["date"] = (
                display_statistics["date"]
            )

            history_series[code] = (
                performance_series
            )

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
                status = "等待基金淨值資料"
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

    return (
        result_df,
        history_series,
        source_df,
    )


# ============================================================
# 8. 走勢圖
# ============================================================

def normalize_series(series, period_name):
    """將所選期間第一筆資料設定為100。"""

    series = clean_series(series)

    if series.empty:
        return pd.Series(dtype=float)

    end_date = pd.Timestamp(
        series.index[-1]
    ).normalize()

    period_start_dates = {
        "1M": end_date - pd.DateOffset(months=1),
        "3M": end_date - pd.DateOffset(months=3),
        "6M": end_date - pd.DateOffset(months=6),
        "YTD": pd.Timestamp(
            year=end_date.year,
            month=1,
            day=1,
        ),
        "1Y": end_date - pd.DateOffset(years=1),
        "2Y": end_date - pd.DateOffset(years=2),
        "3Y": end_date - pd.DateOffset(years=3),
    }

    start_date = period_start_dates.get(
        period_name,
        series.index[0],
    )

    selected = series.loc[
        series.index >= start_date
    ].copy()

    if selected.empty:
        return pd.Series(dtype=float)

    first_value = float(selected.iloc[0])

    if first_value == 0:
        return pd.Series(dtype=float)

    return selected / first_value * 100


def build_performance_chart(
    selected_codes,
    history_series,
    period_name,
):
    """建立績效走勢圖。"""

    colors = {
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

        figure.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=(
                    f"{code} "
                    f"{PRODUCTS[code]['name']}"
                ),
                line={
                    "width": 2.5,
                    "color": colors.get(code),
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
        height=520,
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
        },
        yaxis={
            "title": "標準化績效（起始值＝100）",
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

    selected_group = st.selectbox(
        "基金系列",
        [
            "全部",
            "統一系列",
            "凱基系列",
            "市場基準",
        ],
    )

    selected_type = st.selectbox(
        "商品類型",
        [
            "全部",
            "ETF",
            "共同基金",
        ],
    )

    st.markdown("---")
    st.markdown("### 📈 走勢比較")

    chart_codes = st.multiselect(
        "選擇比較商品",
        options=DISPLAY_ORDER,
        default=DISPLAY_ORDER,
        format_func=lambda code: (
            f"{code}｜{PRODUCTS[code]['name']}"
        ),
    )

    chart_period = st.radio(
        "比較期間",
        [
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


# ============================================================
# 10. 網頁標題
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <h1>📊 台灣基金績效比較 Dashboard</h1>
        <p>
            比較主動式ETF、共同基金及0050市場基準，
            查看 Daily、MTD、YTD 與中長期績效。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 11. 載入資料
# ============================================================

with st.spinner("正在更新資料，請稍候..."):
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


# ============================================================
# 13. 績效總覽
# ============================================================

st.subheader("基金績效總覽")

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
    display_df["最新淨值"].apply(format_nav)
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
        display_df[column].apply(format_return)
    )

styled_df = display_df.style.map(
    performance_style,
    subset=return_columns,
)

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "紅色代表正報酬，綠色代表負報酬；"
    "「-」表示歷史資料不足。"
)


# ============================================================
# 14. 下載資料
# ============================================================

download_df = filtered_df.copy()

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
# 15. 績效走勢圖
# ============================================================

st.subheader("績效走勢比較")

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
            "目前沒有足夠資料可繪製走勢圖。"
        )

    else:
        st.plotly_chart(
            chart,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
            },
        )

        st.caption(
            "所選期間第一筆有效資料設定為100，"
            "用於比較各商品相對績效。"
        )


# ============================================================
# 16. 各基金詳細資料
# ============================================================

st.subheader("各基金詳細資料")

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

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "最新淨值",
    format_nav(detail_row["最新淨值"]),
)

col2.metric(
    "Daily",
    format_return(detail_row["Daily"]),
)

col3.metric(
    "MTD",
    format_return(detail_row["MTD"]),
)

col4.metric(
    "YTD",
    format_return(detail_row["YTD"]),
)

col5, col6, col7 = st.columns(3)

col5.metric(
    "1Y",
    format_return(detail_row["1Y"]),
)

col6.metric(
    "2Y",
    format_return(detail_row["2Y"]),
)

col7.metric(
    "3Y",
    format_return(detail_row["3Y"]),
)


# ============================================================
# 17. 資料來源狀態
# ============================================================

with st.expander("查看資料來源與讀取狀態"):
    st.dataframe(
        source_df,
        use_container_width=True,
        hide_index=True,
    )

    fund_rows = load_fund_data(
        LOCAL_FUND_FILE.stat().st_mtime
        if LOCAL_FUND_FILE.exists()
        else None
    )

    if fund_rows.empty:
        st.warning(
            "fund_nav_history.csv 目前沒有有效淨值資料，"
            "因此 A09012 與 A41209 會顯示無資料。"
        )


# ============================================================
# 18. 頁尾
# ============================================================

st.caption(
    f"最後更新時間："
    f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}｜"
    "本網站資訊僅供資料整理，不構成投資建議。"
)
