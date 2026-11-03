from __future__ import annotations

import re
import time
import unicodedata
from datetime import date, datetime, time as dtime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# 頁面與共用設定
# ============================================================

st.set_page_config(
    page_title="ETF及共同基金績效比較",
    page_icon="📊",
    layout="wide",
)

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
LOCAL_FUND_FILE = Path("fund_nav_history.csv")

PERIOD_COLUMNS = [
    "Daily",
    "MTD",
    "YTD",
    "1Y累積",
    "2Y累積",
    "3Y累積",
]

MIN_RISK_OBSERVATIONS = 60
EXPECTED_FUND_CODES = {"A09012", "A41209"}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


PRODUCTS = {
    "00981A": {
        "name": "00981A 統一台股增長主動式ETF",
        "type": "ETF",
    },
    "A09012": {
        "name": "統一奔騰基金",
        "type": "FUND",
        "urls": [
            "https://www.ezmoney.com.tw/funds/value/36",
            (
                "https://fund.hncb.com.tw/w/wr/"
                "wr02_ACPS10-009007.djhtm"
            ),
            (
                "https://www.moneydj.com/funddj/ya/"
                "yp010000.djhtm?a=ACPS10"
            ),
        ],
    },
    "00407A": {
        "name": "00407A 凱基台灣主動式ETF",
        "type": "ETF",
    },
    "A41209": {
        "name": "凱基台灣精五門基金",
        "type": "FUND",
        "urls": [
            (
                "https://fund.hncb.com.tw/w/wr/"
                "wr02_ACKG15-014013.djhtm"
            ),
            (
                "https://chbfund.moneydj.com/w/wr/"
                "wr02.djhtm?A=ACKG15-9808"
            ),
            (
                "https://tw.stock.yahoo.com/fund/"
                "F0000040SC:FO/history"
            ),
        ],
    },
    "009816": {
        "name": "009816 凱基台灣TOP 50 ETF",
        "type": "ETF",
    },
    "0050": {
        "name": "0050 元大台灣50",
        "type": "ETF",
    },
}


ETF_SPLIT_CONFIG = {
    "0050": [
        {
            "effective_date": pd.Timestamp("2025-06-18"),
            "ratio": 4.0,
        }
    ]
}


# ============================================================
# 共用資料工具
# ============================================================

def empty_history():
    return pd.DataFrame(
        columns=["date", "nav"]
    )


def safe_float(value):
    try:
        text = (
            str(value)
            .strip()
            .replace(",", "")
            .replace("%", "")
            .replace("元", "")
        )

        if text in {
            "",
            "-",
            "--",
            "---",
            "None",
            "nan",
            "NaN",
        }:
            return np.nan

        return float(text)

    except Exception:
        return np.nan


def normalize_history(data):
    if (
        data is None
        or data.empty
        or not {"date", "nav"}.issubset(data.columns)
    ):
        return empty_history()

    result = data[
        ["date", "nav"]
    ].copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["nav"] = (
        result["nav"]
        .map(safe_float)
    )

    try:
        result["date"] = (
            result["date"]
            .dt.tz_localize(None)
        )

    except Exception:
        pass

    result = result.dropna(
        subset=["date", "nav"]
    )

    result = result[
        result["nav"] > 0
    ]

    result["date"] = (
        result["date"]
        .dt.normalize()
    )

    result = (
        result.sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


def parse_roc_or_ad_date(value):
    match = re.search(
        r"(\d{2,4})[/-]"
        r"(\d{1,2})[/-]"
        r"(\d{1,2})",
        str(value),
    )

    if not match:
        return pd.NaT

    year, month, day = map(
        int,
        match.groups(),
    )

    if year < 1911:
        year += 1911

    try:
        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
        )

    except Exception:
        return pd.NaT


def calculate_return(current_value, base_value):
    if (
        pd.isna(current_value)
        or pd.isna(base_value)
        or base_value == 0
    ):
        return np.nan

    return (
        current_value / base_value - 1
    ) * 100


# ============================================================
# 顯示格式工具
# ============================================================

def format_number(value):
    if pd.isna(value):
        return "-"

    return (
        f"{value:,.4f}"
        .rstrip("0")
        .rstrip(".")
    )


def format_percentage(value):
    if pd.isna(value):
        return "-"

    return f"{value:+.2f}%"


def format_daily(value):
    if pd.isna(value):
        return "-"

    if value > 0:
        return f"▲ {value:+.2f}%"

    if value < 0:
        return f"▼ {value:.2f}%"

    return "0.00%"


def return_style(value):
    if pd.isna(value):
        return "#374151", False

    if value > 0:
        return "#d62728", True

    if value < 0:
        return "#138a36", True

    return "#374151", False


def display_character_width(value):
    width = 0

    for character in str(value):
        if (
            unicodedata.east_asian_width(character)
            in {"W", "F", "A"}
        ):
            width += 2
        else:
            width += 1

    return width


def automatic_column_width(
    dataframe,
    column_name,
    extra_characters=5,
    minimum_pixels=75,
    maximum_pixels=360,
):
    values = [
        column_name,
        *dataframe[column_name]
        .astype(str)
        .tolist(),
    ]

    longest_width = max(
        display_character_width(value)
        for value in values
    )

    calculated_width = (
        longest_width
        + extra_characters
    ) * 9

    return max(
        minimum_pixels,
        min(
            calculated_width,
            maximum_pixels,
        ),
    )


def escape_html(value):
    text = str(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def make_table_cell(
    value,
    alignment="left",
    color=None,
    bold=False,
):
    style = (
        f"text-align:{alignment};"
        "white-space:nowrap;"
        "padding:10px 12px;"
        "border-bottom:1px solid #e5e7eb;"
        "font-variant-numeric:tabular-nums;"
    )

    if color:
        style += f"color:{color};"

    if bold:
        style += "font-weight:700;"

    return (
        f'<td style="{style}">'
        f"{escape_html(value)}"
        "</td>"
    )


# ============================================================
# 下次排程
# ============================================================

def get_next_update_text():
    now = datetime.now(TAIPEI_TZ)

    for day_offset in range(8):
        target_date = (
            now.date()
            + timedelta(days=day_offset)
        )

        if target_date.weekday() >= 5:
            continue

        for target_time in [
            dtime(19, 0),
            dtime(21, 30),
        ]:
            candidate = datetime.combine(
                target_date,
                target_time,
                tzinfo=TAIPEI_TZ,
            )

            if candidate <= now:
                continue

            if target_date == now.date():
                prefix = "今日"

            elif (
                target_date
                == now.date()
                + timedelta(days=1)
            ):
                prefix = "明日"

            else:
                weekdays = "一二三四五六日"

                prefix = (
                    f"{candidate:%m/%d}"
                    f"（週"
                    f"{weekdays[candidate.weekday()]}"
                    f"）"
                )

            return (
                f"{prefix} "
                f"{candidate:%H:%M}"
            )

    return "-"


# ============================================================
# ETF資料抓取
# ============================================================

def adjust_etf_price(
    stock_code,
    trade_date,
    close_price,
):
    if (
        pd.isna(trade_date)
        or pd.isna(close_price)
    ):
        return close_price

    adjusted_price = float(close_price)

    for event in ETF_SPLIT_CONFIG.get(
        stock_code,
        [],
    ):
        if (
            trade_date
            < event["effective_date"]
        ):
            adjusted_price /= event["ratio"]

    return adjusted_price


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_twse_month(
    stock_code,
    year,
    month,
):
    url = (
        "https://www.twse.com.tw/"
        "exchangeReport/STOCK_DAY"
    )

    params = {
        "response": "json",
        "date": (
            f"{year}"
            f"{month:02d}"
            "01"
        ),
        "stockNo": stock_code,
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=45,
        )

        response.raise_for_status()

        rows = response.json().get(
            "data",
            [],
        )

    except Exception:
        return empty_history()

    records = []

    for row in rows:
        if len(row) < 7:
            continue

        trade_date = (
            parse_roc_or_ad_date(
                row[0]
            )
        )

        close_price = safe_float(
            row[6]
        )

        adjusted_price = adjust_etf_price(
            stock_code,
            trade_date,
            close_price,
        )

        if (
            pd.notna(trade_date)
            and pd.notna(adjusted_price)
        ):
            records.append(
                {
                    "date": trade_date,
                    "nav": adjusted_price,
                }
            )

    return normalize_history(
        pd.DataFrame(records)
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_etf_history(
    stock_code,
    selected_date,
):
    end_date = pd.Timestamp(
        selected_date
    )

    start_date = (
        end_date
        - pd.DateOffset(
            years=3,
            months=2,
        )
    )

    current_month = pd.Timestamp(
        start_date.year,
        start_date.month,
        1,
    )

    ending_month = pd.Timestamp(
        end_date.year,
        end_date.month,
        1,
    )

    frames = []

    while current_month <= ending_month:
        monthly_data = fetch_twse_month(
            stock_code,
            current_month.year,
            current_month.month,
        )

        if not monthly_data.empty:
            frames.append(monthly_data)

        current_month += (
            pd.offsets.MonthBegin(1)
        )

    if not frames:
        return empty_history()

    history = normalize_history(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    return history[
        history["date"] <= end_date
    ].reset_index(drop=True)


# ============================================================
# CSV讀取及品質檢核
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_fund_csv():
    quality_report = {
        "exists": LOCAL_FUND_FILE.exists(),
        "columns_ok": False,
        "invalid_dates": 0,
        "invalid_nav": 0,
        "duplicate_rows": 0,
        "unknown_codes": [],
        "missing_codes": [],
    }

    empty_result = pd.DataFrame(
        columns=[
            "date",
            "code",
            "name",
            "nav",
        ]
    )

    if not LOCAL_FUND_FILE.exists():
        return empty_result, quality_report

    raw_data = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp950",
        "big5",
    ]:
        try:
            raw_data = pd.read_csv(
                LOCAL_FUND_FILE,
                encoding=encoding,
                dtype={
                    "code": str,
                },
            )

            break

        except Exception:
            continue

    if raw_data is None:
        return empty_result, quality_report

    raw_data.columns = [
        str(column).strip()
        for column in raw_data.columns
    ]

    raw_data = raw_data.rename(
        columns={
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
            "基金淨值": "nav",
            "NAV": "nav",
            "Nav": "nav",
        }
    )

    required_columns = {
        "date",
        "code",
        "name",
        "nav",
    }

    quality_report["columns_ok"] = (
        required_columns.issubset(
            raw_data.columns
        )
    )

    if not quality_report["columns_ok"]:
        return empty_result, quality_report

    parsed_dates = pd.to_datetime(
        raw_data["date"],
        errors="coerce",
    )

    parsed_nav = pd.to_numeric(
        raw_data["nav"],
        errors="coerce",
    )

    quality_report["invalid_dates"] = int(
        parsed_dates.isna().sum()
    )

    quality_report["invalid_nav"] = int(
        (
            parsed_nav.isna()
            | (parsed_nav <= 0)
        ).sum()
    )

    raw_data["date"] = parsed_dates
    raw_data["nav"] = parsed_nav

    raw_data["code"] = (
        raw_data["code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    raw_data["name"] = (
        raw_data["name"]
        .astype(str)
        .str.strip()
    )

    quality_report[
        "duplicate_rows"
    ] = int(
        raw_data.duplicated(
            subset=[
                "code",
                "date",
            ]
        ).sum()
    )

    valid_codes = set(
        raw_data["code"]
        .dropna()
        .unique()
    )

    quality_report[
        "unknown_codes"
    ] = sorted(
        valid_codes
        - EXPECTED_FUND_CODES
    )

    quality_report[
        "missing_codes"
    ] = sorted(
        EXPECTED_FUND_CODES
        - valid_codes
    )

    raw_data = raw_data.dropna(
        subset=[
            "date",
            "code",
            "nav",
        ]
    )

    raw_data = raw_data[
        raw_data["nav"] > 0
    ]

    raw_data = (
        raw_data
        .sort_values(
            [
                "code",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "code",
                "date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return (
        raw_data[
            [
                "date",
                "code",
                "name",
                "nav",
            ]
        ],
        quality_report,
    )


def get_local_fund_history(
    fund_code,
    fund_name,
    local_data,
):
    if local_data.empty:
        return empty_history()

    result = local_data[
        local_data["code"]
        == fund_code
    ].copy()

    if result.empty:
        result = local_data[
            local_data["name"]
            .str.contains(
                fund_name,
                regex=False,
                na=False,
            )
        ].copy()

    if result.empty:
        return empty_history()

    return normalize_history(
        result[
            [
                "date",
                "nav",
            ]
        ]
    )


# ============================================================
# 共同基金線上資料
# ============================================================

def flatten_column_name(column):
    if isinstance(column, tuple):
        return " ".join(
            map(str, column)
        ).strip()

    return str(column).strip()


def find_date_nav_columns(table):
    date_column = None
    nav_column = None

    for column in table.columns:
        column_text = flatten_column_name(
            column
        )

        if (
            date_column is None
            and "日期" in column_text
        ):
            date_column = column

        if (
            nav_column is None
            and "淨值" in column_text
            and "日期" not in column_text
            and "最高" not in column_text
            and "最低" not in column_text
            and "漲跌" not in column_text
        ):
            nav_column = column

    return date_column, nav_column


def parse_fund_tables(html_text):
    records = []

    try:
        tables = pd.read_html(
            StringIO(html_text)
        )

    except Exception:
        tables = []

    for table in tables:
        if table.empty:
            continue

        (
            date_column,
            nav_column,
        ) = find_date_nav_columns(
            table
        )

        if (
            date_column is None
            or nav_column is None
        ):
            continue

        for _, row in table.iterrows():
            parsed_date = (
                parse_roc_or_ad_date(
                    row.get(date_column)
                )
            )

            nav_value = safe_float(
                row.get(nav_column)
            )

            if (
                pd.notna(parsed_date)
                and pd.notna(nav_value)
            ):
                records.append(
                    {
                        "date": parsed_date,
                        "nav": nav_value,
                    }
                )

    return normalize_history(
        pd.DataFrame(records)
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_fund_url(url):
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=45,
            )

            response.raise_for_status()

            response.encoding = (
                response.apparent_encoding
                or response.encoding
            )

            result = parse_fund_tables(
                response.text
            )

            if not result.empty:
                return result, None

            return (
                empty_history(),
                (
                    "頁面可開啟，"
                    "但沒有解析到日期與淨值"
                ),
            )

        except Exception as error:
            last_error = error

            if attempt < 2:
                time.sleep(
                    2 * (attempt + 1)
                )

    return (
        empty_history(),
        (
            f"{type(last_error).__name__}: "
            f"{last_error}"
        ),
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_online_fund_history(urls):
    frames = []
    errors = []
    successful_urls = []

    for url in urls:
        data, error = fetch_fund_url(
            url
        )

        if not data.empty:
            frames.append(data)
            successful_urls.append(url)

        elif error:
            errors.append(
                f"{url}：{error}"
            )

    if not frames:
        return (
            empty_history(),
            errors,
            successful_urls,
        )

    return (
        normalize_history(
            pd.concat(
                frames,
                ignore_index=True,
            )
        ),
        errors,
        successful_urls,
    )


def merge_fund_histories(
    local_history,
    online_history,
):
    frames = []

    if not local_history.empty:
        local_copy = (
            local_history.copy()
        )

        local_copy["priority"] = 1
        frames.append(local_copy)

    if not online_history.empty:
        online_copy = (
            online_history.copy()
        )

        online_copy["priority"] = 2
        frames.append(online_copy)

    if not frames:
        return empty_history()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values(
            [
                "date",
                "priority",
            ]
        )
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
    )

    return normalize_history(
        combined[
            [
                "date",
                "nav",
            ]
        ]
    )


# ============================================================
# 績效及風險計算
# ============================================================

def get_value_on_or_before(
    history,
    target_date,
):
    available = history[
        history["date"]
        <= pd.Timestamp(target_date)
    ]

    if available.empty:
        return None, np.nan

    row = available.iloc[-1]

    return row["date"], row["nav"]


def calculate_period_return(
    history,
    current_value,
    target_date,
    maximum_gap_days=10,
):
    base_date, base_value = (
        get_value_on_or_before(
            history,
            target_date,
        )
    )

    if base_date is None:
        return np.nan

    gap_days = (
        pd.Timestamp(target_date)
        - pd.Timestamp(base_date)
    ).days

    if (
        gap_days < 0
        or gap_days
        > maximum_gap_days
    ):
        return np.nan

    return calculate_return(
        current_value,
        base_value,
    )


def calculate_performance(
    history,
    selected_date,
):
    history = normalize_history(
        history
    )

    if history.empty:
        return None

    (
        current_date,
        current_value,
    ) = get_value_on_or_before(
        history,
        selected_date,
    )

    if current_date is None:
        return None

    previous_data = history[
        history["date"]
        < current_date
    ]

    previous_value = (
        previous_data.iloc[-1]["nav"]
        if not previous_data.empty
        else np.nan
    )

    month_target = (
        current_date.replace(day=1)
        - timedelta(days=1)
    )

    year_target = (
        pd.Timestamp(
            current_date.year,
            1,
            1,
        )
        - timedelta(days=1)
    )

    return {
        "date": current_date,
        "value": current_value,
        "Daily": calculate_return(
            current_value,
            previous_value,
        ),
        "MTD": calculate_period_return(
            history,
            current_value,
            month_target,
        ),
        "YTD": calculate_period_return(
            history,
            current_value,
            year_target,
        ),
        "1Y累積": calculate_period_return(
            history,
            current_value,
            current_date
            - pd.DateOffset(years=1),
        ),
        "2Y累積": calculate_period_return(
            history,
            current_value,
            current_date
            - pd.DateOffset(years=2),
        ),
        "3Y累積": calculate_period_return(
            history,
            current_value,
            current_date
            - pd.DateOffset(years=3),
        ),
    }


def filter_period(
    history,
    selected_date,
    period_name,
):
    end_date = pd.Timestamp(
        selected_date
    )

    offsets = {
        "近30日": pd.DateOffset(
            days=30
        ),
        "近3個月": pd.DateOffset(
            months=3
        ),
        "近6個月": pd.DateOffset(
            months=6
        ),
        "近1年": pd.DateOffset(
            years=1
        ),
        "近2年": pd.DateOffset(
            years=2
        ),
        "近3年": pd.DateOffset(
            years=3
        ),
    }

    if (
        period_name
        == "全部可用期間"
    ):
        return history[
            history["date"]
            <= end_date
        ].copy()

    start_date = (
        end_date
        - offsets[period_name]
    )

    return history[
        (
            history["date"]
            >= start_date
        )
        & (
            history["date"]
            <= end_date
        )
    ].copy()


def calculate_risk_metrics(
    history,
    risk_free_rate,
):
    history = normalize_history(
        history
    )

    daily_returns = (
        history["nav"]
        .pct_change()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if (
        len(daily_returns)
        < MIN_RISK_OBSERVATIONS
    ):
        return None

    observation_years = (
        len(daily_returns)
        / 252
    )

    total_return = (
        history.iloc[-1]["nav"]
        / history.iloc[0]["nav"]
        - 1
    )

    if (
        observation_years > 0
        and 1 + total_return > 0
    ):
        annual_return = (
            (1 + total_return)
            ** (
                1
                / observation_years
            )
            - 1
        )

    else:
        annual_return = np.nan

    annual_volatility = (
        daily_returns.std()
        * np.sqrt(252)
    )

    maximum_drawdown = (
        history["nav"]
        / history["nav"].cummax()
        - 1
    ).min()

    if annual_volatility > 0:
        sharpe_ratio = (
            annual_return
            - risk_free_rate
        ) / annual_volatility
    else:
        sharpe_ratio = np.nan

    return {
        "年化報酬率": (
            annual_return * 100
        ),
        "年化波動率": (
            annual_volatility * 100
        ),
        "最大回撤": (
            maximum_drawdown * 100
        ),
        "Sharpe Ratio": (
            sharpe_ratio
        ),
        "資料筆數": len(history),
    }


# ============================================================
# 側邊選單
# ============================================================

with st.sidebar:
    st.title("📊 分析選單")

    page = st.radio(
        "選擇頁面",
        [
            "📋 績效總覽",
            "📈 走勢分析",
            "⚖️ 風險分析",
            "🔧 資料狀態",
        ],
    )

    selected_date = st.date_input(
        "查詢日期",
        value=date.today(),
        max_value=date.today(),
    )

    chart_period = st.selectbox(
        "走勢分析期間",
        [
            "近30日",
            "近3個月",
            "近6個月",
            "近1年",
            "近2年",
            "近3年",
            "全部可用期間",
        ],
        index=3,
    )

    risk_period = st.selectbox(
        "風險分析期間",
        [
            "近1年",
            "近2年",
            "近3年",
            "全部可用期間",
        ],
    )

    risk_free_rate = st.number_input(
        "無風險利率（年化%）",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.1,
    )

    st.divider()

    st.info(
        "下次預計更新："
        f"{get_next_update_text()}"
        "（台灣時間）"
    )

    st.caption(
        "工作日19:00與21:30排程，"
        "實際啟動可能延遲數分鐘。"
    )


# ============================================================
# 資料載入
# ============================================================

local_fund_data, quality_report = (
    load_fund_csv()
)

result_rows = []
histories = {}
diagnostics = []
status_rows = []

progress_bar = st.progress(
    0,
    text="正在取得資料…",
)


for index, (
    product_code,
    product_info,
) in enumerate(
    PRODUCTS.items()
):
    if product_info["type"] == "ETF":
        history = fetch_etf_history(
            product_code,
            selected_date,
        )

        product_type = "ETF"
        data_source = "TWSE"

    else:
        local_history = (
            get_local_fund_history(
                product_code,
                product_info["name"],
                local_fund_data,
            )
        )

        (
            online_history,
            source_errors,
            successful_urls,
        ) = (
            fetch_online_fund_history(
                tuple(
                    product_info["urls"]
                )
            )
        )

        history = merge_fund_histories(
            local_history,
            online_history,
        )

        product_type = "共同基金"

        if (
            not local_history.empty
            and not online_history.empty
        ):
            data_source = (
                "線上資料 + CSV"
            )

        elif not online_history.empty:
            data_source = (
                "線上基金資料"
            )

        elif not local_history.empty:
            data_source = "CSV"

        else:
            data_source = (
                "無可用資料"
            )

        diagnostics.append(
            {
                "name": (
                    product_info["name"]
                ),
                "online_rows": len(
                    online_history
                ),
                "csv_rows": len(
                    local_history
                ),
                "first_date": (
                    online_history[
                        "date"
                    ].min()
                    if not
                    online_history.empty
                    else None
                ),
                "last_date": (
                    online_history[
                        "date"
                    ].max()
                    if not
                    online_history.empty
                    else None
                ),
                "successful_urls": (
                    successful_urls
                ),
                "errors": (
                    source_errors
                ),
            }
        )

    history = normalize_history(
        history
    )

    history = history[
        history["date"]
        <= pd.Timestamp(
            selected_date
        )
    ].copy()

    histories[
        product_info["name"]
    ] = history

    product_performance = (
        calculate_performance(
            history,
            selected_date,
        )
    )

    result_row = {
        "商品名稱": (
            product_info["name"]
        ),
        "類型": product_type,
        "資料日期": "-",
        "價格／淨值": np.nan,
        "Daily": np.nan,
        "MTD": np.nan,
        "YTD": np.nan,
        "1Y累積": np.nan,
        "2Y累積": np.nan,
        "3Y累積": np.nan,
        "資料來源": data_source,
    }

    if product_performance:
        result_row.update(
            {
                "資料日期": (
                    product_performance[
                        "date"
                    ].strftime(
                        "%Y/%m/%d"
                    )
                ),
                "價格／淨值": (
                    product_performance[
                        "value"
                    ]
                ),
                **{
                    period: (
                        product_performance[
                            period
                        ]
                    )
                    for period
                    in PERIOD_COLUMNS
                },
            }
        )

    result_rows.append(
        result_row
    )

    if history.empty:
        latest_product_date = None
        data_status = "無資料"

    else:
        latest_product_date = (
            history["date"].max()
        )

        age_days = (
            pd.Timestamp(
                selected_date
            )
            - latest_product_date
        ).days

        data_status = (
            "正常"
            if age_days <= 3
            else "資料較舊"
        )

    status_rows.append(
        {
            "商品名稱": (
                product_info["name"]
            ),
            "類型": product_type,
            "最新資料日期": (
                latest_product_date
                .strftime("%Y/%m/%d")
                if latest_product_date
                is not None
                else "-"
            ),
            "資料筆數": len(history),
            "資料來源": data_source,
            "資料狀態": data_status,
        }
    )

    progress_bar.progress(
        (
            index + 1
        )
        / len(PRODUCTS),
        text=(
            "正在處理 "
            f"{product_info['name']}…"
        ),
    )


progress_bar.empty()

result_df = pd.DataFrame(
    result_rows
)

status_df = pd.DataFrame(
    status_rows
)


# ============================================================
# 置中標題與摘要
# ============================================================

st.markdown(
    """
    <h1 style="
        text-align:center;
        margin-top:0.2rem;
        margin-bottom:0.3rem;
    ">
        📊 ETF及共同基金績效比較
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        margin-bottom:1rem;
    ">
        ETF採分割調整後市場收盤價，
        共同基金採正式NAV；
        各期間均為未含配息的累積報酬率。
    </div>
    """,
    unsafe_allow_html=True,
)


valid_dates = pd.to_datetime(
    result_df.loc[
        result_df["資料日期"]
        != "-",
        "資料日期",
    ],
    errors="coerce",
).dropna()

latest_date_text = (
    valid_dates.max()
    .strftime("%Y/%m/%d")
    if not valid_dates.empty
    else "-"
)


metric_col_1, metric_col_2, \
metric_col_3, metric_col_4 = (
    st.columns(4)
)

metric_col_1.metric(
    "查詢日期",
    pd.Timestamp(
        selected_date
    ).strftime("%Y/%m/%d"),
)

metric_col_2.metric(
    "全體最新資料日",
    latest_date_text,
)

metric_col_3.metric(
    "比較商品數",
    len(result_df),
)

metric_col_4.metric(
    "下次預計更新",
    get_next_update_text(),
)


# ============================================================
# 頁面1：績效總覽
# ============================================================

if page == "📋 績效總覽":
    st.subheader(
        "📋 績效總覽"
    )

    display_df = result_df.copy()
    download_df = result_df.copy()

    width_data = display_df.copy()

    width_data[
        "價格／淨值"
    ] = (
        width_data[
            "價格／淨值"
        ].map(format_number)
    )

    width_data[
        "Daily"
    ] = (
        width_data["Daily"]
        .map(format_daily)
    )

    for period in [
        "MTD",
        "YTD",
        "1Y累積",
        "2Y累積",
        "3Y累積",
    ]:
        width_data[period] = (
            width_data[period]
            .map(format_percentage)
        )

    display_columns = [
        "商品名稱",
        "類型",
        "資料日期",
        "價格／淨值",
        "Daily",
        "MTD",
        "YTD",
        "1Y累積",
        "2Y累積",
        "3Y累積",
        "資料來源",
    ]

    column_widths = {
        "商品名稱": automatic_column_width(
            width_data,
            "商品名稱",
            extra_characters=5,
            minimum_pixels=230,
            maximum_pixels=420,
        ),
        "類型": automatic_column_width(
            width_data,
            "類型",
            extra_characters=5,
            minimum_pixels=85,
            maximum_pixels=130,
        ),
        "資料日期": automatic_column_width(
            width_data,
            "資料日期",
            extra_characters=5,
            minimum_pixels=115,
            maximum_pixels=150,
        ),
        "價格／淨值": automatic_column_width(
            width_data,
            "價格／淨值",
            extra_characters=5,
            minimum_pixels=115,
            maximum_pixels=160,
        ),
        "Daily": automatic_column_width(
            width_data,
            "Daily",
            extra_characters=5,
            minimum_pixels=110,
            maximum_pixels=145,
        ),
        "MTD": automatic_column_width(
            width_data,
            "MTD",
            extra_characters=5,
            minimum_pixels=95,
            maximum_pixels=130,
        ),
        "YTD": automatic_column_width(
            width_data,
            "YTD",
            extra_characters=5,
            minimum_pixels=95,
            maximum_pixels=130,
        ),
        "1Y累積": automatic_column_width(
            width_data,
            "1Y累積",
            extra_characters=5,
            minimum_pixels=100,
            maximum_pixels=140,
        ),
        "2Y累積": automatic_column_width(
            width_data,
            "2Y累積",
            extra_characters=5,
            minimum_pixels=100,
            maximum_pixels=140,
        ),
        "3Y累積": automatic_column_width(
            width_data,
            "3Y累積",
            extra_characters=5,
            minimum_pixels=100,
            maximum_pixels=140,
        ),
        "資料來源": automatic_column_width(
            width_data,
            "資料來源",
            extra_characters=5,
            minimum_pixels=130,
            maximum_pixels=220,
        ),
    }

    colgroup_html = "".join(
        (
            '<col style="width:'
            f'{column_widths[column]}'
            'px">'
        )
        for column
        in display_columns
    )

    header_html = "".join(
        (
            '<th style="'
            "text-align:center;"
            "white-space:nowrap;"
            "padding:10px 12px;"
            "background:#f3f4f6;"
            "color:#4b5563;"
            "border-bottom:"
            "1px solid #d1d5db;"
            "font-weight:600;"
            '">'
            f"{escape_html(column)}"
            "</th>"
        )
        for column
        in display_columns
    )

    body_rows = []

    for _, row in display_df.iterrows():
        daily_value = row["Daily"]

        (
            price_color,
            price_bold,
        ) = return_style(
            daily_value
        )

        row_cells = [
            make_table_cell(
                row["商品名稱"],
                alignment="left",
            ),
            make_table_cell(
                row["類型"],
                alignment="center",
            ),
            make_table_cell(
                row["資料日期"],
                alignment="center",
            ),
            make_table_cell(
                format_number(
                    row["價格／淨值"]
                ),
                alignment="right",
                color=price_color,
                bold=price_bold,
            ),
        ]

        (
            daily_color,
            daily_bold,
        ) = return_style(
            daily_value
        )

        row_cells.append(
            make_table_cell(
                format_daily(
                    daily_value
                ),
                alignment="right",
                color=daily_color,
                bold=daily_bold,
            )
        )

        for period in [
            "MTD",
            "YTD",
            "1Y累積",
            "2Y累積",
            "3Y累積",
        ]:
            (
                period_color,
                period_bold,
            ) = return_style(
                row[period]
            )

            row_cells.append(
                make_table_cell(
                    format_percentage(
                        row[period]
                    ),
                    alignment="right",
                    color=period_color,
                    bold=period_bold,
                )
            )

        row_cells.append(
            make_table_cell(
                row["資料來源"],
                alignment="left",
            )
        )

        body_rows.append(
            "<tr>"
            + "".join(row_cells)
            + "</tr>"
        )

    table_html = (
        '<div style="'
        "width:100%;"
        "overflow-x:auto;"
        "border:1px solid #e5e7eb;"
        "border-radius:10px;"
        "margin-bottom:12px;"
        '">'
        '<table style="'
        "width:100%;"
        "min-width:1280px;"
        "border-collapse:collapse;"
        "font-family:Arial,"
        "Microsoft JhengHei,"
        "sans-serif;"
        "font-size:14px;"
        "table-layout:fixed;"
        '">'
        f"<colgroup>{colgroup_html}</colgroup>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )

    st.markdown(
        table_html,
        unsafe_allow_html=True,
    )

    st.caption(
        "註：價格／淨值及Daily以紅色表示上漲、"
        "綠色表示下跌；Daily的▲代表上漲，"
        "▼代表下跌。各期間均為未含配息的"
        "累積報酬率。"
    )

    download_bytes = (
        download_df.to_csv(
            index=False,
        )
        .encode("utf-8-sig")
    )

    st.download_button(
        "⬇️ 下載績效表 CSV",
        data=download_bytes,
        file_name=(
            "performance_"
            f"{pd.Timestamp(selected_date):%Y%m%d}"
            ".csv"
        ),
        mime="text/csv",
    )


# ============================================================
# 頁面2：走勢分析
# ============================================================

elif page == "📈 走勢分析":
    st.subheader(
        "📈 走勢分析"
    )

    chart_frames = []
    normalized_frames = []
    interval_rows = []

    for (
        product_name,
        history,
    ) in histories.items():
        period_data = filter_period(
            history,
            selected_date,
            chart_period,
        )

        if period_data.empty:
            continue

        chart_data = (
            period_data.copy()
        )

        chart_data[
            "商品名稱"
        ] = product_name

        chart_frames.append(
            chart_data
        )

        first_value = (
            period_data.iloc[0]["nav"]
        )

        last_value = (
            period_data.iloc[-1]["nav"]
        )

        if first_value > 0:
            normalized_data = (
                period_data.copy()
            )

            normalized_data[
                "標準化指數"
            ] = (
                normalized_data["nav"]
                / first_value
                * 100
            )

            normalized_data[
                "商品名稱"
            ] = product_name

            normalized_frames.append(
                normalized_data
            )

            interval_rows.append(
                {
                    "商品名稱": (
                        product_name
                    ),
                    "區間累積報酬率": (
                        calculate_return(
                            last_value,
                            first_value,
                        )
                    ),
                }
            )

    st.markdown(
        "#### 價格／淨值走勢"
    )

    if chart_frames:
        combined_chart_data = pd.concat(
            chart_frames,
            ignore_index=True,
        )

        price_figure = px.line(
            combined_chart_data,
            x="date",
            y="nav",
            color="商品名稱",
            labels={
                "date": "日期",
                "nav": "價格／淨值",
                "商品名稱": "商品",
            },
        )

        price_figure.update_layout(
            hovermode="x unified",
            height=520,
        )

        st.plotly_chart(
            price_figure,
            use_container_width=True,
        )

        st.caption(
            "原始價格及淨值尺度不同，"
            "此圖適合查看各商品本身的走勢。"
        )

    else:
        st.info(
            "目前期間沒有可繪製資料。"
        )

    st.markdown(
        "#### 區間累積報酬比較"
    )

    if interval_rows:
        interval_df = (
            pd.DataFrame(
                interval_rows
            )
            .sort_values(
                "區間累積報酬率",
                ascending=False,
            )
        )

        interval_figure = px.bar(
            interval_df,
            x="商品名稱",
            y="區間累積報酬率",
            color="區間累積報酬率",
            color_continuous_scale=(
                "RdYlGn"
            ),
            text_auto=".2f",
            labels={
                "區間累積報酬率": (
                    "累積報酬率 (%)"
                ),
            },
        )

        interval_figure.update_layout(
            height=480,
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            interval_figure,
            use_container_width=True,
        )

    else:
        st.info(
            "目前期間沒有足夠資料"
            "計算區間報酬。"
        )

    st.markdown(
        "#### 標準化績效走勢"
        "（起點 = 100）"
    )

    if normalized_frames:
        normalized_df = pd.concat(
            normalized_frames,
            ignore_index=True,
        )

        normalized_figure = px.line(
            normalized_df,
            x="date",
            y="標準化指數",
            color="商品名稱",
            labels={
                "date": "日期",
                "標準化指數": (
                    "起點 = 100"
                ),
                "商品名稱": "商品",
            },
        )

        normalized_figure.add_hline(
            y=100,
            line_dash="dash",
            line_color="gray",
        )

        normalized_figure.update_layout(
            hovermode="x unified",
            height=520,
        )

        st.plotly_chart(
            normalized_figure,
            use_container_width=True,
        )

    else:
        st.info(
            "目前期間沒有足夠資料"
            "建立標準化走勢。"
        )


# ============================================================
# 頁面3：風險分析
# ============================================================

elif page == "⚖️ 風險分析":
    st.subheader(
        "⚖️ 風險與風險調整績效"
    )

    st.caption(
        f"分析期間：{risk_period}；"
        f"無風險利率：{risk_free_rate:.1f}%；"
        f"至少需要{MIN_RISK_OBSERVATIONS}筆資料。"
    )

    risk_rows = []
    correlation_series = {}

    for (
        product_name,
        history,
    ) in histories.items():
        period_data = filter_period(
            history,
            selected_date,
            risk_period,
        )

        metrics = calculate_risk_metrics(
            period_data,
            risk_free_rate / 100,
        )

        if metrics:
            risk_rows.append(
                {
                    "商品名稱": (
                        product_name
                    ),
                    **metrics,
                }
            )

            correlation_series[
                product_name
            ] = (
                period_data
                .set_index("date")[
                    "nav"
                ]
                .pct_change()
                .rename(product_name)
            )

    if not risk_rows:
        st.info(
            "目前沒有商品具備足夠資料"
            "計算風險指標。"
        )

    else:
        risk_df = pd.DataFrame(
            risk_rows
        )

        st.markdown(
            "#### 風險報酬散點圖"
        )

        risk_return_figure = px.scatter(
            risk_df,
            x="年化波動率",
            y="年化報酬率",
            text="商品名稱",
            size="資料筆數",
            color="Sharpe Ratio",
            color_continuous_scale="RdYlGn",
            labels={
                "年化波動率": (
                    "年化波動率 (%)"
                ),
                "年化報酬率": (
                    "年化報酬率 (%)"
                ),
            },
        )

        risk_return_figure.update_traces(
            textposition="top center"
        )

        risk_return_figure.update_layout(
            height=550
        )

        st.plotly_chart(
            risk_return_figure,
            use_container_width=True,
        )

        risk_col_1, risk_col_2 = (
            st.columns(2)
        )

        with risk_col_1:
            st.markdown(
                "#### 最大回撤"
            )

            drawdown_figure = px.bar(
                risk_df.sort_values(
                    "最大回撤"
                ),
                x="最大回撤",
                y="商品名稱",
                orientation="h",
                text_auto=".2f",
                color_discrete_sequence=[
                    "#d95f59"
                ],
                labels={
                    "最大回撤": (
                        "最大回撤 (%)"
                    ),
                },
            )

            drawdown_figure.update_layout(
                height=430
            )

            st.plotly_chart(
                drawdown_figure,
                use_container_width=True,
            )

        with risk_col_2:
            st.markdown(
                "#### 年化波動率"
            )

            volatility_figure = px.bar(
                risk_df.sort_values(
                    "年化波動率"
                ),
                x="年化波動率",
                y="商品名稱",
                orientation="h",
                text_auto=".2f",
                color_discrete_sequence=[
                    "#4c78a8"
                ],
                labels={
                    "年化波動率": (
                        "年化波動率 (%)"
                    ),
                },
            )

            volatility_figure.update_layout(
                height=430
            )

            st.plotly_chart(
                volatility_figure,
                use_container_width=True,
            )

        st.markdown(
            "#### Sharpe Ratio"
        )

        sharpe_figure = px.bar(
            risk_df.sort_values(
                "Sharpe Ratio",
                ascending=False,
            ),
            x="商品名稱",
            y="Sharpe Ratio",
            color="Sharpe Ratio",
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
        )

        sharpe_figure.add_hline(
            y=0,
            line_color="gray",
        )

        sharpe_figure.update_layout(
            height=470,
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            sharpe_figure,
            use_container_width=True,
        )

        st.markdown(
            "#### 每日報酬相關係數矩陣"
        )

        if (
            len(correlation_series)
            >= 2
        ):
            correlation_df = pd.concat(
                correlation_series.values(),
                axis=1,
            ).corr(
                min_periods=(
                    MIN_RISK_OBSERVATIONS
                )
            )

            if (
                correlation_df
                .notna()
                .any()
                .any()
            ):
                correlation_figure = (
                    px.imshow(
                        correlation_df,
                        text_auto=".2f",
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale=(
                            "RdBu_r"
                        ),
                        labels={
                            "color": (
                                "相關係數"
                            ),
                        },
                    )
                )

                correlation_figure.update_layout(
                    height=600
                )

                st.plotly_chart(
                    correlation_figure,
                    use_container_width=True,
                )

            else:
                st.info(
                    "共同有效日期不足，"
                    "無法計算相關係數。"
                )

        else:
            st.info(
                "至少需要兩項商品具備"
                "足夠資料才能計算相關係數。"
            )

        with st.expander(
            "查看風險指標數據"
        ):
            display_risk_df = (
                risk_df.copy()
            )

            for column in [
                "年化報酬率",
                "年化波動率",
                "最大回撤",
            ]:
                display_risk_df[
                    column
                ] = (
                    display_risk_df[
                        column
                    ].map(
                        lambda value: (
                            f"{value:.2f}%"
                        )
                    )
                )

            display_risk_df[
                "Sharpe Ratio"
            ] = (
                display_risk_df[
                    "Sharpe Ratio"
                ].map(
                    lambda value: (
                        f"{value:.2f}"
                    )
                )
            )

            st.dataframe(
                display_risk_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# 頁面4：資料狀態
# ============================================================

else:
    st.subheader(
        "🔧 資料狀態"
    )

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "#### 🧪 資料品質檢核"
    )

    quality_rows = [
        {
            "檢核項目": "CSV檔案存在",
            "結果": (
                "正常"
                if quality_report["exists"]
                else "異常"
            ),
            "說明": str(
                LOCAL_FUND_FILE
            ),
        },
        {
            "檢核項目": "必要欄位",
            "結果": (
                "正常"
                if quality_report[
                    "columns_ok"
                ]
                else "異常"
            ),
            "說明": (
                "date、code、name、nav"
            ),
        },
        {
            "檢核項目": "無法解析日期",
            "結果": (
                "正常"
                if quality_report[
                    "invalid_dates"
                ] == 0
                else "提醒"
            ),
            "說明": (
                f"{quality_report['invalid_dates']}"
                "筆"
            ),
        },
        {
            "檢核項目": "無效淨值",
            "結果": (
                "正常"
                if quality_report[
                    "invalid_nav"
                ] == 0
                else "提醒"
            ),
            "說明": (
                f"{quality_report['invalid_nav']}"
                "筆"
            ),
        },
        {
            "檢核項目": "重複資料",
            "結果": (
                "正常"
                if quality_report[
                    "duplicate_rows"
                ] == 0
                else "提醒"
            ),
            "說明": (
                f"{quality_report['duplicate_rows']}"
                "筆"
            ),
        },
        {
            "檢核項目": "未知基金代碼",
            "結果": (
                "正常"
                if not quality_report[
                    "unknown_codes"
                ]
                else "異常"
            ),
            "說明": (
                "、".join(
                    quality_report[
                        "unknown_codes"
                    ]
                )
                or "無"
            ),
        },
        {
            "檢核項目": "基金代碼缺漏",
            "結果": (
                "正常"
                if not quality_report[
                    "missing_codes"
                ]
                else "異常"
            ),
            "說明": (
                "、".join(
                    quality_report[
                        "missing_codes"
                    ]
                )
                or "無"
            ),
        },
    ]

    quality_df = pd.DataFrame(
        quality_rows
    )

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True,
    )

    if any(
        row["結果"] == "異常"
        for row in quality_rows
    ):
        st.error(
            "發現資料品質異常，"
            "請查看檢核說明。"
        )

    elif any(
        row["結果"] == "提醒"
        for row in quality_rows
    ):
        st.warning(
            "資料可使用，但有需要留意的"
            "品質提醒。"
        )

    else:
        st.success(
            "資料品質檢核通過。"
        )

    st.markdown(
        "#### 資料時間覆蓋與日期缺口"
    )

    coverage_rows = []

    for (
        product_name,
        history,
    ) in histories.items():
        if history.empty:
            coverage_rows.append(
                {
                    "商品名稱": product_name,
                    "最早日期": "-",
                    "最新日期": "-",
                    "最大日期缺口": "-",
                    "最大缺口區間": "-",
                    "缺口提醒": "無資料",
                }
            )

            continue

        sorted_dates = (
            history["date"]
            .sort_values()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        date_gaps = (
            sorted_dates
            .diff()
            .dt.days
        )

        if date_gaps.dropna().empty:
            maximum_gap = 0
            gap_period = "-"

        else:
            maximum_gap_index = (
                date_gaps.idxmax()
            )

            maximum_gap = int(
                date_gaps.loc[
                    maximum_gap_index
                ]
            )

            gap_start = (
                sorted_dates.loc[
                    maximum_gap_index - 1
                ]
            )

            gap_end = (
                sorted_dates.loc[
                    maximum_gap_index
                ]
            )

            gap_period = (
                f"{gap_start:%Y/%m/%d}"
                " 至 "
                f"{gap_end:%Y/%m/%d}"
            )

        if maximum_gap > 30:
            gap_status = "異常"

        elif maximum_gap > 14:
            gap_status = "提醒"

        else:
            gap_status = "正常"

        coverage_rows.append(
            {
                "商品名稱": product_name,
                "最早日期": (
                    sorted_dates.min()
                    .strftime("%Y/%m/%d")
                ),
                "最新日期": (
                    sorted_dates.max()
                    .strftime("%Y/%m/%d")
                ),
                "最大日期缺口": (
                    f"{maximum_gap}天"
                ),
                "最大缺口區間": (
                    gap_period
                ),
                "缺口提醒": gap_status,
            }
        )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    st.dataframe(
        coverage_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "缺口判定：14天以內正常、"
        "15至30天提醒、超過30天異常；"
        "用以降低春節及連續假期"
        "造成的誤報。"
    )

    with st.expander(
        "共同基金線上來源診斷"
    ):
        for item in diagnostics:
            st.markdown(
                f"**{item['name']}**"
            )

            st.write(
                "線上取得"
                f"{item['online_rows']}筆；"
                "CSV取得"
                f"{item['csv_rows']}筆。"
            )

            if (
                item["first_date"]
                is not None
            ):
                st.write(
                    "線上期間："
                    f"{item['first_date']:%Y/%m/%d}"
                    " 至 "
                    f"{item['last_date']:%Y/%m/%d}"
                )

            if item["successful_urls"]:
                st.write(
                    "成功來源："
                )

                for url in item[
                    "successful_urls"
                ]:
                    st.code(
                        url,
                        language="text",
                    )

            if item["errors"]:
                st.write(
                    "來源錯誤或提醒："
                )

                for error in item[
                    "errors"
                ]:
                    st.code(
                        error,
                        language="text",
                    )

    with st.expander(
        "計算口徑與限制"
    ):
        st.markdown(
            """
- ETF使用臺灣證券交易所市場收盤價，0050已調整2025年6月18日的1拆4。
- 共同基金使用正式NAV，歷史資料由 `fund_nav_history.csv` 提供。
- 線上近期資料與CSV同日期並存時，以線上資料優先。
- Daily、MTD、YTD、1Y、2Y、3Y均為未含配息的累積報酬率。
- 基準資料距目標日期超過10天時，績效顯示「-」。
- 風險指標至少需要60筆每日資料。
"""
        )


# ============================================================
# 頁尾
# ============================================================

st.divider()

st.caption(
    "資料僅供研究與績效比較使用。"
    "ETF正式基金淨值可能與市場收盤價不同；"
    "共同基金正式資料仍以基金公司公告為準。"
)
