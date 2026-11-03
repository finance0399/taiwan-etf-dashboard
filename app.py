from __future__ import annotations

import re
import time
from datetime import date, datetime, time as dt_time, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# 頁面設定
# ============================================================

st.set_page_config(
    page_title="ETF及共同基金績效比較",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# 常數與商品設定
# ============================================================

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

LOCAL_FUND_FILE = Path(
    "fund_nav_history.csv"
)

PERIOD_COLUMNS = [
    "Daily",
    "MTD",
    "YTD",
    "1Y累積",
    "2Y累積",
    "3Y累積",
]

MIN_RISK_OBSERVATIONS = 60
DEFAULT_RISK_FREE_RATE = 1.5

FUND_CODES = {
    "A09012",
    "A41209",
}

EXPECTED_CSV_COLUMNS = {
    "date",
    "code",
    "name",
    "nav",
}


PRODUCTS = {
    "00981A": {
        "name": (
            "00981A "
            "統一台股增長主動式ETF"
        ),
        "type": "ETF",
    },
    "A09012": {
        "name": "統一奔騰基金",
        "type": "FUND",
        "urls": [
            (
                "https://www.ezmoney.com.tw/"
                "funds/value/36"
            ),
            (
                "https://fund.hncb.com.tw/"
                "w/wr/"
                "wr02_ACPS10-009007.djhtm"
            ),
            (
                "https://www.moneydj.com/"
                "funddj/ya/"
                "yp010000.djhtm?a=ACPS10"
            ),
        ],
    },
    "00407A": {
        "name": (
            "00407A "
            "凱基台灣主動式ETF"
        ),
        "type": "ETF",
    },
    "A41209": {
        "name": "凱基台灣精五門基金",
        "type": "FUND",
        "urls": [
            (
                "https://fund.hncb.com.tw/"
                "w/wr/"
                "wr02_ACKG15-014013.djhtm"
            ),
            (
                "https://chbfund.moneydj.com/"
                "w/wr/"
                "wr02.djhtm?A=ACKG15-9808"
            ),
            (
                "https://tw.stock.yahoo.com/"
                "fund/"
                "F0000040SC:FO/history"
            ),
        ],
    },
    "009816": {
        "name": (
            "009816 "
            "凱基台灣TOP 50 ETF"
        ),
        "type": "ETF",
    },
    "0050": {
        "name": "0050 元大台灣50",
        "type": "ETF",
    },
}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/130.0.0.0 "
        "Safari/537.36"
    ),
    "Accept-Language": (
        "zh-TW,zh;q=0.9,en;q=0.8"
    ),
}


ETF_SPLIT_CONFIG = {
    "0050": [
        {
            "effective_date": pd.Timestamp(
                "2025-06-18"
            ),
            "ratio": 4.0,
        }
    ]
}


# ============================================================
# 共用工具
# ============================================================

def empty_history():
    return pd.DataFrame(
        columns=[
            "date",
            "nav",
        ]
    )


def safe_float(value):
    """安全轉換數值。"""

    try:
        text = str(value).strip()

        text = (
            text.replace(",", "")
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
    """統一歷史資料格式。"""

    if (
        data is None
        or data.empty
        or not {
            "date",
            "nav",
        }.issubset(data.columns)
    ):
        return empty_history()

    result = data[
        [
            "date",
            "nav",
        ]
    ].copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    try:
        result["date"] = (
            result["date"]
            .dt.tz_localize(None)
        )

    except (
        TypeError,
        AttributeError,
    ):
        pass

    result["nav"] = (
        result["nav"]
        .apply(safe_float)
    )

    result = result.dropna(
        subset=[
            "date",
            "nav",
        ]
    )

    result["date"] = (
        result["date"]
        .dt.normalize()
    )

    result = result[
        result["nav"] > 0
    ]

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
    """解析民國或西元日期。"""

    text = str(value).strip()

    match = re.search(
        r"(\d{2,4})[/-]"
        r"(\d{1,2})[/-]"
        r"(\d{1,2})",
        text,
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


def adjust_etf_close_price(
    stock_code,
    trade_date,
    close_price,
):
    """
    調整ETF分割造成的價格不連續。

    0050於2025-06-18進行1拆4，
    因此該日期以前的收盤價除以4。
    """

    if (
        pd.isna(trade_date)
        or pd.isna(close_price)
    ):
        return close_price

    adjusted_price = float(
        close_price
    )

    events = ETF_SPLIT_CONFIG.get(
        stock_code,
        [],
    )

    for event in events:

        if (
            trade_date
            < event["effective_date"]
        ):
            adjusted_price /= event[
                "ratio"
            ]

    return adjusted_price


def pct_change(
    current_value,
    base_value,
):
    """計算累積報酬率。"""

    if (
        pd.isna(current_value)
        or pd.isna(base_value)
        or base_value == 0
    ):
        return np.nan

    return (
        current_value
        / base_value
        - 1
    ) * 100


def format_number(value):

    if pd.isna(value):
        return "-"

    return (
        f"{value:,.4f}"
        .rstrip("0")
        .rstrip(".")
    )


def format_pct(value):

    if pd.isna(value):
        return "-"

    return f"{value:+.2f}%"


def performance_color(value):
    """台股習慣：紅漲、綠跌。"""

    if (
        not isinstance(value, str)
        or value == "-"
    ):
        return ""

    try:
        number = float(
            value.replace("%", "")
            .replace("+", "")
        )

        if number > 0:
            return (
                "color:#d62728;"
                "font-weight:bold"
            )

        if number < 0:
            return (
                "color:#138a36;"
                "font-weight:bold"
            )

    except Exception:
        pass

    return ""


# ============================================================
# 下次排程時間
# ============================================================

def next_scheduled_update(
    now=None,
):
    """
    台灣時間工作日：
    19:00及21:30。
    """

    now = (
        now
        or datetime.now(TAIPEI_TZ)
    )

    scheduled_times = [
        dt_time(19, 0),
        dt_time(21, 30),
    ]

    if now.weekday() < 5:

        for scheduled_time in (
            scheduled_times
        ):
            candidate = (
                datetime.combine(
                    now.date(),
                    scheduled_time,
                    tzinfo=TAIPEI_TZ,
                )
            )

            if now < candidate:
                return candidate

    next_day = (
        now.date()
        + timedelta(days=1)
    )

    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return datetime.combine(
        next_day,
        scheduled_times[0],
        tzinfo=TAIPEI_TZ,
    )


def scheduled_update_text():

    now = datetime.now(
        TAIPEI_TZ
    )

    next_update = (
        next_scheduled_update(now)
    )

    if (
        next_update.date()
        == now.date()
    ):
        prefix = "今日"

    elif (
        next_update.date()
        == now.date()
        + timedelta(days=1)
    ):
        prefix = "明日"

    else:
        weekdays = "一二三四五六日"

        prefix = (
            f"{next_update:%m/%d}"
            f"（週"
            f"{weekdays[next_update.weekday()]}"
            f"）"
        )

    return (
        f"{prefix} "
        f"{next_update:%H:%M}"
    )


# ============================================================
# ETF：臺灣證券交易所資料
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_twse_month(
    stock_code,
    year,
    month,
):
    """取得ETF單月市場收盤價。"""

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

    response = None

    for attempt in range(3):

        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=45,
            )

            response.raise_for_status()
            break

        except requests.RequestException:

            response = None

            if attempt < 2:
                time.sleep(
                    2 * (attempt + 1)
                )

    if response is None:
        return empty_history()

    try:
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

        adjusted_price = (
            adjust_etf_close_price(
                stock_code,
                trade_date,
                close_price,
            )
        )

        if (
            pd.notna(trade_date)
            and pd.notna(
                adjusted_price
            )
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


def month_iterator(
    start_date,
    end_date,
):
    """依月份產生查詢年月。"""

    current = pd.Timestamp(
        start_date.year,
        start_date.month,
        1,
    )

    ending = pd.Timestamp(
        end_date.year,
        end_date.month,
        1,
    )

    while current <= ending:

        yield (
            current.year,
            current.month,
        )

        current += (
            pd.offsets.MonthBegin(1)
        )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_etf_history(
    stock_code,
    selected_date,
):
    """取得ETF約三年又兩個月資料。"""

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

    frames = []

    for year, month in month_iterator(
        start_date,
        end_date,
    ):

        monthly_data = (
            fetch_twse_month(
                stock_code,
                year,
                month,
            )
        )

        if not monthly_data.empty:
            frames.append(
                monthly_data
            )

    if not frames:
        return empty_history()

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    history = normalize_history(
        history
    )

    history = history[
        history["date"]
        <= end_date
    ]

    return history.reset_index(
        drop=True
    )


# ============================================================
# 共同基金：CSV讀取及品質檢核
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def inspect_and_load_fund_csv():
    """
    讀取CSV，同時記錄原始資料品質問題。
    """

    report = {
        "exists": (
            LOCAL_FUND_FILE.exists()
        ),
        "columns_ok": False,
        "raw_rows": 0,
        "invalid_dates": 0,
        "invalid_nav": 0,
        "duplicate_rows": 0,
        "unknown_codes": [],
        "missing_codes": [],
        "error": None,
    }

    empty_result = pd.DataFrame(
        columns=[
            "date",
            "code",
            "name",
            "nav",
        ]
    )

    if not report["exists"]:

        report["error"] = (
            "找不到 "
            "fund_nav_history.csv"
        )

        return (
            empty_result,
            report,
        )

    data = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp950",
        "big5",
    ]:

        try:
            data = pd.read_csv(
                LOCAL_FUND_FILE,
                encoding=encoding,
                dtype={
                    "code": str,
                },
            )

            break

        except Exception:
            continue

    if data is None:

        report["error"] = (
            "CSV無法讀取或"
            "編碼不支援"
        )

        return (
            empty_result,
            report,
        )

    report["raw_rows"] = len(data)

    data.columns = [
        str(column).strip()
        for column in data.columns
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
        "基金淨值": "nav",
        "NAV": "nav",
        "Nav": "nav",
    }

    data = data.rename(
        columns=rename_map
    )

    report["columns_ok"] = (
        EXPECTED_CSV_COLUMNS
        .issubset(data.columns)
    )

    if not report["columns_ok"]:

        report["error"] = (
            "缺少必要欄位："
            "date、code、name、nav"
        )

        return (
            empty_result,
            report,
        )

    parsed_dates = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    parsed_nav = pd.to_numeric(
        data["nav"],
        errors="coerce",
    )

    report["invalid_dates"] = int(
        parsed_dates.isna().sum()
    )

    report["invalid_nav"] = int(
        (
            parsed_nav.isna()
            | (parsed_nav <= 0)
        ).sum()
    )

    data["date"] = parsed_dates

    data["code"] = (
        data["code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data["name"] = (
        data["name"]
        .astype(str)
        .str.strip()
    )

    data["nav"] = parsed_nav

    report["duplicate_rows"] = int(
        data.duplicated(
            subset=[
                "code",
                "date",
            ]
        ).sum()
    )

    valid_codes = set(
        data["code"]
        .dropna()
        .unique()
    )

    report["unknown_codes"] = sorted(
        valid_codes
        - FUND_CODES
    )

    report["missing_codes"] = sorted(
        FUND_CODES
        - valid_codes
    )

    data = data.dropna(
        subset=[
            "date",
            "code",
            "nav",
        ]
    )

    data = data[
        data["nav"] > 0
    ]

    data = (
        data.sort_values(
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
        data[
            [
                "date",
                "code",
                "name",
                "nav",
            ]
        ],
        report,
    )


def get_local_history(
    fund_code,
    fund_name,
    local_data,
):
    """取得指定共同基金的CSV歷史資料。"""

    if local_data.empty:
        return empty_history()

    result = local_data[
        local_data["code"]
        == fund_code
    ].copy()

    if result.empty:

        result = local_data[
            local_data[
                "name"
            ].str.contains(
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
# 共同基金：線上近期NAV
# ============================================================

def flatten_column_name(column):

    if isinstance(column, tuple):

        return " ".join(
            map(str, column)
        ).strip()

    return str(column).strip()


def choose_date_nav_columns(
    table,
):
    """尋找日期欄與淨值欄。"""

    date_column = None
    nav_column = None

    for column in table.columns:

        column_text = (
            flatten_column_name(
                column
            )
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

    return (
        date_column,
        nav_column,
    )


def parse_fund_tables(
    html_text,
):
    """解析基金頁面的日期及NAV。"""

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
        ) = choose_date_nav_columns(
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
                    row.get(
                        date_column
                    )
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
                        "date": (
                            parsed_date
                        ),
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
    """下載並解析基金近期NAV。"""

    response = None
    last_error = None

    for attempt in range(3):

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=45,
            )

            response.raise_for_status()
            break

        except requests.RequestException as error:

            response = None
            last_error = error

            if attempt < 2:
                time.sleep(
                    2 * (attempt + 1)
                )

    if response is None:

        return (
            empty_history(),
            (
                "連線失敗或逾時："
                f"{type(last_error).__name__}: "
                f"{last_error}"
            ),
        )

    try:
        response.encoding = (
            response.apparent_encoding
            or response.encoding
        )

        result = parse_fund_tables(
            response.text
        )

        if result.empty:

            return (
                empty_history(),
                (
                    "頁面可開啟，"
                    "但沒有解析到日期與淨值"
                ),
            )

        return result, None

    except Exception as error:

        return (
            empty_history(),
            (
                "解析失敗："
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_online_fund_history(
    urls,
):
    """依序嘗試線上基金來源。"""

    frames = []
    errors = []
    successful_urls = []

    for url in urls:

        data, error = (
            fetch_fund_url(url)
        )

        if not data.empty:

            frames.append(data)

            successful_urls.append(
                url
            )

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

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        normalize_history(combined),
        errors,
        successful_urls,
    )


def merge_histories(
    local_history,
    online_history,
):
    """
    合併CSV及線上近期NAV。

    相同日期以線上資料優先。
    """

    frames = []

    if not local_history.empty:

        local_data = (
            local_history.copy()
        )

        local_data["priority"] = 1

        frames.append(
            local_data
        )

    if not online_history.empty:

        online_data = (
            online_history.copy()
        )

        online_data["priority"] = 2

        frames.append(
            online_data
        )

    if not frames:
        return empty_history()

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    merged = merged.sort_values(
        [
            "date",
            "priority",
        ]
    )

    merged = merged.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return normalize_history(
        merged[
            [
                "date",
                "nav",
            ]
        ]
    )


# ============================================================
# 績效計算
# ============================================================

def get_value_on_or_before(
    history,
    target_date,
):
    """找指定日期以前最近一筆資料。"""

    available = history[
        history["date"]
        <= pd.Timestamp(
            target_date
        )
    ]

    if available.empty:
        return None, np.nan

    row = available.iloc[-1]

    return (
        row["date"],
        row["nav"],
    )


def calculate_period_return(
    history,
    current_value,
    target_date,
    maximum_gap_days=10,
):
    """計算指定期間累積報酬。"""

    (
        base_date,
        base_value,
    ) = get_value_on_or_before(
        history,
        target_date,
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

    return pct_change(
        current_value,
        base_value,
    )


def calculate_performance(
    history,
    selected_date,
):
    """計算各期間累積報酬。"""

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

    if previous_data.empty:
        previous_value = np.nan

    else:
        previous_value = (
            previous_data
            .iloc[-1]["nav"]
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
        "Daily": pct_change(
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
        "1Y累積": (
            calculate_period_return(
                history,
                current_value,
                current_date
                - pd.DateOffset(
                    years=1
                ),
            )
        ),
        "2Y累積": (
            calculate_period_return(
                history,
                current_value,
                current_date
                - pd.DateOffset(
                    years=2
                ),
            )
        ),
        "3Y累積": (
            calculate_period_return(
                history,
                current_value,
                current_date
                - pd.DateOffset(
                    years=3
                ),
            )
        ),
    }


# ============================================================
# 期間篩選及風險計算
# ============================================================

def filter_period(
    history,
    selected_date,
    period_name,
):
    """依選擇期間篩選歷史資料。"""

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
    annual_risk_free_rate,
):
    """
    計算：
    年化報酬率、波動率、
    最大回撤及Sharpe Ratio。
    """

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
        daily_returns.std(ddof=1)
        * np.sqrt(252)
    )

    running_max = (
        history["nav"]
        .cummax()
    )

    drawdown = (
        history["nav"]
        / running_max
        - 1
    )

    maximum_drawdown = (
        drawdown.min()
    )

    if annual_volatility > 0:
        sharpe_ratio = (
            annual_return
            - annual_risk_free_rate
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
        index=0,
    )

    risk_free_rate_pct = (
        st.number_input(
            "無風險利率（年化%）",
            min_value=0.0,
            max_value=10.0,
            value=(
                DEFAULT_RISK_FREE_RATE
            ),
            step=0.1,
        )
    )

    st.divider()

    st.info(
        "下次預計更新："
        f"{scheduled_update_text()}"
        "（台灣時間）"
    )

    st.caption(
        "GitHub Actions於工作日"
        "19:00及21:30排程，"
        "實際啟動可能延遲數分鐘。"
    )


# ============================================================
# 載入CSV
# ============================================================

(
    local_fund_data,
    csv_quality,
) = inspect_and_load_fund_csv()


# ============================================================
# 取得各商品資料
# ============================================================

results = []
histories = {}
diagnostics = []
status_rows = []

progress_bar = st.progress(
    0,
    text="正在取得與整理資料…",
)

product_items = list(
    PRODUCTS.items()
)

for index, (
    product_code,
    product_info,
) in enumerate(product_items):

    is_etf = (
        product_info["type"]
        == "ETF"
    )

    if is_etf:

        history = fetch_etf_history(
            product_code,
            selected_date,
        )

        product_type_text = "ETF"
        data_source = "TWSE"

    else:

        local_history = (
            get_local_history(
                product_code,
                product_info["name"],
                local_fund_data,
            )
        )

        (
            online_history,
            fetch_errors,
            successful_urls,
        ) = (
            fetch_online_fund_history(
                tuple(
                    product_info[
                        "urls"
                    ]
                )
            )
        )

        history = merge_histories(
            local_history,
            online_history,
        )

        product_type_text = (
            "共同基金"
        )

        if (
            not online_history.empty
            and not local_history.empty
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
                    product_info[
                        "name"
                    ]
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
                "errors": fetch_errors,
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

    performance = (
        calculate_performance(
            history,
            selected_date,
        )
    )

    result_row = {
        "商品名稱": (
            product_info["name"]
        ),
        "類型": (
            product_type_text
        ),
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

    if performance:

        result_row.update(
            {
                "資料日期": (
                    performance["date"]
                    .strftime(
                        "%Y/%m/%d"
                    )
                ),
                "價格／淨值": (
                    performance["value"]
                ),
                **{
                    column: (
                        performance[
                            column
                        ]
                    )
                    for column
                    in PERIOD_COLUMNS
                },
            }
        )

    results.append(
        result_row
    )

    if history.empty:

        status_text = "無資料"
        latest_date = None

    else:

        latest_date = (
            history["date"].max()
        )

        age_days = (
            pd.Timestamp(
                selected_date
            )
            - latest_date
        ).days

        if age_days <= 3:
            status_text = "正常"
        else:
            status_text = (
                "資料較舊"
            )

    status_rows.append(
        {
            "商品名稱": (
                product_info["name"]
            ),
            "類型": (
                product_type_text
            ),
            "最新資料日期": (
                latest_date.strftime(
                    "%Y/%m/%d"
                )
                if latest_date
                is not None
                else "-"
            ),
            "資料筆數": len(
                history
            ),
            "資料來源": (
                data_source
            ),
            "資料狀態": (
                status_text
            ),
        }
    )

    progress_bar.progress(
        (
            index + 1
        )
        / len(product_items),
        text=(
            "正在處理 "
            f"{product_info['name']}…"
        ),
    )

progress_bar.empty()

result_df = pd.DataFrame(
    results
)

status_df = pd.DataFrame(
    status_rows
)


# ============================================================
# 共用頁首
# ============================================================

st.title(
    "📊 ETF及共同基金績效比較"
)

st.caption(
    "ETF採分割調整後市場收盤價，"
    "共同基金採正式NAV；"
    "各期間均為未含配息的"
    "累積報酬率。"
)


valid_dates = pd.to_datetime(
    result_df.loc[
        result_df["資料日期"]
        != "-",
        "資料日期",
    ],
    errors="coerce",
).dropna()

if valid_dates.empty:
    latest_date_text = "-"

else:
    latest_date_text = (
        valid_dates.max()
        .strftime("%Y/%m/%d")
    )


metric_col1, metric_col2, \
metric_col3, metric_col4 = (
    st.columns(4)
)

metric_col1.metric(
    "查詢日期",
    pd.Timestamp(
        selected_date
    ).strftime("%Y/%m/%d"),
)

metric_col2.metric(
    "全體最新資料日",
    latest_date_text,
    help=(
        "所有商品中最新的一個"
        "資料日期。個別商品日期"
        "請以表格為準。"
    ),
)

metric_col3.metric(
    "比較商品數",
    len(result_df),
)

metric_col4.metric(
    "下次預計更新",
    scheduled_update_text(),
)


# ============================================================
# 頁面1：績效總覽
# ============================================================

if page == "📋 績效總覽":

    st.subheader(
        "📋 績效總覽"
    )

    display_df = (
        result_df.copy()
    )

    download_df = (
        result_df.copy()
    )

    display_df[
        "價格／淨值"
    ] = (
        display_df[
            "價格／淨值"
        ]
        .apply(format_number)
    )

    for column in PERIOD_COLUMNS:

        display_df[column] = (
            display_df[column]
            .apply(format_pct)
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

    styled_df = (
        display_df[
            display_columns
        ]
        .style.map(
            performance_color,
            subset=PERIOD_COLUMNS,
        )
        .set_properties(
            subset=[
                "類型",
                "資料日期",
                "價格／淨值",
                *PERIOD_COLUMNS,
            ],
            **{
                "text-align": (
                    "center"
                ),
            },
        )
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "註：Daily、MTD、YTD、"
        "1Y累積、2Y累積及3Y累積"
        "皆為未含配息的累積報酬率；"
        "2Y累積與3Y累積"
        "並非年化報酬率。"
    )

    csv_bytes = (
        download_df.to_csv(
            index=False,
            encoding="utf-8-sig",
        )
        .encode("utf-8-sig")
    )

    st.download_button(
        "⬇️ 下載績效表 CSV",
        data=csv_bytes,
        file_name=(
            "performance_"
            f"{pd.Timestamp(selected_date):%Y%m%d}"
            ".csv"
        ),
        mime="text/csv",
    )

    missing_products = result_df[
        result_df["資料來源"]
        == "無可用資料"
    ]["商品名稱"].tolist()

    if missing_products:

        st.warning(
            "目前無法取得可使用資料："
            + "、".join(
                missing_products
            )
        )


# ============================================================
# 頁面2：走勢分析
# ============================================================

elif page == "📈 走勢分析":

    st.subheader(
        "📈 走勢分析"
    )

    st.caption(
        f"目前期間：{chart_period}"
    )

    chart_frames = []
    normalized_frames = []
    interval_rows = []

    for (
        product_name,
        history,
    ) in histories.items():

        sliced = filter_period(
            history,
            selected_date,
            chart_period,
        )

        if sliced.empty:
            continue

        chart_data = (
            sliced.copy()
        )

        chart_data[
            "商品名稱"
        ] = product_name

        chart_frames.append(
            chart_data
        )

        first_value = (
            sliced.iloc[0]["nav"]
        )

        last_value = (
            sliced.iloc[-1]["nav"]
        )

        if first_value > 0:

            normalized_data = (
                sliced.copy()
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
                        (
                            last_value
                            / first_value
                            - 1
                        )
                        * 100
                    ),
                }
            )

    st.markdown(
        "#### 價格／淨值走勢"
    )

    if chart_frames:

        chart_df = pd.concat(
            chart_frames,
            ignore_index=True,
        )

        price_figure = px.line(
            chart_df,
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
            "原始尺度不同，"
            "此圖適合查看各商品本身的"
            "價格或淨值走勢。"
        )

    else:

        st.info(
            "目前期間沒有可繪製資料。"
        )

    st.markdown(
        "#### 區間累積報酬比較"
    )

    if interval_rows:

        interval_df = pd.DataFrame(
            interval_rows
        )

        interval_df = (
            interval_df.sort_values(
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

        st.caption(
            "各商品第一筆資料設為100，"
            "適合比較不同價格尺度商品的"
            "相對績效。"
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
        "無風險利率："
        f"{risk_free_rate_pct:.1f}%；"
        "至少需要"
        f"{MIN_RISK_OBSERVATIONS}"
        "筆每日資料。"
    )

    risk_rows = []
    correlation_series = {}

    for (
        product_name,
        history,
    ) in histories.items():

        sliced = filter_period(
            history,
            selected_date,
            risk_period,
        )

        metrics = (
            calculate_risk_metrics(
                sliced,
                risk_free_rate_pct
                / 100,
            )
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

            series = (
                sliced.set_index(
                    "date"
                )["nav"]
                .pct_change()
                .rename(product_name)
            )

            correlation_series[
                product_name
            ] = series

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

        risk_return_figure = (
            px.scatter(
                risk_df,
                x="年化波動率",
                y="年化報酬率",
                text="商品名稱",
                size="資料筆數",
                color="Sharpe Ratio",
                color_continuous_scale=(
                    "RdYlGn"
                ),
                labels={
                    "年化波動率": (
                        "年化波動率 (%)"
                    ),
                    "年化報酬率": (
                        "年化報酬率 (%)"
                    ),
                },
            )
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

        risk_col1, risk_col2 = (
            st.columns(2)
        )

        with risk_col1:

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
                labels={
                    "最大回撤": (
                        "最大回撤 (%)"
                    ),
                },
                color_discrete_sequence=[
                    "#d95f59"
                ],
            )

            drawdown_figure.update_layout(
                height=430
            )

            st.plotly_chart(
                drawdown_figure,
                use_container_width=True,
            )

        with risk_col2:

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
                labels={
                    "年化波動率": (
                        "年化波動率 (%)"
                    ),
                },
                color_discrete_sequence=[
                    "#4c78a8"
                ],
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
            color_continuous_scale=(
                "RdYlGn"
            ),
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
                .sum()
                .sum()
                > 0
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

            display_risk = (
                risk_df.copy()
            )

            for column in [
                "年化報酬率",
                "年化波動率",
                "最大回撤",
            ]:
                display_risk[
                    column
                ] = (
                    display_risk[column]
                    .map(
                        lambda value: (
                            f"{value:.2f}%"
                        )
                    )
                )

            display_risk[
                "Sharpe Ratio"
            ] = (
                display_risk[
                    "Sharpe Ratio"
                ]
                .map(
                    lambda value: (
                        f"{value:.2f}"
                    )
                )
            )

            st.dataframe(
                display_risk,
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
            "檢核項目": (
                "CSV檔案存在"
            ),
            "結果": (
                "正常"
                if csv_quality[
                    "exists"
                ]
                else "異常"
            ),
            "說明": str(
                LOCAL_FUND_FILE
            ),
        },
        {
            "檢核項目": (
                "必要欄位"
            ),
            "結果": (
                "正常"
                if csv_quality[
                    "columns_ok"
                ]
                else "異常"
            ),
            "說明": (
                "date、code、"
                "name、nav"
            ),
        },
        {
            "檢核項目": (
                "無法解析日期"
            ),
            "結果": (
                "正常"
                if csv_quality[
                    "invalid_dates"
                ] == 0
                else "提醒"
            ),
            "說明": (
                f"{csv_quality['invalid_dates']}"
                " 筆"
            ),
        },
        {
            "檢核項目": (
                "無效淨值"
            ),
            "結果": (
                "正常"
                if csv_quality[
                    "invalid_nav"
                ] == 0
                else "提醒"
            ),
            "說明": (
                f"{csv_quality['invalid_nav']}"
                " 筆"
            ),
        },
        {
            "檢核項目": (
                "重複資料"
            ),
            "結果": (
                "正常"
                if csv_quality[
                    "duplicate_rows"
                ] == 0
                else "提醒"
            ),
            "說明": (
                f"{csv_quality['duplicate_rows']}"
                " 筆"
            ),
        },
        {
            "檢核項目": (
                "未知基金代碼"
            ),
            "結果": (
                "正常"
                if not
                csv_quality[
                    "unknown_codes"
                ]
                else "異常"
            ),
            "說明": (
                "、".join(
                    csv_quality[
                        "unknown_codes"
                    ]
                )
                or "無"
            ),
        },
        {
            "檢核項目": (
                "基金代碼缺漏"
            ),
            "結果": (
                "正常"
                if not
                csv_quality[
                    "missing_codes"
                ]
                else "異常"
            ),
            "說明": (
                "、".join(
                    csv_quality[
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
            "資料可使用，但有需要"
            "留意的品質提醒。"
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
                    "商品名稱": (
                        product_name
                    ),
                    "最早日期": "-",
                    "最新日期": "-",
                    "最大日期缺口": "-",
                    "缺口提醒": (
                        "無資料"
                    ),
                }
            )

            continue

        date_gaps = (
            history["date"]
            .sort_values()
            .diff()
            .dt.days
            .dropna()
        )

        if date_gaps.empty:
            max_gap = 0

        else:
            max_gap = int(
                date_gaps.max()
            )

        coverage_rows.append(
            {
                "商品名稱": (
                    product_name
                ),
                "最早日期": (
                    history["date"]
                    .min()
                    .strftime(
                        "%Y/%m/%d"
                    )
                ),
                "最新日期": (
                    history["date"]
                    .max()
                    .strftime(
                        "%Y/%m/%d"
                    )
                ),
                "最大日期缺口": (
                    f"{max_gap}天"
                ),
                "缺口提醒": (
                    "提醒"
                    if max_gap > 10
                    else "正常"
                ),
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

    with st.expander(
        "共同基金線上來源診斷"
    ):

        for item in diagnostics:

            st.markdown(
                f"**{item['name']}**"
            )

            st.write(
                "線上取得 "
                f"{item['online_rows']} 筆；"
                "CSV取得 "
                f"{item['csv_rows']} 筆。"
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

            if item[
                "successful_urls"
            ]:

                st.write(
                    "成功來源："
                )

                for source_url in item[
                    "successful_urls"
                ]:
                    st.code(
                        source_url,
                        language="text",
                    )

            if item["errors"]:

                st.write(
                    "來源錯誤或提醒："
                )

                for error_message in item[
                    "errors"
                ]:
                    st.code(
                        error_message,
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
- 1Y、2Y、3Y不是年化報酬率。
- 基準資料距目標日超過10天時，績效顯示「-」。
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
