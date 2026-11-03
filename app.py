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
# 基本設定
# ============================================================

st.set_page_config(
    page_title="ETF及共同基金績效比較",
    page_icon="📊",
    layout="wide",
)

TZ = ZoneInfo("Asia/Taipei")
CSV = Path("fund_nav_history.csv")

PERIODS = [
    "Daily",
    "MTD",
    "YTD",
    "1Y累積",
    "2Y累積",
    "3Y累積",
]

MIN_RISK = 60

FUND_CODES = {
    "A09012",
    "A41209",
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


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/130 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


SPLITS = {
    "0050": [
        (
            pd.Timestamp("2025-06-18"),
            4.0,
        )
    ]
}


# ============================================================
# 共用工具
# ============================================================

def empty():
    return pd.DataFrame(
        columns=[
            "date",
            "nav",
        ]
    )


def num(value):
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


def norm(data):
    if (
        data is None
        or data.empty
        or not {
            "date",
            "nav",
        }.issubset(data.columns)
    ):
        return empty()

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

    result["nav"] = result["nav"].map(
        num
    )

    try:
        result["date"] = (
            result["date"]
            .dt.tz_localize(None)
        )
    except Exception:
        pass

    result = result.dropna(
        subset=[
            "date",
            "nav",
        ]
    )

    result = result[
        result["nav"] > 0
    ]

    result["date"] = (
        result["date"]
        .dt.normalize()
    )

    return (
        result
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def parse_date(value):
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
            year,
            month,
            day,
        )
    except Exception:
        return pd.NaT


def pct(
    current_value,
    base_value,
):
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


# ============================================================
# 表格格式
# ============================================================

def fmt_num(value):
    if pd.isna(value):
        return "-"

    return (
        f"{value:,.4f}"
        .rstrip("0")
        .rstrip(".")
    )


def fmt_pct(value):
    if pd.isna(value):
        return "-"

    return f"{value:+.2f}%"


def fmt_daily(value):
    """
    Daily顯示格式：
    上漲：▲ +1.23%
    下跌：▼ -1.23%
    平盤：0.00%
    """

    if pd.isna(value):
        return "-"

    if value > 0:
        return f"▲ {value:+.2f}%"

    if value < 0:
        return f"▼ {value:.2f}%"

    return "0.00%"


def value_color(value):
    """
    台股顯示慣例：
    上漲紅色
    下跌綠色
    """

    if (
        not isinstance(value, str)
        or value == "-"
    ):
        return ""

    try:
        number = float(
            value
            .replace("%", "")
            .replace("+", "")
            .replace("▲", "")
            .replace("▼", "")
            .strip()
        )

        if number > 0:
            return (
                "color:#d62728;"
                "font-weight:700"
            )

        if number < 0:
            return (
                "color:#138a36;"
                "font-weight:700"
            )

    except Exception:
        pass

    return ""


def row_price_color(row):
    """
    價格／淨值依Daily漲跌顯示顏色。
    """

    styles = pd.Series(
        "",
        index=row.index,
    )

    daily_value = row.get(
        "_Daily數值",
        np.nan,
    )

    if pd.notna(daily_value):

        if daily_value > 0:
            styles[
                "價格／淨值"
            ] = (
                "color:#d62728;"
                "font-weight:700;"
                "text-align:right"
            )

        elif daily_value < 0:
            styles[
                "價格／淨值"
            ] = (
                "color:#138a36;"
                "font-weight:700;"
                "text-align:right"
            )

        else:
            styles[
                "價格／淨值"
            ] = (
                "text-align:right"
            )

    return styles


def display_width(value):
    """
    中文、全形字元算2格，
    英文、數字算1格。
    """

    total_width = 0

    for character in str(value):

        if (
            unicodedata.east_asian_width(
                character
            )
            in {
                "W",
                "F",
                "A",
            }
        ):
            total_width += 2
        else:
            total_width += 1

    return total_width


def auto_width(
    dataframe,
    column_name,
    extra=5,
    minimum=75,
    maximum=360,
):
    """
    欄寬依最長內容加5個字元計算。
    回傳整數像素，避免max-width錯誤。
    """

    widths = [
        display_width(
            column_name
        )
    ]

    widths.extend(
        dataframe[
            column_name
        ]
        .astype(str)
        .map(display_width)
        .tolist()
    )

    longest_width = max(
        widths
    )

    calculated_width = (
        longest_width + extra
    ) * 9

    return max(
        minimum,
        min(
            calculated_width,
            maximum,
        ),
    )


# ============================================================
# 下次排程時間
# ============================================================

def next_update():
    now = datetime.now(TZ)

    for day_add in range(8):

        target_day = (
            now.date()
            + timedelta(days=day_add)
        )

        if target_day.weekday() >= 5:
            continue

        for target_time in [
            dtime(19, 0),
            dtime(21, 30),
        ]:

            candidate = (
                datetime.combine(
                    target_day,
                    target_time,
                    tzinfo=TZ,
                )
            )

            if candidate > now:

                if (
                    target_day
                    == now.date()
                ):
                    prefix = "今日"

                elif (
                    target_day
                    == now.date()
                    + timedelta(days=1)
                ):
                    prefix = "明日"

                else:
                    prefix = (
                        f"{candidate:%m/%d}"
                    )

                return (
                    f"{prefix} "
                    f"{candidate:%H:%M}"
                )

    return "-"


# ============================================================
# ETF資料
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def twse_month(
    code,
    year,
    month,
):
    try:
        response = requests.get(
            (
                "https://www.twse.com.tw/"
                "exchangeReport/STOCK_DAY"
            ),
            params={
                "response": "json",
                "date": (
                    f"{year}"
                    f"{month:02d}"
                    "01"
                ),
                "stockNo": code,
            },
            headers=HEADERS,
            timeout=45,
        )

        response.raise_for_status()

        rows = response.json().get(
            "data",
            [],
        )

    except Exception:
        return empty()

    records = []

    for row in rows:

        if len(row) < 7:
            continue

        trade_date = parse_date(
            row[0]
        )

        close_price = num(
            row[6]
        )

        for (
            effective_date,
            split_ratio,
        ) in SPLITS.get(
            code,
            [],
        ):

            if (
                pd.notna(trade_date)
                and trade_date
                < effective_date
            ):
                close_price /= (
                    split_ratio
                )

        if (
            pd.notna(trade_date)
            and pd.notna(close_price)
        ):
            records.append(
                {
                    "date": trade_date,
                    "nav": close_price,
                }
            )

    return norm(
        pd.DataFrame(records)
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def etf_history(
    code,
    end_date,
):
    end_date = pd.Timestamp(
        end_date
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

    while (
        current_month
        <= ending_month
    ):

        monthly_data = twse_month(
            code,
            current_month.year,
            current_month.month,
        )

        if not monthly_data.empty:
            frames.append(
                monthly_data
            )

        current_month += (
            pd.offsets.MonthBegin(1)
        )

    if not frames:
        return empty()

    history = norm(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    return history[
        history["date"]
        <= end_date
    ].reset_index(drop=True)


# ============================================================
# CSV資料及品質檢核
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_csv():
    report = {
        "exists": CSV.exists(),
        "columns": False,
        "bad_date": 0,
        "bad_nav": 0,
        "duplicates": 0,
        "unknown": [],
        "missing": [],
    }

    blank = pd.DataFrame(
        columns=[
            "date",
            "code",
            "name",
            "nav",
        ]
    )

    if not CSV.exists():
        return blank, report

    raw_data = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp950",
        "big5",
    ]:

        try:
            raw_data = pd.read_csv(
                CSV,
                encoding=encoding,
                dtype={
                    "code": str,
                },
            )
            break

        except Exception:
            continue

    if raw_data is None:
        return blank, report

    raw_data.columns = [
        str(column).strip()
        for column
        in raw_data.columns
    ]

    raw_data = raw_data.rename(
        columns={
            "日期": "date",
            "Date": "date",
            "基金代碼": "code",
            "代碼": "code",
            "基金名稱": "name",
            "名稱": "name",
            "淨值": "nav",
            "基金淨值": "nav",
            "NAV": "nav",
        }
    )

    report["columns"] = {
        "date",
        "code",
        "name",
        "nav",
    }.issubset(
        raw_data.columns
    )

    if not report["columns"]:
        return blank, report

    parsed_dates = pd.to_datetime(
        raw_data["date"],
        errors="coerce",
    )

    parsed_nav = pd.to_numeric(
        raw_data["nav"],
        errors="coerce",
    )

    report["bad_date"] = int(
        parsed_dates.isna().sum()
    )

    report["bad_nav"] = int(
        (
            parsed_nav.isna()
            | (parsed_nav <= 0)
        ).sum()
    )

    raw_data["date"] = (
        parsed_dates
    )

    raw_data["nav"] = (
        parsed_nav
    )

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

    report["duplicates"] = int(
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

    report["unknown"] = sorted(
        valid_codes
        - FUND_CODES
    )

    report["missing"] = sorted(
        FUND_CODES
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
    )

    return (
        raw_data[
            [
                "date",
                "code",
                "name",
                "nav",
            ]
        ].reset_index(drop=True),
        report,
    )


def local_history(
    code,
    name,
    data,
):
    result = data[
        data["code"] == code
    ].copy()

    if result.empty:

        result = data[
            data["name"].str.contains(
                name,
                regex=False,
                na=False,
            )
        ].copy()

    if result.empty:
        return empty()

    return norm(
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

def parse_tables(
    html_text,
):
    records = []

    try:
        tables = pd.read_html(
            StringIO(html_text)
        )
    except Exception:
        tables = []

    for table in tables:

        date_column = None
        nav_column = None

        for column in table.columns:

            if isinstance(
                column,
                tuple,
            ):
                column_text = (
                    " ".join(
                        map(str, column)
                    )
                )
            else:
                column_text = str(
                    column
                )

            if (
                date_column is None
                and "日期"
                in column_text
            ):
                date_column = column

            if (
                nav_column is None
                and "淨值"
                in column_text
                and "日期"
                not in column_text
                and "最高"
                not in column_text
                and "最低"
                not in column_text
                and "漲跌"
                not in column_text
            ):
                nav_column = column

        if (
            date_column is None
            or nav_column is None
        ):
            continue

        for _, row in table.iterrows():

            parsed_date = parse_date(
                row.get(
                    date_column
                )
            )

            nav_value = num(
                row.get(
                    nav_column
                )
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

    return norm(
        pd.DataFrame(records)
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fund_url(url):
    last_error = None

    for attempt in range(3):

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=45,
            )

            response.raise_for_status()

            response.encoding = (
                response.apparent_encoding
                or response.encoding
            )

            result = parse_tables(
                response.text
            )

            if not result.empty:
                return result, None

            return (
                empty(),
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
        empty(),
        (
            f"{type(last_error).__name__}: "
            f"{last_error}"
        ),
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def online_history(urls):
    frames = []
    errors = []
    successful_urls = []

    for url in urls:

        data, error = fund_url(
            url
        )

        if not data.empty:

            frames.append(
                data
            )

            successful_urls.append(
                url
            )

        elif error:

            errors.append(
                f"{url}：{error}"
            )

    if not frames:

        return (
            empty(),
            errors,
            successful_urls,
        )

    return (
        norm(
            pd.concat(
                frames,
                ignore_index=True,
            )
        ),
        errors,
        successful_urls,
    )


def merge(
    local_data,
    online_data,
):
    frames = []

    if not local_data.empty:

        local_copy = (
            local_data.copy()
        )

        local_copy[
            "priority"
        ] = 1

        frames.append(
            local_copy
        )

    if not online_data.empty:

        online_copy = (
            online_data.copy()
        )

        online_copy[
            "priority"
        ] = 2

        frames.append(
            online_copy
        )

    if not frames:
        return empty()

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

    return norm(
        combined[
            [
                "date",
                "nav",
            ]
        ]
    )


# ============================================================
# 績效計算
# ============================================================

def before(
    history,
    target_date,
):
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


def period_return(
    history,
    current_value,
    target_date,
    max_gap=10,
):
    base_date, base_value = before(
        history,
        target_date,
    )

    if base_date is None:
        return np.nan

    gap_days = (
        pd.Timestamp(
            target_date
        )
        - base_date
    ).days

    if not (
        0
        <= gap_days
        <= max_gap
    ):
        return np.nan

    return pct(
        current_value,
        base_value,
    )


def performance(
    history,
    selected_date,
):
    history = norm(
        history
    )

    if history.empty:
        return None

    (
        current_date,
        current_value,
    ) = before(
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
        "Daily": pct(
            current_value,
            previous_value,
        ),
        "MTD": period_return(
            history,
            current_value,
            month_target,
        ),
        "YTD": period_return(
            history,
            current_value,
            year_target,
        ),
        "1Y累積": period_return(
            history,
            current_value,
            current_date
            - pd.DateOffset(
                years=1
            ),
        ),
        "2Y累積": period_return(
            history,
            current_value,
            current_date
            - pd.DateOffset(
                years=2
            ),
        ),
        "3Y累積": period_return(
            history,
            current_value,
            current_date
            - pd.DateOffset(
                years=3
            ),
        ),
    }


# ============================================================
# 期間及風險計算
# ============================================================

def slice_period(
    history,
    end_date,
    period_name,
):
    end_date = pd.Timestamp(
        end_date
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

    return history[
        (
            history["date"]
            >= end_date
            - offsets[period_name]
        )
        & (
            history["date"]
            <= end_date
        )
    ].copy()


def risk_metrics(
    history,
    risk_free_rate,
):
    history = norm(
        history
    )

    returns = (
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

    if len(returns) < MIN_RISK:
        return None

    years = (
        len(returns)
        / 252
    )

    total_return = (
        history.iloc[-1]["nav"]
        / history.iloc[0]["nav"]
        - 1
    )

    if (
        years > 0
        and 1 + total_return > 0
    ):
        annual_return = (
            (1 + total_return)
            ** (1 / years)
            - 1
        )
    else:
        annual_return = np.nan

    annual_volatility = (
        returns.std()
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
        "資料筆數": len(
            history
        ),
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

    selected = st.date_input(
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

    risk_free_rate = (
        st.number_input(
            "無風險利率（年化%）",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
        )
    )

    st.divider()

    st.info(
        "下次預計更新："
        f"{next_update()}"
        "（台灣時間）"
    )

    st.caption(
        "工作日19:00與21:30排程，"
        "實際啟動可能延遲數分鐘。"
    )


# ============================================================
# 載入及處理資料
# ============================================================

local_data, quality = load_csv()

results = []
histories = {}
diagnostics = []
statuses = []

progress_bar = st.progress(
    0,
    text="正在取得資料…",
)


for index, (
    code,
    info,
) in enumerate(
    PRODUCTS.items()
):

    if info["type"] == "ETF":

        history = etf_history(
            code,
            selected,
        )

        product_type = "ETF"
        source = "TWSE"

    else:

        local_fund = local_history(
            code,
            info["name"],
            local_data,
        )

        (
            online_fund,
            errors,
            successful_urls,
        ) = online_history(
            tuple(
                info["urls"]
            )
        )

        history = merge(
            local_fund,
            online_fund,
        )

        product_type = (
            "共同基金"
        )

        if (
            not local_fund.empty
            and not online_fund.empty
        ):
            source = (
                "線上資料 + CSV"
            )
        elif not online_fund.empty:
            source = (
                "線上基金資料"
            )
        elif not local_fund.empty:
            source = "CSV"
        else:
            source = "無可用資料"

        diagnostics.append(
            {
                "name": info["name"],
                "online": len(
                    online_fund
                ),
                "csv": len(
                    local_fund
                ),
                "start": (
                    online_fund[
                        "date"
                    ].min()
                    if not
                    online_fund.empty
                    else None
                ),
                "end": (
                    online_fund[
                        "date"
                    ].max()
                    if not
                    online_fund.empty
                    else None
                ),
                "ok": (
                    successful_urls
                ),
                "errors": errors,
            }
        )

    history = norm(
        history
    )

    history = history[
        history["date"]
        <= pd.Timestamp(selected)
    ].copy()

    histories[
        info["name"]
    ] = history

    product_performance = (
        performance(
            history,
            selected,
        )
    )

    result_row = {
        "商品名稱": info["name"],
        "類型": product_type,
        "資料日期": "-",
        "價格／淨值": np.nan,
        "Daily": np.nan,
        "MTD": np.nan,
        "YTD": np.nan,
        "1Y累積": np.nan,
        "2Y累積": np.nan,
        "3Y累積": np.nan,
        "資料來源": source,
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
                    in PERIODS
                },
            }
        )

    results.append(
        result_row
    )

    if history.empty:

        latest_product_date = None
        status_text = "無資料"

    else:

        latest_product_date = (
            history["date"].max()
        )

        age_days = (
            pd.Timestamp(selected)
            - latest_product_date
        ).days

        if age_days <= 3:
            status_text = "正常"
        else:
            status_text = (
                "資料較舊"
            )

    statuses.append(
        {
            "商品名稱": info["name"],
            "類型": product_type,
            "最新資料日期": (
                latest_product_date
                .strftime("%Y/%m/%d")
                if latest_product_date
                is not None
                else "-"
            ),
            "資料筆數": len(
                history
            ),
            "資料來源": source,
            "資料狀態": (
                status_text
            ),
        }
    )

    progress_bar.progress(
        (
            index + 1
        )
        / len(PRODUCTS),
        text=(
            "正在處理 "
            f"{info['name']}…"
        ),
    )


progress_bar.empty()

result = pd.DataFrame(
    results
)

status = pd.DataFrame(
    statuses
)


# ============================================================
# 置中標題及摘要資訊
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
    result.loc[
        result["資料日期"]
        != "-",
        "資料日期",
    ],
    errors="coerce",
).dropna()

if valid_dates.empty:
    latest_date = "-"
else:
    latest_date = (
        valid_dates.max()
        .strftime("%Y/%m/%d")
    )


metric_1, metric_2, \
metric_3, metric_4 = (
    st.columns(4)
)

metric_1.metric(
    "查詢日期",
    pd.Timestamp(
        selected
    ).strftime("%Y/%m/%d"),
)

metric_2.metric(
    "全體最新資料日",
    latest_date,
)

metric_3.metric(
    "比較商品數",
    len(result),
)

metric_4.metric(
    "下次預計更新",
    next_update(),
)


# ============================================================
# 頁面1：績效總覽
# ============================================================

if page == "📋 績效總覽":

    st.subheader(
        "📋 績效總覽"
    )

    display = result.copy()

    download_data = (
        result.copy()
    )

    # 保留Daily原始數值，
    # 供價格／淨值顏色判斷。
    display[
        "_Daily數值"
    ] = (
        display["Daily"]
        .copy()
    )

    display[
        "價格／淨值"
    ] = (
        display[
            "價格／淨值"
        ].map(fmt_num)
    )

    display[
        "Daily"
    ] = (
        display["Daily"]
        .map(fmt_daily)
    )

    for period in [
        "MTD",
        "YTD",
        "1Y累積",
        "2Y累積",
        "3Y累積",
    ]:
        display[period] = (
            display[period]
            .map(fmt_pct)
        )

    columns = [
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

    # 欄寬以最長內容加5個字元計算
    widths = {
        column: auto_width(
            display,
            column,
            extra=5,
        )
        for column in columns
    }

    widths[
        "商品名稱"
    ] = auto_width(
        display,
        "商品名稱",
        extra=5,
        minimum=220,
        maximum=420,
    )

    widths[
        "資料來源"
    ] = auto_width(
        display,
        "資料來源",
        extra=5,
        minimum=120,
        maximum=220,
    )

    table_data = display[
        [
            *columns,
            "_Daily數值",
        ]
    ].copy()

    styled = (
        table_data.style

        # 價格／淨值依Daily顯示紅綠
        .apply(
            row_price_color,
            axis=1,
        )

        # 所有期間報酬紅漲綠跌
        .map(
            value_color,
            subset=PERIODS,
        )

        # 類型及資料日期置中
        .set_properties(
            subset=[
                "類型",
                "資料日期",
            ],
            **{
                "text-align": (
                    "center"
                ),
            },
        )

        # 價格與所有報酬率靠右
        .set_properties(
            subset=[
                "價格／淨值",
                "Daily",
                "MTD",
                "YTD",
                "1Y累積",
                "2Y累積",
                "3Y累積",
            ],
            **{
                "text-align": (
                    "right"
                ),
                "font-variant-numeric": (
                    "tabular-nums"
                ),
            },
        )

        # 商品名稱與來源靠左
        .set_properties(
            subset=[
                "商品名稱",
                "資料來源",
            ],
            **{
                "text-align": (
                    "left"
                ),
            },
        )

        # 表頭置中
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        (
                            "text-align",
                            "center",
                        ),
                    ],
                },
            ]
        )

        # 隱藏輔助欄位
        .hide(
            subset=[
                "_Daily數值",
            ],
            axis="columns",
        )
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            column: (
                st.column_config.TextColumn(
                    column,
                    width=widths[column],
                )
            )
            for column in columns
        },
    )

    st.caption(
        "註：價格／淨值及Daily以紅色表示上漲、"
        "綠色表示下跌；Daily的▲代表上漲，"
        "▼代表下跌。各期間均為未含配息的"
        "累積報酬率。"
    )

    download_bytes = (
        download_data
        .to_csv(
            index=False,
        )
        .encode("utf-8-sig")
    )

    st.download_button(
        "⬇️ 下載績效表 CSV",
        data=download_bytes,
        file_name=(
            "performance_"
            f"{pd.Timestamp(selected):%Y%m%d}"
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

        sliced = slice_period(
            history,
            selected,
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
                        pct(
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

        chart_data = pd.concat(
            chart_frames,
            ignore_index=True,
        )

        figure = px.line(
            chart_data,
            x="date",
            y="nav",
            color="商品名稱",
            labels={
                "date": "日期",
                "nav": "價格／淨值",
            },
        )

        figure.update_layout(
            hovermode="x unified",
            height=520,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    else:

        st.info(
            "目前期間沒有可繪製資料。"
        )

    st.markdown(
        "#### 區間累積報酬比較"
    )

    if interval_rows:

        interval_data = (
            pd.DataFrame(
                interval_rows
            )
            .sort_values(
                "區間累積報酬率",
                ascending=False,
            )
        )

        figure = px.bar(
            interval_data,
            x="商品名稱",
            y="區間累積報酬率",
            color="區間累積報酬率",
            color_continuous_scale=(
                "RdYlGn"
            ),
            text_auto=".2f",
        )

        figure.update_layout(
            height=480,
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            figure,
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

        normalized_data = pd.concat(
            normalized_frames,
            ignore_index=True,
        )

        figure = px.line(
            normalized_data,
            x="date",
            y="標準化指數",
            color="商品名稱",
        )

        figure.add_hline(
            y=100,
            line_dash="dash",
            line_color="gray",
        )

        figure.update_layout(
            hovermode="x unified",
            height=520,
        )

        st.plotly_chart(
            figure,
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

    risk_rows = []
    correlation_series = {}

    for (
        product_name,
        history,
    ) in histories.items():

        sliced = slice_period(
            history,
            selected,
            risk_period,
        )

        metrics = risk_metrics(
            sliced,
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
                sliced
                .set_index("date")[
                    "nav"
                ]
                .pct_change()
                .rename(
                    product_name
                )
            )

    if not risk_rows:

        st.info(
            "目前沒有商品具備足夠資料"
            "計算風險指標。"
        )

    else:

        risk_data = pd.DataFrame(
            risk_rows
        )

        figure = px.scatter(
            risk_data,
            x="年化波動率",
            y="年化報酬率",
            text="商品名稱",
            size="資料筆數",
            color="Sharpe Ratio",
            color_continuous_scale=(
                "RdYlGn"
            ),
        )

        figure.update_traces(
            textposition="top center"
        )

        figure.update_layout(
            height=550
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        risk_col_1, risk_col_2 = (
            st.columns(2)
        )

        with risk_col_1:

            figure = px.bar(
                risk_data.sort_values(
                    "最大回撤"
                ),
                x="最大回撤",
                y="商品名稱",
                orientation="h",
                text_auto=".2f",
                title="最大回撤",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        with risk_col_2:

            figure = px.bar(
                risk_data.sort_values(
                    "年化波動率"
                ),
                x="年化波動率",
                y="商品名稱",
                orientation="h",
                text_auto=".2f",
                title="年化波動率",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        figure = px.bar(
            risk_data.sort_values(
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
            title="Sharpe Ratio",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        if (
            len(correlation_series)
            >= 2
        ):

            correlation = pd.concat(
                correlation_series.values(),
                axis=1,
            ).corr(
                min_periods=MIN_RISK
            )

            if (
                correlation
                .notna()
                .any()
                .any()
            ):

                figure = px.imshow(
                    correlation,
                    text_auto=".2f",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale=(
                        "RdBu_r"
                    ),
                    title=(
                        "每日報酬"
                        "相關係數矩陣"
                    ),
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

            else:

                st.info(
                    "共同有效日期不足，"
                    "無法計算相關係數。"
                )


# ============================================================
# 頁面4：資料狀態
# ============================================================

else:

    st.subheader(
        "🔧 資料狀態"
    )

    st.dataframe(
        status,
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
                if quality["exists"]
                else "異常"
            ),
            "說明": str(CSV),
        },
        {
            "檢核項目": (
                "必要欄位"
            ),
            "結果": (
                "正常"
                if quality["columns"]
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
                if quality["bad_date"]
                == 0
                else "提醒"
            ),
            "說明": (
                f"{quality['bad_date']}筆"
            ),
        },
        {
            "檢核項目": (
                "無效淨值"
            ),
            "結果": (
                "正常"
                if quality["bad_nav"]
                == 0
                else "提醒"
            ),
            "說明": (
                f"{quality['bad_nav']}筆"
            ),
        },
        {
            "檢核項目": (
                "重複資料"
            ),
            "結果": (
                "正常"
                if quality[
                    "duplicates"
                ] == 0
                else "提醒"
            ),
            "說明": (
                f"{quality['duplicates']}筆"
            ),
        },
        {
            "檢核項目": (
                "未知基金代碼"
            ),
            "結果": (
                "正常"
                if not quality[
                    "unknown"
                ]
                else "異常"
            ),
            "說明": (
                "、".join(
                    quality["unknown"]
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
                if not quality[
                    "missing"
                ]
                else "異常"
            ),
            "說明": (
                "、".join(
                    quality["missing"]
                )
                or "無"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(
            quality_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

    if any(
        item["結果"] == "異常"
        for item in quality_rows
    ):

        st.error(
            "發現資料品質異常，"
            "請查看檢核說明。"
        )

    elif any(
        item["結果"] == "提醒"
        for item in quality_rows
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
                    "最大缺口區間": "-",
                    "缺口提醒": (
                        "無資料"
                    ),
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

            max_gap = 0
            gap_period = "-"

        else:

            max_gap_index = (
                date_gaps.idxmax()
            )

            max_gap = int(
                date_gaps.loc[
                    max_gap_index
                ]
            )

            gap_start = (
                sorted_dates.loc[
                    max_gap_index - 1
                ]
            )

            gap_end = (
                sorted_dates.loc[
                    max_gap_index
                ]
            )

            gap_period = (
                f"{gap_start:%Y/%m/%d}"
                " 至 "
                f"{gap_end:%Y/%m/%d}"
            )

        if max_gap > 30:
            gap_status = "異常"
        elif max_gap > 14:
            gap_status = "提醒"
        else:
            gap_status = "正常"

        coverage_rows.append(
            {
                "商品名稱": (
                    product_name
                ),
                "最早日期": (
                    sorted_dates.min()
                    .strftime(
                        "%Y/%m/%d"
                    )
                ),
                "最新日期": (
                    sorted_dates.max()
                    .strftime(
                        "%Y/%m/%d"
                    )
                ),
                "最大日期缺口": (
                    f"{max_gap}天"
                ),
                "最大缺口區間": (
                    gap_period
                ),
                "缺口提醒": (
                    gap_status
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            coverage_rows
        ),
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
                f"{item['online']}筆；"
                "CSV取得"
                f"{item['csv']}筆。"
            )

            if (
                item["start"]
                is not None
            ):

                st.write(
                    "線上期間："
                    f"{item['start']:%Y/%m/%d}"
                    " 至 "
                    f"{item['end']:%Y/%m/%d}"
                )

            if item["ok"]:

                st.write(
                    "成功來源："
                )

                for url in item["ok"]:
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
- 基準資料距目標日超過10天時，績效顯示「-」。
- 風險指標至少需要60筆每日資料。
"""
        )


st.divider()

st.caption(
    "資料僅供研究與績效比較使用。"
    "ETF正式基金淨值可能與市場收盤價不同；"
    "共同基金正式資料仍以基金公司公告為準。"
)
