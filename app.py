import re
import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# 頁面設定
# ============================================================

st.set_page_config(
    page_title="台灣 ETF 及共同基金績效比較",
    page_icon="📊",
    layout="wide",
)

st.title("📊 台灣 ETF 及共同基金績效比較")

st.caption(
    "比較 ETF 與共同基金於指定日期的淨值／收盤價，"
    "以及 Daily、MTD、YTD、1Y、2Y、3Y 績效。"
)


# ============================================================
# 商品設定
# ============================================================

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
            "https://fund.hncb.com.tw/w/wr/wr02_ACPS10-009007.djhtm",
            "https://www.moneydj.com/funddj/ya/yp010000.djhtm?a=ACPS10",
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
            "https://fund.hncb.com.tw/w/wr/wr02.djhtm?a=ACKG15-014013",
            "https://cotabank.moneydj.com/w/wr/wr02_ACKG15.djhtm",
            "https://www.moneydj.com/funddj/ya/yp010000.djhtm?a=ACKG15",
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


# ============================================================
# 共用設定
# ============================================================

LOCAL_FUND_FILE = Path("fund_nav_history.csv")

PERIOD_COLUMNS = [
    "Daily",
    "MTD",
    "YTD",
    "1Y",
    "2Y",
    "3Y",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/130 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


# ============================================================
# ETF 分割還原設定
# ============================================================

ETF_SPLIT_CONFIG = {
    "0050": [
        {
            "effective_date": pd.Timestamp("2025-06-18"),
            "ratio": 4.0,
        }
    ],
}


# ============================================================
# 共用工具
# ============================================================

def empty_history():
    return pd.DataFrame(
        columns=["date", "nav"]
    )


def safe_float(value):
    """安全地將資料轉換成數字。"""

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
    """統一歷史資料欄位與格式。"""

    if data is None or data.empty:
        return empty_history()

    if not {"date", "nav"}.issubset(
        data.columns
    ):
        return empty_history()

    result = data[
        ["date", "nav"]
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
    except TypeError:
        pass

    result["nav"] = result[
        "nav"
    ].apply(safe_float)

    result = result.dropna(
        subset=["date", "nav"]
    )

    if result.empty:
        return empty_history()

    result["date"] = (
        result["date"]
        .dt.normalize()
    )

    result = result[
        result["nav"] > 0
    ]

    result = result.sort_values(
        "date"
    )

    result = result.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return result.reset_index(
        drop=True
    )


def parse_roc_or_ad_date(value):
    """
    支援：
    115/10/27
    2026/10/27
    2026-10-27
    """

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
    將 ETF 分割前價格換算成分割後基準。

    0050 在 2025-06-18 進行 1 拆 4，
    因此該日期以前的原始價格除以 4。
    """

    if pd.isna(trade_date):
        return close_price

    if pd.isna(close_price):
        return close_price

    adjusted_price = float(
        close_price
    )

    split_events = ETF_SPLIT_CONFIG.get(
        stock_code,
        [],
    )

    for split_event in split_events:

        effective_date = split_event[
            "effective_date"
        ]

        split_ratio = split_event[
            "ratio"
        ]

        if trade_date < effective_date:
            adjusted_price = (
                adjusted_price
                / split_ratio
            )

    return adjusted_price


def pct_change(
    current_value,
    base_value,
):
    """計算百分比變化。"""

    if pd.isna(current_value):
        return np.nan

    if pd.isna(base_value):
        return np.nan

    if base_value == 0:
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
    """取得 ETF 單月交易資料。"""

    url = (
        "https://www.twse.com.tw/"
        "exchangeReport/STOCK_DAY"
    )

    params = {
        "response": "json",
        "date": f"{year}{month:02d}01",
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

        adjusted_close_price = (
            adjust_etf_close_price(
                stock_code=stock_code,
                trade_date=trade_date,
                close_price=close_price,
            )
        )

        if (
            pd.notna(trade_date)
            and pd.notna(
                adjusted_close_price
            )
        ):
            records.append(
                {
                    "date": trade_date,
                    "nav": adjusted_close_price,
                }
            )

    return normalize_history(
        pd.DataFrame(records)
    )


def month_iterator(
    start_date,
    end_date,
):
    """依月份產生年月。"""

    current = pd.Timestamp(
        year=start_date.year,
        month=start_date.month,
        day=1,
    )

    ending = pd.Timestamp(
        year=end_date.year,
        month=end_date.month,
        day=1,
    )

    while current <= ending:

        yield (
            current.year,
            current.month,
        )

        current = (
            current
            + pd.offsets.MonthBegin(1)
        )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_etf_history(
    stock_code,
    selected_date,
):
    """取得 ETF 約三年又兩個月資料。"""

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
        monthly_data = fetch_twse_month(
            stock_code,
            year,
            month,
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
        history["date"] <= end_date
    ]

    return history.reset_index(
        drop=True
    )


# ============================================================
# 共同基金：讀取 CSV
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_local_fund_history():
    """讀取 fund_nav_history.csv。"""

    empty_result = pd.DataFrame(
        columns=[
            "date",
            "code",
            "name",
            "nav",
        ]
    )

    if not LOCAL_FUND_FILE.exists():
        return empty_result

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
                dtype={"code": str},
            )

            break

        except Exception:
            continue

    if data is None:
        return empty_result

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
        "NAV": "nav",
        "Nav": "nav",
    }

    data = data.rename(
        columns=rename_map
    )

    if not {
        "date",
        "code",
        "nav",
    }.issubset(data.columns):
        return empty_result

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

    data["nav"] = data[
        "nav"
    ].apply(safe_float)

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

    data = data.sort_values(
        ["code", "date"]
    )

    data = data.drop_duplicates(
        subset=[
            "code",
            "date",
        ],
        keep="last",
    )

    return data.reset_index(
        drop=True
    )


def get_local_history(
    fund_code,
    fund_name,
):
    """取得單一基金的 CSV 歷史資料。"""

    data = load_local_fund_history()

    if data.empty:
        return empty_history()

    result = data[
        data["code"] == fund_code
    ].copy()

    if result.empty:
        result = data[
            data["name"].str.contains(
                fund_name,
                regex=False,
                na=False,
            )
        ].copy()

    if result.empty:
        return empty_history()

    return normalize_history(
        result[
            ["date", "nav"]
        ]
    )


# ============================================================
# 共同基金：解析線上表格
# ============================================================

def flatten_column_name(column):
    if isinstance(column, tuple):
        return " ".join(
            str(part)
            for part in column
        ).strip()

    return str(column).strip()


def choose_date_nav_columns(table):
    """尋找日期欄位及淨值欄位。"""

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


def parse_fund_tables(html_text):
    """從基金網頁表格解析日期與淨值。"""

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
    """下載並解析單一基金頁面。"""

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
                f"{last_error}；"
                f"網址：{url}"
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
                    "頁面可開啟，但沒有解析到"
                    "日期與淨值資料："
                    f"{url}"
                ),
            )

        return result, None

    except Exception as error:

        return (
            empty_history(),
            (
                "解析失敗："
                f"{type(error).__name__}: "
                f"{error}；"
                f"網址：{url}"
            ),
        )


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_online_fund_history(urls):
    """依序嘗試多個基金來源。"""

    frames = []
    errors = []
    successful_urls = []

    for url in urls:

        data, error = fetch_fund_url(
            url
        )

        if not data.empty:

            frames.append(data)

            successful_urls.append(
                url
            )

        elif error:

            errors.append(error)

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

    combined = normalize_history(
        combined
    )

    return (
        combined,
        errors,
        successful_urls,
    )


def merge_histories(
    local_history,
    online_history,
):
    """合併 CSV 與線上資料。"""

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
        ["date", "priority"]
    )

    merged = merged.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return normalize_history(
        merged[
            ["date", "nav"]
        ]
    )


# ============================================================
# 指定日期與績效計算
# ============================================================

def get_value_on_or_before(
    history,
    target_date,
):
    """找指定日期以前最近一筆資料。"""

    available = history[
        history["date"]
        <= pd.Timestamp(target_date)
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
    current_nav,
    target_date,
    maximum_gap_days=10,
):
    """計算指定期間累積報酬。"""

    (
        base_date,
        base_nav,
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
        current_nav,
        base_nav,
    )


def calculate_performance(
    history,
    selected_date,
):
    """計算各期間績效。"""

    history = normalize_history(
        history
    )

    if history.empty:
        return None

    (
        current_date,
        current_nav,
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
        previous_nav = np.nan
    else:
        previous_nav = (
            previous_data
            .iloc[-1]["nav"]
        )

    month_target = (
        current_date.replace(day=1)
        - timedelta(days=1)
    )

    year_target = (
        pd.Timestamp(
            year=current_date.year,
            month=1,
            day=1,
        )
        - timedelta(days=1)
    )

    return {
        "date": current_date,
        "nav": current_nav,

        "Daily": pct_change(
            current_nav,
            previous_nav,
        ),

        "MTD": calculate_period_return(
            history,
            current_nav,
            month_target,
        ),

        "YTD": calculate_period_return(
            history,
            current_nav,
            year_target,
        ),

        "1Y": calculate_period_return(
            history,
            current_nav,
            (
                current_date
                - pd.DateOffset(years=1)
            ),
        ),

        "2Y": calculate_period_return(
            history,
            current_nav,
            (
                current_date
                - pd.DateOffset(years=2)
            ),
        ),

        "3Y": calculate_period_return(
            history,
            current_nav,
            (
                current_date
                - pd.DateOffset(years=3)
            ),
        ),
    }


# ============================================================
# 表格顏色
# ============================================================

def performance_color(value):
    """台股習慣：紅漲、綠跌。"""

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
        return ""

    return ""


# ============================================================
# 日期選擇
# ============================================================

selected_date = st.date_input(
    "📅 選擇查詢日期",
    value=date.today(),
    max_value=date.today(),
)

st.caption(
    "若所選日期為假日或當日尚未公布資料，"
    "系統會採用該日期以前最近一筆可取得資料。"
)


# ============================================================
# 取得資料
# ============================================================

results = []
histories = {}
diagnostics = []

progress_bar = st.progress(0)

product_items = list(
    PRODUCTS.items()
)

for index, (
    product_code,
    product_info,
) in enumerate(product_items):

    if product_info["type"] == "ETF":

        history = fetch_etf_history(
            product_code,
            selected_date,
        )

        data_source = "TWSE"

    else:

        local_history = (
            get_local_history(
                product_code,
                product_info["name"],
            )
        )

        (
            online_history,
            fetch_errors,
            successful_urls,
        ) = fetch_online_fund_history(
            tuple(
                product_info["urls"]
            )
        )

        history = merge_histories(
            local_history,
            online_history,
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
                    fetch_errors
                ),
            }
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
        "基金名稱": (
            product_info["name"]
        ),
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

    if performance is not None:

        result_row.update(
            {
                "淨值日期": (
                    performance["date"]
                    .strftime(
                        "%Y/%m/%d"
                    )
                ),

                "最新淨值": (
                    performance["nav"]
                ),

                "Daily": (
                    performance["Daily"]
                ),

                "MTD": (
                    performance["MTD"]
                ),

                "YTD": (
                    performance["YTD"]
                ),

                "1Y": (
                    performance["1Y"]
                ),

                "2Y": (
                    performance["2Y"]
                ),

                "3Y": (
                    performance["3Y"]
                ),
            }
        )

    results.append(
        result_row
    )

    progress_bar.progress(
        (index + 1)
        / len(product_items)
    )

progress_bar.empty()

result_df = pd.DataFrame(
    results
)


# ============================================================
# 摘要資訊
# ============================================================

metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)

with metric_col1:

    st.metric(
        "查詢日期",
        pd.Timestamp(
            selected_date
        ).strftime(
            "%Y/%m/%d"
        ),
    )

with metric_col2:

    valid_dates = pd.to_datetime(
        result_df.loc[
            result_df["淨值日期"]
            != "-",
            "淨值日期",
        ],
        errors="coerce",
    ).dropna()

    if valid_dates.empty:
        latest_date_text = "-"
    else:
        latest_date_text = (
            valid_dates
            .max()
            .strftime("%Y/%m/%d")
        )

    st.metric(
        "目前最新資料日期",
        latest_date_text,
    )

with metric_col3:

    st.metric(
        "比較商品數",
        len(result_df),
    )


# ============================================================
# 績效表格
# ============================================================

st.subheader("📈 績效比較")

display_df = result_df.copy()

display_df["最新淨值"] = (
    display_df["最新淨值"]
    .apply(format_number)
)

for column in PERIOD_COLUMNS:

    display_df[column] = (
        display_df[column]
        .apply(format_pct)
    )

styled_df = display_df.style.map(
    performance_color,
    subset=PERIOD_COLUMNS,
)

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
)

missing_products = result_df[
    result_df["資料來源"]
    == "無可用資料"
]["基金名稱"].tolist()

if missing_products:

    st.warning(
        "目前無法取得可使用資料："
        + "、".join(
            missing_products
        )
    )


# ============================================================
# 圖表期間
# ============================================================

st.divider()

st.subheader(
    "📉 淨值／收盤價走勢"
)

chart_range = st.selectbox(
    "圖表期間",
    [
        "近30日",
        "近3個月",
        "近6個月",
        "近1年",
        "全部可用資料",
    ],
    index=1,
)

days_map = {
    "近30日": 30,
    "近3個月": 92,
    "近6個月": 184,
    "近1年": 366,
}

if chart_range == "全部可用資料":
    chart_start_date = None

else:
    chart_start_date = (
        pd.Timestamp(selected_date)
        - timedelta(
            days=days_map[
                chart_range
            ]
        )
    )


# ============================================================
# 淨值／收盤價走勢圖
# ============================================================

chart_frames = []

for (
    product_name,
    history,
) in histories.items():

    chart_data = history.copy()

    if chart_start_date is not None:

        chart_data = chart_data[
            chart_data["date"]
            >= chart_start_date
        ]

    if not chart_data.empty:

        chart_data[
            "基金名稱"
        ] = product_name

        chart_frames.append(
            chart_data
        )

if chart_frames:

    chart_df = pd.concat(
        chart_frames,
        ignore_index=True,
    )

    line_chart = px.line(
        chart_df,
        x="date",
        y="nav",
        color="基金名稱",
        markers=False,
        labels={
            "date": "日期",
            "nav": "淨值／收盤價",
            "基金名稱": "商品",
        },
    )

    line_chart.update_layout(
        hovermode="x unified",
        legend_title_text="商品",
        height=520,
    )

    st.plotly_chart(
        line_chart,
        use_container_width=True,
    )

else:

    st.info(
        "目前沒有可繪製的歷史資料。"
    )


# ============================================================
# 各期間績效比較圖
# ============================================================

st.subheader(
    "📊 各期間績效比較"
)

performance_long = result_df.melt(
    id_vars=["基金名稱"],
    value_vars=PERIOD_COLUMNS,
    var_name="期間",
    value_name="報酬率",
)

performance_long = (
    performance_long
    .dropna(
        subset=["報酬率"]
    )
)

if not performance_long.empty:

    bar_chart = px.bar(
        performance_long,
        x="期間",
        y="報酬率",
        color="基金名稱",
        barmode="group",
        text_auto=".2f",
        labels={
            "報酬率": "報酬率 (%)",
            "基金名稱": "商品",
        },
    )

    bar_chart.add_hline(
        y=0,
        line_width=1,
        line_color="gray",
    )

    bar_chart.update_layout(
        height=520,
        legend_title_text="商品",
    )

    st.plotly_chart(
        bar_chart,
        use_container_width=True,
    )

else:

    st.info(
        "目前沒有足夠資料繪製績效比較圖。"
    )


# ============================================================
# 標準化走勢圖
# ============================================================

st.subheader(
    "📐 起點標準化走勢（起點 = 100）"
)

normalized_frames = []

for (
    product_name,
    history,
) in histories.items():

    normalized_data = (
        history.copy()
    )

    if chart_start_date is not None:

        normalized_data = (
            normalized_data[
                normalized_data["date"]
                >= chart_start_date
            ]
        )

    if normalized_data.empty:
        continue

    first_nav = (
        normalized_data
        .iloc[0]["nav"]
    )

    if (
        pd.isna(first_nav)
        or first_nav == 0
    ):
        continue

    normalized_data[
        "標準化指數"
    ] = (
        normalized_data["nav"]
        / first_nav
        * 100
    )

    normalized_data[
        "基金名稱"
    ] = product_name

    normalized_frames.append(
        normalized_data
    )

if normalized_frames:

    normalized_df = pd.concat(
        normalized_frames,
        ignore_index=True,
    )

    normalized_chart = px.line(
        normalized_df,
        x="date",
        y="標準化指數",
        color="基金名稱",
        labels={
            "date": "日期",
            "標準化指數": (
                "起點 = 100"
            ),
            "基金名稱": "商品",
        },
    )

    normalized_chart.add_hline(
        y=100,
        line_dash="dash",
        line_color="gray",
    )

    normalized_chart.update_layout(
        hovermode="x unified",
        height=520,
        legend_title_text="商品",
    )

    st.plotly_chart(
        normalized_chart,
        use_container_width=True,
    )

else:

    st.info(
        "目前沒有可繪製的標準化走勢資料。"
    )


# ============================================================
# 資料抓取診斷
# ============================================================

with st.expander(
    "🔧 資料抓取診斷"
):

    for item in diagnostics:

        st.markdown(
            f"**{item['name']}**"
        )

        st.write(
            f"線上取得 "
            f"{item['online_rows']} 筆，"
            f"CSV 取得 "
            f"{item['csv_rows']} 筆。"
        )

        if (
            item["first_date"]
            is not None
        ):

            first_date_text = (
                item["first_date"]
                .strftime("%Y/%m/%d")
            )

            last_date_text = (
                item["last_date"]
                .strftime("%Y/%m/%d")
            )

            st.write(
                "線上資料期間："
                f"{first_date_text} "
                "至 "
                f"{last_date_text}"
            )

        if item["successful_urls"]:

            st.write(
                "成功取得資料的來源："
            )

            for source_url in item[
                "successful_urls"
            ]:

                st.code(
                    source_url,
                    language="text",
                )

        if item["errors"]:

            if item["successful_urls"]:

                st.caption(
                    "部分備援來源無法連線或解析，"
                    "但已有其他來源成功取得資料，"
                    "不影響目前顯示結果。"
                )

            else:

                st.write(
                    "所有線上來源皆無法取得："
                )

                for error_message in item[
                    "errors"
                ]:

                    st.code(
                        error_message,
                        language="text",
                    )


# ============================================================
# CSV 格式與長期資料說明
# ============================================================

with st.expander(
    "📁 fund_nav_history.csv 格式與長期資料說明"
):

    st.write(
        "fund_nav_history.csv "
        "必須與 app.py 放在 GitHub "
        "Repository 的相同位置。"
    )

    st.code(
        """date,code,name,nav
2026-10-27,A09012,統一奔騰基金,711.97
2026-10-28,A09012,統一奔騰基金,747.96
2026-10-27,A41209,凱基台灣精五門基金,181.37
2026-10-28,A41209,凱基台灣精五門基金,191.01""",
        language="text",
    )

    st.info(
        "公開基金網頁通常只提供近期資料。"
        "長期績效需由 fund_nav_history.csv "
        "持續累積，或由自動更新程式回補。"
    )


# ============================================================
# 績效計算說明
# ============================================================

with st.expander(
    "ℹ️ 績效計算方式"
):

    st.markdown(
        """
- **Daily**：目前淨值 ÷ 前一筆淨值 − 1
- **MTD**：目前淨值 ÷ 上月底最後一筆淨值 − 1
- **YTD**：目前淨值 ÷ 去年底最後一筆淨值 − 1
- **1Y**：目前淨值 ÷ 一年前最近一筆淨值 − 1
- **2Y**：目前淨值 ÷ 兩年前最近一筆淨值 − 1
- **3Y**：目前淨值 ÷ 三年前最近一筆淨值 − 1

若基準日期以前最近一筆資料距離目標日期超過 10 天，
系統會顯示「-」，避免使用錯誤期間計算績效。

0050 在 2025 年 6 月 18 日以前的價格會除以 4，
以處理 1 拆 4造成的價格不連續。
"""
    )


# ============================================================
# 頁尾
# ============================================================

st.divider()

st.caption(
    "ETF 資料來源為臺灣證券交易所；"
    "共同基金使用基金資訊頁面，"
    "並以 fund_nav_history.csv 作為備援。"
)

st.caption(
    "資料僅供研究與績效比較使用，"
    "正式淨值仍以基金公司公告為準。"
)
