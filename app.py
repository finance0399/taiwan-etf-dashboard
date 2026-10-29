import re
from io import StringIO
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# 頁面基本設定
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
        "source_code": "00981A",
    },
    "A09012": {
        "name": "統一奔騰基金",
        "type": "FUND",
        "source_code": "ACPS10",
        "moneydj_url": (
            "https://www.moneydj.com/funddj/ya/"
            "yp010000.djhtm?a=ACPS10"
        ),
    },
    "00407A": {
        "name": "00407A 凱基台灣主動式ETF",
        "type": "ETF",
        "source_code": "00407A",
    },
    "A41209": {
        "name": "凱基台灣精五門基金",
        "type": "FUND",
        "source_code": "ACKG15",
        "moneydj_url": (
            "https://www.moneydj.com/funddj/ya/"
            "yp010000.djhtm?a=ACKG15"
        ),
    },
    "009816": {
        "name": "009816 凱基台灣TOP 50 ETF",
        "type": "ETF",
        "source_code": "009816",
    },
    "0050": {
        "name": "0050 元大台灣50",
        "type": "ETF",
        "source_code": "0050",
    },
}


# ============================================================
# 檔案位置
# ============================================================

LOCAL_FUND_FILE = Path("fund_nav_history.csv")


# ============================================================
# HTTP 共用設定
# ============================================================

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}


# ============================================================
# 共用工具
# ============================================================

def safe_float(value):
    """將文字安全轉換成數字。"""

    try:
        if value is None:
            return np.nan

        text = str(value).strip()
        text = text.replace(",", "")
        text = text.replace("%", "")
        text = text.replace("元", "")

        if text in ["", "-", "--", "---", "None", "nan", "NaN"]:
            return np.nan

        return float(text)

    except Exception:
        return np.nan


def pct_change(current_value, base_value):
    """計算百分比報酬率。"""

    if pd.isna(current_value) or pd.isna(base_value):
        return np.nan

    if base_value == 0:
        return np.nan

    return (current_value / base_value - 1) * 100


def format_number(value):
    if pd.isna(value):
        return "-"

    return f"{value:,.4f}".rstrip("0").rstrip(".")


def format_pct(value):
    if pd.isna(value):
        return "-"

    return f"{value:+.2f}%"


def normalize_history(data):
    """統一歷史資料欄位與格式。"""

    if data is None or data.empty:
        return pd.DataFrame(columns=["date", "nav"])

    result = data.copy()

    if "date" not in result.columns or "nav" not in result.columns:
        return pd.DataFrame(columns=["date", "nav"])

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["nav"] = result["nav"].apply(safe_float)

    result = result.dropna(subset=["date", "nav"])

    result["date"] = result["date"].dt.normalize()

    result = result.sort_values("date")

    result = result.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return result[["date", "nav"]].reset_index(drop=True)


# ============================================================
# ETF：台灣證交所資料
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_twse_month(stock_code, year, month):
    """取得台灣證交所單月收盤價資料。"""

    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    params = {
        "response": "json",
        "date": f"{year}{month:02d}01",
        "stockNo": stock_code,
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=20,
        )

        response.raise_for_status()

        payload = response.json()

        rows = payload.get("data", [])

        records = []

        for row in rows:
            if len(row) < 7:
                continue

            try:
                roc_year, month_text, day_text = str(row[0]).split("/")

                trade_date = pd.Timestamp(
                    year=int(roc_year) + 1911,
                    month=int(month_text),
                    day=int(day_text),
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

        return normalize_history(pd.DataFrame(records))

    except Exception:
        return pd.DataFrame(columns=["date", "nav"])


def month_iterator(start_date, end_date):
    current = pd.Timestamp(
        year=start_date.year,
        month=start_date.month,
        day=1,
    )

    ending_month = pd.Timestamp(
        year=end_date.year,
        month=end_date.month,
        day=1,
    )

    while current <= ending_month:
        yield current.year, current.month
        current = current + pd.offsets.MonthBegin(1)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_history(stock_code, selected_date):
    """取得 ETF 約三年又兩個月的收盤價。"""

    end_date = pd.Timestamp(selected_date)

    start_date = end_date - pd.DateOffset(
        years=3,
        months=2,
    )

    frames = []

    for year, month in month_iterator(start_date, end_date):
        monthly_data = fetch_twse_month(
            stock_code,
            year,
            month,
        )

        if not monthly_data.empty:
            frames.append(monthly_data)

    if not frames:
        return pd.DataFrame(columns=["date", "nav"])

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    history = normalize_history(history)

    history = history[
        history["date"] <= end_date
    ]

    return history.reset_index(drop=True)


# ============================================================
# 共同基金：讀取本機 CSV
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_local_fund_history():
    """讀取 fund_nav_history.csv。"""

    empty_result = pd.DataFrame(
        columns=["date", "code", "name", "nav"]
    )

    if not LOCAL_FUND_FILE.exists():
        return empty_result

    data = None

    for encoding in ["utf-8-sig", "utf-8", "big5", "cp950"]:
        try:
            data = pd.read_csv(
                LOCAL_FUND_FILE,
                encoding=encoding,
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

    data = data.rename(columns=rename_map)

    for required_column in ["date", "code", "nav"]:
        if required_column not in data.columns:
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

    data["nav"] = data["nav"].apply(safe_float)

    data = data.dropna(subset=["date", "nav"])

    data = data.sort_values(["code", "date"])

    data = data.drop_duplicates(
        subset=["code", "date"],
        keep="last",
    )

    return data.reset_index(drop=True)


def get_local_fund_history(product_code, fund_name):
    data = load_local_fund_history()

    if data.empty:
        return pd.DataFrame(columns=["date", "nav"])

    result = data[
        data["code"] == product_code
    ].copy()

    if result.empty:
        result = data[
            data["name"].str.contains(
                fund_name,
                na=False,
                regex=False,
            )
        ].copy()

    if result.empty:
        return pd.DataFrame(columns=["date", "nav"])

    return normalize_history(
        result[["date", "nav"]]
    )


# ============================================================
# 共同基金：MoneyDJ 資料解析
# ============================================================

def find_date_nav_columns(table):
    """從 MoneyDJ 表格中找出日期欄及淨值欄。"""

    date_column = None
    nav_column = None

    for column in table.columns:
        column_text = str(column).strip()

        if date_column is None:
            if "日期" in column_text:
                date_column = column

        if nav_column is None:
            if (
                "淨值" in column_text
                and "日期" not in column_text
                and "最高" not in column_text
                and "最低" not in column_text
            ):
                nav_column = column

    return date_column, nav_column


def parse_moneydj_tables(html_text):
    """從 MoneyDJ HTML 表格解析日期與淨值。"""

    records = []

    try:
        tables = pd.read_html(StringIO(html_text))
    except Exception:
        tables = []

    for table in tables:
        if table.empty:
            continue

        table.columns = [
            " ".join(
                str(part)
                for part in column
            ).strip()
            if isinstance(column, tuple)
            else str(column).strip()
            for column in table.columns
        ]

        date_column, nav_column = find_date_nav_columns(table)

        if date_column is None or nav_column is None:
            continue

        for _, row in table.iterrows():
            raw_date = row.get(date_column)
            raw_nav = row.get(nav_column)

            parsed_date = pd.to_datetime(
                raw_date,
                errors="coerce",
            )

            parsed_nav = safe_float(raw_nav)

            if pd.notna(parsed_date) and pd.notna(parsed_nav):
                records.append(
                    {
                        "date": parsed_date,
                        "nav": parsed_nav,
                    }
                )

    return normalize_history(pd.DataFrame(records))


def parse_moneydj_text(html_text):
    """
    表格解析不到時，從頁面文字尋找：
    YYYY/MM/DD + 淨值
    或 YYYY-MM-DD + 淨值。
    """

    records = []

    cleaned_text = re.sub(
        r"<script.*?</script>",
        " ",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned_text = re.sub(
        r"<style.*?</style>",
        " ",
        cleaned_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned_text = re.sub(
        r"<[^>]+>",
        " ",
        cleaned_text,
    )

    cleaned_text = (
        cleaned_text
        .replace("&nbsp;", " ")
        .replace("&#47;", "/")
        .replace("&quot;", '"')
    )

    pattern = re.compile(
        r"("
        r"(?:19|20)\d{2}"
        r"[/-]"
        r"\d{1,2}"
        r"[/-]"
        r"\d{1,2}"
        r")"
        r"\s+"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)"
    )

    for match in pattern.finditer(cleaned_text):
        parsed_date = pd.to_datetime(
            match.group(1),
            errors="coerce",
        )

        parsed_nav = safe_float(
            match.group(2)
        )

        if pd.notna(parsed_date) and pd.notna(parsed_nav):
            records.append(
                {
                    "date": parsed_date,
                    "nav": parsed_nav,
                }
            )

    return normalize_history(pd.DataFrame(records))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_moneydj_page(url):
    """下載 MoneyDJ 頁面並解析日期及淨值。"""

    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=25,
        )

        response.raise_for_status()

        response.encoding = response.apparent_encoding or "big5"

        html_text = response.text

        table_result = parse_moneydj_tables(html_text)

        if not table_result.empty:
            return table_result, None

        text_result = parse_moneydj_text(html_text)

        if not text_result.empty:
            return text_result, None

        return (
            pd.DataFrame(columns=["date", "nav"]),
            "頁面已開啟，但找不到可辨識的日期及淨值表格。",
        )

    except Exception as error:
        return (
            pd.DataFrame(columns=["date", "nav"]),
            str(error),
        )


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_moneydj_history(moneydj_code, normal_url):
    """
    嘗試多個 MoneyDJ 頁面。

    第一個為較長期間頁面。
    第二個為基金淨值標準頁面。
    """

    candidate_urls = [
        (
            "https://tbb.moneydj.com/w/wr/"
            f"wr02_{moneydj_code}-5704_3_0_0.djhtm"
        ),
        normal_url,
    ]

    frames = []
    errors = []

    for url in candidate_urls:
        page_data, page_error = fetch_moneydj_page(url)

        if not page_data.empty:
            frames.append(page_data)

        if page_error:
            errors.append(page_error)

    if not frames:
        error_message = "；".join(errors)

        return (
            pd.DataFrame(columns=["date", "nav"]),
            error_message,
        )

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    history = normalize_history(history)

    return history, None


def merge_fund_history(local_history, online_history):
    """
    合併 CSV 與 MoneyDJ 資料。

    同一天若兩邊都有資料，MoneyDJ 線上資料優先。
    """

    frames = []

    if local_history is not None and not local_history.empty:
        local_data = local_history.copy()
        local_data["priority"] = 1
        frames.append(local_data)

    if online_history is not None and not online_history.empty:
        online_data = online_history.copy()
        online_data["priority"] = 2
        frames.append(online_data)

    if not frames:
        return pd.DataFrame(columns=["date", "nav"])

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
        merged[["date", "nav"]]
    )


# ============================================================
# 指定日期以前最近一筆資料
# ============================================================

def get_value_on_or_before(history, target_date):
    if history.empty:
        return None, np.nan

    target_date = pd.Timestamp(target_date)

    available = history[
        history["date"] <= target_date
    ].copy()

    if available.empty:
        return None, np.nan

    row = available.iloc[-1]

    return row["date"], row["nav"]


def get_previous_value(history, current_date):
    available = history[
        history["date"] < current_date
    ]

    if available.empty:
        return None, np.nan

    row = available.iloc[-1]

    return row["date"], row["nav"]


def get_period_return(
    history,
    current_date,
    current_nav,
    target_date,
    maximum_gap_days,
):
    """
    找目標日期以前最近一筆資料。

    如果實際找到的基準日期距離目標日期太遠，
    代表歷史資料不足，不計算該期間績效。
    """

    base_date, base_nav = get_value_on_or_before(
        history,
        target_date,
    )

    if base_date is None or pd.isna(base_nav):
        return np.nan

    gap_days = (
        pd.Timestamp(target_date)
        - pd.Timestamp(base_date)
    ).days

    if gap_days < 0 or gap_days > maximum_gap_days:
        return np.nan

    return pct_change(
        current_nav,
        base_nav,
    )


# ============================================================
# 績效計算
# ============================================================

def calculate_performance(history, selected_date):
    history = normalize_history(history)

    if history.empty:
        return None

    selected_timestamp = pd.Timestamp(selected_date)

    current_date, current_nav = get_value_on_or_before(
        history,
        selected_timestamp,
    )

    if current_date is None:
        return None

    previous_date, previous_nav = get_previous_value(
        history,
        current_date,
    )

    daily = pct_change(
        current_nav,
        previous_nav,
    )

    # MTD：以上月底最後可取得資料為基準
    month_start = current_date.replace(day=1)
    month_target = month_start - timedelta(days=1)

    mtd = get_period_return(
        history=history,
        current_date=current_date,
        current_nav=current_nav,
        target_date=month_target,
        maximum_gap_days=10,
    )

    # YTD：以上一年度年底最後可取得資料為基準
    year_start = pd.Timestamp(
        year=current_date.year,
        month=1,
        day=1,
    )

    year_target = year_start - timedelta(days=1)

    ytd = get_period_return(
        history=history,
        current_date=current_date,
        current_nav=current_nav,
        target_date=year_target,
        maximum_gap_days=10,
    )

    one_year = get_period_return(
        history=history,
        current_date=current_date,
        current_nav=current_nav,
        target_date=current_date - pd.DateOffset(years=1),
        maximum_gap_days=10,
    )

    two_year = get_period_return(
        history=history,
        current_date=current_date,
        current_nav=current_nav,
        target_date=current_date - pd.DateOffset(years=2),
        maximum_gap_days=10,
    )

    three_year = get_period_return(
        history=history,
        current_date=current_date,
        current_nav=current_nav,
        target_date=current_date - pd.DateOffset(years=3),
        maximum_gap_days=10,
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
# 取得所有商品資料
# ============================================================

results = []
fund_fetch_messages = []

progress_bar = st.progress(0)

product_items = list(PRODUCTS.items())

for index, (product_code, product_info) in enumerate(product_items):

    product_type = product_info["type"]

    if product_type == "ETF":
        history = fetch_etf_history(
            product_info["source_code"],
            selected_date,
        )

        data_source = "TWSE"

    else:
        local_history = get_local_fund_history(
            product_code,
            product_info["name"],
        )

        online_history, online_error = fetch_moneydj_history(
            product_info["source_code"],
            product_info["moneydj_url"],
        )

        history = merge_fund_history(
            local_history,
            online_history,
        )

        if not online_history.empty and not local_history.empty:
            data_source = "MoneyDJ + CSV"

        elif not online_history.empty:
            data_source = "MoneyDJ"

        elif not local_history.empty:
            data_source = "CSV"

        else:
            data_source = "無可用資料"

        if online_error:
            fund_fetch_messages.append(
                f"{product_info['name']}：{online_error}"
            )

    performance = calculate_performance(
        history,
        selected_date,
    )

    if performance is None:
        results.append(
            {
                "基金名稱": product_info["name"],
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
        )

    else:
        results.append(
            {
                "基金名稱": product_info["name"],
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
        )

    progress_bar.progress(
        (index + 1) / len(product_items)
    )

progress_bar.empty()

result_df = pd.DataFrame(results)


# ============================================================
# 上方資訊
# ============================================================

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        "查詢日期",
        pd.Timestamp(selected_date).strftime(
            "%Y/%m/%d"
        ),
    )

with metric_col2:
    available_date_values = result_df.loc[
        result_df["淨值日期"] != "-",
        "淨值日期",
    ]

    if available_date_values.empty:
        latest_date_text = "-"
    else:
        parsed_dates = pd.to_datetime(
            available_date_values,
            errors="coerce",
        ).dropna()

        if parsed_dates.empty:
            latest_date_text = "-"
        else:
            latest_date_text = (
                parsed_dates.max().strftime("%Y/%m/%d")
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
# 顯示績效表
# ============================================================

st.subheader("📈 績效比較")

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

        if number > 0:
            return "color: #d62728; font-weight: bold;"

        if number < 0:
            return "color: #138a36; font-weight: bold;"

    except Exception:
        return ""

    return ""


styled_df = display_df.style

for column in performance_columns:
    styled_df = styled_df.map(
        performance_color,
        subset=[column],
    )

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 共同基金抓取狀態
# ============================================================

missing_funds = result_df[
    result_df["基金名稱"].isin(
        [
            "統一奔騰基金",
            "凱基台灣精五門基金",
        ]
    )
    & (result_df["淨值日期"] == "-")
]["基金名稱"].tolist()

if missing_funds:
    st.warning(
        "⚠️ 以下共同基金目前無法取得可使用淨值："
        + "、".join(missing_funds)
        + "。請展開下方「資料抓取診斷」查看原因。"
    )

if fund_fetch_messages:
    with st.expander("🔧 資料抓取診斷"):
        for message in fund_fetch_messages:
            st.write(message)

        st.caption(
            "若 MoneyDJ 暫時拒絕 Streamlit Cloud 存取，"
            "系統仍會使用 fund_nav_history.csv 作為備援。"
        )


# ============================================================
# 共同基金查詢連結
# ============================================================

st.divider()

st.subheader("🔎 共同基金資料來源")

link_col1, link_col2 = st.columns(2)

with link_col1:
    st.link_button(
        "統一奔騰基金 MoneyDJ 淨值",
        PRODUCTS["A09012"]["moneydj_url"],
        use_container_width=True,
    )

with link_col2:
    st.link_button(
        "凱基台灣精五門基金 MoneyDJ 淨值",
        PRODUCTS["A41209"]["moneydj_url"],
        use_container_width=True,
    )


# ============================================================
# CSV 說明
# ============================================================

with st.expander("📁 fund_nav_history.csv 格式"):
    st.write(
        "CSV 與 app.py 放在 GitHub Repository 同一層。"
        "MoneyDJ 讀取失敗時，系統會自動改用 CSV。"
    )

    st.code(
        """date,code,name,nav
2026-10-27,A09012,統一奔騰基金,711.97
2026-10-28,A09012,統一奔騰基金,747.96
2026-10-27,A41209,凱基台灣精五門基金,181.37
2026-10-28,A41209,凱基台灣精五門基金,191.01""",
        language="text",
    )


# ============================================================
# 績效計算說明
# ============================================================

with st.expander("ℹ️ 績效計算方式"):
    st.markdown(
        """
- **Daily**：目前淨值 ÷ 前一筆淨值 − 1
- **MTD**：目前淨值 ÷ 上月底淨值 − 1
- **YTD**：目前淨值 ÷ 去年底淨值 − 1
- **1Y**：目前淨值 ÷ 一年前淨值 − 1
- **2Y**：目前淨值 ÷ 兩年前淨值 − 1
- **3Y**：目前淨值 ÷ 三年前淨值 − 1

如果基準日期附近沒有足夠歷史資料，系統會顯示「-」，
不會使用過短的資料錯算完整年度績效。
"""
    )


# ============================================================
# 頁尾
# ============================================================

st.divider()

st.caption(
    "ETF 資料來源為臺灣證券交易所。共同基金資料優先讀取 "
    "MoneyDJ，並以 fund_nav_history.csv 作為備援。"
)

st.caption(
    "本頁資料僅供研究與績效比較使用，"
    "共同基金正式淨值仍以基金公司公告為準。"
)
