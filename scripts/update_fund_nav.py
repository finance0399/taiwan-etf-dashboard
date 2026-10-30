from __future__ import annotations

import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# Repository 路徑與基本設定
# ============================================================

SCRIPT_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_FILE.parent.parent
CSV_FILE = REPOSITORY_ROOT / "fund_nav_history.csv"

OUTPUT_COLUMNS = [
    "date",
    "code",
    "name",
    "nav",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 60

START_DATE = pd.Timestamp(
    "2023-01-01"
)

END_DATE = (
    pd.Timestamp
    .today()
    .normalize()
)


# ============================================================
# 基金設定
# ============================================================

FUND_CONFIG = {
    "A09012": {
        "name": "統一奔騰基金",
        "expected_words": [
            "統一奔騰基金",
        ],
        "history_urls": [
            (
                "https://tbb.moneydj.com/w/wr/"
                "wr02_ACPS10-5704_3_0_0.djhtm"
            ),
            (
                "https://fund.hncb.com.tw/w/wr/"
                "wr02_ACPS10-009007_3_0_0.djhtm"
            ),
        ],
        "recent_urls": [
            (
                "https://kgibank.moneydj.com/w/wr/"
                "wr02_ACPS10-PR01.djhtm"
            ),
            (
                "https://fund.hncb.com.tw/w/wr/"
                "wr02_ACPS10-009007.djhtm"
            ),
        ],
    },
    "A41209": {
        "name": "凱基台灣精五門基金",
        "expected_words": [
            "凱基台灣精五門基金",
        ],
        "history_urls": [
            (
                "https://tbb.moneydj.com/w/wr/"
                "wr02_ACKG15-5704_3_0_0.djhtm"
            ),
            (
                "https://fund.taipeifubon.com.tw/w/wr/"
                "wr02_ACKG15-3908_3_0_0.djhtm"
            ),
            (
                "https://fund.hncb.com.tw/w/wr/"
                "wr02_ACKG15-014013_3_0_0.djhtm"
            ),
        ],
        "recent_urls": [
            (
                "https://fund.hncb.com.tw/w/wr/"
                "wr02_ACKG15-014013.djhtm"
            ),
            (
                "https://fund.taipeifubon.com.tw/w/wr/"
                "wr02_ACKG15-3908.djhtm"
            ),
        ],
    },
}


# ============================================================
# 共用工具
# ============================================================

def empty_history():
    """建立空白歷史資料表。"""

    return pd.DataFrame(
        columns=[
            "date",
            "nav",
        ]
    )


def safe_float(value):
    """安全地將資料轉換成正數。"""

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
            return None

        number = float(text)

        if number <= 0:
            return None

        return number

    except Exception:
        return None


def parse_date(value):
    """解析西元或民國日期。"""

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


def normalize_history(data):
    """統一歷史資料格式並限制回補期間。"""

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

    except TypeError:
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

    if result.empty:
        return empty_history()

    result["date"] = (
        result["date"]
        .dt.normalize()
    )

    result = result[
        (
            result["date"]
            >= START_DATE
        )
        & (
            result["date"]
            <= END_DATE
        )
        & (
            result["nav"]
            > 0
        )
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


def flatten_column(column):
    """處理多層表格欄位名稱。"""

    if isinstance(column, tuple):

        return " ".join(
            str(part)
            for part in column
            if str(part) != "nan"
        ).strip()

    return str(column).strip()


def find_date_nav_columns(table):
    """尋找日期欄位與淨值欄位。"""

    date_column = None
    nav_column = None

    for column in table.columns:

        column_text = (
            flatten_column(
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


# ============================================================
# 網頁表格解析
# ============================================================

def parse_html_tables(html_text):
    """從 HTML 表格解析日期及淨值。"""

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

            parsed_date = parse_date(
                row.get(date_column)
            )

            parsed_nav = safe_float(
                row.get(nav_column)
            )

            if (
                pd.notna(parsed_date)
                and parsed_nav is not None
            ):
                records.append(
                    {
                        "date": parsed_date,
                        "nav": parsed_nav,
                    }
                )

    return normalize_history(
        pd.DataFrame(records)
    )


# ============================================================
# 網頁下載與驗證
# ============================================================

def download_page(url):
    """下載網頁，失敗時最多重試三次。"""

    last_error = None

    for attempt in range(3):

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            response.encoding = (
                response.apparent_encoding
                or response.encoding
            )

            return (
                response.text,
                None,
            )

        except requests.RequestException as error:

            last_error = error

            if attempt < 2:
                time.sleep(
                    2 * (attempt + 1)
                )

    return (
        None,
        (
            f"{type(last_error).__name__}: "
            f"{last_error}"
        ),
    )


def fetch_valid_source(
    url,
    expected_words,
):
    """
    下載來源並檢查基金名稱。

    避免因基金代碼相似而抓到錯誤基金。
    """

    html_text, error = download_page(
        url
    )

    if html_text is None:

        return (
            empty_history(),
            f"連線失敗：{error}",
        )

    if not all(
        word in html_text
        for word in expected_words
    ):

        return (
            empty_history(),
            (
                "頁面基金名稱不符，"
                "拒絕採用"
            ),
        )

    history = parse_html_tables(
        html_text
    )

    if history.empty:

        return (
            empty_history(),
            (
                "頁面沒有解析到"
                "日期與淨值"
            ),
        )

    return history, None


# ============================================================
# 取得單一基金歷史資料
# ============================================================

def fetch_fund_history(config):
    """
    先嘗試多年來源，再以近期來源補最新淨值。
    """

    frames = []
    messages = []

    long_history_succeeded = False

    for url in config[
        "history_urls"
    ]:

        history, error = (
            fetch_valid_source(
                url,
                config[
                    "expected_words"
                ],
            )
        )

        if not history.empty:

            frames.append(
                history
            )

            long_history_succeeded = (
                len(history) >= 200
            )

            messages.append(
                "多年來源成功："
                f"{url}，"
                f"{len(history)} 筆"
            )

            if long_history_succeeded:
                break

        else:

            messages.append(
                "多年來源失敗："
                f"{url}，"
                f"{error}"
            )

    for url in config[
        "recent_urls"
    ]:

        history, error = (
            fetch_valid_source(
                url,
                config[
                    "expected_words"
                ],
            )
        )

        if not history.empty:

            frames.append(
                history
            )

            messages.append(
                "近期來源成功："
                f"{url}，"
                f"{len(history)} 筆"
            )

            break

        messages.append(
            "近期來源失敗："
            f"{url}，"
            f"{error}"
        )

    if not frames:

        return (
            empty_history(),
            messages,
            False,
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
        messages,
        long_history_succeeded,
    )


# ============================================================
# 讀取既有 CSV
# ============================================================

def load_existing_csv():
    """讀取既有 fund_nav_history.csv。"""

    if not CSV_FILE.exists():

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
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
                CSV_FILE,
                encoding=encoding,
                dtype={
                    "code": str,
                },
            )

            break

        except Exception:
            continue

    if data is None:
        raise RuntimeError(
            "無法讀取 "
            "fund_nav_history.csv"
        )

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

    for column in OUTPUT_COLUMNS:

        if column not in data.columns:
            data[column] = None

    data = data[
        OUTPUT_COLUMNS
    ].copy()

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
        data["nav"],
        errors="coerce",
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

    return data.reset_index(
        drop=True
    )


# ============================================================
# 合併與輸出
# ============================================================

def prepare_fund_rows(
    fund_code,
    fund_name,
    history,
):
    """加入基金代碼及名稱欄位。"""

    if history.empty:

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    result = history.copy()

    result["code"] = fund_code
    result["name"] = fund_name

    return result[
        OUTPUT_COLUMNS
    ]


def merge_all_data(
    existing_data,
    new_frames,
):
    """合併既有資料與新下載資料。"""

    frames = []

    if not existing_data.empty:
        frames.append(
            existing_data
        )

    for frame in new_frames:

        if not frame.empty:
            frames.append(
                frame
            )

    if not frames:

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["date"] = pd.to_datetime(
        combined["date"],
        errors="coerce",
    )

    try:
        combined["date"] = (
            combined["date"]
            .dt.tz_localize(None)
        )

    except TypeError:
        pass

    combined["date"] = (
        combined["date"]
        .dt.normalize()
    )

    combined["code"] = (
        combined["code"]
        .astype(str)
        .str.strip()
    )

    combined["name"] = (
        combined["name"]
        .astype(str)
        .str.strip()
    )

    combined["nav"] = pd.to_numeric(
        combined["nav"],
        errors="coerce",
    )

    combined = combined.dropna(
        subset=[
            "date",
            "code",
            "nav",
        ]
    )

    combined = combined[
        (
            combined["date"]
            >= START_DATE
        )
        & (
            combined["date"]
            <= END_DATE
        )
        & (
            combined["nav"]
            > 0
        )
    ]

    combined = combined.drop_duplicates(
        subset=[
            "code",
            "date",
        ],
        keep="last",
    )

    combined = combined.sort_values(
        [
            "code",
            "date",
        ]
    )

    return combined[
        OUTPUT_COLUMNS
    ].reset_index(drop=True)


def validate_fund_rows(
    data,
    fund_code,
):
    """檢查資料覆蓋範圍及異常跳動。"""

    fund_data = data[
        data["code"] == fund_code
    ].copy()

    fund_data = fund_data.sort_values(
        "date"
    )

    if fund_data.empty:

        return [
            "沒有任何資料"
        ]

    issues = []

    earliest_date = (
        fund_data["date"].min()
    )

    if earliest_date > pd.Timestamp(
        "2023-01-31"
    ):

        issues.append(
            "最早日期只有 "
            f"{earliest_date:%Y-%m-%d}，"
            "尚未完整回補至2023年"
        )

    daily_change = (
        fund_data["nav"]
        .pct_change()
        .abs()
    )

    abnormal_rows = fund_data[
        daily_change > 0.30
    ]

    if not abnormal_rows.empty:

        issues.append(
            "發現 "
            f"{len(abnormal_rows)} 筆"
            "單日變動超過30%，"
            "請檢查基金級別或分割事件"
        )

    return issues


def save_csv(data):
    """儲存CSV。"""

    output = data.copy()

    output["date"] = (
        output["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output["nav"] = (
        output["nav"]
        .round(6)
    )

    output.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 主程式
# ============================================================

def main():

    print(
        f"CSV 路徑：{CSV_FILE}"
    )

    print(
        "回補期間："
        f"{START_DATE:%Y-%m-%d} "
        "至 "
        f"{END_DATE:%Y-%m-%d}"
    )

    existing_data = (
        load_existing_csv()
    )

    new_frames = []

    successful_funds = 0

    for (
        fund_code,
        config,
    ) in FUND_CONFIG.items():

        print("=" * 70)

        print(
            f"處理 {fund_code} "
            f"{config['name']}"
        )

        (
            history,
            messages,
            long_history_succeeded,
        ) = fetch_fund_history(
            config
        )

        for message in messages:
            print(message)

        if history.empty:

            print(
                "此基金沒有取得"
                "任何可用資料"
            )

            continue

        print(
            f"合併後取得 "
            f"{len(history)} 筆，"
            f"期間 "
            f"{history['date'].min():%Y-%m-%d} "
            "至 "
            f"{history['date'].max():%Y-%m-%d}"
        )

        if not long_history_succeeded:

            print(
                "警告：多年來源未取得"
                "至少200筆，本次只能補入"
                "目前可取得的近期資料"
            )

        fund_rows = prepare_fund_rows(
            fund_code=fund_code,
            fund_name=config["name"],
            history=history,
        )

        new_frames.append(
            fund_rows
        )

        successful_funds += 1

if successful_funds == 0:

    print(
        "警告：兩檔基金的線上來源本次皆無法更新，"
        "保留原有 fund_nav_history.csv。"
    )

    if existing_data.empty:
        print(
            "目前 fund_nav_history.csv 也是空的，"
            "因此 Dashboard 暫時沒有共同基金歷史資料。"
        )

    return

    final_data = merge_all_data(
        existing_data=existing_data,
        new_frames=new_frames,
    )

    if final_data.empty:

        print(
            "合併後沒有可寫入資料"
        )

        sys.exit(1)

    save_csv(
        final_data
    )

    print(
        f"已寫入 "
        f"{len(final_data)} 筆至 "
        f"{CSV_FILE}"
    )

    for (
        fund_code,
        config,
    ) in FUND_CONFIG.items():

        fund_rows = final_data[
            final_data["code"]
            == fund_code
        ]

        print("-" * 70)

        print(
            f"{config['name']}："
            f"{len(fund_rows)} 筆"
        )

        if not fund_rows.empty:

            print(
                "期間："
                f"{fund_rows['date'].min():%Y-%m-%d} "
                "至 "
                f"{fund_rows['date'].max():%Y-%m-%d}"
            )

        issues = validate_fund_rows(
            final_data,
            fund_code,
        )

        for issue in issues:

            print(
                f"警告：{issue}"
            )


if __name__ == "__main__":
    main()
