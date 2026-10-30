from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


# ============================================================
# Repository 路徑
# ============================================================

SCRIPT_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_FILE.parent.parent
CSV_FILE = REPOSITORY_ROOT / "fund_nav_history.csv"


# ============================================================
# 基金設定
# ============================================================

FUND_CONFIG = {
    "A09012": {
        "name": "統一奔騰基金",
        "yahoo_tw_symbol": "F0HKG05X2C:FO",
        "yahoo_chart_symbols": [
            "F0HKG05X2C.TW",
            "F0HKG05X2C:FO",
        ],
        "history_url": (
            "https://tw.stock.yahoo.com/fund/"
            "F0HKG05X2C:FO/history"
        ),
    },
    "A41209": {
        "name": "凱基台灣精五門基金",
        "yahoo_tw_symbol": "F0000040SC:FO",
        "yahoo_chart_symbols": [
            "F0000040SC.TW",
            "F0000040SC:FO",
        ],
        "history_url": (
            "https://tw.stock.yahoo.com/fund/"
            "F0000040SC:FO/history"
        ),
    },
}


# ============================================================
# HTTP 設定
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/json,"
        "text/plain;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}

REQUEST_TIMEOUT = 30

OUTPUT_COLUMNS = [
    "date",
    "code",
    "name",
    "nav",
]


# ============================================================
# 共用工具
# ============================================================

def empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "nav"]
    )


def safe_float(value):
    try:
        if value is None:
            return None

        text = str(value).strip()

        text = (
            text.replace(",", "")
            .replace("元", "")
            .replace("%", "")
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


def normalize_history(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return empty_history()

    if not {"date", "nav"}.issubset(data.columns):
        return empty_history()

    result = data[
        ["date", "nav"]
    ].copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["nav"] = result["nav"].apply(
        safe_float
    )

    result = result.dropna(
        subset=["date", "nav"]
    )

    if result.empty:
        return empty_history()

    result["date"] = (
        result["date"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    result["nav"] = pd.to_numeric(
        result["nav"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["date", "nav"]
    )

    result = result[
        result["nav"] > 0
    ]

    result = result.sort_values("date")

    result = result.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return result.reset_index(drop=True)


def flatten_column_name(column) -> str:
    if isinstance(column, tuple):
        return " ".join(
            str(part)
            for part in column
            if str(part) != "nan"
        ).strip()

    return str(column).strip()


def find_column(
    columns,
    required_words,
    excluded_words=None,
):
    if excluded_words is None:
        excluded_words = []

    for column in columns:
        text = flatten_column_name(column)

        has_required_words = all(
            word in text
            for word in required_words
        )

        has_excluded_words = any(
            word in text
            for word in excluded_words
        )

        if (
            has_required_words
            and not has_excluded_words
        ):
            return column

    return None


# ============================================================
# 近期資料：Yahoo 台灣基金歷史頁面
# ============================================================

def fetch_recent_yahoo_history(
    history_url: str,
) -> pd.DataFrame:
    print(
        f"讀取近期歷史頁面：{history_url}"
    )

    response = requests.get(
        history_url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    response.encoding = (
        response.apparent_encoding
        or "utf-8"
    )

    try:
        tables = pd.read_html(
            StringIO(response.text)
        )

    except ValueError:
        tables = []

    records = []

    for table in tables:
        if table.empty:
            continue

        date_column = find_column(
            table.columns,
            required_words=["日期"],
        )

        nav_column = find_column(
            table.columns,
            required_words=["淨值"],
            excluded_words=[
                "日期",
                "最高",
                "最低",
                "漲跌",
            ],
        )

        if (
            date_column is None
            or nav_column is None
        ):
            continue

        for _, row in table.iterrows():
            parsed_date = pd.to_datetime(
                row.get(date_column),
                errors="coerce",
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

    result = normalize_history(
        pd.DataFrame(records)
    )

    if result.empty:
        raise RuntimeError(
            "Yahoo 歷史頁面未解析到日期與淨值。"
        )

    print(
        "近期資料取得成功："
        f"{len(result)} 筆，"
        f"{result['date'].min():%Y-%m-%d} 至 "
        f"{result['date'].max():%Y-%m-%d}"
    )

    return result


# ============================================================
# 多年資料：Yahoo Chart JSON
# ============================================================

def build_chart_url(symbol: str) -> str:
    encoded_symbol = quote(
        symbol,
        safe="",
    )

    return (
        "https://query1.finance.yahoo.com/"
        f"v8/finance/chart/{encoded_symbol}"
    )


def fetch_yahoo_chart_history(
    symbol: str,
) -> pd.DataFrame:
    url = build_chart_url(symbol)

    print(
        f"嘗試多年資料代碼：{symbol}"
    )

    response = requests.get(
        url,
        params={
            "period1": "0",
            "period2": "4102444800",
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    chart = payload.get(
        "chart",
        {},
    )

    if chart.get("error"):
        raise RuntimeError(
            json.dumps(
                chart["error"],
                ensure_ascii=False,
            )
        )

    chart_results = chart.get(
        "result"
    )

    if not chart_results:
        return empty_history()

    first_result = chart_results[0]

    timestamps = first_result.get(
        "timestamp",
        [],
    )

    indicators = first_result.get(
        "indicators",
        {},
    )

    quotes = indicators.get(
        "quote",
        [],
    )

    if not quotes:
        return empty_history()

    close_values = quotes[0].get(
        "close",
        [],
    )

    if (
        not timestamps
        or not close_values
    ):
        return empty_history()

    record_count = min(
        len(timestamps),
        len(close_values),
    )

    records = []

    for index in range(record_count):
        timestamp = timestamps[index]
        close_value = close_values[index]

        if (
            timestamp is None
            or close_value is None
        ):
            continue

        parsed_date = pd.to_datetime(
            timestamp,
            unit="s",
            utc=True,
            errors="coerce",
        )

        parsed_nav = safe_float(
            close_value
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
# 多年資料合理性檢查
# ============================================================

def compare_latest_values(
    recent_history: pd.DataFrame,
    long_history: pd.DataFrame,
):
    if (
        recent_history.empty
        or long_history.empty
    ):
        return False, "其中一個資料來源為空"

    recent_latest = recent_history.iloc[-1]
    long_latest = long_history.iloc[-1]

    date_gap = abs(
        (
            recent_latest["date"]
            - long_latest["date"]
        ).days
    )

    recent_nav = float(
        recent_latest["nav"]
    )

    long_nav = float(
        long_latest["nav"]
    )

    if recent_nav == 0:
        return False, "近期淨值為零"

    value_difference = abs(
        long_nav / recent_nav - 1
    )

    print(
        "資料比對："
        f"近期頁面 {recent_latest['date']:%Y-%m-%d}"
        f" = {recent_nav:.4f}；"
        f"多年來源 {long_latest['date']:%Y-%m-%d}"
        f" = {long_nav:.4f}；"
        f"差異 {value_difference:.2%}"
    )

    if date_gap > 10:
        return (
            False,
            f"最新日期相差 {date_gap} 天",
        )

    # 最新淨值差異超過 3%，視為不同資料序列
    if value_difference > 0.03:
        return (
            False,
            "最新淨值差異超過 3%",
        )

    return True, "資料比對通過"


def fetch_validated_long_history(
    chart_symbols,
    recent_history,
):
    for symbol in chart_symbols:
        try:
            candidate = (
                fetch_yahoo_chart_history(
                    symbol
                )
            )

            if candidate.empty:
                print(
                    f"{symbol} 沒有回傳資料。"
                )
                continue

            is_valid, reason = (
                compare_latest_values(
                    recent_history,
                    candidate,
                )
            )

            if is_valid:
                print(
                    f"{symbol} 多年資料驗證成功："
                    f"{len(candidate)} 筆，"
                    f"{candidate['date'].min():%Y-%m-%d} 至 "
                    f"{candidate['date'].max():%Y-%m-%d}"
                )

                return candidate

            print(
                f"{symbol} 多年資料未採用："
                f"{reason}"
            )

        except Exception as error:
            print(
                f"{symbol} 多年資料取得失敗："
                f"{type(error).__name__}: {error}"
            )

    return empty_history()


# ============================================================
# 讀取既有 CSV
# ============================================================

def load_existing_csv() -> pd.DataFrame:
    if not CSV_FILE.exists():
        print(
            "尚無 fund_nav_history.csv，"
            "將建立新檔案。"
        )

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
                    "name": str,
                },
            )

            break

        except Exception:
            continue

    if data is None:
        raise RuntimeError(
            "無法讀取 fund_nav_history.csv。"
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

    print(
        f"既有 CSV：{len(data)} 筆"
    )

    return data.reset_index(drop=True)


# ============================================================
# 組成單一基金輸出
# ============================================================

def prepare_fund_rows(
    fund_code,
    fund_name,
    history,
):
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


# ============================================================
# 合併與輸出 CSV
# ============================================================

def merge_all_data(
    existing_data,
    new_frames,
):
    frames = []

    if (
        existing_data is not None
        and not existing_data.empty
    ):
        frames.append(existing_data)

    for frame in new_frames:
        if (
            frame is not None
            and not frame.empty
        ):
            frames.append(frame)

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
        combined["nav"] > 0
    ]

    combined["date"] = (
        combined["date"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # 新抓取資料放在 existing_data 後方，
    # 同基金、同日期保留最後一筆，也就是新資料。
    combined = combined.drop_duplicates(
        subset=[
            "code",
            "date",
        ],
        keep="last",
    )

    combined = combined.sort_values(
        by=[
            "code",
            "date",
        ],
        ascending=[
            True,
            True,
        ],
    )

    return combined[
        OUTPUT_COLUMNS
    ].reset_index(drop=True)


def save_csv(data):
    output = data.copy()

    output["date"] = (
        pd.to_datetime(
            output["date"],
            errors="coerce",
        )
        .dt.strftime("%Y-%m-%d")
    )

    output["nav"] = (
        pd.to_numeric(
            output["nav"],
            errors="coerce",
        )
        .round(6)
    )

    output.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"已寫入：{CSV_FILE}"
    )

    print(
        f"總資料筆數：{len(output)}"
    )


# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 60)
    print("開始更新共同基金歷史淨值")
    print(f"CSV 路徑：{CSV_FILE}")
    print("=" * 60)

    existing_data = load_existing_csv()

    new_frames = []

    successful_funds = 0
    failed_funds = []

    for fund_code, config in FUND_CONFIG.items():
        fund_name = config["name"]

        print("")
        print("-" * 60)
        print(
            f"處理基金：{fund_code} {fund_name}"
        )
        print("-" * 60)

        try:
            recent_history = (
                fetch_recent_yahoo_history(
                    config["history_url"]
                )
            )

            long_history = (
                fetch_validated_long_history(
                    config[
                        "yahoo_chart_symbols"
                    ],
                    recent_history,
                )
            )

            if not long_history.empty:
                selected_history = (
                    pd.concat(
                        [
                            long_history,
                            recent_history,
                        ],
                        ignore_index=True,
                    )
                )

                selected_history = (
                    normalize_history(
                        selected_history
                    )
                )

                print(
                    "採用多年資料並合併近期資料。"
                )

            else:
                selected_history = (
                    recent_history
                )

                print(
                    "多年來源未通過驗證，"
                    "本次採用近期公開淨值。"
                )

            fund_rows = prepare_fund_rows(
                fund_code=fund_code,
                fund_name=fund_name,
                history=selected_history,
            )

            new_frames.append(
                fund_rows
            )

            successful_funds += 1

            print(
                f"{fund_name} 本次取得 "
                f"{len(fund_rows)} 筆。"
            )

        except Exception as error:
            error_message = (
                f"{fund_name}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            failed_funds.append(
                error_message
            )

            print(
                f"更新失敗：{error_message}"
            )

    if successful_funds == 0:
        print("")
        print("兩檔基金皆更新失敗。")
        print("保留既有 CSV，不覆寫檔案。")

        for failure in failed_funds:
            print(f"- {failure}")

        sys.exit(1)

    final_data = merge_all_data(
        existing_data=existing_data,
        new_frames=new_frames,
    )

    if final_data.empty:
        print(
            "合併後沒有可寫入的有效資料。"
        )

        sys.exit(1)

    save_csv(final_data)

    print("")
    print("=" * 60)
    print(
        f"更新完成：成功 {successful_funds} 檔，"
        f"失敗 {len(failed_funds)} 檔。"
    )

    if failed_funds:
        print("失敗項目：")

        for failure in failed_funds:
            print(f"- {failure}")

    print("=" * 60)


if __name__ == "__main__":
    main()
