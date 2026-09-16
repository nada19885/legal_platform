from __future__ import annotations

import threading
from typing import Iterable

import dataiku
import pandas as pd


_WRITE_LOCK = threading.RLock()


def _clean_frame(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in result.columns:
        if result[column].dtype == "object":
            result[column] = (
                result[column]
                .fillna("")
                .astype(str)
            )

    return result


def read_dataset(
    dataset_name: str,
) -> pd.DataFrame:
    try:
        return dataiku.Dataset(
            dataset_name,
            ignore_flow=True,
        ).get_dataframe()

    except Exception:
        return pd.DataFrame()


def append_rows(
    dataset_name: str,
    rows: Iterable[dict],
) -> int:
    rows = list(rows)

    if not rows:
        return 0

    dataframe = _clean_frame(
        pd.DataFrame(rows)
    )

    with _WRITE_LOCK:
        dataset = dataiku.Dataset(
            dataset_name,
            ignore_flow=True,
        )

        dataset.spec_item[
            "appendMode"
        ] = True

        dataset.write_with_schema(
            dataframe
        )

    return len(dataframe)


def case_rows(
    dataset_name: str,
    case_id: str,
) -> pd.DataFrame:
    dataframe = read_dataset(
        dataset_name
    )

    if (
        dataframe.empty
        or "case_id" not in dataframe.columns
    ):
        return pd.DataFrame()

    return dataframe[
        dataframe["case_id"]
        .astype(str)
        == str(case_id)
    ].copy()


def latest_case(
    case_id: str,
):
    dataframe = case_rows(
        "cases",
        case_id,
    )

    if dataframe.empty:
        return None

    return dataframe.iloc[-1].to_dict()


def list_cases() -> pd.DataFrame:
    dataframe = read_dataset(
        "cases"
    )

    if dataframe.empty:
        return dataframe

    return dataframe.drop_duplicates(
        "case_id",
        keep="last",
    )




