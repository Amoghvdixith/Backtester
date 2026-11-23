"""Market downloader cleaning logic with a FAKE yfinance module (no network)."""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from backtester.market import download


def _fake_yf(frames, fail=False):
    mod = types.ModuleType("yfinance")
    mod.__version__ = "fake"

    def dl(symbols, **kw):
        if fail:
            return pd.DataFrame()
        return pd.concat({s: frames[s] for s in symbols}, axis=1)

    mod.download = dl
    return mod


def _frame(n=5, bad_row=None, tiny=None):
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n), name="Date")
    df = pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5,
                       "Volume": 1000.0}, index=idx)
    if bad_row is not None:
        df.iloc[bad_row, df.columns.get_loc("High")] = 5.0  # big violation -> dropped
    if tiny is not None:
        df.iloc[tiny, df.columns.get_loc("High")] = 10.49  # rounding-size -> repaired
    df.iloc[2, df.columns.get_loc("Close")] = np.nan  # incomplete -> missing bar
    return df


def test_download_cleans_and_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance",
                        _fake_yf({"AAA": _frame(bad_row=1, tiny=3), "BBB": _frame()}))
    meta = download(["AAA", "BBB"], "2020-01-01", "2020-02-01", tmp_path, "t")
    rep = meta["cleaning_report"]
    assert rep == {"repaired_rows": 1, "dropped_rows": 1, "missing_rows": 2}
    df = pd.read_csv(tmp_path / "t.csv")
    assert len(df) == 5 + 5 - 2 - 1
    assert json.loads((tmp_path / "t.meta.json").read_text())["data_mode"] == "market"


def test_download_failure_never_substitutes(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf({}, fail=True))
    with pytest.raises(RuntimeError, match="no data"):
        download(["AAA"], "2020-01-01", "2020-02-01", tmp_path, "t")
    assert not (tmp_path / "t.csv").exists()
