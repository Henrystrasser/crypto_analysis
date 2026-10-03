#!/usr/bin/env python3
"""
Gemeinsame Helfer für Event-Hold-Sims (Berlin-Zeit, Coin-Listen, CLI-Zeitraum).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import FrozenSet, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BERLIN = ZoneInfo("Europe/Berlin")


def ms_to_berlin_str(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=BERLIN)
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def parse_berlin_bound(raw: str, *, end_of_day: bool) -> datetime:
    s = raw.strip()
    if " " in s:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    return dt.replace(tzinfo=BERLIN)


def resolve_event_window(
    from_s: Optional[str],
    to_s: Optional[str],
    lookback_days: Optional[int],
    default_lookback: int,
) -> tuple[datetime, datetime, str]:
    now = datetime.now(BERLIN)
    if from_s:
        start = parse_berlin_bound(from_s, end_of_day=False)
        end = parse_berlin_bound(to_s, end_of_day=True) if to_s else now
        label = f"FROM/TO ({from_s} → {to_s or 'jetzt'})"
        return start, end, label
    days = default_lookback if lookback_days is None else int(lookback_days)
    start = now - timedelta(days=days)
    return start, now, f"LOOKBACK {days}d"


def add_time_range_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--from", dest="from_date", default=None, help="Start Europe/Berlin")
    p.add_argument("--to", dest="to_date", default=None, help="Ende Europe/Berlin")
    p.add_argument("--lookback", type=int, default=None, help="Tage rückwärts, falls kein --from")


def fetch_top_coins_robust(
    n: int,
    exclude_base: str = "BTC",
    stable_symbols: Optional[Set[str]] = None,
    exclude_symbols: Optional[Set[str]] = None,
    session: Optional[requests.Session] = None,
    coingecko: str = "https://api.coingecko.com/api/v3",
) -> List[dict]:
    """Minimaler Fallback: leere Liste — Ein-Coin-Script braucht das Universum nicht."""
    _ = (n, exclude_base, stable_symbols, exclude_symbols, session, coingecko)
    return []
