#!/usr/bin/env python3
"""
Hilfsskript: US Spot BTC ETF Tages-Nettoflows holen oder Template schreiben.

Ziel-CSV (Projektroot): etf_btc_spot_flows.csv
  Spalten: date (YYYY-MM-DD), net_flow_usd (USD, nicht Mio.)

Quellen:
  1) Farside https://farside.co.uk/bitcoin-etf-flow-all-data/ (Total in Mio. → *1e6)
  2) sonst Template + Hinweis (manuell Farside/SoSoValue eintragen)

Nutzung:
  cd /workspace/crypto-laggard
  python scripts/fetch_etf_flows_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etf_flow_hold_sim import (  # noqa: E402
    CSV_FLOWS,
    fetch_farside_flows,
    load_flows_csv,
    write_template_csv,
)


def main() -> None:
    out = ROOT / CSV_FLOWS
    existing = load_flows_csv(str(out)) if out.exists() else None
    if existing is not None and len(existing) > 0:
        print(f"Bereits vorhanden: {out} ({len(existing)} Zeilen)")
        print(existing.tail(3).to_string(index=False))
        return
    print("Fetch Farside …")
    flows = fetch_farside_flows()
    if flows is None:
        write_template_csv(str(out))
        print(
            "Fetch fehlgeschlagen. Template geschrieben — bitte echte Daten "
            "von Farside oder SoSoValue eintragen."
        )
        return
    print(f"OK: {out} ({len(flows)} Tage)")
    print(flows.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
