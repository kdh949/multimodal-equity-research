from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_research.config import SecSettings


@dataclass
class SecEdgarClient:
    settings: SecSettings = field(default_factory=SecSettings)
    cache_dir: Path = Path("data/raw/sec")
    timeout_seconds: float = 15.0
    _last_request_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }
        )

    def get_submissions(self, cik: str) -> dict[str, Any]:
        cik10 = _normalize_cik(cik)
        return self._get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json", f"submissions_{cik10}.json")

    def get_companyfacts(self, cik: str) -> dict[str, Any]:
        cik10 = _normalize_cik(cik)
        return self._get_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json",
            f"companyfacts_{cik10}.json",
        )

    def get_companyconcept(self, cik: str, taxonomy: str, tag: str) -> dict[str, Any]:
        cik10 = _normalize_cik(cik)
        return self._get_json(
            f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik10}/{taxonomy}/{tag}.json",
            f"companyconcept_{cik10}_{taxonomy}_{tag}.json",
        )

    def get_frame(self, taxonomy: str, tag: str, unit: str, period: str) -> dict[str, Any]:
        return self._get_json(
            f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json",
            f"frame_{taxonomy}_{tag}_{unit}_{period}.json",
        )

    def recent_filings(self, cik: str, forms: set[str] | None = None) -> pd.DataFrame:
        submissions = self.get_submissions(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        frame = pd.DataFrame(recent)
        if frame.empty:
            return _empty_filings()
        keep = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"]
        for column in keep:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[keep].rename(
            columns={
                "accessionNumber": "accession_number",
                "filingDate": "filing_date",
                "reportDate": "report_date",
                "primaryDocument": "primary_document",
            }
        )
        frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
        frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
        if forms:
            frame = frame[frame["form"].isin(forms)]
        return frame.reset_index(drop=True)

    def _get_json(self, url: str, cache_name: str) -> dict[str, Any]:
        cache_path = self.cache_dir / cache_name
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        self._throttle()
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload))
        return payload

    def _throttle(self) -> None:
        min_interval = 1.0 / self.settings.max_requests_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()


@dataclass
class SyntheticSecProvider:
    def recent_filings(self, cik: str, forms: set[str] | None = None) -> pd.DataFrame:
        del cik
        forms = forms or {"8-K", "10-Q", "10-K", "4"}
        base = pd.Timestamp.today().normalize() - pd.offsets.BDay(180)
        rows = [
            {"filing_date": base + pd.offsets.BDay(20), "report_date": base, "form": "10-Q", "accession_number": "0001", "primary_document": "10q.htm"},
            {"filing_date": base + pd.offsets.BDay(55), "report_date": base, "form": "8-K", "accession_number": "0002", "primary_document": "8k.htm"},
            {"filing_date": base + pd.offsets.BDay(90), "report_date": base, "form": "4", "accession_number": "0003", "primary_document": "form4.xml"},
            {"filing_date": base + pd.offsets.BDay(130), "report_date": base, "form": "10-K", "accession_number": "0004", "primary_document": "10k.htm"},
        ]
        frame = pd.DataFrame(rows)
        return frame[frame["form"].isin(forms)].reset_index(drop=True)

    def companyfacts_frame(self, cik: str) -> pd.DataFrame:
        del cik
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=8, freq="BQE")
        revenue = pd.Series(range(len(dates)), dtype=float) * 100_000_000 + 1_000_000_000
        net_income = revenue * 0.12
        assets = revenue * 4.0
        return pd.DataFrame(
            {
                "period_end": dates,
                "revenue": revenue,
                "net_income": net_income,
                "assets": assets,
            }
        )


def extract_companyfacts_frame(companyfacts: dict[str, Any]) -> pd.DataFrame:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    concept_map = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
        "Revenues": "revenue",
        "NetIncomeLoss": "net_income",
        "Assets": "assets",
    }
    frames: list[pd.DataFrame] = []
    for sec_tag, column in concept_map.items():
        concept = facts.get(sec_tag, {})
        units = concept.get("units", {})
        unit_values = units.get("USD") or units.get("shares") or []
        if not unit_values:
            continue
        part = pd.DataFrame(unit_values)
        if part.empty or "end" not in part or "val" not in part:
            continue
        part = part[["end", "val", "fy", "fp", "form", "filed"]].copy()
        part["period_end"] = pd.to_datetime(part["end"], errors="coerce")
        part[column] = pd.to_numeric(part["val"], errors="coerce")
        frames.append(part[["period_end", column]])
    if not frames:
        return pd.DataFrame(columns=["period_end", "revenue", "net_income", "assets"])
    merged = frames[0]
    for part in frames[1:]:
        merged = merged.merge(part, on="period_end", how="outer")
    return merged.sort_values("period_end").drop_duplicates("period_end", keep="last").reset_index(drop=True)


def _normalize_cik(cik: str) -> str:
    digits = "".join(char for char in str(cik) if char.isdigit())
    return digits.zfill(10)


def _empty_filings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["accession_number", "filing_date", "report_date", "form", "primary_document"]
    )
