from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_research.config import SecSettings

_SEC_DOCUMENT_CACHE_VERSION = 1


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

    def recent_filings(
        self,
        cik: str,
        forms: set[str] | None = None,
        *,
        include_document_text: bool = False,
    ) -> pd.DataFrame:
        submissions = self.get_submissions(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        frame = pd.DataFrame(recent)
        if frame.empty:
            return _empty_filings()
        keep = [
            "accessionNumber",
            "filingDate",
            "reportDate",
            "form",
            "primaryDocument",
        ]
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
        frame["cik"] = _normalize_cik(cik)
        frame["accession_number_no_dash"] = frame["accession_number"].apply(_normalize_accession_number)
        frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
        frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
        if forms:
            frame = frame[frame["form"].isin(forms)]
        if frame.empty:
            return frame.reset_index(drop=True)
        if include_document_text:
            frame["document_text"] = [
                self.fetch_filing_document(frame["cik"].iloc[0], accession_number, primary_document)
                for accession_number, primary_document in zip(
                    frame["accession_number"],
                    frame["primary_document"],
                    strict=False,
                )
            ]
        return frame.reset_index(drop=True)

    def fetch_filing_document(self, cik: str, accession_number: str, primary_document: str) -> str:
        cik10 = _normalize_cik(cik)
        accession_no_dash = _normalize_accession_number(accession_number)
        if not accession_no_dash or not primary_document:
            return ""
        document_name = primary_document.strip()
        if not document_name:
            return ""

        cache_path = self._filing_document_cache_path(cik10, accession_no_dash, document_name)
        if cache_path.exists():
            cached = _load_cached_filing_document(cache_path)
            if cached is not None:
                return cached

        url = _build_filing_document_url(cik10, accession_no_dash, document_name)
        try:
            self._throttle()
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            text = _extract_readable_sec_text(
                response.content,
                primary_document=document_name,
                headers=getattr(response, "headers", {}),
            )
            _write_cached_filing_document(cache_path, text)
            return text
        except Exception:
            return ""

    def get_filing_document(self, cik: str, accession_number: str, primary_document: str) -> str:
        return self.fetch_filing_document(cik, accession_number, primary_document)

    def _filing_document_cache_path(self, cik10: str, accession_no_dash: str, primary_document: str) -> Path:
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_" for character in primary_document
        )
        if not safe_name:
            safe_name = "document"
        return self.cache_dir / f"filing_{cik10}_{accession_no_dash}_{safe_name}.txt"

    def _get_json(self, url: str, cache_name: str) -> dict[str, Any]:
        cache_path = self.cache_dir / cache_name
        if cache_path.exists():
            cached = _load_json_cache(cache_path)
            if cached is not None:
                return cached
        self._throttle()
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        _write_json_cache(cache_path, payload)
        return payload

    def _throttle(self) -> None:
        min_interval = 1.0 / self.settings.max_requests_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()


@dataclass
class SyntheticSecProvider:
    def recent_filings(
        self,
        cik: str,
        forms: set[str] | None = None,
        *,
        include_document_text: bool = False,
    ) -> pd.DataFrame:
        forms = forms or {"8-K", "10-Q", "10-K", "4"}
        base = pd.Timestamp.today().normalize() - pd.offsets.BDay(180)
        cik10 = _normalize_cik(cik)
        rows = [
            {
                "cik": cik10,
                "accession_number": "0001",
                "accession_number_no_dash": "0001",
                "filing_date": base + pd.offsets.BDay(20),
                "report_date": base,
                "form": "10-Q",
                "primary_document": "10q.htm",
                "document_text": "Synthetic 10-Q filing body",
            },
            {
                "cik": cik10,
                "accession_number": "0002",
                "accession_number_no_dash": "0002",
                "filing_date": base + pd.offsets.BDay(55),
                "report_date": base,
                "form": "8-K",
                "primary_document": "8k.htm",
                "document_text": "Synthetic 8-K filing body",
            },
            {
                "cik": cik10,
                "accession_number": "0003",
                "accession_number_no_dash": "0003",
                "filing_date": base + pd.offsets.BDay(90),
                "report_date": base,
                "form": "4",
                "primary_document": "form4.xml",
                "document_text": "Synthetic form 4 filing body",
            },
            {
                "cik": cik10,
                "accession_number": "0004",
                "accession_number_no_dash": "0004",
                "filing_date": base + pd.offsets.BDay(130),
                "report_date": base,
                "form": "10-K",
                "primary_document": "10k.htm",
                "document_text": "Synthetic 10-K filing body",
            },
        ]
        frame = pd.DataFrame(rows)
        if not include_document_text:
            frame = frame.drop(columns=["document_text"])
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


def extract_companyconcept_frame(companyconcept: dict[str, Any], column: str) -> pd.DataFrame:
    units = companyconcept.get("units", {})
    unit_values = units.get("USD") or units.get("shares") or []
    if not unit_values:
        return pd.DataFrame(columns=["period_end", column])
    frame = pd.DataFrame(unit_values)
    if frame.empty or "end" not in frame or "val" not in frame:
        return pd.DataFrame(columns=["period_end", column])
    frame["period_end"] = pd.to_datetime(frame["end"], errors="coerce")
    frame[column] = pd.to_numeric(frame["val"], errors="coerce")
    return frame[["period_end", column]].dropna(subset=["period_end"]).sort_values("period_end")


def extract_frame_values(frame_payload: dict[str, Any], column: str) -> pd.DataFrame:
    rows = frame_payload.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["cik", column])
    frame = pd.DataFrame(rows)
    if frame.empty or "cik" not in frame or "val" not in frame:
        return pd.DataFrame(columns=["cik", column])
    frame["cik"] = frame["cik"].astype(str).map(_normalize_cik)
    frame[column] = pd.to_numeric(frame["val"], errors="coerce")
    return frame[["cik", column]].dropna(subset=[column]).drop_duplicates("cik", keep="last")


def merge_fact_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty and "period_end" in frame]
    if not usable:
        return pd.DataFrame(columns=["period_end", "revenue", "net_income", "assets"])
    merged = usable[0]
    for frame in usable[1:]:
        merged = merged.merge(frame, on="period_end", how="outer", suffixes=("", "_concept"))
        for column in list(merged.columns):
            if column.endswith("_concept"):
                base_column = column.removesuffix("_concept")
                if base_column in merged:
                    merged[base_column] = merged[base_column].fillna(merged[column])
                    merged = merged.drop(columns=[column])
                else:
                    merged = merged.rename(columns={column: base_column})
    return merged.sort_values("period_end").drop_duplicates("period_end", keep="last").reset_index(drop=True)


def _normalize_cik(cik: str) -> str:
    digits = "".join(char for char in str(cik) if char.isdigit())
    return digits.zfill(10)


def _empty_filings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "accession_number",
            "cik",
            "accession_number_no_dash",
            "filing_date",
            "report_date",
            "form",
            "primary_document",
        ]
    )


def _build_filing_document_url(cik10: str, accession_no_dash: str, primary_document: str) -> str:
    archive_cik = _sec_archive_cik_path(cik10)
    return f"https://www.sec.gov/Archives/edgar/data/{archive_cik}/{accession_no_dash}/{primary_document}"


def _sec_archive_cik_path(cik10: str) -> str:
    digits = "".join(char for char in str(cik10) if char.isdigit())
    stripped = digits.lstrip("0")
    return stripped or "0"


def _extract_readable_sec_text(content: bytes, primary_document: str, headers: dict[str, str] | None = None) -> str:
    text = _decode_bytes(content)
    if not text:
        return ""

    content_type = ""
    if headers:
        content_type = str(headers.get("content-type", "")).lower()
        if not content_type:
            for header_key, header_value in headers.items():
                if str(header_key).lower() == "content-type":
                    content_type = str(header_value).lower()
                    break
    extension = Path(primary_document).suffix.lower()
    if extension in {".htm", ".html", ".xhtml"} or "text/html" in content_type:
        return _extract_text_from_html(text)
    if extension in {".xml", ".xhtml"} or "xml" in content_type:
        return _extract_text_from_xml(text)
    if extension == ".txt" or "text/plain" in content_type:
        return _clean_text(text)
    if "<" in text and ">" in text:
        if extension == ".xhtml":
            return _extract_text_from_html(text)
        return _extract_text_from_xml(text)
    return _clean_text(text)


def _decode_bytes(value: bytes) -> str:
    if not value:
        return ""
    encodings = ("utf-8", "utf-16", "latin-1")
    for encoding in encodings:
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="ignore")


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_text_from_html(value: str) -> str:
    parser = _HtmlTextExtractor()
    try:
        parser.feed(value)
        return _clean_text(parser.value())
    except Exception:
        return _clean_text(_strip_tags_fallback(value))


def _extract_text_from_xml(value: str) -> str:
    parser = _HtmlTextExtractor(allow_text_in_skip_tags=False)
    try:
        parser.feed(value)
        return _clean_text(parser.value())
    except Exception:
        stripped = _strip_tags_fallback(value)
        return _clean_text(stripped)


def _strip_tags_fallback(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


class _HtmlTextExtractor(HTMLParser):
    _BLOCK_ELEMENTS = {
        "p",
        "div",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "td",
        "th",
        "tr",
    }

    def __init__(self, allow_text_in_skip_tags: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self._allow_text_in_skip_tags = allow_text_in_skip_tags
        self._parts: list[str] = []
        self._skip_level = 0
        self._skip_stack: list[str] = []
        self._has_text = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript"}:
            self._skip_level += 1
            self._skip_stack.append(tag_lower)
            return
        if tag_lower in self._BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript"}:
            if self._skip_level > 0:
                self._skip_level -= 1
            if self._skip_stack and self._skip_stack[-1] == tag_lower:
                self._skip_stack.pop()
            if tag_lower == "script" and self._allow_text_in_skip_tags:
                self._parts.append(" ")
            return
        if tag_lower in self._BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_level > 0:
            if self._allow_text_in_skip_tags:
                return
            return
        clean = data.strip()
        if clean:
            self._parts.append(clean)
            self._has_text = True
        elif self._has_text:
            self._parts.append(" ")

    def value(self) -> str:
        return " ".join(part for part in self._parts if part.strip())


def _normalize_accession_number(accession_number: str) -> str:
    return "".join(char for char in str(accession_number) if char.isdigit())


def _load_cached_filing_document(cache_path: Path) -> str | None:
    try:
        raw = cache_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return ""
    if not raw.startswith("{"):
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, dict):
        return ""
    version = payload.get("version", _SEC_DOCUMENT_CACHE_VERSION)
    if str(version) != str(_SEC_DOCUMENT_CACHE_VERSION):
        return None
    text = payload.get("text")
    if isinstance(text, str):
        return text
    return ""


def _write_cached_filing_document(cache_path: Path, text: str) -> None:
    payload = {
        "version": _SEC_DOCUMENT_CACHE_VERSION,
        "text": text,
    }
    _write_json_cache(cache_path, payload)


def _load_json_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        raw = cache_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _write_json_cache(cache_path: Path, payload: dict[str, Any] | list[Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError:
        return
