"""
Cross-modal consistency module (C-Module) for brand alignment analysis.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tldextract

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from src.utils.logging import get_logger


log = get_logger(__name__)


class CModule:
    """
    Computes sentence-level similarity between brand cues extracted across modalities.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        thresh: float = 0.6,
        brand_lexicon_path: Optional[str] = None,
        use_ocr: bool = False,
        metadata_sources: Optional[Iterable[str]] = None,
        max_html_cache: int = 128,
        max_html_chars: int = 8000,
    ) -> None:
        self.model_name = model_name
        self.thresh = float(thresh)
        self.use_ocr = use_ocr
        self.max_html_cache = max(16, int(max_html_cache))
        self.max_html_chars = max(2000, int(max_html_chars))
        self._encoder: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._lexicon: Dict[str, str] = self._load_brand_lexicon(brand_lexicon_path)
        self._registered_sources: List[Path] = []
        self._loaded_sources: set[Path] = set()
        self._records: Dict[str, Dict[str, Any]] = {}
        self._html_cache: OrderedDict[str, str] = OrderedDict()

        if metadata_sources:
            for src in metadata_sources:
                self.register_metadata_source(src)

    # ------------------------------------------------------------------ #
    def setup(self, device: Optional[str] = None) -> None:
        """
        Lazily instantiate the Sentence-BERT encoder.
        """

        if self._encoder or SentenceTransformer is None:
            if SentenceTransformer is None and not hasattr(self, "_warned_st_missing"):
                log.warning("sentence-transformers not available; C-Module disabled.")
                self._warned_st_missing = True
            return

        try:
            self._encoder = SentenceTransformer(self.model_name, device=device)
            log.info("Loaded C-Module encoder: %s", self.model_name)
        except Exception as exc:  # pragma: no cover - dependency/runtime issues
            log.warning(
                "Failed to initialize SentenceTransformer '%s': %s",
                self.model_name,
                exc,
            )
            self._encoder = None

    # ------------------------------------------------------------------ #
    def register_metadata_source(self, csv_path: Optional[str | Path]) -> None:
        if not csv_path:
            return
        path = Path(csv_path)
        if not path.exists():
            log.debug("C-Module metadata source %s missing; skipping.", path)
            return
        if path not in self._registered_sources:
            self._registered_sources.append(path)

    # ------------------------------------------------------------------ #
    def score_consistency(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute consistency metrics for a single sample.

        Args:
            sample: Dictionary that should contain sample_id/id and optional raw fields:
                {
                    "sample_id": "...",
                    "url_text": "...",
                    "html_text": "...",
                    "html_path": "...",
                    "image_path": "...",
                }
        """

        resolved = self._resolve_sample_inputs(sample)
        brands, sources = self._extract_brands(resolved)

        result = self._build_empty_result(
            brands, sources, resolved.get("sample_id") or resolved.get("id")
        )

        active = {mod: text for mod, text in brands.items() if text}
        if len(active) < 2:
            result["meta"]["reason"] = "insufficient_brands"
            return result

        embeddings = self._encode_brands(active)
        if len(embeddings) < 2:
            result["meta"]["reason"] = "encoder_unavailable"
            return result

        pair_scores: Dict[Tuple[str, str], float] = {}
        for (mod_a, emb_a), (mod_b, emb_b) in combinations(embeddings.items(), 2):
            score = float(np.clip(np.dot(emb_a, emb_b), -1.0, 1.0))
            pair_scores[(mod_a, mod_b)] = score

        if not pair_scores:
            result["meta"]["reason"] = "encoder_failed"
            return result

        sims = list(pair_scores.values())
        result["c_mean"] = float(np.mean(sims))
        result["c_min"] = float(np.min(sims))

        per_mod = {mod: [] for mod in embeddings}
        for (mod_a, mod_b), score in pair_scores.items():
            per_mod[mod_a].append(score)
            per_mod[mod_b].append(score)

        for mod in ("url", "html", "visual"):
            scores = per_mod.get(mod) or []
            result[f"c_{mod}"] = float(np.mean(scores)) if scores else math.nan

        result["meta"]["n_pairs"] = len(pair_scores)
        result["meta"]["encoder"] = self.model_name if self._encoder else "unavailable"
        return result

    # ------------------------------------------------------------------ #
    def _build_empty_result(
        self,
        brands: Dict[str, Optional[str]],
        sources: Dict[str, Dict[str, Any]],
        sample_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "c_mean": math.nan,
            "c_min": math.nan,
            "c_url": math.nan,
            "c_html": math.nan,
            "c_visual": math.nan,
            "meta": {
                "brands": brands,
                "sources": sources,
                "sample_id": sample_id,
                "thresh": self.thresh,
                "available_modalities": [mod for mod, text in brands.items() if text],
            },
        }

    # ------------------------------------------------------------------ #
    def _resolve_sample_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(sample)
        sample_id = sample.get("sample_id") or sample.get("id")

        if sample_id and sample_id not in self._records:
            self._maybe_ingest_sources()

        if sample_id and sample_id in self._records:
            record = self._records[sample_id]
            for key in ("url_text", "html_text", "html_path", "image_path", "brand"):
                resolved.setdefault(key, record.get(key))

        resolved.setdefault("sample_id", sample_id)
        return resolved

    # ------------------------------------------------------------------ #
    def _maybe_ingest_sources(self) -> None:
        for path in self._registered_sources:
            if path in self._loaded_sources:
                continue
            self._ingest_metadata(path)
            self._loaded_sources.add(path)

    def _ingest_metadata(self, csv_path: Path) -> None:
        desired = (
            "id",
            "sample_id",
            "url_text",
            "html_text",
            "html_path",
            "img_path",
            "img_path_corrupt",
            "brand",
        )
        usecols = [col for col in desired if col in self._available_columns(csv_path)]

        try:
            if usecols:
                df = pd.read_csv(csv_path, usecols=usecols)
            else:
                df = pd.read_csv(csv_path)
        except Exception as exc:
            log.warning("Failed to read metadata CSV %s: %s", csv_path, exc)
            return

        for _, row in df.iterrows():
            sample_id = row.get("sample_id") or row.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                continue
            record = {
                "url_text": self._safe_str(row.get("url_text")),
                "html_text": self._safe_str(row.get("html_text")),
                "html_path": self._safe_str(row.get("html_path")),
                "image_path": self._safe_str(
                    row.get("img_path_corrupt") or row.get("img_path")
                ),
                "brand": self._safe_str(row.get("brand")),
            }
            self._records[sample_id] = record

    def _available_columns(self, csv_path: Path) -> set[str]:
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns
            return set(cols)
        except Exception as exc:
            log.warning("Failed to parse columns for %s: %s", csv_path, exc)
            return set()

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return str(value)

    # ------------------------------------------------------------------ #
    def _extract_brands(
        self, resolved: Dict[str, Any]
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Dict[str, Any]]]:
        brands: Dict[str, Optional[str]] = {"url": None, "html": None, "visual": None}
        sources: Dict[str, Dict[str, Any]] = {
            "url": {},
            "html": {},
            "visual": {},
        }

        url_brand, sources["url"] = self._brand_from_url(resolved.get("url_text"))
        html_brand, sources["html"] = self._brand_from_html(
            resolved.get("html_text"), resolved.get("html_path")
        )
        visual_brand, sources["visual"] = self._brand_from_visual(
            resolved.get("image_path")
        )

        brands["url"] = url_brand
        brands["html"] = html_brand
        brands["visual"] = visual_brand
        return brands, sources

    def _brand_from_url(
        self, url_text: Optional[str]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        meta = {"method": "tldextract", "raw": None}
        if not url_text:
            meta["reason"] = "missing_url"
            return None, meta

        url_candidate = url_text.strip()
        if not url_candidate.startswith(("http://", "https://")):
            url_candidate = "http://" + url_candidate

        try:
            parsed = tldextract.extract(url_candidate)
            domain = parsed.domain or ""
        except Exception as exc:
            meta["reason"] = f"tldextract_error:{exc}"
            return None, meta

        meta["raw"] = domain
        brand = self._match_brand(domain)
        if not brand and parsed.subdomain:
            tokens = re.split(r"[\W_]+", parsed.subdomain)
            for token in tokens:
                brand = self._match_brand(token)
                if brand:
                    meta["method"] = "subdomain"
                    break

        if not brand:
            brand = domain or None
        elif brand != domain:
            meta["method"] = "lexicon"
        return brand, meta

    def _brand_from_html(
        self, html_text: Optional[str], html_path: Optional[str]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        meta = {"method": None, "raw": None}
        text = html_text or self._load_html_text(html_path)
        if not text:
            meta["reason"] = "missing_html"
            return None, meta

        title = self._extract_title(text)
        if title:
            meta["raw"] = title
            brand = self._match_brand(title) or self._pick_major_token(title)
            if brand:
                meta["method"] = "title"
                return brand, meta

        meta_tag = self._extract_meta_brand(text)
        if meta_tag:
            meta["raw"] = meta_tag
            brand = self._match_brand(meta_tag) or self._pick_major_token(meta_tag)
            if brand:
                meta["method"] = "meta_tag"
                return brand, meta

        lexicon_hit = self._scan_lexicon(text)
        if lexicon_hit:
            meta["raw"] = lexicon_hit
            meta["method"] = "lexicon_scan"
            return lexicon_hit, meta

        meta["reason"] = "no_brand_signal"
        return None, meta

    def _brand_from_visual(
        self, image_path: Optional[str]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        meta = {"method": "ocr_stub", "reason": "ocr_disabled"}
        if self.use_ocr and image_path:
            meta["reason"] = "ocr_not_implemented"
        return None, meta

    # ------------------------------------------------------------------ #
    def _load_html_text(self, html_path: Optional[str]) -> str:
        if not html_path:
            return ""
        cached = self._html_cache.get(html_path)
        if cached is not None:
            return cached
        path = Path(html_path)
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                text = path.read_text(encoding="latin-1", errors="ignore")
            except Exception:
                return ""
        if len(text) > self.max_html_chars:
            text = text[: self.max_html_chars]
        self._html_cache[html_path] = text
        if len(self._html_cache) > self.max_html_cache:
            self._html_cache.popitem(last=False)
        return text

    def _extract_title(self, html_text: str) -> Optional[str]:
        if not html_text:
            return None
        if BeautifulSoup is None:
            match = re.search(
                r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL
            )
            return match.group(1).strip() if match else None
        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            soup = BeautifulSoup(html_text, "html.parser")
        title = soup.title.string if soup.title else None
        return title.strip() if title else None

    @staticmethod
    def _extract_meta_brand(html_text: str) -> Optional[str]:
        match = re.search(
            r'<meta[^>]+(?:property|name)=["\'](?:og:site_name|application-name|twitter:title)["\'][^>]+content=["\']([^"\']+)["\']',
            html_text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return None

    def _scan_lexicon(self, html_text: str) -> Optional[str]:
        if not self._lexicon:
            return None
        normalized = self._normalize_text(html_text)
        for norm, brand in self._lexicon.items():
            if not norm:
                continue
            if norm in normalized:
                return brand
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _pick_major_token(self, text: str) -> Optional[str]:
        tokens = [tok for tok in re.split(r"[\W_]+", text) if tok]
        if not tokens:
            return None
        tokens.sort(key=len, reverse=True)
        candidate = tokens[0]
        return self._match_brand(candidate) or candidate

    # ------------------------------------------------------------------ #
    def _encode_brands(self, brands: Dict[str, str]) -> Dict[str, np.ndarray]:
        self.setup()
        if not self._encoder:
            return {}

        to_encode = [
            text for text in brands.values() if text not in self._embedding_cache
        ]
        if to_encode:
            try:
                vectors = self._encoder.encode(
                    to_encode,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                for text, vec in zip(to_encode, vectors):
                    self._embedding_cache[text] = vec
            except Exception as exc:  # pragma: no cover - runtime guard
                log.warning("Failed to encode brands %s: %s", to_encode, exc)
                return {}

        return {mod: self._embedding_cache[text] for mod, text in brands.items()}

    def _load_brand_lexicon(self, path: Optional[str]) -> Dict[str, str]:
        if not path:
            return {}
        lex_path = Path(path)
        if not lex_path.exists():
            log.warning(
                "Brand lexicon %s not found; lexicon features disabled.", lex_path
            )
            return {}
        lexicon: Dict[str, str] = {}
        for line in lex_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            lexicon[self._normalize_text(entry)] = entry
        log.info("Loaded %d brand lexicon entries from %s", len(lexicon), lex_path)
        return lexicon

    def _match_brand(self, candidate: str) -> Optional[str]:
        if not candidate:
            return None
        norm = self._normalize_text(candidate)
        if norm in self._lexicon:
            return self._lexicon[norm]
        return None
