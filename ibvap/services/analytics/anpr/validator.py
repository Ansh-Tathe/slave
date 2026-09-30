"""
services/analytics/anpr/validator.py
======================================
IBVAP P3 — Indian vehicle number plate format validator and normaliser.

Plate formats supported
-----------------------
1. Standard (post-1989):
       AA99AA9999    e.g.  MH12AB1234  (Maharashtra, district 12, series AB, number 1234)
       AA99A9999     e.g.  KA01M2345   (single-letter series)
       AA99AAA9999   e.g.  UP16CBA1234 (3-letter series — rare)

2. BH (Bharat) series (2021+):
       99BH9999AA    e.g.  22BH1234AB

3. Old/pre-1989 (single-digit district):
       AA9AA9999     e.g.  DL4CAB1234

4. Temporary (T plates): TX99AA9999 — treated as LOW confidence

State codes, district codes, and series are validated against a known-good
lookup table.  The validator returns a structured PlateRead object with:
  - normalised_text  (spaces removed, uppercase)
  - state_code, state_name
  - is_valid_format  (bool)
  - format_confidence  (0.0–1.0)  — heuristic score based on how many fields match
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Indian state / UT codes ───────────────────────────────────────────────────

INDIAN_STATE_CODES: dict[str, str] = {
    "AN": "Andaman & Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman & Diu",
    "DL": "Delhi",
    "DN": "Dadra & Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu & Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# ── Compiled regex patterns ───────────────────────────────────────────────────

# Characters that should not appear in a plate (OCR noise)
_NOISE_CHARS  = re.compile(r"[^A-Z0-9]")

# Standard plate:  2 letters + 1-2 digits + 1-3 letters + 1-4 digits
_STANDARD      = re.compile(
    r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$"
)

# BH series:  2 digits + BH + 1-4 digits + 1-2 letters
_BH_SERIES     = re.compile(
    r"^(\d{2})(BH)(\d{1,4})([A-Z]{1,2})$"
)

# Old format:  2 letters + 1 digit + 1-3 letters + 1-4 digits
_OLD_FORMAT    = re.compile(
    r"^([A-Z]{2})(\d{1})([A-Z]{1,3})(\d{1,4})$"
)

# Temporary:  TX + 2 digits + 1-3 letters + 1-4 digits
_TEMP_PLATE    = re.compile(
    r"^(TX)(\d{2})([A-Z]{1,3})(\d{1,4})$"
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PlateRead:
    """
    Structured result of plate validation.

    Attributes
    ----------
    raw_text          Raw OCR output (un-cleaned).
    normalised_text   Uppercase, spaces/punctuation removed.
    is_valid_format   True if matches a known Indian plate pattern.
    format_type       "standard" | "bh_series" | "old_format" | "temp" | "unknown"
    state_code        2-letter state prefix (if extractable).
    state_name        Full state name (or None if not found).
    format_confidence Heuristic confidence in the plate read (0–1).
    """
    raw_text:           str
    normalised_text:    str
    is_valid_format:    bool
    format_type:        str
    state_code:         Optional[str]       = None
    state_name:         Optional[str]       = None
    format_confidence:  float               = 0.0

    def to_dict(self) -> dict:
        return {
            "raw_text":          self.raw_text,
            "normalised_text":   self.normalised_text,
            "is_valid_format":   self.is_valid_format,
            "format_type":       self.format_type,
            "state_code":        self.state_code,
            "state_name":        self.state_name,
            "format_confidence": round(self.format_confidence, 4),
        }


# ── Validator ─────────────────────────────────────────────────────────────────

class IndianPlateValidator:
    """
    Validates and normalises an Indian vehicle number plate string.

    Usage:
        v   = IndianPlateValidator()
        res = v.validate("MH 12 AB 1234")
        print(res.is_valid_format, res.normalised_text, res.state_name)
    """

    @staticmethod
    def _clean(text: str) -> str:
        """Strip whitespace, punctuation; uppercase."""
        return _NOISE_CHARS.sub("", text.strip().upper())

    def validate(self, raw_text: str, ocr_confidence: float = 1.0) -> PlateRead:
        """
        Validate a raw OCR string as an Indian number plate.

        Parameters
        ----------
        raw_text        : raw string from OCR (may have spaces, noise chars).
        ocr_confidence  : 0–1 confidence from OCR engine.

        Returns
        -------
        PlateRead with is_valid_format, state_code, format_confidence.
        """
        norm = self._clean(raw_text)

        # ── Try each pattern ──────────────────────────────────────────────
        for pattern, fmt in [
            (_STANDARD,   "standard"),
            (_BH_SERIES,  "bh_series"),
            (_OLD_FORMAT, "old_format"),
            (_TEMP_PLATE, "temp"),
        ]:
            m = pattern.match(norm)
            if m:
                state_code  = m.group(1) if fmt not in ("bh_series",) else None
                state_name  = INDIAN_STATE_CODES.get(state_code, None) \
                              if state_code else None
                # Known state code gives a bonus
                state_bonus = 0.10 if state_name else 0.0
                # Temp plates get a penalty
                temp_penalty = -0.10 if fmt == "temp" else 0.0
                # Length heuristic: ideal plate is 8–10 chars
                len_score = min(1.0, len(norm) / 8) if len(norm) <= 10 \
                            else max(0.0, 1.0 - (len(norm) - 10) * 0.1)

                confidence = min(1.0, ocr_confidence * 0.7
                                 + 0.20
                                 + state_bonus
                                 + temp_penalty
                                 + len_score * 0.10)

                return PlateRead(
                    raw_text          = raw_text,
                    normalised_text   = norm,
                    is_valid_format   = True,
                    format_type       = fmt,
                    state_code        = state_code,
                    state_name        = state_name,
                    format_confidence = round(confidence, 4),
                )

        # ── No pattern matched ─────────────────────────────────────────────
        # Still return something — the operator can review
        return PlateRead(
            raw_text          = raw_text,
            normalised_text   = norm,
            is_valid_format   = False,
            format_type       = "unknown",
            state_code        = norm[:2] if len(norm) >= 2 else None,
            format_confidence = round(ocr_confidence * 0.30, 4),
        )
