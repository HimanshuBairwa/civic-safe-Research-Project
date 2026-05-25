"""Taxonomy mappings for standardizing Chicago and NYC crime categories.

Mappings verified against live SODA API responses on 2025-05-25:
  Chicago: https://data.cityofchicago.org/resource/ijzp-q8t2.json
  NYC:     https://data.cityofnewyork.us/resource/qgea-i56i.json
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Unified Categories
# ---------------------------------------------------------------------------
VIOLENT = "violent"
PROPERTY = "property"
DRUG = "drug"

# ---------------------------------------------------------------------------
# Chicago Primary Type → Unified Category
# Verified against live groupby query returning all primary_type values.
# ---------------------------------------------------------------------------
CHICAGO_MAPPING: dict[str, str] = {
    # Violent
    "HOMICIDE": VIOLENT,                    # 14,230 records
    "ASSAULT": VIOLENT,                     # 576,088 records
    "BATTERY": VIOLENT,                     # 1,558,724 records
    "CRIM SEXUAL ASSAULT": VIOLENT,         # 27,224 records
    "CRIMINAL SEXUAL ASSAULT": VIOLENT,     # 12,604 records (variant spelling!)
    "ROBBERY": VIOLENT,                     # 317,206 records
    "KIDNAPPING": VIOLENT,                  # 7,551 records
    "WEAPONS VIOLATION": VIOLENT,           # 127,484 records
    "HUMAN TRAFFICKING": VIOLENT,           # 150 records
    # Property
    "THEFT": PROPERTY,                      # 1,818,025 records
    "BURGLARY": PROPERTY,                   # 452,537 records
    "MOTOR VEHICLE THEFT": PROPERTY,        # 441,163 records
    "ARSON": PROPERTY,                      # 14,623 records
    "CRIMINAL DAMAGE": PROPERTY,            # 972,502 records
    "CRIMINAL TRESPASS": PROPERTY,          # 229,853 records
    "DECEPTIVE PRACTICE": PROPERTY,         # 397,295 records
    # Drug
    "NARCOTICS": DRUG,                      # 767,578 records
    "OTHER NARCOTIC VIOLATION": DRUG,       # 166 records
}

# ---------------------------------------------------------------------------
# NYC KY_CD (Offense Key Code) → Unified Category
# Verified against live groupby query on ky_cd, ofns_desc.
# ---------------------------------------------------------------------------
NYC_MAPPING: dict[int, str] = {
    # Violent
    101: VIOLENT,   # MURDER & NON-NEGL. MANSLAUGHTER (8,284 records)
    104: VIOLENT,   # RAPE (30,078 records)
    105: VIOLENT,   # ROBBERY (346,607 records)
    106: VIOLENT,   # FELONY ASSAULT (423,210 records)
    124: VIOLENT,   # KIDNAPPING & RELATED OFFENSES (3,560 records)
    # Property
    107: PROPERTY,  # BURGLARY (323,118 records)
    109: PROPERTY,  # GRAND LARCENY (880,071 records) — NOT kidnapping!
    110: PROPERTY,  # GRAND LARCENY OF MOTOR VEHICLE (201,767 records)
    114: PROPERTY,  # ARSON (20,652 records)
    341: PROPERTY,  # PETIT LARCENY (1,772,206 records)
    351: PROPERTY,  # CRIMINAL MISCHIEF & RELATED (753,156 records)
    352: PROPERTY,  # CRIMINAL TRESPASS (96,468 records)
    353: PROPERTY,  # UNAUTHORIZED USE OF A VEHICLE (29,332 records)
    # Drug
    117: DRUG,      # DANGEROUS DRUGS — felony (117,626 records)
    235: DRUG,      # DANGEROUS DRUGS — misdemeanor (373,015 records)
}


def get_unified_category(city: str, raw_code: str | int) -> str | None:
    """Map a raw city-specific code to the unified taxonomy.

    Args:
        city: "chicago" or "nyc" (case-insensitive).
        raw_code: For Chicago, a string primary_type.
                  For NYC, an integer or string-encoded KY_CD.

    Returns:
        One of "violent", "property", "drug", or None if unmapped.
    """
    if city.lower() == "chicago":
        return CHICAGO_MAPPING.get(str(raw_code).upper().strip())
    elif city.lower() == "nyc":
        try:
            return NYC_MAPPING.get(int(raw_code))
        except (ValueError, TypeError):
            return None
    return None
