"""Taxonomy mappings for standardizing Chicago and NYC crime categories."""
from __future__ import annotations

# Unified Categories
VIOLENT = "violent"
PROPERTY = "property"
DRUG = "drug"

# Chicago Primary Type mapping
CHICAGO_MAPPING = {
    "HOMICIDE": VIOLENT,
    "ASSAULT": VIOLENT,
    "BATTERY": VIOLENT,
    "CRIM SEXUAL ASSAULT": VIOLENT,
    "ROBBERY": VIOLENT,
    "THEFT": PROPERTY,
    "BURGLARY": PROPERTY,
    "MOTOR VEHICLE THEFT": PROPERTY,
    "ARSON": PROPERTY,
    "NARCOTICS": DRUG,
    "OTHER NARCOTIC VIOLATION": DRUG,
}

# NYC KY_CD (Offense Classification Code) mapping
NYC_MAPPING = {
    101: VIOLENT,  # MURDER
    104: VIOLENT,  # RAPE
    105: VIOLENT,  # ROBBERY
    106: VIOLENT,  # FELONY ASSAULT
    109: VIOLENT,  # KIDNAPPING
    341: PROPERTY, # PETIT LARCENY
    351: PROPERTY, # CRIM MISCHIEF & RELATED OFFENSES
    352: PROPERTY, # CRIMINAL TRESPASS
    361: PROPERTY, # OFF. AGNST PUB ORD SENSBLTY & RGHTS TO PRIV
    114: PROPERTY, # ARSON
    230: DRUG,     # DANGEROUS DRUGS
    231: DRUG,     # POSSESSION
    232: DRUG,     # INTENT TO
}

def get_unified_category(city: str, raw_code: str | int) -> str | None:
    """Map a raw city code to the unified taxonomy."""
    if city.lower() == "chicago":
        return CHICAGO_MAPPING.get(str(raw_code).upper().strip())
    elif city.lower() == "nyc":
        try:
            return NYC_MAPPING.get(int(raw_code))
        except (ValueError, TypeError):
            return None
    return None
