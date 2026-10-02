"""Preprocessing: normalize business names and addresses for blocking & matching."""
import re
from typing import Tuple

# Common legal suffixes to strip for normalization
LEGAL_SUFFIXES = re.compile(
    r"\b(llc|ltd|inc|corp|co|plc|llp|pty|sch\s+bg|gmbh|sa|sas|ab|ag|bv|nv|spz"
    r"|srl|sl|oy|ab|asa|asa|as|oyj|ab|ba|se|sd|bhd|psc|llc|llp|pllc|pllc|pc|pllc"
    r"|ltda|ltd|pvt|private|limited|corporation|company|services|solutions|group"
    r"|international|ventures|holdings|technologies|enterprises|industries|global"
    r"|systems|network|networks|digital|digital|capital|partners|partnership|assoc"
    r"|associates|trading|trading|trading|trade|trading|lending|finance|financial"
    r"|banking|insurance|advisory|consulting|consultancy|engineering|construction"
    r"|development|software|hardware|electronics|manufacturing|manufacturer"
    r"|distributor|distributors|supplier|suppliers|retail|wholesale|wholesaler"
    r"|agency|agencies|broker|brokers|brokerage|exporter|importer|logistics"
    r"|transport|transportation|shipping|courier|courier|warehouse|warehousing"
    r"|real\s*estate|realty|property|properties|facility|facilities|store|stores"
    r"|shop|shops|outlet|outlets|restaurant|restaurants|cafe|cafes|hotel|hotels"
    r"|motel|motels|resort|resorts|clinic|clinics|hospital|hospitals|pharmacy"
    r"|pharmacies|laboratory|laboratories|clinic|center|centers|centre|centres"
    r"|institute|institutes|foundation|foundations|society|societies|association"
    r"|association|organization|organizations|organisation|organisations|union"
    r"|alliance|coalition|consortium|syndicate|trust|trusts|fund|funds|foundation"
    r"|endowment|endowments|fellowship|fellowships|scholarship|scholarships"
    r"|research|research|research\s*institute|research\s*lab|lab|labs"
    r")\b\.?",
    re.IGNORECASE,
)

# Suffix replacement map for consistent normalization
SUFFIX_REPLACEMENTS = {
    "corp": "corporation",
    "co": "company",
    "ltd": "limited",
    "llc": "limited liability company",
    "pllc": "professional limited liability company",
    "pc": "professional corporation",
    "pvt": "private",
    "pty": "proprietary",
    "gmbh": "company",
    "bv": "limited",
    "nv": "company",
    "sa": "company",
    "srl": "limited liability company",
    "sl": "limited",
    "ab": "company",
    "oy": "company",
    "oyj": "company",
    "ag": "company",
    "se": "company",
    "as": "company",
    "asa": "company",
    "sd": "company",
    "psc": "company",
    "bhd": "company",
    "sch bg": "school",
}

# Punctuation / noise to strip
PUNCT_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

# Common abbreviations in addresses
ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "sq": "square", "pk": "park", "wy": "way", "hwy": "highway",
    "trl": "trail", "cir": "circle", "plz": "plaza", "ter": "terrace",
    "expy": "expressway", "fwy": "freeway", "pky": "parkway",
    "cw": "cw", "ccw": "ccw", "neh": "neh", "se": "se", "sw": "sw", "nw": "nw",
    "ste": "suite", "suite": "suite", "unit": "unit", "fl": "floor",
    "floor": "floor", "bldg": "building", "building": "building",
    "dept": "department", "dept": "department", "fk": "fork",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces, normalize suffixes."""
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = PUNCT_RE.sub(" ", t)
    t = MULTI_SPACE_RE.sub(" ", t)
    return t


def normalize_name(name: str) -> str:
    """Normalize a business name for matching: strip suffixes, punctuation, etc."""
    n = normalize_text(name)
    # Replace legal suffixes with canonical forms
    def _replace_suffix(m):
        word = m.group(1).lower().rstrip(".")
        return SUFFIX_REPLACEMENTS.get(word, word)
    n = re.sub(LEGAL_SUFFIXES, _replace_suffix, n)
    # Clean up again after replacement
    n = MULTI_SPACE_RE.sub(" ", n).strip()
    return n


def normalize_address(addr: str) -> str:
    """Normalize an address: expand abbreviations, lowercase, strip punctuation."""
    a = normalize_text(addr)
    tokens = a.split()
    expanded = []
    for tok in tokens:
        # Strip trailing punctuation from token
        clean = tok.rstrip(".")
        expanded.append(ADDR_ABBR.get(clean, clean))
    return " ".join(expanded)


def preprocess_record(entity_id: str, name: str, address: str, country: str) -> Tuple[str, str, str, str]:
    """Return normalized (entity_id, norm_name, norm_address, country)."""
    return (
        entity_id,
        normalize_name(name),
        normalize_address(address),
        country.strip().lower() if isinstance(country, str) else "",
    )
