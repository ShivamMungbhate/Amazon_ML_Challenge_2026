import re
import unicodedata
from collections import Counter

from rapidfuzz import fuzz


LEGAL_SUFFIXES = {
    "llc", "inc", "incorporated", "ltd", "limited",
    "corp", "corporation", "co", "company",
    "pvt", "private", "plc", "gmbh", "ag", "sa",
    "sarl", "pte", "lp", "llp"
}


def clean_text(value):
    """Basic Unicode-aware text normalization."""
    if value is None:
        return ""

    value = str(value).lower().strip()

    # Unicode normalization
    value = unicodedata.normalize("NFKD", value)

    # Remove combining accents
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    # Replace punctuation with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    """Normalize business name and remove trailing legal suffixes."""
    text = clean_text(value)

    if not text:
        return ""

    tokens = text.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def normalize_address(value):
    """Normalize address."""
    return clean_text(value)


def token_set(value):
    if not value:
        return set()

    return set(value.split())


def token_overlap(a, b):
    """Jaccard token overlap."""
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def same_script(a, b):
    """
    Rough script compatibility check.

    Returns:
      1 = same dominant script
      0 = different dominant scripts
    """

    def script_of(text):
        counts = Counter()

        for char in text:
            if not char.isalpha():
                continue

            name = unicodedata.name(char, "")

            if "LATIN" in name:
                counts["latin"] += 1
            elif "DEVANAGARI" in name:
                counts["devanagari"] += 1
            elif "TELUGU" in name:
                counts["telugu"] += 1
            elif "MALAYALAM" in name:
                counts["malayalam"] += 1
            elif "BENGALI" in name:
                counts["bengali"] += 1
            elif "TAMIL" in name:
                counts["tamil"] += 1
            elif "KANNADA" in name:
                counts["kannada"] += 1
            else:
                counts["other"] += 1

        if not counts:
            return "none"

        return counts.most_common(1)[0][0]

    return int(script_of(a) == script_of(b))


def legal_suffix_difference(original_a, original_b):
    """
    Detect cases where names become equal after removing
    legal suffixes but original normalized names differ.
    """

    a = clean_text(original_a)
    b = clean_text(original_b)

    na = normalize_name(original_a)
    nb = normalize_name(original_b)

    if not na or not nb:
        return 0

    return int(na == nb and a != b)


def word_order_difference(a, b):
    """
    Detect whether the same name tokens occur in a different order.
    """

    a_norm = normalize_name(a)
    b_norm = normalize_name(b)

    if not a_norm or not b_norm:
        return 0

    if a_norm == b_norm:
        return 0

    if Counter(a_norm.split()) == Counter(b_norm.split()):
        return 1

    return 0


def make_pair_features(row1, row2):
    """
    Generate all matching features for one S1/S2 or S1/S3 pair.

    row1 and row2 should be dictionaries containing:
        business_name
        address
        country
    """

    name1_raw = row1.get("business_name", "")
    name2_raw = row2.get("business_name", "")

    address1_raw = row1.get("address", "")
    address2_raw = row2.get("address", "")

    country1 = clean_text(row1.get("country", ""))
    country2 = clean_text(row2.get("country", ""))

    name1 = normalize_name(name1_raw)
    name2 = normalize_name(name2_raw)

    address1 = normalize_address(address1_raw)
    address2 = normalize_address(address2_raw)

    # -----------------------------
    # NAME FEATURES
    # -----------------------------

    features = {
        "name_exact": int(name1 == name2 and name1 != ""),

        "name_ratio": (
            fuzz.ratio(name1, name2) / 100.0
            if name1 and name2 else 0.0
        ),

        "name_token_sort": (
            fuzz.token_sort_ratio(name1, name2) / 100.0
            if name1 and name2 else 0.0
        ),

        "name_token_set": (
            fuzz.token_set_ratio(name1, name2) / 100.0
            if name1 and name2 else 0.0
        ),

        "name_partial": (
            fuzz.partial_ratio(name1, name2) / 100.0
            if name1 and name2 else 0.0
        ),

        "name_token_overlap": token_overlap(name1, name2),

        "name_length_diff": abs(len(name1) - len(name2)),

        "name_missing": int(not name1 or not name2),

        "legal_suffix_difference":
            legal_suffix_difference(name1_raw, name2_raw),

        "word_order_difference":
            word_order_difference(name1_raw, name2_raw),

        "same_script":
            same_script(name1_raw, name2_raw),

        # -----------------------------
        # ADDRESS FEATURES
        # -----------------------------

        "address_exact":
            int(
                address1 == address2
                and address1 != ""
            ),

        "address_ratio": (
            fuzz.ratio(address1, address2) / 100.0
            if address1 and address2 else 0.0
        ),

        "address_token_sort": (
            fuzz.token_sort_ratio(address1, address2) / 100.0
            if address1 and address2 else 0.0
        ),

        "address_token_set": (
            fuzz.token_set_ratio(address1, address2) / 100.0
            if address1 and address2 else 0.0
        ),

        "address_token_overlap":
            token_overlap(address1, address2),

        "address_length_diff":
            abs(len(address1) - len(address2)),

        "address_missing":
            int(not address1 or not address2),

        # -----------------------------
        # OTHER FEATURES
        # -----------------------------

        "country_exact":
            int(
                country1 == country2
                and country1 != ""
            ),
    }

    return features