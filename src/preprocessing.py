import re
import unicodedata
import pandas as pd


def normalize_text(value):
    """
    Basic text normalization.

    Keeps the original meaning/script while:
    - handling missing values
    - applying Unicode normalization
    - converting to lowercase
    - replacing punctuation with spaces
    - collapsing repeated whitespace
    """

    if pd.isna(value):
        return ""

    value = str(value)

    
    value = unicodedata.normalize("NFKC", value)


    value = value.casefold()


    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)


    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    """
    Normalize a business name.

    We intentionally do NOT remove legal suffixes here.
    For example:
        'Brightic Data LLC'
    becomes:
        'brightic data llc'

    Later we can create a separate feature that removes
    legal suffixes if our experiments show it helps.
    """
    return normalize_text(value)


def normalize_address(value):
    """
    Basic address normalization.

    We keep numbers because they are often extremely useful
    for matching addresses.
    """
    return normalize_text(value)


def normalize_country(value):
    """
    Normalize country names/codes.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().casefold()


def add_normalized_columns(df):
    """
    Add normalized versions of the important text columns.

    Original columns are NOT modified.

    Expected columns:
        business_name
        business_address
        country
    """

    df = df.copy()

    if "business_name" in df.columns:
        df["business_name_norm"] = df["business_name"].apply(
            normalize_name
        )

    if "business_address" in df.columns:
        df["business_address_norm"] = df["business_address"].apply(
            normalize_address
        )

    if "country" in df.columns:
        df["country_norm"] = df["country"].apply(
            normalize_country
        )

    return df