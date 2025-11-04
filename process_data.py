from pathlib import Path
import re
import pandas as pd

# ---------- Paths ----------
INPUT = Path("raw_data/kaggle_tweets.csv")
OUTPUT = Path("raw_data/help_tweets_strict.csv")

# ---------- Columns present in your file ----------
# date, content, hashtags, like_count, rt_count, followers_count, isVerified, language, coordinates, place, source

# ---------- Config ----------
KEEP_LANGS = {"tr", "en"}
USECOLS = ["date", "content", "hashtags", "language", "coordinates", "place"]  # only what we need
CHUNKSIZE = 200_000  # adjust to your RAM

# Optional event window (set to ISO-8601 strings if you want to restrict by time)
EVENT_START = None  # e.g., "2023-02-06 01:00:00+00:00"
EVENT_END   = None  # e.g., "2023-03-31 23:59:59+00:00"

# ---------- Regex helpers ----------
def kw_regex(phrases: set[str]) -> re.Pattern:
    """
    Case-insensitive regex that matches whole tokens and hashtag forms.
    Works with diacritics. Example: matches 'yardım', 'YARDIM', '#yardım'.
    """
    esc = [re.escape(p) for p in sorted(phrases, key=len, reverse=True)]
    group = "|".join(esc)
    return re.compile(rf"(?i)(?<!\w)#?(?:{group})(?!\w)")

URL_RE      = re.compile(r"https?://\S+", re.IGNORECASE)
PHONE_RE    = re.compile(r"(?i)(?:\+?90[\s\-.]?)?(?:\(?0?\)?[\s\-.]?)?\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4}")
COORD_TEXT_RE = re.compile(r"[-+]?\d{1,2}\.\d+\s*[, ]\s*[-+]?\d{1,3}\.\d+")
ADDRESS_RE  = re.compile(r"(?i)\b(mah\.?|mahalle|cad\.?|cadd?e|sok\.?|sokak|no[:\s]?|apt\.?|apartman|blok|kat[:\s]?)\b")
RETWEET_RE  = re.compile(r"(?i)^\s*rt\s+@")
OFFER_RE    = kw_regex({
    # Turkish offer/announcement
    "bağış", "yardım kampanyası", "yardım topluyoruz", "toplama merkezi", "dağıtıyoruz",
    "ulaştırıyoruz", "kabul ediyoruz", "ihtiyaç listesi", "kampanya",
    # English offer/announcement
    "donation", "fundraiser", "collection center", "drop[- ]off", "we are sending", "we are collecting"
})

# Strong SOS / entrapment phrases (high precision)
SOS_TR = kw_regex({
    "yardım edin", "yardım lazım", "acil yardım", "acil", "imdat",
    "enkazdayım", "enkaz altındayım", "göçük altındayım", "mahsur kaldım",
    "sıkıştım", "nefes alamıyorum", "sesimi duyan var mı", "kurtarın", "ambulans gönderin",
})
SOS_EN = kw_regex({
    "need help", "please help", "urgent help", "help us",
    "trapped", "under rubble", "stuck", "send rescue", "send ambulance",
    "can't move", "cannot move", "we need help"
})

# Location intent words (we'll also require phone/coords/address)
LOC_WORDS = kw_regex({"adres", "konum", "lokasyon", "koordinat", "location", "address", "coordinates"})

# First-person signals (helpful for precision when combined with SOS)
FIRST_PERSON = kw_regex({
    # Turkish
    "ben", "biz", "buradayım", "enkazdayım", "ailem", "annem", "babam", "kardeşim", "çocuğum", "eşim",
    # English
    "i am", "i'm", "we are", "we're", "my family", "my child", "my children"
})

def discover_keep_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    required = {"content", "language"}
    if not required.issubset(set(header.columns)):
        raise ValueError(f"Missing required columns: {sorted(required - set(header.columns))}")
    return [c for c in USECOLS if c in header.columns]

def parse_dates_if_any(df: pd.DataFrame) -> None:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

def within_event_window(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns or (EVENT_START is None and EVENT_END is None):
        return pd.Series(True, index=df.index)
    start = pd.to_datetime(EVENT_START, utc=True) if EVENT_START else None
    end = pd.to_datetime(EVENT_END, utc=True) if EVENT_END else None
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["date"] >= start
    if end is not None:
        mask &= df["date"] <= end
    return mask

def combine_text(df: pd.DataFrame) -> pd.Series:
    txt = df["content"].fillna("")
    tags = df["hashtags"].fillna("") if "hashtags" in df.columns else ""
    # casefold for robust lowercasing with diacritics
    return (txt.astype(str) + " " + tags.astype(str)).str.casefold()

def has_coordinates_column(df: pd.DataFrame) -> pd.Series:
    if "coordinates" not in df.columns:
        return pd.Series(False, index=df.index)
    # If coordinates contain numbers or lat/lon-like text
    col = df["coordinates"].astype(str).fillna("").str.casefold()
    return col.str.contains(r"\d", regex=True) | col.str.contains(COORD_TEXT_RE)

def strict_help_mask(text: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    High-precision rule set:
      1) Must have a strong SOS phrase (TR or EN)
      AND at least one strong locator: phone OR coord text/column OR address tokens OR explicit 'location' word.
      2) Optionally boost if first-person is present.
      3) Exclude clear offers/announcements and retweets.
      4) Exclude URL-only promos (many URLs without SOS).
    """
    m_sos   = text.str.contains(SOS_TR, na=False) | text.str.contains(SOS_EN, na=False)
    m_locw  = text.str.contains(LOC_WORDS, na=False)
    m_phone = text.str.contains(PHONE_RE, na=False)
    m_coord = text.str.contains(COORD_TEXT_RE, na=False) | has_coordinates_column(df)
    m_addr  = text.str.contains(ADDRESS_RE, na=False)
    m_fp    = text.str.contains(FIRST_PERSON, na=False)

    m_offer = text.str.contains(OFFER_RE, na=False)
    m_rt    = text.str.contains(RETWEET_RE, na=False)
    url_cnt = text.str.count(URL_RE)

    # Core precision rule: SOS + (phone OR coord OR address OR 'location' words)
    core = m_sos & (m_phone | m_coord | m_addr | m_locw)

    # Slightly relaxed when strong first-person is present + at least one hard locator
    relaxed = (m_sos & m_fp & (m_phone | m_coord))

    # Drop likely offers/announcements and retweets
    base = (core | relaxed) & ~m_offer & ~m_rt

    # Filter out cases with many URLs and no non-URL content (reduce news/promos)
    non_url_signal = (text.str.len() - text.str.count(URL_RE).fillna(0)*10) > 20  # crude but effective
    return base & non_url_signal

def stream_strict_filter(input_path: Path, output_path: Path, chunksize: int = CHUNKSIZE) -> None:
    keep_cols = discover_keep_columns(input_path)
    first = True
    seen_text = set()  # dedup by normalized text

    total_in = total_lang = total_time = total_kept = 0

    for chunk in pd.read_csv(input_path, usecols=keep_cols, chunksize=chunksize):
        total_in += len(chunk)

        # Language filter
        chunk["language"] = chunk["language"].astype(str).str.casefold()
        chunk = chunk[chunk["language"].isin(KEEP_LANGS)]
        total_lang += len(chunk)
        if chunk.empty:
            continue

        # Optional time window
        parse_dates_if_any(chunk)
        mask_time = within_event_window(chunk)
        chunk = chunk[mask_time]
        total_time += len(chunk)
        if chunk.empty:
            continue

        # Build text
        text = combine_text(chunk)

        # Strict mask
        mask = strict_help_mask(text, chunk)
        filt = chunk[mask].copy()
        if filt.empty:
            continue

        # Deduplicate by normalized text (casefold + strip spaces)
        norm = text.loc[filt.index].str.replace(r"\s+", " ", regex=True).str.strip()
        keep_idx = []
        for i, s in zip(filt.index, norm):
            if s not in seen_text:
                keep_idx.append(i)
                seen_text.add(s)
        filt = filt.loc[keep_idx]
        if filt.empty:
            continue

        # Optional debug flags (helpful for audits)
        t = text.loc[filt.index]
        filt["_hit_sos"]   = t.str.contains(SOS_TR, na=False) | t.str.contains(SOS_EN, na=False)
        filt["_hit_phone"] = t.str.contains(PHONE_RE, na=False)
        filt["_hit_coord"] = t.str.contains(COORD_TEXT_RE, na=False) | has_coordinates_column(filt)
        filt["_hit_addr"]  = t.str.contains(ADDRESS_RE, na=False)
        filt["_hit_locw"]  = t.str.contains(LOC_WORDS, na=False)
        filt["_first_person"] = t.str.contains(FIRST_PERSON, na=False)

        # Write
        filt.to_csv(
            output_path,
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        total_kept += len(filt)
        first = False

    print(
        f"Done. Read: {total_in:,} | After lang: {total_lang:,} | "
        f"After time: {total_time:,} | Kept strict: {total_kept:,} -> {output_path}"
    )

if __name__ == "__main__":
    stream_strict_filter(INPUT, OUTPUT)
