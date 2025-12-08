from pathlib import Path
import pandas as pd
import re
import csv

# ---------------- Paths ----------------
INPUT_DIR  = Path("raw_data/crisis_data")  # folder containing 3 TSV files
OUTPUT_CSV = Path("data/crisis_help_calls.csv")
CHUNKSIZE  = 200_000

# Expected TSV schema: id, event, source, text, lang, lang_conf, class_label
USECOLS = ["id", "event", "source", "text", "lang", "lang_conf", "class_label"]

# How aggressive to be: "strict" | "balanced" | "recall"
MODE = "balanced"
ALLOW_RTS_IF_MATCH = True  # keep RTs only if they look like real help calls

# ---------------- Regex helpers & lexicons (TEXT ONLY) ----------------
def kw_regex(phrases: set[str]) -> re.Pattern:
    esc = [re.escape(p) for p in sorted(phrases, key=len, reverse=True)]
    return re.compile(rf"(?i)(?<!\w)(?:{'|'.join(esc)})(?!\w)")

URL_RE      = re.compile(r"https?://\S+", re.IGNORECASE)
RETWEET_RE  = re.compile(r"(?i)^\s*rt\s+@")

# Contact / location signals in text
PHONE_RE    = re.compile(r"(?i)(?:\+?\d{1,3}[\s\-.]?)?(?:\(?0?\)?[\s\-.]?)?\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4}")
COORD_RE    = re.compile(r"[-+]?\d{1,2}\.\d+\s*[, ]\s*[-+]?\d{1,3}\.\d+")
ADDRESS_RE  = re.compile(r"(?i)\b(mah\.?|mahalle|cad\.?|cadd?e|sok\.?|sokak|no[:\s]?|apt\.?|apartman|blok|kat[:\s]?)\b")
LOC_WORDS_RE= kw_regex({"adres", "konum", "lokasyon", "koordinat", "location", "address", "coordinates", "pin", "share location"})

# Hazards (broad, TR+EN) — used only as TEXT context (no other columns)
HAZ_TR = {
    "deprem", "artçı", "yangın", "orman yangını", "sel", "su baskını", "taşkın",
    "dere taştı", "nehir taştı", "fırtına", "kasırga", "hortum", "heyelan",
    "çamur kayması", "çığ", "tsunami", "volkan", "volkanik patlama", "patlama", "göçük"
}
HAZ_EN = {
    "earthquake", "aftershock", "wildfire", "fire", "flood", "flooded", "flooding",
    "flash flood", "storm", "hurricane", "tornado", "twister", "landslide",
    "mudslide", "avalanche", "tsunami", "volcano", "eruption", "explosion", "blast", "collapse",
    "storm surge"
}
HAZARDS_RE = kw_regex(HAZ_TR | HAZ_EN)

# Strong SOS
SOS_TR = {
    "yardım edin", "yardım lazım", "acil yardım", "imdat",
    "enkazdayım", "enkaz altındayım", "göçük altındayım", "mahsur kaldım",
    "sıkıştım", "nefes alamıyorum", "sesimi duyan var mı", "kurtarın", "ambulans gönderin",
}
SOS_EN = {
    "need help", "please help", "urgent help", "help us",
    "trapped", "under rubble", "stuck", "send rescue", "send ambulance",
    "can't move", "cannot move", "we need help",
}
SOS_RE = kw_regex(SOS_TR | SOS_EN)

# Flood-specific distress cues
FLOOD_SOS_TR = {
    "çatıdayız", "çatıda mahsur", "çatıda kaldık", "su yükseliyor",
    "evleri su bastı", "sular bastı", "sular yükseldi", "boğuluyoruz",
    "bot lazım", "tekne lazım", "tahliye edin", "tahliye lazım"
}
FLOOD_SOS_EN = {
    "on the roof", "roof trapped", "water rising", "water level rising",
    "house flooded", "homes flooded", "need a boat", "need boat",
    "we are drowning", "evacuate us"
}
FLOOD_SOS_RE = kw_regex(FLOOD_SOS_TR | FLOOD_SOS_EN)

# General needs / medical / evacuation
NEEDS_TR = {
    "su lazım", "yemek", "erzak", "barınak", "çadır", "battaniye", "ısıtıcı",
    "generator", "jeneratör", "powerbank", "bebek maması", "mama",
    "ilaç", "insülin", "oksijen", "kan lazım", "kan ihtiyacı", "sedye",
    "ambulans", "doktor", "tahliye", "tahliye edin", "bot", "tekne"
}
NEEDS_EN = {
    "need water", "need food", "food needed", "water needed", "shelter",
    "blanket", "heater", "generator", "power bank", "baby formula",
    "medicine", "meds", "insulin", "oxygen", "blood needed", "stretcher",
    "ambulance", "doctor", "evacuate", "evacuation", "boat"
}
NEEDS_RE = kw_regex(NEEDS_TR | NEEDS_EN)

# First-person (boost precision)
FIRST_PERSON_RE = kw_regex({
    "ben", "biz", "buradayım", "ailem", "annem", "babam", "kardeşim", "çocuğum", "eşim",
    "i am", "i'm", "we are", "we're", "my family", "my child", "my children", "my mom", "my dad",
})

# Exclude offers/announcements
OFFER_RE = kw_regex({
    "bağış", "yardım kampanyası", "yardım topluyoruz", "toplama merkezi", "dağıtıyoruz",
    "ulaştırıyoruz", "kabul ediyoruz", "ihtiyaç listesi", "kampanya", "bağış yapın",
    "donation", "fundraiser", "collection center", "drop[- ]off", "we are sending", "we are collecting",
})

# ---------------- Core filtering (TEXT ONLY) ----------------
def help_mask_text_only(text: pd.Series, mode: str = "balanced") -> pd.Series:
    """
    Keep likely 'help needed' tweets based ONLY on the 'text' field.
    Modes:
      - strict:   SOS + locator   OR  (needs + locator + (first-person OR hazard))
      - balanced: strict  OR  (SOS + (first-person OR hazard))  OR  (flood_SOS + (first-person OR locator OR hazard))
      - recall:   (SOS OR flood_SOS)  OR  (needs + (first-person OR hazard OR locator))
                  (locators optional for SOS/flood_SOS)
    Always excludes obvious offers; RTs excluded unless ALLOW_RTS_IF_MATCH and they match.
    """
    s = text.fillna("").astype(str).str.casefold()

    m_sos    = s.str.contains(SOS_RE, na=False)
    m_flood  = s.str.contains(FLOOD_SOS_RE, na=False)
    m_need   = s.str.contains(NEEDS_RE, na=False)
    m_haz    = s.str.contains(HAZARDS_RE, na=False)
    m_fp     = s.str.contains(FIRST_PERSON_RE, na=False)

    m_phone  = s.str.contains(PHONE_RE, na=False)
    m_coord  = s.str.contains(COORD_RE, na=False)
    m_addr   = s.str.contains(ADDRESS_RE, na=False)
    m_locw   = s.str.contains(LOC_WORDS_RE, na=False)
    locator  = m_phone | m_coord | m_addr | m_locw

    m_offer  = s.str.contains(OFFER_RE, na=False)
    m_rt     = s.str.contains(RETWEET_RE, na=False)
    url_cnt  = s.str.count(URL_RE)

    # URL spam guard
    non_url_signal = (s.str.len() - url_cnt.fillna(0) * 8) > 15

    if mode == "strict":
        core_A = m_sos & locator
        core_B = m_need & locator & (m_fp | m_haz)
        base = core_A | core_B

    elif mode == "recall":
        base = (m_sos | m_flood) | (m_need & (m_fp | m_haz | locator))

    else:  # balanced
        core_A = m_sos & locator
        core_B = m_need & locator & (m_fp | m_haz)
        soft_A = m_sos & (m_fp | m_haz)                  # allow SOS without locator if 1st-person or hazard
        soft_F = m_flood & (m_fp | locator | m_haz)      # flood cues with lighter requirement
        base = core_A | core_B | soft_A | soft_F

    # RT policy
    if ALLOW_RTS_IF_MATCH:
        keep_rt = m_rt & base
        not_rt  = ~m_rt
        rt_filter = keep_rt | not_rt
    else:
        rt_filter = ~m_rt

    return base & rt_filter & ~m_offer & non_url_signal

# ---------------- IO utils ----------------
def iter_tsv_files(dir_path: Path):
    # Only top-level files in crisis_data; change to rglob("*.tsv") if nested
    for p in sorted(dir_path.glob("*.tsv")):
        if p.is_file():
            yield p

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Required columns 'id' and 'text' are missing.")
    # keep all columns, but ensure string types where needed
    df["id"] = df["id"].astype("string")
    df["text"] = df["text"].astype("string")
    return df

def process_file(file_path: Path, first_write: bool, seen_keys: set[str], out_csv: Path) -> tuple[bool, int]:
    wrote = False
    kept  = 0
    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        usecols=[c for c in USECOLS if c in pd.read_csv(file_path, sep="\t", nrows=0).columns],
        dtype={"id": "string"},
        chunksize=CHUNKSIZE,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip"
    ):
        chunk = normalize_columns(chunk)

        mask = help_mask_text_only(chunk["text"], MODE)
        filt = chunk[mask].copy()
        if filt.empty:
            continue

        # Deduplicate by id (fallback to normalized text if id missing)
        keep_idx = []
        for i, tid, txt in zip(filt.index, filt.get("id", ""), filt["text"]):
            key = (tid or "").strip()
            if not key:
                key = re.sub(r"\s+", " ", (txt or "")).strip().casefold()
            if key not in seen_keys:
                seen_keys.add(key)
                keep_idx.append(i)
        filt = filt.loc[keep_idx]
        if filt.empty:
            continue

        # Light debug flags (optional; remove if you want only original columns)
        s = filt["text"].fillna("").astype(str).str.casefold()
        filt["_sos"]   = s.str.contains(SOS_RE, na=False)
        filt["_flood"] = s.str.contains(FLOOD_SOS_RE, na=False)
        filt["_need"]  = s.str.contains(NEEDS_RE, na=False)
        filt["_haz"]   = s.str.contains(HAZARDS_RE, na=False)

        filt.to_csv(
            out_csv,
            index=False,
            mode="w" if first_write and not wrote else "a",
            header=first_write and not wrote,
        )
        kept  += len(filt)
        wrote = True
    return wrote, kept

# ---------------- Main ----------------
def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_tsv_files(INPUT_DIR))
    if not files:
        raise FileNotFoundError(f"No .tsv files found under {INPUT_DIR}")

    seen_keys: set[str] = set()
    first_write = True
    scanned = matched_files = kept_rows = 0

    for f in files:
        wrote, kept = process_file(f, first_write, seen_keys, OUTPUT_CSV)
        scanned += 1
        kept_rows += kept
        if wrote:
            matched_files += 1
            first_write = False
        print(f"Processed: {f} | kept: {kept}")

    print(f"Done. Files scanned: {scanned} | Files with matches: {matched_files} | Kept rows: {kept_rows} -> {OUTPUT_CSV}")
    print(f"Mode: {MODE} | RTs allowed if match: {ALLOW_RTS_IF_MATCH}")

if __name__ == "__main__":
    main()
