# build_disaster_dataset.py
# -*- coding: utf-8 -*-
import argparse, pandas as pd, re, json, hashlib, unicodedata, os, random
from typing import Any, Dict, List, Tuple

# ---------- Helpers ----------
def norm_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\u200b"," ").replace("\xa0"," ")
    return unicodedata.normalize("NFKC", s).strip()

def dedup_id(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\d+", "<num>", t)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]

# 81 provinces (Turkey)
PROVINCES = [
    "Adana","Adıyaman","Afyonkarahisar","Ağrı","Aksaray","Amasya","Ankara","Antalya","Ardahan","Artvin",
    "Aydın","Balıkesir","Bartın","Batman","Bayburt","Bilecik","Bingöl","Bitlis","Bolu","Burdur","Bursa",
    "Çanakkale","Çankırı","Çorum","Denizli","Diyarbakır","Düzce","Edirne","Elazığ","Erzincan","Erzurum",
    "Eskişehir","Gaziantep","Giresun","Gümüşhane","Hakkari","Hatay","Iğdır","Isparta","İstanbul","İzmir",
    "Kahramanmaraş","Karabük","Karaman","Kars","Kastamonu","Kayseri","Kilis","Kırıkkale","Kırklareli",
    "Kırşehir","Kocaeli","Konya","Kütahya","Malatya","Manisa","Mardin","Mersin","Muğla","Muş","Nevşehir",
    "Niğde","Ordu","Osmaniye","Rize","Sakarya","Samsun","Siirt","Sinop","Sivas","Şanlıurfa","Şırnak","Tekirdağ",
    "Tokat","Trabzon","Tunceli","Uşak","Van","Yalova","Yozgat","Zonguldak"
]
PROVINCE_RE = re.compile(r"\b(" + "|".join([re.escape(p) for p in PROVINCES]) + r")\b", flags=re.IGNORECASE)

# Need dictionaries (Turkish cues)
NEED_PATTERNS = {
    "medical": [
        r"\bnefes alam(ıyor|iyor)\b", r"\b(yaralı|yaralandı|kanama|kan(ıyor|iyor))\b",
        r"\bambulans\b", r"\bdoktor\b", r"\bacil tıbbi\b", r"\b112\b", r"\bkalp\b",
        r"\b(baygın|bayıldı)\b", r"\bilk ?yardım\b", r"\byanık\b", r"\bkırık\b"
    ],
    "search_rescue": [
        r"\benkaz\b", r"\bgöçük\b", r"\byık(ıldı|ik)\b", r"\bsıkıştı\b", r"\bkurtar(ma|ın)\b",
        r"\bkayıp\b", r"\bhaber alınam(ıyor|iyor)\b", r"\bulaşılam(ıyor|iyor)\b"
    ],
    "shelter": [r"\bçadır\b", r"\bbarın(ma|acak)\b", r"\bkalacak yer\b", r"\bkonaklama\b", r"\bkonteyner\b", r"\bsığınak\b"],
    "food": [r"\berzak\b", r"\byemek\b", r"\bgıda\b", r"\bekmek\b", r"\bmama\b", r"\bkuru gıda\b"],
    "water": [r"\biçme suyu\b", r"\bsu (yok|lazım|ihtiyaç|gerekiyor)\b", r"\btemiz su\b"],
    "evacuation": [r"\btahliye\b", r"\bboşalt(ın|ılması)\b", r"\byangın\b", r"\bsel\b", r"\btsunami\b", r"\bheyelan\b"],
    "info": [r"\bbilgi\b", r"\bnerede\b", r"\baçık mı\b", r"\btoplanma alan(ı|ları)\b", r"\bnumara\b", r"\bnasıl ulaş(ır|ırım)\b", r"\bdurum nedir\b"],
}
NEED_ORDER = ["medical","search_rescue","evacuation","shelter","water","food","info","other"]

URGENCY_CRITICAL = [
    r"\bacil\b", r"\bhemen\b", r"\bimdat\b", r"\bnefes alam(ıyor|iyor)\b", r"\bkanama\b",
    r"\benkaz altında\b", r"\byangın\b", r"\bhayati\b", r"\bölmek üzere\b", r"!!!"
]
URGENCY_HIGH = [r"\byardım lazım\b", r"\byetişin\b", r"\bihtiyaç\b", r"\bşu anda\b"]

VULN_MAP = {
    "infant": [r"\bbebek\b", r"\byenidoğan\b"],
    "child": [r"\bçocuk(lar)?\b"],
    "elderly": [r"\byaşlı\b", r"\bteyze\b", r"\bdede\b", r"\bnine\b"],
    "pregnant": [r"\bhamile\b", r"\bgebe\b"],
    "disabled": [r"\bengelli\b", r"\btekerlekli\b", r"\bişitme engelli\b", r"\bgörme engelli\b"]
}

def find_need(text: str):
    hits = []; sel = "other"
    for need in NEED_ORDER:
        if need == "other": continue
        for pat in NEED_PATTERNS.get(need, []):
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                hits.append(m.group(0))
                if sel == "other": sel = need
    return sel, list(dict.fromkeys(hits))

def find_urgency(text: str):
    ev = []
    for pat in URGENCY_CRITICAL:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            ev.append(m.group(0)); return "critical", ev
    for pat in URGENCY_HIGH:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            ev.append(m.group(0)); return "high", ev
    return "normal", ev

def find_vulnerable(text: str):
    groups, spans = [], []
    for tag, pats in VULN_MAP.items():
        for pat in pats:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                groups.append(tag); spans.append(m.group(0)); break
    return groups, spans

def extract_location(text: str) -> Dict[str, Any]:
    province = None; district = None; free_text = None; geo_conf = 0.0
    m = PROVINCE_RE.search(text)
    if m:
        province = m.group(0).title()
        after = text[m.end():]
        cand = re.search(r"[ ,]*([A-ZÇĞİIÖŞÜ][A-Za-zÇĞİIÖŞÜçğıiöşü\-]{3,})", after)
        if cand:
            d = cand.group(1)
            if d.lower() not in {"civarı","yakını","merkez","ilçe","mahalle","mah","sokak","cadde","bulvar","bulv","köy","köyü","ilçesi"}:
                district = d.title()
        start = max(0, m.start()-40); end = min(len(text), m.end()+60)
        free_text = text[start:end].strip()
        geo_conf = 0.8 if (province and district) else 0.5
    else:
        loc_m = re.search(r"([A-ZÇĞİIÖŞÜ][^\n,]{0,60}\b(mah(alle)?|cadde|sokak|bulvar|köy(ü)?)\b[^\n,]{0,40})", text)
        if loc_m:
            free_text = loc_m.group(1); geo_conf = 0.2
    return {"province": province,"district": district,"free_text": free_text,"lat": None,"lon": None,"geo_confidence": round(geo_conf, 2)}

def model_confidence(need_hits: List[str], urg: str, vuln_hits: List[str]) -> float:
    base = 0.4 if need_hits else 0.2
    base += 0.3 if urg == "critical" else (0.2 if urg == "high" else 0.0)
    base += 0.1 if vuln_hits else 0.0
    return round(min(0.95, base), 2)

def build_record(text: str) -> Dict[str, Any]:
    t = norm_text(text)
    need, need_ev = find_need(t)
    urg, urg_ev = find_urgency(t)
    vuln, vuln_ev = find_vulnerable(t)
    loc = extract_location(t)
    ev_spans = list(dict.fromkeys(need_ev + urg_ev + vuln_ev))
    return {
        "text": t,
        "need": need,
        "urgency": urg,
        "location": loc,
        "vulnerable_groups": vuln,
        "evidence_spans": ev_spans,
        "model_confidence": model_confidence(need_ev, urg, vuln_ev),
        "dedup_cluster_id": dedup_id(t),
        "recommendations": {"nearest_assembly_points": [], "emergency_number": "112", "notes": None}
    }

# ---------- CSV loading ----------
def load_csv(path: str, text_col: str = None) -> Tuple[pd.DataFrame, str]:
    encodings = [None, "utf-8-sig", "cp1254", "iso-8859-9", "latin-5", "windows-1254"]
    last_err = None; df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", engine="python"); break
        except Exception as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"CSV could not be read: {last_err}")
    if text_col and text_col in df.columns:
        col = text_col
    else:
        cands = [c for c in df.columns if c.lower() in ["content","text","message","tweet","body","mesaj","icerik","içerik"]]
        col = cands[0] if cands else next((c for c in df.columns if df[c].dtype == "object"), None)
        if not col:
            raise RuntimeError("No text-like column found. Pass --text-col.")
    df[col] = df[col].map(norm_text)
    return df, col

# ---------- Packing (USER/ASSISTANT pairs) ----------
def strip_text(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k != "text"}

def to_user_assistant(text: str, rec_no_text: Dict[str, Any]) -> Dict[str, Any]:
    return {"user": text, "assistant": rec_no_text}

# ---------- Cluster split ----------
def cluster_split(items: List[Dict[str, Any]], train: float, val: float, test: float, seed: int = 42):
    assert abs((train + val + test) - 1.0) < 1e-6
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        cid = it.get("_cluster_id")
        buckets.setdefault(cid, []).append(it)
    keys = list(buckets.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = int(round(train*n))
    n_val = int(round(val*n))
    train_k = set(keys[:n_train])
    val_k = set(keys[n_train:n_train+n_val])
    test_k = set(keys[n_train+n_val:])
    def gather(K):
        out = []
        for k in K:
            out.extend(buckets[k])
        return out
    return gather(train_k), gather(val_k), gather(test_k)

def write_json_array(path: str, items: List[Dict[str, Any]], pretty: bool):
    cleaned = [{k:v for k,v in it.items() if k != "_cluster_id"} for it in items]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2 if pretty else None)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Build disaster dataset (JSON array with user/assistant keys)")
    ap.add_argument("--input", default="data/help_tweets_strict.csv", help="Input CSV path")
    ap.add_argument("--output", default="disaster_template_dataset.json", help="Output JSON path (or basename if splitting)")
    ap.add_argument("--text-col", default=None, help="Name of text column (e.g., content)")
    ap.add_argument("--max-rows", type=int, default=None, help="Process first N rows")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    ap.add_argument("--train-ratio", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if round(args.train_ratio + args.val_ratio + args.test_ratio, 6) != 1.0:
        raise SystemExit("splits must sum to 1.0")

    df, text_col = load_csv(args.input, args.text_col)
    if args.max_rows:
        df = df.head(args.max_rows)

    # Build records → user/assistant pairs
    items = []
    for _, row in df.iterrows():
        rec = build_record(row.get(text_col, ""))
        pair = to_user_assistant(rec["text"], strip_text(rec))
        pair["_cluster_id"] = rec["dedup_cluster_id"]  # sidecar for split only
        items.append(pair)

    # Write JSON array(s)
    if args.val_ratio > 0.0 or args.test_ratio > 0.0:
        train, val, test = cluster_split(items, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
        base, ext = os.path.splitext(args.output)
        out_train = base + ".train" + (ext if ext else ".json")
        out_val   = base + ".val"   + (ext if ext else ".json")
        out_test  = base + ".test"  + (ext if ext else ".json")
        write_json_array(out_train, train, args.pretty)
        write_json_array(out_val,   val,   args.pretty)
        write_json_array(out_test,  test,  args.pretty)
        print(f"Wrote {len(train)} → {out_train}")
        print(f"Wrote {len(val)}   → {out_val}")
        print(f"Wrote {len(test)}  → {out_test}")
    else:
        write_json_array(args.output, items, args.pretty)
        print(f"Wrote {len(items)} → {args.output}")

    # Sample
    if items:
        sample = {k:v for k,v in items[0].items() if k != "_cluster_id"}
        print("Sample:")
        print(json.dumps(sample, ensure_ascii=False, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
