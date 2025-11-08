import os, re, io, time, math, textwrap
from datetime import date
from dateutil import parser
from typing import List, Tuple
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# CONFIG & PATHS
# =========================
DATA_DIR = "data"
REPORTS_DIR = "reports"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

WORLD_CUP_YEARS = [1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
RANDOM_SEED = 42  # for reproducible sampling in "user favourites"
REQUEST_DELAY_SEC = 0.8

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CSE-ML-Assignment (Educational, non-commercial)"})


def _get(url, **kwargs):
    """GET with retry + polite delay."""
    for i in range(3):
        try:
            r = SESSION.get(url, timeout=30, **kwargs)
            if r.status_code == 200:
                time.sleep(REQUEST_DELAY_SEC)
                return r
        except requests.RequestException:
            pass
        time.sleep(1 + i)
    raise RuntimeError(f"Failed to fetch: {url}")


def normalize_country(s: str) -> str:
    if pd.isna(s):
        return s
    return (str(s)
            .replace("USA", "United States")
            .replace("IR Iran", "Iran")
            .replace("Korea Republic", "South Korea")
            .replace("Korea DPR", "North Korea")
            .strip())


# =========================
# SCRAPER A: FBref match logs (best effort)
# =========================
def fbref_world_cup_matches(year: int) -> pd.DataFrame:
    candidates = [
        f"https://fbref.com/en/matches/world-cup-{year}",
        f"https://fbref.com/en/comps/1/{year}-matchlogs"
    ]
    html = None
    for url in candidates:
        try:
            r = _get(url)
            if r.status_code == 200 and ("World Cup" in r.text or "world cup" in r.text.lower()):
                html = r.text
                break
        except Exception:
            continue
    if html is None:
        return pd.DataFrame(columns=["date","stage","home","away","score","home_goals","away_goals","year","home_result"])

    soup = BeautifulSoup(html, "lxml")
    # Unwrap comment-embedded tables
    for c in soup.find_all(string=lambda t: isinstance(t, str) and "<table" in t):
        try:
            frag = BeautifulSoup(c, "lxml")
            for tbl in frag.find_all("table"):
                soup.append(tbl)
        except Exception:
            pass

    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame(columns=["date","stage","home","away","score","home_goals","away_goals","year","home_result"])

    # Pick largest table containing a Date header
    best_tbl, best_len = None, 0
    for tbl in tables:
        ths = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        if any("date" in th for th in ths):
            rows = tbl.find_all("tr")
            if len(rows) > best_len:
                best_tbl, best_len = tbl, len(rows)
    if best_tbl is None:
        return pd.DataFrame(columns=["date","stage","home","away","score","home_goals","away_goals","year","home_result"])

    try:
        df = pd.read_html(io.StringIO(str(best_tbl)))[0]
    except Exception:
        return pd.DataFrame(columns=["date","stage","home","away","score","home_goals","away_goals","year","home_result"])

    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "date" in lc and "updated" not in lc: colmap[c] = "date"
        elif "round" in lc or "stage" in lc: colmap[c] = "stage"
        elif "home" in lc or "squad 1" in lc or "team 1" in lc: colmap[c] = "home"
        elif "away" in lc or "squad 2" in lc or "team 2" in lc: colmap[c] = "away"
        elif "score" in lc or "result" in lc: colmap[c] = "score"
    df = df.rename(columns=colmap)
    keep = [c for c in ["date","stage","home","away","score"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["date","stage","home","away","score","home_goals","away_goals","year","home_result"])
    df = df[keep].copy()

    def parse_score(s):
        if isinstance(s, str) and re.search(r"\d+\s*[–-]\s*\d+", s):
            a, b = re.split(r"[–-]", s)
            try:
                return int(a.strip()), int(b.strip())
            except:
                return np.nan, np.nan
        return np.nan, np.nan
    df["home_goals"], df["away_goals"] = zip(*df["score"].apply(parse_score))

    def parse_date_safe(x):
        try: return parser.parse(str(x)).date()
        except: return pd.NaT
    if "date" in df.columns:
        df["date"] = df["date"].apply(parse_date_safe)
    df["year"] = year

    def home_result(r):
        if pd.isna(r["home_goals"]) or pd.isna(r["away_goals"]): return np.nan
        if r["home_goals"] > r["away_goals"]: return "H"
        if r["home_goals"] < r["away_goals"]: return "A"
        return "D"
    df["home_result"] = df.apply(home_result, axis=1)
    return df


# =========================
# SCRAPER B: Wikipedia matches (reliable fallback)
# =========================
def wikipedia_world_cup_matches(year: int) -> pd.DataFrame:
    url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup"
    try:
        r = _get(url)
    except Exception:
        return pd.DataFrame(columns=["date","stage","home","away","score","home_goals","away_goals","year","home_result"])
    soup = BeautifulSoup(r.text, "lxml")

    # Nearest stage header (h2/h3)
    def nearest_stage(node):
        cur = node
        while cur:
            cur = cur.find_previous()
            if cur and cur.name in ("h2","h3"):
                span = cur.find("span", {"class":"mw-headline"})
                if span:
                    txt = span.get_text(" ", strip=True)
                    if any(k in txt.lower() for k in ["group", "round", "quarter", "semi", "final"]):
                        return txt
        return None

    rows = []
    boxes = soup.select(".footballbox")
    if not boxes:
        boxes = soup.select("table.wikitable")

    for box in boxes:
        text = box.get_text(" ", strip=True)
        m = re.search(r"([A-Z][A-Za-z\s\.\-()]+)\s+(\d+)\s*[–-]\s*(\d+)\s+([A-Z][A-Za-z\s\.\-()]+)", text)
        if not m:
            continue
        home, hg, ag, away = m.group(1).strip(), int(m.group(2)), int(m.group(3)), m.group(4).strip()
        dmatch = re.search(r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", text)
        dt = None
        if dmatch:
            try: dt = parser.parse(dmatch.group(1)).date()
            except: dt = pd.NaT
        stage = nearest_stage(box)
        rows.append({
            "date": dt, "stage": stage, "home": home, "away": away,
            "score": f"{hg}–{ag}", "home_goals": hg, "away_goals": ag,
            "year": year, "home_result": "H" if hg>ag else ("A" if hg<ag else "D")
        })
    return pd.DataFrame(rows)


# =========================
# Build match history with fallback & offline sample
# =========================
def build_match_history(years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_matches, finalists = [], []
    for y in tqdm(years, desc="Scraping matches"):
        df = pd.DataFrame()
        try:
            df = fbref_world_cup_matches(y)
        except Exception as e:
            print(f"[FBref WARN {y}] {e}")
        if df.empty or df["home"].dropna().nunique() < 8:
            try:
                df_w = wikipedia_world_cup_matches(y)
                if not df_w.empty:
                    df = df_w
            except Exception as e:
                print(f"[Wikipedia WARN {y}] {e}")

        if not df.empty:
            all_matches.append(df)
            finals_mask = df["stage"].astype(str).str.contains("Final", case=False, na=False)
            finals = df[finals_mask]
            if finals.empty:
                finals = df.sort_values("date").tail(1)
            for _, r in finals.iterrows():
                if pd.notna(r.get("home")): finalists.append({"year": y, "team": r["home"]})
                if pd.notna(r.get("away")): finalists.append({"year": y, "team": r["away"]})

    if not all_matches:
        print("[INFO] Using offline sample (network blocked).")
        matches = pd.DataFrame([
            {"date": date(2018,7,15), "stage":"Final", "home":"France", "away":"Croatia", "score":"4–2",
             "home_goals":4, "away_goals":2, "year":2018, "home_result":"H"},
            {"date": date(2022,12,18), "stage":"Final", "home":"Argentina", "away":"France", "score":"3–3",
             "home_goals":3, "away_goals":3, "year":2022, "home_result":"D"},
        ])
        fins = pd.DataFrame([
            {"year":2018,"team":"France"},{"year":2018,"team":"Croatia"},
            {"year":2022,"team":"Argentina"},{"year":2022,"team":"France"},
        ])
        return matches, fins

    matches = pd.concat(all_matches, ignore_index=True)
    finals_df = pd.DataFrame(finalists).drop_duplicates()
    return matches, finals_df


# =========================
# FIFA rankings snapshot (Wikipedia) — used for features and Top-100
# =========================
WC_START_DATES = {
    1994: date(1994, 6, 17),
    1998: date(1998, 6, 10),
    2002: date(2002, 5, 31),
    2006: date(2006, 6, 9),
    2010: date(2010, 6, 11),
    2014: date(2014, 6, 12),
    2018: date(2018, 6, 14),
    2022: date(2022, 11, 20),
}

def wikipedia_fifa_rankings_snapshot(target_date: date) -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"
    r = _get(url)
    soup = BeautifulSoup(r.text, "lxml")
    tbl = None
    for table in soup.find_all("table", {"class":"wikitable"}):
        ths = [th.get_text(strip=True) for th in table.find_all("th")]
        if any("Rank" in th for th in ths) and any(("Team" in th) or ("Country" in th) or ("Nation" in th) for th in ths):
            tbl = table; break
    if tbl is None:
        return pd.DataFrame(columns=["rank","team_norm","points","snapshot_date"])
    df = pd.read_html(io.StringIO(str(tbl)))[0]

    # Flexible column detection
    rank_col = None; name_col = None; points_col = None
    for c in df.columns:
        lc = str(c).lower()
        if rank_col is None and "rank" in lc and "avg" not in lc: rank_col = c
        if name_col is None and any(k in lc for k in ["team","country","nation"]): name_col = c
        if points_col is None and "points" in lc: points_col = c
    if name_col is None:
        return pd.DataFrame(columns=["rank","team_norm","points","snapshot_date"])

    out = pd.DataFrame()
    out["rank"] = pd.to_numeric(df[rank_col], errors="coerce") if rank_col is not None else np.nan
    out["team_norm"] = (df[name_col].astype(str).str.replace(r"\s+\(.*?\)","",regex=True).str.strip())
    out["points"] = pd.to_numeric(df[points_col], errors="coerce") if points_col is not None else np.nan
    out["snapshot_date"] = pd.to_datetime(target_date)
    return out


def scrape_and_save_top100(path=os.path.join(DATA_DIR, "fifa_top100.csv")) -> pd.DataFrame:
    """Scrape current FIFA table and save Top-100 snapshot."""
    snap = wikipedia_fifa_rankings_snapshot(date.today())
    if snap.empty:
        # minimal fallback if site unavailable
        pd.DataFrame(columns=["team_norm","fifa_rank"]).to_csv(path, index=False)
        return pd.read_csv(path)
    snap = snap.dropna(subset=["rank","team_norm"]).sort_values("rank")
    snap["team_norm"] = snap["team_norm"].apply(normalize_country)
    top100 = snap.head(100)[["team_norm","rank"]].rename(columns={"rank":"fifa_rank"})
    top100.to_csv(path, index=False)
    return top100


# =========================
# Squad pages → average age per team
# =========================
def wikipedia_squad_ages(year: int) -> pd.DataFrame:
    url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup_squads"
    r = _get(url)
    soup = BeautifulSoup(r.text, "lxml")
    tables = soup.find_all("table", {"class":"wikitable"})
    if not tables:
        return pd.DataFrame(columns=["year","team","avg_age_years","players_count"])

    rows = []
    def closest_h2_before(el):
        prev = el
        while prev:
            prev = prev.find_previous()
            if prev and prev.name in ("h2","h3"):
                span = prev.find("span", {"class":"mw-headline"})
                if span: return span.get_text(" ", strip=True)
        return None

    for tbl in tables:
        team = closest_h2_before(tbl)
        if not team or "Group" in team:
            continue
        try:
            df = pd.read_html(io.StringIO(str(tbl)))[0]
        except:
            continue
        dob_col = None
        for c in df.columns:
            if any(k in str(c).lower() for k in ["date of birth","dob","born"]):
                dob_col = c; break
        if dob_col is None:
            continue
        start = WC_START_DATES.get(year, date(year,6,1))
        ages = []
        for val in df[dob_col].astype(str):
            m = re.search(r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})|(\d{4}-\d{2}-\d{2})", val)
            if not m: continue
            try:
                b = parser.parse(m.group(0)).date()
                age = (start - b).days/365.25
                if 15 <= age <= 45: ages.append(age)
            except: pass
        if ages:
            clean_team = (re.sub(r'\s+\(.*?\)','',team)
                          .replace(" national football team","")
                          .strip())
            rows.append({"year":year,"team":clean_team,
                         "avg_age_years":float(np.mean(ages)),"players_count":len(ages)})
    return pd.DataFrame(rows).drop_duplicates(subset=["year","team"])


# =========================
# Feature engineering
# =========================
def team_features_from_matches(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["year","home","home_goals","away_goals"]].copy()
    home["team"] = home["home"]; home["gf"]=home["home_goals"]; home["ga"]=home["away_goals"]
    home["result"] = np.where(home["gf"]>home["ga"],"W", np.where(home["gf"]<home["ga"],"L","D"))
    home = home[["year","team","gf","ga","result"]]

    away = matches[["year","away","home_goals","away_goals"]].copy()
    away["team"] = away["away"]; away["gf"]=away["away_goals"]; away["ga"]=away["home_goals"]
    away["result"] = np.where(away["gf"]>away["ga"],"W", np.where(away["gf"]<away["ga"],"L","D"))
    away = away[["year","team","gf","ga","result"]]

    long = pd.concat([home, away], ignore_index=True)
    agg = (long.groupby(["year","team"])
                .agg(matches_played=("result","count"),
                     wins=("result", lambda s:(s=="W").sum()),
                     draws=("result", lambda s:(s=="D").sum()),
                     losses=("result", lambda s:(s=="L").sum()),
                     goals_for=("gf","sum"),
                     goals_against=("ga","sum"))
                .reset_index())
    agg["goal_diff"] = agg["goals_for"] - agg["goals_against"]
    agg["win_rate"] = agg["wins"] / agg["matches_played"].replace(0,np.nan)
    agg["avg_goal_diff_per_match"] = agg["goal_diff"] / agg["matches_played"].replace(0,np.nan)
    return agg


# =========================
# Week-1 pipeline
# =========================
def run_week1_pipeline(years: List[int] = WORLD_CUP_YEARS,
                       out_csv: str = os.path.join(DATA_DIR,"cleaned_fifa_dataset.csv"),
                       report_md: str = os.path.join(REPORTS_DIR,"week1_data_description.md")) -> pd.DataFrame:
    print("Step 1/6: Scraping matches (FBref → Wikipedia fallback) …")
    matches, finalists = build_match_history(years)
    if matches.empty:
        raise RuntimeError("No matches available (even offline sample failed).")
    print(f" - Matches: {len(matches)} rows; finals entries: {len(finalists)}")

    print("Step 2/6: Team-level features …")
    team_feats = team_features_from_matches(matches)

    print("Step 3/6: FIFA rankings snapshots …")
    ranking_frames = []
    for y in years:
        snap = WC_START_DATES.get(y, date(y,6,1))
        try:
            rnk = wikipedia_fifa_rankings_snapshot(snap); rnk["year"]=y
            ranking_frames.append(rnk)
        except Exception as e:
            print(f"[WARN] Rankings {y}: {e}")
    fifa_rankings = (pd.concat(ranking_frames, ignore_index=True)
                     if ranking_frames else pd.DataFrame(columns=["rank","team_norm","points","snapshot_date","year"]))

    print("Step 4/6: Squad average ages …")
    squad_frames = []
    for y in years:
        try:
            sq = wikipedia_squad_ages(y); squad_frames.append(sq)
        except Exception as e:
            print(f"[WARN] Squads {y}: {e}")
    squad_age = (pd.concat(squad_frames, ignore_index=True)
                 if squad_frames else pd.DataFrame(columns=["year","team","avg_age_years","players_count"]))

    print("Step 5/6: Merge features …")
    team_feats["team_norm"] = (team_feats["team"].apply(normalize_country)
                               .str.replace(r"\s+\(.*?\)","",regex=True))

    # Rankings: ensure team_norm exists
    if fifa_rankings is None or fifa_rankings.empty:
        fifa_rankings = pd.DataFrame(columns=["year","team_norm","rank"])
    else:
        if "team_norm" not in fifa_rankings.columns:
            name_col = None
            for cand in ["country","Team","team","nation","Country","Nation","team_norm"]:
                if cand in fifa_rankings.columns:
                    name_col = cand; break
            if name_col is not None:
                fifa_rankings["team_norm"] = fifa_rankings[name_col].astype(str).apply(normalize_country)
            else:
                fifa_rankings["team_norm"] = np.nan
        if "rank" not in fifa_rankings.columns:
            fifa_rankings["rank"] = np.nan

    # Squad ages
    if squad_age is None or squad_age.empty:
        squad_age = pd.DataFrame(columns=["year","team_norm","avg_age_years"])
    else:
        if "team_norm" not in squad_age.columns:
            squad_age["team_norm"] = squad_age["team"].apply(normalize_country)

    rnk_min = (fifa_rankings[["year","team_norm","rank"]]
               .dropna(subset=["team_norm"])
               .drop_duplicates(["year","team_norm"]))
    age_min = (squad_age[["year","team_norm","avg_age_years"]]
               .drop_duplicates(["year","team_norm"]))

    merged = team_feats.merge(rnk_min, on=["year","team_norm"], how="left") \
                       .merge(age_min, on=["year","team_norm"], how="left")

    # Label: finalist
    if not finalists.empty:
        finalists = finalists.copy()
        finalists["team_norm"] = finalists["team"].apply(normalize_country)
        lab = finalists.drop_duplicates(["year","team_norm"]).assign(finalist=1)[["year","team_norm","finalist"]]
        merged = merged.merge(lab, on=["year","team_norm"], how="left")
    merged["finalist"] = merged["finalist"].fillna(0).astype(int)

    merged = merged.rename(columns={
        "team":"team_original",
        "rank":"fifa_rank",
        "avg_age_years":"team_avg_age"
    }).sort_values(["year","team_norm"]).reset_index(drop=True)

    # Coerce numerics
    for c in ["matches_played","wins","draws","losses","goals_for","goals_against",
              "goal_diff","win_rate","avg_goal_diff_per_match","fifa_rank","team_avg_age"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    print("Step 6/6: Save outputs & report …")
    merged.to_csv(out_csv, index=False)

    col_docs = {
        "year":"World Cup edition year",
        "team_norm":"Normalized team name",
        "team_original":"Team name from match logs",
        "matches_played":"Matches played that year",
        "wins":"Wins", "draws":"Draws", "losses":"Losses",
        "goals_for":"Goals scored", "goals_against":"Goals conceded",
        "goal_diff":"goals_for - goals_against",
        "win_rate":"wins / matches_played",
        "avg_goal_diff_per_match":"goal_diff / matches_played",
        "fifa_rank":"FIFA rank snapshot near tournament start (lower is better)",
        "team_avg_age":"Average squad age at tournament start (years)",
        "finalist":"1 if reached Final, else 0"
    }

    sources = [
        "- **FBref** (matches, where available): https://fbref.com/en/",
        "- **Wikipedia** tournament pages (matches): https://en.wikipedia.org/wiki/{YEAR}_FIFA_World_Cup",
        "- **Wikipedia** FIFA rankings: https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking",
        "- **Wikipedia** squads: https://en.wikipedia.org/wiki/{YEAR}_FIFA_World_Cup_squads",
    ]

    report = f"""# Week 1 Data Description & Scraper Documentation

**Outputs**
- Cleaned dataset: `{out_csv}`
- This report: `{report_md}`

## Data Sources
{chr(10).join(sources)}

## Scraper Strategy
- Try **FBref** match logs; if structure unparseable or blocked, **fallback** to Wikipedia **footballbox** entries.
- FIFA **rankings** and **squad ages** from Wikipedia snapshot pages.
- Robust name normalization and NaN-safe aggregations.

## Cleaning & Feature Engineering
- Parsed score strings → numeric goals; computed team aggregates:
  matches_played, wins, draws, losses, goals_for, goals_against, goal_diff,
  win_rate, avg_goal_diff_per_match.
- Joined FIFA rank snapshot and average squad age per team/year.

## Target Definition
- `finalist` = 1 for teams in matches labeled “Final” (or last match by date as backup).

## Data Dictionary
""" + "\n".join([f"- **{k}**: {v}" for k,v in col_docs.items()])

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Done. Saved:\n - {out_csv}\n - {report_md}")
    return merged


# =========================
# 48-TEAM 2026: Registry & Options
# =========================
def build_wc_frequency(cleaned_path=os.path.join(DATA_DIR, "cleaned_fifa_dataset.csv")) -> pd.DataFrame:
    """Count appearances in last 5 WCs (e.g., 2006–2022) for weighting."""
    df = pd.read_csv(cleaned_path)
    df["team_norm"] = df["team_norm"].apply(normalize_country)
    last5_years = [2006, 2010, 2014, 2018, 2022]
    last5 = df[df["year"].isin(last5_years)]
    freq = (last5.groupby("team_norm")["year"].nunique()
            .rename("wc_freq_last5").reset_index())
    return freq


def init_registry(registry_path=os.path.join(DATA_DIR,"teams_registry.csv"),
                  cleaned_path=os.path.join(DATA_DIR,"cleaned_fifa_dataset.csv"),
                  qualified_path=os.path.join(DATA_DIR,"qualified_teams.csv"),
                  top100_path=os.path.join(DATA_DIR,"fifa_top100.csv")):
    """Create/refresh the canonical registry from historical teams + FIFA Top-100 + qualified list."""
    if not os.path.exists(cleaned_path):
        print("[init_registry] cleaned dataset not found. Running Week-1 pipeline first…")
        run_week1_pipeline()

    hist = pd.read_csv(cleaned_path)
    hist["team_norm"] = hist["team_norm"].apply(normalize_country)
    teams_hist = pd.DataFrame({"team_norm": sorted(hist["team_norm"].dropna().unique())})

    top100 = scrape_and_save_top100(top100_path)
    top100["team_norm"] = top100["team_norm"].apply(normalize_country)

    reg = teams_hist.merge(top100, on="team_norm", how="outer")
    reg = reg.rename(columns={"fifa_rank":"fifa_rank"}).copy()

    freq = build_wc_frequency(cleaned_path)
    reg = reg.merge(freq, on="team_norm", how="left")
    reg["wc_freq_last5"] = reg["wc_freq_last5"].fillna(0).astype(int)

    reg["qualified_2026"] = 0
    reg["source"] = "seed"
    reg["last_update"] = str(date.today())

    if os.path.exists(qualified_path):
        qual = pd.read_csv(qualified_path, header=None, names=["team"])
        qual["team_norm"] = qual["team"].apply(normalize_country)
        reg.loc[reg["team_norm"].isin(qual["team_norm"]), "qualified_2026"] = 1
        reg.loc[reg["team_norm"].isin(qual["team_norm"]), "source"] = "known"

    reg.to_csv(registry_path, index=False)
    print(f"[init_registry] Created/updated {registry_path}. Qualified now: {reg['qualified_2026'].sum()}.")


def update_registry(registry_path=os.path.join(DATA_DIR,"teams_registry.csv"),
                    qualified_path=os.path.join(DATA_DIR,"qualified_teams.csv")):
    """Mark newly qualified teams as they are announced (dynamic updates)."""
    if not os.path.exists(registry_path):
        init_registry(registry_path=registry_path, qualified_path=qualified_path)

    reg = pd.read_csv(registry_path)
    reg["team_norm"] = reg["team_norm"].apply(normalize_country)

    if not os.path.exists(qualified_path):
        print(f"[update_registry] {qualified_path} not found. Create this file with one team per line.")
        return

    qual = pd.read_csv(qualified_path, header=None, names=["team"])
    qual["team_norm"] = qual["team"].apply(normalize_country)

    reg.loc[reg["team_norm"].isin(qual["team_norm"]), "qualified_2026"] = 1
    reg.loc[reg["team_norm"].isin(qual["team_norm"]), "source"] = "known"
    reg["last_update"] = str(date.today())

    reg.to_csv(registry_path, index=False)
    print(f"[update_registry] Updated. Qualified now: {reg['qualified_2026'].sum()}.")


def fill_assumed_20(registry_path=os.path.join(DATA_DIR,"teams_registry.csv"),
                    top100_path=os.path.join(DATA_DIR,"fifa_top100.csv"),
                    cleaned_path=os.path.join(DATA_DIR,"cleaned_fifa_dataset.csv")):
    """Fill remaining slots up to 48 teams using Top-100 minus known, weighted by last-5 frequencies."""
    if not os.path.exists(registry_path):
        init_registry(registry_path=registry_path)

    reg = pd.read_csv(registry_path)
    reg["team_norm"] = reg["team_norm"].apply(normalize_country)

    if not os.path.exists(top100_path):
        scrape_and_save_top100(top100_path)
    top100 = pd.read_csv(top100_path)
    top100["team_norm"] = top100["team_norm"].apply(normalize_country)

    already = set(reg.loc[reg["qualified_2026"]==1, "team_norm"])

    pool = reg[(reg["team_norm"].isin(top100["team_norm"])) & (~reg["team_norm"].isin(already))].copy()

    if "wc_freq_last5" not in pool.columns:
        freq = build_wc_frequency(cleaned_path)
        pool = pool.merge(freq, on="team_norm", how="left")
    pool["wc_freq_last5"] = pool["wc_freq_last5"].fillna(0)

    need = max(0, 48 - reg["qualified_2026"].sum())
    if need == 0:
        print("[fill_assumed_20] Already at 48 teams.")
        return
    k = min(need, 20)

    pool["weight"] = pool["wc_freq_last5"].astype(float) + 0.25
    np.random.seed(RANDOM_SEED)
    if len(pool) < k:
        chosen = pool["team_norm"].tolist()
    else:
        chosen = pool.sample(n=k, weights="weight", replace=False)["team_norm"].tolist()

    reg.loc[reg["team_norm"].isin(chosen), "qualified_2026"] = 1
    reg.loc[reg["team_norm"].isin(chosen), "source"] = "assumed"
    reg["last_update"] = str(date.today())
    reg.to_csv(registry_path, index=False)
    print(f"[fill_assumed_20] Added {len(chosen)} assumed teams. Qualified now: {reg['qualified_2026'].sum()}.")


def export_current_teams(registry_path=os.path.join(DATA_DIR,"teams_registry.csv"),
                         out_path=os.path.join(DATA_DIR,"teams_2026_current.csv")):
    """Export the current list of qualified teams (known + assumed) for downstream modeling."""
    if not os.path.exists(registry_path):
        print("[export_current_teams] Registry not found. Run init_registry first.")
        return
    reg = pd.read_csv(registry_path)
    picked = reg.loc[reg["qualified_2026"]==1, ["team_norm","source","fifa_rank","wc_freq_last5"]].copy()
    picked = picked.sort_values(by=["source","fifa_rank","team_norm"], na_position="last")
    picked.to_csv(out_path, index=False)
    print(f"[export_current_teams] Saved {len(picked)} teams to {out_path}.")


def print_status(registry_path=os.path.join(DATA_DIR,"teams_registry.csv")):
    if not os.path.exists(registry_path):
        print("[status] Registry not found. Run init_registry first.")
        return
    reg = pd.read_csv(registry_path)
    total = reg["qualified_2026"].sum()
    known = reg[(reg["qualified_2026"]==1) & (reg["source"]=="known")].shape[0]
    assumed = reg[(reg["qualified_2026"]==1) & (reg["source"]=="assumed")].shape[0]
    print(f"[status] Qualified: {total} / 48  (known={known}, assumed={assumed})")
    if total:
        print(reg.loc[reg["qualified_2026"]==1, ["team_norm","source","fifa_rank"]]
                .sort_values(["source","fifa_rank","team_norm"], na_position="last")
                .to_string(index=False))

# =========================
# OPTIONAL QUICK EDA
# =========================
def quick_eda_preview(df: pd.DataFrame):
    if df.empty:
        print("No data for EDA."); return
    print(df.describe(include='all'))
    if "team_avg_age" in df.columns:
        df["team_avg_age"].plot(kind="hist", bins=20, title="Distribution: Team Average Age")
        plt.xlabel("Avg Age (years)"); plt.show()
    if {"win_rate","year"}.issubset(df.columns):
        df.boxplot(column="win_rate", by="year")
        plt.title("Win Rate distribution by World Cup"); plt.suptitle(""); plt.xlabel("Year"); plt.ylabel("Win Rate"); plt.show()

# =========================
# HOW TO RUN (in this notebook)
# =========================
# In new Jupyter notebook cell, type the following code to execute this program:

# 1) Build dataset + report
merged = run_week1_pipeline()
merged.head()  # preview

# 2) Initialize registry (uses data/qualified_teams.csv if present)
init_registry()
print_status()

# 3) If you’ve added newly confirmed teams to data/qualified_teams.csv
update_registry()
print_status()

# 4) Fill remaining slots using “User favourites” logic (Top-100 weighted)
fill_assumed_20()
print_status()

# 5) Export current 48 (known + assumed)
export_current_teams()
