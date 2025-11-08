# #######Overview of this program ########
#
# Scrapes the 28 already-qualified teams (Wikipedia, with fallback).
#
# Builds the 48-team pool by adding 20 teams from FIFA Top-100 (excluding already-    #    qualified), biased by last 5 World Cups appearances.

# Reuses your trained model from Week-3 (reads model_path & feature list from reports/#  week3_eval_summary.json).

# Creates the feature matrix using your data/cleaned_fifa_dataset.csv (uses latest     # available year per team, fills missing with column medians).
#
# Outputs:
#
# reports/week4_candidate_pool.csv (qualified vs assumed)
#
# reports/week4_predictions.csv (all 48 teams with probabilities)
#
# reports/week4_summary.json (finalists, pool info, reproducibility seed)
#
#Prints a tidy summary (top-8, top-4, finalists)
#############

# ============================================
# WEEK-4 FINAL PREDICTION 
# ============================================

# ============================================================
# WEEK-4 FINAL PREDICTION 
# ============================================================
# What this does (with detailed logs):
#  1) Scrapes qualified teams (Wikipedia) with status logs; shows why it might be empty
#  2) If scraping yields <2 teams, falls back to:
#       a) qualified_teams_manual.txt (if present), else
#       b) Derive a 48-team pool from your own dataset (cleaned_fifa_dataset.csv)
#  3) Builds features EXACTLY as your fitted pipeline expects (strict feature-name/order)
#     - Logs expected features from the model/imputer
#     - Adds any missing feature columns and logs them
#     - Logs shapes before predict_proba
#  4) Predicts finalist probabilities and saves CSV/JSON
#
# REQUIREMENTS:
#   - data/cleaned_fifa_dataset.csv
#   - reports/week3_eval_summary.json  (contains model_path; features optional)
#   - models/<your_best_model>.pkl
#
# OUTPUTS (in reports/):
#   - week4_candidate_pool.csv
#   - week4_predictions.csv
#   - week4_summary.json
#
# TIP: If your network blocks Wikipedia, create 'qualified_teams_manual.txt'
#      (one team per line) to label real qualified teams in the 'source' column.

import sys, subprocess, json, re, random, warnings, textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# --- Try to import web deps (install if missing, with logs) ---
def _ensure_web_deps():
    try:
        import requests  # noqa
        from bs4 import BeautifulSoup  # noqa
        return True, "requests/bs4 already available"
    except Exception as e:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "beautifulsoup4"])
            return True, "Installed requests + beautifulsoup4"
        except Exception as ee:
            return False, f"Failed to install requests/bs4: {ee}"

ok_deps, dep_msg = _ensure_web_deps()
print(f"[DEBUG] Web-deps status: {dep_msg}")
if ok_deps:
    import requests
    from bs4 import BeautifulSoup

# ----------------
# Paths & constants
# ----------------
DATA_DIR    = Path("data")
REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")
for p in [DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CLEAN_CSV = DATA_DIR / "cleaned_fifa_dataset.csv"
assert CLEAN_CSV.exists(), "Missing data/cleaned_fifa_dataset.csv."

W3_JSON = REPORTS_DIR / "week3_eval_summary.json"
assert W3_JSON.exists(), "Missing reports/week3_eval_summary.json."
W3 = json.loads(W3_JSON.read_text(encoding="utf-8"))

MODEL_PATH = W3.get("model_path")
assert MODEL_PATH and Path(MODEL_PATH).exists(), f"Model not found at {MODEL_PATH}"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------
# Helper functions
# ----------------
def normalize_team(name: str) -> str:
    if not isinstance(name, str): return name
    s = name.strip()
    reps = {
        "United States of America":"USA","United States":"USA",
        "Korea Republic":"South Korea","Republic of Korea":"South Korea",
        "IR Iran":"Iran","Côte d’Ivoire":"Ivory Coast","Cote d'Ivoire":"Ivory Coast",
        "Türkiye":"Turkey","UAE":"United Arab Emirates","DR Congo":"Congo DR","Congo DR":"DR Congo",
    }
    return reps.get(s, s)

def _extract_countryish_text(soup):
    out = []
    for a in soup.find_all("a"):
        t = a.get_text(" ", strip=True)
        if re.match(r"^[A-Za-z .’'()-]{3,}$", t) and len(t) <= 30:
            out.append(t)
    return out

def _try_get(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        return True, r
    except Exception as e:
        return False, e

def scrape_qualified_teams():
    urls = [
        "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification",
        "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
    ]
    teams = set()
    logs = []
    if not ok_deps:
        logs.append("requests/bs4 not available; skipping web scrape.")
    else:
        for url in urls:
            ok, resp = _try_get(url)
            if not ok:
                logs.append(f"GET failed for {url} -> {resp}")
                continue
            if isinstance(resp, Exception):
                logs.append(f"Unexpected error on {url}: {resp}")
                continue
            if resp.status_code != 200:
                logs.append(f"{url} -> HTTP {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            anchors = []
            for h in soup.find_all(["h2","h3"]):
                if "Qualified teams" in h.get_text():
                    nxt = h.find_next_sibling()
                    for _ in range(6):
                        if nxt is None: break
                        if nxt.name in ("ul","ol"):
                            anchors += [li.get_text(" ", strip=True) for li in nxt.find_all("li")]
                        if nxt.name == "table":
                            anchors += _extract_countryish_text(nxt)
                        nxt = nxt.find_next_sibling()
            if not anchors:
                anchors = _extract_countryish_text(soup)
                logs.append(f"No dedicated 'Qualified teams' block parsed in {url}; using broad scan.")

            for t in anchors:
                t = re.sub(r"\[[^\]]+\]","",t).strip()
                if len(t) >= 3 and re.match(r"^[A-Za-z .’'()-]{3,}$", t):
                    teams.add(normalize_team(t))
            logs.append(f"Parsed ~{len(teams)} unique tokens so far from {url}.")
            if len(teams) >= 20:
                break

        # light filter
        bad = {"UEFA","CONMEBOL","CONCACAF","AFC","CAF","OFC","Group","Round","Play-offs","Host","Hosts","United 2026"}
        teams = [t for t in teams if t not in bad and not re.search(r"\d{4}", t)]

    # Manual override file
    manual = Path("qualified_teams_manual.txt")
    if manual.exists():
        listed = [normalize_team(x.strip()) for x in manual.read_text(encoding="utf-8").splitlines() if x.strip()]
        logs.append(f"Using manual qualified list from {manual} (n={len(listed)}).")
        teams = listed

    teams = sorted(set(teams))
    return teams, logs

def scrape_fifa_top100():
    if not ok_deps:
        return [], ["requests/bs4 not available; skipping FIFA Top-100 scrape."]
    url = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"
    logs = []
    ok, resp = _try_get(url)
    if not ok or isinstance(resp, Exception):
        logs.append(f"GET failed for FIFA ranking page: {resp}")
        return [], logs
    if resp.status_code != 200:
        logs.append(f"FIFA ranking page HTTP {resp.status_code}")
        return [], logs
    soup = BeautifulSoup(resp.text, "html.parser")
    cand = []
    for tbl in soup.find_all("table", {"class":"wikitable"})[:6]:
        cand += _extract_countryish_text(tbl)
    seen, out = set(), []
    for c in cand:
        c = normalize_team(c)
        if c not in seen:
            seen.add(c); out.append(c)
    out = out[:120]
    logs.append(f"Collected ~{len(out)} Top-100-ish candidates.")
    return out, logs

def scrape_wc_teams(year):
    if not ok_deps:
        return []
    ok, resp = _try_get(f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup")
    if not ok or isinstance(resp, Exception) or resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    teams = set(_extract_countryish_text(soup))
    bad = {"UEFA","CONMEBOL","CONCACAF","AFC","CAF","OFC","Group","Host"}
    teams = [normalize_team(t) for t in teams if t not in bad and not re.search(r"\d{4}", t)]
    return sorted(set(teams))

def last5_wc_frequent():
    yrs = [2006, 2010, 2014, 2018, 2022]
    bag = []
    for y in yrs:
        bag += scrape_wc_teams(y)
    s = pd.Series(bag) if bag else pd.Series(dtype=str)
    return list(s.value_counts().index)

# -------------------------
# 1) Build the 48-team pool (with logs & fallbacks)
# -------------------------
qualified, qlogs = scrape_qualified_teams()
print("\n[DEBUG] Scrape Qualified Logs:")
print(textwrap.indent("\n".join(qlogs) if qlogs else "(no logs)", "  "))

print(f"[DEBUG] Qualified teams scraped/found: {len(qualified)}")
if qualified:
    print("  Sample:", qualified[:10])

top100, tlogs = scrape_fifa_top100()
print("\n[DEBUG] FIFA Top-100 Logs:")
print(textwrap.indent("\n".join(tlogs) if tlogs else "(no logs)", "  "))
print(f"[DEBUG] FIFA Top-100 candidates collected: {len(top100)}")
if top100:
    print("  Sample:", top100[:10])

freq = last5_wc_frequent()
print(f"\n[DEBUG] Last-5 WCs frequent list size: {len(freq)}")
if freq:
    print("  Sample:", freq[:10])

pool = set(normalize_team(t) for t in qualified)
need = 48 - len(pool)
remaining = [t for t in top100 if t not in pool]
biased   = sorted(remaining, key=lambda t: (t not in freq, remaining.index(t)))

assumed = []
for t in biased:
    if len(assumed) >= max(0, need): break
    if t not in assumed: assumed.append(t)

if len(assumed) < max(0, need):
    more = [t for t in freq if t not in pool and t not in assumed]
    for t in more:
        if len(assumed) >= need: break
        assumed.append(t)
if len(assumed) < max(0, need):
    for t in remaining:
        if len(assumed) >= need: break
        if t not in assumed: assumed.append(t)

pool_list = sorted(set(list(pool) + assumed[:max(0, need)]))
print(f"\n[DEBUG] Pool after scrape: {len(pool_list)} teams")
if pool_list:
    print("  Sample:", pool_list[:12])

# -------------------------
# HARD FALLBACK if scraping failed (<2 teams)
# -------------------------
df_all = pd.read_csv(CLEAN_CSV)
team_col = "team_norm" if "team_norm" in df_all.columns else ("team" if "team" in df_all.columns else None)
assert team_col, "cleaned_fifa_dataset.csv must include 'team_norm' or 'team'."

def derive_pool_from_dataset(df):
    if "year" in df.columns and (df["year"] == 2022).any():
        df2 = df[df["year"] == 2022].copy()
    else:
        df2 = df.sort_values("year" if "year" in df.columns else team_col, ascending=False)\
                .drop_duplicates(subset=[team_col], keep="first").copy()
    # Composite score using available cols
    score = 0
    if "win_rate" in df2.columns: score += df2["win_rate"].fillna(0)
    if "goal_diff" in df2.columns:
        denom = df2.get("matches_played", 1).replace(0,1)
        score += (df2["goal_diff"].fillna(0) / denom).clip(-10, 10)
    if "matches_played" in df2.columns:
        mx = (df2["matches_played"].max() or 1)
        score += (df2["matches_played"].fillna(0) / mx) * 0.1
    df2["_score"] = score
    top = (
        df2.sort_values("_score", ascending=False)
           .drop_duplicates(subset=[team_col], keep="first")
           .head(48)[team_col]
           .map(normalize_team)
           .tolist()
    )
    return sorted(set(top))

if len(pool_list) < 2:
    print("\n[WARN] Scraping produced <2 teams. Falling back to dataset-derived pool of 48.")
    pool_list = derive_pool_from_dataset(df_all)
    print(f"[DEBUG] Derived pool size from dataset: {len(pool_list)}")
    print("  Sample:", pool_list[:12])

# Final safety
if len(pool_list) < 2:
    pool_list = df_all[team_col].dropna().map(normalize_team).drop_duplicates().head(48).tolist()
    print("\n[WARN] Using basic unique-team fallback.")
    print(f"[DEBUG] Final fallback pool size: {len(pool_list)}")
    print("  Sample:", pool_list[:12])

source_map = {t: ("qualified" if t in set(qualified) else "assumed") for t in pool_list}
pool_df = pd.DataFrame({"team": pool_list, "source": [source_map[t] for t in pool_list]})
pool_csv = REPORTS_DIR / "week4_candidate_pool.csv"
pool_df.to_csv(pool_csv, index=False)
print(f"\n[DEBUG] Saved candidate pool -> {pool_csv}")
print(f"[DEBUG] Source counts: qualified={sum(v=='qualified' for v in source_map.values())}, assumed={sum(v=='assumed' for v in source_map.values())}")

# -------------------------
# 2) Build features exactly as model expects (with logs)
# -------------------------
pipe = joblib.load(MODEL_PATH)
print(f"\n[DEBUG] Loaded model: {MODEL_PATH}")

# Resolve expected features from model/imputer, fallback to JSON
expected = None
imputer = pipe.named_steps.get("imputer") if hasattr(pipe, "named_steps") else None
if imputer is not None and hasattr(imputer, "feature_names_in_"):
    expected = [str(c) for c in imputer.feature_names_in_]
    print("[DEBUG] Expected features from SimpleImputer.feature_names_in_")
elif hasattr(pipe, "feature_names_in_"):
    expected = [str(c) for c in pipe.feature_names_in_]
    print("[DEBUG] Expected features from Pipeline.feature_names_in_")
else:
    expected = [str(c).strip() for c in (W3.get("features_used_after_imputer") or [])]
    print("[DEBUG] Expected features from Week-3 JSON (features_used_after_imputer)")

print(f"[DEBUG] n_expected_features = {len(expected)}")
print("  Expected feature list:", expected)

# Build latest snapshot
if "year" in df_all.columns and (df_all["year"] == 2022).any():
    df_latest = df_all[df_all["year"] == 2022].copy()
    still_needed = set(pool_list) - set(df_latest[team_col].map(normalize_team).tolist())
    if still_needed:
        extra = (df_all.sort_values("year" if "year" in df_all.columns else team_col, ascending=False)
                      .drop_duplicates(subset=[team_col], keep="first"))
        extra = extra[extra[team_col].apply(normalize_team).isin(still_needed)]
        df_latest = pd.concat([df_latest, extra], ignore_index=True)
else:
    df_latest = (df_all.sort_values("year" if "year" in df_all.columns else team_col, ascending=False)
                        .drop_duplicates(subset=[team_col], keep="first"))

# Global medians
global_medians = {}
for f in expected:
    if f in df_all.columns and pd.api.types.is_numeric_dtype(df_all[f]):
        med = df_all[f].median()
        global_medians[f] = float(med) if pd.notna(med) else 0.0
    else:
        global_medians[f] = 0.0
print("[DEBUG] Built global medians for expected features.")

# Build rows
rows, filled_counts = [], {f: 0 for f in expected}
for t in pool_list:
    t_norm = normalize_team(t)
    row = df_latest[df_latest[team_col].apply(normalize_team) == t_norm]
    if row.empty:
        s = pd.Series({f: global_medians[f] for f in expected}, name=t)
        for f in expected: filled_counts[f] += 1
        rows.append(s)
    else:
        src = row.iloc[0]
        s = pd.Series(index=expected, dtype="float64", name=t)
        for f in expected:
            if f in row.columns and pd.notna(src.get(f)):
                s[f] = float(src[f])
            else:
                s[f] = global_medians[f]
                filled_counts[f] += 1
        rows.append(s)

X48 = pd.DataFrame(rows)
X48.index.name = "team"
X48 = X48.reset_index()

missing_cols = [f for f in expected if f not in X48.columns]
if missing_cols:
    for f in missing_cols:
        X48[f] = np.nan
    print(f"[WARN] Added missing feature columns to X48: {missing_cols}")

for f in expected:
    X48[f] = pd.to_numeric(X48[f], errors="coerce").fillna(global_medians[f])

# Exact order & log shape
X48 = X48[["team"] + expected]
print(f"[DEBUG] X48 shape: {X48.shape} (rows, cols including 'team')")
print("[DEBUG] Any NaNs left in X48 (expected cols):", bool(pd.isna(X48[expected]).sum().sum()))

# -------------------------
# 3) Predict & save (with logs)
# -------------------------
try:
    proba = pipe.predict_proba(X48[expected])[:, 1]
except Exception as e:
    print("\n[ERROR] predict_proba failed. Dumping debug info:")
    print("  Exception:", repr(e))
    print("  First 3 rows of X48[expected]:")
    print(X48[expected].head(3))
    raise

pred_df = pd.DataFrame({"team": X48["team"], "proba_finalist": proba})
pred_df = pred_df.merge(pool_df, on="team", how="left").sort_values("proba_finalist", ascending=False).reset_index(drop=True)

pred_csv = REPORTS_DIR / "week4_predictions.csv"
pred_df.to_csv(pred_csv, index=False)

finalists = pred_df.head(2)["team"].tolist()
semis     = pred_df.head(4)["team"].tolist()
quarters  = pred_df.head(8)["team"].tolist()

week4_summary = {
    "finalists_top2": finalists,
    "semifinalists_top4": semis,
    "quarterfinalists_top8": quarters,
    "pool_counts": {
        "qualified": int((pred_df["source"] == "qualified").sum()),
        "assumed":   int((pred_df["source"] == "assumed").sum()),
        "total": int(len(pred_df))
    },
    "reproducibility_seed": RANDOM_SEED,
    "model_path": MODEL_PATH,
    "features_expected_by_model": expected,
    "filled_feature_counts": {k:int(v) for k,v in filled_counts.items()}
}
(REPORTS_DIR / "week4_summary.json").write_text(json.dumps(week4_summary, indent=2), encoding="utf-8")

# -------------------------
# 4) Print summary (clear & compact)
# -------------------------
print("\n==================== WEEK-4 SUMMARY (DEBUG) ====================")
print(f"Model     : {MODEL_PATH}")
print(f"Pool size : {len(pred_df)} "
      f"(qualified={week4_summary['pool_counts']['qualified']}, assumed={week4_summary['pool_counts']['assumed']})")
print(f"Finalists : {finalists}")
print(f"Top-4     : {semis}")
print(f"Top-8     : {quarters}")
print("\nFilled feature counts (fallbacks used per feature):")
print(week4_summary['filled_feature_counts'])
print(f"\nSaved files:\n - {pred_csv}\n - {REPORTS_DIR / 'week4_candidate_pool.csv'}\n - {REPORTS_DIR / 'week4_summary.json'}")
print("===============================================================")
