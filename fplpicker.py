"""
fplpicker.py
Greedy FPL picker + live scraping fallback for new signings.

Dependencies:
- pandas
- numpy
- requests
- beautifulsoup4
- lxml
- html5lib
- cloudscraper
- tqdm
- rapidfuzz
- pulp
"""

from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
from tqdm import tqdm
import pulp

# -- Configuration --
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}
SLEEP_RANGE = (1.0, 2.2)

BUDGET = 100.0
SQUAD_STRUCTURE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
CACHE_FILE = "external_stats_cache.csv"
SEASON = "2025/2026"

LEAGUE_DIFFICULTY = {
    "Premier League": 90.6, "La Liga": 84.8, "Serie A": 84.8, "Bundesliga": 84.2,
    "Ligue 1": 84.3, "Eredivisie": 77.2, "Primeira Liga": 78.8, "Belgian Pro League": 80.0,
    "Championship": 79.5, "Scottish Premiership": 74.0, "Brazil Serie A": 80.8,
    "MLS": 74.8, "Argentine Primera": 77.5, "Russian Premier League": 75.0,
    "Turkish Super Lig": 75.0, "Austrian Bundesliga": 75.0,
    "Swiss Super League": 75.7, "Mexican Liga MX": 73.5, "Japanese J1 League": 70.0
}
PL_BASE = LEAGUE_DIFFICULTY["Premier League"]

def sleep_short():
    time.sleep(random.uniform(*SLEEP_RANGE))

def try_request(url, headers=HEADERS, retries=3, timeout=20):
    for i in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
        except Exception:
            pass
        time.sleep(1 + i * 2)
    return None

def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE, index_col=0)
    return pd.DataFrame(columns=[
        "fpl_name", "source_name", "league", "season", "goals", "assists", "minutes", "notes"
    ])

def save_cache(df):
    df.to_csv(CACHE_FILE)

def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = try_request(url)
    r.raise_for_status()
    data = r.json()
    elements = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    elements["pos"] = elements["element_type"].map(pos_map)
    elements["team"] = elements["team"].map(teams.set_index("id")["name"])
    elements["cost"] = elements["now_cost"] / 10.0
    elements["name"] = elements["first_name"].fillna("") + " " + elements["second_name"].fillna("")
    elements["web_name"] = elements["web_name"]
    return elements

# -- Transfermarkt scraping --
def transfermarkt_search_player(name):
    url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={name.replace(' ', '+')}"
    r = try_request(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    a = soup.find("a", class_="spielprofil_tooltip")
    if a and a.get("href"):
        return "https://www.transfermarkt.com" + a["href"]
    return None

def parse_transfermarkt(profile_url):
    r = try_request(profile_url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"class": "items"})
    rows = table.find_all("tr") if table else soup.find_all("tr")
    for tr in rows:
        txt = tr.get_text(" ", strip=True)
        if SEASON in txt:
            cols = [c.get_text(" ", strip=True) for c in tr.find_all(["td","th"])]
            nums = [c for c in cols if c.replace(".", "").isdigit()]
            try:
                goals = int(nums[-3]) if len(nums) >= 3 else 0
                assists = int(nums[-2]) if len(nums) >= 2 else 0
                minutes = int(nums[-1]) if nums else 0
            except:
                goals = assists = minutes = 0
            league = tr.find("a").get_text(" ", strip=True) if tr.find("a") else "Unknown"
            return {"league": league, "goals": goals, "assists": assists, "minutes": minutes}
    return None

def fetch_tm_stats(name):
    pr = transfermarkt_search_player(name)
    if not pr:
        return None
    sleep_short()
    stats = parse_transfermarkt(pr)
    sleep_short()
    return stats

# -- FBref fallback --
def fbref_search(name):
    url = f"https://fbref.com/en/search/search.fcgi?search={name.replace(' ', '+')}"
    r = try_request(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    a = soup.find("div", class_="search-item-name")
    if a and (a:=a.find("a")):
        return "https://fbref.com" + a["href"]
    return None

def parse_fbref(url):
    r = try_request(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    for table in soup.find_all("table"):
        txt = table.get_text(" ", strip=True)
        if "Standard" in txt:
            rows = table.find_all("tr")
            for tr in rows:
                if SEASON.split("/")[0] in tr.get_text():
                    c = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
                    nums = [x for x in c if x.isdigit()]
                    goals = int(nums[-4]) if len(nums) >= 4 else 0
                    assists = int(nums[-3]) if len(nums) >= 3 else 0
                    return {"league": "Unknown (FBref)", "goals": goals, "assists": assists, "minutes": 0}
    return None

def fetch_fbref_stats(name):
    url = fbref_search(name)
    if not url:
        return None
    sleep_short()
    stats = parse_fbref(url)
    sleep_short()
    return stats

def get_external(player_name, web_name, cache):
    # Check cache
    row = cache[(cache["fpl_name"] == player_name) | (cache["source_name"] == web_name)]
    if not row.empty:
        return row.iloc[0].to_dict()
    for src, fn in [("Transfermarkt", fetch_tm_stats), ("FBref", fetch_fbref_stats)]:
        print(f"[lookup] {player_name} via {src}...")
        stats = fn(player_name)
        if stats:
            rec = {
                **stats, "fpl_name": player_name, "source_name": web_name, "season": SEASON, "notes": src
            }
            cache.loc[len(cache)] = rec
            save_cache(cache)
            return rec
    rec = {"fpl_name": player_name, "source_name": web_name, "league": "Unknown", "goals": 0, "assists": 0, "minutes": 0, "season": SEASON, "notes": "none"}
    cache.loc[len(cache)] = rec
    save_cache(cache)
    return rec

def weight_stats(goals, assists, league):
    rating = LEAGUE_DIFFICULTY.get(league, PL_BASE)
    factor = rating / PL_BASE
    return goals * factor, assists * factor

def estimate_fpl_pts(goals, assists, pos):
    if pos == "GK":
        gp = 10
    elif pos == "DEF":
        gp = 6
    elif pos == "MID":
        gp = 5
    else:
        gp = 4
    return goals * gp + assists * 3

def get_external_no_cache_update(player_name, web_name, cache):
    # Check cache first (read-only)
    row = cache[(cache["fpl_name"] == player_name) | (cache["source_name"] == web_name)]
    if not row.empty:
        return row.iloc[0].to_dict()

    # Try Transfermarkt and FBref
    for src, fn in [("Transfermarkt", fetch_tm_stats), ("FBref", fetch_fbref_stats)]:
        print(f"[lookup] {player_name} via {src}...")
        stats = fn(player_name)
        if stats:
            rec = {
                **stats, "fpl_name": player_name, "source_name": web_name, "season": SEASON, "notes": src
            }
            # IMPORTANT: Do NOT update cache here!
            return rec

    # If no stats found
    rec = {
        "fpl_name": player_name, "source_name": web_name,
        "league": "Unknown", "goals": 0, "assists": 0, "minutes": 0,
        "season": SEASON, "notes": "none"
    }
    return rec

def enrich_single(row, cache):
    if pd.isna(row.get("ext_league")):
        pname = row["name"].strip() or row["web_name"].strip()
        rec = get_external_no_cache_update(pname, row["web_name"], cache)
        gw, aw = weight_stats(rec["goals"], rec["assists"], rec["league"])
        est = estimate_fpl_pts(gw, aw, row["pos"])
        row["total_points"] = est
        row["estimated_points"] = est
        row["ext_league"] = rec["league"]
    return row

def enrich(players):
    cache = load_cache()
    enriched_rows = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(enrich_single, row, cache) for _, row in players.iterrows()]
        for future in tqdm(futures, total=len(futures)):
            enriched_rows.append(future.result())
    df = pd.DataFrame(enriched_rows)
    
    # Now update cache once, after all threads are done
    new_records = []  # Collect new records separately inside get_external_no_cache_update
    for r in new_records:
        cache.loc[len(cache)] = r
    save_cache(cache)
    
    return df

""" 
def greedy_select(players):
    squad = []
    budget = BUDGET
    teams = {}

    for pos, cnt in SQUAD_STRUCTURE.items():
        df = players[players["pos"] == pos].copy()
        # Sort by total_points DESC, then cost DESC to maximize cost among top scorers
        df = df.sort_values(by=["total_points", "cost"], ascending=[False, False])

        picked = 0
        for _, p in df.iterrows():
            if picked >= cnt:
                break
            if p["cost"] > budget:
                continue
            if teams.get(p["team"], 0) >= MAX_PER_TEAM:
                continue
            squad.append(p.to_dict())
            budget -= p["cost"]
            teams[p["team"]] = teams.get(p["team"], 0) + 1
            picked += 1

    return pd.DataFrame(squad)
"""

def optimize_squad(players):
    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

    player_vars = {
        idx: pulp.LpVariable(f"player_{idx}", cat="Binary")
        for idx in players.index
    }

    # Maximize points
    prob += pulp.lpSum(players.loc[idx, "total_points"] * var for idx, var in player_vars.items())

    # Budget constraints (both max and min ~99%)
    prob += pulp.lpSum(players.loc[idx, "cost"] * var for idx, var in player_vars.items()) <= BUDGET
    prob += pulp.lpSum(players.loc[idx, "cost"] * var for idx, var in player_vars.items()) >= BUDGET * 0.99

    # Position counts exactly as per squad structure
    for pos, count in SQUAD_STRUCTURE.items():
        prob += pulp.lpSum(player_vars[idx] for idx in players[players["pos"] == pos].index) == count

    # Max 3 players per team
    teams = players["team"].unique()
    for team in teams:
        prob += pulp.lpSum(player_vars[idx] for idx in players[players["team"] == team].index) <= MAX_PER_TEAM

    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"Warning: Solver status: {pulp.LpStatus[prob.status]}")

    selected_idx = [idx for idx, var in player_vars.items() if var.varValue > 0.5]
    squad = players.loc[selected_idx].copy()

    return squad   

if __name__ == "__main__":
    print("Loading FPL data...")
    players = fetch_fpl_data()
    print(f"Loaded {len(players)} players.")
    print("Enriching missing players...")
    players = enrich(players)
    print("Selecting squad...")
    squad = optimize_squad(players)
    
    print(f"Squad cost: £{squad['cost'].sum():.1f}m – Points: {squad['total_points'].sum():.0f}")

    # Sort squad by position order (GK, DEF, MID, FWD)
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad["pos_order"] = squad["pos"].map(pos_order)
    squad = squad.sort_values(by=["pos_order", "total_points"], ascending=[True, False])
    squad = squad.drop(columns=["pos_order"])

    print(squad[["name", "pos", "team", "cost", "total_points", "ext_league"]].reset_index(drop=True))

