# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Fetch Live NFL Odds from Odds API
import requests
import pandas as pd
from datetime import datetime

# Configuration
API_KEY = 'YOUR_ODDS_API_KEY_HERE'  # Get from https://the-odds-api.com
API_BASE = 'https://api.the-odds-api.com/v4'
SPORT = 'americanfootball_nfl'

# NFL team name to abbreviation mapping (Odds API uses full names, nflfastR uses abbr)
TEAM_MAP = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
    'Washington Football Team': 'WAS', 'Washington Redskins': 'WAS',
    'Oakland Raiders': 'LV', 'San Diego Chargers': 'LAC', 'St. Louis Rams': 'LA',
}

# Fetch live NFL odds (spreads market)
print("Fetching live NFL odds from The Odds API...")
url = f'{API_BASE}/sports/{SPORT}/odds'
params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'spreads',
    'oddsFormat': 'american',
    'dateFormat': 'iso',
}
response = requests.get(url, params=params, timeout=30)

if response.status_code != 200:
    print(f'ERROR: API returned {response.status_code}')
    print(response.text[:500])
    raise Exception(f'Odds API failed with status {response.status_code}')

games = response.json()
print(f'API calls remaining: {response.headers.get("x-requests-remaining", "unknown")}')
print(f'Games with odds: {len(games)}')

# Parse the odds - get the consensus spread from multiple books
odds_data = []
for game in games:
    home_team = game.get('home_team', '')
    away_team = game.get('away_team', '')
    home_abbr = TEAM_MAP.get(home_team, '')
    away_abbr = TEAM_MAP.get(away_team, '')
    
    if not home_abbr or not away_abbr:
        print(f'  Skipping - unknown team: {away_team} @ {home_team}')
        continue
    
    # Collect spreads from all bookmakers, use the median/most common
    away_spreads = []
    home_spreads = []
    
    for book in game.get('bookmakers', []):
        for market in book.get('markets', []):
            if market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    if outcome['name'] == away_team:
                        away_spreads.append(outcome['point'])
                    elif outcome['name'] == home_team:
                        home_spreads.append(outcome['point'])
    
    if away_spreads:
        # Use the most common spread (mode); fall back to median if all values are unique
        from statistics import multimode, median
        modes = multimode(away_spreads)
        away_spread = float(modes[0]) if len(modes) < len(away_spreads) else float(median(away_spreads))
        # nflfastR spread_line = away team's spread (positive = away underdog)
        odds_data.append({
            'away_team_abbr': away_abbr,
            'home_team_abbr': home_abbr,
            'away_team': away_team,
            'home_team': home_team,
            'spread_line': away_spread,
            'commence_time': game.get('commence_time', ''),
            'num_books': len(set(b['key'] for b in game.get('bookmakers', []))),
        })
        print(f'  {away_abbr} @ {home_abbr}: spread_line={away_spread} ({len(set(b["key"] for b in game.get("bookmakers", [])))} books)')

print(f'\nTotal games with spread data: {len(odds_data)}')

# Convert to Spark DataFrame and display
if odds_data:
    odds_df = pd.DataFrame(odds_data)
    display(odds_df)
else:
    print('No odds data available. The NFL season may not have upcoming games, or the API key may be invalid.')

# COMMAND ----------

# DBTITLE 1,Add kickoff_central Column
# Add kickoff_central column to NFL tables if it doesn't exist
# This stores game time in America/Chicago timezone (converted from ET gametime)
for table in ['workspace.default.nfl_combined_analysis_enhanced', 'workspace.default.nfl_combined_analysis']:
    try:
        spark.sql(f"ALTER TABLE {table} ADD COLUMN kickoff_central STRING")
        print(f"  Added kickoff_central to {table}")
    except Exception as e:
        if "already exists" in str(e) or "duplicate column" in str(e).lower():
            print(f"  kickoff_central already exists in {table}")
        else:
            raise

# COMMAND ----------

# DBTITLE 1,Update Tables with Live Odds
# Create a temp view from the odds data for SQL MERGE
if odds_data:
    odds_spark_df = spark.createDataFrame(pd.DataFrame(odds_data))
    odds_spark_df.createOrReplaceTempView('live_odds')
    
    print("Updating spread_line for unplayed games across all tables...")
    
    # 1. Update nfl_spreads_historical (only for unplayed games)
    spark.sql("""
    MERGE INTO workspace.default.nfl_spreads_historical AS s
    USING live_odds AS o
    ON s.away_team = o.away_team AND s.home_team = o.home_team
      AND s.home_score IS NULL
    WHEN MATCHED THEN UPDATE SET
      s.spread_line = o.spread_line,
      s.gametime = DATE_FORMAT(FROM_UTC_TIMESTAMP(SUBSTRING(o.commence_time, 1, 19), 'America/New_York'), 'HH:mm')
    """)
    print("  Updated nfl_spreads_historical")
    
    # 2. Update nfl_complete_analysis (only for unplayed games)
    spark.sql("""
    MERGE INTO workspace.default.nfl_complete_analysis AS c
    USING live_odds AS o
    ON c.away_team_abbr = o.away_team_abbr AND c.home_team_abbr = o.home_team_abbr
      AND c.home_score IS NULL
    WHEN MATCHED THEN UPDATE SET
      c.spread_line = o.spread_line,
      c.gametime = DATE_FORMAT(FROM_UTC_TIMESTAMP(SUBSTRING(o.commence_time, 1, 19), 'America/New_York'), 'HH:mm')
    """)
    print("  Updated nfl_complete_analysis")
    
    # 3. Update nfl_combined_analysis_enhanced (only for unplayed games)
    spark.sql("""
    MERGE INTO workspace.default.nfl_combined_analysis_enhanced AS e
    USING (
      SELECT 
        c.game_id, c.spread_line, c.gametime,
        CONVERT_TIMEZONE('America/New_York', 'America/Chicago', TO_TIMESTAMP(CONCAT(c.gameday, ' ', COALESCE(c.gametime, '12:00')), 'yyyy-MM-dd HH:mm')) AS kickoff_central,
        b.Spread AS bet_spread, b.Home_Spread, b.Away_Spread
      FROM workspace.default.nfl_complete_analysis c
      JOIN workspace.default.nfl_betting_analysis b
        ON c.season = b.season AND c.week = b.week AND c.gameday = b.gameday
        AND c.away_team_abbr = b.away_team_abbr AND c.home_team_abbr = b.home_team_abbr
      WHERE c.home_score IS NULL
    ) AS src
    ON e.game_id = src.game_id
    WHEN MATCHED THEN UPDATE SET
      e.spread_line = src.spread_line,
      e.gametime = src.gametime,
      e.kickoff_central = src.kickoff_central,
      e.Spread = src.bet_spread,
      e.Home_Spread = src.Home_Spread,
      e.Away_Spread = src.Away_Spread
    """)
    print("  Updated nfl_combined_analysis_enhanced")
    
    # 4. Refresh nfl_combined_analysis from enhanced (7-day window)
    spark.sql("""
    DELETE FROM workspace.default.nfl_combined_analysis
    WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7)
      AND TO_DATE(gameday) <= DATE_ADD(CURRENT_DATE(), 7)
    """)
    spark.sql("""
    INSERT INTO workspace.default.nfl_combined_analysis
    SELECT DISTINCT *
    FROM workspace.default.nfl_combined_analysis_enhanced
    WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7)
      AND TO_DATE(gameday) <= DATE_ADD(CURRENT_DATE(), 7)
    """)
    print("  Refreshed nfl_combined_analysis (7-day window)")
    
    # 5. Verify the update
    result = spark.sql("""
    SELECT 
      c.gameday, 
      c.gametime AS kickoff_et,
      CONVERT_TIMEZONE('America/New_York', 'America/Chicago', TO_TIMESTAMP(CONCAT(c.gameday, ' ', COALESCE(c.gametime, '12:00')), 'yyyy-MM-dd HH:mm')) AS kickoff_central,
      c.away_team_abbr, c.home_team_abbr, c.spread_line, c.Spread
    FROM workspace.default.nfl_combined_analysis c
    WHERE c.home_score IS NULL
      AND c.spread_line IS NOT NULL
      AND c.season = 2026
    ORDER BY c.gameday
    """)
    print("\nUpdated spreads for upcoming games:")
    display(result)
else:
    print("No odds data to update.")

# COMMAND ----------

# DBTITLE 1,Recent Games and Spreads (Last 7 Days)
from pyspark.sql.functions import col, to_date, date_sub, current_date

recent_games = spark.sql("""
SELECT 
  gameday,
  gametime,
  kickoff_central,
  away_team_abbr,
  home_team_abbr,
  away_score,
  home_score,
  spread_line,
  Spread AS bet_spread,
  Win
FROM workspace.default.nfl_combined_analysis
WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7)
  AND TO_DATE(gameday) <= CURRENT_DATE()
ORDER BY gameday, gametime
""")

print(f"Games in the last 7 days: {recent_games.count()}")
display(recent_games)