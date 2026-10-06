# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,CFB Historical Backfill 2014-2020
# MAGIC %md
# MAGIC # CFB Historical Backfill: 2014-2020
# MAGIC
# MAGIC This notebook fetches game data and betting lines for the 2014-2020 seasons from the College Football Data API and **appends** them to the existing `workspace.default.cfb_merged_data` table.
# MAGIC
# MAGIC **Important**: This uses `mode("append")` — it will NOT overwrite existing data. After running this notebook, re-run the feature engineering query to rebuild `cfb_ml_features` with the expanded dataset.

# COMMAND ----------

# DBTITLE 1,Install packages and import libraries
# MAGIC %pip install requests pandas scipy
# MAGIC import requests
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC from scipy import stats
# MAGIC import ast
# MAGIC import time
# MAGIC from datetime import datetime
# MAGIC from pyspark.sql import SparkSession
# MAGIC from pyspark.sql.functions import to_date
# MAGIC
# MAGIC spark = SparkSession.builder.appName("CFB Historical Backfill").getOrCreate()
# MAGIC
# MAGIC # API Configuration
# MAGIC API_KEY = "YOUR_CFB_API_KEY_HERE"  # Get from https://collegefootballdata.com
# MAGIC years_to_fetch = list(range(2014, 2021))  # 2014 through 2020
# MAGIC
# MAGIC print(f"🏈 Starting CFB historical backfill at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# MAGIC print(f"📅 Fetching data for years: {years_to_fetch}\n")

# COMMAND ----------

# DBTITLE 1,Fetch games from CFBD API
def fetch_cfb_data(endpoint, year, api_key, max_retries=3):
    """Fetch data from CollegeFootballData.com API with retry logic"""
    url = f"https://api.collegefootballdata.com/{endpoint}?year={year}"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return pd.DataFrame(response.json())
            elif response.status_code == 502 and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"⏳ HTTP 502 for {endpoint} {year}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed: HTTP {response.status_code} for {endpoint} in {year}")
                return None
        except Exception as e:
            print(f"❌ Error fetching {endpoint} for {year}: {str(e)}")
            return None
    return None

# Fetch game scores
print("📊 Fetching game scores...")
scores_dfs = []
for year in years_to_fetch:
    df = fetch_cfb_data("games", year, API_KEY)
    if df is not None and not df.empty:
        print(f"  ✅ {year}: {len(df)} games")
        scores_dfs.append(df)
    else:
        print(f"  ❌ {year}: No data")
    time.sleep(1)

full_scores = pd.concat(scores_dfs, ignore_index=True) if scores_dfs else pd.DataFrame()
print(f"\n✅ Total games fetched: {len(full_scores)}")

# COMMAND ----------

# DBTITLE 1,Fetch betting lines from CFBD API
print("💰 Fetching betting lines...")
lines_dfs = []
for year in years_to_fetch:
    df = fetch_cfb_data("lines", year, API_KEY)
    if df is not None and not df.empty:
        print(f"  ✅ {year}: {len(df)} games with lines")
        lines_dfs.append(df)
    else:
        print(f"  ❌ {year}: No lines data")
    time.sleep(1)

full_lines = pd.concat(lines_dfs, ignore_index=True) if lines_dfs else pd.DataFrame()
print(f"\n✅ Total lines fetched: {len(full_lines)}")

# COMMAND ----------

# DBTITLE 1,Process betting lines and calculate mode spreads
def get_mode_spread(lines):
    """Extract most common (mode) spread from lines data"""
    spreads = []
    for line in lines:
        if isinstance(line, dict) and 'spread' in line and line['spread'] is not None:
            try:
                spreads.append(float(line['spread']))
            except ValueError:
                continue
    if not spreads:
        return None
    spreads_array = np.array(spreads)
    mode_result = stats.mode(spreads_array, nan_policy='omit', keepdims=False)
    return mode_result.mode if mode_result.count > 0 else None

def get_mode_spread_open(lines):
    """Extract most common opening spread from lines data"""
    spreads = []
    for line in lines:
        if isinstance(line, dict) and 'spreadOpen' in line and line['spreadOpen'] is not None:
            try:
                spreads.append(float(line['spreadOpen']))
            except ValueError:
                continue
    if not spreads:
        return None
    spreads_array = np.array(spreads)
    mode_result = stats.mode(spreads_array, nan_policy='omit', keepdims=False)
    return mode_result.mode if mode_result.count > 0 else None

if not full_lines.empty:
    print("📊 Processing betting lines...")
    
    # Parse 'lines' column (JSON string to list of dicts)
    full_lines['lines'] = full_lines['lines'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Calculate mode spread and opening spread
    full_lines['mode_spread'] = full_lines['lines'].apply(get_mode_spread)
    full_lines['mode_spread_open'] = full_lines['lines'].apply(get_mode_spread_open)
    
    # Drop the complex 'lines' column
    full_lines = full_lines.drop(columns=['lines'])
    
    print(f"  ✅ Processed spreads for {full_lines['mode_spread'].notna().sum()} games")
    print("✅ Lines processing complete!")
else:
    print("⚠️ No lines data to process")

# COMMAND ----------

# DBTITLE 1,Merge scores with lines and append to table
print("🔄 Converting to Spark DataFrames...")

# Convert pandas to Spark
scores_df = spark.createDataFrame(full_scores)

# Add game_date column (convert startDate string to date type)
scores_df = scores_df.withColumn('game_date', to_date('startDate'))

print(f"  ✅ Scores: {scores_df.count()} rows")

if not full_lines.empty:
    # Select only the columns we need from full_lines to avoid collisions
    lines_subset = full_lines[['id', 'mode_spread', 'mode_spread_open', 'homeScore', 'awayScore']].copy()
    
    print(f"  💰 Lines subset columns: {list(lines_subset.columns)}")
    print(f"  Non-null spreads: {lines_subset['mode_spread'].notna().sum()}")
    
    lines_df = spark.createDataFrame(lines_subset)
    print(f"  ✅ Lines: {lines_df.count()} rows")
    
    # Merge scores with lines on game ID
    print("\n🔗 Merging scores and lines...")
    merged_df = scores_df.join(lines_df, on="id", how="left")
else:
    print("\n⚠️ No lines data to merge")
    merged_df = scores_df

print(f"  ✅ Merged: {merged_df.count()} rows")
print(f"  Total columns: {len(merged_df.columns)}")

# Check for duplicate IDs before appending
existing_ids = spark.sql("SELECT id FROM workspace.default.cfb_merged_data").toPandas()
new_ids = merged_df.select("id").toPandas()
duplicate_count = existing_ids['id'].isin(new_ids['id']).sum()

if duplicate_count > 0:
    print(f"\n⚠️ WARNING: {duplicate_count} duplicate IDs found! Removing duplicates before append...")
    merged_df = merged_df.filter(~merged_df.id.isin(existing_ids['id'].tolist()))
    print(f"  ✅ After dedup: {merged_df.count()} rows")
else:
    print(f"\n✅ No duplicate IDs found")

# APPEND to existing table (NOT overwrite)
print("\n💾 Appending to workspace.default.cfb_merged_data...")

merged_df.write.format("delta") \
    .mode("append") \
    .saveAsTable("workspace.default.cfb_merged_data")

print("✅ Data appended successfully!")

# COMMAND ----------

# DBTITLE 1,Verify backfilled data
print("🔍 Verifying backfilled data...\n")

# Season-by-season summary
summary = spark.sql("""
SELECT 
  season,
  COUNT(*) as game_count,
  SUM(CASE WHEN completed = true THEN 1 ELSE 0 END) as completed_games,
  SUM(CASE WHEN mode_spread IS NOT NULL THEN 1 ELSE 0 END) as games_with_spreads,
  MIN(startDate) as earliest_game,
  MAX(startDate) as latest_game
FROM workspace.default.cfb_merged_data
GROUP BY season
ORDER BY season
""")
summary.show(20, truncate=False)

# Total row count
total = spark.sql("SELECT COUNT(*) as total FROM workspace.default.cfb_merged_data")
print(f"\n✅ Total rows in cfb_merged_data: {total.collect()[0]['total']}")
print(f"\n🏈 Backfill complete! Now re-run the feature engineering query to rebuild cfb_ml_features.")

# COMMAND ----------

# DBTITLE 1,Fetch all sportsbook lines for Tulsa vs FAU (game 400763592)
import requests
import json
from collections import Counter
import numpy as np
from scipy import stats

API_KEY = "k+BfYdbye8wr4fqYOVkiqGoVEEUxTEXaD29KqM3yq1g6PvSMTsJ2/fKDnLNycVLy"
game_id = 400763592

url = f"https://api.collegefootballdata.com/lines?year=2015&gameId={game_id}"
headers = {'Authorization': f'Bearer {API_KEY}'}

response = requests.get(url, headers=headers, timeout=30)
data = response.json()

if data:
    game = data[0]
    lines = game.get('lines', [])
    
    print(f"🏈 {game['awayTeam']} @ {game['homeTeam']}")
    print(f"📅 {game['startDate']}\n")
    print("Sportsbook Lines:")
    print("=" * 60)
    
    spreads = []
    for line in lines:
        provider = line.get('provider', 'Unknown')
        spread = line.get('spread')
        spread_open = line.get('spreadOpen')
        if spread is not None:
            spreads.append(float(spread))
        print(f"{provider:20s} | Close: {str(spread):>6s} | Open: {str(spread_open):>6s}")
    
    print("=" * 60)
    print(f"\n📊 All closing spreads: {spreads}")
    
    # Count frequency
    spread_counts = Counter(spreads)
    print(f"\n📈 Spread frequency:")
    for spread, count in sorted(spread_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {spread}: {count} sportsbooks")
    
    # Show what scipy.stats.mode would return
    import numpy as np
    from scipy import stats
    if spreads:
        mode_result = stats.mode(np.array(spreads), nan_policy='omit', keepdims=False)
        print(f"\n🎯 Calculated mode: {mode_result.mode}")
else:
    print("No data found")

# COMMAND ----------

# DBTITLE 1,Test TheSportsDB API for CFB odds
import requests
import json

# TheSportsDB API - Free tier
# API Doc: https://www.thesportsdb.com/api.php

# Test with a known game: Tulsa vs FAU 2015-09-05
# TheSportsDB uses event lookups by team + date

# First, search for the event
team_name = "Tulsa"
season = "2015"

# Try different endpoints to explore TheSportsDB coverage
print(f"Testing TheSportsDB API for CFB data...\n")

# Test 1: Search for teams by name
print("1️⃣ Searching for 'Tulsa' team...")
search_url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=Tulsa"
search_resp = requests.get(search_url, timeout=30)
if search_resp.status_code == 200:
    teams = search_resp.json().get('teams', [])
    if teams:
        for team in teams:
            if 'College' in team.get('strLeague', '') or 'NCAA' in team.get('strLeague', ''):
                print(f"   Found: {team.get('strTeam')} (ID: {team.get('idTeam')}, League: {team.get('strLeague')})")
    else:
        print("   ❌ No teams found")

# Test 2: Check if they have American Football - NCAA league
print("\n2️⃣ Checking NCAA Football league coverage...")
league_url = f"https://www.thesportsdb.com/api/v1/json/3/search_all_leagues.php?c=USA&s=American%20Football"
league_resp = requests.get(league_url, timeout=30)
if league_resp.status_code == 200:
    leagues = league_resp.json().get('countries', [])
    print(f"   Response: {json.dumps(leagues, indent=2)[:300] if leagues else 'No data'}")

print(f"\n3️⃣ Testing events endpoint with correct ID (136967)...")
url = f"https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?id=136967&s={season}"

response = requests.get(url, timeout=30)

if response.status_code == 200:
    data = response.json()
    events = data.get('events', [])
    
    if events:
        print(f"✅ Found {len(events)} events for Tulsa {season}")
        
        # Look for the FAU game on 2015-09-05
        for event in events:
            event_date = event.get('dateEvent', '')
            home_team = event.get('strHomeTeam', '')
            away_team = event.get('strAwayTeam', '')
            
            if event_date == '2015-09-05' or 'Florida Atlantic' in [home_team, away_team]:
                print(f"\n🎯 Found game: {away_team} @ {home_team}")
                print(f"   Date: {event_date}")
                print(f"   Event ID: {event.get('idEvent')}")
                print(f"   Home Score: {event.get('intHomeScore')}")
                print(f"   Away Score: {event.get('intAwayScore')}")
                print(f"\n   Available betting fields:")
                for key, value in event.items():
                    if 'odd' in key.lower() or 'spread' in key.lower() or 'line' in key.lower():
                        print(f"     {key}: {value}")
                break
    else:
        print("❌ No events found")
        print(f"Response: {json.dumps(data, indent=2)[:500]}")
else:
    print(f"❌ HTTP {response.status_code}")
    print(response.text[:500])

# COMMAND ----------

# DBTITLE 1,Create spread corrections table
# MAGIC %sql
# MAGIC -- Create a table to store verified spread corrections
# MAGIC CREATE TABLE IF NOT EXISTS workspace.default.cfb_spread_corrections (
# MAGIC   game_id BIGINT COMMENT 'CFBD game ID',
# MAGIC   verified_spread DOUBLE COMMENT 'Verified historical spread (home team perspective)',
# MAGIC   source STRING COMMENT 'Source of verification (e.g., Odds Portal, manual research)',
# MAGIC   date_corrected DATE COMMENT 'Date the correction was made',
# MAGIC   notes STRING COMMENT 'Optional notes about the correction'
# MAGIC ) COMMENT 'Manual overrides for incorrect spreads from CFBD API';
# MAGIC
# MAGIC -- Insert the Tulsa vs FAU correction we already made
# MAGIC MERGE INTO workspace.default.cfb_spread_corrections AS target
# MAGIC USING (
# MAGIC   SELECT 
# MAGIC     400763592 AS game_id,
# MAGIC     -7.0 AS verified_spread,
# MAGIC     'Manual verification' AS source,
# MAGIC     CURRENT_DATE() AS date_corrected,
# MAGIC     'CFBD API returned -4.0 (mode of teamrankings/numberfire), but verified consensus was -7.0' AS notes
# MAGIC ) AS source
# MAGIC ON target.game_id = source.game_id
# MAGIC WHEN MATCHED THEN UPDATE SET *
# MAGIC WHEN NOT MATCHED THEN INSERT *;
# MAGIC
# MAGIC -- Verify the table
# MAGIC SELECT * FROM workspace.default.cfb_spread_corrections;

# COMMAND ----------

# DBTITLE 1,Apply verified spread corrections to cfb_merged_data
# MAGIC %sql
# MAGIC -- Apply verified spread corrections to the main data table
# MAGIC -- This updates cfb_merged_data to use verified spreads where available
# MAGIC
# MAGIC MERGE INTO workspace.default.cfb_merged_data AS target
# MAGIC USING workspace.default.cfb_spread_corrections AS corrections
# MAGIC ON target.id = corrections.game_id
# MAGIC WHEN MATCHED THEN 
# MAGIC   UPDATE SET 
# MAGIC     target.mode_spread = corrections.verified_spread;
# MAGIC
# MAGIC -- Show which games were corrected
# MAGIC SELECT 
# MAGIC   m.id,
# MAGIC   m.season,
# MAGIC   m.startDate,
# MAGIC   m.homeTeam,
# MAGIC   m.awayTeam,
# MAGIC   m.mode_spread as corrected_spread,
# MAGIC   c.source,
# MAGIC   c.notes
# MAGIC FROM workspace.default.cfb_merged_data m
# MAGIC INNER JOIN workspace.default.cfb_spread_corrections c
# MAGIC   ON m.id = c.game_id
# MAGIC ORDER BY m.startDate;

# COMMAND ----------

# DBTITLE 1,Helper: Add new spread correction
# Helper function to add new spread corrections

def add_spread_correction(game_id, verified_spread, source="Manual verification", notes=""):
    """
    Add a verified spread correction to override bad API data.
    
    Parameters:
    -----------
    game_id : int
        CFBD game ID (from cfb_merged_data.id)
    verified_spread : float
        The correct spread from home team's perspective (negative = home favored)
    source : str
        Where you verified this spread (e.g., 'Odds Portal', 'Manual verification')
    notes : str
        Additional context about why this correction was needed
    
    Example:
    --------
    # Washington State vs Minnesota Holiday Bowl 2016: should be -5.5, not -8.5
    add_spread_correction(
        game_id=400876095,
        verified_spread=-5.5,
        source="Manual verification",
        notes="CFBD API had -8.5, verified consensus was -5.5"
    )
    """
    from datetime import date
    
    # Insert into corrections table
    spark.sql(f"""
        MERGE INTO workspace.default.cfb_spread_corrections AS target
        USING (
            SELECT 
                {game_id} AS game_id,
                {verified_spread} AS verified_spread,
                '{source}' AS source,
                CURRENT_DATE() AS date_corrected,
                '{notes}' AS notes
        ) AS source
        ON target.game_id = source.game_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    
    # Apply correction to main table
    spark.sql(f"""
        UPDATE workspace.default.cfb_merged_data
        SET mode_spread = {verified_spread}
        WHERE id = {game_id}
    """)
    
    # Show the corrected game
    result = spark.sql(f"""
        SELECT 
            id, season, startDate, homeTeam, awayTeam, 
            homePoints, awayPoints, mode_spread
        FROM workspace.default.cfb_merged_data
        WHERE id = {game_id}
    """)
    
    print(f"✅ Spread correction applied for game {game_id}")
    result.show(truncate=False)
    return result

# Example usage (commented out):
# add_spread_correction(400876095, -5.5, "Manual verification", "CFBD API had -8.5, verified consensus was -5.5")

print("✅ Helper function loaded. Use add_spread_correction() to add new corrections.")

# COMMAND ----------

# DBTITLE 1,Identify low-confidence spreads for spot-checking
# Identify games where CFBD API had very few sportsbooks (low confidence)
# These are candidates for verification against other sources

import requests
import json
from collections import Counter

def check_sportsbook_count(year, max_sportsbooks=2):
    """
    Find games in a given year where the API had only 1-2 sportsbooks.
    These are low-confidence spreads that should be spot-checked.
    """
    url = f"https://api.collegefootballdata.com/lines?year={year}"
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        print(f"❌ Failed to fetch lines for {year}")
        return []
    
    data = response.json()
    low_confidence = []
    
    for game in data:
        lines = game.get('lines', [])
        spreads = [line.get('spread') for line in lines if line.get('spread') is not None]
        
        if 0 < len(spreads) <= max_sportsbooks:
            low_confidence.append({
                'game_id': game['id'],
                'home': game.get('homeTeam'),
                'away': game.get('awayTeam'),
                'date': game.get('startDate', '')[:10],
                'spread': game.get('spread'),
                'sportsbook_count': len(spreads),
                'sportsbooks': [line['provider'] for line in lines if line.get('spread') is not None]
            })
    
    return low_confidence

# Example: Check 2015 season for low-confidence spreads
print("🔍 Scanning 2015 season for low-confidence spreads...\n")
low_conf_2015 = check_sportsbook_count(2015, max_sportsbooks=2)

if low_conf_2015:
    print(f"⚠️ Found {len(low_conf_2015)} games with ≤2 sportsbooks:\n")
    for game in low_conf_2015[:10]:  # Show first 10
        print(f"  Game {game['game_id']}: {game['away']} @ {game['home']} ({game['date']})")
        print(f"    Spread: {game['spread']} | {game['sportsbook_count']} sportsbook(s): {', '.join(game['sportsbooks'])}")
        print()
else:
    print("✅ All games have 3+ sportsbooks")

print("\n💡 To scan other years: low_conf = check_sportsbook_count(2016)")

# COMMAND ----------

# DBTITLE 1,Add Washington State vs Minnesota correction
# Add Washington State vs Minnesota Holiday Bowl 2016 correction
add_spread_correction(
    game_id=400876095,
    verified_spread=-5.5,
    source="Manual verification",
    notes="CFBD API had -8.5, verified consensus was -5.5"
)

# COMMAND ----------

# DBTITLE 1,Scan all backfilled years for low-confidence spreads
# Scan all backfilled years (2014-2020) for low-confidence spreads
import pandas as pd

print("🔍 Scanning all backfilled years for low-confidence spreads...\n")

all_low_conf = []

for year in range(2014, 2021):
    print(f"Scanning {year}...", end=" ")
    low_conf = check_sportsbook_count(year, max_sportsbooks=2)
    all_low_conf.extend(low_conf)
    print(f"{len(low_conf)} games found")
    time.sleep(0.5)  # Rate limiting

print(f"\n⚠️ Total low-confidence games (≤2 sportsbooks): {len(all_low_conf)}\n")

# Convert to DataFrame for easier analysis
df_low_conf = pd.DataFrame(all_low_conf)

# Show summary by year
if not df_low_conf.empty:
    df_low_conf['year'] = pd.to_datetime(df_low_conf['date']).dt.year
    summary = df_low_conf.groupby('year').size().reset_index(name='count')
    print("📊 Low-confidence games by year:")
    print(summary.to_string(index=False))
    
    # Show the 20 most recent low-confidence games
    print("\n📋 Most recent 20 low-confidence games to spot-check:\n")
    recent = df_low_conf.sort_values('date', ascending=False).head(20)
    for _, game in recent.iterrows():
        print(f"  {game['date']}: {game['away']} @ {game['home']}")
        print(f"    Game ID: {game['game_id']} | Spread: {game['spread']} | {game['sportsbook_count']} book(s): {', '.join(game['sportsbooks'])}")
        print()
    
    # Save to table for reference
    spark_df = spark.createDataFrame(df_low_conf[['game_id', 'home', 'away', 'date', 'spread', 'sportsbook_count']])
    spark_df.write.format("delta").mode("overwrite").saveAsTable("workspace.default.cfb_low_confidence_spreads")
    print(f"\n💾 Saved to workspace.default.cfb_low_confidence_spreads for reference")
else:
    print("✅ No low-confidence spreads found")

# COMMAND ----------

# DBTITLE 1,Analyze spread variance across sportsbooks
# Analyze spread variance: Find games where sportsbooks disagreed significantly
# High std dev = market uncertainty or potential data quality issues

import requests
import numpy as np
import pandas as pd
from collections import Counter
import time

API_KEY = "k+BfYdbye8wr4fqYOVkiqGoVEEUxTEXaD29KqM3yq1g6PvSMTsJ2/fKDnLNycVLy"

def analyze_spread_variance(year, min_std=2.0, min_books=3):
    """
    Find games where sportsbooks disagreed significantly on the spread.
    
    Parameters:
    -----------
    year : int
        Season year to analyze
    min_std : float
        Minimum standard deviation to flag (default 2.0 points)
    min_books : int
        Minimum number of sportsbooks required to calculate variance
    
    Returns:
    --------
    list of dicts with high-variance games
    """
    url = f"https://api.collegefootballdata.com/lines?year={year}"
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        print(f"❌ Failed to fetch lines for {year}")
        return []
    
    data = response.json()
    high_variance = []
    
    for game in data:
        lines = game.get('lines', [])
        spreads = [line.get('spread') for line in lines if line.get('spread') is not None]
        
        if len(spreads) >= min_books:
            spreads_array = np.array(spreads)
            std = np.std(spreads_array)
            mean = np.mean(spreads_array)
            
            if std >= min_std:
                sportsbooks = [line['provider'] for line in lines if line.get('spread') is not None]
                spread_list = [(line['provider'], line.get('spread')) for line in lines if line.get('spread') is not None]
                
                high_variance.append({
                    'game_id': game['id'],
                    'home': game.get('homeTeam'),
                    'away': game.get('awayTeam'),
                    'date': game.get('startDate', '')[:10],
                    'mode_spread': game.get('spread'),
                    'mean_spread': round(mean, 2),
                    'std_dev': round(std, 2),
                    'min_spread': min(spreads),
                    'max_spread': max(spreads),
                    'spread_range': round(max(spreads) - min(spreads), 2),
                    'sportsbook_count': len(spreads),
                    'all_spreads': spread_list
                })
    
    return high_variance

# Test on 2015 season
print("🔍 Analyzing spread variance for 2015 season...\n")
print("Looking for games with std dev >= 2.0 points (significant disagreement)\n")

high_var_2015 = analyze_spread_variance(2015, min_std=2.0, min_books=3)

if high_var_2015:
    print(f"⚠️ Found {len(high_var_2015)} games with high spread variance:\n")
    
    # Sort by std dev (highest first)
    high_var_2015_sorted = sorted(high_var_2015, key=lambda x: x['std_dev'], reverse=True)
    
    for game in high_var_2015_sorted[:10]:  # Show top 10
        print(f"📊 Game {game['game_id']}: {game['away']} @ {game['home']} ({game['date']})")
        mode_str = str(game['mode_spread']) if game['mode_spread'] is not None else 'None'
        print(f"   Mean: {game['mean_spread']:>6} | Mode: {mode_str:>6} | Std Dev: {game['std_dev']}")
        print(f"   Range: {game['min_spread']} to {game['max_spread']} ({game['spread_range']} point spread)")
        print(f"   {game['sportsbook_count']} sportsbooks:")
        for book, spread in game['all_spreads'][:5]:  # Show first 5
            print(f"      {book:20s}: {spread}")
        if len(game['all_spreads']) > 5:
            print(f"      ... and {len(game['all_spreads']) - 5} more")
        print()
else:
    print("✅ No high-variance games found")

print("\n💡 To scan other years: high_var = analyze_spread_variance(2016)")

# COMMAND ----------

# DBTITLE 1,Add high-variance flag to cfb_merged_data
# MAGIC %sql
# MAGIC -- Add a spread_data_quality flag to cfb_merged_data
# MAGIC -- Flags games where sportsbooks disagreed significantly (std dev >= 2.0)
# MAGIC
# MAGIC -- First, add the column if it doesn't exist
# MAGIC ALTER TABLE workspace.default.cfb_merged_data 
# MAGIC ADD COLUMNS (
# MAGIC   spread_variance DOUBLE COMMENT 'Standard deviation of spread across sportsbooks',
# MAGIC   spread_data_quality STRING COMMENT 'HIGH_VARIANCE (std>=2.0), LOW_CONFIDENCE (<=2 books), VERIFIED (manual correction), or NORMAL'
# MAGIC );
# MAGIC
# MAGIC -- Update with high variance flags
# MAGIC MERGE INTO workspace.default.cfb_merged_data AS target
# MAGIC USING workspace.default.cfb_high_variance_spreads AS source
# MAGIC ON target.id = source.game_id
# MAGIC WHEN MATCHED THEN UPDATE SET
# MAGIC   target.spread_variance = source.std_dev,
# MAGIC   target.spread_data_quality = 'HIGH_VARIANCE';
# MAGIC
# MAGIC -- Mark verified corrections
# MAGIC UPDATE workspace.default.cfb_merged_data
# MAGIC SET spread_data_quality = 'VERIFIED'
# MAGIC WHERE id IN (SELECT game_id FROM workspace.default.cfb_spread_corrections);
# MAGIC
# MAGIC -- Mark remaining games as NORMAL
# MAGIC UPDATE workspace.default.cfb_merged_data
# MAGIC SET spread_data_quality = 'NORMAL'
# MAGIC WHERE spread_data_quality IS NULL AND mode_spread IS NOT NULL;
# MAGIC
# MAGIC -- Show summary
# MAGIC SELECT 
# MAGIC   spread_data_quality,
# MAGIC   COUNT(*) as game_count,
# MAGIC   AVG(spread_variance) as avg_variance,
# MAGIC   MAX(spread_variance) as max_variance
# MAGIC FROM workspace.default.cfb_merged_data
# MAGIC WHERE mode_spread IS NOT NULL
# MAGIC GROUP BY spread_data_quality
# MAGIC ORDER BY 
# MAGIC   CASE spread_data_quality
# MAGIC     WHEN 'VERIFIED' THEN 1
# MAGIC     WHEN 'HIGH_VARIANCE' THEN 2
# MAGIC     WHEN 'NORMAL' THEN 3
# MAGIC     ELSE 4
# MAGIC   END;

# COMMAND ----------

# DBTITLE 1,Show examples of each data quality flag
# MAGIC %sql
# MAGIC -- Show examples of each data quality category
# MAGIC
# MAGIC -- All verified and high-variance games, plus a few normal ones
# MAGIC SELECT 
# MAGIC   spread_data_quality,
# MAGIC   id,
# MAGIC   season,
# MAGIC   homeTeam,
# MAGIC   awayTeam,
# MAGIC   mode_spread,
# MAGIC   ROUND(spread_variance, 2) as spread_variance
# MAGIC FROM workspace.default.cfb_merged_data
# MAGIC WHERE 
# MAGIC   spread_data_quality IN ('VERIFIED', 'HIGH_VARIANCE')
# MAGIC   OR (spread_data_quality = 'NORMAL' AND id % 1000 = 0)  -- Sample some normal games
# MAGIC ORDER BY 
# MAGIC   CASE spread_data_quality
# MAGIC     WHEN 'VERIFIED' THEN 1
# MAGIC     WHEN 'HIGH_VARIANCE' THEN 2
# MAGIC     WHEN 'NORMAL' THEN 3
# MAGIC   END,
# MAGIC   spread_variance DESC NULLS LAST
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Scan all years 2014-2020 for high-variance spreads
# Scan all backfilled years for high-variance spreads
import pandas as pd
import time

print("🔍 Scanning all years 2014-2020 for high-variance spreads...\n")

all_high_var = []

for year in range(2014, 2021):
    print(f"Scanning {year}...", end=" ")
    high_var = analyze_spread_variance(year, min_std=2.0, min_books=3)
    all_high_var.extend(high_var)
    print(f"{len(high_var)} games found")
    time.sleep(0.5)  # Rate limiting

print(f"\n⚠️ Total high-variance games (std dev ≥2.0): {len(all_high_var)}\n")

# Convert to DataFrame for analysis
df_high_var = pd.DataFrame(all_high_var)

if not df_high_var.empty:
    # Summary by year
    df_high_var['year'] = pd.to_datetime(df_high_var['date']).dt.year
    summary = df_high_var.groupby('year').agg({
        'game_id': 'count',
        'std_dev': ['mean', 'max']
    }).round(2)
    summary.columns = ['game_count', 'avg_std_dev', 'max_std_dev']
    
    print("📊 High-variance games by year:")
    print(summary.to_string())
    
    # Top 20 most problematic games across all years
    print("\n🔥 Top 20 most problematic games (highest std dev):\n")
    top20 = df_high_var.nlargest(20, 'std_dev')
    
    for idx, game in top20.iterrows():
        print(f"{game['date']}: {game['away']} @ {game['home']}")
        mode_str = str(game['mode_spread']) if game['mode_spread'] is not None else 'None'
        print(f"  Game ID: {game['game_id']} | Mean: {game['mean_spread']} | Mode: {mode_str} | Std: {game['std_dev']}")
        print(f"  Range: {game['min_spread']} to {game['max_spread']} ({game['spread_range']} pts) | {game['sportsbook_count']} books")
        print()
    
    # Save to table for reference
    spark_df = spark.createDataFrame(df_high_var[[
        'game_id', 'home', 'away', 'date', 'mode_spread', 'mean_spread', 
        'std_dev', 'min_spread', 'max_spread', 'spread_range', 'sportsbook_count'
    ]])
    spark_df.write.format("delta").mode("overwrite").saveAsTable("workspace.default.cfb_high_variance_spreads")
    print(f"💾 Saved to workspace.default.cfb_high_variance_spreads for review")
    
    # Cross-reference with our corrections
    print("\n🔍 Checking if any corrected games are in the high-variance list...")
    corrections = spark.sql("SELECT game_id FROM workspace.default.cfb_spread_corrections").toPandas()
    corrected_ids = set(corrections['game_id'].tolist())
    high_var_ids = set(df_high_var['game_id'].tolist())
    overlap = corrected_ids.intersection(high_var_ids)
    
    if overlap:
        print(f"✅ {len(overlap)} corrected game(s) found in high-variance list:")
        for game_id in overlap:
            game = df_high_var[df_high_var['game_id'] == game_id].iloc[0]
            print(f"  Game {game_id}: {game['away']} @ {game['home']} (std dev: {game['std_dev']})")
    else:
        print("ℹ️ No overlap between corrections and high-variance list")
else:
    print("✅ No high-variance spreads found")