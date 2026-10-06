# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,College Football Historical Data Import (2007-2022)
# MAGIC %md
# MAGIC # College Football Current Season Data Import (2023-2026)
# MAGIC
# MAGIC This notebook fetches current season game data from the College Football Data API and loads it into `workspace.default.cfb_merged_data`.
# MAGIC
# MAGIC ## Getting an API Key
# MAGIC
# MAGIC 1. Go to https://collegefootballdata.com
# MAGIC 2. Click "Get Started" or "Sign Up"
# MAGIC 3. Create a free account
# MAGIC 4. Navigate to your account settings to get your API key
# MAGIC 5. Paste your API key in the cell below
# MAGIC
# MAGIC **Note**: The API is free for non-commercial use with rate limits.

# COMMAND ----------

# DBTITLE 1,Install required packages and import libraries
# MAGIC %pip install requests pandas

# COMMAND ----------

# DBTITLE 1,Configure API key
# Set your College Football Data API key here
API_KEY = "YOUR_CFB_API_KEY_HERE"  # Get from https://collegefootballdata.com

import requests
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
import time
import json

# COMMAND ----------

# DBTITLE 1,DEBUG - Inspect actual API response structure
# Fetch one game from 2022 to see the actual field structure
url = "https://api.collegefootballdata.com/games"
headers = {"Authorization": f"Bearer {API_KEY}"}
params = {"year": 2022, "seasonType": "regular", "week": 1}

response = requests.get(url, headers=headers, params=params)
if response.status_code == 200:
    games = response.json()
    if games:
        print("First game structure:")
        print("=" * 80)
        import pprint
        pprint.pprint(games[0], width=120)
        print("\n" + "=" * 80)
        print(f"\nAll available fields: {list(games[0].keys())}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# COMMAND ----------

# DBTITLE 1,Define function to fetch games from API
def fetch_games_for_season(season, api_key):
    """
    Fetch all games for a given season from College Football Data API
    """
    url = "https://api.collegefootballdata.com/games"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"year": season}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        games = response.json()
        print(f"✓ Fetched {len(games)} games for {season} season")
        return games
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching {season}: {e}")
        return []

# COMMAND ----------

# DBTITLE 1,Transform API data to match target schema
def safe_float(value):
    """Convert value to float, handling None and empty strings"""
    if value is None or value == '' or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def transform_game_data(games):
    """
    Transform API response to match cfb_merged_data schema
    """
    if not games:
        return []
    
    transformed = []
    for game in games:
        # Note: API uses camelCase for field names
        record = {
            'id': game.get('id'),
            'season': game.get('season'),
            'week': game.get('week'),
            'seasonType': game.get('seasonType'),
            'startDate': game.get('startDate'),
            'startTimeTBD': game.get('startTimeTBD', False),
            'completed': game.get('completed', False),
            'neutralSite': game.get('neutralSite', False),
            'conferenceGame': game.get('conferenceGame', False),
            'attendance': safe_float(game.get('attendance')),
            'venueId': safe_float(game.get('venueId')),
            'venue': game.get('venue'),
            
            # Home team fields
            'homeId': game.get('homeId'),
            'homeTeam': game.get('homeTeam'),
            'homeClassification': game.get('homeClassification'),
            'homeConference': game.get('homeConference'),
            'homePoints': safe_float(game.get('homePoints')),
            'homeLineScores': json.dumps(game.get('homeLineScores', [])) if game.get('homeLineScores') else None,
            'homePostgameWinProbability': safe_float(game.get('homePostgameWinProbability')),
            'homePregameElo': safe_float(game.get('homePregameElo')),
            'homePostgameElo': safe_float(game.get('homePostgameElo')),
            
            # Away team fields
            'awayId': game.get('awayId'),
            'awayTeam': game.get('awayTeam'),
            'awayClassification': game.get('awayClassification'),
            'awayConference': game.get('awayConference'),
            'awayPoints': safe_float(game.get('awayPoints')),
            'awayLineScores': json.dumps(game.get('awayLineScores', [])) if game.get('awayLineScores') else None,
            'awayPostgameWinProbability': safe_float(game.get('awayPostgameWinProbability')),
            'awayPregameElo': safe_float(game.get('awayPregameElo')),
            'awayPostgameElo': safe_float(game.get('awayPostgameElo')),
            
            # Additional fields
            'excitementIndex': safe_float(game.get('excitementIndex')),
            'highlights': None,  # Table schema expects DOUBLE, but API returns string - set to None
            'notes': game.get('notes'),
            
            # Duplicate fields with different names (for schema compatibility)
            'homeTeamId': game.get('homeId'),
            'homeScore': safe_float(game.get('homePoints')),
            'awayTeamId': game.get('awayId'),
            'awayScore': safe_float(game.get('awayPoints')),
            
            # Spread fields (may not be in API response)
            'mode_spread': None,
            'mode_spread_open': None
        }
        transformed.append(record)
    
    return transformed

# COMMAND ----------

# DBTITLE 1,Create backup table before deletion
# MAGIC %sql
# MAGIC -- Create a backup of the entire table before making changes
# MAGIC CREATE OR REPLACE TABLE workspace.colllege_football.cfb_merged_data_backup_20260820
# MAGIC AS SELECT * FROM workspace.colllege_football.cfb_merged_data;
# MAGIC
# MAGIC SELECT 'Backup created successfully!' as status, COUNT(*) as total_rows 
# MAGIC FROM workspace.colllege_football.cfb_merged_data_backup_20260820

# COMMAND ----------

# DBTITLE 1,Delete incorrectly loaded data (2007-2022)
# MAGIC %sql
# MAGIC DELETE FROM workspace.colllege_football.cfb_merged_data 
# MAGIC WHERE season BETWEEN 2007 AND 2022

# COMMAND ----------

# DBTITLE 1,Fetch and load historical data (2007-2022)
# Verify API key is set
if API_KEY == "YOUR_API_KEY_HERE":
    raise ValueError("Please set your API key in the cell above before running this cell")

# Define seasons to fetch
seasons = range(2023, 2027)  # 2023 through 2026

all_games = []

for season in seasons:
    print(f"\nFetching {season} season...")
    games = fetch_games_for_season(season, API_KEY)
    
    if games:
        transformed_games = transform_game_data(games)
        all_games.extend(transformed_games)
        print(f"  Transformed {len(transformed_games)} games")
    
    # Respect API rate limits - add a small delay between requests
    time.sleep(1)

print(f"\n{'='*60}")
print(f"Total games fetched: {len(all_games)}")
print(f"{'='*60}")

# COMMAND ----------

# DBTITLE 1,Convert to Spark DataFrame and insert into table
if all_games:
    # Convert to pandas DataFrame first
    pdf = pd.DataFrame(all_games)
    
    # Replace empty strings with None for proper null handling
    pdf = pdf.replace('', None)
    
    # Convert to Spark DataFrame
    df = spark.createDataFrame(pdf)
    
    print(f"DataFrame created with {df.count()} rows")
    print("\nSchema:")
    df.printSchema()
    
    # Show sample of data
    print("\nSample data:")
    display(df.limit(5))
    
    # Insert into the target table
    print("\nInserting data into workspace.default.cfb_merged_data...")
    df.write.mode("append").saveAsTable("workspace.default.cfb_merged_data")
    
    print(f"✓ Successfully inserted {df.count()} historical games (2007-2022)")
else:
    print("No games to insert")

# COMMAND ----------

# DBTITLE 1,Verify data was loaded
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   *
# MAGIC FROM workspace.colllege_football.cfb_merged_data
# MAGIC WHERE season BETWEEN 2007 AND 2022 
# MAGIC --GROUP BY season
# MAGIC --ORDER BY season

# COMMAND ----------

# DBTITLE 1,Fetch and update spread data for 2007-2022
import time

# Fetch spread data from the /lines endpoint
url = "https://api.collegefootballdata.com/lines"
headers = {"Authorization": f"Bearer {API_KEY}"}

seasons_to_update = range(2023, 2027)
all_spread_data = []

print("Fetching spread data from API...")
for season in seasons_to_update:
    print(f"  Season {season}...", end=" ")
    
    # Fetch lines for entire season
    params = {"year": season}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            lines_data = response.json()
            
            # Process each game's lines
            for game_lines in lines_data:
                game_id = game_lines.get('id')
                lines = game_lines.get('lines', [])
                
                if not lines:
                    continue
                
                # Find consensus/mode spread (look for 'consensus' provider first)
                spread = None
                spread_open = None
                
                for line in lines:
                    if line.get('provider') == 'consensus':
                        spread = line.get('spread')
                        spread_open = line.get('spreadOpen')
                        break
                
                # If no consensus, take the first available line
                if spread is None and lines:
                    spread = lines[0].get('spread')
                    spread_open = lines[0].get('spreadOpen')
                
                # Collect spread data (cast to float for consistent types)
                if game_id and (spread is not None or spread_open is not None):
                    all_spread_data.append({
                        'game_id': int(game_id),
                        'mode_spread': float(spread) if spread is not None else None,
                        'mode_spread_open': float(spread_open) if spread_open is not None else None
                    })
            
            games_with_lines = len([g for g in lines_data if g.get('lines')])
            print(f"✓ {games_with_lines} games")
            
        elif response.status_code == 429:
            print(f"Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            continue
        else:
            print(f"Error {response.status_code}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Rate limiting - be nice to the API
    time.sleep(1)

print(f"\nFetched spread data for {len(all_spread_data)} games")

# Convert to DataFrame
if all_spread_data:
    print("\nCreating DataFrame and merging with table...")
    spread_df = spark.createDataFrame(all_spread_data)
    spread_df.createOrReplaceTempView("spread_updates")
    
    # Use MERGE to update all games in a single operation
    merge_query = """
    MERGE INTO workspace.default.cfb_merged_data AS target
    USING spread_updates AS source
    ON target.id = source.game_id
    WHEN MATCHED THEN UPDATE SET
        target.mode_spread = source.mode_spread,
        target.mode_spread_open = source.mode_spread_open
    """
    
    spark.sql(merge_query)
    
    print(f"\n✓ Successfully updated {len(all_spread_data)} games with spread data!")
else:
    print("\nNo spread data to update.")

# COMMAND ----------

# DBTITLE 1,Final verification - spread data by season
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   season,
# MAGIC   COUNT(*) as total_games,
# MAGIC   SUM(CASE WHEN mode_spread IS NOT NULL THEN 1 ELSE 0 END) as games_with_spread,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN mode_spread IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_with_spread
# MAGIC FROM workspace.colllege_football.cfb_merged_data
# MAGIC WHERE season BETWEEN 2007 AND 2022
# MAGIC GROUP BY season
# MAGIC ORDER BY season DESC

# COMMAND ----------

# DBTITLE 1,Final verification - spread data by season
# MAGIC %sql
# MAGIC