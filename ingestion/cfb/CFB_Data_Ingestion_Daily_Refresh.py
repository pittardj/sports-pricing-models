# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,README: Bug Fix Summary
# MAGIC %md
# MAGIC # 🔄 CFB Data Refresh Strategy
# MAGIC
# MAGIC ## Incremental MERGE Approach
# MAGIC
# MAGIC This notebook uses an **incremental refresh** strategy to efficiently update only recent and upcoming games while preserving all historical data (2014-2024).
# MAGIC
# MAGIC ### What Gets Refreshed
# MAGIC - **Recent games**: 1 week before today (catches late score corrections)
# MAGIC - **Future games**: All upcoming games (spreads change frequently before kickoff)
# MAGIC - **Historical data**: Never touched (2014-2024 remains static)
# MAGIC
# MAGIC ### How It Works
# MAGIC 1. **Cell 2**: Fetches only games from cutoff date onwards (today - 7 days)
# MAGIC 2. **Cell 3**: Processes betting lines for those games
# MAGIC 3. **Cell 4**: Uses **MERGE** to update existing rows and insert new ones
# MAGIC    - `WHEN MATCHED`: Updates scores/spreads for existing games
# MAGIC    - `WHEN NOT MATCHED`: Inserts new games
# MAGIC
# MAGIC ### Benefits
# MAGIC - ✅ **Fast**: Refreshes ~100-200 games instead of 11,000+
# MAGIC - ✅ **Safe**: Historical data (2014-2024) never disappears
# MAGIC - ✅ **Accurate**: Catches late score corrections and spread movements
# MAGIC
# MAGIC ### One-Time Historical Backfill
# MAGIC To restore full 2014-2026 data:
# MAGIC 1. Change Cell 2 line 29 to: `years_to_fetch = list(range(2014, 2027))`
# MAGIC 2. Run the notebook once
# MAGIC 3. Change it back to the incremental logic above
# MAGIC
# MAGIC ### Expected Result
# MAGIC - Daily refresh takes ~30 seconds instead of 5+ minutes
# MAGIC - `workspace.default.cfb_merged_data` maintains complete 2014-2026 history
# MAGIC - Recent/upcoming games stay current with latest scores and spreads

# COMMAND ----------

# DBTITLE 1,Fetch Latest CFB Games and Lines from API
import requests
import pandas as pd
from datetime import datetime
import time

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
                wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                print(f"⏳ HTTP 502 for {endpoint} {year}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed: HTTP {response.status_code} for {endpoint} in {year}")
                return None
        except Exception as e:
            print(f"❌ Error fetching {endpoint} for {year}: {str(e)}")
            return None
    
    return None

# API Configuration
api_key = 'YOUR_CFB_API_KEY_HERE'  # Get from https://collegefootballdata.com
# INCREMENTAL REFRESH: Only fetch recent/upcoming games
# - 1 week before today (catch late score corrections)
# - All future games (spreads change frequently)
from datetime import timedelta

cutoff_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
print(f"🏈 Starting CFB data refresh at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📅 Fetching games from {cutoff_date} onwards (recent + future)\n")

# Determine which years to fetch based on cutoff date
cutoff_year = int(cutoff_date[:4])
current_year = datetime.now().year
years_to_fetch = list(range(cutoff_year, current_year + 1))

# Fetch game scores
print("📊 Fetching game scores...")
scores_dfs = []
for year in years_to_fetch:
    df = fetch_cfb_data('games', year, api_key)
    if df is not None:
        scores_dfs.append(df)
        print(f"  ✅ {year}: {len(df)} games")
    else:
        print(f"  ⚠️  {year}: No data")
    time.sleep(1)  # Rate limit: 1 second between requests

if scores_dfs:
    full_scores = pd.concat(scores_dfs, ignore_index=True)
    # Filter to only games from cutoff date onwards
    full_scores['startDate_parsed'] = pd.to_datetime(full_scores['startDate'])
    full_scores = full_scores[full_scores['startDate_parsed'] >= cutoff_date].copy()
    full_scores = full_scores.drop(columns=['startDate_parsed'])
    print(f"\n✅ Total games fetched (from {cutoff_date} onwards): {len(full_scores)}")
else:
    raise Exception("No scores data fetched - aborting")

# Fetch betting lines
print("\n💰 Fetching betting lines...")
lines_dfs = []
for year in years_to_fetch:
    df = fetch_cfb_data('lines', year, api_key)
    if df is not None:
        lines_dfs.append(df)
        print(f"  ✅ {year}: {len(df)} games with lines")
    else:
        print(f"  ⚠️  {year}: No data")
    time.sleep(1)  # Rate limit: 1 second between requests

if lines_dfs:
    full_lines = pd.concat(lines_dfs, ignore_index=True)
    print(f"\n✅ Total lines fetched: {len(full_lines)}")
else:
    print("\n⚠️  Warning: No betting lines data fetched")
    full_lines = pd.DataFrame()

print("\n✅ API fetch complete!")

# COMMAND ----------

# DBTITLE 1,Process Betting Lines and Calculate Mode Spreads
import ast
from scipy import stats
import numpy as np

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
    full_lines = full_lines.drop(columns=['lines'], errors='ignore')
    
    # Drop duplicate columns that exist in scores
    lines_columns_to_drop = [
        "season", "seasonType", "week", "homeTeam", "awayTeam",
        "homePoints", "awayPoints", "homeConference", "awayConference",
        "homeClassification", "awayClassification", "startDate"
    ]
    full_lines = full_lines.drop(columns=[c for c in lines_columns_to_drop if c in full_lines.columns])
    
    games_with_spreads = full_lines['mode_spread'].notna().sum()
    print(f"  ✅ Processed spreads for {games_with_spreads} games")
else:
    print("⚠️  Skipping lines processing - no data available")

print("✅ Lines processing complete!")

# COMMAND ----------

# DBTITLE 1,Merge and Save to Delta Table
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("CFB Data Refresh").getOrCreate()

# NOTE: To preserve mode_spread and mode_spread_open columns in the final table,
# we explicitly select only the columns we need from full_lines before joining.
# This prevents column collisions and ensures the spread data makes it through the join.

print("🔄 Converting to Spark DataFrames...")

# Convert pandas to Spark
scores_df = spark.createDataFrame(full_scores)

# Add properly formatted game_date column (convert startDate string to date type)
from pyspark.sql.functions import to_date, to_timestamp
scores_df = scores_df.withColumn('game_date', to_date('startDate'))

print(f"  ✅ Scores: {scores_df.count()} rows")

if not full_lines.empty:
    # CRITICAL: Select only the columns we need from full_lines to avoid collisions
    # Keep: id (for join), mode_spread, mode_spread_open, homeScore, awayScore
    lines_subset = full_lines[['id', 'mode_spread', 'mode_spread_open', 'homeScore', 'awayScore']].copy()
    
    print(f"\n💰 Lines subset columns: {list(lines_subset.columns)}")
    print(f"  Non-null spreads: {lines_subset['mode_spread'].notna().sum()}")
    
    lines_df = spark.createDataFrame(lines_subset)
    print(f"  ✅ Lines: {lines_df.count()} rows with columns: {lines_df.columns}")
    
    # Merge scores with lines on game ID
    print("\n🔗 Merging scores and lines...")
    merged_df = scores_df.join(lines_df, on="id", how="left")
else:
    print("\n⚠️  No lines data to merge")
    merged_df = scores_df

print(f"  ✅ Merged: {merged_df.count()} rows")
print(f"  Total columns: {len(merged_df.columns)}")

# Add Central Time columns for Power BI display
from pyspark.sql.functions import from_utc_timestamp
merged_df = merged_df.withColumn('start_central',
    from_utc_timestamp(to_timestamp('startDate'), 'America/Chicago'))
merged_df = merged_df.withColumn('game_date_central',
    to_date(from_utc_timestamp(to_timestamp('startDate'), 'America/Chicago')))
print(f"  ✅ Added start_central and game_date_central columns")

# Save to Delta table using MERGE (preserves historical data)
print("\n💾 Merging into workspace.default.cfb_merged_data...")

# Create or get the target table
table_exists = spark.catalog.tableExists("workspace.default.cfb_merged_data")

if not table_exists:
    print("  ⚠️ Table doesn't exist - creating fresh table...")
    merged_df.write.format("delta") \
        .mode("overwrite") \
        .saveAsTable("workspace.default.cfb_merged_data")
    print("  ✅ Table created successfully!")
else:
    print("  📊 Table exists - performing incremental MERGE...")
    
    # Write new data to a temp view
    merged_df.createOrReplaceTempView("cfb_updates")
    
    # MERGE: Update existing rows, insert new ones
    merge_sql = """
    MERGE INTO workspace.default.cfb_merged_data AS target
    USING cfb_updates AS source
    ON target.id = source.id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
    
    spark.sql(merge_sql)
    print("  ✅ MERGE completed successfully!")

# VERIFY: Check that spread columns are in the saved table
print("\n🔍 Verifying saved table schema...")
saved_table = spark.table("workspace.default.cfb_merged_data")
print(f"  Total columns in table: {len(saved_table.columns)}")
print(f"  ✅ mode_spread in table: {'mode_spread' in saved_table.columns}")
print(f"  ✅ mode_spread_open in table: {'mode_spread_open' in saved_table.columns}")

if 'mode_spread' in saved_table.columns:
    spread_count = saved_table.filter(saved_table.mode_spread.isNotNull()).count()
    print(f"  ✅ Non-null spreads in saved table: {spread_count}")
else:
    print("  ❌ ERROR: mode_spread column is MISSING from saved table!")

# Summary statistics
print("\n" + "="*50)
print("🏈 CFB DATA REFRESH SUMMARY")
print("="*50)
print(f"Total games:        {merged_df.count()}")
print(f"Completed games:    {merged_df.filter(merged_df.completed == True).count()}")
print(f"Future games:       {merged_df.filter(merged_df.completed == False).count()}")
if 'mode_spread' in merged_df.columns:
    print(f"Games with spreads: {merged_df.filter(merged_df.mode_spread.isNotNull()).count()}")
else:
    print(f"Games with spreads: 0 (no betting lines data available)")
print(f"\nRefresh completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*50)
print("✅ DONE! Data is ready for feature engineering.")