# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Fetch nflfastR Data
import pandas as pd
import requests
from io import BytesIO

# Fetch latest nflfastR games data from nflverse GitHub
url = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
print(f"Fetching data from: {url}")
df_nfl_games = pd.read_csv(url)

# Filter to 2015-2026, regular + playoff games
df_filtered = df_nfl_games[
    (df_nfl_games['season'] >= 2015) & 
    (df_nfl_games['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB']))
].copy()

print(f"Total games fetched: {len(df_nfl_games)}")
print(f"Filtered to 2015-2026: {len(df_filtered)} games")
print(f"Games with spread data: {df_filtered['spread_line'].notna().sum()}")

# Select key columns (must match existing nfl_spreads_historical schema)
columns_to_keep = [
    'game_id', 'season', 'game_type', 'week', 'gameday', 'weekday', 'gametime',
    'away_team', 'away_score', 'home_team', 'home_score',
    'spread_line', 'away_spread_odds', 'home_spread_odds',
    'total_line', 'over_odds', 'under_odds',
    'away_moneyline', 'home_moneyline',
    'location', 'result', 'total', 'overtime',
    'div_game', 'roof', 'surface', 'temp', 'wind',
    'away_qb_name', 'home_qb_name', 'stadium'
]
df_spreads = df_filtered[columns_to_keep].copy()

# Convert to Spark DataFrame and MERGE into table (non-destructive)
spark_df = spark.createDataFrame(df_spreads)
spark_df.createOrReplaceTempView('staging_nfl_spreads')

spark.sql("""
MERGE INTO workspace.default.nfl_spreads_historical AS t
USING staging_nfl_spreads AS s
ON t.game_id = s.game_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

print(f"Merged {len(df_spreads)} games into workspace.default.nfl_spreads_historical")
print(f"Years: {df_spreads['season'].min()} to {df_spreads['season'].max()}")

# COMMAND ----------

# DBTITLE 1,Merge Spreads into nfl_complete_analysis
# MAGIC %sql
# MAGIC -- Sync updated spreads, scores, and odds from nfl_spreads_historical into nfl_complete_analysis
# MAGIC MERGE INTO workspace.default.nfl_complete_analysis AS c
# MAGIC USING workspace.default.nfl_spreads_historical AS s
# MAGIC ON c.game_id = s.game_id
# MAGIC WHEN MATCHED THEN UPDATE SET
# MAGIC   -- Always sync spread/odds from nfl_spreads_historical (has authoritative closing lines)
# MAGIC   c.spread_line = s.spread_line,
# MAGIC   c.away_spread_odds = s.away_spread_odds,
# MAGIC   c.home_spread_odds = s.home_spread_odds,
# MAGIC   c.total_line = s.total_line,
# MAGIC   c.over_odds = s.over_odds,
# MAGIC   c.under_odds = s.under_odds,
# MAGIC   c.away_moneyline = s.away_moneyline,
# MAGIC   c.home_moneyline = s.home_moneyline,
# MAGIC   -- Always update scores (picks up newly completed games)
# MAGIC   c.away_score = s.away_score,
# MAGIC   c.home_score = s.home_score,
# MAGIC   c.overtime = s.overtime,
# MAGIC   c.gametime = s.gametime

# COMMAND ----------

# DBTITLE 1,Update Derived Columns
# MAGIC %sql
# MAGIC -- Update derived columns for games with scores
# MAGIC UPDATE workspace.default.nfl_complete_analysis
# MAGIC SET 
# MAGIC   point_differential = home_score - away_score,
# MAGIC   total_points = home_score + away_score,
# MAGIC   game_result = CASE 
# MAGIC     WHEN home_score > away_score THEN 'home_win'
# MAGIC     WHEN away_score > home_score THEN 'away_win'
# MAGIC     WHEN home_score = away_score THEN 'tie'
# MAGIC     ELSE NULL
# MAGIC   END,
# MAGIC   spread_result = CASE 
# MAGIC     WHEN away_score IS NULL OR home_score IS NULL OR spread_line IS NULL THEN NULL
# MAGIC     WHEN ABS((home_score + (-spread_line)) - away_score) < 0.01 THEN 'push'
# MAGIC     WHEN (home_score + (-spread_line)) > away_score THEN 'home_covered'
# MAGIC     ELSE 'away_covered'
# MAGIC   END,
# MAGIC   over_under_result = CASE
# MAGIC     WHEN away_score IS NULL OR home_score IS NULL OR total_line IS NULL THEN NULL
# MAGIC     WHEN (home_score + away_score) > total_line THEN 'over'
# MAGIC     WHEN (home_score + away_score) < total_line THEN 'under'
# MAGIC     ELSE 'push'
# MAGIC   END
# MAGIC WHERE home_score IS NOT NULL AND away_score IS NOT NULL

# COMMAND ----------

# DBTITLE 1,Merge into Enhanced Table
# MAGIC %sql
# MAGIC -- Sync nfl_combined_analysis_enhanced with refreshed data from complete + betting view
# MAGIC MERGE INTO workspace.default.nfl_combined_analysis_enhanced AS e
# MAGIC USING (
# MAGIC   SELECT 
# MAGIC     c.game_id, c.spread_line, c.away_score, c.home_score,
# MAGIC     c.point_differential, c.total_points, c.overtime,
# MAGIC     c.gametime,
# MAGIC     c.game_result, c.spread_result, c.over_under_result,
# MAGIC     c.away_spread_odds, c.home_spread_odds, c.total_line,
# MAGIC     c.over_odds, c.under_odds, c.away_moneyline, c.home_moneyline,
# MAGIC     b.Spread AS bet_spread, b.Home_Spread, b.Away_Spread,
# MAGIC     b.Win, b.MOV, b.Win_Abs, b.MOV_Abs, b.Spread_Abs,
# MAGIC     b.spread_result_corrected, b.Team_Cover, b.spread_result_original,
# MAGIC     b.Betting_Recommendation, b.Bet_Result
# MAGIC   FROM workspace.default.nfl_complete_analysis c
# MAGIC   JOIN workspace.default.nfl_betting_analysis b 
# MAGIC     ON c.season = b.season AND c.week = b.week AND c.gameday = b.gameday 
# MAGIC     AND c.away_team_abbr = b.away_team_abbr AND c.home_team_abbr = b.home_team_abbr
# MAGIC ) AS src
# MAGIC ON e.game_id = src.game_id
# MAGIC WHEN MATCHED THEN UPDATE SET
# MAGIC   -- Always sync spread/odds from nfl_complete_analysis (has authoritative closing lines)
# MAGIC   e.spread_line = src.spread_line,
# MAGIC   e.away_spread_odds = src.away_spread_odds,
# MAGIC   e.home_spread_odds = src.home_spread_odds,
# MAGIC   e.total_line = src.total_line,
# MAGIC   e.over_odds = src.over_odds,
# MAGIC   e.under_odds = src.under_odds,
# MAGIC   e.away_moneyline = src.away_moneyline,
# MAGIC   e.home_moneyline = src.home_moneyline,
# MAGIC   -- Always update scores (picks up newly completed games)
# MAGIC   e.away_score = src.away_score,
# MAGIC   e.home_score = src.home_score,
# MAGIC   e.point_differential = src.point_differential,
# MAGIC   e.total_points = src.total_points,
# MAGIC   e.overtime = src.overtime,
# MAGIC   e.gametime = src.gametime,
# MAGIC   e.game_result = src.game_result,
# MAGIC   e.spread_result = src.spread_result,
# MAGIC   e.over_under_result = src.over_under_result,
# MAGIC   -- Always sync betting spread columns from nfl_betting_analysis
# MAGIC   e.Spread = src.bet_spread,
# MAGIC   e.Home_Spread = src.Home_Spread,
# MAGIC   e.Away_Spread = src.Away_Spread,
# MAGIC   e.Win = src.Win,
# MAGIC   e.MOV = src.MOV,
# MAGIC   e.Win_Abs = src.Win_Abs,
# MAGIC   e.MOV_Abs = src.MOV_Abs,
# MAGIC   e.Spread_Abs = src.Spread_Abs,
# MAGIC   e.spread_result_corrected = src.spread_result_corrected,
# MAGIC   e.Team_Cover = src.Team_Cover,
# MAGIC   e.spread_result_original = src.spread_result_original,
# MAGIC   e.Betting_Recommendation = src.Betting_Recommendation,
# MAGIC   e.Bet_Result = src.Bet_Result,
# MAGIC   e.home_covered = CASE 
# MAGIC     WHEN src.away_score IS NULL OR src.home_score IS NULL OR src.spread_line IS NULL THEN NULL
# MAGIC     WHEN (src.home_score + (-src.spread_line)) > src.away_score THEN 1 
# MAGIC     ELSE 0 
# MAGIC   END,
# MAGIC   e.cover = CASE 
# MAGIC     WHEN src.away_score IS NULL OR src.home_score IS NULL OR src.spread_line IS NULL THEN NULL
# MAGIC     WHEN (src.home_score + (-src.spread_line)) > src.away_score THEN 'HOME'
# MAGIC     WHEN ABS((src.home_score + (-src.spread_line)) - src.away_score) < 0.01 THEN 'PUSH'
# MAGIC     ELSE 'AWAY'
# MAGIC   END,
# MAGIC   e.ats_result = CASE 
# MAGIC     WHEN src.away_score IS NULL OR src.home_score IS NULL OR src.spread_line IS NULL THEN NULL
# MAGIC     WHEN ABS((src.home_score + (-src.spread_line)) - src.away_score) < 0.01 THEN 'PUSH'
# MAGIC     WHEN (src.home_score + (-src.spread_line)) > src.away_score THEN 'HOME_COVERED'
# MAGIC     ELSE 'AWAY_COVERED'
# MAGIC   END

# COMMAND ----------

# DBTITLE 1,Refresh Combined Table (7-Day Window)
# MAGIC %sql
# MAGIC -- Refresh nfl_combined_analysis from enhanced (7-day window to match scheduled job)
# MAGIC DELETE FROM workspace.default.nfl_combined_analysis
# MAGIC WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7)
# MAGIC   AND TO_DATE(gameday) <= DATE_ADD(CURRENT_DATE(), 7);
# MAGIC
# MAGIC INSERT INTO workspace.default.nfl_combined_analysis
# MAGIC SELECT DISTINCT *
# MAGIC FROM workspace.default.nfl_combined_analysis_enhanced
# MAGIC WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7)
# MAGIC   AND TO_DATE(gameday) <= DATE_ADD(CURRENT_DATE(), 7);
# MAGIC
# MAGIC SELECT COUNT(*) as refreshed_rows FROM workspace.default.nfl_combined_analysis WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7) AND TO_DATE(gameday) <= DATE_ADD(CURRENT_DATE(), 7);

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Verify gametime populated
# MAGIC %sql
# MAGIC -- Verify gametime is populated
# MAGIC SELECT 
# MAGIC   CASE WHEN gametime IS NOT NULL THEN 'populated' ELSE 'NULL' END AS gametime_status,
# MAGIC   COUNT(*) AS game_count
# MAGIC FROM workspace.default.nfl_spreads_historical
# MAGIC GROUP BY CASE WHEN gametime IS NOT NULL THEN 'populated' ELSE 'NULL' END;
# MAGIC
# MAGIC -- Show sample of upcoming games with Central Time conversion
# MAGIC -- nflverse gametime is in US Eastern Time; convert to Central
# MAGIC SELECT 
# MAGIC   gameday, 
# MAGIC   gametime AS kickoff_et,
# MAGIC   CONVERT_TIMEZONE('America/New_York', 'America/Chicago', TO_TIMESTAMP(CONCAT(gameday, ' ', COALESCE(gametime, '12:00')), 'yyyy-MM-dd HH:mm')) AS kickoff_central,
# MAGIC   away_team, home_team, spread_line
# MAGIC FROM workspace.default.nfl_spreads_historical
# MAGIC WHERE home_score IS NULL AND season = 2026
# MAGIC ORDER BY gameday
# MAGIC LIMIT 10;