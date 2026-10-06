# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,CFB Odds API - Spread Ingestion


# COMMAND ----------

# DBTITLE 1,Fetch odds from The Odds API
import requests
import json
from datetime import datetime
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, LongType
)

def parse_ts(ts_str):
    """Parse ISO 8601 timestamp string to datetime object."""
    if not ts_str:
        return None
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

# Widget parameter for API key
dbutils.widgets.text("api_key", "", "The Odds API Key")
api_key = dbutils.widgets.get("api_key")

if not api_key:
    raise ValueError("Please set the 'api_key' widget parameter with your The Odds API key.")

# Fetch all current CFB events with odds
SPORT_KEY = "americanfootball_ncaaf"
BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"

params = {
    "apiKey": api_key,
    "regions": "us",
    "markets": "spreads,h2h,totals",
    "oddsFormat": "american",
}

print(f"Fetching odds from The Odds API for {SPORT_KEY}...")
response = requests.get(BASE_URL, params=params, timeout=30)

# Extract quota info from headers (visible even on error responses)
requests_used = response.headers.get("x-requests-used", "unknown")
requests_remaining = response.headers.get("x-requests-remaining", "unknown")
requests_last_month = response.headers.get("x-requests-used-last-month", "unknown")

response.raise_for_status()
events = response.json()

print(f"Fetched {len(events)} events with odds.")
print(f"API requests used: {requests_used}, remaining: {requests_remaining}")

# Parse JSON into flat rows: one row per event/bookmaker/market/outcome
rows = []
for event in events:
    event_id = event["id"]
    commence_time = event["commence_time"]
    home_team = event["home_team"]
    away_team = event["away_team"]

    for book in event.get("bookmakers", []):
        bookmaker = book["title"]
        book_last_update = book.get("last_update")

        for market in book.get("markets", []):
            market_key = market["key"]
            market_last_update = market.get("last_update", book_last_update)

            for outcome in market.get("outcomes", []):
                rows.append((
                    event_id,
                    parse_ts(commence_time),
                    home_team,
                    away_team,
                    bookmaker,
                    market_key,
                    outcome.get("name", ""),
                    float(outcome["price"]) if outcome.get("price") is not None else None,
                    float(outcome["point"]) if outcome.get("point") is not None else None,
                    parse_ts(market_last_update),
                ))

# Create Spark DataFrame
schema = StructType([
    StructField("event_id", StringType(), True),
    StructField("commence_time", TimestampType(), True),
    StructField("home_team", StringType(), True),
 StructField("away_team", StringType(), True),
    StructField("bookmaker", StringType(), True),
    StructField("market", StringType(), True),
    StructField("outcome_name", StringType(), True),
    StructField("outcome_price", DoubleType(), True),
    StructField("outcome_point", DoubleType(), True),
    StructField("last_update", TimestampType(), True),
])

raw_df = spark.createDataFrame(rows, schema)

print(f"\nRaw odds: {raw_df.count()} rows across {len(events)} events")
raw_df.printSchema()
raw_df.show(10, truncate=False)

# Write raw table (overwrite each run)
raw_df.write.mode("overwrite").saveAsTable("workspace.default.cfb_odds_api_raw")
print("\nWrote raw odds to workspace.default.cfb_odds_api_raw")

# COMMAND ----------

# DBTITLE 1,Normalize team names and compute consensus lines
from collections import Counter
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType

# ─── Load CFBD team names for matching ───
cfbd_teams_df = spark.sql("SELECT DISTINCT homeTeam as team FROM workspace.default.cfb_merged_data WHERE homeClassification = 'fbs' UNION ALL SELECT DISTINCT awayTeam FROM workspace.default.cfb_merged_data WHERE homeClassification = 'fbs'")
cfbd_teams = set([r["team"] for r in cfbd_teams_df.collect()])
print(f"Loaded {len(cfbd_teams)} CFBD team names")

# ─── Team name normalization: Odds API → CFBD names ───
# Special mappings for teams where prefix matching won't work
TEAM_NAME_MAP = {
    "Ole Miss Rebels": "Ole Miss",
    "Southern California Trojans": "USC",
    "Central Florida Knights": "UCF",
    "Texas Christian Horned Frogs": "TCU",
    "Louisiana State Tigers": "LSU",
    "Pitt Panthers": "Pittsburgh",
    "Pittsburgh Panthers": "Pittsburgh",
    "North Carolina State Wolfpack": "NC State",
    "NC State Wolfpack": "NC State",
    "Virginia Military Institute Keydets": "VMI",
    "Florida International Panthers": "Florida International",
    "Florida Intl. Panthers": "Florida International",
    "Middle Tennessee Blue Raiders": "Middle Tennessee",
    "Middle Tennessee State Blue Raiders": "Middle Tennessee",
    "Southern Mississippi Golden Eagles": "Southern Miss",
    "Louisiana-Monroe Warhawks": "UL Monroe",
    "UL Monroe Warhawks": "UL Monroe",
    "Appalachian State Mountaineers": "App State",
    "Georgia Southern Eagles": "Georgia Southern",
    "Jacksonville State Gamecocks": "Jacksonville State",
    "Sam Houston State Bearkats": "Sam Houston",
    "Kennesaw State Owls": "Kennesaw State",
    "UT Martin Skyhawks": "UT Martin",
    "Tennessee-Martin Skyhawks": "UT Martin",
    "Alabama-Birmingham Blazers": "UAB",
    "Texas at El Paso Miners": "UTEP",
    "Texas-San Antonio Roadrunners": "UTSA",
    "Hawaii Rainbow Warriors": "Hawai'i",
    "Hawai'i Rainbow Warriors": "Hawai'i",
    "Massachusetts Minutemen": "Massachusetts",
    "UMass Minutemen": "Massachusetts",
    "Louisiana Ragin' Cajuns": "Louisiana",
    "Louisiana-Lafayette Ragin' Cajuns": "Louisiana",
    "Miami (FL) Hurricanes": "Miami",
    "Miami Hurricanes": "Miami",
    "Miami (OH) RedHawks": "Miami (OH)",
    "Connecticut Huskies": "UConn",
    "Southern Cal Trojans": "USC",
    "Army Black Knights": "Army",
    "Navy Midshipmen": "Navy",
    "Air Force Falcons": "Air Force",
    "Florida A&M Rattlers": "Florida A&M",
    "San Jose State Spartans": "San José State",
}

def normalize_team(name):
    """Map Odds API team name (with mascot) to CFBD team name (without mascot)."""
    # 1. Check special mappings first
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # 2. Try prefix matching: find the longest CFBD name that is a prefix of the Odds API name
    best_match = None
    best_len = 0
    for cfbd_name in cfbd_teams:
        if name.startswith(cfbd_name + " ") and len(cfbd_name) > best_len:
            best_match = cfbd_name
            best_len = len(cfbd_name)
    if best_match:
        return best_match
    # 3. Fall back: return as-is
    return name

normalize_udf = F.udf(normalize_team, StringType())

# Apply normalization to raw table
raw_normalized = spark.table("workspace.default.cfb_odds_api_raw").withColumn(
    "home_team_norm", normalize_udf(F.col("home_team"))
).withColumn(
    "away_team_norm", normalize_udf(F.col("away_team"))
).withColumn(
    "outcome_team_norm", normalize_udf(F.col("outcome_name"))
)

# Flag any unmatched team names for review
all_raw_teams = set()
events_data = raw_normalized.select("home_team", "away_team").distinct().collect()
for row in events_data:
    all_raw_teams.add(row["home_team"])
    all_raw_teams.add(row["away_team"])

unmatched = [t for t in sorted(all_raw_teams) if normalize_team(t) == t and t not in TEAM_NAME_MAP]
if unmatched:
    print("# TODO: These team names have no normalization mapping — verify they match CFBD names:")
    for t in unmatched:
        print(f"#   '{t}' -> '{t}' (unmapped)")
else:
    print("All team names have normalization mappings.")

# ─── Compute consensus lines per game ───
# Get unique events (one row per game)
events_df = raw_normalized.select(
    "event_id", "commence_time", "home_team_norm", "away_team_norm"
).distinct()

# --- Consensus Spread (mode of home team's spread point across books) ---
spread_rows = raw_normalized.filter(F.col("market") == "spreads") \
    .filter(F.col("outcome_team_norm") == F.col("home_team_norm")) \
    .select("event_id", "outcome_point", "bookmaker") \
    .dropDuplicates(["event_id", "bookmaker"])  # one spread per book per game

# Compute mode: group by event_id, find most common outcome_point
spread_mode = spread_rows.groupBy("event_id").agg(
    F.collect_list("outcome_point").alias("points")
).withColumn(
    "consensus_spread",
    # Mode = most frequent value; on tie, take the average of tied values
    F.udf(lambda pts: max(set(pts), key=pts.count) if pts else None, DoubleType())(F.col("points"))
).select("event_id", "consensus_spread")

# Count books for spread
spread_book_count = spread_rows.groupBy("event_id").agg(
    F.countDistinct("bookmaker").alias("spread_books")
)

# --- Consensus H2H (best price for each side) ---
h2h_home = raw_normalized.filter(F.col("market") == "h2h") \
    .filter(F.col("outcome_team_norm") == F.col("home_team_norm")) \
    .groupBy("event_id").agg(F.max("outcome_price").alias("consensus_home_ml"))

h2h_away = raw_normalized.filter(F.col("market") == "h2h") \
    .filter(F.col("outcome_team_norm") == F.col("away_team_norm")) \
    .groupBy("event_id").agg(F.max("outcome_price").alias("consensus_away_ml"))

h2h_books = raw_normalized.filter(F.col("market") == "h2h") \
    .groupBy("event_id").agg(F.countDistinct("bookmaker").alias("h2h_books"))

# --- Consensus Total (mode of total points across books) ---
totals_rows = raw_normalized.filter(F.col("market") == "totals") \
    .filter(F.col("outcome_name") == "Over") \
    .select("event_id", "outcome_point", "bookmaker") \
    .dropDuplicates(["event_id", "bookmaker"])

totals_mode = totals_rows.groupBy("event_id").agg(
    F.collect_list("outcome_point").alias("points")
).withColumn(
    "consensus_total",
    F.udf(lambda pts: max(set(pts), key=pts.count) if pts else None, DoubleType())(F.col("points"))
).select("event_id", "consensus_total")

totals_books = raw_normalized.filter(F.col("market") == "totals") \
    .groupBy("event_id").agg(F.countDistinct("bookmaker").alias("total_books"))

# --- Join everything together ---
consensus_df = events_df \
    .join(spread_mode, "event_id", "left") \
    .join(spread_book_count, "event_id", "left") \
    .join(h2h_home, "event_id", "left") \
    .join(h2h_away, "event_id", "left") \
    .join(h2h_books, "event_id", "left") \
    .join(totals_mode, "event_id", "left") \
    .join(totals_books, "event_id", "left") \
    .select(
        "event_id",
        F.col("commence_time").alias("game_time"),
        F.col("home_team_norm").alias("home_team"),
        F.col("away_team_norm").alias("away_team"),
        "consensus_spread",
        "spread_books",
        "consensus_home_ml",
        "consensus_away_ml",
        "h2h_books",
        "consensus_total",
        "total_books",
    )

# Write consensus table using CREATE OR REPLACE for safe overwrite
consensus_df.createOrReplaceTempView("consensus_tmp")
spark.sql("CREATE OR REPLACE TABLE workspace.default.cfb_odds_api_consensus AS SELECT * FROM consensus_tmp")
print(f"\nWrote consensus odds to workspace.default.cfb_odds_api_consensus")
print(f"Consensus rows: {consensus_df.count()}")
display(consensus_df.orderBy(F.col("game_time")))

# COMMAND ----------

# DBTITLE 1,Print summary stats
# ─── Summary stats ───
print("=" * 60)
print("THE ODDS API — CFB Odds Ingestion Summary")
print("=" * 60)

# Games fetched
num_games = consensus_df.count()
print(f"\nGames fetched: {num_games}")

# Bookmakers per game
books_per_game = spark.sql("""
    SELECT event_id, home_team, away_team, 
           spread_books, h2h_books, total_books
    FROM workspace.default.cfb_odds_api_consensus
    ORDER BY game_time
""")
print("\nBookmakers per game:")
display(books_per_game)

# Games with each market available
games_with_spread = consensus_df.filter(F.col("consensus_spread").isNotNull()).count()
games_with_h2h = consensus_df.filter(F.col("consensus_home_ml").isNotNull()).count()
games_with_total = consensus_df.filter(F.col("consensus_total").isNotNull()).count()

print(f"\nGames with spread lines: {games_with_spread} / {num_games}")
print(f"Games with moneyline:  {games_with_h2h} / {num_games}")
print(f"Games with totals:     {games_with_total} / {num_games}")

# API quota
print(f"\nAPI requests used:      {requests_used}")
print(f"API requests remaining: {requests_remaining}")
print(f"Requests last month:   {requests_last_month}")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,SQL: Show new spread coverage for previously NULL games
# MAGIC %sql
# MAGIC -- Show upcoming games that now have spread data from The Odds API
# MAGIC -- but were previously NULL in cfb_merged_data
# MAGIC SELECT 
# MAGIC   md.season,
# MAGIC   md.week,
# MAGIC   md.homeTeam,
# MAGIC   md.awayTeam,
# MAGIC   DATE(md.startDate) as game_date,
# MAGIC   md.mode_spread as existing_cfb_spread,
# MAGIC   c.consensus_spread as odds_api_spread,
# MAGIC   c.spread_books as books_reporting,
# MAGIC   c.consensus_home_ml,
# MAGIC   c.consensus_away_ml,
# MAGIC   c.consensus_total
# MAGIC FROM workspace.default.cfb_merged_data md
# MAGIC JOIN workspace.default.cfb_odds_api_consensus c
# MAGIC   ON c.home_team = md.homeTeam 
# MAGIC   AND c.away_team = md.awayTeam
# MAGIC   AND DATE(c.game_time) = DATE(md.startDate)
# MAGIC WHERE md.homeClassification = 'fbs'
# MAGIC   AND md.mode_spread IS NULL
# MAGIC   AND c.consensus_spread IS NOT NULL
# MAGIC ORDER BY md.season, md.week, md.startDate

# COMMAND ----------

# DBTITLE 1,UPDATE spreads in cfb_merged_data
# MAGIC %sql
# MAGIC -- Merge spreads from Odds API into cfb_merged_data
# MAGIC MERGE INTO workspace.default.cfb_merged_data AS target
# MAGIC USING (
# MAGIC   SELECT 
# MAGIC     md.id,
# MAGIC     c.consensus_spread
# MAGIC   FROM workspace.default.cfb_merged_data md
# MAGIC   JOIN workspace.default.cfb_odds_api_consensus c
# MAGIC     ON c.home_team = md.homeTeam 
# MAGIC     AND c.away_team = md.awayTeam
# MAGIC     AND DATE(c.game_time) = DATE(md.startDate)
# MAGIC   WHERE md.homeClassification = 'fbs'
# MAGIC     AND md.mode_spread IS NULL
# MAGIC     AND c.consensus_spread IS NOT NULL
# MAGIC ) AS source
# MAGIC ON target.id = source.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET target.mode_spread = source.consensus_spread