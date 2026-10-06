# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Imports and Team Name Mappings
import requests
import re
import time
import pandas as pd
from pyspark.sql import functions as F

# Team abbreviation → TeamRankings URL nickname
# Key fixes: OAK maps to 'raiders' (same as LV), WAS changes name by season
TEAM_NICKNAMES = {
    'ARI': 'cardinals', 'ATL': 'falcons', 'BAL': 'ravens', 'BUF': 'bills',
    'CAR': 'panthers', 'CHI': 'bears', 'CIN': 'bengals', 'CLE': 'browns',
    'DAL': 'cowboys', 'DEN': 'broncos', 'DET': 'lions', 'GB': 'packers',
    'HOU': 'texans', 'IND': 'colts', 'JAX': 'jaguars', 'KC': 'chiefs',
    'LV': 'raiders', 'OAK': 'raiders',  # Oakland → Las Vegas, same nickname
    'LAC': 'chargers', 'SD': 'chargers',  # San Diego → LA Chargers, same nickname
    'LA': 'rams', 'STL': 'rams',  # St. Louis → LA Rams, same nickname
    'MIA': 'dolphins',
    'MIN': 'vikings', 'NE': 'patriots', 'NO': 'saints', 'NYG': 'giants',
    'NYJ': 'jets', 'PHI': 'eagles', 'PIT': 'steelers', 'SF': '49ers',
    'SEA': 'seahawks', 'TB': 'buccaneers', 'TEN': 'titans',
    # Washington name changed over the years
    'WAS': 'commanders',     # 2022+ (default)
    # For 2015-2019 use 'redskins', for 2020-2021 use 'football-team'
    # This is handled in the get_nickname() function below
}

def get_nickname(abbr, season):
    """Get TeamRankings URL nickname for a team, accounting for historical name changes."""
    if abbr == 'WAS':
        if season <= 2019:
            return 'redskins'
        elif season <= 2021:
            return 'football-team'
        else:
            return 'commanders'
    return TEAM_NICKNAMES.get(abbr, abbr.lower())

print('✅ Libraries imported and team mappings configured')
print(f'   OAK → raiders (Oakland Raiders 2015-2019)')
print(f'   WAS → redskins (2015-2019) / football-team (2020-2021) / commanders (2022+)')
print(f'   SD  → chargers (San Diego Chargers, pre-2017)')

# COMMAND ----------

# DBTITLE 1,Get Games Needing Backfill
# Get all games needing backfill (spread_line present, opening_spread NULL)
games_df = spark.sql("""
    SELECT DISTINCT 
        game_id, away_team_abbr, home_team_abbr, 
        spread_line, season, week, game_type
    FROM workspace.default.nfl_combined_analysis_enhanced
    WHERE spread_line IS NOT NULL 
        AND opening_spread IS NULL
    ORDER BY season, week
""")

games = games_df.collect()
print(f'Total games to backfill: {len(games)}')

# Show breakdown by season
from collections import Counter
season_counts = Counter(g.season for g in games)
for season in sorted(season_counts):
    print(f'  {season}: {season_counts[season]} games')

# COMMAND ----------

# DBTITLE 1,Fetch Opening Spreads (Single-Threaded with Delay)
# Batch processing: 100 games per batch, 120s pause between batches
# Rate-limited to 100 requests per batch with pauses to keep request volume low
# Playoff URL patterns: WC→wild-card, DIV→divisional, SB→super-bowl, CON→skip (not available)

ROUND_MAP = {
    'REG': None,       # Use week-{week}
    'WC': 'wild-card',
    'DIV': 'divisional',
    'CON': None,       # Not available on TeamRankings — skip
    'SB': 'super-bowl',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

def fetch_opening_spread(game_id, away_abbr, home_abbr, week, season, spread_line, game_type='REG'):
    """Fetch opening spread from TeamRankings spread-movement page."""
    away_nick = get_nickname(away_abbr, season)
    home_nick = get_nickname(home_abbr, season)
    
    # Determine URL round segment based on game type
    round_name = ROUND_MAP.get(game_type)
    if game_type == 'CON':
        return {'game_id': game_id, 'opening_spread': None, 'status': 'skip_CON'}
    if round_name:
        url = f'https://www.teamrankings.com/nfl/matchup/{away_nick}-{home_nick}-{round_name}-{season}/spread-movement'
    else:
        url = f'https://www.teamrankings.com/nfl/matchup/{away_nick}-{home_nick}-week-{week}-{season}/spread-movement'
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return {'game_id': game_id, 'opening_spread': None, 'status': response.status_code}
        
        # Check for redirect to current schedule page
        title_match = re.search(r'<title>(.*?)</title>', response.text, re.DOTALL)
        if title_match and '2026 NFL Football Schedules' in title_match.group(1):
            return {'game_id': game_id, 'opening_spread': None, 'status': 'redirect'}
        
        # Parse opening line from table
        table_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', response.text, re.DOTALL)
        for row in table_rows[:10]:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
            clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if len(clean) >= 5 and 'Open' in clean:
                try:
                    open_val = clean[2]
                    if 'pick' in open_val.lower():
                        opening_spread = 0.0
                    else:
                        opening_spread = -float(open_val)  # negate: TR uses negative=favored
                    line_movement = float(spread_line) - opening_spread
                    return {
                        'game_id': game_id,
                        'opening_spread': float(opening_spread),
                        'line_movement': float(line_movement),
                        'spread_line': float(spread_line),
                        'status': 200
                    }
                except (ValueError, IndexError):
                    pass
        return {'game_id': game_id, 'opening_spread': None, 'status': 'no_data'}
    except Exception as e:
        return {'game_id': game_id, 'opening_spread': None, 'status': str(e)}

# Fetch all games in batches of 100 with 120s pause between batches
results = []
success = 0
fail = 0
rate_limited = False

for i, game in enumerate(games):
    if rate_limited:
        print(f'\n⚠️  Rate limited (403). Stopping at game {i}/{len(games)}.')
        print(f'   Run this notebook again later to retry remaining games.')
        break
    
    result = fetch_opening_spread(
        game.game_id, game.away_team_abbr, game.home_team_abbr,
        game.week, game.season, game.spread_line, game.game_type
    )
    results.append(result)
    
    if result.get('opening_spread') is not None:
        success += 1
    else:
        fail += 1
        if result.get('status') == 403:
            rate_limited = True
            continue
    
    # Progress reporting
    total = i + 1
    if total % 50 == 0 or total == len(games) or total <= 5:
        print(f'  [{total}/{len(games)}] {game.away_team_abbr}@{game.home_team_abbr} {game.game_type} W{game.week} {game.season}: ', end='')
        if result.get('opening_spread') is not None:
            print(f'open={result["opening_spread"]:.1f}, close={game.spread_line}, move={result["line_movement"]:+.1f}')
        else:
            print(f'NOT FOUND ({result.get("status")})')
    
    # Batch pause: every 100 games, wait 120s to avoid rate limit
    if total % 100 == 0 and total < len(games) and not rate_limited:
        print(f'  -- Batch {total // 100} complete. Pausing 120s to avoid rate limit... --')
        time.sleep(120)
    else:
        time.sleep(0.3)  # Short delay between requests within a batch

print(f'\nResults: {success} found, {fail} not found out of {len(results)} processed')
if rate_limited:
    remaining = len(games) - len(results)
    print(f'⚠️  {remaining} games not attempted due to rate limiting. Retry later.')

# COMMAND ----------

# DBTITLE 1,Update Tables and Check Coverage
# Update both tables with successful results
successful = [r for r in results if r.get('opening_spread') is not None]

if successful:
    results_df = pd.DataFrame(successful)
    results_df['home_line_open'] = -results_df['opening_spread']
    results_df['away_line_open'] = results_df['opening_spread']
    results_df['home_line_close'] = -results_df['spread_line']
    results_df['away_line_close'] = results_df['spread_line']
    
    print(f'Updating {len(results_df)} games with opening spread data')
    
    spark_df = spark.createDataFrame(results_df)
    spark_df.createOrReplaceTempView('opening_lines_update')
    
    # Merge into enhanced table
    spark.sql("""
        MERGE INTO workspace.default.nfl_combined_analysis_enhanced AS e
        USING opening_lines_update AS s
        ON e.game_id = s.game_id
        WHEN MATCHED THEN UPDATE SET
          e.opening_spread = s.opening_spread,
          e.line_movement = s.line_movement,
          e.home_line_open = s.home_line_open,
          e.away_line_open = s.away_line_open,
          e.home_line_close = s.home_line_close,
          e.away_line_close = s.away_line_close
    """)
    
    # Merge into combined table
    spark.sql("""
        MERGE INTO workspace.default.nfl_combined_analysis AS t
        USING opening_lines_update AS s
        ON t.game_id = s.game_id
        WHEN MATCHED THEN UPDATE SET
          t.opening_spread = s.opening_spread,
          t.line_movement = s.line_movement,
          t.home_line_open = s.home_line_open,
          t.away_line_open = s.away_line_open,
          t.home_line_close = s.home_line_close,
          t.away_line_close = s.away_line_close
    """)
    
    print(f'✅ Updated both nfl_combined_analysis_enhanced and nfl_combined_analysis with {len(results_df)} games')
else:
    print('⚠️  No successful results to update')

# Final coverage check
coverage = spark.sql("""
    SELECT 
      season,
      COUNT(*) as total,
      SUM(CASE WHEN opening_spread IS NOT NULL THEN 1 ELSE 0 END) as has_opening,
      SUM(CASE WHEN spread_line IS NOT NULL AND opening_spread IS NULL THEN 1 ELSE 0 END) as missing
    FROM workspace.default.nfl_combined_analysis_enhanced
    WHERE spread_line IS NOT NULL
    GROUP BY season
    ORDER BY season
""")
coverage.show()

# COMMAND ----------

