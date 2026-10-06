-- Databricks notebook source
-- NFL Combined Analysis - 7-Day Window Refresh
-- Refreshes nfl_combined_analysis for games within a 7-day window (past 7 days to future 7 days)
-- Source: nfl_combined_analysis_enhanced (has the full dataset with latest odds/scores)
-- This is idempotent: DELETE then INSERT the same window each run

-- Step 1: Remove existing games in the 7-day window
DELETE FROM workspace.default.nfl_combined_analysis
WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7)
  AND TO_DATE(gameday) <= DATE_ADD(CURRENT_DATE(), 7);

-- Step 2: Re-insert fresh data from the enhanced table for the same window
-- SELECT DISTINCT handles any duplicate game_ids in the source
INSERT INTO workspace.default.nfl_combined_analysis
SELECT DISTINCT
  game_id, season, week, gameday, away_team_abbr, home_team_abbr, game_type, weekday,
  away_score, home_score, point_differential, total_points, overtime, away_elo_pre,
  home_elo_pre, elo_diff_pre, spread_line, away_spread_odds, home_spread_odds, total_line,
  over_odds, under_odds, away_moneyline, home_moneyline, away_games_played, away_wins,
  away_losses, away_season_points_for, away_season_points_against, away_cumulative_margin,
  away_avg_margin, home_games_played, home_wins, home_losses, home_season_points_for,
  home_season_points_against, home_cumulative_margin, home_avg_margin, location, div_game,
  roof, surface, temp, wind, stadium, away_qb_name, home_qb_name, game_result, spread_result,
  over_under_result, home_cover_prob, home_covered, side_covered, confidence, pick, cover,
  ats_result, strategy, Spread, Home_Spread, Away_Spread, Win, MOV, Win_Abs, MOV_Abs,
  Spread_Abs, spread_result_corrected, Team_Cover, spread_result_original,
  Betting_Recommendation, Bet_Result, opening_spread, line_movement, home_line_open,
  home_line_close, away_line_open, away_line_close, home_odds_open, home_odds_close,
  away_odds_open, away_odds_close, total_score_open, total_score_close, over_odds_open,
  over_odds_close, under_odds_open, under_odds_close, home_line_odds_open, home_line_odds_close,
  away_line_odds_open, away_line_odds_close
FROM workspace.default.nfl_combined_analysis_enhanced
WHERE TO_DATE(gameday) >= DATE_SUB(CURRENT_DATE(), 7)
  AND TO_DATE(gameday) <= DATE_ADD(CURRENT_DATE(), 7);