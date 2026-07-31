#!/bin/bash
BASE=/mnt/c/Users/henry/lrspeiser/ai-grand-prix
echo "== dirs/files matching finish/official/win/lap/complete =="
find "$BASE/outputs" -maxdepth 2 \( -iname "*finish*" -o -iname "*official*" \
  -o -iname "*win*" -o -iname "*fullcourse*" -o -iname "*complete*" \
  -o -iname "*lap*" \) 2>/dev/null | head -25
echo "== l-prefixed poselog-style dirs (win1 used l###.jpg frames) =="
ls "$BASE/outputs" 2>/dev/null | head -20
echo "== any leaderboard/results files repo-wide (fast name scan) =="
find "$BASE" -maxdepth 2 \( -iname "*result*" -o -iname "*leaderboard*" \
  -o -iname "*race_time*" -o -iname "*times*" \) 2>/dev/null | head -15
