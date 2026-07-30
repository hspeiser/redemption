#!/bin/bash
base=/mnt/c/Users/henry/lrspeiser/ai-grand-prix
echo "== newest entries in repo (top-level, last 12h) =="
find "$base" -maxdepth 2 -newermt "-12 hours" -type d 2>/dev/null | head -20
echo "== newest files (last 6h, excluding frames) =="
find "$base" -maxdepth 3 -newermt "-6 hours" -type f \
  ! -name "*.jpg" ! -name "*.png" 2>/dev/null | head -30
