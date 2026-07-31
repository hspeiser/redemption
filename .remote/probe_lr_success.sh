#!/bin/bash
BASE=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/telemetry
echo "== finishrun.csv head =="
head -3 "$BASE/race_logs/finishrun.csv"
echo "== manifest.csv tail =="
tail -5 "$BASE/race_logs/manifest.csv"
echo "== run_outcomes count + newest =="
ls "$BASE/run_outcomes" | wc -l
ls -t "$BASE/run_outcomes" | head -3
echo "== newest outcome sample =="
newest=$(ls -t "$BASE/run_outcomes" | head -1)
python3 -c "
import json
d = json.load(open('$BASE/run_outcomes/$newest'))
print(json.dumps(d, indent=1)[:900])
"
