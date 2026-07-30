#!/bin/bash
base=/mnt/c/Users/henry/lrspeiser/ai-grand-prix
echo "== outputs subdirs =="
ls "$base/outputs" 2>/dev/null | head -15
echo "== find imu.jsonl (depth 4, first 6) =="
find "$base" -maxdepth 4 -name imu.jsonl 2>/dev/null | head -6
echo "== find cmd.jsonl =="
find "$base" -maxdepth 4 -name cmd.jsonl 2>/dev/null | head -4
echo "== datasets =="
ls "$base/datasets" 2>/dev/null | head -8
