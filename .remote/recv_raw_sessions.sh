#!/bin/bash
mkdir -p /mnt/c/Users/henry/aigp_raw
cat > /mnt/c/Users/henry/aigp_raw/raw_sessions.tar
cd /mnt/c/Users/henry/aigp_raw
tar xf raw_sessions.tar && rm raw_sessions.tar
echo "EXTRACTED:"
ls raw_sessions | wc -l
du -sh raw_sessions
