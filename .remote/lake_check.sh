#!/bin/bash
echo "== ~/aigp/lake =="
ls ~/aigp/lake 2>/dev/null | head -8
du -sh ~/aigp/lake 2>/dev/null
echo "== windows lrspeiser =="
ls /mnt/c/Users/henry/lrspeiser/ai-grand-prix 2>/dev/null | head -8
echo "== telemetry =="
ls /mnt/c/Users/henry/lrspeiser/ai-grand-prix/telemetry 2>/dev/null | head -6
find /mnt/c/Users/henry/lrspeiser/ai-grand-prix -maxdepth 2 -iname "*sysid*" 2>/dev/null | head -4
