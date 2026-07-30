#!/bin/bash
if pgrep -f fastsim_train_ppo >/dev/null; then echo ALIVE; else echo DEAD; fi
tail -1 ~/aigp/data/fastsim_runs/ppo_v1/train_log.jsonl 2>/dev/null
