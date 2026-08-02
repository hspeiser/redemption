"""Build the exact v77-config teacher eval command from config.json."""
import json

cfg = json.load(open(
    r"D:\ai-gp\training\vq2_sac_runs\gate10_left15_gpu10_awr_v77"
    r"\20260730_200627\config.json"))["args"]

skip = {"episodes", "eval_only", "output_root", "record_root",
        "full_recording", "smoke", "replay_dir", "eval_interval"}
parts = [
    r".venv-train\Scripts\python.exe scripts\train_vq2_sac_live.py",
    "--eval-only", "--episodes 4",
    r"--output-root D:\ai-gp\training\vq2_teachercheck2",
]
for key, val in cfg.items():
    if key in skip or val is None:
        continue
    flag = "--" + key.replace("_", "-")
    if isinstance(val, bool):
        if val:
            parts.append(flag)
        continue
    sval = str(val)
    if sval == "" or " " in sval or "\\" in sval:
        parts.append(f'{flag} "{sval}"')
    else:
        parts.append(f"{flag} {sval}")
print(" ".join(parts))
