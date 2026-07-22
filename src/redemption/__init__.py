"""Redemption: synthetic YOLO-Pose gate-corner detection + PnP pose recovery.

A fully config-driven (TOML) pipeline:

    datagen  ->  distribution plots  ->  YOLO-Pose training  ->  PnP eval  ->  report

See the top-level README.md and the configs/ directory for usage.
"""

__version__ = "0.1.0"
