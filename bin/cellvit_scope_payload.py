#!/usr/bin/env python3
"""Inert scope gate: wait for host resource admission before CellViT starts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.wait_seconds <= 60:
        parser.error("A bounded native command is required")
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        if args.gate.is_file():
            gate = json.loads(args.gate.read_text(encoding="utf-8"))
            if (gate.get("schema") != "cellvit-live-scope-admission-v1" or
                    gate.get("unit") != args.unit or
                    gate.get("plan_sha256") != args.plan_sha256 or
                    gate.get("resource_controls_verified") is not True):
                raise RuntimeError("CellViT live scope gate is not bound to this case")
            os.execv(command[0], command)
        time.sleep(0.1)
    raise TimeoutError("CellViT live scope was never admitted; no GPU work started")


if __name__ == "__main__":
    main()
