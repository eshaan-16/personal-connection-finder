#!/usr/bin/env python3
"""Thin launcher so you can run without the -m flag:

    python run.py "Bill Gates" --context "Microsoft"

Equivalent to: python -m connection_finder.cli ...
"""
from connection_finder.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
