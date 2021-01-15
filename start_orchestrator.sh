#!/bin/bash
set -e

export FL_R=20
export FL_K_FIT="1.0"
export FL_K_EVAL="1.0"
export FL_MIN=10

python main_orchestrator.py
