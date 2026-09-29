#!/bin/sh
# One command: tests -> 30-seed experiment -> all figures.
set -e
python -m tests.run_tests
python -m experiments.run_all --seeds 30
python -m experiments.make_diagrams
python -m experiments.make_figures
echo "Done. See results/report.json and docs/figures/"
