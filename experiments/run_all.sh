#!/usr/bin/env bash
# Tum v2 boru hatti. 01 ve 02 bittikten sonra uc paralel kol:
#   A (GPU)  05_classify            -> 09_covariates
#   B (CPU)  03 -> 04 -> 07 -> 10 -> 06
#   C (CPU)  08_microstates
# Sonunda 11_stats. Loglar outputs/_log_*.txt
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8
[ -f "${TEZ_CACHE:-../cache}/subjects.csv" ] || "$PY" 01_preprocess.py > outputs/_log_01.txt 2>&1
"$PY" 02_fidelity.py > outputs/_log_02_fidelity.txt 2>&1   # subjects.csv'ye fit_split ekler
( "$PY" 05_classify.py > outputs/_log_05.txt 2>&1 && "$PY" 09_covariates.py > outputs/_log_09.txt 2>&1 ) &
( for s in 03_sensitivity 04_abba_vs_sax 07_codebook 10_minute_scale 06_within_probe; do
    "$PY" $s.py > outputs/_log_$s.txt 2>&1 || echo "HATA: $s"; done ) &
( "$PY" 08_microstates.py > outputs/_log_08.txt 2>&1 ) &
wait
"$PY" 11_stats.py > outputs/_log_11.txt 2>&1
echo BITTI
