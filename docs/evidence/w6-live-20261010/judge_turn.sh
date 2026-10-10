#!/usr/bin/env bash
# live-w6 B44: run scripts/launch.sh's judge-pass block VERBATIM (extracted between its >>> / <<< markers, as
# tests/test_judge_proof_turn.py does) over a launcher-style receipt, with the local Qwen judge config.
# usage: judge_turn.sh <state_dir>
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
REPO=$LV/cascade
STATE_DIR=${1:?state dir}
PY=/home/johnny/Projects/demo/cascade/.venv/bin/python
JUDGE=vlm
export CASCADE_JUDGE_CONFIG=$LV/scratch/judge-qwen-local.json
export PYTHONPATH=$REPO/src CUDA_VISIBLE_DEVICES=-1
unset CASCADE_JUDGE_TIMEOUT_S
log() { printf '[launch] %s\n' "$*"; }
BLOCK="$(sed -n '/^    # >>> judge pass$/,/^    # <<< judge pass$/p' $REPO/scripts/launch.sh)"
[[ -n "$BLOCK" ]] || { echo "judge block not found"; exit 1; }
printf '%s\n' "$BLOCK" > $LV/b44/judge_block_extracted.sh
sha256sum "$STATE_DIR/proof.json" > $LV/b44/proof_sha_before.txt
EV="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['evidence_dir'])" "$STATE_DIR/proof.json")"
sha256sum "$EV/proof.json" >> $LV/b44/proof_sha_before.txt
T0=$(date +%s.%N)
eval "$BLOCK"
T1=$(date +%s.%N)
echo "JUDGE_NOTE=$JUDGE_NOTE"
echo "elapsed_s=$(python3 -c "print(round($T1-$T0,2))")"
sha256sum "$STATE_DIR/proof.json" "$EV/proof.json" > $LV/b44/proof_sha_after.txt
cp "$EV/run-summary.json" "$EV/judge.json" "$EV/judge-run.log" $LV/b44/ 2>/dev/null
cp "$STATE_DIR/judge-pass.log" $LV/b44/ 2>/dev/null
cp "$STATE_DIR/proof.json" $LV/b44/proof.json
diff <(cut -d' ' -f1 $LV/b44/proof_sha_before.txt) <(cut -d' ' -f1 $LV/b44/proof_sha_after.txt) && echo "PROOF_UNCHANGED"
