#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# dots.ocr vision-encoder precision A/B/C driver.
#
# For each precision tier (default: baseline hifi4 bf16) this:
#   1. frees the service port,
#   2. launches the tt-inference-server local vLLM server with
#      DOTS_OCR_VISION_PRECISION=<tier> (identical command every arm),
#   3. waits for the OpenAI /v1/models endpoint to answer 200,
#   4. confirms the server log shows the requested precision tier,
#   5. (bf16 only) runs a 1-image smoke gate; skips the full run if it fails,
#   6. runs the demo over all images into results/ab_<tier>/,
#   7. copies the server log into that dir, stops the server, settles.
#
# Everything is overridable via env vars (see below). Safe to re-run: each arm
# writes to its own results/ab_<tier>/ dir.
set -uo pipefail

TIERS="${TIERS:-baseline hifi4 bf16}"
TT_INF="${TT_INF:-/home/aroberge/tt-inference-server}"
TT_METAL_HOME="${TT_METAL_HOME:-/home/aroberge/tt-metal}"
VLLM_DIR="${VLLM_DIR:-/home/aroberge/vllm}"
IMAGE_DIR="${IMAGE_DIR:-/home/aroberge/t1_pages}"
OUT_ROOT="${OUT_ROOT:-/home/aroberge/t1_pages/results}"
PORT="${PORT:-8000}"
LIMIT="${LIMIT:-9}"
JWT_SECRET="${JWT_SECRET:-test123}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"   # seconds to wait for server boot (weights+device init)
SETTLE="${SETTLE:-45}"                    # seconds to wait for device release between arms
PYTHON="${PYTHON:-python}"               # interpreter to run run.py (system python has the deps)
DEMO_PYTHON="${DEMO_PYTHON:-python}"     # interpreter to run the demo (needs requests+PIL)
TT_SMI="${TT_SMI:-tt-smi}"               # for device reset between arms
BASE_URL="http://127.0.0.1:${PORT}/v1"

export JWT_SECRET
RUN_TS="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG_DIR="${OUT_ROOT}/ab_driver_logs"
mkdir -p "$OUT_ROOT" "$DRIVER_LOG_DIR"

log() { echo "[ab-driver $(date +%H:%M:%S)] $*"; }

mint_jwt() {
  python3 - "$JWT_SECRET" <<'PY'
import sys, hmac, hashlib, base64, json
secret = sys.argv[1]
b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
h = b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
p = b64(json.dumps({"team_id": "tenstorrent", "token_id": "debug-test"}, separators=(",", ":")).encode())
sig = b64(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
print(f"{h}.{p}.{sig}")
PY
}
TOKEN="$(mint_jwt)"

stop_server() {
  # Best-effort stop: named PID (if known) + anything on the port + the vLLM entrypoint.
  local pid="${1:-}"
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  fuser -k "${PORT}/tcp" 2>/dev/null
  pkill -f "run_vllm_api_server.py" 2>/dev/null
  pkill -f "rednote-hilab/dots.ocr" 2>/dev/null
  pkill -f "run.py --model dots.ocr" 2>/dev/null   # any lingering (blocking) launcher
  # wait for the port to actually free
  for _ in $(seq 1 "$SETTLE"); do
    if ! fuser "${PORT}/tcp" >/dev/null 2>&1; then break; fi
    sleep 1
  done
}

reset_device() {
  # A hard-killed server leaves the TT device wedged ("ARC core failed to start"
  # on the next open). Reset between arms so every arm opens a clean device.
  log "resetting TT device (${TT_SMI} -r) ..."
  timeout 300 "$TT_SMI" -r >/dev/null 2>&1
  log "device reset exit=$? ; settling ${SETTLE}s"
  sleep "$SETTLE"
}

wait_ready() {
  # Poll /v1/models until HTTP 200 or timeout. Returns 0 ready, 1 timeout, 2 server died.
  # Boot on T3K takes minutes (weights + 8-device init), and the vLLM startup logs
  # contain benign lines like "CUDA platform is not available" -- so DO NOT infer a
  # crash from log grep. The only reliable early-abort signal is the launched server
  # PID no longer being alive (with the port also closed).
  local server_pid="$1" deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
      -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/models" 2>/dev/null || echo 000)"
    if [ "$code" = "200" ]; then return 0; fi
    if [ -n "$server_pid" ] && ! kill -0 "$server_pid" 2>/dev/null \
       && ! fuser "${PORT}/tcp" >/dev/null 2>&1; then
      return 2   # the server process exited and nothing is on the port -> crashed
    fi
    sleep 5
  done
  return 1
}

run_arm() {
  local tier="$1"
  local out_dir="${OUT_ROOT}/ab_${tier}"
  local launcher_log="${DRIVER_LOG_DIR}/launcher_${tier}_${RUN_TS}.log"
  log "=== ARM: ${tier} -> ${out_dir} ==="
  rm -rf "$out_dir"; mkdir -p "$out_dir"

  stop_server ""   # ensure clean slate
  reset_device     # recover the device from any prior arm's hard-kill
  log "[${tier}] launching server (DOTS_OCR_VISION_PRECISION=${tier}) ..."
  ( cd "$TT_INF" && MODEL_SPECS_ENV=dev DOTS_OCR_VISION_PRECISION="$tier" "$PYTHON" run.py \
      --model dots.ocr --tt-device t3k --workflow server --local-server --dev-mode \
      --service-port "$PORT" --tt-metal-home "$TT_METAL_HOME" --vllm-dir "$VLLM_DIR" \
      --disable-trace-capture --skip-system-sw-validation ) >"$launcher_log" 2>&1 &
  local launcher_bg=$!
  sleep 10   # let run.py create the server log + (for SERVER workflow) print PID and exit

  # run.py exits after spawning the detached server; recover PID + server log path.
  local server_pid server_log
  server_pid="$(grep -oiE "process PID: [0-9]+" "$launcher_log" | grep -oE "[0-9]+" | tail -1)"
  server_log="$(grep -oiE "log file: .+\.log" "$launcher_log" | sed -E 's/.*log file: //' | tail -1)"
  if [ -z "$server_log" ]; then
    server_log="$(ls -t "$TT_INF"/workflow_logs/local_server/vllm_local_*.log 2>/dev/null | head -1)"
    [ -z "$server_log" ] && server_log="$(find "$TT_INF" /home/aroberge -maxdepth 6 -name 'vllm_local_*.log' -mmin -5 2>/dev/null | head -1)"
  fi
  log "[${tier}] server PID=${server_pid:-?} log=${server_log:-?}"

  log "[${tier}] waiting for readiness (timeout ${READY_TIMEOUT}s) ..."
  wait_ready "${server_pid:-}"; local rc=$?
  if [ "$rc" -ne 0 ]; then
    log "[${tier}] SERVER NOT READY (rc=$rc) -- recording infeasible, skipping demo."
    { echo "{\"tier\":\"${tier}\",\"status\":\"server_not_ready\",\"rc\":${rc}}"; } > "${out_dir}/INFEASIBLE.json"
    [ -f "$server_log" ] && cp "$server_log" "${out_dir}/server.log"
    cp "$launcher_log" "${out_dir}/launcher.log"
    stop_server "${server_pid:-}"
    return 1
  fi
  log "[${tier}] server READY."

  # Confirm the intended tier actually loaded (env var is authoritative; this is a guard).
  if [ -f "$server_log" ] && grep -q "DOTS_OCR_VISION_PRECISION tier=${tier}" "$server_log"; then
    log "[${tier}] confirmed precision tier in server log."
  else
    log "[${tier}] WARN: precision audit line 'tier=${tier}' not found in server log (continuing)."
  fi

  # bf16 smoke gate: one image first; bail to full-run only if it succeeds.
  if [ "$tier" = "bf16" ]; then
    log "[bf16] smoke gate: 1-image probe ..."
    JWT_SECRET="$JWT_SECRET" "$DEMO_PYTHON" "$TT_INF/demo/dots_ocr_endpoint_demo.py" \
      --base-url "$BASE_URL" --image-dir "$IMAGE_DIR" --limit 1 \
      --out "${out_dir}/_smoke" >"${out_dir}/smoke.log" 2>&1
    local ok
    ok="$(python3 -c "import json,sys;
try:
  d=json.load(open('${out_dir}/_smoke/results.json')); r=d['results'][0]
  print('1' if (r.get('ok') and (r.get('text') or '').strip()) else '0')
except Exception: print('0')" 2>/dev/null)"
    if [ "$ok" != "1" ]; then
      log "[bf16] SMOKE FAILED -- bf16 infeasible on T3K as-configured. Skipping full run."
      { echo "{\"tier\":\"bf16\",\"status\":\"smoke_failed\"}"; } > "${out_dir}/INFEASIBLE.json"
      [ -f "$server_log" ] && cp "$server_log" "${out_dir}/server.log"
      stop_server "${server_pid:-}"
      return 1
    fi
    log "[bf16] smoke OK."
  fi

  log "[${tier}] running demo (limit=${LIMIT}) ..."
  JWT_SECRET="$JWT_SECRET" "$DEMO_PYTHON" "$TT_INF/demo/dots_ocr_endpoint_demo.py" \
    --base-url "$BASE_URL" --image-dir "$IMAGE_DIR" --limit "$LIMIT" \
    --out "$out_dir" >"${out_dir}/demo.log" 2>&1
  local demo_rc=$?
  log "[${tier}] demo exit=${demo_rc}"

  [ -f "$server_log" ] && cp "$server_log" "${out_dir}/server.log"
  cp "$launcher_log" "${out_dir}/launcher.log"
  log "[${tier}] stopping server ..."
  stop_server "${server_pid:-}"
  log "[${tier}] done."
  return 0
}

log "Tiers: ${TIERS} | port ${PORT} | out ${OUT_ROOT} | limit ${LIMIT}"
for tier in $TIERS; do
  run_arm "$tier" || log "arm ${tier} did not complete a full demo run."
done
log "ALL ARMS COMPLETE."
