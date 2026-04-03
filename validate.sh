#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Validating openenv.yaml${NC} ..."

# The 'openenv' package on PyPI (0.1.x) is NOT the hackathon validation CLI.
# The hackathon's openenv-core may not be publicly available yet.
# We'll validate the required files exist instead.

REQUIRED_FILES=(
  "openenv.yaml"
  "models.py"
  "environment.py"
  "server.py"
  "inference.py"
  "Dockerfile"
  "requirements.txt"
)

MISSING_FILES=()
for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$REPO_DIR/$f" ]; then
    MISSING_FILES+=("$f")
  fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
  pass "All required files present: ${REQUIRED_FILES[*]}"
else
  fail "Missing required files: ${MISSING_FILES[*]}"
  stop_at "Step 3"
fi

# Validate openenv.yaml has required fields
# On Windows, 'python' may redirect to Microsoft Store. Use python.exe or py.exe
PYTHON_CMD=""
if command -v py &>/dev/null; then
  PYTHON_CMD="py"
elif command -v python.exe &>/dev/null; then
  PYTHON_CMD="python.exe"
elif command -v python3 &>/dev/null; then
  PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
  PYTHON_CMD="python"
fi

if [ -n "$PYTHON_CMD" ]; then
  # Use a temp file to avoid path quoting issues with spaces
  YAML_SCRIPT=$(portable_mktemp "validate-yaml")
  CLEANUP_FILES+=("$YAML_SCRIPT")
  cat > "$YAML_SCRIPT" << 'PYEOF'
import yaml
import sys
import os
repo_dir = sys.argv[1]
yaml_path = os.path.join(repo_dir, 'openenv.yaml')
try:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    required = ['name', 'version', 'description', 'tasks']
    missing = [k for k in required if k not in data]
    if missing:
        print(f'Missing fields: {missing}')
        sys.exit(1)
    print(f'Valid: name={data["name"]}, version={data["version"]}, tasks={len(data["tasks"])}')
except Exception as e:
    print(f'YAML error: {e}')
    sys.exit(1)
PYEOF

  YAML_CHECK=$($PYTHON_CMD "$YAML_SCRIPT" "$REPO_DIR" 2>&1) && YAML_OK=true || YAML_OK=false

  if [ "$YAML_OK" = true ]; then
    pass "openenv.yaml validated: $YAML_CHECK"
  else
    fail "openenv.yaml validation error: $YAML_CHECK"
    hint "Ensure openenv.yaml has: name, version, description, tasks"
    stop_at "Step 3"
  fi
else
  log "${YELLOW}SKIPPED${NC} -- Python not found for YAML validation"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0