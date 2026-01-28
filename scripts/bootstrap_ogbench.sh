#!/usr/bin/env bash
set -euo pipefail

# Ensures `third_party/ogbench` exists (cloned at a pinned commit) and applies local patches.
#
# This repo does NOT vendor the full upstream OGBench tree by default (to keep clones light).
# Run this once on your server/login node before installing/running.

OGBENCH_URL="${OGBENCH_URL:-https://github.com/seohongpark/ogbench.git}"
OGBENCH_COMMIT="${OGBENCH_COMMIT:-1d4140997f60c52c6fb0702ec100dc988b18c548}"
OGBENCH_DIR="${OGBENCH_DIR:-third_party/ogbench}"
PATCH_FILE="${PATCH_FILE:-patches/ogbench_impls.patch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -d "${OGBENCH_DIR}" ]]; then
  if [[ ! -f "${OGBENCH_DIR}/pyproject.toml" ]]; then
    echo "Error: ${OGBENCH_DIR} exists but is missing pyproject.toml." >&2
    echo "Delete it and re-run, or set FORCE=1 to overwrite." >&2
    if [[ "${FORCE:-0}" != "1" ]]; then
      exit 2
    fi
    rm -rf "${OGBENCH_DIR}"
  fi
fi

if [[ ! -d "${OGBENCH_DIR}" ]]; then
  mkdir -p "$(dirname "${OGBENCH_DIR}")"
  git clone "${OGBENCH_URL}" "${OGBENCH_DIR}"
fi

if [[ -d "${OGBENCH_DIR}/.git" ]]; then
  git -C "${OGBENCH_DIR}" fetch --all --tags --prune
  git -C "${OGBENCH_DIR}" checkout -q "${OGBENCH_COMMIT}"
else
  echo "Error: ${OGBENCH_DIR} is not a git checkout." >&2
  exit 3
fi

if [[ -f "${PATCH_FILE}" ]]; then
  if git -C "${OGBENCH_DIR}" apply --reverse --check "${REPO_ROOT}/${PATCH_FILE}" >/dev/null 2>&1; then
    echo "Patch already applied: ${PATCH_FILE}"
  else
    git -C "${OGBENCH_DIR}" apply --whitespace=nowarn "${REPO_ROOT}/${PATCH_FILE}"
    echo "Applied patch: ${PATCH_FILE}"
  fi
fi

echo "OK: OGBench ready at ${OGBENCH_DIR} ($(git -C "${OGBENCH_DIR}" rev-parse --short HEAD))"
