#!/usr/bin/env bash
# Build submission.tar.gz for the Kaggle Crawl competition.
#
# Usage:
#   ./make_submission.sh                      # bundles ./ppo_crawl.zip
#   ./make_submission.sh checkpoints/run/ppo_crawl_500000_steps.zip
#
# The given model zip is packed as ppo_crawl.zip at the tarball root, which is
# where main.py loads it from (/kaggle_simulations/agent/ppo_crawl.zip).
set -euo pipefail

MODEL_ZIP="ppo_crawl_fir.zip"
OUT="submission.tar.gz"

if [[ ! -f "$MODEL_ZIP" ]]; then
  echo "Model zip not found: $MODEL_ZIP" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Kaggle requires the entrypoint to be main.py at the tarball root, so our
# submission_agent.py is staged under that name (keeps the repo's main.py free
# for other uses).
cp submission_agent.py "$STAGE/main.py"
cp model.py env.py constants.py opponent.py kaggle_patches.py "$STAGE/"
cp "$MODEL_ZIP" "$STAGE/ppo_crawl.zip"

# Vendor sb3_contrib (absent in the runtime) and stable_baselines3 (mismatched
# version) at the exact versions this model was trained with. main.py prepends
# _vendor/ to sys.path so these win over the runtime packages.
SITE_PACKAGES="$(python -c 'import sb3_contrib, os; print(os.path.dirname(os.path.dirname(sb3_contrib.__file__)))')"
mkdir -p "$STAGE/_vendor"
for pkg in stable_baselines3 sb3_contrib; do
  if [[ ! -d "$SITE_PACKAGES/$pkg" ]]; then
    echo "Vendored package not found: $SITE_PACKAGES/$pkg" >&2
    exit 1
  fi
  # Exclude bytecode caches to keep the tarball lean.
  rsync -a --exclude='__pycache__' "$SITE_PACKAGES/$pkg" "$STAGE/_vendor/"
done

tar -czf "$OUT" -C "$STAGE" \
  main.py model.py env.py constants.py opponent.py kaggle_patches.py ppo_crawl.zip _vendor

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1)) from model $MODEL_ZIP"
