#!/bin/bash
# ORIN. Pull the latest detbench code from GitHub and copy it into the working folder.
#   ~/odin_test/detbench/update_from_github.sh            (branch go2_rf_detr_bench)
# The clone (~/Odin-Nav-Stack2) is sparse and shallow: only go2_agx/agx/detbench, latest commit (< 1 MB).
# Only code is copied (*.py, *.sh, README.md); models/, data/, results/ and .venv/ in ~/odin_test/detbench are never touched.
set -e
REPO=${DETBENCH_REPO:-$HOME/Odin-Nav-Stack2}
BRANCH=${DETBENCH_BRANCH:-go2_rf_detr_bench}
DEST=${DETBENCH_DIR:-$HOME/odin_test/detbench}
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse -b "$BRANCH" https://github.com/datamentors/Odin-Nav-Stack2.git "$REPO"
  git -C "$REPO" sparse-checkout set go2_agx/agx/detbench
fi
git -C "$REPO" fetch --depth 1 origin "$BRANCH"
git -C "$REPO" reset --hard FETCH_HEAD      # the clone is read-only: local edits there are discarded
echo "GitHub: $(git -C "$REPO" log -1 --format='%h %s (%cr)')"
rsync -a --itemize-changes --include='*.py' --include='*.sh' --include='README.md' --exclude='*' \
  "$REPO/go2_agx/agx/detbench/" "$DEST/" | sed 's/^/  updated: /'
echo "Done. Code in $DEST matches GitHub ($BRANCH)."
