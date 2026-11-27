#!/bin/bash
# Pull the three images that run the stack from AWS ECR and give them the local names the scripts use. Run it ON the new Go2.
#   odin_ons2 -> ons2:go2      odin_yolo -> odin-yolo:0144      odin_gpu -> odin-gpu:go2   (repository layout: see the settings below)
# After it, ./go2_agx/install.sh --ons2 --engine builds the workspace and the engine; build_images.sh is not needed.
#   REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com VERSION=1.0.0 ./go2_agx/pull_images.sh          pull what is missing
#   REGISTRY=... VERSION=1.0.0 ./go2_agx/pull_images.sh --force                                       replace images that exist (they are untagged, not deleted)
# Settings (environment): REGISTRY (required), VERSION (required), AWS_REGION (default: from the registry name), and ONE of:
#   - nothing: three repositories <PREFIX>/odin_ons2, odin_yolo, odin_gpu (PREFIX default robotics), tag = VERSION
#   - REPO=<name>: all three in ONE existing repository, tags ons2-<VERSION>, yolo-<VERSION>, gpu-<VERSION>
# Needs: docker, the aws CLI, and an AWS login that may pull from ECR (set up by you). The login password is never stored.
set -euo pipefail
FORCE=0
for a in "$@"; do case $a in --force) FORCE=1 ;; *) echo "unknown option $a"; exit 1 ;; esac; done
: "${REGISTRY:?set REGISTRY, e.g. 123456789012.dkr.ecr.eu-west-1.amazonaws.com}"
: "${VERSION:?set VERSION, e.g. 1.0.0}"
[ "$VERSION" != latest ] || { echo "VERSION must be a version, not 'latest'"; exit 1; }
PREFIX="${PREFIX:-robotics}"
AWS_REGION="${AWS_REGION:-$(sed -n 's/.*\.ecr\.\([a-z0-9-]*\)\.amazonaws\.com.*/\1/p' <<< "$REGISTRY")}"
[ -n "$AWS_REGION" ] || { echo "cannot read the region from REGISTRY: set AWS_REGION"; exit 1; }
[ "$(uname -m)" = aarch64 ] || { echo "this machine is $(uname -m), the images are arm64 (Jetson): run this on the Go2"; exit 1; }

# local image | repository name (three-repository mode) | tag prefix (single-repository mode)
IMAGES=("ons2:go2|odin_ons2|ons2" "odin-yolo:0144|odin_yolo|yolo" "odin-gpu:go2|odin_gpu|gpu")
split() {   # sets LOCAL, ECR_REPO, TAG, REMOTE
  local name short; IFS='|' read -r LOCAL name short <<< "$1"
  if [ -n "${REPO:-}" ]; then ECR_REPO="$REPO"; TAG="$short-$VERSION"; else ECR_REPO="$PREFIX/$name"; TAG="$VERSION"; fi
  REMOTE="$REGISTRY/$ECR_REPO:$TAG"
}

todo=()
for e in "${IMAGES[@]}"; do
  split "$e"
  if docker image inspect "$LOCAL" >/dev/null 2>&1 && [ $FORCE = 0 ]; then echo "  skip     $LOCAL exists (--force replaces it)"; else todo+=("$e"); fi
done
[ ${#todo[@]} -gt 0 ] || { echo "nothing to pull"; exit 0; }

command -v aws >/dev/null || { echo "the aws CLI is not installed on this machine"; exit 1; }
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1 || { echo "not logged in to AWS (aws sso login, or export keys in this shell)"; exit 1; }
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
echo "== docker logged in to $REGISTRY"
for e in "${todo[@]}"; do
  split "$e"
  echo "== pulling $REMOTE (large: this takes a while)"
  docker pull "$REMOTE" | tail -1
  docker tag "$REMOTE" "$LOCAL"
  docker rmi "$REMOTE" >/dev/null   # only the ECR name; the image keeps its local name
  echo "   now: $LOCAL ($(docker image inspect "$LOCAL" --format '{{.Architecture}}'))"
done
echo "== done. Next: ./go2_agx/install.sh --check, then ./go2_agx/install.sh --ons2 --engine"
