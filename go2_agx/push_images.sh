#!/bin/bash
# Publish the three images that run the stack to AWS ECR. Run it ON the machine that has the images (the Go2).
#   ons2:go2 -> <REGISTRY>/<PREFIX>/odin_ons2:<VERSION>      odin-yolo:0144 -> .../odin_yolo:<VERSION>      odin-gpu:go2 -> .../odin_gpu:<VERSION>
# It is a DRY RUN until you add --push: it only checks the images and prints what it would do.
#   REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com VERSION=1.0.0 ./go2_agx/push_images.sh            dry run
#   REGISTRY=... VERSION=1.0.0 ./go2_agx/push_images.sh --push                                          log in to ECR and push
#   REGISTRY=... VERSION=1.0.0 ./go2_agx/push_images.sh --push --create-repos                           also create the missing repositories
# Settings (environment): REGISTRY (required), VERSION (required, never "latest"), PREFIX (default robotics), AWS_REGION (default: from the registry name)
# Needs: docker, the aws CLI, and an AWS login that may push to ECR (SSO or keys, set up by you; nothing is read from or written to a file here).
# The ECR login password is piped straight into docker login and never stored.
set -euo pipefail
export LC_NUMERIC=C
PUSH=0; CREATE=0
for a in "$@"; do case $a in --push) PUSH=1 ;; --create-repos) CREATE=1 ;; *) echo "unknown option $a"; exit 1 ;; esac; done
: "${REGISTRY:?set REGISTRY, e.g. 123456789012.dkr.ecr.eu-west-1.amazonaws.com}"
: "${VERSION:?set VERSION, e.g. 1.0.0}"
[ "$VERSION" != latest ] || { echo "VERSION must be a version, not 'latest'"; exit 1; }
PREFIX="${PREFIX:-robotics}"
AWS_REGION="${AWS_REGION:-$(sed -n 's/.*\.ecr\.\([a-z0-9-]*\)\.amazonaws\.com.*/\1/p' <<< "$REGISTRY")}"
[ -n "$AWS_REGION" ] || { echo "cannot read the region from REGISTRY: set AWS_REGION"; exit 1; }

# local image : repository name
IMAGES=("ons2:go2:odin_ons2" "odin-yolo:0144:odin_yolo" "odin-gpu:go2:odin_gpu")
split() { local e=$1; LOCAL="${e%:*}"; REPO="${e##*:}"; REMOTE="$REGISTRY/$PREFIX/$REPO:$VERSION"; }

echo "== images on this machine"
FAIL=0
for e in "${IMAGES[@]}"; do
  split "$e"
  if docker image inspect "$LOCAL" >/dev/null 2>&1; then
    arch=$(docker image inspect "$LOCAL" --format '{{.Architecture}}'); size=$(docker image inspect "$LOCAL" --format '{{.Size}}')
    printf "  ok       %-16s %-6s %5.1f GB on disk  ->  %s\n" "$LOCAL" "$arch" "$(awk -v s="$size" 'BEGIN{print s/1e9}')" "$REMOTE"
    [ "$arch" = arm64 ] || { echo "           WARNING: expected arm64"; }
  else
    echo "  MISSING  $LOCAL (build it: ./go2_agx/build_images.sh)"; FAIL=1
  fi
done
[ $FAIL = 0 ] || exit 1

if [ $PUSH = 0 ]; then
  echo
  echo "DRY RUN: nothing was pushed. To push: add --push (and --create-repos if the repositories do not exist yet)."
  echo "Repositories needed in $AWS_REGION: $(for e in "${IMAGES[@]}"; do split "$e"; printf '%s/%s  ' "$PREFIX" "$REPO"; done)"
  echo "Recommended repository settings: private, tag immutability ON, scan on push ON."
  exit 0
fi

command -v aws >/dev/null || { echo "the aws CLI is not installed on this machine"; exit 1; }
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1 || { echo "not logged in to AWS (aws sso login, or export keys in this shell)"; exit 1; }
echo "== AWS login OK: $(aws sts get-caller-identity --query Arn --output text --region "$AWS_REGION")"

for e in "${IMAGES[@]}"; do
  split "$e"
  if ! aws ecr describe-repositories --repository-names "$PREFIX/$REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
    if [ $CREATE = 1 ]; then
      aws ecr create-repository --repository-name "$PREFIX/$REPO" --region "$AWS_REGION" --image-tag-mutability IMMUTABLE \
        --image-scanning-configuration scanOnPush=true >/dev/null && echo "  created repository $PREFIX/$REPO (immutable tags, scan on push)"
    else
      echo "repository $PREFIX/$REPO does not exist in $AWS_REGION: create it in the console, or add --create-repos"; exit 1
    fi
  fi
  if aws ecr describe-images --repository-name "$PREFIX/$REPO" --image-ids imageTag="$VERSION" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "$PREFIX/$REPO:$VERSION already exists in ECR (tags are meant to be final): use a new VERSION"; exit 1
  fi
done

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
echo "== docker logged in to $REGISTRY"
for e in "${IMAGES[@]}"; do
  split "$e"
  echo "== pushing $LOCAL as $REMOTE (large: this takes a while)"
  docker tag "$LOCAL" "$REMOTE"
  docker push "$REMOTE" | tail -2
  echo "   digest: $(docker image inspect "$REMOTE" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo unknown)"
done
echo "== done. On another Go2: REGISTRY=$REGISTRY VERSION=$VERSION ./go2_agx/pull_images.sh"
