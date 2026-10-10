#!/bin/sh
# make_sandbox.sh - create Bob's throwaway sandbox container: Alpine, NO network, 64 MB RAM, 64 processes max,
# all Linux capabilities dropped, no privilege escalation. Bob's actions run only in here, in /fly.
# Needs Docker (use "sudo docker" if your user isn't in the docker group: DOCKER="sudo docker" ./make_sandbox.sh)
set -e
DOCKER=${DOCKER:-docker}
$DOCKER run -d --name flybrain-sandbox --label made-by=bob-flybrain \
  --network none --memory 64m --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges \
  alpine:3.20 sleep infinity
$DOCKER exec flybrain-sandbox mkdir -p /fly
echo "sandbox ready: flybrain-sandbox (no network)"
