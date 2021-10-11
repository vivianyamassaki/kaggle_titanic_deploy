#!/usr/bin/env bash

set -e

COMMAND=${1:-"web"}
export WORKER_CLASS=${WORKER_CLASS:-"uvicorn.workers.UvicornWorker"}
case "$COMMAND" in
 web)
   exec gunicorn predictor.api.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ;;
 *)
   exec sh -c "$*"
   ;;
esac