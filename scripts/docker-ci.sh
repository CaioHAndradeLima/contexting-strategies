#!/usr/bin/env sh
set -eu

PROFILE="${1:-ci-all-strategies}"
IMAGE="${CONTEXT_LAB_IMAGE:-contexting-strategies:local}"

mkdir -p reports
docker build -t "${IMAGE}" .

docker run --rm -v "$(pwd)/reports:/reports" "${IMAGE}" \
  measure --profile "${PROFILE}" --output "/reports/measure-${PROFILE}.json"

docker run --rm -v "$(pwd)/reports:/reports" "${IMAGE}" \
  check-measure --profile "${PROFILE}" --report "/reports/measure-${PROFILE}.json"

echo "Measure profile ${PROFILE} passed. Report: reports/measure-${PROFILE}.json"
