#!/usr/bin/env bash
set -euo pipefail
API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-dev-insecure-key-change-before-deploying}"
N="${N:-100}"

echo "Benchmarking ${API_URL}/recommend/{user_id} (${N} iterations)"
start=$(date +%s%N)
for i in $(seq 1 "$N"); do
    curl -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" \
        "${API_URL}/recommend/$((i % 100))"
done
end=$(date +%s%N)
total_ms=$(( (end - start) / 1000000 ))
avg_ms=$(awk "BEGIN { printf \"%.2f\", ${total_ms} / ${N} }")
echo "Total: ${total_ms}ms, Avg: ${avg_ms}ms, RPS: $(awk "BEGIN { printf \"%.1f\", ${N} * 1000 / ${total_ms} }")"
