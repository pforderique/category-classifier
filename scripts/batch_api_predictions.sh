#!/usr/bin/env bash
set -u

API_URL="${API_URL:-http://127.0.0.1:8000/prediction/}"

declare -a EXAMPLES=(
  "Monthly Rent|2200.00"
  "Spotify Family Plan|16.99"
  "Whole Foods Groceries|84.23"
  "Uber Ride Downtown|27.50"
  "Netflix Subscription|15.49"
  "Big Way Hot Pot|22.13"
)

echo "Sending ${#EXAMPLES[@]} sample requests to ${API_URL}"

for example in "${EXAMPLES[@]}"; do
  item_name="${example%%|*}"
  price="${example##*|}"

  echo
  echo "item_name=${item_name} | price=${price}"
  if response=$(curl -sS --fail --get "${API_URL}" \
    --data-urlencode "item_name=${item_name}" \
    --data-urlencode "price=${price}"); then
    echo "response=${response}"
  else
    echo "response=ERROR (request failed)"
  fi
done
