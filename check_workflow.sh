#!/bin/bash

# Get the latest CI/CD Pipeline workflow run
echo "Getting latest CI/CD Pipeline workflow run..."
echo "============================================="

# Get the latest workflow run
latest_run=$(curl -s -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/kathanparagshah/Customer-Churn-Analysis/actions/workflows/ci.yml/runs?branch=main&per_page=1")

run_info=$(echo "$latest_run" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data['workflow_runs']:
    run = data['workflow_runs'][0]
    print(f'{run[\"id\"]}|{run[\"run_number\"]}|{run[\"status\"]}|{run[\"conclusion\"]}|{run[\"created_at\"]}')
else:
    print('No runs found')
")

if [ "$run_info" = "No runs found" ]; then
    echo "No workflow runs found"
    exit 1
fi

# Parse run info
IFS='|' read -r run_id run_number status conclusion created_at <<< "$run_info"

echo "Latest Run #$run_number (ID: $run_id)"
echo "Status: $status"
echo "Conclusion: $conclusion"
echo "Created: $created_at"
echo

# Get jobs for this run
echo "Jobs in this workflow run:"
echo "-------------------------"
curl -s -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/kathanparagshah/Customer-Churn-Analysis/actions/runs/$run_id/jobs" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
failed_jobs = []
for job in data['jobs']:
    print(f'Job: {job[\"name\"]} - Status: {job[\"status\"]} - Conclusion: {job[\"conclusion\"]} - ID: {job[\"id\"]}')
    if job['conclusion'] == 'failure':
        failed_jobs.append(job['id'])
        print(f'  Failed job ID: {job[\"id\"]}')
        print(f'  Started: {job[\"started_at\"]}')
        print(f'  Completed: {job[\"completed_at\"]}')
        print(f'  HTML URL: {job[\"html_url\"]}')
        print()

if failed_jobs:
    print('\\nFailed job IDs:')
    for job_id in failed_jobs:
        print(job_id)
else:
    print('\\nNo failed jobs found!')
"