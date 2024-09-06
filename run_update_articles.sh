#!/bin/bash

# File to track the last run date
LAST_RUN_FILE="/tmp/last_update_articles_run"

# Get today's date
TODAY=$(date +%Y-%m-%d)

# Check if the script has already run today
if [ -f "$LAST_RUN_FILE" ] && [ "$(cat $LAST_RUN_FILE)" == "$TODAY" ]; then
    echo "Script has already run today. Exiting." >> /tmp/update_articles.log
    exit 0
fi

# Log start time
echo "Script started at $(date)" >> /tmp/update_articles.log

# Run the Python script
/Users/jakepappo/micromamba/envs/sauron/bin/python3 /Users/jakepappo/Documents/Stuff/Projects/news_agg/update_articles.py >> /tmp/update_articles.log 2>&1

# Log end time
echo "Script finished at $(date)" >> /tmp/update_articles.log

# Update the last run date
echo "$TODAY" > "$LAST_RUN_FILE"