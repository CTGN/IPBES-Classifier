#!/usr/bin/env bash

# List processes matching 'uv run'
ps -ef | grep '[u]v run'
ps -ef | grep 'launch_ipbes_pipeline.sh'

# Kill processes running 'uv run'

ps -ef | grep 'launch_ipbes_pipeline.sh' | awk '{print $2}' | xargs -r kill
ps -ef | grep '[u]v run' | awk '{print $2}' | xargs -r kill