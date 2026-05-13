#!/usr/bin/env bash

# Get the directory where this script is located
THREEDPRINT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export variables
export THREEDPRINT

# Add script directory to PATH if needed
export PATH="$THREEDPRINT:$PATH"
