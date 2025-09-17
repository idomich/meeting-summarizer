#!/bin/bash

# Run mypy on all provided files
if [ "$#" -gt 0 ]; then
    mypy --config-file mypy.ini --no-incremental "$@"
fi