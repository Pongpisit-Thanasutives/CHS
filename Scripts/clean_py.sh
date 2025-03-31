#!/bin/bash

# Check if correct number of arguments is passed
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Assign arguments to variables
input_file=$1

# Clean the notebook by removing unwanted lines
sed -i '' '/^%/d;/# %%/d' "$input_file"
black "$input_file"

