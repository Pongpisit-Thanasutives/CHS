#!/bin/bash

# Check if correct number of arguments is passed
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Assign arguments to variables
input_file=$1

jupytext --to py:percent --opt comment_magics=false "$input_file"
