#!/bin/bash
# Run this locally to request resources on the hpc for running a vscode server

ssh -t submit-a \
    "salloc -A kt-lab --partition=preempt --time=3-00:00:00 -c 12 --nodes=1 --mem=16G --nodelist=cn-v-[1-9]"
