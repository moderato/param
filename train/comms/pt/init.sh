#!/bin/bash

if [ ! -f hfile.txt ];
then
    echo "localhost" > hfile.txt
fi

mkdir -p execution_graphs
mkdir -p traces
