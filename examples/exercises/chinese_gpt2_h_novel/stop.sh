#!/usr/bin/env bash

kill -9 `ps -aef | grep 'run.sh' | grep -v grep | awk '{print $2}' | sed 's/\n/ /'`

kill -9 `ps -aef | grep 'Transformers/bin/python3' | grep -v grep | awk '{print $2}' | sed 's/\n/ /'`
