#!/bin/sh

RUN_PATH=`dirname $0`
# echo $RUN_PATH

python $RUN_PATH/atari_breakout/main.py $@
