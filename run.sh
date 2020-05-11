#!/bin/sh

RUN_PATH=`dirname $0`
# echo $RUN_PATH

python $RUN_PATH/src/main.py $@
