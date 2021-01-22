#!/bin/bash
set -e

SERVER_HOST=${FL_SERVERHOST:="localhost:5000"}
NUM_DEVICES=${FL_NUM_DEVICES:=10}
START_PORT=${FL_START_PORT:=5100}
NK=${FL_NK:=100}
DEVICE_PORT=$START_PORT
MODEL=${FL_MODEL:="mnist"}

dev_pids=()
echo "Starting $NUM_DEVICES clients."
for ((i = 1; i <= $NUM_DEVICES; i++))
do
  export FL_PORT=$DEVICE_PORT
  export FL_NK=$NK
  export FL_MODEL=$MODEL
  export FL_SERVERHOST=$SERVER_ADDRESS
  echo "Starting client $i/$NUM_DEVICES clients on port $DEVICE_PORT"
  python main_device.py &
  dev_pids+=($!)
  DEVICE_PORT=$((DEVICE_PORT+1))
done
echo "Started $NUM_CLIENTS clients."

read -p "Press any key to stop the devices... " -n1 -s

for dev in "${dev_pids[@]}"
do
  echo "killing $dev"
  kill $dev
done
