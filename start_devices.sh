#!/bin/bash
set -e

SERVER_ADDRESS=${SERVER_ADDRESS:="[::]:8080"}
NUM_DEVICES=${NUM_DEVICES:=10}
START_PORT=${START_PORT:=5100}
DEVICE_PORT=$START_PORT

dev_pids=()
echo "Starting $NUM_DEVICES clients."
for ((i = 1; i <= $NUM_DEVICES; i++))
do
  export FL_PORT=$DEVICE_PORT
  export FL_NK=100
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
