#!/bin/bash
set -e

export FL_ROUNDS=${FL_ROUNDS:=15}
export FL_CONTROL=1
export FL_EXPORT_METRICS=true
export FL_TERMINATE=true

cd ../..
for mparams in "mnist",100; do
  IFS=',' read model nk <<< "${mparams}"
  echo "$model $nk"
  for num_devices in 10 20 50; do
    echo "$num_devices"
    for tparams in 2,8, 2,16, 2,32, 2,64, 2,128, 4,8, 4,16, 4,32, 4,64, 4,128, 8,8, 8,16, 8,32, 8,64, 8,128, 16,8, 16,16, 16,32, 16,64, 16,128, 32,8, 32,16, 32,32, 32,64, 32,128; do
      IFS=',' read epochs batch_size <<< "${tparams}"
      echo "starting FL for model $model with C=$num_devices, E=${epochs} and B=${batch_size}"
      export FL_EPOCHS=${epochs}
      export FL_BATCHSIZE=${batch_size}
      export FL_MODEL=$model
      export FL_NUM_DEVICES=$num_devices
      export FL_MIN=$num_devices
      export FL_NK=$nk

      python main_orchestrator.py &
      orchestrator_pid=$!
      ./start_devices.sh &

      tail --pid=$orchestrator_pid -f /dev/null
      echo "next training..."
    done
  done
done

