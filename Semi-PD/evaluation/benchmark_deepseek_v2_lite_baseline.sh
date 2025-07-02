
# pass from command line
MODEL_PATH=$1
DATASET_PATH=$2

PWD=$(pwd)

nohup python -m sglang.launch_server   --model-path  $MODEL_PATH --trust-remote-code \
   --context-length 10240 --watchdog-timeout 60000 --dist-timeout 3600   --enable-metrics  --disable-radix-cache  --host 0.0.0.0 \
      --served-model-name  deepseek --mem-fraction-static 0.82  --tp 1 > $PWD/deepseek_v2_lite_baseline.log 2>&1 &

echo "Waiting for server to start..."

while true; do
  if nc -z 0.0.0.0 30000; then
    $(cat $PWD/deepseek_v2_lite_baseline.log)
    break
  else
    sleep 1
  fi
done

python3 -m sglang.bench_serving --backend sglang --dataset-name  sharegpt  --host 0.0.0.0 --port 30000  \
--model  $MODEL_PATH  --dataset-path  $DATASET_PATH  --num-prompt 1000  --benchmark-save-path $PWD/result_v2_baseline --request-rate 5 --request-rate-extent 10

# kill the server
$(pkill -f sglang)
