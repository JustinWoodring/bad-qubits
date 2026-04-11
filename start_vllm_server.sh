#!/usr/bin/env bash
# Start the vLLM generation server across both GPUs before running training.
#
# Usage:
#   ./start_vllm_server.sh          # start server in background, then train
#   ./start_vllm_server.sh --fg     # foreground (for debugging)
#
# The server uses tensor parallelism across GPU 0 and GPU 1 so neither card
# needs to hold the full model.  Training still loads the model in 4-bit via
# unsloth on GPU 0.

MODEL="Qwen/Qwen2.5-Coder-7B"
HOST="0.0.0.0"
PORT=8000
TP=2
GPU_MEM=0.85   # fraction of each GPU reserved for vLLM KV cache
MAX_MODEL_LEN=2048

if [[ "$1" == "--fg" ]]; then
    exec trl vllm-serve \
        --model "$MODEL" \
        --tensor-parallel-size "$TP" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu_memory_utilization "$GPU_MEM" \
        --max_model_len "$MAX_MODEL_LEN"
fi

# Background mode: write log to vllm_server.log
echo "Starting vLLM server (TP=$TP) on ${HOST}:${PORT} …"
trl vllm-serve \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu_memory_utilization "$GPU_MEM" \
    --max_model_len "$MAX_MODEL_LEN" \
    > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID  (log: vllm_server.log)"
echo "$VLLM_PID" > vllm_server.pid

# Wait until the server is ready
echo -n "Waiting for server to be ready"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "Server is up.  Run training with:"
echo "  python main.py train"
echo ""
echo "To stop the server afterwards:"
echo "  kill \$(cat vllm_server.pid)"
