#!/bin/bash

#LLAMA_SERVER_BIN=~/llama.cpp/build/bin/llama-server
LLAMA_SERVER_BIN=llama-server

#Default values for CLI arguments 
MODEL_DIR=""
PORT=8000
HOST="127.0.0.1"
CTX_SIZE=1024
N_GPU_LAYERS=99

# Subdirectory names within MODEL_DIR
REASONING_SUBDIR="reasoning"
NON_REASONING_SUBDIR="non_reasoning"

# CLI arguments parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-dir)
      MODEL_DIR=$2
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --ctx-size)
      CTX_SIZE="$2"
      shift 2
      ;;
    --ngl)
      N_GPU_LAYERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model-dir /path/to/models [--port 8000] [--host 127.0.0.1]"
      exit 1
      ;;
  esac
done

if [ -z "$MODEL_DIR" ]; then
  echo "--model-dir is required"
  echo "Usage: $0 --model-dir /path/to/models [--port 8000] [--host 127.0.0.1]"
  exit 1
fi 

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory does not exist: $MODEL_DIR"
  exit 1
fi

PYTHON_PID=""
SERVER_PID=""

cleanup() {
  if [ ! -z "$PYTHON_PID" ]; then
    echo "Stopping main.py (PID: $PYTHON_PID)..."
    kill $PYTHON_PID 2>/dev/null
    wait $PYTHON_PID 2>/dev/null
    PYTHON_PID=""
  fi
  
  if [ ! -z "$SERVER_PID" ]; then
    echo "Shutting down llama-server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    SERVER_PID=""
  fi
  exit
}
trap cleanup INT TERM

process_models(){
  local dir="$1"
  local is_reasoning="$2"
  
  # Check if directory exists before looping to prevent error output
  if [ ! -d "$dir" ]; then
    echo "Directory $dir does not exist, skipping..."
    return
  fi
  
  for MODEL_PATH in "$dir"/*.gguf; do
    [ -e "$MODEL_PATH" ] || continue 
    
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    echo ""
    echo "CURRENT MODEL : $MODEL_NAME"
    echo ""
    
    CMD=(
      "$LLAMA_SERVER_BIN"
      "-m" "$MODEL_PATH"
      "--port" "$PORT"
      "--host" "$HOST"
      "-c" "$CTX_SIZE"
      "-ngl" "$N_GPU_LAYERS"
    )
    
    # Check if the model is a reasoning model and append the disable flag
    if [ "$is_reasoning" = "true" ]; then
      echo "Disabling reasoning ..."
      CMD+=("--chat-template-kwargs" '{"enable_thinking": false}')
    fi

    # Start the llama.cpp server in the background
    "${CMD[@]}" > "llama_server_${MODEL_NAME}.log" 2>&1 &
    SERVER_PID=$! # capture llama-server pid

    echo "llama.cpp server initiating (PID: $SERVER_PID)..."
    
    # A while true loop to ensure that llama.cpp is up and running 
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while true; do
      # Ping llama.cpp health endpoint
      if curl -s --max-time 2 "http://$HOST:$PORT/health" | grep -q "ok"; then
        echo "Server is Live !! Initiating test script"
        break
      fi
      
      RETRY_COUNT=$((RETRY_COUNT + 1))
      if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
        echo "ERROR: Server failed to start within the timeout period."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null || true
        SERVER_PID="" 
        break 
      fi
      
      sleep 2
    done
    
    # If server failed to start, skip to next model
    if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
      continue
    fi

    # Execute python script and capture the PID
    python3 main.py --model_name "$MODEL_NAME" --port "$PORT" --verbose &
    PYTHON_PID=$!
    
    # Wait for python script to finish 
    wait $PYTHON_PID
    PYTHON_EXIT_CODE=$?
    PYTHON_PID=""
    
    if [ "$PYTHON_EXIT_CODE" -ne 0 ]; then
      echo "python script failed to launch or encountered an error "
    fi

    # Killing llama.cpp process
    echo "Shutting down server..."
    kill $SERVER_PID

    # Wait for the process to fully exit 
    wait $SERVER_PID 2>/dev/null || true 
    SERVER_PID="" 

    sleep 5 
    echo ""
  done
}

echo "Starting automated model benchmarking script"
echo "Model directory: $MODEL_DIR"
echo "llama server config: "
echo "Port: $PORT | HOST: $HOST | Context: $CTX_SIZE | GPU Layers: $N_GPU_LAYERS"

process_models "$MODEL_DIR/$NON_REASONING_SUBDIR" "false"
process_models "$MODEL_DIR/$REASONING_SUBDIR" "true"

echo "Completed processing all models in the queue"
