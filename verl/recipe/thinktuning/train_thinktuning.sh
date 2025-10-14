#!/bin/bash
set -x

source .venv/bin/activate

# HF and wandb credentials
# HF and wandb credentials
export HF_TOKEN=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
export WANDB_API_KEY=167f77df3e2d14177e53333df1eb899324ea1450
export WANDB_API_KEY=c09d92ce856d5de5fb37c7eb46706bff574d7b0e
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export HOME="/scr/arrv"
export HF_HOME="/scr/arrv/tmp"
export TRANSFORMERS_CACHE="/scr/arrv/tmp/transformers"
export HF_DATASETS_CACHE="/scr/arrv/tmp/datasets"
export HUGGINGFACE_HUB_CACHE="/scr/arrv/tmp/hub"


# KINGKONG3 Specific Environment Variables
export NCCL_CUDA_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs
export LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH


echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "HF_MODULES_CACHE: $HF_MODULES_CACHE"
echo "HOME: $HOME"

export RAY_ADDRESS="local"
export RAY_TMPDIR="/scr/arrv/tmp"
export TMPDIR="/scr/arrv/tmp"
export REPO_ROOT=$(pwd)
echo "REPO_ROOT: $REPO_ROOT"
echo "HOME DIR: $HOME"


export DATA_ROOT="$REPO_ROOT/data"

# Environment setup
# module load mamba
# module load cuda-12.4.1-gcc-12.1.0
# module load gcc-12.1.0-gcc-11.2.0
# source activate verl-final

# Configuration variables
CUDA_DEVICES=${CUDA_DEVICES:-"6,7"}  # Default to CUDA device 1
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-3B-Instruct"}  
DATASET=${DATASET:-"gsm8k"}  # Default to GSM8K dataset (options: gsm8k, math, dapo_math_5k, dapo_math_7k, dapo_math_12k, dapo_math_full, deep_math_6.5d (or other difficulties))

USE_EVALUATION_DATASET=${USE_EVALUATION_DATASET:-"False"}

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=1

# Function to prepare data if it doesn't exist
prepare_dataset() {
    local dataset_name=$1
    local data_dir="$DATA_ROOT/$dataset_name"
    local train_file="$data_dir/train.parquet"
    local test_file="$data_dir/test.parquet"
    local evaluation_file="$data_dir/evaluation.parquet"

    if [ "$USE_EVALUATION_DATASET" = "True" ]; then
    
        if [ -f "$train_file" ] && [ -f "$test_file" ] && [ -f "$evaluation_file" ]; then
            echo "✓ $dataset_name dataset and the evaluation dataset already exists at $data_dir"
            return 0
        fi
    else
        if [ -f "$train_file" ] && [ -f "$test_file" ]; then
            echo "✓ $dataset_name dataset already exists at $data_dir"
            return 0
        fi
    fi

    if [ ! -f "$train_file" ] && [ ! -f "$test_file" ]; then
    
        echo "📥 Downloading and preparing $dataset_name dataset..."
        mkdir -p "$data_dir"
        
        # Get script directory and verl root  
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
        echo "VERL_ROOT: $VERL_ROOT"
        cd "$VERL_ROOT"

        
        case $dataset_name in
            "gsm8k")
                PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/gsm8k.py --local_dir "$data_dir"
                ;;
            "math")
                PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/math_dataset.py --local_dir "$data_dir"
                ;;
            "dapo_math_5k")
                PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/dapo_math.py --local_dir "$data_dir" --split "5k"
                ;;
            "dapo_math_7k")
                PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/dapo_math.py --local_dir "$data_dir" --split "7k"
                ;;
            "dapo_math_12k")
                PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/dapo_math.py --local_dir "$data_dir" --split "12k"
                ;;
            "dapo_math_full")
                PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/dapo_math.py --local_dir "$data_dir" --split "full"
                ;;
            "deep_math_6.5d")
                PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/deep_math.py --local_dir "$data_dir" --split "6.5d"
                ;;
            *)
                echo "❌ Unknown dataset: $dataset_name"
                exit 1
                ;;
        esac

        if [ -f "$train_file" ] && [ -f "$test_file" ]; then
            echo "✅ Successfully prepared $dataset_name dataset"
        else
            echo "❌ Failed to prepare $dataset_name dataset"
            exit 1
        fi
    fi

    # Prepare evaluation dataset if needed
    if [ "$USE_EVALUATION_DATASET" = "True" ] && [ ! -f "$evaluation_file" ]; then
        echo "📥 Downloading and preparing $dataset_name with evaluation dataset..."        
        # Get script directory and verl root  
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
        echo "VERL_ROOT: $VERL_ROOT"
        cd "$VERL_ROOT"

        PYTHONPATH="$VERL_ROOT:$PYTHONPATH" python3 examples/data_preprocess/evaluation_dataset.py --local_dir "$data_dir" --data_source "/scr/arrv/AdvThinking/data/misc/processed_evaluation_dataset.json"
        
        if [ -f "$evaluation_file" ]; then
            echo "✅ Successfully prepared $dataset_name evaluation dataset"
        else
            echo "❌ Failed to prepare $dataset_name evaluation dataset"
            exit 1
        fi
    fi
}

# Prepare dataset
echo "🔍 Checking dataset: $DATASET"
prepare_dataset "$DATASET"

# Set file paths
train_files="['$DATA_ROOT/$DATASET/train.parquet']"
test_files="['$DATA_ROOT/$DATASET/test.parquet']"

echo "📊 Training files: $train_files"
echo "📊 Test files: $test_files"


EXPERIMENT_NAME="PrayToGod4"

echo "Starting trainer with experiment: $EXPERIMENT_NAME"


python3 -m recipe.multitask.main_multitask \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@