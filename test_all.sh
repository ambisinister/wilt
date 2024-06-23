#!/bin/bash

models=(
    "llama3-70b-8192"
    "llama3-8b-8192"
    "gemma-7b-it"
    "mixtral-8x7b-32768"
)

PYTHON_SCRIPT="eval.py"

mkdir -p results

run_model() {
    local model=$1
    echo "Running tests for model: $model"
    python "$PYTHON_SCRIPT" --model "$model"
    echo "Completed tests for model: $model"
    echo "----------------------------------------"
}

for model in "${models[@]}"; do
    run_model "$model"
done

echo "Combining CSV results..."
{
    echo "model,accuracy,avg_guesses,points"
    for model in "${models[@]}"; do
        if [[ -f "./results/${model}_results.csv" ]]; then
            tail -n 1 "./results/${model}_results.csv" | sed "s/^/${model},/"
        fi
    done
} > results/all_models_results.csv

echo "All tests completed. Combined results saved in combined_results/all_models_results.csv"
