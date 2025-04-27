# data/t0/t0-task-web_questions_short_general_knowledge_q.json/
# DATA_NAME_OR_PATH=data/t0/t0-ljson-sample.json
# OUTPUT_DIR=results_gpt2
#
MODEL_NAME_OR_PATH=gpt2
#HUB_TOKEN=

#export WANDB_API_KEY=
HUB_TOKEN=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

export WANDB_API_KEY=74e8368d36e7abf19bc9d56a66e74a7bea1033e5

wandb login --cloud --host https://api.wandb.ai --relogin $WANDB_API_KEY

# run_task <task_name> <data_dir> <output_dir> <retry_num>

LOG_FILE=t0_all_tasks_runs.log
MAX_RETRIES=5
echo "--------------------------------------------" >>$LOG_FILE
echo "$(date) : Started T0 Fine-Tuning Stage" >>$LOG_FILE

run_task() {

    if [ $4 -eq 0 ]; then
        echo "ERROR: Retries exceeded in Fine-Tuning for task : $1" >>$LOG_FILE
        return
    fi

    echo "Fine-Tuning for task : $1" >>$LOG_FILE

    accelerate launch --config_file configs/acc_tuner_config.yaml src/instruction_tuner.py \
        --dataset_name_or_path $2 \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --load_data_from_disk \
        --hf_access_token $HUB_TOKEN \
        --torch_dtype bfloat16 \
        --max_seq_length 4096 \
        --learning_rate 2e-3 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --preprocessing_num_workers 12 \
        --use_peft \
        --peft_lora_r 8 \
        --peft_lora_alpha 32 \
        --peft_lora_dropout 0.1 \
        --peft_target_modules c_proj,c_attn \
        --seed 23 \
        --num_train_epochs 10 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --with_tracking \
        --report_to wandb \
        --output_dir $3 \
        --hub_token $HUB_TOKEN \
        --private_repo

    if [ $? -ne 0 ]; then
        echo "WARNING : Retrying Fine-Tuning for task : $1" >>$LOG_FILE
        next_retry=$(expr $4 - 1)
        run_task $1 $2 $3 $next_retry
    else
        echo "Done Fine-Tuning for task : $1" >>$LOG_FILE
    fi
}

for t0_task_name in data/t0/*.json; do

    base_name=$(basename "$t0_task_name")
    task_name=${base_name%.json}
    output_dir=$(echo -e "models/result_${MODEL_NAME_OR_PATH}_${task_name}")
    if [ -d "$output_dir" ]; then
        echo "Skipping, since already done : $output_dir" >>$LOG_FILE
        continue
    fi
    run_task $task_name $t0_task_name $output_dir $MAX_RETRIES
done
