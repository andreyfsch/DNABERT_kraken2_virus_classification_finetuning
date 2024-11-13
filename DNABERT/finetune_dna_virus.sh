cd examples

export KMER=6
export MODEL_PATH=../6-new-12w-0

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnavirus \
    --do_train \
    --do_eval \
    --data_dir ../../kraken2_viral_data \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir ../../DNABERT_output \
    --evaluate_during_training \
    --logging_steps 500 \
    --save_steps 100 \
    --max_steps 5000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8 \
    --fp16
