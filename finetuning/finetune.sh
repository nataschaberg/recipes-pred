mkdir experiments

for epoch in 2
do
    python lm_finetuning.py \
        --model_name_or_path distilgpt2 \
        --model_type gpt2 \
        --train_data_file data_train.txt \
        --output_dir experiments/epochs_$epoch \
        --do_train \
        --overwrite_output_dir \
        --num_train_epochs $epoch \
        --save_total_limit 5
done