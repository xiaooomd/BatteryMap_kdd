#!/bin/bash
# Batch train CPLSTM on all small datasets

# datasets=(
#     "HUST" "MATR" "SNL" "RWTH" "MICH" "MICH_EXP" 
#     "CALCE" "HNEI" "Tongji" "Stanford" "ISU_ILCC" "XJTU"
#     "ZN_2024" "CALB_2024" "NAion_2024"
# )

datasets=(
    "HUST" "MATR" "SNL" "RWTH" "MICH" "MICH_EXP" 
    "CALCE" "HNEI" "Tongji" "Stanford" "ISU_ILCC" "XJTU"
)

# CPLSTM training parameters
model_name=CPLSTM
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00005
master_port=29507
num_process=2
batch_size=16
seq_len=1
accumulation_steps=1
lstm_layers=2
d_model=256
d_ff=128
loss=MSE
e_layers=3
d_layers=2
dropout=0.1
charge_discharge_length=300
patience=5
lradj=constant
n_heads=8
seed=2021

data=Dataset_original
root_path=./dataset
comment='CPLSTM'
task_name=classification

for dataset in "${datasets[@]}"
do
    echo "=========================================="
    echo "Start training dataset: $dataset"
    echo "=========================================="
    
    checkpoints=./checkpoints/CPLSTM_${dataset}_$(date +%Y%m%d_%H%M%S)
    
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes $num_process --num_machines 1 --mixed_precision no --dynamo_backend no --main_process_port $master_port run_main.py \
      --task_name $task_name \
      --data $data \
      --is_training 1 \
      --root_path $root_path \
      --model_id CPLSTM_${dataset} \
      --model $model_name \
      --features MS \
      --seq_len $seq_len \
      --label_len 50 \
      --factor 3 \
      --enc_in 3 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1 \
      --seed $seed \
      --class_num 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --model_comment $comment \
      --accumulation_steps $accumulation_steps \
      --charge_discharge_length $charge_discharge_length \
      --dataset $dataset \
      --num_workers 32 \
      --e_layers $e_layers \
      --lstm_layers $lstm_layers \
      --d_layers $d_layers \
      --patience $patience \
      --n_heads $n_heads \
      --early_cycle_threshold $early_cycle_threshold \
      --dropout $dropout \
      --lradj $lradj \
      --loss $loss \
      --checkpoints $checkpoints
    
    echo "Dataset $dataset training completed!"
    echo ""
done

echo "=========================================="
echo "All datasets training completed!"
echo "=========================================="
