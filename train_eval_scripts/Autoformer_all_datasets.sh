#!/bin/bash
# Batch train Autoformer on all small datasets

# Define all datasets (based on datasets with complete train/val/test in data_split_recorder.py)
# NA, ZN, CALB uniformly use the 2024 version
# datasets=(
#     "HUST" "MATR" "SNL" "RWTH" "MICH" "MICH_EXP" 
#     "CALCE" "HNEI" "Tongji" "Stanford" "ISU_ILCC" "XJTU"
#     "ZN_2024" "CALB_2024" "NAion_2024"
# )

datasets=(
    "HUST" "MATR" "SNL" "RWTH" "MICH" "MICH_EXP" 
    "CALCE" "HNEI" "Tongji" "Stanford" "ISU_ILCC" "XJTU"
)

# Shared training parameters
model_name=Autoformer
train_epochs=100
early_cycle_threshold=100
learning_rate=0.00005
master_port=29501
num_process=2
batch_size=4
n_heads=4
seq_len=1
accumulation_steps=4
lstm_layers=6
e_layers=2
d_layers=2
d_model=128
d_ff=256
dropout=0.1
charge_discharge_length=300
patience=5
lradj=constant
loss=MSE
patch_len=50
stride=50
seed=2021

data=Dataset_original
root_path=./dataset
comment='Autoformer'
task_name=classification

# Loop to train each dataset
for dataset in "${datasets[@]}"
do
    echo "=========================================="
    echo "Start training dataset: $dataset"
    echo "=========================================="

    # Set a dedicated checkpoint path for this dataset
    checkpoints=./checkpoints/Autoformer_${dataset}_$(date +%Y%m%d_%H%M%S)
    
    CUDA_VISIBLE_DEVICES=4,5 accelerate launch --multi_gpu --num_processes $num_process --num_machines 1 --mixed_precision no --dynamo_backend no --main_process_port $master_port run_main.py \
      --task_name $task_name \
      --data $data \
      --is_training 1 \
      --root_path $root_path \
      --model_id Autoformer_${dataset} \
      --model $model_name \
      --features MS \
      --seed $seed \
      --seq_len $seq_len \
      --label_len 50 \
      --factor 3 \
      --enc_in 3 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1 \
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
