if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=ConvTimeNet

CTX=1
root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2023
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --CTX $CTX \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --enable_res_param 1 \
      --re_param 1 \
      --e_layers 6 \
      --dw_ks '9,11,15,21,29,39' \
      --patch_ks 32 \
      --patch_sd 0.5 \
      --d_model 64 \
      --d_ff 256 \
      --dropout 0.2 \
      --des 'Exp' \
      --train_epochs 10 \
      --itr 3 \
      --batch_size 256 \
      --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done