python train.py '../liputan6/data_dir_xNLU' \
  --task bert_xymasked_wp_seq2seq -s de -t en --memory-efficient-fp16 \
  -a bert2bert --train-from-scratch --no-noisy-source --save-interval-updates 10000 \
  --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr '1e-07' \
  --lr 0.0005 --min-lr '1e-09' --device-id 0 --finetune-whole-encoder \
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 \
  --weight-decay 0.01 --max-tokens 2000 --update-freq 16 --max-update 200000 \
  --left-pad-source False --adapter-dimension 512 --max-source-positions 512 --max-target-positions 512 \
  --bert-model-name indolem/indobert-base-uncased --decoder-bert-model-name indobenchmark/indobert-base-p2