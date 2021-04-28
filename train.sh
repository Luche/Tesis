python train.py '../liputan6/data_dir' \
  --task bert_xymasked_wp_seq2seq -s de -t en \
  -a bert2bert --train-from-scratch --no-noisy-source \
  --optimizer adam --adam-betas '(0.9, 0.999)' \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 25.0 \
  --lr 1e-5 --warmup-init-lr 1e-7 --min-lr 1e-9 --device-id 0 \
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 \
  --weight-decay 0.0 --max-tokens 512 --update-freq 1 --max-epoch 10 \
  --left-pad-source False --max-source-positions 512 --max-target-positions 512 \
  --bert-model-name indolem/indobert-base-uncased --decoder-bert-model-name indolem/indobert-base-uncased 
  # --save-dir /mnt/9d3e4e21-bdd9-4bca-b801-87c08124cc05/LUCKY/checkpoints