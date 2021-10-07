python train.py '../data/data_dir' \
  --task bert_xymasked_wp_seq2seq -s de -t en --fp16 \
  -a bert2bert --train-from-scratch \
  --optimizer adam --adam-betas '(0.9, 0.999)' --finetune-whole-encoder --finetune-embeddings \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 25.0 \
  --lr 5e-5 --warmup-init-lr 1e-7 --min-lr 1e-9 --device-id 0 --no-epoch-checkpoints --save-interval-updates 5000 \
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 \
  --weight-decay 0.0 --max-tokens 4096 --update-freq 5 --max-update 300000 \
  --left-pad-source False --max-source-positions 512 --max-target-positions 512 \
  --bert-model-name indolem/indobert-base-uncased --decoder-bert-model-name indolem/indobert-base-uncased