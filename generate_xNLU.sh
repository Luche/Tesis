python generate_cmlm.py '../liputan6/data_dir_xNLU' \
  --path checkpoints/checkpoint_last.pt \
  --task bert_xymasked_wp_seq2seq --remove-bpe wordpiece --max-sentences 5 \
  --decoding-iterations 10 --decoding-strategy mask_predict \
  --bert-model-name indolem/indobert-base-uncased --no-repeat-ngram-size 3