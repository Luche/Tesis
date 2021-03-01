python generate_cmlm.py '../liputan6/data_dir' \
  --path checkpoints/checkpoint_last.pt \
  --task bert_xymasked_wp_seq2seq --remove-bpe --max-sentences 5 \
  --decoding-iterations 20 --decoding-strategy mask_predict \
  --bert-model-name indolem/indobert-base-uncased --no-repeat-ngram-size 3 \
  --length-beam 3   