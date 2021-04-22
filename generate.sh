python generate_cmlm.py '../liputan6/data_dir' \
  --path '../checkpoint9.pt' \
  --task bert_xymasked_wp_seq2seq --remove-bpe wordpiece --max-sentences 5 \
  --decoding-iterations 30 --decoding-strategy mask_predict \
  --bert-model-name indolem/indobert-base-uncased --no-repeat-ngram-size 3