python preprocess.py -train_src ./data/train.txt.src.tokenized.fixed.cleaned.final.truncated \
                     -train_tgt ./data/train.txt.src.tokenized.fixed.cleaned.final.truncated \
                     -valid_src ./data/val.txt.src.tokenizd.fixed.cleaned.final.truncated \
                     -valid_tgt ./data/val.txt.tgt.tokenizd.fixed.cleaned.final.truncated \
                     -save_data ./newser_sent_500/newser_sents \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -src_seq_length_trunc 500 \
                     -tgt_seq_length_trunc 300 \
                     -dynamic_dict \
                     -share_vocab \
                     -max_shard_size 10000000
read -p "hold a min..."