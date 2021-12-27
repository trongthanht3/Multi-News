python preprocess.py -train_src ./data/train.txt.src.tokenized.fixed.cleaned.final.truncated.txt \
                     -train_tgt ./data/train.txt.src.tokenized.fixed.cleaned.final.truncated.txt \
                     -valid_src ./data/val.txt.src.tokenized.fixed.cleaned.final.truncated.txt \
                     -valid_tgt ./data/val.txt.tgt.tokenized.fixed.cleaned.final.truncated.txt \
                     -save_data ./newser_sent_500/newser_sents \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -src_seq_length_trunc 500 \
                     -tgt_seq_length_trunc 300 \
                     -dynamic_dict \
                     -share_vocab \
                     -max_shard_size 10000000
read -p "hold a min..."