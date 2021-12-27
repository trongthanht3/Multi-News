import os
import fastapi
import sys
from fastapi import FastAPI, Form
sys.path.append('Hi_MAP')
from onmt.translate import TranslationServer, ServerModelError
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

from time import time
import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import os
import argparse

STATUS_OK = "ok"
STATUS_ERROR = "error"
# parser = argparse.ArgumentParser()

class OPT_C:
    models = ['Hi_MAP/model_newser_atten/Feb17__step_20000.pt']
    data_type = 'text'
    src = "Hi_MAP/data/test.txt.src.tokenized.fixed.cleaned.final.truncated"
    src_dir = ''
    tgt = None
    output = "output.txt"
    report_bleu = False
    report_rouge = False
    dynamic_dict = False
    share_vocab = False
    fast = False
    beam_size = 4
    min_length = 200
    max_length = 300
    max_sent_length = None
    stepwise_penalty = True
    length_penalty = 'wu'
    coverage_penalty = 'summary'
    alpha = 0.9
    beta = 5.0
    block_ngram_repeat = 3
    ignore_when_blocking = ['story_separator_special_tag']
    replace_unk = False
    verbose = True
    log_file = ''
    attn_debug = False
    dump_beam = ''
    n_best = 1
    batch_size = 8
    gpu = -1
    sample_rate = 16000
    window_size = 0.02
    window_stride = 0.01
    window = 'hamming'
    image_channel_size = 3
    def __init__(self):
        self.models = ['Hi_MAP/model_newser_atten/Feb17__step_20000.pt']
        self.data_type = 'text'
        self.src = "temp/file.txt"
        self.src_dir = ''
        self.tgt = None
        self.output = "output.txt"
        self.report_bleu = False
        self.report_rouge = False
        self.dynamic_dict = False
        self.share_vocab = False
        self.fast = False
        self.beam_size = 4
        self.min_length = 200
        self.max_length = 300
        self.max_sent_length = None
        self.stepwise_penalty = True
        self.length_penalty = 'wu'
        self.coverage_penalty = 'summary'
        self.alpha = 0.9
        self.beta = 5.0
        self.block_ngram_repeat = 3
        self.ignore_when_blocking = ['story_separator_special_tag']
        self.replace_unk = False
        self.verbose = True
        self.log_file = ''
        self.attn_debug = False
        self.dump_beam = ''
        self.n_best = 1
        self.batch_size = 8
        self.gpu = -1
        self.sample_rate = 16000
        self.window_size = 0.02
        self.window_stride = 0.01
        self.window = 'hamming'
        self.image_channel_size = 3

def showme():
    print(os.getcwd())

opt = OPT_C
print("Loading Hi_MAP module...")
translator = build_translator(opt, report_score=True)
print("Loaded success")

def hi_map_summarize(text):
    f_name = str(time()) + '.txt'
    f = open("Hi_MAP/temp/"+f_name, "w", encoding='utf-8')
    f.write(text)
    f.close()

    translator.translate(src_path="Hi_MAP/temp/"+f_name,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug,
                         output_temp="Hi_MAP/temp/"+f_name+".trans")

    f_o = open("Hi_MAP/temp/"+f_name+".trans", 'r', encoding='utf-8')
    data = f_o.read()
    f_o.close()
    os.remove("Hi_MAP/temp/"+f_name)
    os.remove("Hi_MAP/temp/" + f_name + ".trans")
    # print("this is data: ", data)
    return data