# -*- coding: utf-8 -*-
"""
Streaming processing demo

"""

import argparse
import numpy as np
import pandas as pd
import wave
import time
import torch
import soundfile as sf
# import textgrid
# import glob
import os
from tqdm import tqdm


def get_recognize_frame(data, speech2text, mode='non_parallel'):
    output_list = []
    enct_list = []
    dect_list = [] 
    speech = data#.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    #sim_chunk_length =  800 # 640
    
    size = 4 # 20
    kernel_2, stride_2 = 3, 2  # sub = 4
    sim_chunk_length = (((size + 2) * 2) + (kernel_2 - 1) * stride_2) * 128

    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            hyps, enc_time, st_dec = speech2text.streaming_decode(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if mode == 'parallel':
                hyps = hyps[1]
            results = speech2text.hypotheses_to_results(hyps)    
            
            enct_list.append(enc_time)
            

            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
            else:
                output_list.append("")
            et_dec = time.time() 
            dec_time = et_dec - st_dec
            dect_list.append(dec_time)

        hyps, enc_time, st_dec = speech2text.streaming_decode(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
        if mode == 'parallel':
                hyps = hyps[1]
        results = speech2text.hypotheses_to_results(hyps)
        
    else:
        hyps, enc_time, st_dec = speech2text.streaming_decode(speech, is_final=True)
        if mode == 'parallel':
                hyps = hyps[1]
        results = speech2text.hypotheses_to_results(hyps)      
    
    #nbests = [text for text, token, token_int, hyp in results]
    if results is not None and len(results) > 0:
        nbests = [text for text, token, token_int, hyp in results]
        text = nbests[0] if nbests is not None and len(nbests) > 0 else ""        
        output_list.append(text)
    else:
        output_list.append("")
    enct_list.append(enc_time)
    et_dec = time.time()
    dec_time = et_dec - st_dec
    dect_list.append(dec_time)
    # if mode == "parallel":
    enct_avg = round(sum(enct_list) / (len(speech)//sim_chunk_length),5)
    dect_avg = round(sum(dect_list) / (len(speech)//sim_chunk_length),5)
    # else:
    #     enct_list2 = [enct_list[0]]
    #     for i in range(1,len(enct_list)):
    #         t = enct_list[i] + enct_list[i-1] + dect_list[i-1]
    #         enct_list2.append(t)
    #     enct_avg = round(sum(enct_list2) / (len(speech)//sim_chunk_length),5)
    #     dect_avg = round(sum(dect_list) / (len(speech)//sim_chunk_length),5)
    # print("Encoding avg delay:{}, Decoding avg delay:{}".format(enct_avg, dect_avg))
        
    return output_list, enct_avg, dect_avg

def run(args, speech2text, split='test'):
    
    size = 4
    kernel_2, stride_2 = 3, 2  # sub = 4
    sim_chunk_length = (((size + 2) * 2) + (kernel_2 - 1) * stride_2) * 128
        
    with open('dump/raw/{}/wav.scp'.format(split), 'r') as f:
        lines = f.readlines()
        
    paths = []
    names = []
    for line in lines:
        name, path = line.split()
        if '.flac' in path:
            names.append(name)
            paths.append(path)
            
    with open('dump/raw/{}/text'.format(split), 'r') as f:
        lines = f.readlines()

    transcripts = []
    for i, line in enumerate(lines):
        splits = line.split()
        name = splits[0]
        text = splits[1:]
        assert name == names[i], 'file is not much'
        if '.flac' in path:
            transcripts.append(text)
       
    results = []
    texts = []
    ets = []
    dts = []
    is_ok = []
    # for i in tqdm(range(len(paths))):
    for i in tqdm(range(5)):
        name = names[i]
        wav_path = paths[i]
        # tg_path = os.path.join(TEXTGRID , split, "{}.TextGrid".format(name))
        # print("name:{}".format(name))
        wav, samplerate = sf.read(wav_path)
        # tg = textgrid.TextGrid.fromFile(tg_path)
        
        text, et, dt = get_recognize_frame(wav, speech2text, args.mode)
        # print("text: {}".format(text))
        texts.append(text)
        ets.append(et)
        dts.append(dt)
    # print(texts[-1][-1])
    time1 = np.array(ets)
    time2 = np.array(dts)
    time3 = np.add(time1,time2)
    print('Enc_time 50: {}s, Enc_time 90: {}s'.format(np.percentile(time1, 50), np.percentile(time1, 90)))  
    print('Dec_time 50: {}s, Dec_time 90: {}s'.format(np.percentile(time2, 50), np.percentile(time2, 90)))
    print('Total_time 50: {}s, Total_time 90: {}s'.format(np.percentile(time3, 50), np.percentile(time3, 90)))  
    
                
# TEXTGRID = './alignments'
if __name__ == '__main__':

    # asr_train_asr_cbs_transducer_onepass_8412_raw_en_bpe80
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='asr_train_asr_cbst_conformer_op_raw_en_bpe500_sp', help='exp name')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu id')
    parser.add_argument('--beamsize', type=int, default=10, help='beam size')
    parser.add_argument('--mode', type=str, default="parallel", help='decoding mode', choices=('non_parallel', 'parallel'),)
    args = parser.parse_args()            
       
    exp = args.exp
    if args.gpuid >= 0:
        device = 'cuda:{}'.format(args.gpuid)
    else:
        device = 'cpu'
    
    
    
    from espnet2.bin.asr_dual_transducer_inference import Speech2Text
    # /mnt/kiso-qnap/zhao/g10work/espnet/egs2/wsj/asr1/exp/{}/config.yaml
    # /mnt/kiso-qnap/zhao/g11work/espnet/egs2/wsj/asr1/exp/asr_train_asr_cbs_transducer_onepass_8412_raw_en_bpe80/config.yaml
    #  asr_model_file="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/wsj/asr1/exp/{}/valid.loss_transducer.ave_10best.pth".format(exp),  
    speech2text = Speech2Text(
        asr_train_config="/groups/gca50130/zhao/work/onepass/espnet/egs2/tedlium2/asr1/exp/{}/config.yaml".format(exp),
        asr_model_file="/groups/gca50130/zhao/work/onepass/espnet/egs2/tedlium2/asr1/exp/{}/valid.loss.ave_10best.pth".format(exp),
        token_type=None,
        bpemodel=None,
        beam_search_config={"search_type": "maes"},
        beam_size=args.beamsize,
        lm_weight=0.0,
        nbest=1,
        device = device,
        streaming = True,
    )

    splits = ['test']
    for split in splits:
        run(args, speech2text, split)