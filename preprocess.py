# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:13:43 2024

@author: 28257
"""

import pandas as pd
import pickle
from transformers import AutoTokenizer,AutoModel
from collections import defaultdict
import argparse
import random
import torch
import numpy as np
from opencc import OpenCC

def set_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', default='raw/data.xlsx')
    parser.add_argument('--bert_path', default='bert/')
    parser.add_argument('--num_period', default=3, type=int)
    parser.add_argument('--bert_max_len', default=130, type=int)
    parser.add_argument('--query_ratio', default=0.4, type=float)
    parser.add_argument('--opt_dir', default='data/')

    args = parser.parse_args([])
    return args

def get_bio(text,bio2id,id2bio):
    labels=[]
    cnt=0
    id_cnt=len(bio2id)
    while cnt<len(text):
        char=text[cnt]
        if char=='{':
            cnt+=1
            char=text[cnt]
            tmp=[]
            while char!='|':
                tmp.append(char)
                cnt+=1
                char=text[cnt]
            cnt+=1
            char=text[cnt]
            label=''
            while char!='}':
                label+=char
                cnt+=1
                char=text[cnt]
            if label!='' and 'B-'+label not in bio2id:
                bio2id['B-'+label]=id_cnt
                bio2id['I-'+label]=id_cnt+1
                id2bio[id_cnt]='B-'+label
                id2bio[id_cnt+1]='I-'+label
                id_cnt+=2
            if label!='' and len(tmp)>0:
                labels.append(bio2id['B-'+label])
                for _ in tmp[1:]:
                    labels.append(bio2id['I-'+label])
        else:
            labels.append(bio2id['O'])
        cnt+=1
    return labels

if __name__=='__main__':
    args = set_args()
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    period_dict={'北齊書':0,'北史':0,'金史':2,'舊唐書':1,'舊五代史':1,'遼史':2,'南史':0,
                 '隋書':1,'魏書':0,'新唐書':1,'新五代史':1,'元史':2,'周書':0}
    bio2id={'O':0}
    id2bio={0:'O'}
    test_sets=['北史','舊唐書','金史']
    trains=defaultdict(dict)
    tests=defaultdict(dict)
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    df=pd.read_excel(args.data_path)
    converter = OpenCC('t2s')
    
    for _, row in df.iterrows():
        text=row['text']
        clean_text=row['clean_text']
        source=row['source']
        
        #clean_text=converter.convert(clean_text)
        period=[1 if i==period_dict[source] else 0 for i in range(args.num_period)]
        inputs=tokenizer(text = clean_text,
                         truncation = True,
                         padding = 'max_length',
                         max_length = args.bert_max_len,
                         return_tensors = "pt",
                         return_token_type_ids = True,
                         return_attention_mask = True,
                         return_special_tokens_mask = True,
                         return_length = True)
        input_ids=inputs.input_ids[0].tolist()
        attention_masks=inputs.attention_mask[0].tolist()
        bio=get_bio(text, bio2id, id2bio)
        
        if source in test_sets:
            if 'period' not in tests[source]:
                tests[source]['period']=period
                tests[source]['text']=[]
                tests[source]['mask']=[]
                tests[source]['bio']=[]
            tests[source]['text'].append(input_ids)
            tests[source]['mask'].append(attention_masks)
            tests[source]['bio'].append([0]+bio+[0]*(args.bert_max_len-len(bio)-1))
        else:
            if 'period' not in trains[source]:
                trains[source]['period']=period
                trains[source]['text']=[]
                trains[source]['mask']=[]
                trains[source]['bio']=[]
            trains[source]['text'].append(input_ids)
            trains[source]['mask'].append(attention_masks)
            trains[source]['bio'].append([0]+bio+[0]*(args.bert_max_len-len(bio)-1))
    
    for book in trains:
        data_size=len(trains[book]['text'])
        trains[book]['query_index']=random.sample(range(data_size), int(data_size*args.query_ratio))
    for book in tests:
        data_size=len(tests[book]['text'])
        tests[book]['query_index']=random.sample(range(data_size), int(data_size*args.query_ratio))
    
    with open('data/dict.pkl', 'wb') as fp:
        pickle.dump([period_dict,bio2id,id2bio],fp)
    with open('data/train.pkl', 'wb') as fp:
        pickle.dump(trains,fp)
    with open('data/test.pkl', 'wb') as fp:
        pickle.dump(tests,fp)
        
    

