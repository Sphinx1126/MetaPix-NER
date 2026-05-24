# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:06:39 2024

@author: 28257
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import random

class MetaPixNER(Dataset):
    def __init__(self, 
                 data_dir, mode,
                 metatrain_iterations,meta_batch_size,update_batch_size,query_size,
                 mixup=False,mixup_prob=0.0,mixup_u=0.0):
        super(MetaPixNER, self).__init__()
        self.mode = mode
        self.metatrain_iterations=metatrain_iterations
        self.meta_batch_size=meta_batch_size
        self.update_batch_size=update_batch_size
        self.mixup=mixup
        if self.mixup:
            self.mixup_prob=mixup_prob
            self.mixup_u=mixup_u
        self.query_size=query_size
        
        with open(data_dir+self.mode+'.pkl','rb+') as fp:
            self.data=pickle.load(fp)
    
    def __len__(self):
        return self.metatrain_iterations if self.mode=='train' else 1
    
    def sample(self,book,sample_size,mode='support'):
        if mode=='support':
            data_size=len(self.data[book]['text'])
            sample_size=min(data_size-len(self.data[book]['query_index']),sample_size)
            sample_index=random.sample(list(set(range(data_size))-set(self.data[book]['query_index'])), sample_size)
        else:
            data_size=len(self.data[book]['query_index'])
            sample_size=min(data_size,sample_size)
            sample_index=random.sample(self.data[book]['query_index'], sample_size)
        
        period=[self.data[book]['period']]*sample_size
        text=[self.data[book]['text'][i] for i in sample_index]
        mask=[self.data[book]['mask'][i] for i in sample_index]
        ner_label=[self.data[book]['bio'][i] for i in sample_index]
        
        return text,mask,ner_label,period
    
    def __getitem__(self, index):
        texts_spt, masks_spt, ner_labels_spt, periods_spt=[],[],[],[]
        texts_qry, masks_qry, ner_labels_qry, periods_qry=[],[],[],[]
        
        if self.mode=='train':
            for i in range(self.meta_batch_size):
                if self.mixup and random.random()<self.mixup_prob:
                    cand_book=random.sample(self.data.keys(), 2)
                    sample_size1_spt=int(self.update_batch_size*self.mixup_u)
                    text1_spt,mask1_spt,ner_label1_spt,period1_spt=self.sample(cand_book[0], sample_size1_spt, 'support')
                    text2_spt,mask2_spt,ner_label2_spt,period2_spt=self.sample(cand_book[0], self.update_batch_size-len(text1_spt), 'support')
                    text_spt=torch.tensor(text1_spt+text2_spt)
                    mask_spt=torch.tensor(mask1_spt+mask2_spt)
                    ner_label_spt=torch.tensor(ner_label1_spt+ner_label2_spt)
                    period_spt=torch.mean(torch.tensor(period1_spt+period2_spt).float(),dim=0)
                    
                    sample_size1_qry=int(self.query_size*self.mixup_u)
                    text1_qry,mask1_qry,ner_label1_qry,period1_qry=self.sample(cand_book[1], sample_size1_qry, 'query')
                    text2_qry,mask2_qry,ner_label2_qry,period2_qry=self.sample(cand_book[1], self.query_size-sample_size1_qry, 'query')
                    text_qry=torch.tensor(text1_qry+text2_qry)
                    mask_qry=torch.tensor(mask1_qry+mask2_qry)
                    ner_label_qry=torch.tensor(ner_label1_qry+ner_label2_qry)
                    period_qry=torch.mean(torch.tensor(period1_qry+period2_qry).float(),dim=0)
                else:
                    cand_book=random.sample(self.data.keys(), 1)
                    text_spt,mask_spt,ner_label_spt,period_spt=self.sample(cand_book[0], self.update_batch_size, 'support')
                    text_spt=torch.tensor(text_spt)
                    mask_spt=torch.tensor(mask_spt)
                    ner_label_spt=torch.tensor(ner_label_spt)
                    period_spt=torch.mean(torch.tensor(period_spt).float(),dim=0)
                    
                    text_qry,mask_qry,ner_label_qry,period_qry=self.sample(cand_book[0], self.query_size, 'query')
                    text_qry=torch.tensor(text_qry)
                    mask_qry=torch.tensor(mask_qry)
                    ner_label_qry=torch.tensor(ner_label_qry)
                    period_qry=torch.mean(torch.tensor(period_qry).float(),dim=0)
                    
                texts_spt.append(text_spt)
                masks_spt.append(mask_spt)
                ner_labels_spt.append(ner_label_spt)
                periods_spt.append(period_spt)
                texts_qry.append(text_qry)
                masks_qry.append(mask_qry)
                ner_labels_qry.append(ner_label_qry)
                periods_qry.append(period_qry)
        else:
            for book in self.data:
                text_spt,mask_spt,ner_label_spt,period_spt=self.sample(book, self.update_batch_size, 'support')
                text_spt=torch.tensor(text_spt)
                mask_spt=torch.tensor(mask_spt)
                ner_label_spt=torch.tensor(ner_label_spt)
                period_spt=torch.mean(torch.tensor(period_spt).float(),dim=0)
                
                text_qry,mask_qry,ner_label_qry,period_qry=self.sample(book, self.query_size, 'query')
                text_qry=torch.tensor(text_qry)
                mask_qry=torch.tensor(mask_qry)
                ner_label_qry=torch.tensor(ner_label_qry)
                period_qry=torch.mean(torch.tensor(period_qry).float(),dim=0)
                
                texts_spt.append(text_spt)
                masks_spt.append(mask_spt)
                ner_labels_spt.append(ner_label_spt)
                periods_spt.append(period_spt)
                texts_qry.append(text_qry)
                masks_qry.append(mask_qry)
                ner_labels_qry.append(ner_label_qry)
                periods_qry.append(period_qry)
        
        
        return torch.stack(texts_spt), torch.stack(masks_spt), torch.stack(ner_labels_spt), torch.stack(periods_spt), \
            torch.stack(texts_qry), torch.stack(masks_qry), torch.stack(ner_labels_qry), torch.stack(periods_qry)           
        '''
        return texts_spt, masks_spt, ner_labels_spt, periods_spt, texts_qry, masks_qry, ner_labels_qry, periods_qry
        '''
        
        


if __name__=='__main__':
    ds=MetaPixNER('../data/','train',
                  15000,25,5,20,
                  True,0.8,0.3)
    ds2=MetaPixNER('../data/','test',
                  15000,25,5,20)
    texts_spt, masks_spt, ner_labels_spt, periods_spt, texts_qry, masks_qry, ner_labels_qry, periods_qry=ds2[0]