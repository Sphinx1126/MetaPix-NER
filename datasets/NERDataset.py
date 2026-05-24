# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:57:47 2024

@author: 28257
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import random
from enum import Enum
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from os.path import join
from loguru import logger
import re

class NERDataset(Dataset):
    def __init__(self, 
                 data_path, book, mode, sample_rate=1.0, sample_num=None):
        super(NERDataset, self).__init__()
        self.book=book
        self.mode=mode
        with open(data_path,'rb+') as fp:
            data=pickle.load(fp)[self.book]
            if self.mode=='train':
                cand_index=list(set(range(len(data['bio'])))-set(data['query_index']))
                sample_num=int(len(cand_index)*sample_rate) if sample_num is None else sample_num
                sample_index=random.sample(cand_index, sample_num)
                self.labels=[torch.tensor(data['bio'][i]) for i in range(len(data['bio'])) if i in sample_index]
                self.texts=[torch.tensor(data['text'][i]) for i in range(len(data['bio'])) if i in sample_index]
                self.masks=[torch.tensor(data['mask'][i]) for i in range(len(data['bio'])) if i in sample_index]
            else:
                self.labels=[torch.tensor(data['bio'][i]) for i in range(len(data['bio'])) if i in data['query_index']]
                self.texts=[torch.tensor(data['text'][i]) for i in range(len(data['bio'])) if i in data['query_index']]
                self.masks=[torch.tensor(data['mask'][i]) for i in range(len(data['bio'])) if i in data['query_index']]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        texts=self.texts[index]
        masks=self.masks[index]
        labels=self.labels[index]
        return texts,masks,labels