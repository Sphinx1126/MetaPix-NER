# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:23:44 2024

@author: 28257
"""

#A14eve6y1z123
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.MetaPixDataset import MetaPixNER
from datasets.NERDataset import NERDataset
from models.maml import MAML
import transformers
from transformers import AutoTokenizer,AutoModel
import time
from loguru import logger
from tensorboardX import SummaryWriter
from os.path import join
from tqdm import tqdm
import pickle
from seqeval.metrics import precision_score,recall_score,f1_score

def set_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--train', default=True, type=bool)

    parser.add_argument('--output_path', default='output/')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--bert_path', default='SIKU-BERT/sikubert')
    
    parser.add_argument('--num_period', default=3, type=int)
    parser.add_argument('--crf_num_labels', default=7, type=int)
    parser.add_argument('--bert_max_len', default=130, type=int)
    
    parser.add_argument('--bottleneck_dim', default=64, type=int)
    parser.add_argument('--lstm_hidden_size', default=200, type=int)
    parser.add_argument('--lstm_num_layers', default=1, type=int)
    parser.add_argument('--lstm_batch_first', default=False, type=bool)
    parser.add_argument('--lstm_bidirectional', default=True, type=bool)
    parser.add_argument('--loss_lam', default=0.8, type=float)
    
    parser.add_argument('--metatrain_iterations', default=2000, type=int,
                        help='number of metatraining iterations.')
    parser.add_argument('--warmup_steps', default=800, type=int)
    parser.add_argument('--meta_batch_size', default=25, type=int, help='number of tasks sampled per meta-update')
    parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
    parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
    parser.add_argument('--num_updates', default=4, type=int, help='num_updates in maml')
    
    parser.add_argument('--update_batch_size', default=8, type=int,
                        help='number of examples used for inner gradient update (K for K-shot learning).')
    parser.add_argument('--query_size', default=16, type=int)
    parser.add_argument('--mixup', default=False, type=bool)
    parser.add_argument('--mixup_prob', default=0.0, type=float)
    parser.add_argument('--mixup_u', default=0.25, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    
    parser.add_argument('--update_batch_size_test', default=32, type=int,
                        help='number of examples used for inner gradient test (K for K-shot learning).')
    parser.add_argument('--num_updates_test', default=8, type=int, help='num_updates in maml')
    parser.add_argument('--query_size_test', default=64, type=int,
                        help='number of examples used for inner gradient test (K for K-shot learning).')

    
    parser.add_argument('--adversarial_temperature', default=5, type=int)
    
    parser.add_argument('--bs_train', default=8)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--warmup_test', default=50)
    parser.add_argument('--eval_step', default=50)
    parser.add_argument('--bs_eval', default=32)
    parser.add_argument('--save_step', default=100)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-randomSeed', default=0, type=int)
    parser.add_argument('-save', '--save_path', default=None, type=str)

    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    args = set_args()
    args.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'test-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    
    maml = MAML(args).to(args.device)
    
    
    logger.info("start testing")
    books=['北史','舊唐書','金史'] #
    book2warmup={'北史':80,'舊唐書':250,'金史':120}
    with open(args.data_dir+'dict.pkl','rb+') as fp:
        period_dict,bio2id,id2bio=pickle.load(fp)
    P,R,F1=[],[],[]
    for book in books:
        torch.manual_seed(args.randomSeed)
        np.random.seed(args.randomSeed)
        random.seed(args.randomSeed)

        logger.info("Testing in "+book)
        train_dataset = NERDataset(args.data_dir+'test.pkl',book,'train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.update_batch_size_test, shuffle=True)
        del train_dataset
        test_dataset = NERDataset(args.data_dir+'test.pkl',book,'test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.update_batch_size_test, shuffle=False)
        del test_dataset
        
        maml.load_state_dict(torch.load(args.output_path+'Model', map_location=torch.device('cpu')))
        model=maml.learner

        t_total = len(train_dataloader) * args.epochs
        test_optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
        test_scheduler = transformers.get_linear_schedule_with_warmup(
            test_optimizer, num_warmup_steps=book2warmup[book], num_training_steps=t_total
        )
        model.train()
        for epoch in range(args.epochs):
            for batch_idx, data in enumerate(train_dataloader):
                step = epoch * len(train_dataloader) + batch_idx + 1
                
                texts,masks,labels=data
                texts=texts.to(args.device)
                masks = masks.to(args.device)
                labels=labels.to(args.device)
                
                model.train()
                _,loss = model(input_ids=texts,attention_mask=masks,ner_labels=labels)
                
                if step % args.eval_step == 0 or step==t_total:
                    model.eval()
                    preds,masks,truths=[],[],[]
                    with torch.no_grad():
                        for data in test_dataloader:
                            test_texts,test_masks,test_labels=data
                            test_texts=test_texts.to(args.device)
                            test_masks = test_masks.to(args.device)
                            test_labels=test_labels.to(args.device)
                            
                            pred,_ = model(input_ids=test_texts,attention_mask=test_masks,ner_labels=test_labels)
                            preds+=pred.tolist()
                            masks+=test_masks.tolist()
                            truths+=test_labels.tolist()
                    preds=torch.tensor(preds)
                    truths=torch.tensor(truths)
                    preds=torch.argmax(preds,dim=2)
                    preds=[[id2bio[int(preds[i][j])] for j in range(1,preds.size(1)) if masks[i][j]==1] for i in range(preds.size(0))]
                    truths=[[id2bio[int(truths[i][j])] for j in range(1,truths.size(1)) if masks[i][j]==1] for i in range(truths.size(0))]
                    test_p=precision_score(truths,preds)
                    test_r=recall_score(truths,preds)
                    test_f1=f1_score(truths,preds)
                    
                    
                    if step==t_total:
                        logger.info('Test in {}: P is {}, R is {}, F1 is {}.'.format(book, test_p, test_r, test_f1))
                        P.append(test_p)
                        R.append(test_r)
                        F1.append(test_f1)
                    else:
                        logger.info('Test in {} Step {}: P is {}, R is {}, F1 is {}.'.format(book, step, test_p, test_r, test_f1))
                    model.train()
                
                loss.backward()
                test_optimizer.step()
                test_scheduler.step()
                test_optimizer.zero_grad()
    P=sum(P)/len(books)
    R=sum(R)/len(books)
    F1=sum(F1)/len(books)
    logger.info('Test P is {}, R is {}, F1 is {}.'.format(P, R, F1))