# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:23:44 2024

@author: 28257
"""

#p=0.8
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
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--meta_batch_size', default=25, type=int, help='number of tasks sampled per meta-update')
    parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
    parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
    parser.add_argument('--num_updates', default=4, type=int, help='num_updates in maml')
    
    parser.add_argument('--update_batch_size', default=8, type=int,
                        help='number of examples used for inner gradient update (K for K-shot learning).')
    parser.add_argument('--query_size', default=24, type=int)
    parser.add_argument('--mixup', default=True, type=bool)
    parser.add_argument('--mixup_prob', default=0.6, type=float)
    parser.add_argument('--mixup_u', default=0.25, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    
    parser.add_argument('--update_batch_size_test', default=16, type=int,
                        help='number of examples used for inner gradient test (K for K-shot learning).')
    parser.add_argument('--num_updates_test', default=8, type=int, help='num_updates in maml')
    parser.add_argument('--query_size_test', default=64, type=int,
                        help='number of examples used for inner gradient test (K for K-shot learning).')

    
    parser.add_argument('--adversarial_temperature', default=5, type=int)
    
    parser.add_argument('--bs_train', default=8)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--eval_step', default=100)
    parser.add_argument('--bs_eval', default=32)
    parser.add_argument('--save_step', default=100)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-randomSeed', default=0, type=int)
    parser.add_argument('-save', '--save_path', default=None, type=str)

    args = parser.parse_args([])
    return args

def train(args, maml, optimiser, scheduler):
    logger.info("Loading Data.")
    dataloader = MetaPixNER(args.data_dir, 'train',
                            args.metatrain_iterations,args.meta_batch_size,args.update_batch_size,args.query_size,
                            args.mixup,args.mixup_prob,args.mixup_u)
    with open(args.data_dir+'dict.pkl','rb+') as fp:
        period_dict,bio2id,id2bio=pickle.load(fp)


    logger.info("start training")
    maml.train()
    device = args.device
    update_lr=args.update_lr
    warmup_step=args.metatrain_iterations//2
    with torch.backends.cudnn.flags(enabled=False):
        for step, data in enumerate(tqdm(dataloader)):
            if step>dataloader.__len__():
                break
            texts_spt, masks_spt, ner_labels_spt, periods_spt, texts_qry, masks_qry, ner_labels_qry, periods_qry=data
            texts_spt=texts_spt.to(device)
            masks_spt=masks_spt.to(device)
            ner_labels_spt=ner_labels_spt.to(device)
            periods_spt=periods_spt.to(device)
            texts_qry=texts_qry.to(device)
            masks_qry=masks_qry.to(device)
            ner_labels_qry=ner_labels_qry.to(device)
            periods_qry=periods_qry.to(device)
            
            task_losses = []
            #task_acc = []

            if step>warmup_step:
                update_lr/=10
                warmup_step+=args.metatrain_iterations//2

            for meta_batch in range(args.meta_batch_size):
                pred_qry,loss_qry,loss_ner,loss_period = maml(texts_spt[meta_batch], masks_spt[meta_batch], ner_labels_spt[meta_batch], periods_spt[meta_batch],
                                                              texts_qry[meta_batch], masks_qry[meta_batch], ner_labels_qry[meta_batch], periods_qry[meta_batch],
                                                              update_lr)
                task_losses.append(loss_qry)
                #task_acc.append(acc_val)

            meta_batch_loss = torch.stack(task_losses).mean()
            #meta_batch_acc = torch.stack(task_acc).mean()

            meta_batch_loss.backward()
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            if step % args.eval_step == 0:
                pred_val=torch.argmax(pred_qry,dim=2)
                pred_val=[[id2bio[int(pred_val[i][j])] for j in range(1,pred_val.size(1)) if masks_qry[meta_batch][i][j]==1] for i in range(pred_val.size(0))]
                truth_val=[[id2bio[int(ner_labels_qry[meta_batch][i][j])] for j in range(1,ner_labels_qry[meta_batch].size(1)) if masks_qry[meta_batch][i][j]==1] for i in range(ner_labels_qry[meta_batch].size(0))]
                f1_val=f1_score(truth_val,pred_val)
                
                test_loss_spt,test_f1_spt,loss_test,f1_test=evaluate(args, maml, update_lr, id2bio)
                
                logger.info('step {}: Val loss is {}, Val F1 is {}, Period Loss is {}, NER Loss is {}.'.format(
                    step, loss_qry.item(), f1_val,loss_period.item(),loss_ner.item()))
                logger.info('step {}: Test Support loss is {}, Test Support F1 is {}; Test Support loss is {}, Test Support F1 is {}.'.format(
                    step, test_loss_spt.item(), test_f1_spt,loss_test.item(),f1_test))

                
def evaluate(args, model,lr,id2bio):
    device = args.device
    dataloader = MetaPixNER(args.data_dir, 'test',
                            args.metatrain_iterations,args.meta_batch_size,args.update_batch_size_test,args.query_size_test)
    device = args.device
    task_loss_spt,task_f1_spt=[],[]
    task_loss,task_f1=[],[]
    for step, data in enumerate(dataloader):
        if step > dataloader.__len__():
            break
        texts_spt, masks_spt, ner_labels_spt, periods_spt, texts_qry, masks_qry, ner_labels_qry, periods_qry=data
        texts_spt=texts_spt.to(device)
        masks_spt=masks_spt.to(device)
        ner_labels_spt=ner_labels_spt.to(device)
        periods_spt=periods_spt.to(device)
        texts_qry=texts_qry.to(device)
        masks_qry=masks_qry.to(device)
        ner_labels_qry=ner_labels_qry.to(device)
        periods_qry=periods_qry.to(device)
        
        for meta_batch in range(len(dataloader.data)):
            f1_spt,loss_spt,pred,truth,f1,loss = maml.forward_test(texts_spt[meta_batch], masks_spt[meta_batch], ner_labels_spt[meta_batch], periods_spt[meta_batch],
                                                                   texts_qry[meta_batch], masks_qry[meta_batch], ner_labels_qry[meta_batch], periods_qry[meta_batch],
                                                                   lr, id2bio)
            task_loss_spt.append(loss_spt)
            task_f1_spt.append(f1_spt)
            task_loss.append(loss)
            task_f1.append(f1)

    loss_spt = torch.stack(task_loss_spt).mean()
    loss_test = torch.stack(task_loss).mean()
    f1_spt = sum(task_f1_spt)/len(task_f1_spt)
    f1_test = sum(task_f1)/len(task_f1)
    
    return loss_spt,f1_spt,loss_test,f1_test

if __name__ == '__main__':
    args = set_args()
    args.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    
    maml = MAML(args).to(args.device)
    
    optimizer = transformers.AdamW(maml.parameters(), 
                                   lr=args.meta_lr, weight_decay=args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.metatrain_iterations
    )
    
    train(args, maml, optimizer, scheduler)
    torch.save(maml.state_dict(),args.output_path+'Model')