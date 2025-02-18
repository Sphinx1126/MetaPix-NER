# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:22:51 2024

@author: 28257
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import AutoTokenizer,AutoModel
from models.learner import NERLearner
import numpy as np
from seqeval.metrics import f1_score

class MAML(nn.Module):
    def __init__(self, args):
        super(MAML, self).__init__()
        self.args = args
        pretrained_bert=AutoModel.from_pretrained(args.bert_path)
        self.learner = NERLearner(pretrained_bert,args.bottleneck_dim,args.num_period,
                                  args.lstm_hidden_size,args.lstm_num_layers,args.lstm_batch_first,args.lstm_bidirectional,
                                  args.crf_num_labels,args.device)
        self.num_updates = self.args.num_updates
        self.num_updates_test = self.args.num_updates_test
        self.loss_lam=args.loss_lam
    
    def forward(self, 
                texts_spt, masks_spt, ner_label_spt, period_spt,
                texts_qry, masks_qry, ner_label_qry, period_qry):
        create_graph = True

        fast_weights = OrderedDict([(name,param) for name,param in self.learner.named_parameters() if param.requires_grad])

        for inner_batch in range(self.num_updates):
            pred_spt,loss_spt,_ = self.learner.functional_forward(fast_weights=fast_weights,
                                                                  input_ids=texts_spt,attention_mask=masks_spt,
                                                                  ner_labels=ner_label_spt,period_label=period_spt)
            gradients=torch.autograd.grad(loss_spt, 
                                          fast_weights.values(), 
                                          create_graph=create_graph,
                                          allow_unused=True)
            warmup_step=int(self.num_updates*0.3)
            step=inner_batch+1
            lr=self.args.update_lr*step/warmup_step if step<=warmup_step else self.args.update_lr*2**(warmup_step-step)
            fast_weights = OrderedDict(
                (name, param - lr * grad) if grad is not None else (name, param)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        
        _,_,period_loss_spt = self.learner.functional_forward(fast_weights=fast_weights,
                                                              input_ids=texts_spt,attention_mask=masks_spt,
                                                              ner_labels=ner_label_spt,period_label=period_spt)
        pred_qry,ner_loss_qry,_ = self.learner.functional_forward(fast_weights=fast_weights,
                                                                  input_ids=texts_qry,attention_mask=masks_qry,
                                                                  ner_labels=ner_label_qry,period_label=period_qry)
        loss_qry=self.loss_lam*ner_loss_qry+(1-self.loss_lam)*period_loss_spt

        return pred_qry,loss_qry
    
    def forward_test(self, 
                     texts_spt, masks_spt, ner_label_spt, period_spt,
                     texts_qry, masks_qry, ner_label_qry, period_qry,
                     id2bio):
        create_graph = True
        self.train()

        fast_weights = OrderedDict([(name,param) for name,param in self.learner.named_parameters() if param.requires_grad])

        for inner_batch in range(self.num_updates_test):
            pred_spt,loss_spt,_ = self.learner.functional_forward(fast_weights=fast_weights,
                                                                  input_ids=texts_spt,attention_mask=masks_spt,
                                                                  ner_labels=ner_label_spt,period_label=period_spt)
            gradients=torch.autograd.grad(loss_spt, 
                                          fast_weights.values(), 
                                          create_graph=create_graph,
                                          allow_unused=True)
            warmup_step=int(self.num_updates_test*0.3)
            step=inner_batch+1
            lr=self.args.update_lr*step/warmup_step if step<=warmup_step else self.args.update_lr*2**(warmup_step-step)
            fast_weights = OrderedDict(
                (name, param - lr * grad) if grad is not None else (name, param)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        
        pred_spt=torch.argmax(pred_spt,dim=2)
        pred_spt=[[id2bio[int(pred_spt[i][j])] for j in range(1,pred_spt.size(1)) if masks_spt[i][j]==1] for i in range(pred_spt.size(0))]
        truth_spt=[[id2bio[int(ner_label_spt[i][j])] for j in range(1,ner_label_spt.size(1)) if masks_spt[i][j]==1] for i in range(ner_label_spt.size(0))]
        f1_spt=f1_score(truth_spt,pred_spt)
        
        self.eval()
        with torch.no_grad():
            pred_qry,ner_loss_qry,_ = self.learner.functional_forward(fast_weights=fast_weights,
                                                                      input_ids=texts_qry,attention_mask=masks_qry,
                                                                      ner_labels=ner_label_qry,period_label=period_qry)
            pred_test=torch.argmax(pred_qry,dim=2)
            pred_test=[[id2bio[int(pred_test[i][j])] for j in range(1,pred_test.size(1)) if masks_qry[i][j]==1] for i in range(pred_test.size(0))]
            truth_test=[[id2bio[int(ner_label_qry[i][j])] for j in range(1,ner_label_qry.size(1)) if masks_qry[i][j]==1] for i in range(ner_label_qry.size(0))]
            f1_test=f1_score(truth_test,pred_test)
            
        self.train()

        return f1_spt,loss_spt,pred_test,truth_test,f1_test,ner_loss_qry