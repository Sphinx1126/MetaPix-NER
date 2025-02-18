# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:04:46 2024

@author: 28257
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from lstm import functional_lstm
from crf import functional_CRF
from AdaptedBERT import AdaptedBERT
from collections import OrderedDict

class NERLearner(nn.Module):
    def __init__(self, 
                 pretrained_bert, bottleneck_dim, num_period,
                 lstm_hidden_size, lstm_num_layers, lstm_batch_first, lstm_bidirectional,
                 crf_num_labels, device,):
        super(NERLearner, self).__init__()
        self.device=device
        self.num_labels=crf_num_labels
        self.bottleneck_size=bottleneck_dim
        self.num_period=num_period
        self.lstm_hidden_size=lstm_hidden_size
        self.lstm_num_layers=lstm_num_layers
        self.lstm_batch_first=lstm_batch_first
        self.lstm_bidirectional=lstm_bidirectional
        
        self.bert=AdaptedBERT(pretrained_bert, self.bottleneck_size, self.device)
        self.config=self.bert.config
        self.bert_hidden_size=self.bert.hidden_size
        self.period_classifier=nn.Linear(self.bert_hidden_size, self.num_period)
        self.softmax=nn.Softmax(dim=0)
        self.lstm=functional_lstm(self.bert_hidden_size, self.lstm_hidden_size//2, 
                                  num_layers=self.lstm_num_layers,
                                  batch_first=self.lstm_batch_first,bidirectional=self.lstm_bidirectional)
        self.linear=nn.Linear(self.lstm_hidden_size, self.num_labels)
        self.crf=functional_CRF(device=self.device,num_labels=self.num_labels)
    
    def forward(self,
                input_ids,attention_mask,
                ner_labels=None,period_label=None):
        embs=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        if period_label is not None:
            period_emb=torch.mean(embs[1],dim=0)
            period_pred=self.softmax(self.period_classifier(period_emb))
            period_loss=torch.sum(-period_label.float()*torch.log(period_pred+1e-15))
        embs=self.lstm(embs[0]) if self.lstm_batch_first else self.lstm(embs[0].permute((1, 0, 2)))[0].permute((1, 0, 2))
        emission=self.linear(embs)
        if ner_labels is not None:
            ner_loss=self.crf(emission,ner_labels,attention_mask)
            ner_loss=-torch.mean(ner_loss)
        opt=(emission,)
        if ner_labels is not None:
            opt+=(ner_loss,)
        if period_label is not None:
            opt+=(period_loss,)
        return opt
    
    def functional_forward(self,
                           fast_weights,
                           input_ids,attention_mask,
                           ner_labels=None,period_label=None):

        embs=self.bert.functional_forward(fast_weights['bert.adapter.down_project.weight'], fast_weights['bert.adapter.down_project.bias'], 
                                          fast_weights['bert.adapter.up_project.weight'], fast_weights['bert.adapter.up_project.bias'],
                                          input_ids=input_ids,attention_mask=attention_mask)
        if period_label is not None:
            period_emb=torch.mean(embs[1],dim=0)
            period_pred=F.softmax(F.linear(period_emb,
                                           fast_weights['period_classifier.weight'],fast_weights['period_classifier.bias']),dim=0)
            period_loss=torch.sum(-period_label.float()*torch.log(period_pred+1e-15))
        embs=embs[0] if self.lstm_batch_first else embs[0].permute((1, 0, 2))
        bilstm_weights=[fast_weights[name] for name in fast_weights if 'lstm' in name]
        embs=self.lstm.functional_forward(embs, bilstm_weights)[0]
        embs=embs if self.lstm_batch_first else embs.permute((1, 0, 2))
        emission=F.linear(embs,fast_weights['linear.weight'],fast_weights['linear.bias'])
        if ner_labels is not None:
            ner_loss=self.crf.functional_forward(emission,ner_labels,attention_mask,
                                                 fast_weights['crf.trans_matrix'], fast_weights['crf.start_matrix'], fast_weights['crf.end_matrix'])
            ner_loss=-torch.mean(ner_loss)
        opt=(emission,)
        if ner_labels is not None:
            opt+=(ner_loss,)
        if period_label is not None:
            opt+=(period_loss,)
        return opt
    
    

if __name__=='__main__':
    pretrained_bert=AutoModel.from_pretrained("../bert/")
    device = torch.device("cuda:0")
    model=NERLearner(pretrained_bert,8,5,
                     200,1,False,True,
                     6,device).cuda()
    

    text1=[114,114,514,0,0]
    text2=[19,810,0,0,0]
    mask1=[1,1,1,0,0]
    mask2=[1,1,0,0,0]
    texts=torch.tensor([text1,text2]).cuda()
    masks=torch.tensor([mask1,mask2]).cuda()
    period=torch.tensor([0,1,0,0,0]).cuda()
    label=torch.tensor([[1,0,0,1,2],[0,1,2,2,0]]).cuda()
    opt=model(input_ids=texts,attention_mask=masks,ner_labels=label,period_label=period)
    
    
    fast_weights=OrderedDict([(name,param) for name,param in model.named_parameters() if param.requires_grad])
    for i in range(10):
        opt_functional,ner_loss,period_loss=model.functional_forward(fast_weights=fast_weights,
                                                                     input_ids=texts,attention_mask=masks,
                                                                     ner_labels=label,period_label=period)
        gradients=torch.autograd.grad(ner_loss, fast_weights.values(), create_graph=True,allow_unused=True)
        fast_weights = OrderedDict(
            (name, param - 0.01 * grad) if grad is not None else (name, param)
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )
    '''
    opt=NERLearner(input_ids=texts,attention_mask=masks)
    weights_down=perta.adapter.down_project.weight
    bias_down=perta.adapter.down_project.bias
    weights_up=perta.adapter.up_project.weight
    bias_up=perta.adapter.up_project.bias
    opt_functional=perta.functional_forward(weights_down, bias_down, weights_up, bias_up,input_ids=texts,attention_mask=masks)
    gradients = torch.autograd.grad(torch.mean(opt_functional[0]), weights_down, create_graph=True)
    
    from collections import OrderedDict
    fast_weights = OrderedDict(perta.adapter.named_parameters())
    opt_functional=perta.functional_forward(fast_weights['down_project.weight'], fast_weights['down_project.bias'], 
                                            fast_weights['up_project.weight'], fast_weights['up_project.bias'],
                                            input_ids=texts,attention_mask=masks)
    gradients2=torch.autograd.grad(torch.mean(opt_functional[0]), fast_weights.values(), create_graph=True)
    '''