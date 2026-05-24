# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 05:49:11 2023

@author: 28257
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self,input_dim,bottleneck_dim,opt_dim=None):
        super(Adapter, self).__init__()
        self.input_dim=input_dim
        self.bottleneck_dim=bottleneck_dim
        self.opt_dim=self.input_dim if opt_dim is None else opt_dim
        self.down_project=nn.Linear(self.input_dim, self.bottleneck_dim)
        self.nonlinearity=nn.GELU()
        self.up_project=nn.Linear(self.bottleneck_dim, self.opt_dim)

        torch.nn.init.normal_(self.down_project.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.up_project.weight, mean=0.0, std=0.01)

    def forward(self,x):
        opt=self.down_project(x)
        opt=self.nonlinearity(opt)
        opt=self.up_project(opt)
        opt+=x
        return opt
    
    def functional_forward(self,x,
                           weights_down,bias_down,
                           weights_up,bias_up):
        opt=F.linear(x,weights_down,bias_down)
        opt=F.gelu(opt)
        opt=F.linear(opt,weights_up,bias_up)
        opt+=x
        return opt


class AdaptedBERT(nn.Module):
    def __init__(self, pretrained_bert,
                 #prompt_len,
                 bottleneck_dim,device):
        super(AdaptedBERT, self).__init__()
        for param in pretrained_bert.parameters():
            param.requires_grad=False
        self.device=device
        self.config = pretrained_bert.config
        self.embeddings = pretrained_bert.embeddings
        self.hidden_size=pretrained_bert.config.hidden_size
        #self.prompt_len=prompt_len
        self.bottleneck_dim=bottleneck_dim
        self.encoder = pretrained_bert.encoder
        self.adapter=Adapter(self.hidden_size,self.bottleneck_dim)
        self.pooler = pretrained_bert.pooler
        self.get_extended_attention_mask=pretrained_bert.get_extended_attention_mask
        self.invert_attention_mask=pretrained_bert.invert_attention_mask
        self.get_head_mask=pretrained_bert.get_head_mask
        
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def encoding(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return encoder_outputs
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        encoder_outputs=self.encoding(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      head_mask=head_mask,
                                      inputs_embeds=inputs_embeds,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_attention_mask,
                                      past_key_values=past_key_values,
                                      use_cache=use_cache,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output=self.adapter(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
    def functional_forward(
        self,
        weights_down,bias_down,
        weights_up,bias_up,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        encoder_outputs=self.encoding(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      head_mask=head_mask,
                                      inputs_embeds=inputs_embeds,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_attention_mask,
                                      past_key_values=past_key_values,
                                      use_cache=use_cache,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output=self.adapter.functional_forward(sequence_output, weights_down, bias_down, weights_up, bias_up)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )




if __name__=='__main__':
    pretrained_bert=AutoModel.from_pretrained("../bert/")
    perta=AdaptedBERT(pretrained_bert,8)
    
    len1=128
    len2=255
    text1=[514]*len1+[103]*(512-len1)
    text2=[114]*len2+[103]*(512-len2)
    mask1=[1]*len1+[0]*(512-len1)
    mask2=[1]*len2+[0]*(512-len2)
    texts=torch.tensor([text1,text2])
    masks=torch.tensor([mask1,mask2])
    
    opt=perta(input_ids=texts,attention_mask=masks)
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