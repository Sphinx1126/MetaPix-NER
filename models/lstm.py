# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:01:42 2024

@author: 28257
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch._VF as _VF

class functional_lstm(nn.LSTM):
    def functional_forward(self, 
                           input,
                           weights,
                           hx=None):  # noqa: F811
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        do_permute = False
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            if hx is None:
                h_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, real_hidden_size,
                                      dtype=input.dtype, device=input.device)
                c_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, self.hidden_size,
                                      dtype=input.dtype, device=input.device)
                hx = (h_zeros, c_zeros)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            if input.dim() not in (2, 3):
                raise ValueError(f"LSTM: Expected input to be 2D or 3D, got {input.dim()}D instead")
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                h_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, real_hidden_size,
                                      dtype=input.dtype, device=input.device)
                c_zeros = torch.zeros(self.num_layers * num_directions,
                                      max_batch_size, self.hidden_size,
                                      dtype=input.dtype, device=input.device)
                hx = (h_zeros, c_zeros)
                self.check_forward_args(input, hx, batch_sizes)
            else:
                if is_batched:
                    if (hx[0].dim() != 3 or hx[1].dim() != 3):
                        msg = ("For batched 3-D input, hx and cx should "
                               f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = ("For unbatched 2-D input, hx and cx should "
                               f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                self.check_forward_args(input, hx, batch_sizes)
                hx = self.permute_hidden(hx, sorted_indices)

        if batch_sizes is None:
            result = _VF.lstm(input, hx, weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)

if __name__=='__main__':
    bilstm=functional_lstm(10, 20,bidirectional=True)
    #weights=bilstm._flat_weights
    from collections import OrderedDict
    fast_weights=OrderedDict(bilstm.named_parameters())
    weights=list(fast_weights.values())
    
    input = torch.randn(3, 5, 10)
    opt,_ = bilstm(input.permute((1, 0, 2)))
    opt_functional,_=bilstm.functional_forward(input,weights)

    