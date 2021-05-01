# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F
import torch
import copy 

from . import register_strategy
from .easy_first import EasyFirst
from .strategy_utils import duplicate_encoder_out, generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('hybrid')
class Hybrid(EasyFirst):
    
    def __init__(self, args):
        super().__init__(args)
        self.iterations = args.decoding_iterations
    
    def generate(self, model, encoder_out, tokens, tgt_dict):
        bsz, seq_len = tokens.size()
        pad_mask = tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = seq_len if self.iterations is None else self.iterations
        encoder_out_2 = copy.deepcopy(encoder_out)

        duplicate_encoder_out(encoder_out, bsz, self.beam_size)
        tokens = tokens.unsqueeze(1).repeat(1, self.beam_size, 1)
        lprobs = tokens.new(bsz, self.beam_size).float().fill_(float('-inf'))
        lprobs[:, 0] = 0

        
        # for batch in range(bsz):
        #     for beam in range(self.beam_size):
        #         print("Initialization: ", convert_tokens(tgt_dict, tokens[batch, beam]))
        # print()
        

        for position in range(seq_len):
            tokens = tokens.view(bsz * self.beam_size, seq_len) # merge beam with batch
            decoder_out = model.decoder(input_ids=tokens, encoder_hidden_states=encoder_out['encoder_out'], output_hidden_states=True)
            candidate_lprobs = self.generate_candidates(decoder_out, tokens, tgt_dict.mask(), position)
            tokens = tokens.view(bsz, self.beam_size, seq_len) # separate beam from batch
            candidate_lprobs = candidate_lprobs.view(bsz, self.beam_size, seq_len, -1) # separate beam from batch
            tokens, _ = self.select_best(tokens, lprobs, candidate_lprobs)

            
            # for batch in range(bsz):
            #     for beam in range(self.beam_size):
            #         print("Prediction: ", convert_tokens(tgt_dict, tokens[batch, beam]))
            # print()
            

        # return tokens[:, 0, :], lprobs[:, 0]

        tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_out_2, tokens[:, 0, :])
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        print("\nInitialization: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

            # print("Step: ", counter+1)
            # print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))
            decoder_out = model.decoder(input_ids=tgt_tokens, encoder_hidden_states=encoder_out_2['encoder_out'], output_hidden_states=True)
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            # print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs
    
    def generate_candidates(self, decoder_out, tokens, mask, position):
        candidate_probs = F.softmax(decoder_out[0], dim=-1)
        candidate_probs = candidate_probs * tokens.eq(mask).float().unsqueeze(-1)
        candidate_probs[:, :, mask] = 0
        candidate_probs[:, position + 1:, :] = 0
        return candidate_probs.log()

    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder(input_ids=tgt_tokens, encoder_hidden_states=encoder_out['encoder_out'], output_hidden_states=True)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)