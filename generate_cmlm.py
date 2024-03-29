# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch
import numpy as np
import math
import torch.nn.functional as F
import re

from fairseq import options, progress_bar, tasks, tokenizer, utils, strategies, checkpoint_utils
from fairseq.meters import TimeMeter, StopwatchMeter
from fairseq.strategies.strategy_utils import duplicate_encoder_out


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'
    
    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)
    
    use_cuda = torch.cuda.is_available() and not args.cpu
    torch.manual_seed(args.seed)

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    #src_dict = task.source_dictionary
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary
    dict = tgt_dict

    # Load decoding strategy
    strategy = strategies.setup_strategy(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    # models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
        bert_ratio=args.bert_ratio if args.change_ratio else None,
        encoder_ratio=args.encoder_ratio if args.change_ratio else None,
        geargs=args,
    )

    if args.aver:
        print("\nAveraging the models...")
        model_aver = {}
        model_aver = models[0].state_dict()
        for key in models[0].state_dict():
            for i in range(len(models)-1):
                model_aver[key] = model_aver[key] + models[i+1].state_dict()[key]
            model_aver[key] = model_aver[key] / float(len(models))
        models[0].load_state_dict(model_aver)
        del models[1:]
        print("Models are averaged.")
    
    if use_cuda:
        models = [model.cuda() for model in models]

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        # if use_cuda:
        #     model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    
    results = []
    num_sentences = 0
    has_target = True
    timer = TimeMeter()

    with progress_bar.build_progress_bar(args, itr) as t:

        gen_timer = TimeMeter()
        if args.use_amp:
            with torch.cuda.amp.autocast():
                translations = generate_batched_itr(t, strategy, models, tgt_dict, length_beam_size=args.length_beam, use_gold_target_len=args.gold_target_len, cuda=use_cuda)
        else:
            translations = generate_batched_itr(t, strategy, models, tgt_dict, length_beam_size=args.length_beam, use_gold_target_len=args.gold_target_len, cuda=use_cuda)

        # print('Generation time = {}'.format(gen_timer.elapsed_time))

        for sample_id, src_tokens, target_tokens, hypos in translations:
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                src_str = dict.string(src_tokens, args.remove_bpe)
                if args.dehyphenate:
                    src_str = dehyphenate(src_str)
                if has_target:
                    target_str = dict.string(target_tokens, args.remove_bpe, escape_unk=True)
                    if args.dehyphenate:
                        target_str = dehyphenate(target_str)

            if not args.quiet:
                print('S-{}\t{}'.format(sample_id, src_str))
                    
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str))
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypos.int().cpu(),
                        src_str=src_str,
                        alignment= None,
                        align_dict=align_dict,
                        tgt_dict=dict,
                        remove_bpe=args.remove_bpe,
                    )
                    if args.dehyphenate:
                        hypo_str = dehyphenate(hypo_str)

                    if not args.quiet:
                        print('H-{}\t{}'.format(sample_id, hypo_str))
                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))
                        print()
                        
                        # Score only the top hypothesis
                        if has_target:
                            if align_dict is not None or args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)

                            results.append((target_str, hypo_str))

                    num_sentences += 1

        if has_target:
            print('Time = {}'.format(timer.elapsed_time))
            refs, outs = zip(*results)
            print('| Generate {} with beam={}: , '.format(args.gen_subset, args.beam))
            # Save files for scoring
            print('Saving files...')
            with open("ref.txt", "w") as f:
                for ref in refs:
                    f.write("{} \n".format(ref))
            with open("out.txt", "w") as f:
                for out in outs:
                    f.write("{} \n".format(out))
            print('Done saving files...')

def dehyphenate(sent):
    return re.sub(r'(\S)-(\S)', r'\1 ##AT##-##AT## \2', sent).replace('##AT##', '@')


def generate_batched_itr(data_itr, strategy, models, tgt_dict, length_beam_size=None, use_gold_target_len=False, cuda=True):
    """Iterate over a batched dataset and yield individual translations.
     Args:
        maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
    """
    for sample in data_itr:
        s = utils.move_to_cuda(sample) if cuda else sample
        if 'net_input' not in s:
            continue
        input = s['net_input']

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in input.items()
            if k != 'prev_output_tokens'
        }
        
        with torch.no_grad():
            gold_target_len = s['target'].ne(tgt_dict.pad()).sum(-1) if use_gold_target_len else None
            hypos = generate(strategy, encoder_input, models, tgt_dict, length_beam_size, gold_target_len)
            for batch in range(hypos.size(0)):
                src = utils.strip_pad(input['src_tokens'][batch].data, tgt_dict.pad())
                ref = utils.strip_pad(s['target'][batch].data, tgt_dict.pad()) if s['target'] is not None else None
                hypo = utils.strip_pad(hypos[batch], tgt_dict.pad())
                example_id = s['id'][batch].data
                yield example_id, src, ref, hypo


def generate(strategy, encoder_input, models, tgt_dict, length_beam_size, gold_target_len):
    assert len(models) == 1
    model = models[0]
    src_tokens = encoder_input['src_tokens']
    src_tokens = src_tokens.new(src_tokens.tolist())
    bsz = src_tokens.size(0)
    bert_encoder_padding_mask = src_tokens.eq(tgt_dict.pad())

    # print(encoder_input)
    bert_encoder_out, _, predicted_lengths = model.encoder(input_ids=src_tokens, output_hidden_states=True)
    encoder_out = {
            'encoder_out': bert_encoder_out,
            'encoder_padding_mask': 1-bert_encoder_padding_mask.float(),
            'predicted_lengths': predicted_lengths,
    }
    beam = predict_length_beam(gold_target_len, encoder_out['predicted_lengths'], length_beam_size)
    
    max_len = beam.max().item()
    length_mask = torch.triu(src_tokens.new(max_len, max_len).fill_(1).long(), 1)
    length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(bsz)], dim=0)
    tgt_tokens = src_tokens.new(bsz, length_beam_size, max_len).fill_(tgt_dict.mask())
    tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * tgt_dict.pad()
    tgt_tokens = tgt_tokens.view(bsz * length_beam_size, max_len)

    duplicate_encoder_out(encoder_out, bsz, length_beam_size)

    hypotheses, lprobs = strategy.generate(model, encoder_out, tgt_tokens, tgt_dict)
    
    hypotheses = hypotheses.view(bsz, length_beam_size, max_len)
    lprobs = lprobs.view(bsz, length_beam_size)
    tgt_lengths = (1 - length_mask).sum(-1)
    avg_log_prob = lprobs / tgt_lengths.float()
    best_lengths = avg_log_prob.max(-1)[1]
    hypotheses = torch.stack([hypotheses[b, l, :] for b, l in enumerate(best_lengths)], dim=0)
    
    return hypotheses


def predict_length_beam(gold_target_len, predicted_lengths, length_beam_size):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        beam = torch.stack([torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
    beam[beam < 2] = 2
    beam[beam > 40] = 40
    return beam


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
