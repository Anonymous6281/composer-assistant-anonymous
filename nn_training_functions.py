import copy
import json
import math
import pickle
import random
import time
import bisect

import constants as cs
import midisong as ms
import encoding_functions as enc
import preprocessing_functions as pre
import os
import torch
import collections

from tokenizer_functions import spm_type_to_note_off_treatment


# test written
def aug_bpm(S: "ms.MidiSongByMeasure"):
    """in place operation"""
    amt = random.randint(-1, 1) * 5 * random.random() * .01
    for t in S.tempo_changes:
        t.val += amt * t.val


# test written
def aug_vel(S: "ms.MidiSongByMeasure"):
    """in place operation"""
    def fix_vel(v):
        v = int(v)
        if v < 1:
            v = 1
        if v > 127:
            v = 127
        return v

    amt = random.randint(-1, 1) * 5 * random.random() * .01
    for track in S.tracks:
        for tbm in track.tracks_by_measure:
            for n in tbm.note_ons:
                n.vel += amt * n.vel
                n.vel = fix_vel(n.vel)


# test written
def uncorrupt(input_ids: "list[int]", labels: "list[int]", tokenizer) -> "list[int]":
    """input_ids and labels lists of integers. Returns a list of integers."""
    corruption_markers = []
    extra_ids = [';<extra_id_{}>'.format(x) for x in range(256)]
    for marker in tokenizer.Encode(extra_ids):
        corruption_markers.append(marker[0])
    corruption_markers_set = set(corruption_markers)

    res = []
    len_labels = len(labels)
    for i, e in enumerate(input_ids):
        if e in corruption_markers_set:
            done = False
            try:
                i_labels = labels.index(e)
            except ValueError:
                done = True
            while not done:
                i_labels += 1
                if i_labels >= len_labels:
                    done = True

                if not done:
                    e_labels = labels[i_labels]
                    if e_labels in corruption_markers_set:
                        done = True

                if not done:
                    res.append(e_labels)
        else:
            res.append(e)
    return res


# test written
def corrupt_pretrain(list_of_ints, tokenizer):
    """for corrupted span objective"""
    n_corruptions = math.floor(0.15*len(list_of_ints)/3)  # 0 or more corruptions
    n_corruptions = min(256, n_corruptions)  # at most 256 corruptions

    labels = []
    corrupted = []

    # first, get corruption indices
    sample_population = range(len(list_of_ints) - 2 * n_corruptions - 1 - n_corruptions)
    indices = sorted(random.sample(sample_population, n_corruptions))
    corruption_start_indices = []
    for i, x in enumerate(indices):
        corruption_start_indices.append(x + 1 + 3 * i)

    i = 0
    n_spans_done = 0
    done = False
    # iterate over L from left to right; randomize first corruption marker
    first_extra_id = random.randint(0, 256-n_corruptions)  # assumes tokenizer has extra_id's 0 thru 255
    while not done:
        if i in corruption_start_indices:
            corrupt_marker = tokenizer.encode(';<extra_id_{}>'.format(first_extra_id + n_spans_done))[0]
            labels.append(corrupt_marker)
            corrupted.append(corrupt_marker)
            labels.extend(list_of_ints[i:i + 3])
            n_spans_done += 1
            i += 3
        else:
            corrupted.append(list_of_ints[i])
            i += 1

        done = i >= len(list_of_ints)

    return corrupted, labels


# test written
def get_random_mask(S: "ms.MidiSongByMeasure",
                    measure_slice: "tuple[int, int]" = None,
                    pattern_type: int = 0,
                    extra_params: "dict" = None
                    ) -> "set[tuple[int, int]]":
    """returns a set of tuples of the form (track_index, measure_index)"""
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    if extra_params is None:
        extra_params = {}

    res = set()

    if pattern_type == 0:  # random measures and instruments.
        p = extra_params['mask_probability']

        for track_index in range(len(S.tracks)):
            for measure_index in range(*measure_slice):
                if random.random() < p:
                    res.add((track_index, measure_index))

    elif pattern_type == 1:  # random tracks, all measures
        if len(S.tracks) < 2:
            return res
        n_tracks_to_mask = random.randint(1, len(S.tracks) - 1)
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for measure_index in range(*measure_slice):
            for track_index in tracks_to_mask:
                res.add((track_index, measure_index))

    elif pattern_type == 2:  # random measure, all tracks
        if 'masked_measure' in extra_params:
            measure_to_mask = extra_params['masked_measure']
        else:
            measure_to_mask = random.randint(measure_slice[0], measure_slice[1] - 1)

        for track_index in range(len(S.tracks)):
            res.add((track_index, measure_to_mask))

    elif pattern_type == 3:  # random measure, most tracks
        if len(S.tracks) // 2 > len(S.tracks) - 1:
            return res
        measure_to_mask = random.randint(measure_slice[0], measure_slice[1] - 1)
        n_tracks_to_mask = random.randint(len(S.tracks) // 2, len(S.tracks) - 1)
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for track_index in tracks_to_mask:
            res.add((track_index, measure_to_mask))

    elif pattern_type == 4:  # 2, 3, or 4 consecutive measures chosen at random, all tracks
        m_slice_size = measure_slice[1] - measure_slice[0]
        if m_slice_size < 3:
            return res
        if 'masked_measures' in extra_params:
            masked_measures = extra_params['masked_measures']
        else:
            n_masked_measures = random.randint(2, min(m_slice_size, 4))
            first_masked_measure = random.randint(0, m_slice_size - n_masked_measures)
            masked_measures = [first_masked_measure + x for x in range(n_masked_measures)]

        for measure_index in masked_measures:
            for track_index in range(len(S.tracks)):
                res.add((track_index, measure_index))

    elif pattern_type == 5:  # 2, 3, or 4 consecutive measures chosen at random, most tracks
        m_slice_size = measure_slice[1] - measure_slice[0]

        if m_slice_size < 3:
            return res

        if len(S.tracks) // 2 > len(S.tracks)-1:
            return res

        if 'masked_measures' in extra_params:
            masked_measures = extra_params['masked_measures']
        else:
            n_masked_measures = random.randint(2, min(m_slice_size, 4))
            first_masked_measure = random.randint(0, m_slice_size - n_masked_measures)
            masked_measures = [first_masked_measure + x for x in range(n_masked_measures)]

        n_tracks_to_mask = random.randint(len(S.tracks) // 2, len(S.tracks) - 1)
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for measure_index in masked_measures:
            for track_index in tracks_to_mask:
                res.add((track_index, measure_index))

    elif pattern_type == 6:  # random tracks; each track gets 1 or 2 random spans.
        if len(S.tracks) == 0:
            return res
        n_tracks_to_mask = random.randint(1, len(S.tracks))
        tracks_to_mask = random.sample(range(len(S.tracks)), n_tracks_to_mask)

        for track_index in tracks_to_mask:
            n_spans = random.randint(1, 2)
            if n_spans == 1:
                st_measure = random.randint(measure_slice[0], measure_slice[1] - 1)
                end_measure = random.randint(st_measure + 1, measure_slice[1])  # set up for range()
                # ignore the following; full masking is fine.
                # # make sure this is not a full masking of the instrument...
                # if st_measure == measure_slice[0] and end_measure == measure_slice[1]:
                #     # ...unless there's only one measure
                #     if end_measure - st_measure > 1:
                #         # then randomly increment the st measure or decrement the end measure
                #         if random.randint(0, 1):
                #             st_measure += 1
                #         else:
                #             end_measure -= 1

                for measure_index in range(st_measure, end_measure):
                    res.add((track_index, measure_index))

            elif n_spans == 2:
                if measure_slice[1] - measure_slice[0] < 3:
                    return res
                span_1_st = measure_slice[0]
                span_2_end = measure_slice[1]
                if random.randint(0, 1):
                    # then choose span 1 end first
                    span_1_end = random.randint(span_1_st + 1, span_2_end - 2)
                    span_2_st = random.randint(span_1_end + 1, span_2_end - 1)
                else:
                    # then choose span 2 st first
                    span_2_st = random.randint(span_1_st + 2, span_2_end - 1)
                    span_1_end = random.randint(span_1_st + 1, span_2_st - 1)

                for measure_index in range(span_1_st, span_1_end):
                    res.add((track_index, measure_index))
                for measure_index in range(span_2_st, span_2_end):
                    res.add((track_index, measure_index))

    return res


# test written
def get_trans_amt(epoch, i):
    trans_range = cs.AUG_TRANS_MAX - cs.AUG_TRANS_MIN + 1
    trans_amt = (epoch + i) % trans_range
    trans_amt += cs.AUG_TRANS_MIN
    return trans_amt


# test written
def count_tracks_in_str(s):
    inst_rep_tuples = set()
    s_split = s.split(';')
    for i, instruction in enumerate(s_split):
        if ':' in instruction:
            a, b = instruction.split(':')
            if a == 'I':
                inst = int(b)
                if i + 1 < len(s_split) and ':' in s_split[i + 1] and s_split[i + 1].split(':')[0] == 'R':
                    rep = int(s_split[i + 1].split(':')[1])
                else:
                    rep = 0
                inst_rep_tuples.add((inst, rep))
    return len(inst_rep_tuples)


# helper function for build_val_test_finetune_data
def _build_val_test_finetune_data(T):
    i, k, v, tokenizer, epoch, incl_poly_mono, mask_pattern_type, n_measures = T

    # random.seed(epoch)

    S = pre.midisongbymeasure_from_save_dict(v)

    if n_measures == 'all':
        n_measures = S.get_n_measures()

    if n_measures > S.get_n_measures():
        print('Notice: Could not generate example (not enough measures): Wanted {} measures; {}'.format(n_measures, k))
        return []

    transpose_amt = get_trans_amt(epoch=epoch, i=i)
    S.transpose(amt=transpose_amt)
    enc.transpose_into_acceptable_ranges_TT(S)
    S.sort_tracks_by_inst_and_avg_note_pitch()
    for t in S.tracks:
        t.sort()

    extra_id_max = 999999999

    n_tries = 25
    for try_ in range(n_tries):
        measure_st = random.randint(0, S.get_n_measures() - n_measures)
        measure_end = measure_st + n_measures
        measure_slice = (measure_st, measure_end)

        if mask_pattern_type == '2random':
            mask_pattern = 2
        elif mask_pattern_type == '2last':
            mask_pattern = 2
        elif mask_pattern_type == '0half':
            mask_pattern = 0
        elif mask_pattern_type == '0quarter':
            mask_pattern = 0
        else:
            mask_pattern = mask_pattern_type

        extra_params = None
        if mask_pattern_type == '2last':
            extra_params = {'masked_measure': measure_end - 1}
        elif mask_pattern_type == 4:
            extra_params = {'masked_measures': [measure_end - 2, measure_end - 1]}
        elif mask_pattern_type == "0half":
            extra_params = {'mask_probability': 0.5}
        elif mask_pattern_type == '0quarter':
            extra_params = {'mask_probability': 0.25}

        mask = get_random_mask(S, measure_slice=measure_slice, pattern_type=mask_pattern, extra_params=extra_params)
        if incl_poly_mono:
            poly_mono_commands = get_poly_mono_commands(S, mask=mask)
        else:
            poly_mono_commands = None

        # make test and val data even if there are more masks than our tokenizer can handle
        if len(mask) < 256 or True:
            # extra_id_st = random.randint(0, extra_id_max + 1 - len(mask))
            # always start val and test extra id's at 0
            extra_id_st = 0
            iids, labels = enc.encode_midisongbymeasure_with_masks(S,
                                                                   note_off_treatment=spm_type_to_note_off_treatment(cs.SPM_TYPE),
                                                                   mask_locations=mask,
                                                                   measure_slice=measure_slice,
                                                                   include_heads_for_empty_masked_measures=False,
                                                                   poly_mono_commands=poly_mono_commands,
                                                                   return_labels_too=True,
                                                                   extra_id_st=extra_id_st,
                                                                   extra_id_max=extra_id_max
                                                                   )
            iids_track_count = count_tracks_in_str(iids)
            if iids and labels and iids_track_count > 1:  # make sure we have at least 2 tracks in this example
                return [(k, {'input_ids': iids,
                             'labels': labels,
                             'processed_source': k,
                             'measure_slice': measure_slice,
                             'mask': list(mask),  # sets are not json serializable
                             'note_off_treatment': cs.SPM_TYPE,
                             'extra_id_st': extra_id_st,
                             'transpose': transpose_amt
                             })]

    to_print = 'Notice: Could not generate example after {} tries {}'.format(n_tries, k)
    to_print += ' params: mask_pattern_type={}, n_measures={}'.format(mask_pattern_type, n_measures)
    print(to_print)
    return []


def build_val_test_finetune_data(tokenizer, epoch, pool, mode,
                                 incl_poly_mono=True,
                                 mask_pattern_type=0,
                                 n_measures=5):
    """
    mode = 'val' or 'test'
    incl_poly_mono = True or False
    mask_pattern_type = 0 or 1 or "2random" or "2last" or 4 or 6
    n_measures = 5 or 9 or 17 or 33 or whatever or 'all'
    """
    t0 = time.time()

    P = pool
    to_write = []

    if mode == 'val':
        path = cs.PATH_TO_PROCESSED_VAL_MIDI
    elif mode == 'test':
        path = cs.PATH_TO_PROCESSED_TEST_MIDI
    else:
        raise ValueError('mode not recognized: {}'.format(mode))

    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            with open(os.path.join(folder, fname)) as infile:
                d = json.load(infile)
            d_keys = sorted(list(d.keys()))
            items = [(i, k, d[k], tokenizer, epoch, incl_poly_mono, mask_pattern_type, n_measures) for i, k in enumerate(d_keys)]
            print('file {} loaded'.format(fname))

            for i, res in enumerate(P.imap_unordered(_build_val_test_finetune_data, items, chunksize=10)):
                to_write.extend(res)

                if (i + 1) % 1000 == 0:
                    print(i + 1, 'songs processed from this file so far')

    # to compensate for random return order from imap_unordered
    to_write.sort(key=lambda x: x[0])
    to_write = [x[1] for x in to_write]
    dict_to_write = {}
    for i, thing in enumerate(to_write):
        dict_to_write[i] = thing

    if mode == 'val':
        with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'finetune_validation_{}_{}.txt'.format(mask_pattern_type, n_measures)), 'w') as outfile:
            json.dump(dict_to_write, outfile)
    elif mode == 'test':
        with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'finetune_test_{}_{}.txt'.format(mask_pattern_type, n_measures)), 'w') as outfile:
            json.dump(dict_to_write, outfile)
    to_print = 'finished building {} data ('.format(mode)
    to_print += 'incl_poly_mono={}'.format(incl_poly_mono)
    to_print += ', mask_pattern_type={}'.format(mask_pattern_type)
    to_print += ', n_measures={})'.format(n_measures)
    to_print += ' in {} sec'.format(time.time() - t0)
    print(to_print)


# test written
def weighted_choose_one(dict_of_weights: dict):
    """dict of weights a dict with keys the objects you are trying to choose one from, and associated keys
    positive numbers"""
    total = sum(v for v in dict_of_weights.values())
    items = sorted(list(dict_of_weights.keys()))
    index_weights = [0]
    for k in items:
        if dict_of_weights[k] < 0:
            raise ValueError('all weights must be nonnegative')
        index_weights.append(index_weights[-1] + dict_of_weights[k])
    index = random.random() * total
    index = bisect.bisect_right(index_weights, index) - 1
    return items[index]


# test written
def get_poly_mono_commands(S: "ms.MidiSongByMeasure", mask: "set[tuple[int, int]]"):
    poly_mono_commands = collections.defaultdict(str)
    for mask_tuple in mask:
        tr_i, m_i = mask_tuple
        if not S.tracks[tr_i].is_drum:  # drum tracks do not get poly/mono commands
            if S.is_poly(track_idx=tr_i, measure_idx=m_i):
                poly_mono_commands[(tr_i, m_i)] = ';<poly>'
            else:
                poly_mono_commands[(tr_i, m_i)] = ';<mono>'
    return poly_mono_commands


# test written
def _build_finetune_train_data_recursive_helper(S: "ms.MidiSongByMeasure",
                                                pattern_type: "int",
                                                p_include_poly_mono: "float",
                                                p_extend_one_more_measure: "float",
                                                p_truncate: "float",
                                                extra_id_max: "int",
                                                tokenizer,
                                                is_first_example: "bool",
                                                n_measures_to_get: "int",
                                                try_countdown: "int"
                                                ):

    if try_countdown <= 0 or n_measures_to_get <= 0:
        return None

    while random.random() < p_extend_one_more_measure:
        n_measures_to_get += 1
    n_measures_to_get = min(S.get_n_measures(), n_measures_to_get)  # can't get more measures than we have

    # decide the start measure for this example
    if is_first_example:
        # first example starts at measure 0
        measure_st = 0
    else:
        # other examples start at random places
        measure_st = random.randint(0, S.get_n_measures() - n_measures_to_get)

    # artificially get fewer measures sometimes
    if random.random() < p_truncate:
        n_measures_to_drop = random.randint(0, n_measures_to_get - 1)
        n_measures_to_get -= n_measures_to_drop

    # get mask
    if pattern_type == 0:
        extra_params = {'mask_probability': 0.5}
    else:
        extra_params = None
    measure_slice = (measure_st, measure_st + n_measures_to_get)
    mask = get_random_mask(S, measure_slice=measure_slice,
                           pattern_type=pattern_type,
                           extra_params=extra_params)

    # define extra_id_st
    if len(mask) > extra_id_max:
        extra_id_st = 0
    else:
        extra_id_st = random.randint(0, extra_id_max - len(mask))

    # get poly mono commands sometimes
    if random.random() < p_include_poly_mono:
        poly_mono_commands = get_poly_mono_commands(S, mask=mask)
    else:
        poly_mono_commands = None

    continue_this_attempt = True
    try:
        input_ids_str, labels_str = enc.encode_midisongbymeasure_with_masks(
            S,
            note_off_treatment=spm_type_to_note_off_treatment(cs.SPM_TYPE),
            mask_locations=mask,
            measure_slice=measure_slice,
            include_heads_for_empty_masked_measures=False,
            poly_mono_commands=poly_mono_commands,
            return_labels_too=True,
            extra_id_st=extra_id_st,
            extra_id_max=extra_id_max
        )
    except ValueError:
        continue_this_attempt = False

    if continue_this_attempt:
        input_ids = tokenizer.Encode(input_ids_str)
        labels = tokenizer.encode(labels_str)
        if len(input_ids) > cs.MAX_LEN or len(labels) > cs.MAX_LEN - 1:
            continue_this_attempt = False

    if continue_this_attempt:
        input_ids_str = tokenizer.decode(input_ids)
        if (';N:' in input_ids_str or ';D:' in input_ids_str) and labels:
            this_res = {'input_ids': input_ids, 'labels': labels + [tokenizer.eos_id()]}
        else:
            continue_this_attempt = False

    if continue_this_attempt:
        return this_res

    else:
        if n_measures_to_get > 1:
            n_measures_to_get -= 1
        return _build_finetune_train_data_recursive_helper(S=S,
                                                           pattern_type=pattern_type,
                                                           p_include_poly_mono=p_include_poly_mono,
                                                           p_extend_one_more_measure=p_extend_one_more_measure,
                                                           p_truncate=p_truncate,
                                                           extra_id_max=extra_id_max,
                                                           tokenizer=tokenizer,
                                                           is_first_example=is_first_example,
                                                           n_measures_to_get=n_measures_to_get,
                                                           try_countdown=try_countdown - 1)


def _build_finetune_train_data(T):
    i_, k_, v_, tokenizer, epoch = T
    # setup
    S_orig = pre.midisongbymeasure_from_save_dict(v_)

    n_examples_to_get = 1 + S_orig.get_n_measures() // 16  # get at least 1 example per song
    n_examples_to_get = min(16, n_examples_to_get)  # get at most 16 examples per song
    n_tries = n_examples_to_get * 2

    # probabilities
    p_drop_tracks = 0.0
    p_include_poly_mono = 0.75
    p_extend_one_more_measure = 0.7  # must be < 1
    p_truncate = 0.15

    pattern_weights = {0: 4, 1: 6, 2: 1, 3: 1, 4: 1, 5: 1, 6: 4}

    extra_id_max = 255
    res = []

    if len(S_orig.tracks) == 0:
        return res

    done = False
    n_tries_so_far = 0
    while not done:
        S = copy.copy(S_orig)
        # each example we generate from this song gets a different transposition
        transpose_amt = get_trans_amt(epoch=epoch, i=i_ + len(res))
        S.transpose(amt=transpose_amt)
        enc.transpose_into_acceptable_ranges_TT(S)
        aug_bpm(S)  # each example we generate from this song gets a different BPM
        aug_vel(S)  # each example we generate from this song gets a different velocity augmentation
        S.sort_tracks_by_inst_and_avg_note_pitch()
        for t in S.tracks:
            t.sort()

        if random.random() < p_drop_tracks and len(S.tracks) > 1:
            # then drop at least one track. Do not drop all tracks.
            tracks_to_drop = random.sample(range(len(S.tracks)), random.randint(1, len(S.tracks) - 1))
            tracks_to_drop.sort()
            for i, pop_val in enumerate(tracks_to_drop):
                S.tracks.pop(pop_val - i)

        if len(S.tracks) == 1:
            pattern_type = weighted_choose_one({0: pattern_weights[0], 4: pattern_weights[4], 6: pattern_weights[6]})
        else:
            pattern_type = weighted_choose_one(pattern_weights)

        this_res = _build_finetune_train_data_recursive_helper(S=S,
                                                               pattern_type=pattern_type,
                                                               p_include_poly_mono=p_include_poly_mono,
                                                               p_extend_one_more_measure=p_extend_one_more_measure,
                                                               p_truncate=p_truncate,
                                                               extra_id_max=extra_id_max,
                                                               tokenizer=tokenizer,
                                                               is_first_example=len(res) == 0,
                                                               n_measures_to_get=32,
                                                               try_countdown=10)
        if this_res is not None:
            this_res['measure coverage'] = (tokenizer.decode(this_res['input_ids']).count(';M'), S_orig.get_n_measures())
            this_res['track coverage'] = (len(S.tracks), len(S_orig.tracks))
            this_res['pattern type'] = pattern_type
            this_res['source'] = k_
            res.append((k_, this_res))
            # print(' measr coverage', this_res['measure coverage'])
            # print(' pt and lens', this_res['pattern type'], len(this_res['input_ids']), len(this_res['labels']))
            # print()
        else:
            pass
            # print('got None res')
            # print()

        n_tries_so_far += 1
        done = (len(res) == n_examples_to_get) or (n_tries_so_far >= n_tries)

    # print('n examples found', len(res))
    return res

# def _get_measure_ints(tokenizer) -> "set[int]":
#     measure_ints = []
#     for x in range(tokenizer.vocab_size()):
#         if ';M' in tokenizer.decode(x):
#             measure_ints.append(x)
#     measure_ints = set(measure_ints)
#     return measure_ints
#
#
# def _get_extra_id_ints(tokenizer) -> "set[int]":
#     extra_id_ints = []
#     for x in range(tokenizer.vocab_size()):
#         if ';<extra_id_' in tokenizer.decode(x):
#             extra_id_ints.append(x)
#     extra_id_ints = set(extra_id_ints)
#     return extra_id_ints
#
#
# def _drop_one_measure(L, measure_ints) -> "list[int]":
#     # drop from left since it's easier
#     skip = True
#     for i, elt in enumerate(L):
#         if elt in measure_ints:
#             if skip:
#                 skip = False
#             else:
#                 L = L[i:]
#                 return L
#     return []
#
#
# def _cleanup_labels(labels: "list[int]", input_ids: "list[int]", extra_id_ints) -> "list[int]":
#     # relies on the fact that input ids were shortened by dropping on the left
#     for i, x in enumerate(labels):
#         if x in extra_id_ints and x in input_ids:
#             res = labels[i:]
#             return res
#     return []


# works; not needed anymore
# def build_finetune_train_data_one_thread(tokenizer, epoch, pool, path):
#     t0 = time.time()
#     print('Building finetune train data for epoch = {}'.format(epoch))
#
#     for folder, _, fnames in os.walk(path):
#         for fname in fnames:
#             with open(os.path.join(folder, fname)) as infile:
#                 d = json.load(infile)
#             d_keys = sorted(list(d.keys()))
#             items = [(i, k, d[k], tokenizer, epoch) for i, k in enumerate(d_keys)]
#             print('file {} loaded'.format(fname))
#
#             to_write = []
#             for i, T in enumerate(items):
#                 res = _build_finetune_train_data(T)
#                 to_write.extend(res)
#
#                 if (i + 1) % 1 == 0:
#                     print(i + 1, 'songs processed from this file so far')
#
#             print('writing finetune train data for epoch {}'.format(epoch))
#             # to compensate for random return order from imap_unordered
#             to_write.sort(key=lambda x: x[0])
#             to_write = [x[1] for x in to_write]
#             with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'finetune_epoch_{}_{}'.format(epoch, fname)), 'wb') as outfile:
#                 pickle.dump(to_write, outfile)
#
#     print('finished building finetune train data for epoch {} in {} sec'.format(epoch, time.time() - t0))


def build_finetune_train_data(tokenizer, epoch, pool, path):
    t0 = time.time()
    print('Building finetune train data for epoch = {}'.format(epoch))

    P = pool

    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            with open(os.path.join(folder, fname)) as infile:
                d = json.load(infile)
            d_keys = sorted(list(d.keys()))
            items = [(i, k, d[k], tokenizer, epoch) for i, k in enumerate(d_keys)]
            print('file {} loaded'.format(fname))

            to_write = []
            for i, res in enumerate(P.imap_unordered(_build_finetune_train_data, items, chunksize=10)):
                to_write.extend(res)

                if (i + 1) % 100 == 0:
                    print(i + 1, 'songs processed from this file so far')

            print('writing finetune train data for epoch {}'.format(epoch))
            # to compensate for random return order from imap_unordered
            to_write.sort(key=lambda x: x[0])
            to_write = [x[1] for x in to_write]
            with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'finetune_epoch_{}_{}'.format(epoch, fname)), 'wb') as outfile:
                pickle.dump(to_write, outfile)

    print('finished building finetune train data for epoch {} in {} sec'.format(epoch, time.time() - t0))


# helper function for build_pretrain_data
# test written
def _build_pretrain_data(T):
    i, k, v, tokenizer, epoch, target_len = T
    # random.seed(epoch)
    S = pre.midisongbymeasure_from_save_dict(v)
    S.transpose(amt=get_trans_amt(epoch=epoch, i=i))
    enc.transpose_into_acceptable_ranges_TT(S)
    aug_bpm(S)
    aug_vel(S)
    S.sort_tracks_by_inst_and_avg_note_pitch()
    for t in S.tracks:
        t.sort()
    s = enc.encode_midisongbymeasure(S, note_off_treatment=spm_type_to_note_off_treatment(cs.SPM_TYPE))
    ints = tokenizer.Encode(s)
    i = 0
    done = False
    res = []
    while not done:
        sl = ints[i * target_len: (i + 1) * target_len]
        if sl:
            corrupted = corrupt_pretrain(list_of_ints=sl, tokenizer=tokenizer)
            if corrupted[0] and corrupted[1]:
                this_res = {'input_ids': corrupted[0], 'labels': corrupted[1] + [tokenizer.eos_id()]}
                res.append((k, this_res))
            i += 1
        else:
            done = True
    return res


def build_pretrain_data(tokenizer, epoch, pool, mode):
    t0 = time.time()

    if mode in ('train', 'val_short'):
        max_len = 512 if epoch < cs.N_EPOCHS_SHORT else cs.MAX_LEN
    elif mode == 'val_long':
        max_len = cs.MAX_LEN
    else:
        raise ValueError('mode {} not recognized'.format(mode))

    addl = ' epoch = {}'.format(epoch) if mode == 'train' else ''
    print('Building pretrain data for mode = {}{}'.format(mode, addl))

    target_len = math.floor(max_len * 1 / (.85 + .15 / 3))
    n_corruptions_max = math.floor(0.15 * target_len / 3)  # 0 or more corruptions
    n_corruptions_max = min(256, n_corruptions_max)  # at most 256 corruptions
    while target_len - 2 * n_corruptions_max > max_len:
        target_len -= 1
        n_corruptions_max = math.floor(0.15 * target_len / 3)  # 0 or more corruptions
        n_corruptions_max = min(256, n_corruptions_max)  # at most 256 corruptions

    P = pool
    to_write = []

    if mode == 'train':
        path = cs.PATH_TO_PROCESSED_TRAIN_MIDI
    elif mode in ('val_short', 'val_long'):
        path = cs.PATH_TO_PROCESSED_VAL_MIDI
    else:
        raise ValueError('mode = {} not recognized'.format(mode))

    for folder, _, fnames in os.walk(path):
        for fname in fnames:
            with open(os.path.join(folder, fname)) as infile:
                d = json.load(infile)
            d_keys = sorted(list(d.keys()))
            items = [(i, k, d[k], tokenizer, epoch, target_len) for i, k in enumerate(d_keys)]
            print('file {} loaded'.format(fname))

            for i, res in enumerate(P.imap_unordered(_build_pretrain_data, items, chunksize=10)):
                to_write.extend(res)

                if (i + 1) % 1000 == 0:
                    print(i + 1, 'songs processed from this file so far')

            if mode == 'train':
                print('writing pretrain data for epoch {}'.format(epoch))
                # to compensate for random return order from imap_unordered
                to_write.sort(key=lambda x: x[0])
                to_write = [x[1] for x in to_write]
                with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_epoch_{}_{}'.format(epoch, fname)), 'wb') as outfile:
                    pickle.dump(to_write, outfile)
                to_write = []

    # Training data is already written above in chunks. Here we handle writing validation data to a single file.
    if mode == 'val_short':
        print('writing pretrain validation data (short inputs)')
    elif mode == 'val_long':
        print('writing pretrain validation data (long inputs)')

    # to compensate for random return order from imap_unordered
    to_write.sort(key=lambda x: x[0])
    to_write = [x[1] for x in to_write]

    if mode == 'val_long':
        with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_long.txt'), 'wb') as outfile:
            pickle.dump(to_write, outfile)
        print('finished building pretrain validation data in {} sec'.format(time.time() - t0))
    elif mode == 'val_short':
        with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_short.txt'), 'wb') as outfile:
            pickle.dump(to_write, outfile)
        print('finished building pretrain validation data in {} sec'.format(time.time() - t0))
    elif mode == 'train':
        print('finished building pretrain training data for epoch {} in {} sec'.format(epoch, time.time() - t0))


class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, epoch=None, mode='train'):
        """mode = 'train' or 'val_short' or 'val_long'"""
        self.data = []
        # load ALL training data into memory
        t0 = time.time()
        if mode == 'train':
            for folder, _, fnames in os.walk(cs.PATH_TO_TEMP_FILES):
                for fname in fnames:
                    if fname.find('pretrain_epoch_{}_'.format(epoch)) == 0:
                        with open(os.path.join(folder, fname), 'rb') as infile:
                            print('Loading files for PreTrainDataset mode={} epoch={}...'.format(mode, epoch))
                            d = pickle.load(infile)
                            self.data.extend(d)
        elif mode == 'val_short':
            with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_short.txt'), 'rb') as infile:
                print('Loading files for PreTrainDataset mode={}...'.format(mode))
                d = pickle.load(infile)
                self.data.extend(d)
        elif mode == 'val_long':
            with open(os.path.join(cs.PATH_TO_TEMP_FILES, 'pretrain_validation_long.txt'), 'rb') as infile:
                print('Loading files for PreTrainDataset mode={}...'.format(mode))
                d = pickle.load(infile)
                self.data.extend(d)

        if not self.data:
            if mode == 'train':
                err = 'pretrain epoch {}'.format(epoch)
            elif mode == 'val_short':
                err = 'validation (short sequences)'
            elif mode == 'val_long':
                err = 'validation (long sequences)'
            else:
                raise ValueError('mode {} not recognized'.format(mode))
            raise RuntimeError('No data for {} found. Did you run build_pretrain_data.py?'.format(err))

        n_tokens = 0
        for d in self.data:
            n_tokens += len(d['input_ids']) + len(d['labels'])
        print('PreTrainDataset containing {} examples ({} tokens; {} tokens/example) loaded in {} sec'.format(
            len(self.data), n_tokens, round(n_tokens/len(self.data), 1), time.time() - t0))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class FineTuneTrainDataset(torch.utils.data.Dataset):
    def __init__(self, epoch, max_len_override=None, slice_in_epoch=0, n_slices_in_epoch=1):
        """max_len_override just for development purposes"""
        self.data = []

        # load ALL training data into memory
        t0 = time.time()
        for folder, _, fnames in os.walk(cs.PATH_TO_TEMP_FILES):
            for fname in fnames:
                if fname.find('finetune_epoch_{}_'.format(epoch)) == 0:
                    print('Loading files from {} for FineTuneTrainDataset'.format(fname))
                    with open(os.path.join(folder, fname), 'rb') as infile:
                        d = pickle.load(infile)
                        self.data.extend(d)

        if not self.data:
            raise ValueError('No data for finetune epoch {} found. Did you run build_finetune_train_data.py?'.format(epoch))

        # remove extraneous keys from finetune examples
        for d in self.data:
            keys_to_del = []
            for k in d:
                if k not in ('input_ids', 'labels'):
                    keys_to_del.append(k)
            for k in keys_to_del:
                del d[k]

        # handle max_len_override
        if max_len_override is not None:
            new_data = []
            for d in self.data:
                if len(d['input_ids']) < max_len_override and len(d['labels']) < max_len_override:
                    new_data.append(d)
            self.data = new_data

        # handle slice within epoch
        new_data = []
        for i, d in enumerate(self.data):
            if i % n_slices_in_epoch == slice_in_epoch:
                new_data.append(d)
        self.data = new_data

        n_tokens = 0
        for d in self.data:
            n_tokens += len(d['input_ids']) + len(d['labels'])
        print('FineTuneTrainDataset containing {} examples ({} tokens; {} tokens/example) loaded in {} sec'.format(
            len(self.data), n_tokens, round(n_tokens/len(self.data), 1), time.time() - t0))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class FineTuneValTestDataset(torch.utils.data.Dataset):
    def __init__(self, mode, mask_pattern_type, n_measures):
        """
        mode = 'val' or 'test'

        mask_pattern_type in ("0half", "0quarter", 1, "2random", "2last", 4, 6)

        n_measures either 'all' or an int
        """
        self.data = []

        t0 = time.time()

        if mode == 'val':
            s = 'finetune_validation_'
        elif mode == 'test':
            s = 'finetune_test_'
        else:
            raise ValueError('mode {} not recognized'.format(mode))

        s += '{}_'.format(mask_pattern_type)
        s += '{}.txt'.format(n_measures)

        path = os.path.join(cs.PATH_TO_TEMP_FILES, s)
        if not os.path.exists(path):
            raise ValueError('No data file named {} found. Did you run build_val_and_test_finetune_data.py?'.format(s))

        with open(path) as infile:
            d = json.load(infile)
        for k in sorted([int(x) for x in d]):
            this_example = d[str(k)]
            self.data.append(this_example)

        print('FineTuneValTestDataset containing {} examples loaded in {} sec'.format(len(self.data), time.time()-t0))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def batch_padder(batch, tokenizer, max_padding=None, add_attention_mask=True):
    """uses tokenizer.pad_id() to pad on right.
    max_padding = an integer, if you ALWAYS want to pad to that amount. Otherwise, pads to max len in the batch.
    """
    keys_to_pad = ('input_ids', 'labels')

    res = collections.defaultdict(list)
    pad_id = tokenizer.pad_id()

    if max_padding is not None:
        max_input_len_dict = {k: max_padding for k in keys_to_pad}
    else:
        max_input_len_dict = {}
        for k in keys_to_pad:
            max_input_len = 0
            for b in batch:
                max_input_len = max(len(b[k]), max_input_len)
            max_input_len_dict[k] = max_input_len

    for b in batch:
        for k, v in b.items():
            if k == 'labels':
                to_pad = -100
            else:
                to_pad = pad_id

            res[k].append(v + [to_pad] * (max_input_len_dict[k] - len(v)))

        if add_attention_mask:
            k = 'input_ids'
            res['attention_mask'].append(
                [1] * len(b[k]) + [0] * (max_input_len_dict[k] - len(b[k]))
            )

    # convert to tensors:
    for k in res:
        res[k] = torch.tensor(res[k], dtype=torch.long)
    return res
