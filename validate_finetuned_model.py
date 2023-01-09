import json
import statistics
import time
import constants as cs
import os
import transformers
import torch
import nn_training_functions as fn
import functools
import nn_output_parser as nns
import collections
import tokenizer_functions as tok
import nn_input_chunker as chunker
import midisong as ms

EPOCHS_TO_ANALYZE = [(11, 0)]
BATCH_SIZE = 25
DO_SAMPLE = 0  # 0 or False for greedy decoding; 1 or higher for number of return sequences for top-p (p=0.95) sampling
# note that when DO_SAMPLE > 0, actual batch size is BATCH_SIZE * DO_SAMPLE
TRUNCATE_DATASET = False  # False (or 0), or number of samples
CHUNK_BY_INPUT_LEN = True
DELETE_MONO_POLY = False
DELETE_MONO_POLY_BEFORE_CHUNKING = False  # this is a switch that only matters when DELETE_MONO_POLY is true.

# If 0, use neural net.
# If 1, just predict the most common note for each pitched instrument and most common drum beat.
# If 2, predict by copying the most recent prior measure for this instrument (if no prior measure, then
#       copy closest subsequent measure; if no subsequent measure, then copy a nearby instrument in same measure;
#       if that fails, then use strategy 1.)
NAIVE = 0

# EXPLORE = 0  # 0 (False) or 1 or 2; 1 explores the dataset elements that get chunked; 2 evaluates those elements without chunking them
# EXPLORE_F1_0 = False

TOKENIZER = tok.get_tokenizer()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DO_SAMPLE:
    transformers.set_seed(42)  # for top_p reproducibility


def get_model(epoch, slice_in_epoch):
    base_load_path = os.path.join(cs.PATH_TO_MODELS, 'finetuned_epoch_{}_{}'.format(epoch, slice_in_epoch))
    print('Loading model from: {}...'.format(base_load_path))
    if not os.path.exists(base_load_path):
        print('No model found for epoch {}_{}. Skipping.'.format(epoch, slice_in_epoch))
        return None
    else:
        M = transformers.T5ForConditionalGeneration.from_pretrained(os.path.join(base_load_path, 'model'))
        M.eval()
        M = M.to(DEVICE)
        print('Model loaded.')
        return M


def batch_padder_for_val_and_test(batch, tokenizer, add_attention_mask=True):
    res = collections.defaultdict(list)
    pad_id = tokenizer.pad_id()

    max_input_len_dict = {}
    for k in ('input_ids', 'labels'):
        max_input_len = 0
        for b in batch:
            max_input_len = max(len(b[k]), max_input_len)
        max_input_len_dict[k] = max_input_len

    for b in batch:
        for k, v in b.items():
            if k == 'labels':
                to_pad = -100
            elif k == 'input_ids':
                to_pad = pad_id
            else:
                to_pad = None

            if to_pad is not None:
                res[k].append(v + [to_pad] * (max_input_len_dict[k] - len(v)))
            else:
                res[k].append(v)

        if add_attention_mask:
            k = 'input_ids'
            res['attention_mask'].append(
                [1] * len(b[k]) + [0] * (max_input_len_dict[k] - len(b[k]))
            )

    # convert to tensors:
    for k in res:
        if k in ('input_ids', 'labels', 'attention_mask'):
            res[k] = torch.tensor(res[k], dtype=torch.long)
    return res


def get_generated_samples_greedy(model, dataset):
    collator = functools.partial(batch_padder_for_val_and_test, tokenizer=TOKENIZER)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=collator)
    for i, x in enumerate(dataloader):
        # gen_len = x['labels'].shape[1] + 1  # start with the decoder_start_token, then generate enough outputs
        max_label_len = x['labels'].shape[1]
        print('max label len: {}'.format(max_label_len))
        output_generated = model.generate(input_ids=x['input_ids'].to(DEVICE),
                                          num_return_sequences=1,
                                          # do_sample=use_sampling,
                                          # temperature=0.8,
                                          # remove_invalid_values=True,
                                          # top_k=100,
                                          # top_p=top_p,
                                          # min_length=min_length,
                                          # max_new_tokens=cs.MAX_LEN,
                                          max_new_tokens=min(max_label_len * 2, 3000),
                                          decoder_start_token_id=TOKENIZER.pad_id(),
                                          pad_token_id=TOKENIZER.pad_id(),
                                          bos_token_id=TOKENIZER.bos_id(),
                                          eos_token_id=TOKENIZER.eos_id(),
                                          use_cache=True,
                                          # force_words_ids=forced_ids,
                                          # encoder_no_repeat_ngram_size=enc_no_repeat_ngram_size,
                                          # repetition_penalty=1.01
                                          )
        output_generated = output_generated[:, 1:]  # drop the decoder_start_token at the beginning of each generation
        output_generated = output_generated.to('cpu')

        # roughly range(BATCH_SIZE), but accounts for smaller chunk at the end as well
        for output_i in range(x['input_ids'].shape[0]):
            this_output = output_generated[output_i, :].tolist()
            this_labels = x['labels'][output_i, :].tolist()
            this_labels = [a for a in this_labels if a >= 0]
            this_input = x['input_ids'][output_i, :].tolist()
            this_input = [a for a in this_input if a != TOKENIZER.pad_id()]
            this_output_index = 0
            yield this_output, this_labels, x['processed_source'][output_i], this_input, this_output_index


def get_generated_samples_top_p(model, dataset):
    collator = functools.partial(batch_padder_for_val_and_test, tokenizer=TOKENIZER)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=collator)
    for i, x in enumerate(dataloader):
        # gen_len = x['labels'].shape[1] + 1  # start with the decoder_start_token, then generate enough outputs
        max_label_len = x['labels'].shape[1]
        print('max label len: {}'.format(max_label_len))
        output_generated = model.generate(input_ids=x['input_ids'].to(DEVICE),
                                          num_return_sequences=DO_SAMPLE,
                                          do_sample=True,
                                          # temperature=0.8,
                                          # remove_invalid_values=True,
                                          # top_k=100,
                                          top_p=0.95,
                                          max_new_tokens=min(max_label_len * 2, 3000),
                                          decoder_start_token_id=TOKENIZER.pad_id(),
                                          pad_token_id=TOKENIZER.pad_id(),
                                          bos_token_id=TOKENIZER.bos_id(),
                                          eos_token_id=TOKENIZER.eos_id(),
                                          use_cache=True,
                                          )
        output_generated = output_generated[:, 1:]  # drop the decoder_start_token at the beginning of each generation
        output_generated = output_generated.to('cpu')

        # roughly range(BATCH_SIZE), but accounts for smaller chunk at the end as well
        for output_i in range(output_generated.shape[0]):
            this_output = output_generated[output_i, :].tolist()
            this_labels = x['labels'][output_i // DO_SAMPLE, :].tolist()
            this_labels = [a for a in this_labels if a >= 0]
            this_input = x['input_ids'][output_i // DO_SAMPLE, :].tolist()
            this_input = [a for a in this_input if a != TOKENIZER.pad_id()]
            this_processed_source = x['processed_source'][output_i // DO_SAMPLE]
            this_output_index = output_i % DO_SAMPLE

            yield this_output, this_labels, this_processed_source, this_input, this_output_index


# tested via testing test_get_naive_sample
def _get_naive_sample_strat_2_get_closest_match(k, d, measure_len):
    inst, inst_rep, measure_i = k

    # copy from most recent prior measure
    for m_i in range(measure_i - 1, -1, -1):
        if (inst, inst_rep, m_i) in d:
            res = d[(inst, inst_rep, m_i)]
            if res and '<extra_id_' not in res:
                return res

    # if nothing in prior measure, copy from closest subsequent measure
    max_measure_i = max(x[2] for x in d)
    for m_i in range(measure_i + 1, max_measure_i + 1):
        if (inst, inst_rep, m_i) in d:
            res = d[(inst, inst_rep, m_i)]
            if res and '<extra_id_' not in res:
                return res

    # if still nothing, copy from another instrument in same measure; don't copy drums
    skip_this = inst == 128
    if not skip_this:
        keys = [x for x in d if x[2] == measure_i and x[0] != 128]
        keys.sort(key=lambda x: x[1])
        keys.sort(key=lambda x: abs(x[0] - inst))
        for x in keys:
            res = d[x]
            if res and '<extra_id_' not in res:
                return res

    # if still nothing, use strategy 1
    example_str = f';M:5;B:5;L:{measure_len};I:{inst};<extra_id_0>'
    return get_naive_sample(example_str=example_str, strategy=1, include_extra_ids=False)


# test written
def get_naive_sample(example_str, strategy=1, include_extra_ids=True):
    most_common = {
        0: (62, 12),
        1: (62, 12),
        2: (62, 12),
        3: (62, 12),
        4: (62, 12),
        5: (62, 12),
        6: (69, 6),
        7: (67, 6),
        8: (81, 6),
        9: (96, 12),
        10: (79, 6),
        11: (67, 12),
        12: (67, 12),
        13: (79, 12),
        14: (67, 24),
        15: (69, 6),
        16: (69, 6),
        17: (74, 6),
        18: (72, 6),
        19: (67, 12),
        20: (67, 12),
        21: (67, 3),
        22: (69, 12),
        23: (69, 3),
        24: (62, 12),
        25: (57, 12),
        26: (64, 12),
        27: (62, 6),
        28: (57, 6),
        29: (52, 12),
        30: (52, 12),
        31: (57, 12),
        32: (38, 12),
        33: (33, 12),
        34: (33, 12),
        35: (38, 12),
        36: (33, 12),
        37: (36, 12),
        38: (36, 6),
        39: (36, 12),
        40: (74, 12),
        41: (62, 12),
        42: (50, 12),
        43: (38, 12),
        44: (67, 6),
        45: (62, 12),
        46: (67, 12),
        47: (48, 3),
        48: (62, 6),
        49: (62, 96),
        50: (62, 96),
        51: (64, 12),
        52: (67, 12),
        53: (64, 12),
        54: (67, 12),
        55: (67, 12),
        56: (67, 12),
        57: (55, 12),
        58: (43, 12),
        59: (69, 6),
        60: (62, 12),
        61: (67, 6),
        62: (67, 6),
        63: (69, 6),
        64: (69, 12),
        65: (67, 12),
        66: (60, 12),
        67: (48, 12),
        68: (74, 12),
        69: (67, 12),
        70: (50, 12),
        71: (67, 12),
        72: (91, 6),
        73: (79, 12),
        74: (79, 12),
        75: (74, 12),
        76: (72, 6),
        77: (74, 12),
        78: (74, 12),
        79: (72, 6),
        80: (67, 6),
        81: (67, 6),
        82: (74, 12),
        83: (72, 6),
        84: (60, 12),
        85: (67, 12),
        86: (60, 3),
        87: (41, 6),
        88: (67, 12),
        89: (62, 96),
        90: (67, 6),
        91: (67, 12),
        92: (67, 12),
        93: (38, 6),
        94: (71, 6),
        95: (62, 96),
        96: (69, 6),
        97: (42, 6),
        98: (74, 6),
        99: (69, 6),
        100: (72, 6),
        101: (90, 3),
        102: (69, 12),
        103: (72, 6),
        104: (64, 12),
        105: (62, 12),
        106: (64, 6),
        107: (69, 6),
        108: (67, 6),
        109: (69, 12),
        110: (69, 12),
        111: (69, 6),
        112: (79, 12),
        113: (62, 6),
        114: (67, 6),
        115: (60, 6),
        116: (48, 6),
        117: (48, 6),
        118: (36, 6),
        119: (60, 48),
        120: (48, 6),
        121: (62, 0),
        122: (45, 0),
        123: (54, 0),
        124: (76, 4),
        125: (65, 4),
        126: (42, 3),
        127: (64, 6),
    }
    max_measure_len = ms.extended_lcm(cs.QUANTIZE) * 8
    this_output = ""
    infos = nns.infos_by_extra_id(example_str)

    if strategy == 1:
        for k in infos:
            inst = infos[k]['inst']
            if 0 <= inst <= 127:
                cur_m_len = 0
                T = most_common[inst]
                dur = T[1]
                wait = dur if dur != 0 else 6
                s = f';{k};d:{dur}' if include_extra_ids else f';d:{dur}'
                while cur_m_len < max_measure_len:
                    s += ';N:{};w:{}'.format(T[0], wait)
                    cur_m_len += wait
            elif inst == 128:
                s = f';{k}' if include_extra_ids else ''
                s += ';D:36;D:42;w:12;D:42;w:12;D:38;D:42;w:12;D:42;w:12' * (max_measure_len // 48)

            else:
                raise ValueError(f'inst {inst} unrecognized')
            this_output += s

        return this_output

    # If 2, predict by copying the most recent prior measure for this instrument (if no prior measure, then
    #       copy closest subsequent measure; if no subsequent measure, then copy a nearby instrument in same measure
    elif strategy == 2:
        d = nns.deconstructed_input_str(example_str)
        for k in infos:
            inst = infos[k]['inst']
            inst_rep = infos[k]['inst_rep']
            measure_i = infos[k]['measure_index']
            measure_len = infos[k]['measure_len']
            s = f';{k}' if include_extra_ids else ''
            s += _get_naive_sample_strat_2_get_closest_match(k=(inst, inst_rep, measure_i),
                                                             d=d, measure_len=measure_len)
            this_output += s

        return this_output


def get_naive_samples(dataset, strategy=1):
    for x in dataset:
        input_ids_s = TOKENIZER.decode(x['input_ids'])
        this_output = get_naive_sample(example_str=input_ids_s, strategy=strategy)
        this_output = TOKENIZER.encode(this_output)
        this_labels = x['labels']
        this_processed_source = x['processed_source']
        this_input = x['input_ids']
        this_output_index = 0
        yield this_output, this_labels, this_processed_source, this_input, this_output_index


# test written
def get_note_onset_F1(output_dict: "collections.defaultdict[str, list[tuple]]",
                      intention_dict: "collections.defaultdict[str, list[tuple]]",
                      return_dict: "bool" = False,
                      ) -> float or None or dict:
    # recall = # relevant retrieved / # total relevant
    # precision = # relevant retrieved / # total retrieved

    numerator = 0
    total_relevant = 0
    total_retrieved = 0
    # all_keys = set(output_dict.keys()).union(set(intention_dict.keys()))
    all_keys = intention_dict.keys()  # only evaluate the net on its outputs that have a corresponding mask in the input
    for k in all_keys:
        retrieved = set(output_dict[k])
        relevant = set(intention_dict[k])
        numerator += len(retrieved.intersection(relevant))
        total_relevant += len(relevant)
        total_retrieved += len(retrieved)

    if return_dict:
        res = {'numerator': numerator, 'n_relevant': total_relevant, 'n_retrieved': total_retrieved}
        return res

    else:  # then return F1
        if total_relevant == 0:
            return None
        else:
            recall = numerator / total_relevant

        if total_retrieved == 0:
            precision = 0
        else:
            precision = numerator / total_retrieved

        if precision + recall == 0:
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        return F1


# test written via testing generic_F1
def generic_precision(numerator, n_retrieved):
    if n_retrieved == 0:
        return 0
    else:
        return numerator/n_retrieved


# test written via testing generic_F1
def generic_recall(numerator, n_relevant):
    if n_relevant == 0:
        return None
    else:
        return numerator / n_relevant


# test written
def generic_F1(numerator, n_relevant, n_retrieved):
    recall = generic_recall(numerator=numerator, n_relevant=n_relevant)
    precision = generic_precision(numerator=numerator, n_retrieved=n_retrieved)

    if recall is None:  # if there are no relevant documents, F1 is None (undefined)
        return None

    denom = recall + precision
    if denom == 0:
        return 0

    return 2 * precision * recall / denom


# test written
def list_of_instructions_to_list_of_pitch_pos_DN_tuples(L: "list[str]", cutoff_pos=None) -> "list[tuple]":
    if cutoff_pos is None:
        cutoff_pos = 9999999

    res = []
    cur_duration = 0
    cur_pos = 0
    for instruction in L:
        if instruction[0] in ('w', 'd', 'N', 'D'):
            s = instruction.split(':')
            if s[0] == 'd':
                cur_duration = int(s[1])
            elif s[0] == 'w':
                cur_pos += int(s[1])
            else:
                pitch = int(s[1])
                # vel and inst arbitrary
                # note = ms.Note(pitch=pitch, vel=96, click=cur_pos, end=cur_pos + cur_duration, inst=None)
                note = (pitch, cur_pos, s[0])  # ex: (pitch) 36, (pos) 12, (D or N) N
                if cur_pos < cutoff_pos:
                    res.append(note)
    return res


# test written
def get_only_tuples_whose_last_entry_is_x(d: "collections.defaultdict[str, list[tuple]]", x):
    res = collections.defaultdict(list)
    for k, L in d.items():
        for v in L:
            if v[-1] == x:
                res[k].append(v)
    return res


def labels_to_dict_of_pitch_pos_DN_tuples_by_extra_id(L: "list[int]", extra_id_to_infos: "dict") -> "collections.defaultdict[str, list[tuple]]":
    item = TOKENIZER.decode(L)
    item = nns.instructions_by_extra_id(item)
    res = collections.defaultdict(list)
    for k in item:
        try:
            measure_len = extra_id_to_infos[k]["measure_len"]
        except KeyError:
            print(f"warning: generated extra id {k} not in input_ids")
            measure_len = None
        res[k] = list_of_instructions_to_list_of_pitch_pos_DN_tuples(item[k], cutoff_pos=measure_len)
    return res


def apply_chunk_by_input_len(L: "iterable", tokenizer):
    new_L = []
    for x in L:
        for chunk in chunker.chunk_by_input_len(tokenizer=tokenizer, input_ids=x['input_ids'], labels=x['labels'],
                                                max_input_len=cs.MAX_LEN):
            new_ex = {'input_ids': chunk[0], 'labels': chunk[1], 'processed_source': x['processed_source']}
            new_L.append(new_ex)
    return new_L


def apply_grab_right_chunks_only(L: "iterable", tokenizer):
    new_L = []
    for x in L:
        for chunk in chunker.get_rightmost_chunk(tokenizer=tokenizer, input_ids=x['input_ids'], labels=x['labels'],
                                                 max_input_len=cs.MAX_LEN):
            new_ex = {'input_ids': chunk[0], 'labels': chunk[1], 'processed_source': x['processed_source']}
            new_L.append(new_ex)
    return new_L


def apply_chunk_by_max_extra_id(L: "iterable"):
    new_L = []
    for x in L:
        for chunk in chunker.chunk_by_max_extra_id(input_ids=x['input_ids'], labels=x['labels'], max_extra_id=255):
            new_ex = {'input_ids': chunk[0], 'labels': chunk[1], 'processed_source': x['processed_source']}
            new_L.append(new_ex)
    return new_L


# test written
def to_mod_n(n: int, d: "collections.defaultdict[str, list[tuple]]") -> "collections.defaultdict[str, list[tuple]]":
    res = collections.defaultdict(list)
    for k in d:
        res[k] = []
        for T in d[k]:
            a, b, c = T
            new_tuple = (a % n, b, c)
            if new_tuple not in res[k]:
                res[k].append(new_tuple)
    return res


# test written
def remove_extra_ids_in_str(s: str = "", min_extra_id=0, max_extra_id=255) -> str:
    """naive implementation"""
    for k in range(min_extra_id, max_extra_id + 1):
        s = s.replace(f";<extra_id_{k}>", '')
    return s


# test written
def valid_str_to_MidiSong(s: str = "", cpq=24) -> "ms.MidiSong":
    """only works if no note offs or if note off treatment is 'duration'"""
    import encoding_functions as enc

    sL = s.split(';M:')[1:]
    notes_by_measure = []
    loudness_by_measure = []
    bpm_by_measure = []
    clicks_by_measure = []
    insts = set()

    for measure_str in sL:
        measure_str = 'M:' + measure_str
        measure_loudness_setting = 5
        measure_bpm_setting = 4
        measure_n_clicks = cpq * 4

        measure_L = measure_str.split(';')
        cur_pos_in_measure = 0
        cur_inst = 0
        cur_inst_rep = 0
        cur_note_dur = 0
        notes = collections.defaultdict(list)
        for instruction in measure_L:
            if ':' in instruction:
                a, b = instruction.split(':')
                if a == 'B':
                    measure_bpm_setting = int(b)
                elif a == 'M':
                    measure_loudness_setting = int(b)
                elif a == 'L':
                    measure_n_clicks = int(b)
                elif a == 'I':
                    cur_inst = int(b)
                    cur_inst_rep = 0
                    cur_note_dur = 0
                    cur_pos_in_measure = 0
                elif a == 'R':
                    cur_inst_rep = int(b)
                elif a in ('N', 'D'):
                    insts.add((cur_inst, cur_inst_rep))
                    vel = enc.DYNAMICS_SLICER[measure_loudness_setting] if measure_loudness_setting < 7 else 124
                    notes[(cur_inst, cur_inst_rep)].append(ms.Note(pitch=int(b),
                                                                   vel=round(vel),
                                                                   click=cur_pos_in_measure,
                                                                   end=cur_pos_in_measure + cur_note_dur))
                elif a == 'd':
                    cur_note_dur = int(b)
                elif a == 'w':
                    cur_pos_in_measure += int(b)
        notes_by_measure.append(notes)
        loudness_by_measure.append(measure_loudness_setting)
        bpm_by_measure.append(measure_bpm_setting)
        clicks_by_measure.append(measure_n_clicks)

    inst_to_max_rep = {}
    for notes in notes_by_measure:
        for T in notes:
            inst, rep = T
            if inst not in inst_to_max_rep:
                inst_to_max_rep[inst] = rep
            else:
                inst_to_max_rep[inst] = max(inst_to_max_rep[inst], rep)
    insts = sorted(list(set(inst_to_max_rep.keys())))

    inst_rep_tuple_to_tr_index = {}
    tr_index_to_inst_rep_tuple = {}
    tr_i = 0
    for inst in insts:
        max_rep = inst_to_max_rep[inst]
        for k in range(max_rep + 1):
            inst_rep_tuple_to_tr_index[(inst, k)] = tr_i
            tr_index_to_inst_rep_tuple[tr_i] = (inst, k)
            tr_i += 1

    MEs = [sum(clicks_by_measure[:x]) for x in range(len(clicks_by_measure)+1)]
    TempoChanges = [ms.TempoChange(val=round(enc.BPM_SLICER[x]) if x < 7 else 200, click=MEs[i]) for i, x in enumerate(bpm_by_measure)]
    S = ms.MidiSongByMeasure(tracks=[],
                             measure_endpoints=MEs,
                             tempo_changes=TempoChanges,
                             cpq=cpq)
    Tracks = []
    for tr_i in sorted(list(tr_index_to_inst_rep_tuple.keys())):
        Tracks.append(ms.Track(inst=tr_index_to_inst_rep_tuple[tr_i][0]))
    S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=True)
    S.tracks = Tracks
    for measure_i, notes_in_measure in enumerate(notes_by_measure):
        measure_st_click = MEs[measure_i]
        for T, notes_for_T in notes_in_measure.items():
            track_i = inst_rep_tuple_to_tr_index[T]
            for note in notes_for_T:
                note.end += measure_st_click
                note.click += measure_st_click
                S.tracks[track_i].notes.append(note)

    return S


def index_of_max(L):
    m_v, m_i = None, None
    for i, v in enumerate(L):
        if m_v is None:
            m_v = v
            m_i = i
        else:
            if v > m_v:
                m_v = v
                m_i = i
    return m_i


def analyze(epoch_tuple, mode, mask_pattern_type, n_measures):
    t0 = time.time()
    M = get_model(epoch=epoch_tuple[0], slice_in_epoch=epoch_tuple[1])
    if M is None:
        return

    EXPLORE_F1_0_COUNTER = 0
    n_outputs_per_source = DO_SAMPLE if DO_SAMPLE else 1

    dataset = fn.FineTuneValTestDataset(mode=mode, mask_pattern_type=mask_pattern_type, n_measures=n_measures)
    if TRUNCATE_DATASET:
        dataset = torch.utils.data.Subset(dataset, torch.arange(0, TRUNCATE_DATASET))
    # if EXPLORE_F1_0:
    #     dataset = torch.utils.data.Subset(dataset, torch.arange(9200, 9300))
    if DELETE_MONO_POLY and DELETE_MONO_POLY_BEFORE_CHUNKING:
        for x in dataset:
            x['input_ids'] = x['input_ids'].replace(';<mono>', '')
            x['input_ids'] = x['input_ids'].replace(';<poly>', '')
    if CHUNK_BY_INPUT_LEN:
        if 'last' not in mask_pattern_type:
            print('chunking by input length')
            dataset = apply_chunk_by_input_len(dataset, tokenizer=TOKENIZER)
        else:
            print('chunking by grabbing largest possible rightmost chunk only')
            dataset = apply_grab_right_chunks_only(dataset, tokenizer=TOKENIZER)
    dataset = apply_chunk_by_max_extra_id(dataset)
    # dataset is now a list of dicts; each dict has keys 'input_ids', 'labels', and 'processed_source'
    print(f'dataset chunked; {time.time() - t0} sec elapsed so far')
    print(f'number of pieces in dataset after chunking: {len(dataset)}')
    # if EXPLORE:
    #     new_dataset = []
    #     sources = [x['processed_source'] for x in dataset]
    #     sources_seen = set()
    #     for source in sources:
    #         if sources.count(source) > 1 and source not in sources_seen:
    #             for x in dataset:
    #                 if x['processed_source'] == source:
    #                     new_dataset.append(x)
    #         sources_seen.add(source)
    #     print(f'new dataset size (0) = {len(new_dataset)}')
    #     dataset = new_dataset
    #
    #     if EXPLORE == 2:
    #         new_dataset = fn.FineTuneValTestDataset(mode=mode, mask_pattern_type=mask_pattern_type, n_measures=n_measures)
    #         sources = set([x['processed_source'] for x in dataset])
    #         new_dataset = [x for x in new_dataset if x['processed_source'] in sources]
    #         print(f'new dataset size (1) = {len(new_dataset)}')
    #         dataset = new_dataset
    #         dataset = apply_chunk_by_max_extra_id(dataset)  # only chunk by extra id, not by tokenized length
    #         print(f'new dataset size (2) = {len(dataset)}')

    if DELETE_MONO_POLY and not DELETE_MONO_POLY_BEFORE_CHUNKING:
        for x in dataset:
            x['input_ids'] = x['input_ids'].replace(';<mono>', '')
            x['input_ids'] = x['input_ids'].replace(';<poly>', '')

    for d in dataset:
        d['input_ids'] = TOKENIZER.encode(d['input_ids'])  # input_ids is a list of ints; labels is a string
        d['labels'] = TOKENIZER.encode(d['labels'])  # appending eos is not needed since labels are never fed into model

        # clean up keys
        # to_del = []
        # for k in d:
        #     if k not in ('labels', 'input_ids'):
        #         to_del.append(k)
        # for k in to_del:
        #     del d[k]

    dataset_len = len(dataset)

    source_to_outputs_and_intentions = collections.defaultdict(list)
    if NAIVE:
        sampler = get_naive_samples(dataset=dataset, strategy=NAIVE)
    elif DO_SAMPLE:
        sampler = get_generated_samples_top_p(model=M, dataset=dataset)
    else:
        sampler = get_generated_samples_greedy(model=M, dataset=dataset)

    for i, sample in enumerate(sampler):
        output, intention, source, input_ids, output_index = sample
        extra_info = {'input_ids': input_ids, 'labels': intention, 'output': output, 'output_index': output_index}
        extra_id_to_infos = nns.infos_by_extra_id(TOKENIZER.decode(input_ids))
        output = labels_to_dict_of_pitch_pos_DN_tuples_by_extra_id(output, extra_id_to_infos)
        intention = labels_to_dict_of_pitch_pos_DN_tuples_by_extra_id(intention, extra_id_to_infos)
        source_to_outputs_and_intentions[source].append((output, intention, extra_info))
        print(i + 1, f'/ {n_outputs_per_source * dataset_len} outputs from neural net done so far')

    print(f'dataset evaluated by neural net; {time.time() - t0} sec elapsed so far')

    # collate by source and output index
    source_to_F1_stats = {}
    for source, L in source_to_outputs_and_intentions.items():
        source_to_F1_stats[source] = []
        for output_i in range(n_outputs_per_source):
            source_to_F1_stats[source].append({'full': {}, 'drums': {}, 'pitched': {}, 'pitched_mod_12': {}, 'pitched_mod_1': {}})
        for output_i in range(n_outputs_per_source):
            for k, v in source_to_F1_stats[source][output_i].items():
                v['numerator'] = 0
                v['n_relevant'] = 0
                v['n_retrieved'] = 0

        for (output, intention, extra_info) in L:
            output_pitched_insts_only = get_only_tuples_whose_last_entry_is_x(output, 'N')
            output_drums_only = get_only_tuples_whose_last_entry_is_x(output, 'D')

            intention_pitched_insts_only = get_only_tuples_whose_last_entry_is_x(intention, 'N')
            intention_drums_only = get_only_tuples_whose_last_entry_is_x(intention, 'D')

            output_pitched_mod_12 = to_mod_n(n=12, d=output_pitched_insts_only)
            output_pitched_mod_1 = to_mod_n(n=1, d=output_pitched_insts_only)
            intention_pitched_mod_12 = to_mod_n(n=12, d=intention_pitched_insts_only)
            intention_pitched_mod_1 = to_mod_n(n=1, d=intention_pitched_insts_only)

            F1_full = get_note_onset_F1(output_dict=output, intention_dict=intention, return_dict=True)
            F1_drums = get_note_onset_F1(output_dict=output_drums_only, intention_dict=intention_drums_only,
                                         return_dict=True)
            F1_pitched = get_note_onset_F1(output_dict=output_pitched_insts_only,
                                           intention_dict=intention_pitched_insts_only, return_dict=True)
            F1_pitched_mod_12 = get_note_onset_F1(output_dict=output_pitched_mod_12,
                                                  intention_dict=intention_pitched_mod_12, return_dict=True)
            F1_pitched_mod_1 = get_note_onset_F1(output_dict=output_pitched_mod_1,
                                                 intention_dict=intention_pitched_mod_1, return_dict=True)

            output_i = extra_info['output_index']
            for k, v in F1_full.items():
                source_to_F1_stats[source][output_i]['full'][k] += v
            for k, v in F1_drums.items():
                source_to_F1_stats[source][output_i]['drums'][k] += v
            for k, v in F1_pitched.items():
                source_to_F1_stats[source][output_i]['pitched'][k] += v
            for k, v in F1_pitched_mod_12.items():
                source_to_F1_stats[source][output_i]['pitched_mod_12'][k] += v
            for k, v in F1_pitched_mod_1.items():
                source_to_F1_stats[source][output_i]['pitched_mod_1'][k] += v

            # if EXPLORE_F1_0 and F1_full['numerator'] == 0 and F1_full['n_relevant'] > 0:
            #     this_prompt = TOKENIZER.decode(extra_info['input_ids'])
            #     this_intention = TOKENIZER.decode(fn.uncorrupt(input_ids=extra_info['input_ids'], labels=extra_info['labels'], tokenizer=TOKENIZER))
            #     this_output = TOKENIZER.decode(fn.uncorrupt(input_ids=extra_info['input_ids'], labels=extra_info['output'], tokenizer=TOKENIZER))
            #     this_output = ';' + ';'.join(nns.parse_instruction_str(this_output))
            #     print(this_prompt)
            #     print(output)
            #     print()
            #     S_prompt = valid_str_to_MidiSong(s=this_prompt)
            #     S_prompt.dump(r'D:\temp\{}_prompt.mid'.format(EXPLORE_F1_0_COUNTER))
            #     S_intention = valid_str_to_MidiSong(s=this_intention)
            #     S_intention.dump(r'D:\temp\{}_intention.mid'.format(EXPLORE_F1_0_COUNTER))
            #     S_output = valid_str_to_MidiSong(s=this_output)
            #     S_output.dump(r'D:\temp\{}_output.mid'.format(EXPLORE_F1_0_COUNTER))
            #
            #     EXPLORE_F1_0_COUNTER += 1

    # For each source and evaluation target, choose best F1 (used when n_outputs_per_source > 1).
    # Always use song F1 to score the song regardless of f1_method below.
    source_to_best_F1_output = {}
    for source in source_to_F1_stats:
        for eval_target in ('full', 'drums', 'pitched', 'pitched_mod_12', 'pitched_mod_1'):
            F1s = []
            for output_i in range(n_outputs_per_source):
                numerator = source_to_F1_stats[source][output_i][eval_target]['numerator']
                n_relevant = source_to_F1_stats[source][output_i][eval_target]['n_relevant']
                n_retrieved = source_to_F1_stats[source][output_i][eval_target]['n_retrieved']
                F1s.append(generic_F1(numerator=numerator, n_relevant=n_relevant, n_retrieved=n_retrieved))
            for i, x in enumerate(F1s):
                if x is None:
                    F1s[i] = -1.0
            best_output_i = index_of_max(F1s)
            source_to_best_F1_output[(source, eval_target)] = best_output_i

    # compute dataset F1 scores
    res = {'full': {}, 'drums': {}, 'pitched': {}, 'pitched_mod_12': {}, 'pitched_mod_1': {}}
    for eval_target in res:
        for f1_method in ('avg_f1', 'dataset_prec_and_recall', 'avg_precs_and_recalls'):

            if f1_method == 'avg_f1':
                F1_L = []
                for source in source_to_F1_stats:
                    best_output_i = source_to_best_F1_output[(source, eval_target)]
                    numerator = source_to_F1_stats[source][best_output_i][eval_target]['numerator']
                    n_relevant = source_to_F1_stats[source][best_output_i][eval_target]['n_relevant']
                    n_retrieved = source_to_F1_stats[source][best_output_i][eval_target]['n_retrieved']
                    F1 = generic_F1(numerator=numerator, n_relevant=n_relevant, n_retrieved=n_retrieved)
                    if F1 is not None:
                        F1_L.append(F1)
                if F1_L:
                    res[eval_target][f1_method] = statistics.mean(F1_L)
                else:
                    res[eval_target][f1_method] = None

            elif f1_method == 'dataset_prec_and_recall':
                numerator = n_retrieved = n_relevant = 0
                for source in source_to_F1_stats:
                    best_output_i = source_to_best_F1_output[(source, eval_target)]
                    numerator += source_to_F1_stats[source][best_output_i][eval_target]['numerator']
                    n_relevant += source_to_F1_stats[source][best_output_i][eval_target]['n_relevant']
                    n_retrieved += source_to_F1_stats[source][best_output_i][eval_target]['n_retrieved']
                F1 = generic_F1(numerator=numerator, n_relevant=n_relevant, n_retrieved=n_retrieved)
                res[eval_target][f1_method] = F1

            elif f1_method == 'avg_precs_and_recalls':
                precision_L, recall_L = [], []
                for source in source_to_F1_stats:
                    best_output_i = source_to_best_F1_output[(source, eval_target)]
                    numerator = source_to_F1_stats[source][best_output_i][eval_target]['numerator']
                    n_relevant = source_to_F1_stats[source][best_output_i][eval_target]['n_relevant']
                    n_retrieved = source_to_F1_stats[source][best_output_i][eval_target]['n_retrieved']
                    precision = generic_precision(numerator=numerator, n_retrieved=n_retrieved)
                    recall = generic_recall(numerator=numerator, n_relevant=n_relevant)
                    if recall is not None:
                        precision_L.append(precision)
                        recall_L.append(recall)
                if precision_L:
                    precision = statistics.mean(precision_L)
                else:
                    precision = 0
                if recall_L:
                    recall = statistics.mean(recall_L)
                else:
                    recall = None
                if recall is None:
                    F1 = None
                else:
                    denom = recall + precision
                    if denom == 0:
                        F1 = 0
                    else:
                        F1 = 2 * precision * recall / denom
                res[eval_target][f1_method] = F1

    # write output
    base_save_path = os.path.join(cs.PATH_TO_MODELS, 'finetuned_epoch_{}_{}'.format(epoch_tuple[0], epoch_tuple[1]),
                                  'metrics_{}_{}_{}'.format(mode, mask_pattern_type, n_measures))
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)

    prepend_s = f'chunked={CHUNK_BY_INPUT_LEN}_'
    if NAIVE:
        prepend_s += f'NAIVE ({NAIVE})_'
    # if EXPLORE:
    #     prepend_s += f'EXPLORE={EXPLORE}_'
    if not NAIVE:
        if DELETE_MONO_POLY:
            prepend_s += 'NO_MONO_POLY_'
            if DELETE_MONO_POLY_BEFORE_CHUNKING:
                prepend_s += '(DEL_BEFORE_CHUNKING)_'
        if DO_SAMPLE:
            prepend_s += f'DO_SAMPLE={DO_SAMPLE}_'
        prepend_s += f'BATCH_SIZE={BATCH_SIZE}_'

    prepend_s += f'TRUNCATE_DATASET={TRUNCATE_DATASET}_'
    with open(os.path.join(base_save_path, prepend_s+'metrics.txt'), 'w') as outfile:
        json.dump(res, outfile)

    with open(os.path.join(base_save_path, prepend_s+'source_to_F1_stats.txt'), 'w') as outfile:
        json.dump(source_to_F1_stats, outfile)

    print('analysis done for epoch_tuple = {}, mode = {}, mask_pattern_type = {}, n_measures = {} done in {} sec'.format(
        epoch_tuple, mode, mask_pattern_type, n_measures, time.time() - t0))


def go():
    mode = 'val'
    for epoch_tuple in EPOCHS_TO_ANALYZE:
        for mask_pattern_type in ['1', '2last', '0half']:
            for n_measures in [8, 16, 32]:
                print('analyzing epoch_tuple = {}, mode = {}, mask_pattern_type = {}, n_measures = {}'.format(
                    epoch_tuple, mode, mask_pattern_type, n_measures))
                analyze(epoch_tuple=epoch_tuple, mode=mode, mask_pattern_type=mask_pattern_type, n_measures=n_measures)


if __name__ == '__main__':
    go()
