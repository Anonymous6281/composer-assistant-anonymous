import constants
import midisong as ms
import collections
import bisect
import statistics as stat

# this file contains functions for encoding midisongbymeasure objects to strings.
# to encode such strings to sequences of integers, use nn_training_functions.get_tokenizer()

# learned from train data; transfers well to val data
BPM_SLICER = [59.00000885, 74.9, 89.999955, 105.00157502, 119.9, 138.45653273, 165.000165]

# learned from train data; transfers well to val data, except lowest dynamic level is underrepresented in val data
DYNAMICS_SLICER = [64.4, 76.66666667, 81.9, 89.36666667, 95.9, 100.5, 109.9]

# based on DYNAMICS_SLICER
DYNAMICS_DEFAULTS = [58, 70, 79, 85, 93, 98, 105, 115]


def get_bpm_level(b: float) -> int:
    """b a bmp float value. Returns an integer in [0,...,7]"""
    return bisect.bisect_right(BPM_SLICER, b)


def get_loudness_level(x: float) -> int:
    """x an average velocity level for a measure. Returns an integer in [0,...,7]"""
    return bisect.bisect_right(DYNAMICS_SLICER, x)


def _avg_vel_of_tr(tr):
    res = [n.vel for n in tr.note_ons]
    if res:
        return stat.mean(res)
    else:
        return None


def _avg_vel_of_measure(m):
    """empty measures are considered to have 0.0 avg vel"""
    res = [_avg_vel_of_tr(tr) for tr in m]
    res = [x for x in res if x is not None]
    if res:
        return stat.mean(res)
    else:
        return 0.0


def transpose_into_acceptable_ranges_TT(S: "ms.MidiSongByMeasure"):
    """in place operation. TT stands for train/test, because that's where we use it in this project."""
    for t in S.tracks:
        t.transpose_by_octaves_into_range(range_=constants.ACCEPTABLE_NOTE_RANGE_BY_INST_TRAIN_TEST[t.inst],
                                          cleanup_note_duplicates=True)


def encode_midisongbymeasure_with_masks(S: "ms.MidiSongByMeasure",
                                        note_off_treatment: str = 'duration',
                                        mask_locations: None or "set[tuple[int, int]]" = None,
                                        measure_slice: None or "tuple[int, int]" = None,
                                        include_heads_for_empty_masked_measures: bool = False,
                                        poly_mono_commands: None or "dict[tuple[int, int], str]" = None,
                                        return_labels_too: bool = True,
                                        extra_id_st: int = 0,
                                        extra_id_max: int = 255,
                                        velocity_overrides: None or "dict[int, int]" = None,
                                        ) -> "tuple[str, str]":
    """
    mask_locations a set of tuples of the form (track_index, measure_index).
    Note that measure indexes are relative to the start of the song, even if measure_slice targets an area of the song
    following the start.

    measure_slice a tuple of the form (start_measure, end_measure).
    Like range() and slicing operations, start_measure is included in the result and end_measure is not.
    So, to get the string encoding measures 14 and 15, for instance, you would set measure_slice = (14, 16).

    Set include_heads_for_empty_masked_measures to False for fine-tuning, validation, and testing; set it to
    True for inference (e.g., in Reaper). Note that empty unmasked measures NEVER include heads.

    poly_mono_commands a (default)dict with keys of the form (track_index, measure_index) and corresponding values of
    either ';<poly>' or ';<mono>' or ''. ;<poly> and ;<mono> commands will only be inserted for masked
    (track_index, measure_index) pairs.

    velocity_overrides a dict, *not* a defaultdict, of the form measure_i: avg_vel_of_measure_i.
    This is useful for inference in Reaper when there are no labels and a whole measure is masked.
    Measure indexes not in velocity_overrides use the computed avg velocity from S.

    If return_labels is True, the output will be a tuple of the form (input_ids, labels); otherwise, the output
    will be a tuple of the form (input_ids, ''). Use True for fine-tuning, validation, and testing. Use False for
    inference.
    """
    if mask_locations is None:
        mask_locations = set()

    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())

    if poly_mono_commands is None:
        poly_mono_commands = collections.defaultdict(str)

    if velocity_overrides is None:
        velocity_overrides = {}

    measure_lens = S.get_measure_lengths()
    tempos = S.get_tempo_at_start_of_each_measure()

    measure_st_strings = collections.defaultdict(str)

    for measure_i in range(*measure_slice):
        if measure_i in velocity_overrides:
            avg_vel = velocity_overrides[measure_i]
        else:
            avg_vel = _avg_vel_of_measure(S.get_measure(measure_idx=measure_i))
        s = ';M:{}'.format(get_loudness_level(avg_vel))  # M: how loud (scale of 0 to 7)
        s += ';B:{}'.format(get_bpm_level(tempos[measure_i]))  # B: how fast (tempo; scale of 0 to 7)
        s += ';L:{}'.format(measure_lens[measure_i])  # L: how long (number of clicks)
        measure_st_strings[measure_i] = s

    input_ids, labels = '', ''
    extra_id = extra_id_st
    heads, tails = get_string_encoding_matrices(S, note_off_treatment=note_off_treatment, measure_slice=measure_slice)
    for measure_i in range(*measure_slice):
        input_ids += measure_st_strings[measure_i]
        for tr_i in range(len(S.tracks)):
            cur_T = (tr_i, measure_i)
            head = heads[cur_T]
            tail = tails[cur_T]

            # if we are masking this tr + measure:
            if cur_T in mask_locations:
                if tail or include_heads_for_empty_masked_measures:
                    extra_id_str = ';<extra_id_{}>'.format(extra_id)
                    if extra_id > extra_id_max:
                        raise ValueError('extra_id_max not large enough or extra_id_st too large')
                    extra_id += 1

                    labels += extra_id_str + tail

                    input_ids += head
                    input_ids += poly_mono_commands[cur_T]
                    # insert other commands here if applicable
                    input_ids += extra_id_str

            else:
                # if this tr + measure is not masked, only include it if there is a non-'' tail
                if tail:
                    input_ids += head + tail

    if return_labels_too:
        return input_ids, labels
    else:
        return input_ids, ''


def encode_midisongbymeasure(S: "ms.MidiSongByMeasure", note_off_treatment='duration') -> str:
    """Mainly for testing and pretraining. Template for finetuning."""
    measure_lens = S.get_measure_lengths()
    tempos = S.get_tempo_at_start_of_each_measure()

    measure_st_strings = []
    for measure_i in range(S.get_n_measures()):
        avg_vel = _avg_vel_of_measure(S.get_measure(measure_idx=measure_i))
        s = ';M:{}'.format(get_loudness_level(avg_vel))  # M: how loud (scale of 0 to 7)
        s += ';B:{}'.format(get_bpm_level(tempos[measure_i]))  # B: how fast (tempo; scale of 0 to 7)
        s += ';L:{}'.format(measure_lens[measure_i])  # L: how long (number of clicks)
        measure_st_strings.append(s)

    heads, tails = get_string_encoding_matrices(S, note_off_treatment=note_off_treatment)

    # put it all together
    res = ''
    for measure_i in range(S.get_n_measures()):
        res += measure_st_strings[measure_i]
        for tr_i in range(len(S.tracks)):
            head = heads[(tr_i, measure_i)]
            tail = tails[(tr_i, measure_i)]
            # only include this measure and track in the string if there is a non-'' tail
            if tail:
                res += head + tail
    return res


# test written via test_encode_midisongbymeasure_no_note_offs
def _get_string_encoding_matrices_no_note_offs(
        S: "ms.MidiSongByMeasure",
        measure_slice: None or "tuple[int, int]" = None,
        ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":
    """we assume that note on's and note off's are already sorted by click and pitch"""
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;N:Y;w:Z... etc
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    low, high = measure_slice
    for tr_i, tr in enumerate(S.tracks):
        for measure_i, measure_t in enumerate(tr.tracks_by_measure):
            if low <= measure_i < high:
                # compute tail for this track and measure
                s_tail = ''
                cur_click = 0
                for n in measure_t.note_ons:
                    if n.click != cur_click:
                        s_tail += ';w:{}'.format(n.click - cur_click)
                        cur_click = n.click
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        s_tail += ';N:{}'.format(n.pitch)

                res_array_tails[(tr_i, measure_i)] = s_tail

                # compute head for this track and measure
                if instrument_repetition_counter[tr.inst]:
                    inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
                else:
                    inst_rep_str = ''

                s_head = ';I:{}'.format(tr.inst) + inst_rep_str

                res_array_heads[(tr_i, measure_i)] = s_head

        # note that instrument repetition counters are song-wide, regardless of the measure slice.
        instrument_repetition_counter[tr.inst] += 1

    return res_array_heads, res_array_tails


# test written via test_encode_midisongbymeasure_including_note_duration_commands
def _get_string_encoding_matrices_with_note_duration_commands(
        S: "ms.MidiSongByMeasure",
        measure_slice: None or "tuple[int, int]" = None,
        ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;d:X;N:Y;w:Z... etc (d: means "duration for new notes in this measure")
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    low, high = measure_slice

    MLs = S.get_measure_lengths()
    max_note_length = 8 * ms.extended_lcm(constants.QUANTIZE)  # 8 QN's max length
    for tr_i, tr in enumerate(S.tracks):

        noteidx_tracker = tr.get_noteidx_info_dict(measure_lengths=MLs)

        for measure_i, measure_t in enumerate(tr.tracks_by_measure):
            if low <= measure_i < high:
                # compute tail for this track and measure
                s_tail = ''
                cur_click = 0
                cur_length = -1
                # cur_vel = -1
                for n in measure_t.note_ons:
                    if n.click != cur_click:
                        s_tail += ';w:{}'.format(n.click - cur_click)
                        cur_click = n.click
                    # if I wanted to add velocity commands, here's where I'd do it
                    # if vel_bucket(n.vel) != cur_vel:
                    #     s_tail += ';v:{}'.format(vel_bucket(n.vel))
                    #     cur_vel = vel_bucket(n.vel)
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        this_note_length = noteidx_tracker[n.noteidx].length
                        if this_note_length is None:
                            this_note_length = 0
                        length = min(this_note_length, max_note_length)
                        if length != cur_length:
                            s_tail += ';d:{}'.format(length)  # always put "duration" commands right before note on's
                            cur_length = length  # and only add duration commands for non-drum notes
                        s_tail += ';N:{}'.format(n.pitch)

                res_array_tails[(tr_i, measure_i)] = s_tail

                # compute head for this track and measure
                if instrument_repetition_counter[tr.inst]:
                    inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
                else:
                    inst_rep_str = ''

                s_head = ';I:{}'.format(tr.inst) + inst_rep_str

                res_array_heads[(tr_i, measure_i)] = s_head

        instrument_repetition_counter[tr.inst] += 1

    return res_array_heads, res_array_tails


# test written via test_encode_midisongbymeasure_including_note_lengths
def _get_string_encoding_matrices_with_note_lengths(
        S: "ms.MidiSongByMeasure",
        measure_slice: None or "tuple[int, int]" = None,
        ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;N:X:Y;w:Z... etc  # N:X:Y means note pitch X, duration Y.
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    if measure_slice is None:
        measure_slice = (0, S.get_n_measures())
    low, high = measure_slice

    MLs = S.get_measure_lengths()
    max_note_length = 8 * ms.extended_lcm(constants.QUANTIZE)  # 8 QN's max length
    for tr_i, tr in enumerate(S.tracks):

        noteidx_tracker = tr.get_noteidx_info_dict(measure_lengths=MLs)

        for measure_i, measure_t in enumerate(tr.tracks_by_measure):
            if low <= measure_i < high:
                # compute tail for this track and measure
                s_tail = ''
                cur_click = 0
                for n in measure_t.note_ons:
                    if n.click != cur_click:
                        s_tail += ';w:{}'.format(n.click - cur_click)
                        cur_click = n.click
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        this_note_length = noteidx_tracker[n.noteidx].length
                        if this_note_length is None:
                            this_note_length = 0
                        length = min(this_note_length, max_note_length)
                        s_tail += ';N:{}:{}'.format(n.pitch, length)

                res_array_tails[(tr_i, measure_i)] = s_tail

                # compute head for this track and measure
                if instrument_repetition_counter[tr.inst]:
                    inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
                else:
                    inst_rep_str = ''

                s_head = ';I:{}'.format(tr.inst) + inst_rep_str

                res_array_heads[(tr_i, measure_i)] = s_head

        instrument_repetition_counter[tr.inst] += 1

    return res_array_heads, res_array_tails


# test written via test_encode_midisongbymeasure_including_note_offs
# uses ties and explicit note off messages
def _get_string_encoding_matrices_including_note_offs(
        S: "ms.MidiSongByMeasure",
        measure_slice: None or "tuple[int, int]" = None,
        ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":
    """we assume that note on's and note off's are already sorted by click and pitch"""
    # keys of the form (tr_i, measure_i)
    # values of the form ;I:X or ;I:X;R:X
    # There is always a non-'' head, regardless of whether there is a '' tail
    res_array_heads = collections.defaultdict(str)

    # keys of the form (tr_i, measure_i)
    # values of the form ;T:X;N:Y;w:Z;/N:Q... etc
    res_array_tails = collections.defaultdict(str)

    # build up our result one track at a time
    instrument_repetition_counter = collections.Counter()
    for tr_i, tr in enumerate(S.tracks):

        noteidx_tracker = tr.get_noteidx_info_dict()

        # compute where we will have ties for this track
        if not tr.is_drum:
            ties_by_measure = collections.defaultdict(set)
            for idx, info in noteidx_tracker.items():
                if info.measure_note_on < info.measure_note_off:
                    if info.measure_note_off > info.measure_note_on + 1 or info.note_off.click > 0:
                        # then create ties
                        if info.note_off.click == 0:
                            upper = info.measure_note_off
                        else:
                            upper = info.measure_note_off + 1
                        for measure_i in range(info.measure_note_on + 1, upper):
                            ties_by_measure[measure_i].add(info.note_on.pitch)
            # sort ties by pitch
            d = collections.defaultdict(list)
            for measure_i, L in ties_by_measure.items():
                d[measure_i] = sorted(list(L))
            ties_by_measure = d
        else:
            ties_by_measure = collections.defaultdict(list)

        # next, build up our result one measure at a time for this track
        for measure_i, measure_t in enumerate(tr.tracks_by_measure):

            # separate note ons by click and sort by pitch at each click
            note_ons_by_click = collections.defaultdict(list)
            for n in measure_t.note_ons:
                note_ons_by_click[n.click].append(n)

            # separate note offs into "before" and "after" at each click
            note_offs_before_by_click = collections.defaultdict(list)
            note_offs_after_by_click = collections.defaultdict(list)
            if not tr.is_drum:
                for n in measure_t.note_offs:
                    if noteidx_tracker[n.noteidx].measure_note_on == measure_i and noteidx_tracker[n.noteidx].note_on.click == n.click:
                        note_offs_after_by_click[n.click].append(n)
                    else:
                        if n.click != 0:
                            note_offs_before_by_click[n.click].append(n)

            # get our list of clicks for this measure and track
            clicks = set(note_ons_by_click.keys())
            if not tr.is_drum:
                clicks = clicks.union(set(note_offs_before_by_click.keys()))
                clicks = clicks.union(set(note_offs_after_by_click.keys()))
            clicks = sorted(list(clicks))

            # create string for this track and measure
            s_tail = ''
            # handle ties first
            for tie in ties_by_measure[measure_i]:
                s_tail += ';T:{}'.format(tie)
            # build the rest of the string click by click
            prev_click = 0
            for click in clicks:
                if click != prev_click:
                    s_tail += ';w:{}'.format(click - prev_click)
                    prev_click = click
                for n in note_offs_before_by_click[click]:
                    s_tail += ';/N:{}'.format(n.pitch)
                for n in note_ons_by_click[click]:
                    if tr.is_drum:
                        s_tail += ';D:{}'.format(n.pitch)
                    else:
                        s_tail += ';N:{}'.format(n.pitch)
                for n in note_offs_after_by_click[click]:
                    s_tail += ';/N:{}'.format(n.pitch)

            res_array_tails[(tr_i, measure_i)] = s_tail

            if instrument_repetition_counter[tr.inst]:
                inst_rep_str = ';R:{}'.format(instrument_repetition_counter[tr.inst])
            else:
                inst_rep_str = ''

            s_head = ';I:{}'.format(tr.inst) + inst_rep_str

            res_array_heads[(tr_i, measure_i)] = s_head

        instrument_repetition_counter[tr.inst] += 1

    # This function is fairly inefficient: Since we need to compute ties, we need to encode the whole song before
    # restricting to the input measure_slice.
    def do_del(heads_or_tails):
        low, high = measure_slice
        to_del = set()
        for k, v in heads_or_tails.items():
            if k[1] < low or k[1] >= high:
                to_del.add(k)
        for k in to_del:
            del heads_or_tails[k]

    if measure_slice is not None:
        do_del(res_array_heads)
        do_del(res_array_tails)

    return res_array_heads, res_array_tails


def get_string_encoding_matrices(S: "ms.MidiSongByMeasure",
                                 note_off_treatment: str = 'duration',
                                 measure_slice: None or "tuple[int, int]" = None,
                                 ) -> "tuple[collections.defaultdict[tuple[int, int], str], collections.defaultdict[tuple[int, int], str]]":

    for tr in S.tracks:
        for tm in tr.tracks_by_measure:
            if tm.notes:
                raise ValueError('cannot encode MidiSongByMeasure to string if any of its tracks T has a ByMeasureTrack'
                                 'in its .tracks_by_measure with .notes. Use only .note_ons and .note_offs instead.')
            for x in tm.note_ons:
                if not hasattr(x, "noteidx") or x.noteidx is None:
                    raise ValueError('cannot encode MidiSongByMeasure to string because some note_on lacks a .noteidx')
            for x in tm.note_offs:
                if not hasattr(x, "noteidx") or x.noteidx is None:
                    raise ValueError('cannot encode MidiSongByMeasure to string because some note_off lacks a .noteidx')

    if note_off_treatment == 'include':
        return _get_string_encoding_matrices_including_note_offs(S, measure_slice=measure_slice)
    elif note_off_treatment == 'exclude':
        return _get_string_encoding_matrices_no_note_offs(S, measure_slice=measure_slice)
    elif note_off_treatment == 'length':
        return _get_string_encoding_matrices_with_note_lengths(S, measure_slice=measure_slice)
    elif note_off_treatment == 'duration':
        return _get_string_encoding_matrices_with_note_duration_commands(S, measure_slice=measure_slice)
    else:
        raise ValueError("note_off_treatment must be one of 'include', 'exclude', 'length', or 'duration'.")
