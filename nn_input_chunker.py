import bisect


# test written
def min_and_max_extra_id_in(s='') -> "tuple[int, int]":
    index = -1
    m = -1
    sm = 999999999999999999999
    start = True
    found = False
    while index != -1 or start:
        index = s.find('<extra_id_', index + 1)
        index_end = s.find('>', index)
        if index != -1:
            cur = int(s[index + 10: index_end])
            found = True
            m = max(m, cur)
            sm = min(sm, cur)
        start = False
    if found:
        return sm, m
    else:
        raise ValueError("no extra_id's in s")


# no tests written. function not used elsewhere
def _get_tokenized_measure_lens_and_n_extra_ids_by_measure(input_ids, tokenizer):
    def replace_extra_ids(s):
        first_loc = s.find('<extra_id_')
        if first_loc != -1:
            second_loc = s.find('>', first_loc)
            s = s[:first_loc] + '<e>' + s[second_loc + 1:]
            return replace_extra_ids(s)
        else:
            return s.replace('<e>', '<extra_id_0>')

    input_ids_temp = replace_extra_ids(input_ids)
    encoded_iids_temp = tokenizer.encode(input_ids_temp)
    measure_st_tokens = set(tokenizer.encode([';M:{}'.format(x) for x in range(8)]))
    eid_token = tokenizer.encode(';<extra_id_0>')[0]
    measure_lens = []
    eids = []
    this_measure_len = 0
    this_measure_eids = 0
    for token in encoded_iids_temp:
        if token in measure_st_tokens:
            if this_measure_len > 0:
                measure_lens.append(this_measure_len)
                eids.append(this_measure_eids)
                this_measure_len = 0
                this_measure_eids = 0
        else:
            this_measure_len += 1

        if token == eid_token:
            this_measure_eids += 1

    return measure_lens, eids


# test written
def _M_locations_in(s=""):
    res = []
    last_loc = -1
    start = True
    while last_loc != -1 or start:
        last_loc = s.find(';M', last_loc + 1)
        if last_loc != -1:
            res.append(last_loc)
        start = False
    return res


# test written
def _cil_grab_greedy(tokenizer, input_ids='', labels='', max_input_len=1024, from_left=True):
    M_locs = _M_locations_in(input_ids)
    if not M_locs:
        return input_ids, labels

    M_locs.append(len(input_ids))
    n_M_locs = len(M_locs)

    n_measures_to_grab = 1

    done = False
    while not done:
        if n_measures_to_grab >= len(M_locs):
            n_measures_to_grab = len(M_locs) - 1
            done = True
        else:
            if from_left:
                grabbed = input_ids[:M_locs[n_measures_to_grab]]
            else:
                grabbed = input_ids[M_locs[n_M_locs - 1 - n_measures_to_grab]:]
            if len(tokenizer.encode(grabbed)) > max_input_len:
                n_measures_to_grab -= 1
                done = True
            else:
                n_measures_to_grab += 1

    if n_measures_to_grab == 0:
        n_measures_to_grab = 1

    if from_left:
        grabbed = input_ids[:M_locs[n_measures_to_grab]]
    else:
        grabbed = input_ids[M_locs[n_M_locs - 1 - n_measures_to_grab]:]
    labels_grabbed = _grab_labels(grabbed, labels)

    return grabbed, labels_grabbed


# test written
def _cil_grab_approx_half(tokenizer, input_ids='', labels='', max_input_len=1024):
    M_locs = _M_locations_in(input_ids)
    if not M_locs:
        return input_ids, labels

    M_locs.append(len(input_ids))

    lens = []
    for n_measures_grabbed in range(1, len(M_locs)):
        grabbed = input_ids[:M_locs[n_measures_grabbed]]
        lens.append(len(tokenizer.encode(grabbed)))

    if lens[0] > max_input_len:
        n_measures_to_grab = 1  # gotta grab at least one measure
    else:
        scores = []
        total_len = len(tokenizer.encode(input_ids))
        for x in lens:
            score = abs(2 * x - total_len) if x <= max_input_len else 999999999999999999
            scores.append(score)  # lower score is better
        min_score = min(scores)
        min_score_i = scores.index(min_score)
        n_measures_to_grab = min_score_i + 1

    grabbed = input_ids[:M_locs[n_measures_to_grab]]
    labels_grabbed = _grab_labels(grabbed, labels)

    return grabbed, labels_grabbed


# tested via testing _cil functions
def _grab_labels(grabbed, labels):
    try:
        min_eid, max_eid = min_and_max_extra_id_in(grabbed)
        continue_ = True
    except ValueError:
        min_eid, max_eid = -1, -1
        continue_ = False
    if continue_:
        min_pos = labels.find(';<extra_id_{}>'.format(min_eid))
        max_pos = labels.find(';<extra_id_{}>'.format(max_eid + 1))
        if max_pos > -1:
            labels_grabbed = labels[min_pos: max_pos]
        else:
            labels_grabbed = labels[min_pos:]
    else:
        labels_grabbed = ''
    return labels_grabbed


# test written
def _ceid_grab_greedy(input_ids='', labels='', max_extra_id=255):
    """extra_ids in input_ids and labels must start at 0."""
    M_locs = _M_locations_in(input_ids)

    # If no measures, just return the input
    if not M_locs:
        return input_ids, labels

    first_bad_eid_loc = input_ids.find('<extra_id_{}>'.format(max_extra_id + 1))
    if first_bad_eid_loc == -1:
        return input_ids, labels
    else:
        first_bad_measure = bisect.bisect_right(M_locs, first_bad_eid_loc) - 1
        if first_bad_measure <= 0:
            first_bad_measure = 1  # always grab at least one measure
        grabbed = input_ids[:M_locs[first_bad_measure]]
        labels_grabbed = _grab_labels(grabbed, labels)
        return grabbed, labels_grabbed


# test written
def _ceid_grab_approx_half(input_ids='', labels='', max_extra_id=255):
    """extra_ids in input_ids and labels must start at 0. Tries to grab approximately half of the input ids in
    input_ids. max_extra_id is only used to """
    M_locs = _M_locations_in(input_ids)
    if not M_locs:
        return input_ids, labels

    M_locs.append(len(input_ids))

    n_eids_in = []
    for n_measures_grabbed in range(1, len(M_locs)):
        grabbed = input_ids[:M_locs[n_measures_grabbed]]
        try:
            a, b = min_and_max_extra_id_in(grabbed)
            count = b - a + 1
        except ValueError:
            count = 0
        n_eids_in.append(count)

    n_eids = n_eids_in[-1]
    if n_eids_in[0] > max_extra_id + 1:
        n_measures_to_grab = 1  # gotta grab at least one measure
    else:
        scores = []
        for x in n_eids_in:
            score = abs(n_eids - 2 * x) if x <= max_extra_id + 1 else 99999999999999999999
            scores.append(score)  # lower score is better
        min_score = min(scores)
        min_score_i = scores.index(min_score)
        n_measures_to_grab = min_score_i + 1

    grabbed = input_ids[:M_locs[n_measures_to_grab]]
    labels_grabbed = _grab_labels(grabbed, labels)

    return grabbed, labels_grabbed


# test written
def _replace_eids(input_ids='', labels=''):
    try:
        m_i, _ = min_and_max_extra_id_in(input_ids)
    except ValueError:
        m_i = None
    try:
        m_l, _ = min_and_max_extra_id_in(labels)
    except ValueError:
        m_l = None

    if m_i is not None and m_l is not None:
        m = min(m_i, m_l)
    elif m_i is None and m_l is not None:
        m = m_l
    elif m_i is not None and m_l is None:
        m = m_i
    else:
        m = 0

    try:
        m_0, m_1 = min_and_max_extra_id_in(input_ids)
    except ValueError:
        m_0, m_1 = 0, -1
    for k in range(m_0, m_1 + 1):
        input_ids = input_ids.replace('<extra_id_{}>'.format(k), '<extra_id_{}>'.format(k - m))

    try:
        m_0, m_1 = min_and_max_extra_id_in(labels)
    except ValueError:
        m_0, m_1 = 0, -1
    for k in range(m_0, m_1 + 1):
        labels = labels.replace('<extra_id_{}>'.format(k), '<extra_id_{}>'.format(k - m))

    return input_ids, labels


# tested via test for chunk_by_input_len
def _chunk_by_input_len_recursive(done_so_far: "list", tokenizer, input_ids='', labels='', max_input_len=1024):
    int_iids = tokenizer.encode(input_ids)

    # base case
    if len(int_iids) <= max_input_len:
        grabbed_iids, grabbed_labels = _replace_eids(input_ids=input_ids, labels=labels)
        if grabbed_iids or grabbed_labels:
            done_so_far.append((grabbed_iids, grabbed_labels))
        return done_so_far

    # recursive step
    else:
        if len(int_iids) > 2 * max_input_len:
            grabbed_iids, grabbed_labels = _cil_grab_greedy(tokenizer=tokenizer,
                                                            input_ids=input_ids,
                                                            labels=labels,
                                                            max_input_len=max_input_len)
        else:
            grabbed_iids, grabbed_labels = _cil_grab_approx_half(tokenizer=tokenizer,
                                                                 input_ids=input_ids,
                                                                 labels=labels,
                                                                 max_input_len=max_input_len)
        next_input_ids = input_ids[len(grabbed_iids):]
        next_labels = labels[len(grabbed_labels):]
        grabbed_iids, grabbed_labels = _replace_eids(input_ids=grabbed_iids, labels=grabbed_labels)
        next_input_ids, next_labels = _replace_eids(input_ids=next_input_ids, labels=next_labels)
        done_so_far.append((grabbed_iids, grabbed_labels))
        return _chunk_by_input_len_recursive(done_so_far=done_so_far, tokenizer=tokenizer,
                                             input_ids=next_input_ids, labels=next_labels, max_input_len=max_input_len)


# test written
def chunk_by_input_len(tokenizer, input_ids='', labels='', max_input_len=1024):
    return _chunk_by_input_len_recursive(done_so_far=[], tokenizer=tokenizer, input_ids=input_ids,
                                         labels=labels, max_input_len=max_input_len)


def get_rightmost_chunk(tokenizer, input_ids='', labels='', max_input_len=1024):
    res = _cil_grab_greedy(tokenizer=tokenizer, input_ids=input_ids, labels=labels,
                           max_input_len=max_input_len, from_left=False)
    return [res]  # list to be consistent with other chunker outputs


def _chunk_by_max_extra_id_recursive(done_so_far: "list", input_ids='', labels='', max_extra_id=255):
    # base case
    try:
        min_eid, max_eid = min_and_max_extra_id_in(input_ids)
        if max_eid <= max_extra_id:
            done = True
        else:
            done = False
    except ValueError:
        done = True
    if done:
        grabbed_iids, grabbed_labels = _replace_eids(input_ids, labels)
        if grabbed_iids or grabbed_labels:
            done_so_far.append((grabbed_iids, grabbed_labels))
        return done_so_far

    # recursive step
    else:  # if we get here, we know max_eid is defined, and max_eid > max_extra_id.
        if max_eid > 2 * max_extra_id:
            grabbed_iids, grabbed_labels = _ceid_grab_greedy(input_ids=input_ids, labels=labels,
                                                             max_extra_id=max_extra_id)
        else:
            grabbed_iids, grabbed_labels = _ceid_grab_approx_half(input_ids=input_ids, labels=labels,
                                                                  max_extra_id=max_extra_id)
        next_input_ids = input_ids[len(grabbed_iids):]
        next_labels = labels[len(grabbed_labels):]
        grabbed_iids, grabbed_labels = _replace_eids(input_ids=grabbed_iids, labels=grabbed_labels)
        next_input_ids, next_labels = _replace_eids(input_ids=next_input_ids, labels=next_labels)
        done_so_far.append((grabbed_iids, grabbed_labels))
        return _chunk_by_max_extra_id_recursive(done_so_far=done_so_far, input_ids=next_input_ids,
                                                labels=next_labels, max_extra_id=max_extra_id)


def chunk_by_max_extra_id(input_ids='', labels='', max_extra_id=255):
    return _chunk_by_max_extra_id_recursive(done_so_far=[], input_ids=input_ids,
                                            labels=labels, max_extra_id=max_extra_id)
