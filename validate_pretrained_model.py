import json
import statistics
import time
import constants as cs
import os
import transformers
import torch
import nn_training_functions as fn
import functools
import tokenizer_functions as tok

EPOCHS_TO_ANALYZE = range(9, 10)
# batch size for get_losses (short)
# batch size for get_losses (long) is scaled by a factor of 0.5
# batch size for get_accuracies is scaled up by a large factor since it uses generate()
BATCH_SIZE_SHORT = 2
GET_LOSS_SHORT = False
GET_LOSS_LONG = False
GET_ACCURACY_SHORT = True
GET_ACCURACY_LONG = False
TRUNCATE = 8000  # False or number of samples

TOKENIZER = tok.get_tokenizer()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(epoch):
    base_load_path = os.path.join(cs.PATH_TO_MODELS, 'pretrained_epoch_{}'.format(epoch))
    if not os.path.exists(base_load_path):
        print('No model found for epoch {}. Skipping.'.format(epoch))
        return None
    else:
        M = transformers.T5ForConditionalGeneration.from_pretrained(os.path.join(base_load_path, 'model'))
        M.eval()
        M = M.to(DEVICE)
        return M


def get_losses(model, dataset, short_or_long):
    collator = functools.partial(fn.batch_padder, tokenizer=TOKENIZER)
    if short_or_long == 'short':
        BATCH_SIZE = BATCH_SIZE_SHORT
    else:
        BATCH_SIZE = max(1, BATCH_SIZE_SHORT // 2)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=collator)
    losses = []
    for i, x in enumerate(dataloader):
        output = model(input_ids=x['input_ids'].to(DEVICE), labels=x['labels'].to(DEVICE))
        losses.append(output.loss.item())

        print((i+1) * BATCH_SIZE, 'examples done so far (losses)')

    return losses


def get_accuracies(model, dataset, batch_size_mult=1):
    local_bs = BATCH_SIZE_SHORT * batch_size_mult
    collator = functools.partial(fn.batch_padder, tokenizer=TOKENIZER)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=local_bs, collate_fn=collator)
    accuracies = []
    for i, x in enumerate(dataloader):
        gen_len = x['labels'].shape[1]  # start with the decoder_start_token, then generate enough outputs
        output_generated = model.generate(input_ids=x['input_ids'].to(DEVICE),
                                          min_length=gen_len,
                                          max_new_tokens=gen_len,
                                          decoder_start_token_id=TOKENIZER.pad_id(),
                                          pad_token_id=TOKENIZER.pad_id(),
                                          bos_token_id=TOKENIZER.bos_id(),
                                          eos_token_id=TOKENIZER.eos_id(),
                                          use_cache=True)
        output_generated = output_generated[:, 1:]  # drop the decoder_start_token at the beginning of each generation
        output_generated = output_generated.to('cpu')

        # print('o1', output_generated.shape)
        # print('o2', x['labels'].shape)

        # this works because generated tokens are always >= 0 and labels corresponding to pad tokens are -100
        acc = (output_generated == x['labels']).sum() / (x['labels'] >= 0).sum()
        accuracies.append(acc.item())

        print((i+1) * local_bs, 'examples done so far (accuracies)')

    return accuracies


def analyze(epoch):
    M = get_model(epoch)
    if M is None:
        return

    if GET_LOSS_SHORT or GET_ACCURACY_SHORT:
        val_dataset_short = fn.PreTrainDataset(epoch=0, mode='val_short')
        if TRUNCATE:
            val_dataset_short = torch.utils.data.Subset(val_dataset_short, torch.arange(0, TRUNCATE))

    if GET_LOSS_LONG or GET_ACCURACY_LONG:
        val_dataset_long = fn.PreTrainDataset(epoch=0, mode='val_long')
        if TRUNCATE:
            val_dataset_long = torch.utils.data.Subset(val_dataset_long, torch.arange(0, TRUNCATE))

    res = {}
    if GET_LOSS_SHORT:
        t0 = time.time()
        L_short = get_losses(model=M, dataset=val_dataset_short, short_or_long='short')
        print('Got short-sequence loss in {} sec'.format(time.time() - t0))
        res['loss_short_mean'] = statistics.mean(L_short)
        res['loss_short_std'] = statistics.stdev(L_short)
        res['losses_short'] = L_short

    if GET_LOSS_LONG:
        t0 = time.time()
        L_long = get_losses(model=M, dataset=val_dataset_long, short_or_long='long')
        print('Got long-sequence loss in {} sec'.format(time.time() - t0))
        res['loss_long_mean'] = statistics.mean(L_long)
        res['loss_long_std'] = statistics.stdev(L_long)
        res['losses_long'] = L_long

    if GET_ACCURACY_SHORT:
        t0 = time.time()
        A_short = get_accuracies(model=M, dataset=val_dataset_short, batch_size_mult=50)
        print('Got short-sequence accuracy in {} sec'.format(time.time() - t0))
        res['acc_short_mean'] = statistics.mean(A_short)
        res['acc_short_std'] = statistics.stdev(A_short)
        res['accuracies_short'] = A_short

    if GET_ACCURACY_LONG:
        t0 = time.time()
        A_long = get_accuracies(model=M, dataset=val_dataset_long, batch_size_mult=20)
        print('Got long-sequence accuracy in {} sec'.format(time.time() - t0))
        res['acc_long_mean'] = statistics.mean(A_long)
        res['acc_long_std'] = statistics.stdev(A_long)
        res['accuracies_long'] = A_long

    base_save_path = os.path.join(cs.PATH_TO_MODELS, 'pretrained_epoch_{}'.format(epoch), 'metrics')
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)

    with open(os.path.join(base_save_path, 'metrics.txt'), 'w') as outfile:
        json.dump(res, outfile)


def go():
    for epoch in EPOCHS_TO_ANALYZE:
        t0 = time.time()
        print('analyzing epoch {}'.format(epoch))
        analyze(epoch)
        print('epoch {} analyzed in {} sec'.format(epoch, time.time() - t0))


if __name__ == '__main__':
    go()
    print('all done')
