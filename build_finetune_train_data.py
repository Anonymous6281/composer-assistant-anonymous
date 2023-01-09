import nn_training_functions as fn
from multiprocessing import Pool
import os
import constants as cs
import tokenizer_functions as tok

# Instructions: Edit range(x, y) to set the epochs for which you're building pretrain data. Then run this file.
# If you are just starting the finetuning process, begin with epoch 0.
EPOCHS = range(0, 1)

TOKENIZER = tok.get_tokenizer()

if __name__ == '__main__':
    P = Pool()

    if not os.path.exists(cs.PATH_TO_TEMP_FILES):
        os.mkdir(cs.PATH_TO_TEMP_FILES)

    path = cs.PATH_TO_PROCESSED_TRAIN_MIDI
    for epoch in EPOCHS:
        fn.build_finetune_train_data(tokenizer=TOKENIZER, epoch=epoch, pool=P, path=path)

    print('All done')
