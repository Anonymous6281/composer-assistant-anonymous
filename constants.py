########################################################################################################################
# REAPER USER SETTINGS:

# To decide what instrument a given Reaper track is, the program takes the first of these that matches any part of
# the track name.
# Example: If the track name is 'grand pianos', that corresponds to instrument 0, since 'piano' appears in the name.
# Example: If the track name is 'harpsichord ep1', that corresponds to instrument 4, since 'ep1' is the first matching
#          instrument in the name.
# You may edit this to reflect your preferred track naming conventions in Reaper.
INST_TO_MATCHING_STRINGS_RPR = {0: ['piano', 'key'],  # ac grand piano
                                1: ['bright'],  # bright ac piano
                                2: [],  # electric grand piano
                                3: ['honk'],  # honky-tonk piano
                                4: ['ep1'],  # electric piano 1
                                5: ['ep2'],  # electric piano 2
                                6: ['harpsichord'],  # harpsichord
                                7: ['clavi'],  # clavi
                                8: ['celesta'],  # celesta
                                9: ['glock'],  # glock
                                10: ['box'],  # music box
                                11: ['vib'],  # vibraphone
                                12: ['mar'],  # marimba
                                13: ['xyl'],  # xylophone
                                14: ['bell', 'tubu'],  # tubular bells
                                15: ['dulc'],  # dulcimer
                                16: ['org'],  # drawbar organ
                                17: [],  # percussive organ
                                18: [],  # rock organ
                                19: [],  # church organ
                                20: [],  # reed organ
                                21: ['acc'],  # accordian
                                22: ['harmonica'],  # harmonica
                                23: [],  # tango accordian
                                24: ['ngtr'],  # ac guitar (nylon)
                                25: ['sgtr'],  # ac guitar (steel)
                                26: ['jgtr'],  # elec guitar (jazz)
                                27: ['cgtr'],  # elec guitar (clean)
                                28: ['mgtr'],  # elec guitar (muted)
                                29: ['ogtr'],  # overdriven guitar
                                30: ['gtr', 'guit'],  # distortion guitar
                                31: ['harmon'],  # guitar harmonics ("harmon" doesn't conflict with "harmonica" above)
                                32: [],  # ac bass
                                33: ['finger'],  # electric bass (finger)
                                34: ['bass'],  # electric bass (pick)
                                35: ['fretless'],  # fretless bass
                                36: ['slap'],  # slap bass 1
                                37: [],  # slap bass 2
                                38: [],  # synth bass 1
                                39: [],  # synth bass 2
                                40: ['violin', 'vn1', 'vn2'],  # violin
                                41: ['viola', 'vn3'],  # viola
                                42: ['cell', 'vn4'],  # cello
                                43: ['contrabass', 'vn5'],  # contrabass
                                44: ['trem'],  # tremolo strings
                                45: ['pizz'],  # pizz strings
                                46: ['harp'],  # orchestral harp
                                47: ['timpani'],  # timpani
                                48: ['str'],  # string ensemble 1
                                49: [],  # string ensemble 2
                                50: [],  # synth strings 1
                                51: [],  # synth strings 2
                                52: ['choir', 'aah'],  # choir aahs
                                53: ['voice', 'ooh'],  # voice oohs
                                54: ['svoice'],  # synth voice
                                55: ['hit'],  # orchestra hit
                                56: ['trumpet', 'tp'],  # trumpet
                                57: ['trombone'],  # trombone
                                58: ['tuba'],  # tuba
                                59: [],  # muted trumpet
                                60: ['french horn', 'fh'],  # french horn
                                61: ['brass'],  # brass section
                                62: [],  # synth brass 1
                                63: [],  # synth brass 2
                                64: ['sax'],  # soprano sax
                                65: [],  # alto sax
                                66: [],  # tenor sax
                                67: [],  # baritone sax
                                68: ['oboe'],  # oboe
                                69: ['english horn'],  # english horn
                                70: [],  # bassoon
                                71: ['clarinet'],  # clarinet
                                72: ['pic'],  # piccolo
                                73: ['flute', 'ft'],  # flute
                                74: ['recorder'],  # recorder
                                75: [],  # pan flute
                                76: ['bottle'],  # blown bottle
                                77: [],  # shakuhachi
                                78: ['whistle'],  # whistle
                                79: ['oca'],  # ocarina
                                80: ['square'],  # lead 1 (square)
                                81: ['saw'],  # lead 2 (sawtooth)
                                82: [],  # lead 3 (calliope)
                                83: [],  # lead 4 (chiff)
                                84: [],  # lead 5 (charang)
                                85: [],  # lead 6 (voice)
                                86: [],  # lead 7 (fifths)
                                87: [],  # lead 8 (bass + lead)
                                88: ['pad1'],  # pad 1 (new age)
                                89: ['pad2'],  # pad 2 (warm)
                                90: ['pad3'],  # pad 3 (polysynth)
                                91: ['pad4'],  # pad 4 (choir)
                                92: ['pad5'],  # pad 5 (bowed)
                                93: ['pad6'],  # pad 6 (metallic)
                                94: ['pad7'],  # pad 7 (halo)
                                95: ['pad8'],  # pad 8 (sweep)
                                96: ['fx1'],  # FX 1 (rain)
                                97: ['fx2'],  # FX 2 (soundtrack)
                                98: ['fx3'],  # FX 3 (crystal)
                                99: ['fx4'],  # FX 4 (atmosphere)
                                100: ['fx5'],  # FX 5 (brightness)
                                101: ['fx6'],  # FX 6 (goblins)
                                102: ['fx7'],  # FX 7 (echoes)
                                103: ['fx8'],  # FX 8 (sci-fi)
                                104: ['sitar'],  # sitar
                                105: ['banjo'],  # banjo
                                106: [],  # shamisen
                                107: ['koto'],  # koto
                                108: [],  # kalimba
                                109: ['bag'],  # bag pipe
                                110: ['fid'],  # fiddle
                                111: [],  # shanai
                                112: ['tink'],  # tinkle bell
                                113: [],  # agogo
                                114: ['steel'],  # steel drums
                                115: ['wood'],  # woodblock
                                116: ['taiko'],  # taiko
                                117: ['mtom'],  # melodic tom
                                118: ['sdrum'],  # synth drum
                                119: ['rev'],  # reverse cymbal
                                120: ['fret'],  # guitar fret noise
                                121: ['breath'],  # breath noise
                                122: [],  # seashore
                                123: ['bird', 'tweet'],  # bird tweet
                                124: ['phone'],  # telephone ring
                                125: ['heli'],  # helicopter
                                126: [],  # applause
                                127: [],  # gunshot
                                128: ['drum', 'kick', 'kik', 'sn', 'tom', 'hat', 'crash']
                                # GM drums on channel "10" (channel 9 in 0-based indexing systems)
                                }

# Edit the following to force neural net outputs (via octave transposition) into the ranges defined below.
# Every note range must be at least one octave. Values represent closed intervals.
# The default values below should be good for most people, but you can change them if you want.
# (21, 108) is a standard 88-key grand piano.
# A typical 61 key keyboard would be (36, 96)
ACCEPTABLE_NOTE_RANGE_BY_INST_RPR = {0: (21, 108),  # ac grand piano
                                     1: (21, 108),  # bright ac piano
                                     2: (24, 96),  # electric grand piano
                                     3: (24, 96),  # honky-tonk piano
                                     4: (24, 96),  # electric piano 1
                                     5: (24, 96),  # electric piano 2
                                     6: (21, 89),  # harpsichord
                                     7: (21, 88),  # clavi
                                     8: (60, 108),  # celesta
                                     9: (72, 108),  # glock
                                     10: (36, 96),  # music box
                                     11: (36, 96),  # vibraphone
                                     12: (36, 96),  # marimba
                                     13: (36, 96),  # xylophone
                                     14: (36, 96),  # tubular bells
                                     15: (36, 96),  # dulcimer
                                     16: (36, 96),  # drawbar organ
                                     17: (36, 96),  # percussive organ
                                     18: (36, 96),  # rock organ
                                     19: (36, 96),  # church organ
                                     20: (36, 96),  # reed organ
                                     21: (36, 96),  # accordian
                                     22: (36, 96),  # harmonica
                                     23: (36, 96),  # tango accordian
                                     24: (36, 96),  # ac guitar (nylon)
                                     25: (36, 96),  # ac guitar (steel)
                                     26: (36, 96),  # elec guitar (jazz)
                                     27: (36, 96),  # elec guitar (clean)
                                     28: (36, 96),  # elec guitar (muted)
                                     29: (36, 96),  # overdriven guitar
                                     30: (36, 96),  # distortion guitar
                                     31: (36, 96),  # guitar harmonics
                                     32: (36, 96),  # ac bass
                                     33: (36, 96),  # electric bass (finger)
                                     34: (36, 96),  # electric bass (pick)
                                     35: (36, 96),  # fretless bass
                                     36: (36, 96),  # slap bass 1
                                     37: (36, 96),  # slap bass 2
                                     38: (36, 96),  # synth bass 1
                                     39: (36, 96),  # synth bass 2
                                     40: (36, 96),  # violin
                                     41: (36, 96),  # viola
                                     42: (36, 96),  # cello
                                     43: (36, 96),  # contrabass
                                     44: (36, 96),  # tremolo strings
                                     45: (36, 96),  # pizz strings
                                     46: (36, 96),  # orchestral harp
                                     47: (36, 96),  # timpani
                                     48: (36, 96),  # string ensemble 1
                                     49: (36, 96),  # string ensemble 2
                                     50: (36, 96),  # synth strings 1
                                     51: (36, 96),  # synth strings 2
                                     52: (36, 96),  # choir aahs
                                     53: (36, 96),  # voice oohs
                                     54: (36, 96),  # synth voice
                                     55: (36, 96),  # orchestra hit
                                     56: (36, 96),  # trumpet
                                     57: (36, 96),  # trombone
                                     58: (36, 96),  # tuba
                                     59: (36, 96),  # muted trumpet
                                     60: (36, 96),  # french horn
                                     61: (36, 96),  # brass section
                                     62: (36, 96),  # synth brass 1
                                     63: (36, 96),  # synth brass 2
                                     64: (36, 96),  # soprano sax
                                     65: (36, 96),  # alto sax
                                     66: (36, 96),  # tenor sax
                                     67: (36, 96),  # baritone sax
                                     68: (36, 96),  # oboe
                                     69: (36, 96),  # english horn
                                     70: (36, 96),  # bassoon
                                     71: (36, 96),  # clarinet
                                     72: (36, 96),  # piccolo
                                     73: (36, 96),  # flute
                                     74: (36, 96),  # recorder
                                     75: (36, 96),  # pan flute
                                     76: (36, 96),  # blown bottle
                                     77: (36, 96),  # shakuhachi
                                     78: (36, 96),  # whistle
                                     79: (36, 96),  # ocarina
                                     80: (36, 96),  # lead 1 (square)
                                     81: (36, 96),  # lead 2 (sawtooth)
                                     82: (36, 96),  # lead 3 (calliope)
                                     83: (36, 96),  # lead 4 (chiff)
                                     84: (36, 96),  # lead 5 (charang)
                                     85: (36, 96),  # lead 6 (voice)
                                     86: (36, 96),  # lead 7 (fifths)
                                     87: (36, 96),  # lead 8 (bass + lead)
                                     88: (36, 96),  # pad 1 (new age)
                                     89: (36, 96),  # pad 2 (warm)
                                     90: (36, 96),  # pad 3 (polysynth)
                                     91: (36, 96),  # pad 4 (choir)
                                     92: (36, 96),  # pad 5 (bowed)
                                     93: (36, 96),  # pad 6 (metallic)
                                     94: (36, 96),  # pad 7 (halo)
                                     95: (36, 96),  # pad 8 (sweep)
                                     96: (36, 96),  # FX 1 (rain)
                                     97: (36, 96),  # FX 2 (soundtrack)
                                     98: (36, 96),  # FX 3 (crystal)
                                     99: (36, 96),  # FX 4 (atmosphere)
                                     100: (36, 96),  # FX 5 (brightness)
                                     101: (36, 96),  # FX 6 (goblins)
                                     102: (36, 96),  # FX 7 (echoes)
                                     103: (36, 96),  # FX 8 (sci-fi)
                                     104: (36, 96),  # sitar
                                     105: (36, 96),  # banjo
                                     106: (36, 96),  # shamisen
                                     107: (36, 96),  # koto
                                     108: (36, 96),  # kalimba
                                     109: (36, 96),  # bag pipe
                                     110: (36, 96),  # fiddle
                                     111: (36, 96),  # shanai
                                     112: (36, 96),  # tinkle bell
                                     113: (36, 96),  # agogo
                                     114: (36, 96),  # steel drums
                                     115: (36, 96),  # woodblock
                                     116: (36, 96),  # taiko
                                     117: (36, 96),  # melodic tom
                                     118: (36, 96),  # synth drum
                                     119: (36, 96),  # reverse cymbal
                                     120: (36, 96),  # guitar fret noise
                                     121: (36, 96),  # breath noise
                                     122: (36, 96),  # seashore
                                     123: (36, 96),  # bird tweet
                                     124: (36, 96),  # telephone ring
                                     125: (36, 96),  # helicopter
                                     126: (36, 96),  # applause
                                     127: (36, 96),  # gunshot
                                     128: (0, 127)  # GM drums on channel "10" (channel 9 in 0-based indexing systems)
                                     }

########################################################################################################################
# NEURAL NET TRAINING SETTINGS
# Do not change any of the following unless you are training a new model from scratch

QUANTIZE = (8, 6)

PATH_TO_TRAIN_MIDI = r'C:\delete\midi_train'
PATH_TO_PROCESSED_TRAIN_MIDI = r'C:\delete\midi_train_processed'

# You can ignore this. It is only for the code author's convenience during development.
PATH_TO_PROCESSED_TRAIN_MIDI_SUBSET = r'C:\delete\midi_train_processed_subset'

# Even if you are not planning to run validation, you should put at least one midi file in PATH_TO_VAL_MIDI
PATH_TO_VAL_MIDI = r'C:\delete\midi_val'
PATH_TO_PROCESSED_VAL_MIDI = r'C:\delete\midi_val_processed'

# Even if you are not planning to run tests, you should put at least one midi file in PATH_TO_TEST_MIDI
PATH_TO_TEST_MIDI = r'C:\delete\midi_test'
PATH_TO_PROCESSED_TEST_MIDI = r'C:\delete\midi_test_processed'

# make PATH_TO_MODELS empty if you are training from scratch; otherwise, make this the path that you are placing
#   already-pretrained or already-finetuned models in.
# Set UNJOINED = True to use a basic event-based vocab.
# Set UNJOINED = False to use a joined-event sentencepiece vocab learned from train data.
UNJOINED = False
if UNJOINED:
    PATH_TO_MODELS = r'C:\delete\unjoined composer assistant models'
else:
    PATH_TO_MODELS = r'C:\delete\composer assistant models'

# Empty folder needed for temporary storage during training. If it doesn't exist, it will be created for you.
PATH_TO_TEMP_FILES = r'C:\delete\composer assistant temp'

# During training, validation, and testing, as a form of data augmentation, we transpose our songs
# randomly by an integer amount in the closed interval [AUG_TRANS_MIN, AUG_TRANS_MAX]
# Defaults:
# AUG_TRANS_MIN, AUG_TRANS_MAX = -5, 6
AUG_TRANS_MIN = -5
AUG_TRANS_MAX = 6

# Our primary model is SPM_TRAIN_MODEL_WITH_NOTE_DURATION_COMMANDS.
# Train other models as well if you want to experiment with them.
SPM_TRAIN_MODEL_WITH_NOTE_OFFS = True
SPM_TRAIN_MODEL_WITH_NOTE_LENGTHS = True
SPM_TRAIN_MODEL_WITH_NOTE_DURATION_COMMANDS = True
SPM_TRAIN_MODEL_WITHOUT_NOTE_OFFS = True

# Train each sentencepiece model from approximately this number of examples.
# Each example is a string representing a single instrument in a single measure.
# Default: 5000000; The default amount requires a significant amount of RAM + CPU.
SPM_NUM_EXAMPLES = 5000000

# Number of chords/short musical phrases for sentencepiece vocab to learn
# Actual vocab size will be slightly higher (usually about 1500 higher; the exact amount depends on QUANTIZE)
N_PHRASES_FOR_VOCAB = 15000

# neural net parameters; additional parameters may be edited in pretrain_model.py.
MAX_LEN = 1024  # max number of tokens to train the net on per example; You should set MAX_LEN >= 512.
# If you want to set MAX_LEN to something higher than 4096, then you will need to edit pretrain_model.py.
D_MODEL = 512
N_LAYERS = 12
N_HEADS = 8
N_EPOCHS_SHORT = 5  # number of epochs to train on inputs of length 512

# SPM_TYPE: Choose one of the following to use to train your model.
# 'duration' (if SPM_TRAIN_MODEL_WITH_NOTE_DURATION_COMMANDS was True; default)
# 'length' (if SPM_TRAIN_MODEL_WITH_NOTE_LENGTHS was True; not recommended; use 'duration' instead)
# 'include_note_offs' (if SPM_TRAIN_MODEL_WITH_NOTE_OFFS was True; not recommended; use 'duration' instead)
# 'exclude_note_offs' (if SPM_TRAIN_MODEL_WITHOUT_NOTE_OFFS was True)
# 'unjoined_include_note_duration_commands' (always available; basic sentencepiece-like model containing no event merges)
# 'unjoined_include_note_length' (always available; basic sentencepiece-like model containing no event merges)
# 'unjoined_include_note_offs' (always available; basic sentencepiece-like model containing no event merges)
# 'unjoined_exclude_note_offs' (always available; basic sentencepiece-like model containing no event merges)
if UNJOINED:
    SPM_TYPE = 'unjoined_include_note_duration_commands'
else:
    SPM_TYPE = 'duration'

# Map drum notes to -1 to disable them. Set this map up before running preprocess_midi.py.
SIMPLIFIED_DRUM_MAP = {0: -1,
                       1: -1,
                       2: -1,
                       3: -1,
                       4: -1,
                       5: -1,
                       6: -1,
                       7: -1,
                       8: -1,
                       9: -1,
                       10: -1,
                       11: -1,
                       12: -1,
                       13: -1,
                       14: -1,
                       15: -1,
                       16: -1,
                       17: -1,
                       18: -1,
                       19: -1,
                       20: -1,
                       21: -1,
                       22: -1,
                       23: -1,
                       24: -1,
                       25: -1,
                       26: 38,  # snap
                       27: 36,  # boopy kick
                       28: 38,  # cardboard
                       29: 36,  # record scratch
                       30: 36,  # record scratch 2
                       31: 46,  # stick click
                       32: 50,  # L bongo?
                       33: 37,  # rim click?
                       34: 53,  # telephone ding
                       35: 36,  # kick
                       36: 36,  # kick - primary
                       37: 37,  # sidestick
                       38: 38,  # snare - primary
                       39: 39,  # clap
                       40: 38,  # snare
                       41: 41,  # lowest tom
                       42: 42,  # closed HH - primary
                       43: 43,  # tom
                       44: 42,  # closed HH
                       45: 45,  # tom
                       46: 46,  # open HH - primary
                       47: 47,  # tom
                       48: 48,  # tom
                       49: 49,  # crash - primary
                       50: 50,  # highest tom
                       51: 51,  # ride
                       52: 52,  # china
                       53: 53,  # ride bell
                       54: 54,  # tambourine
                       55: 49,  # crash
                       56: 56,  # cowbell
                       57: 49,  # crash
                       58: 73,  # frogs? (vibraslap)
                       59: 51,  # ride 2
                       60: 50,  # hi bongo
                       61: 47,  # low bongo
                       62: 48,  # mute hi conga
                       63: 45,  # open hi conga
                       64: 43,  # low conga
                       65: 48,  # hi timbale
                       66: 47,  # low timbale
                       67: 53,  # hi agogo (bell)
                       68: 51,  # low agogo (bell)
                       69: 69,  # shaker
                       70: 70,  # maraca
                       71: -1,  # whistle
                       72: -1,  # whistle
                       73: 73,  # short guiro (frog)
                       74: 73,  # long guiro (frog)
                       75: 76,  # claves (woodblockish)
                       76: 76,  # hi wood block
                       77: 76,  # low wood block
                       78: -1,  # mute cuica (honky voice?)
                       79: -1,  # open cuica (honky voice?)
                       80: 51,  # muted triangle
                       81: 53,  # unmuted ("open") triangle
                       82: 69,  # shaker
                       83: 46,  # sleigh bells
                       84: -1,  # small chimes
                       85: 39,  # snap
                       86: 48,  # hi tom
                       87: 47,  # lower tom
                       88: -1,
                       89: -1,
                       90: -1,
                       91: -1,
                       92: -1,
                       93: -1,
                       94: -1,
                       95: -1,
                       96: -1,
                       97: -1,
                       98: -1,
                       99: -1,
                       100: -1,
                       101: -1,
                       102: -1,
                       103: -1,
                       104: -1,
                       105: -1,
                       106: -1,
                       107: -1,
                       108: -1,
                       109: -1,
                       110: -1,
                       111: -1,
                       112: -1,
                       113: -1,
                       114: -1,
                       115: -1,
                       116: -1,
                       117: -1,
                       118: -1,
                       119: -1,
                       120: -1,
                       121: -1,
                       122: -1,
                       123: -1,
                       124: -1,
                       125: -1,
                       126: -1,
                       127: -1,
                       }

# every note range must be at least one octave
# values represent closed intervals
# (21, 108) is a standard 88-key grand piano
# Set this up before running spm_create_train_data.py.
ACCEPTABLE_NOTE_RANGE_BY_INST_TRAIN_TEST = {0: (0, 127),  # ac grand piano
                                            1: (0, 127),  # bright ac piano
                                            2: (0, 127),  # electric grand piano
                                            3: (0, 127),  # honky-tonk piano
                                            4: (0, 127),  # electric piano 1
                                            5: (0, 127),  # electric piano 2
                                            6: (0, 127),  # harpsichord
                                            7: (0, 127),  # clavi
                                            8: (0, 127),  # celesta
                                            9: (0, 127),  # glock
                                            10: (0, 127),  # music box
                                            11: (0, 127),  # vibraphone
                                            12: (0, 127),  # marimba
                                            13: (0, 127),  # xylophone
                                            14: (0, 127),  # tubular bells
                                            15: (0, 127),  # dulcimer
                                            16: (0, 127),  # drawbar organ
                                            17: (0, 127),  # percussive organ
                                            18: (0, 127),  # rock organ
                                            19: (0, 127),  # church organ
                                            20: (0, 127),  # reed organ
                                            21: (0, 127),  # accordian
                                            22: (0, 127),  # harmonica
                                            23: (0, 127),  # tango accordian
                                            24: (0, 127),  # ac guitar (nylon)
                                            25: (0, 127),  # ac guitar (steel)
                                            26: (0, 127),  # elec guitar (jazz)
                                            27: (0, 127),  # elec guitar (clean)
                                            28: (0, 127),  # elec guitar (muted)
                                            29: (0, 127),  # overdriven guitar
                                            30: (0, 127),  # distortion guitar
                                            31: (0, 127),  # guitar harmonics
                                            32: (0, 127),  # ac bass
                                            33: (0, 127),  # electric bass (finger)
                                            34: (0, 127),  # electric bass (pick)
                                            35: (0, 127),  # fretless bass
                                            36: (0, 127),  # slap bass 1
                                            37: (0, 127),  # slap bass 2
                                            38: (0, 127),  # synth bass 1
                                            39: (0, 127),  # synth bass 2
                                            40: (0, 127),  # violin
                                            41: (0, 127),  # viola
                                            42: (0, 127),  # cello
                                            43: (0, 127),  # contrabass
                                            44: (0, 127),  # tremolo strings
                                            45: (0, 127),  # pizz strings
                                            46: (0, 127),  # orchestral harp
                                            47: (0, 127),  # timpani
                                            48: (0, 127),  # string ensemble 1
                                            49: (0, 127),  # string ensemble 2
                                            50: (0, 127),  # synth strings 1
                                            51: (0, 127),  # synth strings 2
                                            52: (0, 127),  # choir aahs
                                            53: (0, 127),  # voice oohs
                                            54: (0, 127),  # synth voice
                                            55: (0, 127),  # orchestra hit
                                            56: (0, 127),  # trumpet
                                            57: (0, 127),  # trombone
                                            58: (0, 127),  # tuba
                                            59: (0, 127),  # muted trumpet
                                            60: (0, 127),  # french horn
                                            61: (0, 127),  # brass section
                                            62: (0, 127),  # synth brass 1
                                            63: (0, 127),  # synth brass 2
                                            64: (0, 127),  # soprano sax
                                            65: (0, 127),  # alto sax
                                            66: (0, 127),  # tenor sax
                                            67: (0, 127),  # baritone sax
                                            68: (0, 127),  # oboe
                                            69: (0, 127),  # english horn
                                            70: (0, 127),  # bassoon
                                            71: (0, 127),  # clarinet
                                            72: (0, 127),  # piccolo
                                            73: (0, 127),  # flute
                                            74: (0, 127),  # recorder
                                            75: (0, 127),  # pan flute
                                            76: (0, 127),  # blown bottle
                                            77: (0, 127),  # shakuhachi
                                            78: (0, 127),  # whistle
                                            79: (0, 127),  # ocarina
                                            80: (0, 127),  # lead 1 (square)
                                            81: (0, 127),  # lead 2 (sawtooth)
                                            82: (0, 127),  # lead 3 (calliope)
                                            83: (0, 127),  # lead 4 (chiff)
                                            84: (0, 127),  # lead 5 (charang)
                                            85: (0, 127),  # lead 6 (voice)
                                            86: (0, 127),  # lead 7 (fifths)
                                            87: (0, 127),  # lead 8 (bass + lead)
                                            88: (0, 127),  # pad 1 (new age)
                                            89: (0, 127),  # pad 2 (warm)
                                            90: (0, 127),  # pad 3 (polysynth)
                                            91: (0, 127),  # pad 4 (choir)
                                            92: (0, 127),  # pad 5 (bowed)
                                            93: (0, 127),  # pad 6 (metallic)
                                            94: (0, 127),  # pad 7 (halo)
                                            95: (0, 127),  # pad 8 (sweep)
                                            96: (0, 127),  # FX 1 (rain)
                                            97: (0, 127),  # FX 2 (soundtrack)
                                            98: (0, 127),  # FX 3 (crystal)
                                            99: (0, 127),  # FX 4 (atmosphere)
                                            100: (0, 127),  # FX 5 (brightness)
                                            101: (0, 127),  # FX 6 (goblins)
                                            102: (0, 127),  # FX 7 (echoes)
                                            103: (0, 127),  # FX 8 (sci-fi)
                                            104: (0, 127),  # sitar
                                            105: (0, 127),  # banjo
                                            106: (0, 127),  # shamisen
                                            107: (0, 127),  # koto
                                            108: (0, 127),  # kalimba
                                            109: (0, 127),  # bag pipe
                                            110: (0, 127),  # fiddle
                                            111: (0, 127),  # shanai
                                            112: (0, 127),  # tinkle bell
                                            113: (0, 127),  # agogo
                                            114: (0, 127),  # steel drums
                                            115: (0, 127),  # woodblock
                                            116: (0, 127),  # taiko
                                            117: (0, 127),  # melodic tom
                                            118: (0, 127),  # synth drum
                                            119: (0, 127),  # reverse cymbal
                                            120: (0, 127),  # guitar fret noise
                                            121: (0, 127),  # breath noise
                                            122: (0, 127),  # seashore
                                            123: (0, 127),  # bird tweet
                                            124: (0, 127),  # telephone ring
                                            125: (0, 127),  # helicopter
                                            126: (0, 127),  # applause
                                            127: (0, 127),  # gunshot
                                            128: (0, 127)
                                            # GM drums on channel "10" (channel 9 in 0-based indexing systems)
                                            }


# Server overrides for the code author. You can ignore these.
# PATH_TO_PROCESSED_TRAIN_MIDI = r'midi_train_processed'
# PATH_TO_PROCESSED_VAL_MIDI = r'midi_val_processed'
# PATH_TO_PROCESSED_TEST_MIDI = r'midi_test_processed'
# PATH_TO_MODELS = r'composer assistant models'
# PATH_TO_TEMP_FILES = r'composer assistant temp'
