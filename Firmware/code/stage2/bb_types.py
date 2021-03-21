import os 

# Create abs path from relative path
SOUNDPACK_DIR = os.path.abspath("../../../Data/sf2_soundpacks/")
MODEL_DIR = os.path.abspath("../../../ML/model_checkpoints/")

GENRE_TO_ID = {
        "generic" : 0,
        "rock" : 1,
        "jazz" : 2,
        "pop" : 3,
        "hiphop" : 4,
        }

ID_TO_MODEL = {
        0 : "groovae_all/allgenre.tar",
        1 : "groovae_rock/groovae_rock.tar",
        2 : "groovae_jazz/jazz.tar",
        3 : "groovae_pop/pop.tar",
        4 : "groovae_hiphop/hiphop.tar",
        }

TIMRE_TO_ID = {
        "Default" : 0,
        "Secondary default" : 1,
        "808 pack" : 2,
        "Jazz" : 3,
        "Woodblocks" : 4,
        "Electronic sounds" : 5,
        "Synth Drums" : 6,
        "Hard Rock" : 7,
        "Brushes" : 8,
        "Warm Pad" : 9,
        }

ID_TO_SOUNDPACK = {
        0 : "GoodStandard.sf2",
        1 : "GoodStandard2.sf2",
        2 : "808s.sf2",
        3 : "Jazz.sf2",
        4 : "Woodblocks.sf2",
        5 : "Electronic.sf2",
        6 : "SynthDrums.sf2",
        7 : "HardRockDrums.sf2",
        8 : "Brushy.sf2",
        9 : "WarmPad.sf2",
        }
