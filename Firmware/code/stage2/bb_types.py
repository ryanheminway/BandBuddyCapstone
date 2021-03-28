import os 

# Create abs path from relative path
SOUNDPACK_DIR = os.path.abspath("../../../Data/sf2_soundpacks/")
MODEL_DIR = os.path.abspath("../../../ML/model_checkpoints/")

GENRE_TO_ID = (
        (0, "Generic"),
        (1, "Rock"),
        (2, "Jazz"),
        (3, "Pop"),
        (4, "Hiphop"),
)

ID_TO_MODEL = {
        0 : "groovae_all/allgenre.tar",
        1 : "groovae_rock/groovae_rock.tar",
        2 : "groovae_jazz/jazz.tar",
        3 : "groovae_pop/pop.tar",
        4 : "groovae_hiphop/hiphop.tar",
        }

TIMBRE_TO_ID = (
        (0, "Default"),
        (1, "Secondary Default"),
        (2, "808 Pack"),
        (3, "Jazz"),
        (4, "Woodblocks"),
        (5, "Electronic Sounds"),
        (6, "Synth Drums"),
        (7, "Hard Rock"),
        (8, "Brushes"),
        (9, "Warm Pad"),
)

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

BARS_TO_VALUE = (
        (2, "2 Bars"),
        (4, "4 Bars"),
)
