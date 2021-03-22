#!/bin/bash

# Load button log script
. /usr/local/pisound/scripts/common/common.sh

# Send a SIGINT to Big Brother
killall -s SIGINT big_brother

# Send SIGINTs to Stages 1 and 3 - let each program deal with them
# ANALOG2WAV=$(pgrep -x analog2wav)
# if [[ -n "$ANALOG2WAV" ]]; then 
#     log "analog2wav running - sending SIGINT!"
#     killall -s SIGINT analog2wav
# else 
#     # Record a failure 
#     log "analog2wav not running!!!"
# fi

# WAVS2ANALOG=$(pgrep -x wavs2analog)
# if [[ -n "$WAVS2ANALOG" ]]; then
#     log "wavs2analog found - sending SIGINT!"
#     killall -s SIGINT wavs2analog
# else 
#     # Record a failure 
#     log "wavs2analog not running!!!"
# fi