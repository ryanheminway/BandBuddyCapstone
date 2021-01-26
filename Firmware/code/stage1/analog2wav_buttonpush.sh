#!/bin/bash

# Load button log script
echo "got here 1"  >> /home/patch/hmm.hmm
. /usr/local/pisound/scripts/common/common.sh
echo "got here 2" >> /home/patch/hmm.hmm

# If the program is running, send it a SIGINT 
PGREP=$(pgrep -x analog2wav)
if [[ -n "$PGREP" ]]; then 
    # Simply SIGINT the analog2wav process by name
    log "analog2wav running - sending SIGINT!"
    killall -s SIGINT analog2wav
else 
    # Spawn the analog2wav process
    log "analog2wav not running - spawning now!"
    /home/patch/BandBuddyCapstone/Firmware/code/stage1/analog2wav &
fi