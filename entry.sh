#!/bin/bash

tmux new-session -d -s colin;                           # start new detached tmux session
tmux split-window;                                      # split the detached tmux session
tmux send 'cd /seminar/datscieo-0/colin/runs' ENTER;    # send 2nd command to 2nd pane. I believe there's a `--target` option to target specific pane.
tail -f /dev/null                                       # keeps container running (if debugging and no script should be launched)
