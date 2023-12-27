#!/bin/bash

tmux new-window -n:colin 'tmux source ~/.tmux.conf'     # sources config, doesnt open window however
# tail -f /dev/null                                     # keeps container running (if debugging and no script should be launched)
python training.py

