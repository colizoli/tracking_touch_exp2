"""
Setting up the EyeLink eye tracker
Psychopy 2024.2.4
O.Colizoli 2026

Notes:
Make sure the cable is plugged into the camera, lens cap off, and eye link computer in eyelink mode.
Have to hit ENTER twice to get camera on screen first.
ESC will exit each mode.
"""

# Import necessary modules
from psychopy import core, visual, monitors
import funcs_pylink3 as eye
# from IPython import embed as shell # for Olympia debugging only, comment out if crashes

# Screen-specific parameters lab MM 00.478
scnWidth, scnHeight = (1920, 1080)
screen_width        = 53.5 # centimeters (double check this)
screen_dist         = 58.0
grey = [128,128,128]

# Set-up window:
mon = monitors.Monitor('myMac15', width=screen_width, distance=screen_dist)
mon.setSizePix((scnWidth, scnHeight))
win = visual.Window(
    (scnWidth, scnHeight),
    color = grey,
    colorSpace = 'rgb255',
    monitor = mon,
    fullscr = True,
    units = 'pix',
    allowStencil = True,
    autoLog = False)
win.setMouseVisible(False)

subject_ID = 5000
task = 'eyelink_setup'
eye.config(subject_ID,task)
eye.run_calibration(win, scnWidth, scnHeight)
# eye.start_recording()
# eye.send_message('subject_ID sub-{} task-{} timestamp {}'.format(subject_ID, task ,timestr))

eye.stop_recording('finished', task)    