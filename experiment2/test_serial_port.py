#!/usr/bin/env python

# import the rusocsci.buttonbox module
from psychopy import core, visual, event, gui, monitors
import serial
import solenoid_functions

# Set-up window:
scnWidth, scnHeight = (800, 600)
screen_width        = 53.5 # centimeters
screen_dist         = 58.0
mon = monitors.Monitor('myMac15', width=screen_width, distance=screen_dist)
mon.setSizePix((scnWidth, scnHeight))
win = visual.Window(
    (scnWidth, scnHeight),
    color = (118,118,118),
    colorSpace = 'rgb255',
    monitor = mon,
    units = 'pix',
    allowStencil = True,
    autoLog = False)
win.setMouseVisible(False)

stim_instr   = visual.TextStim(win, text="test", color='black', pos=(0.0, 0.0),)

serial_port = solenoid_functions.define_serial_port()

for t in [1,2,3,1,2,3,1,2,3]:
    # Welcome instructions
    stim_instr.setText(str(t) + " push a button to continue")
    stim_instr.draw()
    win.flip()
    solenoid_functions.send_solnenoid_pulses(t, serial_port)
    core.wait(0.25)
    event.waitKeys()


