#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.3),
    on April 20, 2026, at 16:22
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
prefs.hardware['audioLib'] = 'pygame'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from preparations
import os
import random
import numpy as np
import pandas as pd
from itertools import islice
from functools import reduce
from operator import add

def chunked_iterable(iterable, size):
    it = iter(iterable)
    return iter(lambda: tuple(islice(it, size)), ())

def shuffle(fnames, aud_dir, n_topics=7, n_sents=12):
    chunked_fnames = [list(chunk) for chunk in chunked_iterable(fnames, n_sents)]
    
    shuffled_fnames = list()
    for i in range(n_sents):
        chunk = list()
        for j in range(n_topics):
            fname = np.random.choice(chunked_fnames[j])
            chunk.append(fname)
            chunked_fnames[j].remove(fname)
        shuffled_fnames.append(chunk)
    
    for i in range(len(shuffled_fnames)):
        if i == 0:
            random.shuffle(shuffled_fnames[i])
        else:
            random.shuffle(shuffled_fnames[i])
            while shuffled_fnames[i-1][-1].split('_')[0] == shuffled_fnames[i][0].split('_')[0]:
                random.shuffle(shuffled_fnames[i])
    
    shuffled_fnames = reduce(add, shuffled_fnames)
    shuffled_fnames = [
        os.path.join(aud_dir, fname) for fname in shuffled_fnames
    ]
    
    return shuffled_fnames

aud_dir = './stimulus/aud/volcano_tts/aud_final'
num_topics = 7
num_sents_per_topic = 12 # 4 * 3

# Prepare blocks
fnames_designed_block = os.listdir(aud_dir)
fnames_designed_block = shuffle(fnames_designed_block, aud_dir)

num_trials_per_block = 21
blocks = list(chunked_iterable(
    fnames_designed_block,
    num_trials_per_block
))
random.shuffle(blocks)

# Define cue
cue = '句子是否正确'
# Run 'Before Experiment' code from trial_code
# Define response cue
response_cue = ['正确', '不正确']

from pypixxlib.propixx import PROPixxCTRL
from pypixxlib._libdpx import DPxUpdateRegCache, DPxSelectDevice, DPxStopAllScheds, DPxSetDoutValue

bitmask = 0x00FFFFFF

PROPixxCTRL()
DPxSelectDevice('PROPixx Ctrl')
DPxStopAllScheds()
DPxSetDoutValue(0, bitmask)
DPxUpdateRegCache()



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.3'
expName = 'ConjSemantic'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='E:\\ConjSemProj\\experiment_ver-3\\paradigm\\ConjSemantic_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1536, 864], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "Waiting" ---
waiting_text = visual.TextStim(win=win, name='waiting_text',
    text='实验即将开始，请稍候',
    font='Open Sans',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
    color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
waiting_key = keyboard.Keyboard()

# --- Initialize components for Routine "Welcome" ---
# Run 'Begin Experiment' code from preparations
# Initialize
block_counter = 0
trial_counter = 0
n_blocks = len(blocks)
n_trials = len(blocks[0])

if_prompt = 1
block_cue = f'Block {block_counter + 1} 即将开始\n' + \
            '请注意问题:\n\n' + \
            cue
welcome_text = visual.TextStim(win=win, name='welcome_text',
    text='欢迎参加本次实验',
    font='Open Sans',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
    color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "Prompt" ---
prompt_text1 = visual.TextStim(win=win, name='prompt_text1',
    text='本次实验分为4个组块\n每个组块中你会听到一系列句子',
    font='Open Sans',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
prompt_text2 = visual.TextStim(win=win, name='prompt_text2',
    text='请根据你听到的句子\n对如下问题做出判断：\n\n“句子是否正确”\n\n并按照指示进行按键',
    font='Open Sans',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
prompt_text3 = visual.TextStim(win=win, name='prompt_text3',
    text='每个组块结束后，会出现\n\n“休息”\n\n的字样，表明这一组实验已经结束\n并进入持续5s的间隔',
    font='Open Sans',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
prompt_text4 = visual.TextStim(win=win, name='prompt_text4',
    text='现在，实验正式开始',
    font='Open Sans',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
prompt_text5 = visual.TextStim(win=win, name='prompt_text5',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# --- Initialize components for Routine "Cue" ---
cue_text = visual.TextStim(win=win, name='cue_text',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
    color=[0.6549, 0.6549, 0.6549], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "Fixation" ---
polygon_2 = visual.ShapeStim(
    win=win, name='polygon_2', vertices='cross',
    size=(0.10, 0.10),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.6549, 0.6549, 0.6549], fillColor=[0.6549, 0.6549, 0.6549],
    opacity=None, depth=0.0, interpolate=True)

# --- Initialize components for Routine "Trial" ---
sound_stim = sound.Sound('A', secs=-1, stereo=True, hamming=True,
    name='sound_stim')
sound_stim.setVolume(1.0)
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.10, 0.10),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.6549, 0.6549, 0.6549], fillColor=[0.6549, 0.6549, 0.6549],
    opacity=None, depth=-2.0, interpolate=True)

# --- Initialize components for Routine "Response" ---
key_resp = keyboard.Keyboard()
response_prompt = visual.TextStim(win=win, name='response_prompt',
    text='',
    font='Open Sans',
    pos=(0, 0.20), height=0.08, wrapWidth=None, ori=0.0, 
    color=[0.6549, 0.6549, 0.6549], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
left_cue = visual.TextStim(win=win, name='left_cue',
    text='',
    font='Open Sans',
    pos=(-0.3, -0.1), height=0.08, wrapWidth=None, ori=0.0, 
    color=[0.6549, 0.6549, 0.6549], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
right_cue = visual.TextStim(win=win, name='right_cue',
    text='',
    font='Open Sans',
    pos=(0.3, -0.1), height=0.08, wrapWidth=None, ori=0.0, 
    color=[0.6549, 0.6549, 0.6549], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# --- Initialize components for Routine "ITI" ---
iti_text = visual.TextStim(win=win, name='iti_text',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "IBI" ---
ibi_text = visual.TextStim(win=win, name='ibi_text',
    text='休息5秒',
    font='Open Sans',
    pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "End" ---
end_text = visual.TextStim(win=win, name='end_text',
    text='本次实验结束\n感谢您的参与',
    font='Open Sans',
    pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "Waiting" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
waiting_key.keys = []
waiting_key.rt = []
_waiting_key_allKeys = []
# keep track of which components have finished
WaitingComponents = [waiting_text, waiting_key]
for thisComponent in WaitingComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "Waiting" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *waiting_text* updates
    if waiting_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        waiting_text.frameNStart = frameN  # exact frame index
        waiting_text.tStart = t  # local t and not account for scr refresh
        waiting_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(waiting_text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'waiting_text.started')
        waiting_text.setAutoDraw(True)
    
    # *waiting_key* updates
    waitOnFlip = False
    if waiting_key.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
        # keep track of start time/frame for later
        waiting_key.frameNStart = frameN  # exact frame index
        waiting_key.tStart = t  # local t and not account for scr refresh
        waiting_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(waiting_key, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'waiting_key.started')
        waiting_key.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(waiting_key.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(waiting_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if waiting_key.status == STARTED and not waitOnFlip:
        theseKeys = waiting_key.getKeys(keyList=['space'], waitRelease=False)
        _waiting_key_allKeys.extend(theseKeys)
        if len(_waiting_key_allKeys):
            waiting_key.keys = _waiting_key_allKeys[-1].name  # just the last key pressed
            waiting_key.rt = _waiting_key_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in WaitingComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "Waiting" ---
for thisComponent in WaitingComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if waiting_key.keys in ['', [], None]:  # No response was made
    waiting_key.keys = None
thisExp.addData('waiting_key.keys',waiting_key.keys)
if waiting_key.keys != None:  # we had a response
    thisExp.addData('waiting_key.rt', waiting_key.rt)
thisExp.nextEntry()
# the Routine "Waiting" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "Welcome" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
WelcomeComponents = [welcome_text]
for thisComponent in WelcomeComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "Welcome" ---
while continueRoutine and routineTimer.getTime() < 3.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *welcome_text* updates
    if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome_text.frameNStart = frameN  # exact frame index
        welcome_text.tStart = t  # local t and not account for scr refresh
        welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'welcome_text.started')
        welcome_text.setAutoDraw(True)
    if welcome_text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > welcome_text.tStartRefresh + 3.0-frameTolerance:
            # keep track of stop time/frame for later
            welcome_text.tStop = t  # not accounting for scr refresh
            welcome_text.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.stopped')
            welcome_text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in WelcomeComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "Welcome" ---
for thisComponent in WelcomeComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# Run 'End Routine' code from preparations
if not expInfo['participant'].endswith('run1'):
    if_prompt = 0
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-3.000000)

# set up handler to look after randomisation of conditions etc
skip_prompt = data.TrialHandler(nReps=if_prompt, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='skip_prompt')
thisExp.addLoop(skip_prompt)  # add the loop to the experiment
thisSkip_prompt = skip_prompt.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisSkip_prompt.rgb)
if thisSkip_prompt != None:
    for paramName in thisSkip_prompt:
        exec('{} = thisSkip_prompt[paramName]'.format(paramName))

for thisSkip_prompt in skip_prompt:
    currentLoop = skip_prompt
    # abbreviate parameter names if possible (e.g. rgb = thisSkip_prompt.rgb)
    if thisSkip_prompt != None:
        for paramName in thisSkip_prompt:
            exec('{} = thisSkip_prompt[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "Prompt" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # keep track of which components have finished
    PromptComponents = [prompt_text1, prompt_text2, prompt_text3, prompt_text4, prompt_text5]
    for thisComponent in PromptComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Prompt" ---
    while continueRoutine and routineTimer.getTime() < 22.4:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prompt_text1* updates
        if prompt_text1.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            prompt_text1.frameNStart = frameN  # exact frame index
            prompt_text1.tStart = t  # local t and not account for scr refresh
            prompt_text1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prompt_text1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prompt_text1.started')
            prompt_text1.setAutoDraw(True)
        if prompt_text1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prompt_text1.tStartRefresh + 4.0-frameTolerance:
                # keep track of stop time/frame for later
                prompt_text1.tStop = t  # not accounting for scr refresh
                prompt_text1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_text1.stopped')
                prompt_text1.setAutoDraw(False)
        
        # *prompt_text2* updates
        if prompt_text2.status == NOT_STARTED and tThisFlip >= 4.2-frameTolerance:
            # keep track of start time/frame for later
            prompt_text2.frameNStart = frameN  # exact frame index
            prompt_text2.tStart = t  # local t and not account for scr refresh
            prompt_text2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prompt_text2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prompt_text2.started')
            prompt_text2.setAutoDraw(True)
        if prompt_text2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prompt_text2.tStartRefresh + 8.0-frameTolerance:
                # keep track of stop time/frame for later
                prompt_text2.tStop = t  # not accounting for scr refresh
                prompt_text2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_text2.stopped')
                prompt_text2.setAutoDraw(False)
        
        # *prompt_text3* updates
        if prompt_text3.status == NOT_STARTED and tThisFlip >= 12.3-frameTolerance:
            # keep track of start time/frame for later
            prompt_text3.frameNStart = frameN  # exact frame index
            prompt_text3.tStart = t  # local t and not account for scr refresh
            prompt_text3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prompt_text3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prompt_text3.started')
            prompt_text3.setAutoDraw(True)
        if prompt_text3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prompt_text3.tStartRefresh + 6.0-frameTolerance:
                # keep track of stop time/frame for later
                prompt_text3.tStop = t  # not accounting for scr refresh
                prompt_text3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_text3.stopped')
                prompt_text3.setAutoDraw(False)
        
        # *prompt_text4* updates
        if prompt_text4.status == NOT_STARTED and tThisFlip >= 18.4-frameTolerance:
            # keep track of start time/frame for later
            prompt_text4.frameNStart = frameN  # exact frame index
            prompt_text4.tStart = t  # local t and not account for scr refresh
            prompt_text4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prompt_text4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prompt_text4.started')
            prompt_text4.setAutoDraw(True)
        if prompt_text4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prompt_text4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                prompt_text4.tStop = t  # not accounting for scr refresh
                prompt_text4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_text4.stopped')
                prompt_text4.setAutoDraw(False)
        
        # *prompt_text5* updates
        if prompt_text5.status == NOT_STARTED and tThisFlip >= 21.4-frameTolerance:
            # keep track of start time/frame for later
            prompt_text5.frameNStart = frameN  # exact frame index
            prompt_text5.tStart = t  # local t and not account for scr refresh
            prompt_text5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prompt_text5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prompt_text5.started')
            prompt_text5.setAutoDraw(True)
        if prompt_text5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prompt_text5.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                prompt_text5.tStop = t  # not accounting for scr refresh
                prompt_text5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_text5.stopped')
                prompt_text5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in PromptComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Prompt" ---
    for thisComponent in PromptComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-22.400000)
# completed if_prompt repeats of 'skip_prompt'


# set up handler to look after randomisation of conditions etc
Blocks_loop = data.TrialHandler(nReps=n_blocks, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='Blocks_loop')
thisExp.addLoop(Blocks_loop)  # add the loop to the experiment
thisBlocks_loop = Blocks_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
if thisBlocks_loop != None:
    for paramName in thisBlocks_loop:
        exec('{} = thisBlocks_loop[paramName]'.format(paramName))

for thisBlocks_loop in Blocks_loop:
    currentLoop = Blocks_loop
    # abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
    if thisBlocks_loop != None:
        for paramName in thisBlocks_loop:
            exec('{} = thisBlocks_loop[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "Cue" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from cue_code
    trial_counter = 0
    n_trials = len(blocks[block_counter])
    
    iti = 1.0
    
    block_cue = f'Block {block_counter + 1} 即将开始\n' + \
                '请注意问题:\n\n' + \
                cue
    cue_text.setText(block_cue)
    # keep track of which components have finished
    CueComponents = [cue_text]
    for thisComponent in CueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Cue" ---
    while continueRoutine and routineTimer.getTime() < 3.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cue_text* updates
        if cue_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            cue_text.frameNStart = frameN  # exact frame index
            cue_text.tStart = t  # local t and not account for scr refresh
            cue_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cue_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cue_text.started')
            cue_text.setAutoDraw(True)
        if cue_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cue_text.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                cue_text.tStop = t  # not accounting for scr refresh
                cue_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_text.stopped')
                cue_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in CueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Cue" ---
    for thisComponent in CueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.500000)
    
    # set up handler to look after randomisation of conditions etc
    Trials_loop = data.TrialHandler(nReps=n_trials, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='Trials_loop')
    thisExp.addLoop(Trials_loop)  # add the loop to the experiment
    thisTrials_loop = Trials_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
    if thisTrials_loop != None:
        for paramName in thisTrials_loop:
            exec('{} = thisTrials_loop[paramName]'.format(paramName))
    
    for thisTrials_loop in Trials_loop:
        currentLoop = Trials_loop
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
        if thisTrials_loop != None:
            for paramName in thisTrials_loop:
                exec('{} = thisTrials_loop[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "Fixation" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # keep track of which components have finished
        FixationComponents = [polygon_2]
        for thisComponent in FixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fixation" ---
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon_2* updates
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_2.started')
                polygon_2.setAutoDraw(True)
            if polygon_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_2.tStop = t  # not accounting for scr refresh
                    polygon_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_2.stopped')
                    polygon_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation" ---
        for thisComponent in FixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Trial" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trial_code
        # Audio File Name
        aud_file = blocks[block_counter][trial_counter]
        aud_fname = os.path.basename(aud_file)
        block_type = aud_fname.split('_')[1]
        Trials_loop.addData(
            'Stim_fname',
            aud_fname
        )
        
        # Random Inter Trial Intervval
        iti_time = random.random() * 0.2 + 0.9
        
        # Random Key Response
        correct = None
        incorrect = None
        if 'sent-1' in aud_fname or 'sent-4' in aud_fname:
            correct = response_cue[0]
            incorrect = response_cue[1]
        else:
            correct = response_cue[1]
            incorrect = response_cue[0]
        
        rand_num = random.random()
        
        left = correct
        right = incorrect
        if rand_num < 0.5:
            left = '左：' + correct
            right = '右：' + incorrect
            correct_key = '1'
        else:
            left = '左：' + incorrect
            right = '右：' + correct
            correct_key = '4'
        
        # Set Trigger
        stim_topic_index = int(
            aud_fname.split('_')[0][-1]
        )
        stim_block_index = int(
            aud_fname.split('_')[1][-1]
        )
        stim_sent_index = int(
            aud_fname.replace('.wav', '').split('_')[2][-1]
        )
        
        trigger_value = (stim_topic_index - 1) * num_sents_per_topic \
                        + (stim_block_index - 1) * num_sents_per_topic / 3 \
                        + stim_sent_index
        trigger_value = int(trigger_value)
        
        Trials_loop.addData(
            'Stim_trigger',
            trigger_value
        )
        DPxSetDoutValue(trigger_value, bitmask)
        DPxUpdateRegCache()
        core.wait(0.3)
        DPxSetDoutValue(0, bitmask)
        DPxUpdateRegCache()
        sound_stim.setSound(aud_file, secs=4.5, hamming=True)
        sound_stim.setVolume(1.0, log=False)
        # keep track of which components have finished
        TrialComponents = [sound_stim, polygon]
        for thisComponent in TrialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Trial" ---
        while continueRoutine and routineTimer.getTime() < 5.6:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # start/stop sound_stim
            if sound_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_stim.frameNStart = frameN  # exact frame index
                sound_stim.tStart = t  # local t and not account for scr refresh
                sound_stim.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_stim.started', tThisFlipGlobal)
                sound_stim.play(when=win)  # sync with win flip
            if sound_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_stim.tStartRefresh + 4.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_stim.tStop = t  # not accounting for scr refresh
                    sound_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_stim.stopped')
                    sound_stim.stop()
            
            # *polygon* updates
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                polygon.setAutoDraw(True)
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 5.6-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.stopped')
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TrialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Trial" ---
        for thisComponent in TrialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from trial_code
        trial_counter += 1
        sound_stim.stop()  # ensure sound has stopped at end of routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.600000)
        
        # --- Prepare to start Routine "Response" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        # Run 'Begin Routine' code from key_response_code
        trigger_value = 100
        
        DPxSetDoutValue(trigger_value, bitmask)
        DPxUpdateRegCache()
        core.wait(0.1)
        DPxSetDoutValue(0, bitmask)
        DPxUpdateRegCache()
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        response_prompt.setText('请回答')
        left_cue.setText(left)
        right_cue.setText(right)
        # keep track of which components have finished
        ResponseComponents = [key_resp, response_prompt, left_cue, right_cue]
        for thisComponent in ResponseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Response" ---
        while continueRoutine and routineTimer.getTime() < 2.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp* updates
            waitOnFlip = False
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['1','4'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    # was this correct?
                    if (key_resp.keys == str(correct_key)) or (key_resp.keys == correct_key):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *response_prompt* updates
            if response_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                response_prompt.frameNStart = frameN  # exact frame index
                response_prompt.tStart = t  # local t and not account for scr refresh
                response_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(response_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'response_prompt.started')
                response_prompt.setAutoDraw(True)
            if response_prompt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > response_prompt.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    response_prompt.tStop = t  # not accounting for scr refresh
                    response_prompt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_prompt.stopped')
                    response_prompt.setAutoDraw(False)
            
            # *left_cue* updates
            if left_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_cue.frameNStart = frameN  # exact frame index
                left_cue.tStart = t  # local t and not account for scr refresh
                left_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_cue.started')
                left_cue.setAutoDraw(True)
            if left_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > left_cue.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    left_cue.tStop = t  # not accounting for scr refresh
                    left_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_cue.stopped')
                    left_cue.setAutoDraw(False)
            
            # *right_cue* updates
            if right_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_cue.frameNStart = frameN  # exact frame index
                right_cue.tStart = t  # local t and not account for scr refresh
                right_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_cue.started')
                right_cue.setAutoDraw(True)
            if right_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > right_cue.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    right_cue.tStop = t  # not accounting for scr refresh
                    right_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_cue.stopped')
                    right_cue.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ResponseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Response" ---
        for thisComponent in ResponseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for Trials_loop (TrialHandler)
        Trials_loop.addData('key_resp.keys',key_resp.keys)
        Trials_loop.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            Trials_loop.addData('key_resp.rt', key_resp.rt)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "ITI" ---
        continueRoutine = True
        routineForceEnded = False
        # update component parameters for each repeat
        iti_text.setText('')
        # keep track of which components have finished
        ITIComponents = [iti_text]
        for thisComponent in ITIComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ITI" ---
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *iti_text* updates
            if iti_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                iti_text.frameNStart = frameN  # exact frame index
                iti_text.tStart = t  # local t and not account for scr refresh
                iti_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(iti_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'iti_text.started')
                iti_text.setAutoDraw(True)
            if iti_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > iti_text.tStartRefresh + iti_time-frameTolerance:
                    # keep track of stop time/frame for later
                    iti_text.tStop = t  # not accounting for scr refresh
                    iti_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'iti_text.stopped')
                    iti_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ITIComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ITI" ---
        for thisComponent in ITIComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "ITI" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed n_trials repeats of 'Trials_loop'
    
    
    # --- Prepare to start Routine "IBI" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # keep track of which components have finished
    IBIComponents = [ibi_text]
    for thisComponent in IBIComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "IBI" ---
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ibi_text* updates
        if ibi_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ibi_text.frameNStart = frameN  # exact frame index
            ibi_text.tStart = t  # local t and not account for scr refresh
            ibi_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ibi_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ibi_text.started')
            ibi_text.setAutoDraw(True)
        if ibi_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ibi_text.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                ibi_text.tStop = t  # not accounting for scr refresh
                ibi_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ibi_text.stopped')
                ibi_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in IBIComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "IBI" ---
    for thisComponent in IBIComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # Run 'End Routine' code from ibi_code
    block_counter += 1
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
# completed n_blocks repeats of 'Blocks_loop'


# --- Prepare to start Routine "End" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
EndComponents = [end_text]
for thisComponent in EndComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "End" ---
while continueRoutine and routineTimer.getTime() < 5.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *end_text* updates
    if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end_text.frameNStart = frameN  # exact frame index
        end_text.tStart = t  # local t and not account for scr refresh
        end_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'end_text.started')
        end_text.setAutoDraw(True)
    if end_text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > end_text.tStartRefresh + 5.0-frameTolerance:
            # keep track of stop time/frame for later
            end_text.tStop = t  # not accounting for scr refresh
            end_text.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.stopped')
            end_text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in EndComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "End" ---
for thisComponent in EndComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-5.000000)
# Run 'End Experiment' code from trial_code
DPxStopAllScheds()
DPxSetDoutValue(0, bitmask)
DPxUpdateRegCache()


# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
