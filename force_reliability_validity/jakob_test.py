#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on Oktober 08, 2026, at 10:47
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard
from psychopy_apparatus.hardware.apparatus import Apparatus

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'jakob_test'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\PhD-s\\dfg-dynamind\\force_reliability_validity\\jakob_test.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # initialize 'apparatus'
    deviceManager.addDevice(
        deviceName='apparatus',
        deviceClass='psychopy.hardware.ApparatusDevice',
        port='COM4',
        baudrate='921600',
        simulate=False,
        debug=False
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "kraftmessung" ---
    
    apparatusForce = Apparatus('apparatus')
    apparatusForce_raw_buffer = []
    
    apparatusLED_3 = Apparatus(None)
    
    # --- Initialize components for Routine "stop" ---
    
    apparatusLED_2 = Apparatus('apparatus')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "kraftmessung" ---
    # create an object to store info about Routine kraftmessung
    kraftmessung = data.Routine(
        name='kraftmessung',
        components=[apparatusForce, apparatusLED_3],
    )
    kraftmessung.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    apparatusLED_3_led_off_sent = False
    # store start times for kraftmessung
    kraftmessung.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    kraftmessung.tStart = globalClock.getTime(format='float')
    kraftmessung.status = STARTED
    thisExp.addData('kraftmessung.started', kraftmessung.tStart)
    kraftmessung.maxDuration = None
    # keep track of which components have finished
    kraftmessungComponents = kraftmessung.components
    for thisComponent in kraftmessung.components:
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
    
    # --- Run Routine "kraftmessung" ---
    thisExp.currentRoutine = kraftmessung
    kraftmessung.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # if apparatusForce is starting this frame...
        if apparatusForce.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            apparatusForce.frameNStart = frameN  # exact frame index
            apparatusForce.tStart = t  # local t and not account for scr refresh
            apparatusForce.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(apparatusForce, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('apparatusForce.started', t)
            # update status
            apparatusForce.status = STARTED
            apparatusForce.startForceMeasurement(100, 'white')
        
        # if apparatusForce is active this frame...
        if apparatusForce.status == STARTED:
            # update params
            pass
            apparatusForce.updateForceMeasurement()
        
        # if apparatusForce is stopping this frame...
        if apparatusForce.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > apparatusForce.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                apparatusForce.tStop = t  # not accounting for scr refresh
                apparatusForce.tStopRefresh = tThisFlipGlobal  # on global time
                apparatusForce.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.addData('apparatusForce.stopped', t)
                # update status
                apparatusForce.status = FINISHED
                apparatusForce.stopForceMeasurement()
        
        # if apparatusLED_3 is starting this frame...
        if apparatusLED_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            apparatusLED_3.frameNStart = frameN  # exact frame index
            apparatusLED_3.tStart = t  # local t and not account for scr refresh
            apparatusLED_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(apparatusLED_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('apparatusLED_3.started', t)
            # update status
            apparatusLED_3.status = STARTED
            apparatusLED_3.setLights("all", 'red')
        
        # if apparatusLED_3 is active this frame...
        if apparatusLED_3.status == STARTED:
            # update params
            pass
            
        
        # if apparatusLED_3 is stopping this frame...
        if apparatusLED_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > apparatusLED_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                apparatusLED_3.tStop = t  # not accounting for scr refresh
                apparatusLED_3.tStopRefresh = tThisFlipGlobal  # on global time
                apparatusLED_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.addData('apparatusLED_3.stopped', t)
                # update status
                apparatusLED_3.status = FINISHED
                if True:
                   apparatusLED_3.turnOffLights("all")
                   apparatusLED_3_led_off_sent = True
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=kraftmessung,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            kraftmessung.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if kraftmessung.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in kraftmessung.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "kraftmessung" ---
    for thisComponent in kraftmessung.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for kraftmessung
    kraftmessung.tStop = globalClock.getTime(format='float')
    kraftmessung.tStopRefresh = tThisFlipGlobal
    thisExp.addData('kraftmessung.stopped', kraftmessung.tStop)
    thisExp.addData('apparatusForce.rate', 100)
    thisExp.addData('apparatusForce.dynamometer', 'white')
    thisExp.addData('apparatusForce.maxWhiteForce', apparatusForce.maxWhiteForce)
    thisExp.addData('apparatusForce.maxBlueForce', apparatusForce.maxBlueForce)
    if True:
        _loop = thisExp
        _trial_index = _loop.thisN if _loop is not None and hasattr(_loop, 'thisN') else -1
        _trial_name = _loop.name if _loop is not None and hasattr(_loop, 'name') else ''
        _identifier = str("test") if "test" is not None else ''
        _records = apparatusForce.forceRows if hasattr(apparatusForce, 'forceRows') else []
        for _i, _record in enumerate(_records):
            apparatusForce_raw_buffer.append({
                'participant': expInfo.get('participant', ''),
                'session': expInfo.get('session', ''),
                'routine': 'kraftmessung',
                'component': 'apparatusForce',
                'trial_index': _trial_index,
                'trial_name': _trial_name,
                'identifier': _identifier,
                'sample_index': _i,
                'white_time': _record['white_time'],
                'blue_time': _record['blue_time'],
                'time': _record['time'],
                'white_force': _record['white_force'],
                'blue_force': _record['blue_force'],
                'white_force_raw_counts': _record['white_force_raw_counts'] if _record['white_force_raw_counts'] is not None else '',
                'blue_force_raw_counts': _record['blue_force_raw_counts'] if _record['blue_force_raw_counts'] is not None else '',
            })
    if False and not apparatusLED_3_led_off_sent:
        apparatusLED_3.turnOffLights("all")
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if kraftmessung.maxDurationReached:
        routineTimer.addTime(-kraftmessung.maxDuration)
    elif kraftmessung.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "stop" ---
    # create an object to store info about Routine stop
    stop = data.Routine(
        name='stop',
        components=[apparatusLED_2],
    )
    stop.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    apparatusLED_2_led_off_sent = False
    # store start times for stop
    stop.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    stop.tStart = globalClock.getTime(format='float')
    stop.status = STARTED
    thisExp.addData('stop.started', stop.tStart)
    stop.maxDuration = None
    # keep track of which components have finished
    stopComponents = stop.components
    for thisComponent in stop.components:
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
    
    # --- Run Routine "stop" ---
    thisExp.currentRoutine = stop
    stop.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # if apparatusLED_2 is starting this frame...
        if apparatusLED_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            apparatusLED_2.frameNStart = frameN  # exact frame index
            apparatusLED_2.tStart = t  # local t and not account for scr refresh
            apparatusLED_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(apparatusLED_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('apparatusLED_2.started', t)
            # update status
            apparatusLED_2.status = STARTED
            apparatusLED_2.setLights("all", 'green')
        
        # if apparatusLED_2 is active this frame...
        if apparatusLED_2.status == STARTED:
            # update params
            pass
            
        
        # if apparatusLED_2 is stopping this frame...
        if apparatusLED_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > apparatusLED_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                apparatusLED_2.tStop = t  # not accounting for scr refresh
                apparatusLED_2.tStopRefresh = tThisFlipGlobal  # on global time
                apparatusLED_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.addData('apparatusLED_2.stopped', t)
                # update status
                apparatusLED_2.status = FINISHED
                if True:
                   apparatusLED_2.turnOffLights("all")
                   apparatusLED_2_led_off_sent = True
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=stop,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            stop.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if stop.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in stop.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "stop" ---
    for thisComponent in stop.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for stop
    stop.tStop = globalClock.getTime(format='float')
    stop.tStopRefresh = tThisFlipGlobal
    thisExp.addData('stop.stopped', stop.tStop)
    if False and not apparatusLED_2_led_off_sent:
        apparatusLED_2.turnOffLights("all")
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if stop.maxDurationReached:
        routineTimer.addTime(-stop.maxDuration)
    elif stop.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    if True and apparatusForce_raw_buffer:
        _raw_path = thisExp.dataFileName + '_force_long.tsv'
        _write_header = not os.path.exists(_raw_path)
        with open(_raw_path, 'a', encoding='utf-8') as _f:
            if _write_header:
                _f.write('participant\tsession\troutine\tcomponent\ttrial_index\ttrial_name\tidentifier\tsample_index\twhite_time\tblue_time\ttime\twhite_force\tblue_force\twhite_force_raw_counts\tblue_force_raw_counts\n')
            for _row in apparatusForce_raw_buffer:
                _f.write('\t'.join(str(_row[_k]) for _k in ['participant','session','routine','component','trial_index','trial_name','identifier','sample_index','white_time','blue_time','time','white_force','blue_force','white_force_raw_counts','blue_force_raw_counts']) + '\n')
    # Run 'End Experiment' code from code
    force = []
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
