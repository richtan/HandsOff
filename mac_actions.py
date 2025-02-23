from subprocess import Popen, PIPE, run
import Quartz


def open_app(app: str):
    try:
        Popen("open /Applications/" + app + ".app", shell=True, errors=True).check_output()
    except:
        Popen("open /System/Applications/" + app + ".app", shell=True)


def open_app_fullscreen(app: str, AXFullScreen=False):
    s = f"""
        tell application "System Events"
	    set appRunning to (name of processes) contains "{app}"
        end tell

        tell application "{app}"
            if not appRunning then
                activate
                delay 0.5
            end if
            
            reopen
            activate 
            
            tell application "System Events"
                tell process "{app}"
                    try
                        {AXFullScreen and 'set value of attribute "AXFullScreen" of window 1 to true'}
                        {not AXFullScreen and 'set position of window 1 to {0, 0}'}
                        {not AXFullScreen and 'set size of window 1 to {4000, 4000}'} 
                    on error
                        perform action "AXRaise" of window 1
                    end try
                end tell
            end tell
        end tell
        """
    # set value of attribute "AXFullScreen" of window 1 to true
    # set position of window 1 to {0, 0}
    # set size of window 1 to {4000, 4000}
    Popen(["osascript", "-e", s])


def go_to_last_applilcation():
    script = f"""
    tell application "System Events"
        key code 48 using command down  -- (Tab + Command)
    end tell
    """
    run(["osascript", "-e", script])


def switch_desktop_left():
    script = f"""
    tell application "System Events"
        key code 123 using control down  -- Left arrow (Ctrl + Left)
    end tell
    """
    run(["osascript", "-e", script])


def switch_desktop_right():
    script = f"""
    tell application "System Events"
        key code 124 using control down  -- Right arrow (Ctrl + Right)
    end tell
    """
    run(["osascript", "-e", script])


def switch_desktop(dir: bool, count=1):
    script = f"""
    tell application "System Events"
        key code 123 using control down  -- Left arrow (Ctrl + Left)
    end tell
    """

    if dir == "right":
        script = f"""
        tell application "System Events"
            key code 124 using control down  -- Right arrow (Ctrl + Right)
        end tell
        """

    script = script * count

    run(["osascript", "-e", script])


def toggle_play_pause_music():
    s = '''
    on is_running(appName)
        tell application "System Events" to (name of processes) contains appName
    end is_running

    if is_running("Music") then
        tell application "Music"
            playpause
        end tell

    else if is_running("Spotify") then
        tell application "Spotify"
            playpause
        end tell
    end if
    '''
    run(["osascript", "-e", s])


def toggle_play_pause_on_app(delay=0.5):
    script = f"""
    delay {delay}
    tell application "System Events"
        key code 49 -- (Space)
    end tell
    """
    run(["osascript", "-e", script])


def volume_up(amount=6.25):
    s = f"""set theOutput to output volume of (get volume settings)
        if not (theOutput = 100) then
            set newVolume to theOutput + {amount}
            if newVolume > 100 then set newVolume to 100
            set volume output volume newVolume
            display notification "Volume set to " & newVolume & "%" with title "Volume Up"
        end if
        """

    Popen(["osascript", "-e", s])


def volume_down(amount=6.25):
    s = f"""set theOutput to output volume of (get volume settings)
        if not (theOutput = 0) then
            set newVolume to theOutput - {amount}
            if newVolume < 0 then set newVolume to 0
            set volume output volume newVolume
            display notification "Volume set to " & newVolume & "%" with title "Volume Down"
        end if
        """

    Popen(["osascript", "-e", s])


def exit_window():
    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell

    tell application frontApp to quit
    """
    run(["osascript", "-e", script])


def minimize_front_window(includeEsc=False):
    s = f'''
    tell application "System Events"
        key code 46 using command down  -- (Cmd + M)
    end tell
    '''
    run(["osascript", "-e", s])


# NSEvent.h
NSSystemDefined = 14

# hidsystem/ev_keymap.h
NX_KEYTYPE_SOUND_UP = 0
NX_KEYTYPE_SOUND_DOWN = 1
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18
NX_KEYTYPE_FAST = 19
NX_KEYTYPE_REWIND = 20


def HIDPostAuxKey(key):
    def doKey(down):
        ev = Quartz.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            NSSystemDefined,  # type
            (0, 0),  # location
            0xA00 if down else 0xB00,  # flags
            0,  # timestamp
            0,  # window
            0,  # ctx
            8,  # subtype
            (key << 16) | ((0xA if down else 0xB) << 8),  # data1
            -1,  # data2
        )
        cev = ev.CGEvent()
        Quartz.CGEventPost(0, cev)

    doKey(True)
    doKey(False)


def q_play_pause():
    HIDPostAuxKey(NX_KEYTYPE_PLAY)


def q_sound_up():
    HIDPostAuxKey(NX_KEYTYPE_SOUND_UP)


def q_sound_down():
    HIDPostAuxKey(NX_KEYTYPE_SOUND_DOWN)
