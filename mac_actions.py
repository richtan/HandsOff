from subprocess import Popen, PIPE, run


def open_app(app: str):
    try:
        Popen("open /Applications/" + app + ".app", shell=True, errors=True).check_output()
    except:
        Popen("open /System/Applications/" + app + ".app", shell=True)


def open_app_fullscreen(app: str):
    s = f"""
        tell application "System Events"
	    set appRunning to (name of processes) contains "{app}"
        end tell

        tell application "Safari"
            if not appRunning then
                activate
                delay 0.5
            end if
            
            reopen
            activate 
            
            tell application "System Events"
                tell process "{app}"
                    try
                        set value of attribute "AXFullScreen" of window 1 to true
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


def minimize_front_window():
    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell
    tell application frontApp
    try
        set miniaturized of window 1 to true
    on error
        display dialog "This app does not support minimizing via AppleScript."
    end try
    end tell
    """
    run(["osascript", "-e", script])


def exit_window():
    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell

    tell application frontApp to quit
    """
    run(["osascript", "-e", script])
