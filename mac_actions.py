from subprocess import Popen, PIPE, run


def open_app(app: str):
    try:
        Popen(
            "open /Applications/" + app + ".app", shell=True, errors=True
        ).check_output()
    except:
        Popen("open /System/Applications/" + app + ".app", shell=True)


def open_app_fullscreen(app: str):
    s = f"""
        tell application "System Events"
          set appRunning to (name of processes) contains "{app}"
        end tell

        
        tell application "{app}"
          if not appRunning then
            make new document
          end if
          
          activate
          reopen
          
          tell application "System Events" to tell process "{app}"
            set value of attribute "AXFullScreen" of window 1 to true
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
    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell
    
    if frontApp is "Spotify" then
        tell application "Spotify" to playpause
    else if frontApp is "QuickTime Player" then
        tell application "QuickTime Player" to tell document 1 to play
    else if frontApp is "VLC" then
        tell application "VLC" to play
    else
        tell application "System Events" to key code 49 -- Simulates pressing the spacebar
    end if
    """
    run(["osascript", "-e", script])


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