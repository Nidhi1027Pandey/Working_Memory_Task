# -*- coding: utf-8 -*-
"""
Lateralized Sternberg Working-Memory Task
==========================================
Fixation -> Encoding (quadrant-distributed letters) -> Maintenance -> Probe
-> Response (response pad / keyboard, QUADRANT 1-4) -> Feedback (smiley/frown)

One combined block of 50 trials: 10 trials each of 5 difficulty levels
(interleaved by default -- see INTERLEAVE_CONDITIONS). Trial conditions
are fully editable in TRIAL_CONDITIONS below.

Hardware integrations (StimTracker, Riponda response pad, Pupil Core eye
tracker, optional LSL) are each kept in their own clearly-marked,
flag-gated block so you can disable them by flipping the USE_* flags, or
by commenting out the whole marked section if you want it fully gone
while debugging.

*** HARDWARE PIPELINE (post-port) ***
- Both Cedrus devices (StimTracker Quad + Riponda pad) are discovered in
  ONE pyxid2 scan and told apart by product_id (StimTracker=b'S',
  Riponda=b'5') -- no more "devices[0]" guessing.
- The response pad is now read through pyxid2, so RT comes from the
  Cedrus device's OWN hardware timer (r['time']), not the host clock.
- A new "response_onset" marker fires when the response screen appears,
  distinct from "response" at the button press.
- An OPTIONAL LSL marker outlet (USE_LSL_MARKERS) fans every marker to an
  LSL stream for LabRecorder / eye-tracker sync into XDF. Off by default.

*** RESPONSE PAD MAPPING: MUST RE-CONFIRM ***
The pad is now on pyxid2, whose per-button 'key' numbers are NOT
guaranteed to be 1/2/3/4. Run  `python <this_file>.py --capture`  press
each button, and fill in PAD_KEY_TO_QUADRANT below before real data.
"""

from psychopy import visual, core, event, gui
import random
import math
import string
import csv
import os
from datetime import datetime
import pylsl

# pyxid2 is required for the Cedrus devices. Guarded so the task can still
# run keyboard-only on a machine without it (device scan just returns None).
try:
    import pyxid2
except ImportError:
    pyxid2 = None
    print("pyxid2 not installed -- StimTracker + response pad disabled, "
          "keyboard (1-4) + CSV-only markers will be used.")


#############################
# EXPERIMENT SETTINGS
#############################

# ----------------------------------------------------------------------
# TRIAL CONDITIONS -- fully mutable. Add/remove/edit conditions freely;
# everything downstream (trial generation, CSV logging) reads from this.
# ----------------------------------------------------------------------
TRIAL_CONDITIONS = [
    {"level": 1, "n_letters": 2, "delay": 2.0, "n_trials": 30},
    {"level": 2, "n_letters": 4, "delay": 2.0, "n_trials": 30},
    {"level": 3, "n_letters": 6, "delay": 2.0, "n_trials": 30},
    {"level": 4, "n_letters": 8, "delay": 2.0, "n_trials": 30},
    {"level": 5, "n_letters": 8, "delay": 5.0, "n_trials": 30},
]

# True = shuffle all conditions' trials together into one interleaved
# block (matches "one block of 50 trials"). False = present as 5
# consecutive sub-blocks, one per condition, in random order.
INTERLEAVE_CONDITIONS = True

FIXATION_DURATION = 1.0
ENCODING_DURATION = 1.3
PROBE_DURATION = 1.0
RESPONSE_DURATION = 2.0
FEEDBACK_DURATION = 0.30

BREAK_EVERY_N_TRIALS = 50   # self-paced break frequency; None to disable

# ----------------------------------------------------------------------
# FEATURE FLAGS
# ----------------------------------------------------------------------
USE_APRILTAG = True
USE_STIMTRACKER = True
USE_RESPONSE_PAD = True      # Cedrus Riponda via pyxid2 (keyboard 1-4 always works too)
USE_EYETRACKER = False       
USE_LSL_MARKERS = True      

#APRILTAG_PATH = "C:/Users/hprai/OneDrive/Desktop/Python/Healtcare/Neural/Working memory task/apriltag.png"
APRILTAG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "apriltag.png"
)
# ----------------------------------------------------------------------
# CEDRUS XID DEVICE IDENTIFICATION
# Both the StimTracker and the Riponda enumerate as XID devices; we tell
# them apart by product_id instead of grabbing devices[0]. Confirm these
# against your own hardware with a diagnose scan if a device isn't found.
# ----------------------------------------------------------------------
RIPONDA_PRODUCT_ID = b'5'
STIMTRACKER_PRODUCT_ID = b'S'

# Pad button 'key' number (from pyxid2 get_next_response) -> quadrant (1-4).
# !! CONFIRM with capture_response_pad_codes()  (python <file>.py --capture)
#    before real data -- pyxid2 key numbers depend on your pad's layout and
#    are NOT guaranteed to be 1/2/3/4. Unmapped presses simply fall through
#    to the keyboard, so a wrong map won't crash, it just won't register. !!
PAD_KEY_TO_QUADRANT = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Quadrant -> keyboard key (always active, hardware-independent fallback)
QUADRANT_KEYS = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}


#############################
# POSITION GENERATION -- fixed reference-image coordinates
#############################
# Each trial randomly picks ONE of two pre-measured position sets (from
# your reference images), then uses n_letters/2 of that set's 4 fixed
# slots on each side. Positions are fixed points, not randomly sampled
# within a region -- so there's no MIN_DISTANCE/FIXATION_RADIUS geometry
# to tune here; the reference-image coordinates already keep letters
# well clear of the central crosshair and of each other.


POSITION_SETS = {
    1: {  # Random Set 1
        'left':  [(-0.390, 0.275), (-0.225, 0.215), (-0.205, -0.190), (-0.300, -0.255)],
        'right': [(0.390, 0.255), (0.260, 0.085), (0.335, -0.115), (0.235, -0.285)],
    },

    2: {  # Random Set 2
        'left':  [(-0.400, 0.275), (-0.295, 0.169), (-0.220, -0.148), (-0.390, -0.255)],
        'right': [(0.355, 0.285), (0.240, 0.095), (0.390, -0.085), (0.275, -0.265)],
    },
}


EXCLUDED_LETTERS = {'I', 'O'}      # visually confusable letters, excluded
LETTER_POOL = [c for c in string.ascii_uppercase if c not in EXCLUDED_LETTERS]

QUADRANT_LABEL_POS = {
    2: (-0.35, 0.2),   # top-left
    3: (0.35, 0.2),    # top-right
    1: (-0.35, -0.2),  # bottom-left
    4: (0.35, -0.2),   # bottom-right
}


def quadrant_of(pos):
    """Derive quadrant number (1-4) from an (x, y) position, based on sign.
    1 = top-left, 2 = top-right, 3 = bottom-left, 4 = bottom-right."""
    x, y = pos
    if x < 0 and y > 0:
        return 2
    elif x > 0 and y > 0:
        return 3
    elif x < 0 and y < 0:
        return 1
    else:
        return 4


def generate_memory_positions(n_letters):
    """Picks a random position set (1 or 2), then samples n_letters/2
    fixed slots from each side of that set. Returns (position_quadrant_pairs,
    random_set_id) -- random_set_id is logged per trial so you can see
    which reference layout was used."""
    n_per_side = n_letters // 2
    random_set_id = random.choice([1, 2])
    left_slots = random.sample(POSITION_SETS[random_set_id]['left'], n_per_side)
    right_slots = random.sample(POSITION_SETS[random_set_id]['right'], n_per_side)

    position_quadrant_pairs = [(pos, quadrant_of(pos)) for pos in left_slots + right_slots]
    return position_quadrant_pairs, random_set_id


#############################
# TRIAL GENERATOR
#############################

def build_trial(n_letters, delay, level):
    """Creates one randomized memory trial."""
    position_quadrant_pairs, random_set_id = generate_memory_positions(n_letters)
    letters = random.sample(LETTER_POOL, n_letters)

    memory_items = []
    for letter, (pos, quadrant) in zip(letters, position_quadrant_pairs):
        memory_items.append({
            "letter": letter,
            "position": pos,
            "quadrant": quadrant,
        })

    random.shuffle(memory_items)  # shuffle draw/probe-eligibility order

    probe_item = random.choice(memory_items)

    trial = {
        "level": level,
        "n_letters": n_letters,
        "maintenance_duration": delay,
        "memory_items": memory_items,
        "probe_letter": probe_item["letter"],
        "correct_quadrant": probe_item["quadrant"],
    }
    return trial


def build_all_trials():
    """Generate all trials across every condition in TRIAL_CONDITIONS."""
    all_trials = []

    for condition in TRIAL_CONDITIONS:
        condition_trials = [
            build_trial(condition["n_letters"], condition["delay"], condition["level"])
            for _ in range(condition["n_trials"])
        ]

        if INTERLEAVE_CONDITIONS:
            all_trials.extend(condition_trials)
        else:
            random.shuffle(condition_trials)
            all_trials.append(condition_trials)  # sub-block, kept together

    if INTERLEAVE_CONDITIONS:
        random.shuffle(all_trials)
        return [all_trials]  # one combined block
    else:
        random.shuffle(all_trials)  # randomize sub-block order
        return all_trials


#############################
# EVENT MARKER SYSTEM
#############################

MARKERS = {
    "experiment_start": 101,
    "instructions": 102,

    "trial_start": 110,

    "fixation": 120,

    "encoding": 130,
   
    "retention_start_2500": 140,     # 2.5 s maintenance
    "retention_start_5000": 141,     # 5.0 s maintenance

    "probe": 150,

    "response_onset": 160,   # response screen appears (distinct from the press below)
    "response": 161,

    "feedback_correct": 170,
    "feedback_wrong": 171,

    "trial_end": 190,
    "experiment_end": 109,
}


RETENTION_MARKERS = {
    2.0: "retention_start_2500",
    5.0: "retention_start_5000",
}


class EventLogger:
    """Handles sending + logging event markers.

    send() prints, logs to events.csv, fires a StimTracker TTL pulse (if
    connected), writes the marker into the ioHub HDF5 message stream (if
    Pupil Core/ioHub is connected), AND pushes it onto an LSL outlet (if
    enabled). This is the ONLY function that needs to change to add/remove
    a hardware integration -- every call site elsewhere is untouched.
    """

    def __init__(self, filepath, stim_device=None, io=None, lsl_outlet=None):
        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "marker_code", "event_name"])
        self.clock = core.Clock()
        self.stim_device = stim_device   # pyxid2 StimTracker, or None
        self.io = io                     # ioHub connection, or None
        self.lsl_outlet = lsl_outlet     # pylsl StreamOutlet, or None

    def send(self, event_name):
        marker_code = MARKERS[event_name]
        timestamp = self.clock.getTime()

        print(f"Marker sent: {marker_code} ({event_name}) at {timestamp:.4f}s")

        if self.stim_device is not None:
            try:
                self.stim_device.activate_line(bitmask=marker_code)
            except Exception as e:
                print(f"StimTracker: failed to send marker {marker_code} ({e})")

        if self.io is not None:
            try:
                self.io.sendMessageEvent(text=f"{event_name}_{marker_code}", category=event_name)
            except Exception as e:
                print(f"ioHub: failed to send marker {marker_code} ({e})")

        if self.lsl_outlet is not None:
            try:
                self.lsl_outlet.push_sample([int(marker_code)])
            except Exception as e:
                print(f"LSL: failed to push marker {marker_code} ({e})")

        self.writer.writerow([f"{timestamp:.4f}", marker_code, event_name])
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# BEGIN CEDRUS XID DISCOVERY BLOCK -- one pyxid2 scan for BOTH devices.
# Comment out this whole block (and its call site in run_experiment) if you
# don't want any Cedrus code running at all.
# ============================================================================

STIMTRACKER_PULSE_MS = 20


def _reset_pad_rt(dev):
    """Reset the pad's reaction-time timer, tolerating pyxid2 API name
    differences (reset_rt_timer vs reset_timer across versions)."""
    for meth in ("reset_rt_timer", "reset_timer"):
        fn = getattr(dev, meth, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


def discover_xid_devices():
    """Scan the XID bus ONCE and return (pad_dev, stim_dev), each selected
    by product_id. Soft-fail: a device that isn't found comes back None
    (with a warning) so the task still runs on keyboard / CSV-only markers."""
    pad_dev = None
    stim_dev = None

    if pyxid2 is None or not (USE_RESPONSE_PAD or USE_STIMTRACKER):
        return None, None

    try:
        devices = pyxid2.get_xid_devices()
    except Exception as e:
        print(f"pyxid2: device scan failed ({e}) -- keyboard + CSV-only fallback.")
        return None, None

    if not devices:
        print("pyxid2: no XID devices found -- keyboard + CSV-only fallback.")
        return None, None

    ripondas = [d for d in devices if getattr(d, "product_id", None) == RIPONDA_PRODUCT_ID]
    stims = [d for d in devices if getattr(d, "product_id", None) == STIMTRACKER_PRODUCT_ID]

    if USE_RESPONSE_PAD:
        if not ripondas:
            print(f"Response pad: no Riponda (product_id={RIPONDA_PRODUCT_ID!r}) found "
                  "-- keyboard (1-4) fallback.")
        else:
            if len(ripondas) > 1:
                print(f"Response pad: {len(ripondas)} Ripondas found -- using the first.")
            pad_dev = ripondas[0]

    if USE_STIMTRACKER:
        if not stims:
            print(f"StimTracker: no device (product_id={STIMTRACKER_PRODUCT_ID!r}) found "
                  "-- print/CSV markers only.")
        else:
            if len(stims) > 1:
                print(f"StimTracker: {len(stims)} devices found -- using the first.")
            stim_dev = stims[0]

    if pad_dev is not None:
        print(f"Response pad selected: {pad_dev} | product_id="
              f"{getattr(pad_dev, 'product_id', None)!r}")
    if stim_dev is not None:
        print(f"StimTracker selected: {stim_dev} | product_id="
              f"{getattr(stim_dev, 'product_id', None)!r}")

    return pad_dev, stim_dev


def setup_stimtracker(stim_dev):
    """Apply pulse settings to an already-discovered StimTracker. Returns
    the device on success, or None so EventLogger falls back to CSV-only."""
    if stim_dev is None:
        return None
    try:
        stim_dev.reset_base_timer()
        stim_dev.reset_rt_timer()
        stim_dev.set_pulse_duration(STIMTRACKER_PULSE_MS)
        print(f"StimTracker configured (pulse {STIMTRACKER_PULSE_MS} ms).")
        return stim_dev
    except Exception as e:
        print(f"StimTracker: setup failed ({e}) -- print/CSV markers only.")
        return None


def setup_response_pad(pad_dev):
    """Put an already-discovered Riponda into a clean state. Returns the
    device on success, or None so responses fall back to the keyboard."""
    if pad_dev is None:
        return None
    try:
        _reset_pad_rt(pad_dev)
        pad_dev.poll_for_response()
        pad_dev.clear_response_queue()
        print("Response pad ready.")
        return pad_dev
    except Exception as e:
        print(f"Response pad: setup failed ({e}) -- keyboard (1-4) fallback.")
        return None

# ============================================================================
# END CEDRUS XID DISCOVERY BLOCK
# ============================================================================


# ============================================================================
# BEGIN LSL MARKER BLOCK -- OPTIONAL. Pushes every marker code onto an LSL
# stream so LabRecorder (or an LSL-aware eye tracker / EEG amp) can record
# them into one synchronized XDF. Comment out this whole block (and its call
# site in run_experiment) if you don't want LSL code running at all.
#
# Requires: pip install pylsl. Off by default (USE_LSL_MARKERS = False) so
# nothing breaks if pylsl isn't installed / LabRecorder isn't set up.
# ============================================================================

LSL_STREAM_NAME = "WM_Task_Markers"
LSL_STREAM_TYPE = "Markers"
LSL_SOURCE_ID = "wm_task_marker_stream"


def init_lsl():
    """Create an int32 LSL marker outlet, or return None on any failure."""
    if not USE_LSL_MARKERS:
        return None
    try:
        from pylsl import StreamInfo, StreamOutlet
        info = StreamInfo(
            name=LSL_STREAM_NAME,
            type=LSL_STREAM_TYPE,
            channel_count=1,
            nominal_srate=0,          # irregular rate -- markers are event-driven
            channel_format="int32",
            source_id=LSL_SOURCE_ID,
        )
        outlet = StreamOutlet(info)
        print(f"LSL: marker outlet '{LSL_STREAM_NAME}' created.")
        return outlet
    except ImportError:
        print("LSL: pylsl not installed -- skipping LSL markers.")
        return None
    except Exception as e:
        print(f"LSL: outlet creation failed ({e}) -- skipping LSL markers.")
        return None

# ============================================================================
# END LSL MARKER BLOCK
# ============================================================================


# ============================================================================
# BEGIN PUPIL CORE EYE TRACKER BLOCK -- comment out this whole block (and its
# call sites in run_experiment) if you don't want eye-tracking code running.
#
# Requires: Pupil Capture running with headset connected, the
# "psychopy-eyetracker-pupil-labs" plugin installed, AprilTag markers
# visible on screen (see APRILTAG_PATH / USE_APRILTAG above), and --
# done manually inside Pupil Capture itself, NOT from this script -- a
# Surface Tracker surface defined around those tags and renamed to match
# SURFACE_NAME below. Core also needs calibration; run_calibration() below
# triggers Pupil Capture's calibration choreography once, at experiment
# start.
# ============================================================================

PUPIL_REMOTE_ADDRESS = "127.0.0.1"
PUPIL_REMOTE_PORT = 50020
PUPIL_REMOTE_TIMEOUT_MS = 5000

PUPIL_CAPTURE_RECORDING = True
GAZE_CONFIDENCE_THRESHOLD = 0.6
PUPILLOMETRY_ONLY = False
SURFACE_NAME = "psychopy_iohub_surface"

HDF5_FILENAME = "gaze_data"   # ioHub appends .hdf5 automatically


def init_iohub(data_dir, win):
    """Launch ioHub with the Pupil Core eye tracker device. Returns
    (io, eyetracker), or (None, None) on failure -- in which case the
    task runs with no eye tracking. NOTE: the exact runtime_settings key
    names below are a best-effort translation of the plugin's documented
    Builder-panel fields; cross-check against your installed plugin's
    default_eyetracker_devices.yaml if connection fails on a settings
    error specifically."""
    if not USE_EYETRACKER:
        return None, None
    try:
        from psychopy.iohub import launchHubServer

        iohub_config = {
            "eyetracker.hw.pupil_labs.pupil_core.EyeTracker": {
                "name": "tracker",
                "runtime_settings": {
                    "pupil_remote": {
                        "ip_address": PUPIL_REMOTE_ADDRESS,
                        "port": PUPIL_REMOTE_PORT,
                        "timeout_ms": PUPIL_REMOTE_TIMEOUT_MS,
                    },
                    "pupil_capture_recording": {
                        "enabled": PUPIL_CAPTURE_RECORDING,
                    },
                    "gaze_confidence_threshold": GAZE_CONFIDENCE_THRESHOLD,
                    "pupillometry_only": PUPILLOMETRY_ONLY,
                    "surface_name": SURFACE_NAME,
                },
            }
        }

        io = launchHubServer(
            window=win,
            datastore_name=os.path.join(data_dir, HDF5_FILENAME),
            **iohub_config,
        )
        eyetracker = io.devices.tracker
        if eyetracker is None:
            print("Pupil Core: eye tracker device not found -- running without eye tracking.")
            io.quit()
            return None, None
        print("Pupil Core eye tracker connected via ioHub.")
        return io, eyetracker
    except Exception as e:
        print(f"Pupil Core: connection failed ({e}) -- running without eye tracking.")
        return None, None


def run_calibration(eyetracker):
    """Run Pupil Capture's calibration choreography. Blocks until
    complete. Skipped in pupillometry-only mode or if no tracker."""
    if eyetracker is None or PUPILLOMETRY_ONLY:
        return
    try:
        print("Pupil Core: starting calibration...")
        result = eyetracker.runSetupProcedure()
        print(f"Pupil Core: calibration finished (result={result}).")
    except Exception as e:
        print(f"Pupil Core: calibration failed ({e}) -- continuing without it.")

# ============================================================================
# END PUPIL CORE EYE TRACKER BLOCK
# ============================================================================


#############################
# GLOBAL ESCAPE HANDLING
#############################

def check_escape():
    if event.getKeys(keyList=["escape"]):
        core.quit()


def wait_with_escape(duration, poll_interval=0.02):
    clock = core.Clock()
    while clock.getTime() < duration:
        check_escape()
        core.wait(min(poll_interval, duration - clock.getTime()))
    check_escape()


#############################
# PARTICIPANT INFO / OUTPUT FILES
#############################

def get_participant_info():
    exp_info = {"Participant ID": "", "Session": "001",
                "Age": "", "Gender": ["Female", "Male", "Other/Prefer not to say"]}
    dlg = gui.DlgFromDict(dictionary=exp_info, title="Lateralized Sternberg WM Task")
    if not dlg.OK:
        core.quit()
    exp_info["date"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return exp_info


def make_data_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "wm_task_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


#############################
# BUILD STIMULI
#############################

def build_stimuli(win):
    background = None
    if USE_APRILTAG:
        try:
            background = visual.ImageStim(win, image=APRILTAG_PATH, units="norm", size=(2, 2))
        except Exception as e:
            print("AprilTag could not be loaded:", e)
            background = None

    fixation = visual.TextStim(win, text="+", color="white", height=0.08, units="height")

    memory_letter = visual.TextStim(win, text="", color="yellow", height=0.10, units="height")

    probe = visual.TextStim(win, text="", color="orange", height=0.15, units="height", pos=(0, 0))

    # Quadrant response screen: crosshair lines + "1","2","3","4" labels
    v_line = visual.Line(win, start=(0, -0.5), end=(0, 0.5), lineColor="grey", units="height")
    h_line = visual.Line(win, start=(-0.8, 0), end=(0.8, 0), lineColor="grey", units="height")
    quadrant_labels = {
        q: visual.TextStim(win, text=str(q), color="white", height=0.08,
                            units="height", pos=QUADRANT_LABEL_POS[q])
        for q in [1, 2, 3, 4]
    }
    response_prompt = visual.TextStim(
        win, text="Which quadrant was the letter in?\n(press 1, 2, 3, 4, or 5 if unsure)",
        color="white", height=0.045, units="height", pos=(0, 0.42), wrapWidth=1.5
    )

    correct_feedback = visual.TextStim(win, text="\u263A", color="limegreen", height=0.20, units="height")
    wrong_feedback = visual.TextStim(win, text="\u2639", color="red", height=0.20, units="height")

    break_text = visual.TextStim(win, text="", color="white", height=0.05, units="height", wrapWidth=1.5)

    return {
        "background": background,
        "fixation": fixation,
        "memory_letter": memory_letter,
        "probe": probe,
        "response_prompt": response_prompt,
        "v_line": v_line,
        "h_line": h_line,
        "quadrant_labels": quadrant_labels,
        "correct_feedback": correct_feedback,
        "wrong_feedback": wrong_feedback,
        "break_text": break_text,
    }


def draw_background(stims):
    if stims["background"] is not None:
        stims["background"].draw()


#############################
# TRIAL ROUTINES  (each phase sends its marker)
#############################

def run_fixation(win, stims, logger):
    check_escape()
    draw_background(stims)
    stims["fixation"].draw()
    logger.send("fixation")
    win.flip()
    wait_with_escape(FIXATION_DURATION)


def run_encoding(win, stims, trial, logger):
    check_escape()
    draw_background(stims)
    for item in trial["memory_items"]:
        stims["memory_letter"].text = item["letter"]
        stims["memory_letter"].pos = item["position"]
        stims["memory_letter"].draw()
    logger.send("encoding")
    win.flip()
    wait_with_escape(ENCODING_DURATION)



def run_maintenance(win, stims, trial, logger):
    check_escape()
    draw_background(stims)
    win.flip()
    delay = trial["maintenance_duration"]
    marker = RETENTION_MARKERS.get(round(delay, 3), "retention_start")
    logger.send(marker)
    wait_with_escape(delay)


def run_probe(win, stims, trial, logger):
    check_escape()
    draw_background(stims)
    stims["probe"].text = trial["probe_letter"]
    stims["probe"].draw()
    logger.send("probe")
    win.flip()
    wait_with_escape(PROBE_DURATION)


# ----------------------------------------------------------------------
# RESPONSE COLLECTION -- Riponda via pyxid2 (device-timer RT), with
# keyboard (1/2/3/4) as a hardware-independent fallback that always works.
# pad_dev is the pyxid2 device from discover_xid_devices(), or None.
# ----------------------------------------------------------------------

def get_response(win, stims, logger, pad_dev):
    check_escape()
    draw_background(stims)
    stims["response_prompt"].draw()
    stims["v_line"].draw()
    stims["h_line"].draw()
    for label in stims["quadrant_labels"].values():
        label.draw()
    win.flip()

    # Marker for the moment the response screen appears (distinct from the
    # button press). Fires across all enabled sinks via EventLogger.
    logger.send("response_onset")

    # Clear anything left in the pad queue and zero its RT timer so r['time']
    # is measured from the response-screen onset.
    if pad_dev is not None:
        try:
            pad_dev.poll_for_response()
            pad_dev.clear_response_queue()
            _reset_pad_rt(pad_dev)
        except Exception as e:
            print(f"Response pad: pre-trial reset failed ({e}) -- keyboard fallback.")

    timer = core.Clock()

    while timer.getTime() < RESPONSE_DURATION:
        check_escape()

        # ---- Response pad (pyxid2): RT comes from the device timer ----


        if pad_dev is not None:
            try:
                pad_dev.poll_for_response()
                if pad_dev.has_response():
                    r = pad_dev.get_next_response()
                    pad_dev.clear_response_queue()
                    if r["pressed"] and r["key"] in PAD_KEY_TO_QUADRANT:
                        quadrant = PAD_KEY_TO_QUADRANT[r["key"]]
                        rt = r["time"] / 1000.0   # pyxid2 reports ms -> seconds
                        logger.send("response")
                        return quadrant, rt
            except Exception as e:
                print(f"Response pad: read failed ({e}) -- keyboard fallback.")

        # ---- Keyboard fallback: RT from the PsychoPy response clock ----
        keys = event.getKeys(keyList=list(QUADRANT_KEYS.values()) + ["escape"], timeStamped=timer)
        if keys:
            key, rt = keys[0]
            if key == "escape":
                core.quit()
            for quadrant, key_code in QUADRANT_KEYS.items():
                if key == key_code:
                    logger.send("response")
                    return quadrant, rt

    return None, None  # timeout


def run_feedback(win, stims, correct, logger):
    check_escape()
    draw_background(stims)
    if correct:
        stims["correct_feedback"].draw()
    else:
        stims["wrong_feedback"].draw()
    logger.send("feedback_correct" if correct else "feedback_wrong")
    win.flip()
    wait_with_escape(FEEDBACK_DURATION)


def run_trial(win, stims, trial, logger, pad_dev):
    logger.send("trial_start")

    run_fixation(win, stims, logger)
    run_encoding(win, stims, trial, logger)
    run_maintenance(win, stims, trial, logger)
    run_probe(win, stims, trial, logger)

    response, rt = get_response(win, stims, logger, pad_dev)
    correct = response == trial["correct_quadrant"]

    run_feedback(win, stims, correct, logger)

    logger.send("trial_end")

    return {"response": response, "rt": rt, "correct": correct}


#############################
# INSTRUCTIONS / BREAKS
#############################

INSTRUCTIONS = (
    "In this task, you will briefly see a set of letters appear around the "
    "screen, split across four quadrants.\n\n"
    "Remember the letters and, importantly, WHICH QUADRANT each one appeared in.\n\n"
    "After a short delay, a single letter (the PROBE) will appear in the center "
    "of the screen.\n\n"
    "You will then press the number (1, 2, 3, or 4) corresponding to the "
    "quadrant the probe letter was originally shown in or press 5 if unsure.\n\n"
    "Respond as quickly and accurately as you can.\n\n"
    "Press any button to begin."
)


def show_instructions(win, logger, pad_dev=None):
    logger.send("instructions")
    stim = visual.TextStim(win, text=INSTRUCTIONS, color="white", height=0.045, wrapWidth=1.5)
    while True:
        check_escape()
        stim.draw()
        win.flip()

        if pad_dev is not None:
            pad_dev.poll_for_response()

        if pad_dev is not None and pad_dev.has_response():
            r = pad_dev.get_next_response()
            pad_dev.clear_response_queue()

            if r["pressed"]:
                break

    # Keyboard fallback
        if event.getKeys(keyList=["space"]):
            break

    core.wait(0.01)

   



def show_break(win, current, total, pad_dev):
    txt = visual.TextStim(
        win,
        text=f"Completed {current}/{total} trials.\n\nTake a short break.\n\nPress any button to continue.",
        color="white", height=0.05
    )
    while True:
            check_escape()
            txt.draw()
            win.flip()
    
            if pad_dev is not None:
                pad_dev.poll_for_response()
    
            if pad_dev is not None and pad_dev.has_response():
                r = pad_dev.get_next_response()
                pad_dev.clear_response_queue()
    
                if r["pressed"]:
                    break
    
        # Keyboard fallback
            if event.getKeys(keyList=["space"]):
                break
    
    core.wait(0.01)


#############################
# OUTPUT FILE
#############################

def create_output_file(data_dir, exp_info):
    filename = os.path.join(
        data_dir, f"response_{exp_info['Participant ID']}_sess{exp_info['Session']}_{exp_info['date']}.csv"
    )
    f = open(filename, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "participant", "session", "trial", "level", "n_letters", "delay",
        "memory_letters", "memory_quadrants",
        "probe", "correct_quadrant",
        "response", "accuracy", "rt",
    ])
    return f, writer


#############################
# MAIN EXPERIMENT
#############################

def run_experiment(win, exp_info, data_dir):
    stims = build_stimuli(win)

    events_filename = os.path.join(
        data_dir, f"events_{exp_info['Participant ID']}_sess{exp_info['Session']}_{exp_info['date']}.csv"
    )

    # --- Cedrus devices: ONE scan, both selected by product_id ---
    # (comment out these three lines + the discovery block above to fully disable)
    pad_dev, stim_dev = discover_xid_devices()
    stim_device = setup_stimtracker(stim_dev)
    pad_dev = setup_response_pad(pad_dev)

    # --- LSL marker outlet (comment out this line + the LSL block above to disable) ---
    lsl_outlet = init_lsl()

    # --- Pupil Core eye tracker (comment out these two lines + block above to disable) ---
    io, eyetracker = init_iohub(data_dir, win)

    logger = EventLogger(events_filename, stim_device=stim_device, io=io, lsl_outlet=lsl_outlet)
    outfile, writer = create_output_file(data_dir, exp_info)

    show_instructions(win, logger, pad_dev)

    run_calibration(eyetracker)   # no-op if eyetracker is None or pupillometry-only

    if eyetracker is not None:
        state = eyetracker.setRecordingState(True)
        print(f"Pupil Core: setRecordingState(True) -> recording state = {state}")

    logger.send("experiment_start")

    trial_blocks = build_all_trials()
    total_trials = sum(len(block) for block in trial_blocks)

    trial_counter = 0
    for block in trial_blocks:
        for trial in block:
            trial_counter += 1

            result = run_trial(win, stims, trial, logger, pad_dev)

            writer.writerow([
                exp_info["Participant ID"], exp_info["Session"], trial_counter,
                trial["level"], trial["n_letters"], trial["maintenance_duration"],
                "".join(it["letter"] for it in trial["memory_items"]),
                "".join(str(it["quadrant"]) for it in trial["memory_items"]),
                trial["probe_letter"], trial["correct_quadrant"],
                result["response"], result["correct"], result["rt"],
            ])
            outfile.flush()

            if BREAK_EVERY_N_TRIALS and trial_counter % BREAK_EVERY_N_TRIALS == 0 \
                    and trial_counter != total_trials:
                show_break(win, trial_counter, total_trials)

    logger.send("experiment_end")

    if eyetracker is not None:
        state = eyetracker.setRecordingState(False)
        print(f"Pupil Core: setRecordingState(False) -> recording state = {state}")
    if io is not None:
        io.quit()   # flushes and finalizes the HDF5 file

    outfile.close()
    logger.close()


def main():
    exp_info = get_participant_info()
    data_dir = make_data_dir()

    win = visual.Window(fullscr=True, color="black", units="height")

    run_experiment(win, exp_info, data_dir)

    thanks = visual.TextStim(win, text="Thank you!", color="white", height=0.07)
    thanks.draw()
    win.flip()
    wait_with_escape(2.0)

    win.close()
    core.quit()


#############################
# RESPONSE PAD CODE CAPTURE UTILITY  (pyxid2)
# Run this standalone (python this_file.py --capture) to print the pyxid2
# 'key' number for each button press, so you can confirm/correct
# PAD_KEY_TO_QUADRANT above before real data collection.
#############################

def capture_response_pad_codes():
    if pyxid2 is None:
        print("pyxid2 not installed -- cannot capture pad codes.")
        return
    try:
        devices = pyxid2.get_xid_devices()
    except Exception as e:
        print(f"pyxid2 scan failed: {e}")
        return

    ripondas = [d for d in devices if getattr(d, "product_id", None) == RIPONDA_PRODUCT_ID]
    if not ripondas:
        print(f"No Riponda (product_id={RIPONDA_PRODUCT_ID!r}) found. "
              f"Devices seen: {devices}")
        return

    pad = ripondas[0]
    _reset_pad_rt(pad)
    pad.poll_for_response()
    pad.clear_response_queue()

    print("Press each response-pad button one at a time. Ctrl+C to stop.")
    print("Note the 'key' number for each physical button, then set "
          "PAD_KEY_TO_QUADRANT so each quadrant maps to the right button.\n")
    try:
        while True:
            pad.poll_for_response()
            if pad.has_response():
                r = pad.get_next_response()
                print(f"key={r['key']}  pressed={r['pressed']}  "
                      f"port={r.get('port')}  time={r['time']} ms")
                pad.clear_response_queue()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    import sys
    if "--capture" in sys.argv:
        capture_response_pad_codes()
    else:
        main()
