import os
import time
import csv
import html
from collections import Counter, deque
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import joblib
import serial
import streamlit as st
from tensorflow.keras.models import load_model

# =========================================================
# SETTINGS
# =========================================================

PORT = "COM3"
BAUD_RATE = 115200

EXPECTED_WINDOW_SIZE = 91
STEP_SIZE = 35

# Save every raw IMU sample during a monitoring run.
READING_LOG_FILE = "maintenaction_readings.csv"

# Save one diagnostic row for every prediction window.
WINDOW_LOG_FILE = "maintenaction_windows.csv"

# You can keep using your latest model/preprocessing files.
# IMPORTANT:
# - If the model still contains an "idle" output, this app IGNORES that output.
# - Idle is decided ONLY by the rule below.
MODEL_PATH = "best_cnn_lstm_accuracy12.keras"
PREPROCESSING_PATH = "imu_preprocessing12.npz"

# Dedicated sequence model for the three between-wrench moves.
# Place this .joblib file in the same folder as the Streamlit app.
CROSS_ADJ_MODEL_PATH = "cac_aca_logistic_model.joblib"

# =========================================================
# SMOOTHING / CONFIRMATION
# =========================================================

VOTE_WINDOW = 3

# Insert bolt uses its own stricter consecutive-window confirmation.
# All other classes still use the normal VOTE_WINDOW above.
INSERT_BOLT_CONFIRMATION_WINDOWS = 4

# After enough votes agree on a class, their AVERAGE confidence
# must be at least this value before the action is confirmed.
MIN_COMBINED_CONFIDENCE = 0.70

CONFIRMATION_REQUIRED = {
    'clean_inspect': 2,
    'place_align_gasket': 2,
    'insert_bolt_nut': 4,
    'hand_tighten': 3,
    'wrench_tighten': 3,
    'lubrication': 2,
}

EVENT_COOLDOWN_SEC = {
    'clean_inspect': 3.0,
    'place_align_gasket': 3.0,
    'insert_bolt_nut': 3.0,
    'hand_tighten': 2.0,
    'wrench_tighten': 3.0,
    'lubrication': 3.0,
}

REQUIRE_RELEASE_FOR_REPEAT = True

# =========================================================
# RULE-BASED IDLE
# =========================================================

IDLE_ENTER_GYRO_P25_THRESHOLD = 350.0
IDLE_EXIT_GYRO_P25_THRESHOLD = 2300.0
IDLE_EXIT_CONSECUTIVE_WINDOWS = 2

# =========================================================
# PROCEDURE DEFINITION
# =========================================================

ACTION_DISPLAY = {
    'clean_inspect': 'Cleaning / inspection',
    'place_align_gasket': 'Place / align gasket',
    'insert_bolt_nut': 'Insert bolt + nut',
    'hand_tighten': 'Hand tighten',
    'wrench_tighten': 'Wrench tighten',
    'lubrication': 'Lubrication',
    'idle': 'Idle',
}

# =========================================================
# WRENCH RE-ARM + TRANSITION CAPTURE
# =========================================================

GYRO_SCALE = 131.0
SAMPLE_RATE_HZ = 45.45

# Wrench re-arm:
# Learn the dominant wrist-rotation axis from the confirmed wrench window.
# Then inspect the most recent ~1 second of motion.
# Re-arm only when there is substantial movement AND enough of that
# movement occurs perpendicular to the learned tightening axis.
WRENCH_REARM_WINDOW_SAMPLES = max(1, int(round(SAMPLE_RATE_HZ * 1.0)))
WRENCH_REARM_MIN_TOTAL_ROTATION_DEG = 45.0
WRENCH_REARM_MIN_PERPENDICULAR_FRACTION = 0.30
WRENCH_REARM_MAX_SAMPLES = WRENCH_REARM_WINDOW_SAMPLES

WRENCH_REARM_REFERENCE_SAMPLES = 12

# Wider transition evidence buffer for inertial trajectory reconstruction.
# Re-arm detection itself is unchanged; this only preserves more samples
# around the detected move so the relocation path can be reconstructed.
WRENCH_REPOSITION_PREBUFFER_SAMPLES = 90
WRENCH_MAX_TRANSITION_SAMPLES = int(SAMPLE_RATE_HZ * 8.0)

# After reposition is detected, capture a short fixed amount of
# additional movement. Cross/adj classification no longer waits
# for the next wrench prediction.
WRENCH_TRANSITION_POST_SAMPLES = 90

# =========================================================
# LUBRICATION STROKE COUNTING
# =========================================================

LUBRICATION_PREBUFFER_SAMPLES = int(SAMPLE_RATE_HZ * 4.0)
LUBRICATION_MAX_SAMPLES = int(SAMPLE_RATE_HZ * 30.0)
LUBRICATION_END_NONMATCH_WINDOWS = 2

# Lubrication stroke detector calibrated from the improved pump-count
# recordings. The detector looks for repeated short acceleration pulses,
# rejects broad high-gyro bottle-handling movements, and re-checks broad
# low-gyro regions at a finer time scale so fast pumps are not merged.
LUB_GRAVITY_SMOOTH_SAMPLES = 29
LUB_COARSE_ENVELOPE_SMOOTH_SAMPLES = 12
LUB_FINE_ENVELOPE_SMOOTH_SAMPLES = 6

LUB_PEAK_HEIGHT_PERCENTILE = 70.0
LUB_PEAK_PROMINENCE_RATIO = 0.25
LUB_MIN_PEAK_DISTANCE_SAMPLES = 12

LUB_MIN_PEAK_WIDTH_SAMPLES = 3
LUB_MAX_NORMAL_PEAK_WIDTH_SAMPLES = 26

LUB_GYRO_CONTEXT_SAMPLES = 8
LUB_BOTTLE_GYRO_P95_THRESHOLD = 12000.0
LUB_BOTTLE_EXCLUSION_SAMPLES = int(round(SAMPLE_RATE_HZ * 0.80))

LUB_FINE_MIN_PEAK_DISTANCE_SAMPLES = 10
LUB_FINE_PEAK_PROMINENCE_RATIO = 0.15

LUB_PUMP_BOUT_MAX_GAP_SAMPLES = int(round(SAMPLE_RATE_HZ * 2.30))

LUB_WEAK_TAIL_RATIO = 0.45
LUB_WEAK_TAIL_MIN_PRIOR_PUMPS = 3

# Leading handling cleanup.
# When the extended prebuffer includes bottle pickup / hand positioning,
# reject an obvious leading handling pulse before the repeated pump rhythm.
LUB_LEADING_GYRO_RATIO = 1.80
LUB_LEADING_WIDTH_RATIO = 0.75
LUB_LEADING_STRENGTH_RATIO = 0.85

# Adaptive cadence-tail protection.
# After 3 established pumps, a much later candidate is treated as
# post-pump handling rather than another stroke.
LUB_CADENCE_MIN_PRIOR_PUMPS = 3
LUB_CADENCE_MAX_GAP_RATIO = 1.80

# =========================================================
# PAGE
# =========================================================

st.set_page_config(page_title='MaintenAction', layout='wide', initial_sidebar_state='expanded')

st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --navy-0:#01070d;
    --navy-1:#020d18;
    --navy-2:#04182b;
    --navy-3:#06233d;
    --panel:#041a2e;
    --panel-2:#06243e;
    --panel-3:#082b49;
    --blue:#008cff;
    --blue-2:#18a8ff;
    --cyan:#63d6ff;
    --ice:#dff5ff;
    --white:#f2f8fb;
    --muted:#8ba7b7;
    --muted-2:#5d7a8d;
    --steel:#64798a;
    --steel-2:#33495a;
    --steel-3:#172b3a;
    --green:#41e18a;
    --red:#ff6573;
    --amber:#ffaf16;
}

html, body, [class*="css"] {
    font-family:"Inter","Segoe UI",sans-serif;
}

html, body, .stApp {
    background:#01070d !important;
}

.stApp {
    background:
      radial-gradient(circle at 58% -12%, rgba(0,126,255,.16), transparent 33%),
      linear-gradient(rgba(43,127,184,.030) 1px, transparent 1px),
      linear-gradient(90deg, rgba(43,127,184,.024) 1px, transparent 1px),
      linear-gradient(180deg,#03101c 0%,#01070d 100%) !important;
    background-size:auto, 38px 38px,38px 38px,auto !important;
    color:var(--white);
}

/* Remove Streamlit chrome. */
header[data-testid="stHeader"] {height:0!important;min-height:0!important;background:transparent!important;}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none!important;}
[data-testid="stSidebarCollapseButton"] {display:none!important;}

.block-container {
    max-width:none!important;
    width:100%!important;
    padding:10px 12px 28px 10px!important;
    margin:0!important;
}

/* Main chassis edge. */
[data-testid="stMain"] {
    background:
      linear-gradient(90deg, rgba(91,119,139,.18), transparent 6px),
      transparent!important;
    border-top:1px solid #728899;
    border-bottom:1px solid #253b4b;
    box-shadow:inset 0 2px 0 #101f2b, inset 0 -2px 0 #000, 0 0 30px rgba(0,0,0,.55);
}

h1,h2,h3,h4,p,label {color:var(--white)!important;}

/* =========================================================
   SIDEBAR — BLUE COMMAND
   ========================================================= */
section[data-testid="stSidebar"] {
    width:300px!important;
    min-width:300px!important;
    background:
      linear-gradient(180deg,rgba(11,39,63,.98),rgba(2,12,22,.995))!important;
    border-right:3px solid #1d3749!important;
    box-shadow:
      inset -1px 0 0 #6a7d8b,
      inset -5px 0 0 #07131d,
      10px 0 25px rgba(0,0,0,.42)!important;
}
section[data-testid="stSidebar"] > div {padding:0!important;}

.ma-sidebar-shell {
    min-height:100vh;
    position:relative;
    padding:14px 13px 18px;
    background:
      radial-gradient(circle at 16px 16px,#9fb0ba 0 2px,#111c24 2.3px 4px,transparent 4.3px),
      radial-gradient(circle at calc(100% - 16px) 16px,#9fb0ba 0 2px,#111c24 2.3px 4px,transparent 4.3px),
      linear-gradient(180deg,rgba(12,48,78,.74),rgba(2,14,25,.92));
    box-shadow:
      inset 0 0 0 1px #567083,
      inset 0 0 0 5px #06121c,
      inset 0 0 0 7px #1c3547;
}
.ma-command-tab {
    margin:-3px 7px 12px;
    height:36px;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:7px;
    color:#d8e7ef;
    font-family:"Barlow Condensed",sans-serif;
    font-size:1.05rem;
    letter-spacing:.05em;
    font-weight:700;
    text-transform:uppercase;
    background:linear-gradient(180deg,#293e4d,#102433 54%,#0a1823);
    border:1px solid #5f7484;
    clip-path:polygon(14px 0,calc(100% - 14px) 0,100% 50%,calc(100% - 14px) 100%,14px 100%,0 50%);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.16),0 3px 8px rgba(0,0,0,.4);
}
.ma-command-tab span {color:#8bcfff;font-size:1.25rem;letter-spacing:-.15em;}

.ma-brand {
    margin:0 12px 10px;
    padding:6px 4px 14px;
    border-bottom:1px solid #355266;
}
.ma-brand-name {
    font-family:"Barlow Condensed",sans-serif;
    font-size:2.5rem;
    line-height:.92;
    font-weight:800;
    letter-spacing:-.035em;
    color:#eff9ff;
    text-shadow:0 0 12px rgba(36,162,255,.16);
    white-space:nowrap;
}
.ma-brand-name span {color:#149eff;text-shadow:0 0 13px rgba(0,147,255,.48);}
.ma-brand-sub {
    margin-top:8px;
    color:#91a8b8;
    font-family:"Barlow Condensed",sans-serif;
    font-size:.9rem;
    line-height:1.1;
    font-weight:700;
    letter-spacing:.055em;
    text-transform:uppercase;
}

.ma-nav {padding:0 8px;}
.ma-nav-row {
    height:47px;
    display:grid;
    grid-template-columns:35px minmax(0,1fr) 18px;
    align-items:center;
    padding:0 9px;
    color:#d4e0e7;
    font-family:"Barlow Condensed",sans-serif;
    font-size:1.22rem;
    font-weight:700;
    letter-spacing:.018em;
    text-transform:uppercase;
    border-left:2px solid transparent;
    border-right:1px solid transparent;
}
.ma-nav-row + .ma-nav-row {border-top:1px solid rgba(78,107,126,.11);}
.ma-nav-icon {
    width:26px;height:26px;display:grid;place-items:center;
    color:#d8ebf6;font-size:1.08rem;
    filter:drop-shadow(0 0 3px rgba(86,180,255,.20));
}
.ma-nav-row.active {
    position:relative;
    color:white;
    background:linear-gradient(90deg,#0365c7 0%,#087fe4 66%,#0753a0 100%);
    border:1px solid #0c9bff;
    box-shadow:inset 0 1px 0 rgba(159,221,255,.28),0 0 13px rgba(0,125,255,.40);
    clip-path:polygon(0 0,calc(100% - 13px) 0,100% 50%,calc(100% - 13px) 100%,0 100%);
}
.ma-nav-row.active::before {
    content:"";position:absolute;left:-4px;top:8px;bottom:8px;width:4px;background:#66d5ff;box-shadow:0 0 9px #00a6ff;
}
.ma-nav-arrow {color:#43baff;font-size:1.5rem;line-height:1;}
.ma-nav-row.active .ma-nav-arrow {color:#9be5ff;}

.ma-sidebar-divider {
    margin:10px 12px 12px;height:1px;
    background:linear-gradient(90deg,transparent,#608095 15%,#315167 85%,transparent);
    box-shadow:0 1px 0 #02070c;
}

.ma-device-card {
    margin:0 11px;
    min-height:225px;
    position:relative;
    overflow:hidden;
    padding:14px 14px 13px;
    background:
      radial-gradient(circle at 9px 9px,#9db0bc 0 1.7px,#1c2a34 2px 4px,transparent 4.2px),
      radial-gradient(circle at calc(100% - 9px) 9px,#9db0bc 0 1.7px,#1c2a34 2px 4px,transparent 4.2px),
      radial-gradient(circle at 9px calc(100% - 9px),#9db0bc 0 1.7px,#1c2a34 2px 4px,transparent 4.2px),
      radial-gradient(circle at calc(100% - 9px) calc(100% - 9px),#9db0bc 0 1.7px,#1c2a34 2px 4px,transparent 4.2px),
      linear-gradient(180deg,#082844,#031526);
    border:1px solid #698096;
    clip-path:polygon(10px 0,calc(100% - 10px) 0,100% 10px,100% calc(100% - 10px),calc(100% - 10px) 100%,10px 100%,0 calc(100% - 10px),0 10px);
    box-shadow:inset 0 0 0 4px #020c15,inset 0 0 0 5px #1d4562,0 8px 18px rgba(0,0,0,.40);
}
.ma-device-title {
    font-family:"Barlow Condensed",sans-serif;font-size:1.9rem;font-weight:800;color:#f2f8fb;letter-spacing:.03em;
}
.ma-device-status {margin-top:10px;display:flex;align-items:center;gap:8px;color:#ff6975;font-family:"Barlow Condensed",sans-serif;font-weight:700;font-size:1.08rem;letter-spacing:.04em;}
.ma-device-status.connected {color:var(--green);}
.ma-led {width:10px;height:10px;border-radius:50%;background:#ff6975;box-shadow:0 0 10px rgba(255,80,92,.55);}
.ma-device-status.connected .ma-led {background:var(--green);box-shadow:0 0 11px rgba(65,225,138,.65);}
.ma-device-port-label {margin-top:18px;color:#91a8b7;font-family:"Barlow Condensed",sans-serif;font-size:.9rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;}
.ma-device-port {color:#e6f4fb;font-family:"Barlow Condensed",sans-serif;font-size:1.5rem;font-weight:700;margin-top:1px;}
.ma-chip-scene {position:absolute;right:22px;top:96px;width:76px;height:76px;transform:rotate(-12deg);}
.ma-chip-body {position:absolute;inset:14px;background:linear-gradient(135deg,#42586b,#07101a 42%,#1b3850);border:2px solid #7c8d98;transform:rotate(45deg);box-shadow:0 0 0 3px #06121d,0 0 17px rgba(0,141,255,.32);}
.ma-chip-body::after {content:"";position:absolute;inset:10px;background:radial-gradient(circle,#159cff 0 10%,#05243f 22%,#020a10 67%);border:1px solid #2f8dcc;box-shadow:0 0 14px #008cff;}
.ma-signal {position:absolute;right:18px;bottom:55px;color:#40e685;font-size:1.24rem;font-weight:900;letter-spacing:2px;}
.ma-device-connect {
    position:absolute;left:15px;right:15px;bottom:14px;height:38px;display:grid;place-items:center;
    color:#eaf8ff;font-family:"Barlow Condensed",sans-serif;font-size:1.05rem;font-weight:800;letter-spacing:.04em;
    background:linear-gradient(180deg,#087ee1,#0458ae);border:1px solid #25adff;
    box-shadow:inset 0 1px 0 rgba(190,232,255,.3),0 0 0 3px #03111c,0 0 0 4px #365d78;
}

/* =========================================================
   TOP HEADER — refinery + session status
   ========================================================= */
.ma-top-shell {
    position:relative;
    min-height:160px;
    overflow:hidden;
    padding:18px 20px 16px;
    margin:0 0 9px;
    background:
      radial-gradient(circle at 10px 10px,#abb9c2 0 1.8px,#182733 2px 4px,transparent 4.3px),
      radial-gradient(circle at calc(100% - 10px) 10px,#abb9c2 0 1.8px,#182733 2px 4px,transparent 4.3px),
      radial-gradient(circle at 10px calc(100% - 10px),#778a98 0 1.8px,#182733 2px 4px,transparent 4.3px),
      radial-gradient(circle at calc(100% - 10px) calc(100% - 10px),#778a98 0 1.8px,#182733 2px 4px,transparent 4.3px),
      linear-gradient(180deg,rgba(11,61,103,.90),rgba(2,22,40,.96));
    border:1px solid #6d8393;
    clip-path:polygon(12px 0,calc(100% - 12px) 0,100% 12px,100% calc(100% - 12px),calc(100% - 12px) 100%,12px 100%,0 calc(100% - 12px),0 12px);
    box-shadow:inset 0 0 0 5px #020a12,inset 0 0 0 7px #173d59,inset 0 1px 0 rgba(193,222,240,.25),0 10px 22px rgba(0,0,0,.38);
}
.ma-top-shell::before {
    content:"";position:absolute;left:12px;right:12px;top:8px;height:2px;background:linear-gradient(90deg,transparent,#0da4ff 15%,#39b8ff 52%,transparent 91%);opacity:.75;
}
.ma-top-grid {position:relative;z-index:2;display:grid;grid-template-columns:minmax(440px,1fr) 430px;gap:12px;align-items:start;}
.ma-live-title-wrap {display:flex;align-items:center;gap:14px;padding-top:8px;}
.ma-live-logo {width:55px;height:58px;position:relative;filter:drop-shadow(0 0 9px rgba(0,153,255,.75));}
.ma-live-logo::before {content:"";position:absolute;left:20px;top:2px;width:19px;height:47px;background:linear-gradient(180deg,#8be9ff,#138eff 60%,#035ab2);clip-path:polygon(47% 0,100% 12%,81% 87%,38% 100%,0 74%,17% 13%);border:1px solid #a1ebff;}
.ma-live-logo::after {content:"";position:absolute;left:6px;top:17px;width:27px;height:27px;background:rgba(0,118,255,.34);clip-path:polygon(50% 0,100% 50%,50% 100%,0 50%);border-left:2px solid #36bbff;}
.ma-live-kicker {font-family:"Barlow Condensed",sans-serif;color:#83a7bc;font-size:.92rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.ma-live-title {font-family:"Barlow Condensed",sans-serif;color:#f7fbff;font-size:2.8rem;line-height:.96;font-weight:800;letter-spacing:.015em;text-transform:uppercase;text-shadow:0 0 8px rgba(99,213,255,.20);}
.ma-live-sub {margin-top:5px;color:#a7c1d0;font-family:"Barlow Condensed",sans-serif;font-size:1.12rem;font-weight:600;letter-spacing:.03em;text-transform:uppercase;}

.ma-plant {position:absolute;z-index:1;right:380px;bottom:0;width:480px;height:118px;opacity:.88;pointer-events:none;filter:drop-shadow(0 0 9px rgba(0,126,255,.25));}
.ma-plant svg {width:100%;height:100%;}

.ma-top-meta {display:grid;grid-template-columns:1.15fr 1.1fr .75fr;margin-top:8px;background:rgba(2,16,30,.30);border-left:1px solid #2e6f99;border-right:1px solid #1d4f70;}
.ma-meta-cell {min-height:78px;padding:9px 15px;border-left:1px solid rgba(91,138,167,.45);display:flex;flex-direction:column;justify-content:center;}
.ma-meta-cell:first-child {border-left:0;}
.ma-meta-label {color:#a8c0cf;font-family:"Barlow Condensed",sans-serif;font-size:.9rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;}
.ma-meta-value {margin-top:4px;color:#eaf7ff;font-family:"Barlow Condensed",sans-serif;font-size:1.55rem;font-weight:800;letter-spacing:.02em;white-space:nowrap;}
.ma-meta-value.timer {color:#62e69b;text-shadow:0 0 8px rgba(65,225,138,.24);}
.ma-live-indicator {display:flex;align-items:center;gap:7px;color:#55eb94!important;}
.ma-live-dot {width:10px;height:10px;border-radius:50%;border:2px solid #75ffc0;background:#20ca69;box-shadow:0 0 9px rgba(52,235,128,.65);}
.ma-live-indicator.standby {color:#84a0b0!important;}
.ma-live-indicator.standby .ma-live-dot {border-color:#687f8d;background:#344a58;box-shadow:none;}

/* =========================================================
   STREAMLIT CONTROL BUTTONS — glossy blue hardware
   ========================================================= */
[data-testid="stHorizontalBlock"] {gap:.55rem!important;}
.stButton > button, .stDownloadButton > button {
    min-height:49px!important;width:100%!important;border-radius:2px!important;
    color:#eff9ff!important;
    font-family:"Barlow Condensed",sans-serif!important;font-size:1.22rem!important;font-weight:800!important;letter-spacing:.028em!important;text-transform:uppercase!important;
    background:linear-gradient(180deg,#0b8bee 0%,#0569c6 46%,#034b99 100%)!important;
    border:1px solid #27b4ff!important;
    box-shadow:
      inset 0 1px 0 rgba(191,236,255,.34),
      inset 0 -2px 0 rgba(0,25,57,.58),
      0 0 0 3px #020c15,
      0 0 0 4px #4c6679,
      0 5px 12px rgba(0,0,0,.34),
      0 0 13px rgba(0,118,255,.16)!important;
}
.stButton > button:hover, .stDownloadButton > button:hover {background:linear-gradient(180deg,#16a4ff,#0777da 48%,#0556a5)!important;border-color:#7dd7ff!important;box-shadow:inset 0 1px 0 rgba(230,250,255,.42),0 0 0 3px #020c15,0 0 0 4px #657f90,0 7px 14px rgba(0,0,0,.39),0 0 18px rgba(0,141,255,.30)!important;}
.stButton > button:active {transform:translateY(1px);}
.stButton > button *, .stDownloadButton > button * {color:#eff9ff!important;}

/* Make the three main control columns look embedded in the header chassis. */
div[data-testid="stHorizontalBlock"]:has(.stButton) {margin-top:0;margin-bottom:11px;}

/* =========================================================
   SENSOR STATUS RIBBON
   ========================================================= */
.ma-sensor-ribbon {
    min-height:67px;margin:0 0 9px;padding:8px 10px;
    display:grid;grid-template-columns:1.25fr 1.25fr .9fr .9fr .8fr;gap:0;
    background:linear-gradient(180deg,#082640,#031728);border:1px solid #3b6078;
    box-shadow:inset 0 0 0 3px #020c14,inset 0 0 0 4px #1b3f57,0 6px 15px rgba(0,0,0,.28);
    clip-path:polygon(8px 0,calc(100% - 8px) 0,100% 8px,100% calc(100% - 8px),calc(100% - 8px) 100%,8px 100%,0 calc(100% - 8px),0 8px);
}
.ma-ribbon-cell {padding:5px 13px;border-left:1px solid rgba(65,120,157,.40);display:flex;flex-direction:column;justify-content:center;min-width:0;}
.ma-ribbon-cell:first-child {border-left:0;}
.ma-ribbon-label {color:#6f9ab4;font-family:"Barlow Condensed",sans-serif;font-size:.82rem;font-weight:700;letter-spacing:.075em;text-transform:uppercase;}
.ma-ribbon-value {margin-top:2px;color:#eaf7ff;font-family:"Barlow Condensed",sans-serif;font-size:1.23rem;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.ma-ribbon-value.blue {color:#4cc3ff;text-shadow:0 0 7px rgba(0,151,255,.26);}
.ma-ribbon-value.green {color:#48e18c;}

/* =========================================================
   INDUSTRIAL PANELS
   ========================================================= */
.ma-panel {
    position:relative;overflow:hidden;min-height:575px;padding:11px 11px 12px;
    background:
      radial-gradient(circle at 8px 8px,#9babb6 0 1.5px,#192934 1.8px 3.5px,transparent 3.8px),
      radial-gradient(circle at calc(100% - 8px) 8px,#9babb6 0 1.5px,#192934 1.8px 3.5px,transparent 3.8px),
      radial-gradient(circle at 8px calc(100% - 8px),#768895 0 1.5px,#192934 1.8px 3.5px,transparent 3.8px),
      radial-gradient(circle at calc(100% - 8px) calc(100% - 8px),#768895 0 1.5px,#192934 1.8px 3.5px,transparent 3.8px),
      linear-gradient(180deg,#082842 0%,#031729 100%);
    border:1px solid #5f7688;
    clip-path:polygon(9px 0,calc(100% - 9px) 0,100% 9px,100% calc(100% - 9px),calc(100% - 9px) 100%,9px 100%,0 calc(100% - 9px),0 9px);
    box-shadow:inset 0 0 0 4px #020a12,inset 0 0 0 5px #173c56,inset 0 1px 0 rgba(189,222,239,.19),0 9px 20px rgba(0,0,0,.36);
}
.ma-panel::after {content:"";position:absolute;left:9px;right:9px;bottom:7px;height:2px;background:linear-gradient(90deg,transparent,#087de0 25%,#29a9ff 65%,transparent);opacity:.30;pointer-events:none;}
.ma-panel-head {height:39px;padding:0 5px 7px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #31516a;box-shadow:0 1px 0 #02080d;}
.ma-panel-title {color:#dfeef7;font-family:"Barlow Condensed",sans-serif;font-size:1.5rem;font-weight:800;letter-spacing:.03em;text-transform:uppercase;}
.ma-panel-badge {color:#dceeff;background:linear-gradient(180deg,#0b6fc8,#06478a);border:1px solid #188dE8;border-radius:3px;padding:3px 9px;font-family:"Barlow Condensed",sans-serif;font-size:.95rem;font-weight:800;box-shadow:inset 0 1px 0 rgba(174,225,255,.2),0 0 8px rgba(0,113,214,.16);}

/* Timeline */
.ma-timeline {position:relative;margin-top:9px;max-height:500px;overflow-y:auto;padding:0 2px 4px 20px;scrollbar-width:thin;scrollbar-color:#1d6ca6 transparent;}
.ma-timeline::before {content:"";position:absolute;left:8px;top:15px;bottom:15px;width:1px;background:linear-gradient(#71d7ff,#17649b 30%,#29475b);}
.ma-event {position:relative;display:grid;grid-template-columns:42px minmax(0,1fr) 54px;gap:8px;align-items:center;min-height:58px;margin-bottom:7px;padding:7px 8px;background:linear-gradient(180deg,#0a3152,#05213a);border:1px solid #236b99;box-shadow:inset 0 1px 0 rgba(114,197,242,.10),inset 0 -1px 0 #02101b;clip-path:polygon(5px 0,100% 0,100% calc(100% - 5px),calc(100% - 5px) 100%,0 100%,0 5px);}
.ma-event::before {content:"";position:absolute;left:-17px;width:7px;height:7px;border-radius:50%;background:#8ddfff;border:1px solid #d8f5ff;box-shadow:0 0 7px #16a8ff;}
.ma-event-index {height:38px;display:grid;place-items:center;color:#eaf8ff;background:linear-gradient(180deg,#0f8ef4,#0560b5);border:1px solid #35b8ff;border-radius:3px;font-family:"Barlow Condensed",sans-serif;font-size:1.12rem;font-weight:800;box-shadow:inset 0 1px 0 rgba(208,242,255,.25),0 0 8px rgba(0,133,255,.23);}
.ma-event-name {color:#f1f8fc;font-family:"Barlow Condensed",sans-serif;font-size:1.2rem;font-weight:700;line-height:1.05;}
.ma-event-meta {margin-top:4px;color:#9eb8c7;font-family:"Barlow Condensed",sans-serif;font-size:.9rem;font-weight:500;}
.ma-event-confidence {justify-self:end;padding:5px 7px;color:#eaf8ff;background:#0c2c45;border:1px solid #507b96;border-radius:5px;font-family:"Barlow Condensed",sans-serif;font-size:1.05rem;font-weight:800;}
.ma-transition-note {grid-column:2 / 4;color:#82b6d3;font-size:.82rem;margin-top:-2px;}
.ma-transition-note .ok {color:#52e592;}.ma-transition-note .bad {color:#ff7e89;}
.ma-empty {height:430px;display:grid;place-items:center;text-align:center;color:#5c8098;font-family:"Barlow Condensed",sans-serif;font-size:1.12rem;border:1px dashed rgba(61,116,151,.25);background:rgba(0,20,36,.28);}

/* Workflow */
.ma-workflow {margin-top:9px;display:flex;flex-direction:column;gap:7px;}
.ma-step {display:grid;grid-template-columns:49px 34px minmax(0,1fr);gap:9px;align-items:center;min-height:88px;padding:8px 9px;background:linear-gradient(180deg,#092e4d,#041d33);border:1px solid #245e85;box-shadow:inset 0 1px 0 rgba(123,193,232,.09);clip-path:polygon(5px 0,100% 0,100% calc(100% - 5px),calc(100% - 5px) 100%,0 100%,0 5px);}
.ma-step-icon {width:47px;height:47px;display:grid;place-items:center;color:#e3f4ff;background:linear-gradient(180deg,#0b68b6,#063a6d);border:1px solid #2589d2;clip-path:polygon(7px 0,calc(100% - 7px) 0,100% 7px,100% calc(100% - 7px),calc(100% - 7px) 100%,7px 100%,0 calc(100% - 7px),0 7px);font-size:1.3rem;text-shadow:0 0 5px #0eb2ff;}
.ma-step-no {color:#dceefa;font-family:"Barlow Condensed",sans-serif;font-size:1.18rem;font-weight:800;text-align:center;}
.ma-step-title {color:#f0f8fc;font-family:"Barlow Condensed",sans-serif;font-size:1.28rem;font-weight:700;line-height:1.05;}
.ma-step-note {margin-top:5px;color:#aec1cd;font-family:"Barlow Condensed",sans-serif;font-size:1.02rem;line-height:1.15;}
.ma-sequence {margin-top:11px;padding:10px 12px 14px;position:relative;background:linear-gradient(180deg,#082640,#031729);border:1px solid #35607a;box-shadow:inset 0 0 0 3px #020b13;clip-path:polygon(7px 0,calc(100% - 7px) 0,100% 7px,100% calc(100% - 7px),calc(100% - 7px) 100%,7px 100%,0 calc(100% - 7px),0 7px);}
.ma-sequence-label {color:#9db8c8;font-family:"Barlow Condensed",sans-serif;font-size:.95rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;}
.ma-sequence-value {margin-top:5px;color:#eaf7ff;font-family:"Barlow Condensed",sans-serif;font-size:1.35rem;font-weight:800;letter-spacing:.02em;}
.ma-sequence-hazard {height:7px;margin-top:9px;background:repeating-linear-gradient(135deg,#ffae11 0 14px,#071521 14px 26px);border-top:1px solid #7d5f20;border-bottom:1px solid #04080b;}

/* Compliance */
.ma-compliance-panel {min-height:575px;}
.ma-gauge-zone {height:170px;position:relative;display:flex;justify-content:center;align-items:flex-end;padding-bottom:17px;border-bottom:1px solid #274b63;}
.ma-gauge-arc {--score:0;position:absolute;top:25px;width:210px;height:105px;overflow:hidden;}
.ma-gauge-arc::before {content:"";position:absolute;width:210px;height:210px;border-radius:50%;background:conic-gradient(from 270deg,#27bfff calc(var(--score)*1.8deg),#173d58 0 180deg,transparent 0);filter:drop-shadow(0 0 6px rgba(0,151,255,.45));}
.ma-gauge-arc::after {content:"";position:absolute;left:20px;top:20px;width:170px;height:170px;border-radius:50%;background:#041a2e;box-shadow:inset 0 0 0 1px #31516a;}
.ma-gauge-value {position:relative;z-index:2;color:#edf8ff;font-family:"Barlow Condensed",sans-serif;font-size:3.4rem;font-weight:800;text-shadow:0 0 10px rgba(59,185,255,.22);}
.ma-gauge-scale {position:absolute;left:30px;right:30px;bottom:12px;display:flex;justify-content:space-between;color:#8ba8b9;font-family:"Barlow Condensed",sans-serif;font-size:.95rem;}
.ma-score-metrics {padding:8px 2px 0;}
.ma-score-row {padding:10px 4px 11px;border-bottom:1px solid rgba(56,91,115,.40);}
.ma-score-label {color:#b9cfdb;font-family:"Barlow Condensed",sans-serif;font-size:.96rem;font-weight:700;letter-spacing:.025em;text-transform:uppercase;}
.ma-score-line {display:grid;grid-template-columns:78px minmax(0,1fr);gap:10px;align-items:end;margin-top:2px;}
.ma-score-value {color:#f0f9ff;font-family:"Barlow Condensed",sans-serif;font-size:2.1rem;font-weight:800;line-height:1;}
.ma-score-value small {font-size:1rem;color:#bbccd5;margin-left:3px;}
.ma-progress {height:8px;background:#16364e;border:1px solid #274c65;border-radius:6px;overflow:hidden;box-shadow:inset 0 1px 2px rgba(0,0,0,.55);}
.ma-progress > div {height:100%;background:linear-gradient(90deg,#0b7ee2,#23b8ff);box-shadow:0 0 6px rgba(0,147,255,.38);border-radius:5px;}
.ma-status-box {margin:10px 2px 0;padding:10px 10px;display:grid;grid-template-columns:44px minmax(0,1fr) 54px;gap:9px;align-items:center;background:linear-gradient(180deg,#072a47,#041b30);border:1px solid #276b98;clip-path:polygon(6px 0,100% 0,100% calc(100% - 6px),calc(100% - 6px) 100%,0 100%,0 6px);}
.ma-status-icon {width:39px;height:39px;display:grid;place-items:center;color:#e5f7ff;border:2px solid #24a8ff;clip-path:polygon(50% 0,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);font-weight:800;text-shadow:0 0 7px #05a8ff;}
.ma-status-title {color:#eaf8ff;font-family:"Barlow Condensed",sans-serif;font-size:1.04rem;font-weight:800;letter-spacing:.02em;text-transform:uppercase;}
.ma-status-detail {color:#96b0c0;font-family:"Barlow Condensed",sans-serif;font-size:.82rem;line-height:1.15;margin-top:3px;}
.ma-status-pct {color:#eaf8ff;background:#0b2f4b;border:1px solid #4b7894;border-radius:5px;padding:6px 4px;text-align:center;font-family:"Barlow Condensed",sans-serif;font-size:1.25rem;font-weight:800;}
.ma-mini-grid {display:grid;grid-template-columns:1fr 1fr 1fr;margin-top:10px;border:1px solid #31546a;background:#031728;}
.ma-mini {padding:8px 5px;text-align:center;border-left:1px solid #31546a;min-height:66px;}
.ma-mini:first-child {border-left:0;}
.ma-mini-label {color:#8eabbc;font-family:"Barlow Condensed",sans-serif;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em;}
.ma-mini-value {margin-top:3px;color:#e6f5fd;font-family:"Barlow Condensed",sans-serif;font-size:1.6rem;font-weight:800;line-height:1;}
.ma-mini-note {margin-top:4px;color:#8ca6b6;font-family:"Barlow Condensed",sans-serif;font-size:.75rem;text-transform:uppercase;}
.ma-mini-note.green {color:#48e18c;}

/* Post-run detail bays */
.ma-detail-grid {display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:10px;}
.ma-detail-panel {background:linear-gradient(180deg,#08253d,#031626);border:1px solid #405f73;padding:10px 11px;box-shadow:inset 0 0 0 3px #020a11,inset 0 0 0 4px #17384f;clip-path:polygon(7px 0,calc(100% - 7px) 0,100% 7px,100% calc(100% - 7px),calc(100% - 7px) 100%,7px 100%,0 calc(100% - 7px),0 7px);}
.ma-detail-title {color:#edf7fc;font-family:"Barlow Condensed",sans-serif;font-size:1.05rem;font-weight:800;text-transform:uppercase;border-bottom:1px solid #315069;padding-bottom:7px;}
.ma-table-row {display:grid;grid-template-columns:minmax(0,1fr) 60px 60px 70px;gap:6px;align-items:center;min-height:39px;border-top:1px solid rgba(70,109,133,.25);color:#b8ceda;font-size:.68rem;}
.ma-table-row.head {min-height:30px;color:#7799ad;text-transform:uppercase;font-size:.6rem;border-top:0;}
.ma-badge {display:inline-flex;justify-content:center;min-width:52px;padding:3px 5px;border-radius:3px;font-family:"Barlow Condensed",sans-serif;font-size:.63rem;font-weight:800;text-transform:uppercase;}
.ma-pass {color:#65eba2;background:rgba(28,107,70,.28);border:1px solid rgba(69,223,139,.38);}.ma-fail {color:#ff8992;background:rgba(115,42,50,.34);border:1px solid rgba(255,101,115,.38);}.ma-warn {color:#ffc966;background:rgba(119,81,17,.30);border:1px solid rgba(255,175,22,.35);}
.ma-rule-row {display:grid;grid-template-columns:65px 1fr;gap:7px;align-items:center;min-height:45px;border-top:1px solid rgba(70,109,133,.25);color:#b8ceda;font-size:.69rem;}
.ma-wrench-grid {margin-top:9px;}
.ma-wrench-row {display:grid;grid-template-columns:1.2fr .8fr .8fr .65fr;gap:8px;align-items:center;min-height:39px;border-top:1px solid rgba(70,109,133,.25);color:#b8ceda;font-size:.68rem;}
.ma-wrench-row.head {min-height:30px;color:#7799ad;text-transform:uppercase;font-size:.6rem;border-top:0;}

/* Alerts / expanders / native charts. */
div[data-testid="stAlert"] {background:#06243a!important;border:1px solid #2b607f!important;color:#bde8ff!important;border-radius:2px!important;box-shadow:inset 0 0 0 2px #020c14!important;}
div[data-testid="stExpander"] {background:#031623!important;border:1px solid #29485d!important;border-radius:2px!important;color:#cde8f5!important;}
[data-testid="stMetric"] {background:#061f34!important;border:1px solid #305873!important;border-radius:2px!important;padding:8px!important;}
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {color:#dff4ff!important;}

@media(max-width:1180px) {
    section[data-testid="stSidebar"] {width:260px!important;min-width:260px!important;}
    .ma-top-grid {grid-template-columns:1fr 350px;}.ma-plant{right:310px;width:390px;opacity:.58;}
    .ma-live-title{font-size:2rem}.ma-sensor-ribbon{grid-template-columns:1fr 1fr 1fr}.ma-ribbon-cell:nth-child(4),.ma-ribbon-cell:nth-child(5){display:none;}
}
@media(max-width:900px) {
    section[data-testid="stSidebar"] {display:none!important;}
    .block-container{padding:7px!important}.ma-top-grid{grid-template-columns:1fr}.ma-top-meta{grid-template-columns:1fr 1fr 1fr}.ma-plant{display:none}.ma-detail-grid{grid-template-columns:1fr}.ma-panel{min-height:auto}.ma-timeline{max-height:430px}
}
</style>
    """,
    unsafe_allow_html=True,
)

is_connected_ui = bool(st.session_state.get("running", False))
device_status_class = "connected" if is_connected_ui else ""
device_status_text = "CONNECTED" if is_connected_ui else "OFFLINE"

sidebar_html = (
    '<div class="ma-sidebar-shell">'
    '<div class="ma-command-tab"><span>»»</span> BLUE COMMAND</div>'
    '<div class="ma-brand">'
    '<div class="ma-brand-name">Mainten<span>Action</span></div>'
    '<div class="ma-brand-sub">Live maintenance action recognition</div>'
    '</div>'
    '<div class="ma-nav">'
    '<div class="ma-nav-row"><div class="ma-nav-icon">⌂</div><div>DASHBOARD</div><div class="ma-nav-arrow"></div></div>'
    '<div class="ma-nav-row active"><div class="ma-nav-icon">✺</div><div>LIVE MONITOR</div><div class="ma-nav-arrow">›</div></div>'
    '<div class="ma-nav-row"><div class="ma-nav-icon">◷</div><div>ACTIVITY HISTORY</div><div class="ma-nav-arrow"></div></div>'
    '<div class="ma-nav-row"><div class="ma-nav-icon">◇</div><div>MODEL INFO</div><div class="ma-nav-arrow"></div></div>'
    '<div class="ma-nav-row"><div class="ma-nav-icon">⛓</div><div>DEVICES</div><div class="ma-nav-arrow"></div></div>'
    '<div class="ma-nav-row"><div class="ma-nav-icon">▤</div><div>PROCEDURES</div><div class="ma-nav-arrow"></div></div>'
    '<div class="ma-nav-row"><div class="ma-nav-icon">▧</div><div>REPORTS</div><div class="ma-nav-arrow"></div></div>'
    '<div class="ma-nav-row"><div class="ma-nav-icon">⚙</div><div>SETTINGS</div><div class="ma-nav-arrow"></div></div>'
    '</div>'
    '<div class="ma-sidebar-divider"></div>'
    '<div class="ma-device-card">'
    '<div class="ma-device-title">ESP32</div>'
    f'<div class="ma-device-status {device_status_class}"><span class="ma-led"></span><span>{device_status_text}</span></div>'
    '<div class="ma-device-port-label">PORT</div>'
    f'<div class="ma-device-port">{PORT}</div>'
    '<div class="ma-chip-scene"><div class="ma-chip-body"></div></div>'
    '<div class="ma-signal">▂▄▆█</div>'
    '<div class="ma-device-connect">CONNECT</div>'
    '</div>'
    '</div>'
)

with st.sidebar:
    st.markdown(sidebar_html, unsafe_allow_html=True)

def _format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

@st.fragment(run_every=1.0)
def render_command_header():
    running = bool(st.session_state.get("running", False))
    started = st.session_state.get("monitoring_started_at")
    if running and started:
        elapsed = _format_elapsed(time.time() - float(started))
    else:
        elapsed = "00:00:00"

    session_id = st.session_state.get("session_id", "#7F3A-28B1")
    live_class = "" if running else " standby"
    live_text = "LIVE" if running else "STANDBY"

    refinery_svg = r'''<svg viewBox="0 0 520 130" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#159cff"/><stop offset="1" stop-color="#06213a"/></linearGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <g fill="#041423" stroke="#168ee0" stroke-width="1" opacity=".94">
        <path d="M0 119H520V130H0z" fill="#03111d"/>
        <path d="M30 119V78H44V58H51V119M64 119V88H81V68H89V119M103 119V73H119V47H125V119M134 119V92H153V62H161V119M176 119V55H191V28H197V119M206 119V81H225V42H232V119M244 119V69H261V51H268V119M282 119V43H296V18H303V119M315 119V79H331V57H338V119M350 119V65H366V35H373V119M389 119V88H407V49H414V119M431 119V72H447V44H454V119M470 119V91H489V69H496V119"/>
        <path d="M20 99H164M158 83H244M230 94H338M329 74H438M414 101H510" fill="none"/>
        <path d="M38 58H57M111 47H132M185 28H203M290 18H308M357 35H381M438 44H461" fill="none"/>
        <path d="M45 43v15M118 32v15M192 13v15M297 3v15M365 20v15M446 29v15" fill="none"/>
      </g>
      <g fill="#24aaff" filter="url(#glow)">
        <circle cx="51" cy="91" r="2"/><circle cx="89" cy="104" r="2"/><circle cx="125" cy="65" r="2"/><circle cx="197" cy="43" r="2"/><circle cx="232" cy="91" r="2"/><circle cx="303" cy="31" r="2"/><circle cx="338" cy="97" r="2"/><circle cx="373" cy="55" r="2"/><circle cx="454" cy="80" r="2"/><circle cx="496" cy="104" r="2"/>
      </g>
      <g fill="#f4a51d" opacity=".9"><circle cx="185" cy="75" r="1.5"/><circle cx="297" cy="72" r="1.5"/><circle cx="365" cy="88" r="1.5"/><circle cx="447" cy="61" r="1.5"/></g>
    </svg>'''

    header_html = (
        '<div class="ma-top-shell">'
        f'<div class="ma-plant">{refinery_svg}</div>'
        '<div class="ma-top-grid">'
        '<div class="ma-live-title-wrap">'
        '<div class="ma-live-logo"></div>'
        '<div>'
        '<div class="ma-live-kicker">BLUE COMMAND / MAINTENANCE CONTROL</div>'
        '<div class="ma-live-title">LIVE MONITORING</div>'
        '<div class="ma-live-sub">ESP32 + MPU6050 SENSOR FUSION</div>'
        '</div></div>'
        '<div class="ma-top-meta">'
        '<div class="ma-meta-cell"><div class="ma-meta-label">SESSION ID</div>'
        f'<div class="ma-meta-value">{html.escape(str(session_id))}</div></div>'
        '<div class="ma-meta-cell"><div class="ma-meta-label">ELAPSED TIME</div>'
        f'<div class="ma-meta-value timer">{elapsed}</div></div>'
        '<div class="ma-meta-cell"><div class="ma-meta-label">LIVE</div>'
        f'<div class="ma-meta-value ma-live-indicator{live_class}"><span class="ma-live-dot"></span>{live_text}</div></div>'
        '</div></div></div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

render_command_header()

# =========================================================
# LOAD MODEL + PREPROCESSING
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(PREPROCESSING_PATH):
    st.error(f"Preprocessing file not found: {PREPROCESSING_PATH}")
    st.stop()

if not os.path.exists(CROSS_ADJ_MODEL_PATH):
    st.error(f"Cross/adjacent model file not found: {CROSS_ADJ_MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_everything(model_path, preprocessing_path, model_modified, prep_modified,):
    model = load_model(model_path)

    prep = np.load(preprocessing_path, allow_pickle=True,)

    mean = prep["mean"]
    std = prep["std"]
    classes = prep["classes"]

    return model, mean, std, classes

model_modified = os.path.getmtime(MODEL_PATH)
prep_modified = os.path.getmtime(PREPROCESSING_PATH)

model, mean, std, classes = load_everything(MODEL_PATH, PREPROCESSING_PATH, model_modified, prep_modified,)

@st.cache_resource
def load_cross_adj_sequence_model(model_path, model_modified):
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Invalid CAC/ACA model bundle.")
    return bundle

cross_adj_model_modified = os.path.getmtime(CROSS_ADJ_MODEL_PATH)
cross_adj_model_bundle = load_cross_adj_sequence_model(CROSS_ADJ_MODEL_PATH, cross_adj_model_modified,)
cross_adj_sequence_model = cross_adj_model_bundle["model"]
cross_adj_label_map = cross_adj_model_bundle.get("label_map", {0: 'ACA', 1: 'CAC'},)

classes = np.array([str(c) for c in classes])

model_window_size = int(model.input_shape[1])

model_channels = int(model.input_shape[2])

if model_channels != 6:
    st.error(f"Model expects {model_channels} channels. " "Expected 6: ax, ay, az, gx, gy, gz.")
    st.stop()

if model_window_size != EXPECTED_WINDOW_SIZE:
    st.error(f"Model expects {model_window_size} samples, " f"but app expects {EXPECTED_WINDOW_SIZE}.")
    st.stop()

if mean.shape[-1] != 6 or std.shape[-1] != 6:
    st.error("Preprocessing file does not contain " "six-channel normalization values.")
    st.stop()

if model.output_shape[-1] != len(classes):
    st.error("Model output count does not match " "saved class names.")
    st.stop()

WINDOW_SIZE = model_window_size

# We deliberately ignore a learned idle output if one exists.
MAINTENANCE_CLASS_INDICES = [i for i, class_name in enumerate(classes) if class_name != 'idle']

MAINTENANCE_CLASSES = [classes[i] for i in MAINTENANCE_CLASS_INDICES]

required_actions = {'clean_inspect', 'place_align_gasket', 'insert_bolt_nut', 'hand_tighten', 'wrench_tighten', 'lubrication'}

missing_model_actions = required_actions - set(MAINTENANCE_CLASSES)

if missing_model_actions:
    st.error("Model/preprocessing is missing required classes: " + ", ".join(sorted(missing_model_actions)))
    st.stop()

# =========================================================
# HELPERS
# =========================================================

def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def gyro_p25_raw(window_2d):
    gyro = np.asarray(window_2d[:, 3:6], dtype=np.float64)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    return float(np.percentile(gyro_mag, 25))

def rule_based_idle(window_2d):
    score = gyro_p25_raw(window_2d)
    if not st.session_state.is_rule_idle:
        if score < IDLE_ENTER_GYRO_P25_THRESHOLD:
            st.session_state.is_rule_idle = True
            st.session_state.idle_exit_counter = 0
    else:
        if score >= IDLE_EXIT_GYRO_P25_THRESHOLD:
            st.session_state.idle_exit_counter += 1
        else:
            st.session_state.idle_exit_counter = 0
        if st.session_state.idle_exit_counter >= IDLE_EXIT_CONSECUTIVE_WINDOWS:
            st.session_state.is_rule_idle = False
            st.session_state.idle_exit_counter = 0
    return bool(st.session_state.is_rule_idle), score

def learn_wrench_rotation_axis(samples):
    """
    Learn the dominant rotation axis during the confirmed wrench action.

    The sign of an axis is irrelevant here: tightening/ratcheting may move
    forward and backward along the same physical axis.
    """
    arr = np.asarray(samples, dtype=np.float64)

    if len(arr) < 3:
        return None

    gyro_dps = arr[:, 3:6] / GYRO_SCALE

    # Ignore almost-zero rows so stillness does not dominate the estimate.
    magnitudes = np.linalg.norm(gyro_dps, axis=1)
    moving = gyro_dps[magnitudes > 1e-8]

    if len(moving) < 3:
        return None

    # Principal direction of the 3-D gyro cloud.
    # This captures the physical wrenching axis even when motion reverses.
    _, _, vh = np.linalg.svd(moving, full_matrices=False)
    axis = np.asarray(vh[0], dtype=np.float64)

    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return None

    return axis / norm

def wrench_reposition_metrics(samples, wrench_axis):
    """
    Adaptive wrench re-arm metrics:
    - total angular travel in the recent window
    - fraction of that angular travel perpendicular to the learned
      tightening axis
    """
    arr = np.asarray(samples, dtype=np.float64)

    if len(arr) < 2 or wrench_axis is None:
        return 0.0, 0.0

    axis = np.asarray(wrench_axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm <= 1e-8:
        return 0.0, 0.0

    axis = axis / axis_norm

    gyro_dps = arr[:, 3:6] / GYRO_SCALE

    gyro_mag = np.linalg.norm(gyro_dps, axis=1,)

    total_rotation_deg = float(np.sum(gyro_mag) / SAMPLE_RATE_HZ)

    parallel_scalar = gyro_dps @ axis

    parallel_vec = (parallel_scalar[:, None] * axis[None, :])

    perpendicular_vec = (gyro_dps - parallel_vec)

    perpendicular_mag = np.linalg.norm(perpendicular_vec, axis=1,)

    perpendicular_rotation_deg = float(np.sum(perpendicular_mag) / SAMPLE_RATE_HZ)

    perpendicular_fraction = float(perpendicular_rotation_deg / (total_rotation_deg + 1e-8))

    return (total_rotation_deg, perpendicular_fraction,)

def _rotation_from_two_vectors(source, target):
    """Return scipy Rotation that maps source direction onto target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    source /= np.linalg.norm(source) + 1e-12
    target /= np.linalg.norm(target) + 1e-12

    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot < -0.9999:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis) + 1e-12
        return Rot.from_rotvec(axis * np.pi)

    s = np.sqrt((1.0 + dot) * 2.0)
    if s <= 1e-12:
        return Rot.identity()

    quat_xyzw = np.array([cross[0] / s, cross[1] / s, cross[2] / s, s / 2.0], dtype=np.float64,)
    return Rot.from_quat(quat_xyzw)

def _sequence_model_orientation_path(samples):
    """
    Exact transition feature used to train cac_aca_logistic_model.joblib.

    The trained feature is the cumulative orientation-path length from a
    90-sample prebuffer + 90-sample postbuffer, with no accelerometer
    complementary correction (kp=0).  Keeping this implementation aligned
    with training is critical for valid live inference.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if len(arr) < 24:
        return None

    accel = arr[:, :3] / 16384.0
    gyro_rad_s = (arr[:, 3:6] / GYRO_SCALE) * (np.pi / 180.0)

    accel_unit = accel / (np.linalg.norm(accel, axis=1, keepdims=True) + 1e-12)

    rotation = _rotation_from_two_vectors(accel_unit[0], np.array([0.0, 0.0, 1.0], dtype=np.float64),)

    dt = 1.0 / SAMPLE_RATE_HZ
    rotation_vectors = []

    for i in range(len(arr)):
        # Training configuration was kp=0: gyro-only orientation propagation
        # after the initial gravity alignment.
        rotation = rotation * Rot.from_rotvec(gyro_rad_s[i] * dt)
        rotation_vectors.append(rotation.as_rotvec())

    rotation_vectors = np.asarray(rotation_vectors, dtype=np.float64)
    if len(rotation_vectors) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(rotation_vectors, axis=0), axis=1,)))

def classify_wrench_transition(samples):
    """Extract the specialist CAC/ACA model feature for one transition."""
    arr = np.asarray(samples, dtype=np.float64)

    orientation_path = _sequence_model_orientation_path(arr)
    if orientation_path is None or not np.isfinite(orientation_path):
        return {
            'observed': 'unknown',
            'cross_probability': None,
            'relative_score': None,
            'model_orientation_path': None,
            'reason': 'specialist_model_feature_failed',
        }

    gyro_raw = arr[:, 3:6]
    gyro_median_raw = float(np.median(np.linalg.norm(gyro_raw, axis=1)))

    return {
        'observed': 'pending',
        'cross_probability': None,
        'relative_score': None,
        'gyro_median_raw': gyro_median_raw,
        'transition_sample_count': int(len(arr)),
        'model_orientation_path': float(orientation_path),
        'reason': 'pending_cac_aca_logistic_model',
    }

def finalize_wrench_transitions_relative_multiwindow():
    """
    Finalize all three moves with the trained CAC/ACA logistic model.

    Exact trained scalar feature:
        X = mean(orientation_path(T1), orientation_path(T3))
            - orientation_path(T2)

    The model predicts the whole sequence as CAC or ACA.  Individual
    transition labels are then assigned from that predicted sequence.
    """
    transitions = st.session_state.wrench_transitions

    if len(transitions) < 3:
        return

    first_three = transitions[:3]
    values = []

    for item in first_three:
        value = item.get("model_orientation_path")
        if value is None or not np.isfinite(value):
            for failed_item in first_three:
                failed_item["observed"] = "unknown"
                failed_item["ok"] = None
                failed_item["reason"] = "missing_specialist_model_feature"
            st.session_state.last_transition_result = first_three[-1]
            return
        values.append(float(value))

    t1, t2, t3 = values
    feature_value = float(((t1 + t3) * 0.5) - t2)

    x = np.asarray([[feature_value]], dtype=np.float64)
    predicted_class = int(cross_adj_sequence_model.predict(x)[0])
    sequence_pattern = str(cross_adj_label_map.get(predicted_class, predicted_class)).upper()

    sequence_probability = None
    probability_cac = None

    if hasattr(cross_adj_sequence_model, "predict_proba"):
        probabilities = cross_adj_sequence_model.predict_proba(x)[0]
        model_classes = list(cross_adj_sequence_model.classes_)

        if predicted_class in model_classes:
            predicted_index = model_classes.index(predicted_class)
            sequence_probability = float(probabilities[predicted_index])

        cac_classes = [class_id for class_id, label in cross_adj_label_map.items() if str(label).upper() == "CAC"]
        if cac_classes and cac_classes[0] in model_classes:
            cac_index = model_classes.index(cac_classes[0])
            probability_cac = float(probabilities[cac_index])

    if sequence_pattern == "CAC":
        observed_values = ["cross", "adjacent", "cross"]
    elif sequence_pattern == "ACA":
        observed_values = ["adjacent", "cross", "adjacent"]
    else:
        observed_values = ["unknown", "unknown", "unknown"]

    for transition_index, (item, observed) in enumerate(zip(first_three, observed_values)):
        item["observed"] = observed
        item["relative_score"] = feature_value
        item["sequence_model_feature"] = feature_value
        item["sequence_model_prediction"] = sequence_pattern
        item["sequence_model_probability"] = sequence_probability
        item["model_orientation_path_t1"] = t1
        item["model_orientation_path_t2"] = t2
        item["model_orientation_path_t3"] = t3
        item["reason"] = "cac_aca_logistic_sequence_model"

        # Cross probability can be derived from P(CAC): T1/T3 are cross in
        # CAC, while T2 is cross in ACA.
        if probability_cac is not None:
            item["cross_probability"] = (probability_cac if transition_index in (0, 2) else 1.0 - probability_cac)
        else:
            item["cross_probability"] = None

        expected = item.get("expected")
        item["ok"] = (expected is not None and observed == expected)

    st.session_state.last_transition_result = first_three[-1]

def expected_transition_for_wrench_number(wrench_number,):
    return {2: 'cross', 3: 'adjacent', 4: 'cross'}.get(wrench_number)

def _moving_average_same(values, window_size):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return values

    window_size = max(1, min(int(window_size), len(values),),)

    # Edge-safe centered moving average:
    # use only real samples near the start/end instead of zero-padding.
    left = window_size // 2
    right = window_size - 1 - left

    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64,),))

    indices = np.arange(len(values))

    starts = np.maximum(0, indices - left,)

    ends = np.minimum(len(values), indices + right + 1,)

    sums = (cumulative[ends] - cumulative[starts])

    counts = (ends - starts)

    return (sums / counts)

def _moving_average_matrix_same(values, window_size):
    values = np.asarray(values, dtype=np.float64,)

    if len(values) == 0:
        return values

    smoothed = np.zeros_like(values, dtype=np.float64,)

    for col in range(values.shape[1]):
        smoothed[:, col] = (_moving_average_same(values[:, col], window_size,))

    return smoothed

def _lub_peak_prominence(signal, index, radius,):
    left_start = max(0, index - radius,)

    right_end = min(len(signal), index + radius + 1,)

    left_min = float(np.min(signal[left_start:index + 1]))

    right_min = float(np.min(signal[index:right_end]))

    return float(signal[index] - max(left_min, right_min,))

def _lub_peak_width_half_prominence(signal, index, prominence,):
    half_level = float(signal[index] - 0.5 * prominence)

    left = int(index)

    while (left > 0 and signal[left] > half_level):
        left -= 1

    right = int(index)

    while (right < len(signal) - 1 and signal[right] > half_level):
        right += 1

    return int(right - left)

def _lub_select_spaced_peaks(indices, signal, min_distance,):
    selected = []

    for idx in sorted(int(i) for i in indices):

        if (not selected or idx - selected[-1] >= min_distance):
            selected.append(idx)

        elif (signal[idx] > signal[selected[-1]]):
            selected[-1] = (idx)

    return selected

def count_lubrication_strokes(samples):
    arr = np.asarray(samples, dtype=np.float64,)

    if len(arr) < 35:
        return 0

    acc = arr[:, :3]
    gyro = arr[:, 3:6]

    # -----------------------------------------------------
    # 1) Remove slow gravity/orientation acceleration.
    # -----------------------------------------------------
    gravity_like = _moving_average_matrix_same(acc, LUB_GRAVITY_SMOOTH_SAMPLES)

    dynamic_mag = np.linalg.norm(acc - gravity_like, axis=1,)

    # Coarse envelope gives one clear hump for a normal pump.
    coarse_envelope = _moving_average_same(dynamic_mag, LUB_COARSE_ENVELOPE_SMOOTH_SAMPLES)

    if len(coarse_envelope) < 3:
        return 0

    p10 = float(np.percentile(coarse_envelope, 10,))

    p90 = float(np.percentile(coarse_envelope, 90,))

    min_height = float(np.percentile(coarse_envelope, LUB_PEAK_HEIGHT_PERCENTILE,))

    min_prominence = float(LUB_PEAK_PROMINENCE_RATIO * max(p90 - p10, 1.0,))

    # -----------------------------------------------------
    # 2) Find coarse acceleration-pulse candidates.
    # -----------------------------------------------------
    raw_candidates = []

    for i in range(1, len(coarse_envelope) - 1,):

        if not (
            coarse_envelope[i] > coarse_envelope[i - 1]
            and
            coarse_envelope[i] >= coarse_envelope[i + 1]
            and
            coarse_envelope[i] >= min_height
        ):
            continue

        prominence = (_lub_peak_prominence(coarse_envelope, i, LUB_MIN_PEAK_DISTANCE_SAMPLES,))

        if (prominence < min_prominence):
            continue

        raw_candidates.append(int(i))

    coarse_candidates = (_lub_select_spaced_peaks(raw_candidates, coarse_envelope, LUB_MIN_PEAK_DISTANCE_SAMPLES,))

    if not coarse_candidates:
        return 0

    gyro_mag = np.linalg.norm(gyro, axis=1,)

    accepted = []
    broad_low_gyro = []
    bottle_handling = []

    # -----------------------------------------------------
    # 3) Classify each coarse pulse by width + gyro.
    #
    # Short pulse:
    #     normal pump candidate.
    #
    # Broad + high gyro:
    #     bottle handling / pick-up / put-down.
    #
    # Broad + normal gyro:
    #     may contain multiple fast pumps merged together,
    #     so inspect it again using finer smoothing.
    # -----------------------------------------------------
    for idx in coarse_candidates:

        prominence = (_lub_peak_prominence(coarse_envelope, idx, LUB_MIN_PEAK_DISTANCE_SAMPLES,))

        peak_width = (_lub_peak_width_half_prominence(coarse_envelope, idx, prominence,))

        gyro_start = max(0, idx - LUB_GYRO_CONTEXT_SAMPLES,)

        gyro_end = min(len(gyro_mag), idx + LUB_GYRO_CONTEXT_SAMPLES + 1,)

        local_gyro_p95 = float(np.percentile(gyro_mag[gyro_start:gyro_end], 95,))

        if (peak_width > LUB_MAX_NORMAL_PEAK_WIDTH_SAMPLES and local_gyro_p95 > LUB_BOTTLE_GYRO_P95_THRESHOLD):
            bottle_handling.append(idx)
            continue

        if (peak_width > LUB_MAX_NORMAL_PEAK_WIDTH_SAMPLES):
            broad_low_gyro.append(idx)
            continue

        if (peak_width >= LUB_MIN_PEAK_WIDTH_SAMPLES):
            accepted.append(idx)

    # -----------------------------------------------------
    # 4) Re-check broad LOW-gyro regions at a finer scale.
    #    This prevents two or more fast pumps from being
    #    merged into one coarse peak.
    # -----------------------------------------------------
    if broad_low_gyro:

        fine_envelope = (_moving_average_same(dynamic_mag, LUB_FINE_ENVELOPE_SMOOTH_SAMPLES,))

        fine_p10 = float(np.percentile(fine_envelope, 10,))

        fine_p90 = float(np.percentile(fine_envelope, 90,))

        fine_min_prominence = float(LUB_FINE_PEAK_PROMINENCE_RATIO * max(fine_p90 - fine_p10, 1.0,))

        for coarse_idx in broad_low_gyro:

            coarse_prominence = (_lub_peak_prominence(coarse_envelope, coarse_idx, LUB_MIN_PEAK_DISTANCE_SAMPLES,))

            coarse_width = (_lub_peak_width_half_prominence(coarse_envelope, coarse_idx, coarse_prominence,))

            search_radius = max(LUB_MAX_NORMAL_PEAK_WIDTH_SAMPLES, coarse_width,)

            search_start = max(1, coarse_idx - search_radius,)

            search_end = min(len(fine_envelope) - 1, coarse_idx + search_radius + 1,)

            local_region = (fine_envelope[search_start:search_end])

            if len(local_region) < 3:
                accepted.append(coarse_idx)
                continue

            local_height = float(np.percentile(local_region, 60,))

            fine_candidates = []

            for i in range(search_start, search_end,):

                if not (
                    fine_envelope[i] > fine_envelope[i - 1]
                    and
                    fine_envelope[i] >= fine_envelope[i + 1]
                    and
                    fine_envelope[i] >= local_height
                ):
                    continue

                prominence = (_lub_peak_prominence(fine_envelope, i, LUB_FINE_MIN_PEAK_DISTANCE_SAMPLES,))

                if (prominence >= fine_min_prominence):
                    fine_candidates.append(i)

            fine_selected = _lub_select_spaced_peaks(fine_candidates, fine_envelope, LUB_FINE_MIN_PEAK_DISTANCE_SAMPLES)

            # Split only when the fine signal contains at
            # least two distinct pulses. Otherwise retain
            # the original coarse candidate.
            if len(fine_selected) >= 2:
                accepted.extend(fine_selected)
            else:
                accepted.append(coarse_idx)

    if not accepted:
        return 0

    accepted = sorted(set(int(i) for i in accepted))

    # -----------------------------------------------------
    # 5) Bottle handling exclusion.
    #    Do not count pulses immediately around a broad,
    #    high-gyro handling event.
    # -----------------------------------------------------
    if bottle_handling:
        accepted = [
            idx
            for idx in accepted
            if all((abs(idx - handling_idx) > LUB_BOTTLE_EXCLUSION_SAMPLES for handling_idx in bottle_handling))
        ]

    if not accepted:
        return 0

    # Re-enforce minimum physical spacing after fine/coarse
    # candidates have been combined.
    strength_signal = _moving_average_same(dynamic_mag, LUB_FINE_ENVELOPE_SMOOTH_SAMPLES)

    accepted = (_lub_select_spaced_peaks(accepted, strength_signal, LUB_FINE_MIN_PEAK_DISTANCE_SAMPLES,))

    # -----------------------------------------------------
    # 6) Remove obvious LEADING hand/bottle handling.
    #
    # The lubrication model can be confirmed after pumping has
    # already started, so we now keep a 4-second prebuffer.
    # That captures the missing early pump, but it may also include
    # bottle pickup / hand positioning. Remove only a clearly
    # different leading pulse relative to the next 3 candidates.
    # -----------------------------------------------------
    while len(accepted) >= 4:

        first_idx = accepted[0]
        next_three = accepted[1:4]

        first_gyro_start = max(0, first_idx - LUB_GYRO_CONTEXT_SAMPLES,)

        first_gyro_end = min(len(gyro_mag), first_idx + LUB_GYRO_CONTEXT_SAMPLES + 1,)

        first_gyro_median = float(np.median(gyro_mag[first_gyro_start:first_gyro_end]))

        next_gyro_medians = []

        for idx in next_three:

            gyro_start = max(0, idx - LUB_GYRO_CONTEXT_SAMPLES,)

            gyro_end = min(len(gyro_mag), idx + LUB_GYRO_CONTEXT_SAMPLES + 1,)

            next_gyro_medians.append(float(np.median(gyro_mag[gyro_start:gyro_end])))

        next_gyro_level = float(np.median(next_gyro_medians))

        first_prominence = (_lub_peak_prominence(coarse_envelope, first_idx, LUB_MIN_PEAK_DISTANCE_SAMPLES,))

        first_width = (_lub_peak_width_half_prominence(coarse_envelope, first_idx, first_prominence,))

        next_widths = []

        for idx in next_three:

            prominence = (_lub_peak_prominence(coarse_envelope, idx, LUB_MIN_PEAK_DISTANCE_SAMPLES,))

            next_widths.append(_lub_peak_width_half_prominence(coarse_envelope, idx, prominence,))

        next_width_level = float(np.median(next_widths))

        first_strength = float(strength_signal[first_idx])

        next_strength_level = float(np.median([strength_signal[idx] for idx in next_three]))

        high_rotation_lead = (next_gyro_level > 0.0 and first_gyro_median > LUB_LEADING_GYRO_RATIO * next_gyro_level)

        narrow_weak_lead = (
            next_width_level > 0.0
            and
            first_width < LUB_LEADING_WIDTH_RATIO * next_width_level
            and
            first_strength < LUB_LEADING_STRENGTH_RATIO * next_strength_level
        )

        if (high_rotation_lead or narrow_weak_lead):
            accepted = accepted[1:]
        else:
            break

    if not accepted:
        return 0

    # -----------------------------------------------------
    # 7) Keep the main repeated pumping bout.
    #    Isolated movements far away from the repeated pulse
    #    sequence are not treated as pump strokes.
    # -----------------------------------------------------
    groups = []
    current_group = []

    for idx in accepted:

        if (not current_group or idx - current_group[-1] <= LUB_PUMP_BOUT_MAX_GAP_SAMPLES):
            current_group.append(idx)

        else:
            groups.append(current_group)

            current_group = [idx]

    if current_group:
        groups.append(current_group)

    pump_group = max(groups, key=lambda group: (len(group), sum(float(strength_signal[idx]) for idx in group),),)

    # -----------------------------------------------------
    # 8) Adaptive cadence-tail protection.
    #
    # Only applied AFTER the main repeated pump bout has been
    # selected. This avoids cutting a genuine early/slow pump.
    # Once 3 pumps establish the recent rhythm, a later candidate
    # whose gap is > 1.8x the recent median interval is treated
    # as post-pump handling.
    # -----------------------------------------------------
    if (
        len(pump_group) >= LUB_CADENCE_MIN_PRIOR_PUMPS + 1
    ):

        cadence_cut_index = None

        for j in range(LUB_CADENCE_MIN_PRIOR_PUMPS, len(pump_group),):

            recent_start = max(1, j - LUB_CADENCE_MIN_PRIOR_PUMPS + 1,)

            recent_gaps = [pump_group[k] - pump_group[k - 1] for k in range(recent_start, j,)]

            if not recent_gaps:
                continue

            recent_gap_level = float(np.median(recent_gaps))

            current_gap = float(pump_group[j] - pump_group[j - 1])

            if (recent_gap_level > 0.0 and current_gap > LUB_CADENCE_MAX_GAP_RATIO * recent_gap_level):
                cadence_cut_index = j
                break

        if cadence_cut_index is not None:
            pump_group = (pump_group[:cadence_cut_index])

    # -----------------------------------------------------
    # 9) Weak end-tail protection.
    #    After a stable run of pumps, two consecutive peaks
    #    that are both much weaker than the recent pump level
    #    are treated as the bottle-handling tail.
    # -----------------------------------------------------
    if (
        len(pump_group) >= LUB_WEAK_TAIL_MIN_PRIOR_PUMPS + 2
    ):

        strengths = [float(strength_signal[idx]) for idx in pump_group]

        cut_index = None

        for j in range(LUB_WEAK_TAIL_MIN_PRIOR_PUMPS, len(pump_group) - 1,):

            recent_start = max(0, j - LUB_WEAK_TAIL_MIN_PRIOR_PUMPS,)

            recent_level = float(np.median(strengths[recent_start:j]))

            weak_limit = (LUB_WEAK_TAIL_RATIO * recent_level)

            if (strengths[j] < weak_limit and strengths[j + 1] < weak_limit):
                cut_index = j
                break

        if cut_index is not None:
            pump_group = (pump_group[:cut_index])

    return int(len(pump_group))

def event_display_name(action, number,):
    base = ACTION_DISPLAY.get(action, action,)

    if action in {"insert_bolt_nut", "hand_tighten", "wrench_tighten",}:
        return (f"{base} #{number}")

    if number > 1:
        return (f"{base} #{number}")

    return base

def reset_monitoring_state():
    st.session_state.buffer.clear()
    st.session_state.time_buffer.clear()

    st.session_state.raw_prediction = "Waiting..."
    st.session_state.raw_confidence = 0.0

    st.session_state.confirmed_prediction = "Waiting..."
    st.session_state.confirmed_confidence = 0.0

    st.session_state.prediction_history.clear()
    st.session_state.insert_bolt_confirmation_history.clear()

    st.session_state.vote_count = 0
    st.session_state.votes_required = 0

    st.session_state.raw_line = "No data yet"
    st.session_state.samples_since_prediction = 0
    st.session_state.first_prediction_done = False
    st.session_state.probabilities = None

    st.session_state.reading_log_count = 0
    st.session_state.saved_log_path = None

    # Added diagnostic log state
    st.session_state.window_log_count = 0
    st.session_state.saved_window_log_path = None
    st.session_state.prediction_window_index = 0

    st.session_state.idle_score = None
    st.session_state.is_rule_idle = False
    st.session_state.idle_exit_counter = 0

    st.session_state.action_events = []
    st.session_state.action_counts = Counter()

    st.session_state.last_event_time = {}
    st.session_state.last_counted_action = None
    st.session_state.released_since_last_event = True

    st.session_state.wrench_count = 0
    st.session_state.wrench_rearmed = True

    st.session_state.collect_transition = False
    st.session_state.transition_samples = []
    st.session_state.wrench_transitions = []
    st.session_state.last_transition_result = None

    st.session_state.wrench_transition_state = "idle"
    st.session_state.wrench_rearm_reference_acc = None
    st.session_state.wrench_rearm_sample_count = 0
    st.session_state.wrench_rearm_counter = 0
    st.session_state.wrench_rearm_angle_deg = 0.0
    st.session_state.wrench_rearm_accel_delta = 0.0
    st.session_state.wrench_transition_post_count = 0
    st.session_state.wrench_motion_prebuffer = deque(maxlen=WRENCH_REPOSITION_PREBUFFER_SAMPLES)

    st.session_state.wrench_rearm_samples = []
    st.session_state.wrench_rearm_metrics = None
    st.session_state.wrench_reference_axis = None

    st.session_state.recent_raw_samples = deque(maxlen=LUBRICATION_PREBUFFER_SAMPLES)
    st.session_state.lubrication_active = False
    st.session_state.lubrication_samples = []
    st.session_state.lubrication_nonmatch_windows = 0
    st.session_state.lubrication_stroke_count = 0
    st.session_state.last_lubrication_stroke_count = 0

    st.session_state.monitoring_started_at = (time.time())

    st.session_state.accept_events = True

    st.session_state.wrench_waiting_for_fresh_prediction = False
def can_count_event(action):
    if action == "idle":
        return False

    now = time.time()

    last_time = (st.session_state.last_event_time.get(action, 0.0,))

    cooldown = (EVENT_COOLDOWN_SEC.get(action, 2.0,))

    if (now - last_time < cooldown):
        return False

    # Wrench is special:
    # after the first wrench, the next wrench is allowed only after
    # cooldown + a detected reposition movement have re-armed it.
    if action == "wrench_tighten":
        if st.session_state.wrench_count == 0:
            return True

        return bool(st.session_state.wrench_rearmed)

    # Other repeated actions use release.
    if (
        REQUIRE_RELEASE_FOR_REPEAT
        and
        st.session_state.last_counted_action == action
        and
        not st.session_state.released_since_last_event
    ):
        return False

    return True

def record_confirmed_action(action, confidence,):
    """
    Record one confirmed physical action.

    Does NOT judge the full procedure live.
    That is done after Stop.
    """

    # Stop must freeze the procedure history immediately.
    if not st.session_state.accept_events:
        return

    if action == "idle":
        return

    if not can_count_event(action):
        return

    now = time.time()

    # -----------------------------------------------------
    # WRENCH EVENT
    # -----------------------------------------------------
    if action == "wrench_tighten":
        next_wrench_number = st.session_state.wrench_count + 1
        st.session_state.wrench_count = next_wrench_number
        st.session_state.prediction_history.clear()
        st.session_state.vote_count = 0
        st.session_state.votes_required = 0

        if next_wrench_number < 4:
            st.session_state.wrench_rearmed = False
            st.session_state.wrench_waiting_for_fresh_prediction = False
            st.session_state.wrench_transition_state = "waiting_reposition"
            st.session_state.collect_transition = False
            st.session_state.transition_samples = []

            # Learn this worker's current tightening axis from the same
            # rolling IMU window that exists when the wrench is confirmed.
            current_wrench_window = list(st.session_state.buffer)

            st.session_state.wrench_reference_axis = (learn_wrench_rotation_axis(current_wrench_window))

            recent_for_reference = list(st.session_state.recent_raw_samples)

            if recent_for_reference:
                reference_count = min(WRENCH_REARM_REFERENCE_SAMPLES, len(recent_for_reference),)

                reference_arr = np.asarray(recent_for_reference[-reference_count:], dtype=np.float64,)

                st.session_state.wrench_rearm_reference_acc = (reference_arr[:, :3].mean(axis=0))
            else:
                st.session_state.wrench_rearm_reference_acc = None

            st.session_state.wrench_rearm_sample_count = 0
            st.session_state.wrench_rearm_counter = 0
            st.session_state.wrench_rearm_angle_deg = 0.0
            st.session_state.wrench_rearm_accel_delta = 0.0
            st.session_state.wrench_transition_post_count = 0
            st.session_state.wrench_motion_prebuffer.clear()
            st.session_state.wrench_rearm_samples = []
        else:
            # All three between-wrench transitions are now available.
            # Finalize them together using the trained CAC/ACA sequence model.
            finalize_wrench_transitions_relative_multiwindow()

            st.session_state.wrench_rearmed = False
            st.session_state.wrench_transition_state = "idle"
            st.session_state.collect_transition = False
            st.session_state.transition_samples = []

    # -----------------------------------------------------
    # LUBRICATION EVENT
    # -----------------------------------------------------
    if action == "lubrication" and not st.session_state.lubrication_active:
        st.session_state.lubrication_active = True
        st.session_state.lubrication_nonmatch_windows = 0
        st.session_state.lubrication_samples = [np.asarray(s, dtype=np.float32).copy() for s in st.session_state.recent_raw_samples]
        st.session_state.lubrication_stroke_count = count_lubrication_strokes(st.session_state.lubrication_samples)

    # -----------------------------------------------------
    # RECORD ACTION EVENT
    # -----------------------------------------------------
    st.session_state.action_counts[action] += 1

    action_number = (st.session_state.action_counts[action])

    elapsed = (now - st.session_state.monitoring_started_at)

    st.session_state.action_events.append(
        {
            'action': action,
            'number': action_number,
            'display': event_display_name(action, action_number),
            'confidence': float(confidence),
            'elapsed_sec': float(elapsed),
            "lubrication_strokes": (
                int(st.session_state.lubrication_stroke_count) if action == 'lubrication' else None
            ),
        }
    )

    st.session_state.last_event_time[action] = now

    st.session_state.last_counted_action = (action)

    st.session_state.released_since_last_event = False
def render_action_history():
    events = st.session_state.action_events
    event_rows = []

    transition_map = {int(item.get("to_wrench_number", -1)): item for item in st.session_state.wrench_transitions}

    for idx, event in enumerate(events, start=1):
        transition_html = ""
        if event.get("action") == "wrench_tighten":
            wrench_number = int(event.get("number", 0))
            transition = transition_map.get(wrench_number)
            if transition is not None and wrench_number >= 2:
                observed = str(transition.get("observed", "unknown")).upper()
                expected = str(transition.get("expected", "unknown")).upper()
                ok = bool(transition.get("ok", False))
                state_class = "ok" if ok else "bad"
                transition_html = (
                    '<div class="ma-transition-note">'
                    f'Transition: <b>{html.escape(observed)}</b> · expected {html.escape(expected)} '
                    f'<span class="{state_class}">{"PASS" if ok else "CHECK"}</span>'
                    '</div>'
                )

        elif event.get("action") == "lubrication":
            strokes = event.get("lubrication_strokes")
            if strokes is not None:
                transition_html = f'<div class="ma-transition-note">Pump strokes detected: <b>{int(strokes)}</b></div>'

        event_rows.append(
            '<div class="ma-event">'
            f'<div class="ma-event-index">{idx:02d}</div>'
            '<div>'
            f'<div class="ma-event-name">{html.escape(str(event["display"]))}</div>'
            f'<div class="ma-event-meta">Detected at {float(event["elapsed_sec"]):.1f}s</div>'
            '</div>'
            f'<div class="ma-event-confidence">{float(event["confidence"]):.0%}</div>'
            f'{transition_html}'
            '</div>'
        )

    if event_rows:
        body_html = '<div class="ma-timeline">' + "".join(event_rows) + '</div>'
    else:
        body_html = (
            '<div class="ma-empty"><div>WAITING FOR CONFIRMED ACTIONS<br>'
            '<span style="font-size:.72rem;color:#4c7690">Session events will populate automatically.</span>'
            '</div></div>'
        )

    panel_html = (
        '<div class="ma-panel">'
        '<div class="ma-panel-head">'
        '<div class="ma-panel-title">SESSION TIMELINE</div>'
        f'<div class="ma-panel-badge">{len(events)} EVENTS</div>'
        '</div>'
        + body_html
        + '</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)

def render_expected_procedure():
    steps = [
        ("⌁", "01", "Cleaning / inspection", "Inspect and prepare the flange surfaces."),
        ("◇", "02", "Place / align gasket", "Position the gasket correctly."),
        ("●", "03", "Lubrication", "Lubricate bolt threads and bearing surfaces before insertion."),
        ("⬡", "04", "Insert bolt • nut × 4", "Bolt #1 → Bolt #2 → Bolt #3 → Bolt #4"),
        ("✣", "05", "Hand tighten × 4", "Hand tighten #1 → #2 → #3 → #4"),
        ("⌕", "06", "Wrench tighten × 4", "Final tightening across the four bolts."),
    ]

    step_rows = []
    for icon, number, title, note in steps:
        step_rows.append(
            '<div class="ma-step">'
            f'<div class="ma-step-icon">{html.escape(icon)}</div>'
            f'<div class="ma-step-no">{number}</div>'
            '<div>'
            f'<div class="ma-step-title">{html.escape(title)}</div>'
            f'<div class="ma-step-note">{html.escape(note)}</div>'
            '</div></div>'
        )

    panel_html = (
        '<div class="ma-panel">'
        '<div class="ma-panel-head">'
        '<div class="ma-panel-title">REFERENCE WORKFLOW</div>'
        '<div class="ma-panel-badge">6 STAGES</div>'
        '</div>'
        '<div class="ma-workflow">' + "".join(step_rows) + '</div>'
        '<div class="ma-sequence">'
        '<div class="ma-sequence-label">WRENCH SEQUENCE</div>'
        '<div class="ma-sequence-value">CROSS → ADJACENT → CROSS</div>'
        '<div class="ma-sequence-hazard"></div>'
        '</div>'
        '</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)

def evaluate_procedure_compliance():
    """
    Fast post-Stop compliance evaluation.

    Bolt work is flexible:
    insert -> hand -> insert -> hand is allowed,
    and several inserts may happen before hand-tightening.

    Rules:
    - cleaning before assembly work
    - gasket before bolt work
    - hand-tighten count can never get ahead of insert count
    - lubrication occurs before bolt insertion and tightening
    - wrench starts only after 4 inserts + 4 hand-tightens
    - required action counts are checked
    - wrench transition pattern is CROSS -> ADJACENT -> CROSS
    """
    events = [event["action"] for event in st.session_state.action_events]

    counts = Counter(events)

    expected_counts = {
        'clean_inspect': 1,
        'place_align_gasket': 1,
        'insert_bolt_nut': 4,
        'hand_tighten': 4,
        'lubrication': 1,
        'wrench_tighten': 4,
    }

    # -----------------------------------------------------
    # 1) ACTION COMPLETION / COUNT ACCURACY = 50%
    # Missing and extra confirmed events both reduce the score.
    # -----------------------------------------------------
    required_total = sum(expected_counts.values())

    total_count_error = sum(abs(counts.get(action, 0) - expected) for action, expected in expected_counts.items())

    completion_ratio = max(0.0, 1.0 - (total_count_error / required_total),)
    completion_score = completion_ratio * 50.0

    # -----------------------------------------------------
    # 2) PROCEDURE ORDER RULES = 30%
    # -----------------------------------------------------
    order_checks = []
    order_applicable = []

    # Cleaning before any assembly work.
    if "clean_inspect" in events:
        clean_idx = events.index("clean_inspect")
        later_indices = [
            i for i, action in enumerate(events)
            if action in {'place_align_gasket', 'insert_bolt_nut', 'hand_tighten', 'lubrication', 'wrench_tighten'}
        ]
        clean_ok = (not later_indices or clean_idx < min(later_indices))
    else:
        clean_ok = False

    order_checks.append(("Cleaning before assembly work", clean_ok))
    clean_applicable = (
        "clean_inspect" in events
        or any(action in {'place_align_gasket', 'lubrication', 'insert_bolt_nut', 'hand_tighten', 'wrench_tighten'} for action in events)
    )
    order_applicable.append(clean_applicable)

    # Gasket before bolt work.
    if "place_align_gasket" in events:
        gasket_idx = events.index("place_align_gasket")
        bolt_indices = [i for i, action in enumerate(events) if action in {'lubrication', 'insert_bolt_nut', 'hand_tighten', 'wrench_tighten'}]
        gasket_ok = (not bolt_indices or gasket_idx < min(bolt_indices))
    else:
        gasket_ok = False

    order_checks.append(("Gasket before bolt work", gasket_ok))
    gasket_applicable = (
        "place_align_gasket" in events
        or any(action in {'lubrication', 'insert_bolt_nut', 'hand_tighten', 'wrench_tighten'} for action in events)
    )
    order_applicable.append(gasket_applicable)

    # Insert/hand-tighten may alternate, but hand-tighten
    # can never get ahead of inserted bolts.
    inserts_seen = 0
    hands_seen = 0
    hand_order_ok = True

    for action in events:
        if action == "insert_bolt_nut":
            inserts_seen += 1

        elif action == "hand_tighten":
            hands_seen += 1

            if hands_seen > inserts_seen:
                hand_order_ok = False
                break

    order_checks.append(("Hand-tighten only after a bolt is available", hand_order_ok,))
    order_applicable.append("hand_tighten" in events)

    # Lubrication must happen before bolt insertion and tightening.
    if "lubrication" in events:
        lubrication_idx = events.index("lubrication")
        bolt_work_indices = [
            i for i, action in enumerate(events)
            if action in {'insert_bolt_nut', 'hand_tighten', 'wrench_tighten'}
        ]
        lubrication_ok = (not bolt_work_indices or lubrication_idx < min(bolt_work_indices))
    else:
        lubrication_ok = False

    order_checks.append(("Lubrication before bolt insertion", lubrication_ok))
    lubrication_applicable = (
        "lubrication" in events
        or any(action in {'insert_bolt_nut', 'hand_tighten', 'wrench_tighten'} for action in events)
    )
    order_applicable.append(lubrication_applicable)

    # First wrench may begin only after all 4 bolts have been
    # inserted and hand-tightened.
    inserts_before_wrench = 0
    hands_before_wrench = 0
    first_wrench_found = False

    for action in events:
        if action == "wrench_tighten":
            first_wrench_found = True
            break

        if action == "insert_bolt_nut":
            inserts_before_wrench += 1
        elif action == "hand_tighten":
            hands_before_wrench += 1

    wrench_start_ok = (first_wrench_found and inserts_before_wrench >= 4 and hands_before_wrench >= 4)

    order_checks.append(("Wrench starts after all 4 bolts are hand-tightened", wrench_start_ok,))
    order_applicable.append(first_wrench_found)

    # Score only rules that have actually become testable.
    # This prevents untouched rules from earning free compliance points at startup.
    active_order_checks = [
        passed
        for (_, passed), applicable in zip(order_checks, order_applicable)
        if applicable
    ]

    if active_order_checks:
        passed_order = sum(1 for passed in active_order_checks if passed)
        order_score = passed_order / len(active_order_checks) * 30.0
    else:
        order_score = 0.0

    # -----------------------------------------------------
    # 3) WRENCH SEQUENCE = 20%
    # -----------------------------------------------------
    expected_wrench_sequence = ['cross', 'adjacent', 'cross']

    transition_by_target = {result.get("to_wrench_number"): result for result in st.session_state.wrench_transitions}

    wrench_rows = []
    correct_wrench_transitions = 0

    for transition_number, expected in enumerate(expected_wrench_sequence, start=2,):
        result = transition_by_target.get(transition_number)

        observed = (result.get("observed", "unknown") if result else "missing")

        correct = observed == expected

        if correct:
            correct_wrench_transitions += 1

        wrench_rows.append(
            {
                'Transition': f'Wrench #{transition_number - 1} -> #{transition_number}',
                'Expected': expected.upper(),
                'Detected': observed.upper(),
                'Status': 'RIGHT' if correct else 'WRONG',
            }
        )

    wrench_score = (correct_wrench_transitions / len(expected_wrench_sequence) * 20.0)

    overall_score = (completion_score + order_score + wrench_score)

    all_counts_exact = all(counts.get(action, 0) == expected for action, expected in expected_counts.items())

    all_order_ok = all(passed for _, passed in order_checks)

    all_wrench_ok = (correct_wrench_transitions == len(expected_wrench_sequence))

    if (all_counts_exact and all_order_ok and all_wrench_ok):
        status = "PROCEDURE COMPLIANT"
    elif overall_score >= 75:
        status = "COMPLETED WITH DEVIATIONS"
    else:
        status = "REVIEW REQUIRED"

    completion_rows = []

    for action, expected in expected_counts.items():
        detected = counts.get(action, 0)

        if detected == expected:
            row_status = "RIGHT"
        elif detected < expected:
            row_status = "MISSING"
        else:
            row_status = "EXTRA"

        completion_rows.append(
            {
                'Action': ACTION_DISPLAY[action],
                'Required': expected,
                'Detected': detected,
                'Status': row_status,
            }
        )

    return {
        'overall_score': overall_score,
        'completion_score': completion_score,
        'order_score': order_score,
        'wrench_score': wrench_score,
        'status': status,
        'completion_rows': completion_rows,
        'order_checks': order_checks,
        'wrench_rows': wrench_rows,
    }

def render_procedure_compliance(show_details=False, show_compact=True):
    result = evaluate_procedure_compliance()

    overall = float(result["overall_score"])
    actions_pct = min(100.0, max(0.0, result["completion_score"] / 50.0 * 100.0))
    order_pct = min(100.0, max(0.0, result["order_score"] / 30.0 * 100.0))
    wrench_pct = min(100.0, max(0.0, result["wrench_score"] / 20.0 * 100.0))

    if result["status"] == "PROCEDURE COMPLIANT":
        status_icon = "✓"
        status_label = "PROCEDURE COMPLIANT"
        status_detail = "All required actions, order rules and wrench transitions passed."
        status_short = "NOMINAL"
    elif result["status"] == "COMPLETED WITH DEVIATIONS":
        status_icon = "!"
        status_label = "COMPLETED WITH DEVIATIONS"
        status_detail = "The procedure was completed, but one or more compliance checks need attention."
        status_short = "REVIEW"
    else:
        status_icon = "!"
        status_label = "REVIEW REQUIRED"
        status_detail = "The procedure has missing, extra or incorrectly ordered steps."
        status_short = "CHECK"

    expected_total = sum(int(row["Required"]) for row in result["completion_rows"])
    detected_total = sum(min(int(row["Detected"]), int(row["Required"])) for row in result["completion_rows"])
    completion_percent = (detected_total / expected_total * 100.0) if expected_total else 0.0
    deviations = sum(1 for row in result["completion_rows"] if row["Status"] != "RIGHT")
    deviations += sum(1 for _, passed in result["order_checks"] if not passed)
    deviations += sum(1 for row in result["wrench_rows"] if row["Status"] != "RIGHT")

    compact_html = (
        '<div class="ma-panel ma-compliance-panel">'
        '<div class="ma-panel-head"><div class="ma-panel-title">COMPLIANCE SCORE</div></div>'
        '<div class="ma-gauge-zone">'
        f'<div class="ma-gauge-arc" style="--score:{overall:.1f}"></div>'
        f'<div class="ma-gauge-value">{overall:.0f}%</div>'
        '<div class="ma-gauge-scale"><span>0</span><span>100</span></div>'
        '</div>'
        '<div class="ma-score-metrics">'
        '<div class="ma-score-row"><div class="ma-score-label">ACTION COMPLETION</div>'
        '<div class="ma-score-line">'
        f'<div class="ma-score-value">{result["completion_score"]:.0f}<small>/50</small></div>'
        f'<div class="ma-progress"><div style="width:{actions_pct:.1f}%"></div></div>'
        '</div></div>'
        '<div class="ma-score-row"><div class="ma-score-label">PROCEDURE ORDER</div>'
        '<div class="ma-score-line">'
        f'<div class="ma-score-value">{result["order_score"]:.0f}<small>/30</small></div>'
        f'<div class="ma-progress"><div style="width:{order_pct:.1f}%"></div></div>'
        '</div></div>'
        '<div class="ma-score-row"><div class="ma-score-label">WRENCH SEQUENCE</div>'
        '<div class="ma-score-line">'
        f'<div class="ma-score-value">{result["wrench_score"]:.0f}<small>/20</small></div>'
        f'<div class="ma-progress"><div style="width:{wrench_pct:.1f}%"></div></div>'
        '</div></div>'
        '</div>'
        '<div class="ma-status-box">'
        f'<div class="ma-status-icon">{html.escape(status_icon)}</div>'
        '<div>'
        f'<div class="ma-status-title">{html.escape(status_label)}</div>'
        f'<div class="ma-status-detail">{html.escape(status_detail)}</div>'
        '</div>'
        f'<div class="ma-status-pct">{overall:.0f}%</div>'
        '</div>'
        '<div class="ma-mini-grid">'
        '<div class="ma-mini"><div class="ma-mini-label">ACTIONS</div>'
        f'<div class="ma-mini-value">{completion_percent:.0f}%</div><div class="ma-mini-note">COMPLETED</div></div>'
        '<div class="ma-mini"><div class="ma-mini-label">DEVIATIONS</div>'
        f'<div class="ma-mini-value">{deviations}</div><div class="ma-mini-note">REQUIRING REVIEW</div></div>'
        '<div class="ma-mini"><div class="ma-mini-label">STATUS</div>'
        f'<div class="ma-mini-value" style="font-size:1.0rem;margin-top:7px">SYSTEM</div><div class="ma-mini-note green">{html.escape(status_short)}</div></div>'
        '</div>'
        '</div>'
    )
    if show_compact:
        st.markdown(compact_html, unsafe_allow_html=True)

    if not show_details:
        return

    completion_rows_html = []
    for row in result["completion_rows"]:
        badge = "ma-pass" if row["Status"] == "RIGHT" else "ma-warn"
        completion_rows_html.append(
            '<div class="ma-table-row">'
            f'<div>{html.escape(str(row["Action"]))}</div>'
            f'<div style="text-align:center">{row["Required"]}</div>'
            f'<div style="text-align:center">{row["Detected"]}</div>'
            f'<div style="text-align:center"><span class="ma-badge {badge}">{html.escape(str(row["Status"]))}</span></div>'
            '</div>'
        )

    rules_rows_html = []
    for label, passed in result["order_checks"]:
        rules_rows_html.append(
            '<div class="ma-rule-row">'
            f'<div><span class="ma-badge {"ma-pass" if passed else "ma-fail"}">{"PASS" if passed else "FAIL"}</span></div>'
            f'<div>{html.escape(label)}</div>'
            '</div>'
        )

    wrench_rows_html = []
    for row in result["wrench_rows"]:
        correct = row["Status"] == "RIGHT"
        wrench_rows_html.append(
            '<div class="ma-wrench-row">'
            f'<div>{html.escape(str(row["Transition"]).replace("->", "→"))}</div>'
            f'<div>{html.escape(str(row["Expected"]))}</div>'
            f'<div>{html.escape(str(row["Detected"]))}</div>'
            f'<div><span class="ma-badge {"ma-pass" if correct else "ma-fail"}">{"PASS" if correct else "FAIL"}</span></div>'
            '</div>'
        )

    detail_html = (
        '<div class="ma-detail-grid">'
        '<div class="ma-detail-panel"><div class="ma-detail-title">ACTION COMPLETION</div>'
        '<div class="ma-table-row head"><div>ACTION</div><div>REQ.</div><div>DET.</div><div>STATUS</div></div>'
        + ''.join(completion_rows_html) + '</div>'
        '<div class="ma-detail-panel"><div class="ma-detail-title">PROCEDURE RULES</div>'
        + ''.join(rules_rows_html) + '</div>'
        '</div>'
        '<div class="ma-detail-panel ma-wrench-grid">'
        '<div class="ma-detail-title">WRENCH TIGHTENING SEQUENCE</div>'
        '<div class="ma-wrench-row head"><div>TRANSITION</div><div>EXPECTED</div><div>DETECTED</div><div>STATUS</div></div>'
        + ''.join(wrench_rows_html) + '</div>'
    )
    st.markdown(detail_html, unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

init_state('ser', None)

init_state("running", False,)

init_state("session_id", "#7F3A-28B1",)

if ("buffer" not in st.session_state or st.session_state.buffer.maxlen != WINDOW_SIZE):
    st.session_state.buffer = deque(maxlen=WINDOW_SIZE)

# Added only so each prediction window can be mapped
# to its exact raw sample timestamps.
if (
    'time_buffer' not in st.session_state or st.session_state.time_buffer.maxlen != WINDOW_SIZE
):
    st.session_state.time_buffer = deque(maxlen=WINDOW_SIZE)

init_state("raw_prediction", "Waiting...",)

init_state("raw_confidence", 0.0,)

init_state("confirmed_prediction", "Waiting...",)

init_state("confirmed_confidence", 0.0,)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=VOTE_WINDOW)

if (
    "insert_bolt_confirmation_history" not in st.session_state
    or st.session_state.insert_bolt_confirmation_history.maxlen != INSERT_BOLT_CONFIRMATION_WINDOWS
):
    st.session_state.insert_bolt_confirmation_history = deque(maxlen=INSERT_BOLT_CONFIRMATION_WINDOWS)

init_state("vote_count", 0,)

init_state("votes_required", 0,)

init_state("raw_line", "No data yet",)

init_state("samples_since_prediction", 0,)

init_state("first_prediction_done", False,)

init_state("probabilities", None,)

init_state("reading_log_count", 0,)

init_state("saved_log_path", None,)

# Added diagnostic CSV state
init_state('window_log_count', 0)

init_state("saved_window_log_path", None,)

init_state("prediction_window_index", 0,)

init_state("idle_score", None,)

init_state("is_rule_idle", False,)

init_state("idle_exit_counter", 0,)

init_state("action_events", [],)

init_state("action_counts", Counter(),)

init_state("last_event_time", {},)

init_state("last_counted_action", None,)

init_state("released_since_last_event", True,)

init_state("wrench_count", 0,)

init_state("wrench_rearmed", True,)

init_state("collect_transition", False,)

init_state("transition_samples", [],)

init_state("wrench_transitions", [],)

init_state("last_transition_result", None,)

init_state("wrench_rearm_samples", [],)

init_state("wrench_rearm_metrics", None,)

init_state("wrench_reference_axis", None,)

init_state("wrench_transition_state", "idle")
init_state("wrench_rearm_reference_acc", None)
init_state("wrench_rearm_sample_count", 0)
init_state("wrench_rearm_counter", 0)
init_state("wrench_rearm_angle_deg", 0.0)
init_state("wrench_rearm_accel_delta", 0.0)
init_state("wrench_transition_post_count", 0)
if (
    "wrench_motion_prebuffer" not in st.session_state
    or
    st.session_state.wrench_motion_prebuffer.maxlen != WRENCH_REPOSITION_PREBUFFER_SAMPLES
):
    st.session_state.wrench_motion_prebuffer = deque(maxlen=WRENCH_REPOSITION_PREBUFFER_SAMPLES)
if "recent_raw_samples" not in st.session_state:
    st.session_state.recent_raw_samples = deque(maxlen=LUBRICATION_PREBUFFER_SAMPLES)
init_state("lubrication_active", False)
init_state("lubrication_samples", [])
init_state("lubrication_nonmatch_windows", 0)
init_state("lubrication_stroke_count", 0)
init_state("last_lubrication_stroke_count", 0)

init_state("monitoring_started_at", time.time(),)

init_state("accept_events", True,)

init_state("wrench_waiting_for_fresh_prediction", False,)

# =========================================================
# RAW READING CSV
# =========================================================

READING_LOG_FIELDS = [
    "time_ms",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "label",
    "raw_prediction",
    "raw_confidence",
    "confirmed_confidence",
    "rule_idle",
    "idle_gyro_p25",
    "wrench_count",
    "wrench_rearmed",
    "collect_transition",
]

# =========================================================
# PREDICTION WINDOW CSV
# =========================================================

WINDOW_LOG_BASE_FIELDS = [
    "window_index",
    "window_start_ms",
    "window_end_ms",
    "window_label",
    "rule_idle",
    "idle_gyro_p25",
    "model_top_prediction",
    "model_top_confidence",
    "raw_prediction",
    "raw_confidence",
    "candidate",
    "vote_count",
    "votes_required",
    "combined_confidence",
    "prediction_history",
    "confirmation_result",
    "confirmed_prediction",
    "confirmed_confidence",
    "new_action_event",
    "event_action",
    "event_number",
    "event_display",
    "event_elapsed_sec",
    "wrench_count",
    "wrench_rearmed",
    "collect_transition",
]

WINDOW_LOG_PROB_FIELDS = [f"prob_{str(class_name)}" for class_name in classes]

WINDOW_LOG_FIELDS = (WINDOW_LOG_BASE_FIELDS + WINDOW_LOG_PROB_FIELDS)

def initialize_reading_log():

    with open(READING_LOG_FILE, "w", newline="", encoding="utf-8",) as f:

        writer = csv.DictWriter(f, fieldnames=(READING_LOG_FIELDS),)

        writer.writeheader()

    st.session_state.saved_log_path = (READING_LOG_FILE)

    st.session_state.reading_log_count = 0

def append_reading_rows(rows):

    if not rows:
        return

    with open(READING_LOG_FILE, "a", newline="", encoding="utf-8",) as f:

        writer = csv.DictWriter(f, fieldnames=(READING_LOG_FIELDS),)

        writer.writerows(rows)

    st.session_state.reading_log_count += (len(rows))

# =========================================================
# WINDOW DIAGNOSTIC CSV FUNCTIONS
# =========================================================

def initialize_window_log():

    with open(WINDOW_LOG_FILE, "w", newline="", encoding="utf-8",) as f:

        writer = csv.DictWriter(f, fieldnames=(WINDOW_LOG_FIELDS),)

        writer.writeheader()

    st.session_state.saved_window_log_path = (WINDOW_LOG_FILE)

    st.session_state.window_log_count = 0

def append_window_log(row):

    with open(WINDOW_LOG_FILE, "a", newline="", encoding="utf-8",) as f:

        writer = csv.DictWriter(f, fieldnames=(WINDOW_LOG_FIELDS),)

        writer.writerow(row)

    st.session_state.window_log_count += 1

# =========================================================
# CONTROLS
# =========================================================

control_1, control_2, control_3 = st.columns(3)

with control_1:

    if st.button("▶  START MONITORING", use_container_width=True,):

        try:

            if (st.session_state.ser is not None):
                try:
                    st.session_state.ser.close()
                except Exception:
                    pass

            st.session_state.ser = (serial.Serial(PORT, BAUD_RATE, timeout=0.05,))

            time.sleep(2)

            st.session_state.ser.reset_input_buffer()

            reset_monitoring_state()
            initialize_reading_log()
            initialize_window_log()

            st.session_state.running = True
            st.session_state.accept_events = True
            st.success(f"Sensor connected on {PORT}.")

        except Exception as e:

            st.session_state.running = False
            st.error(f"Could not connect to " f"{PORT}: {e}")

with control_2:

    if st.button("■  STOP MONITORING", use_container_width=True,):

        # Freeze event creation immediately.
        st.session_state.accept_events = False

        st.session_state.running = False
        # CSV has already been written continuously during monitoring.
        # Nothing expensive is done here.

        if (
            st.session_state.ser is not None
        ):
            try:
                st.session_state.ser.close()
            except Exception:
                pass

        st.session_state.ser = None

with control_3:

    if st.button("↻  RESET SESSION", use_container_width=True,):

        if (st.session_state.ser is not None):
            try:
                st.session_state.ser.reset_input_buffer()
            except Exception:
                pass

        reset_monitoring_state()

        st.success("Session reset.")

# =========================================================
# LIVE MONITOR
# =========================================================

@st.fragment(run_every=0.25)
def live_monitor():

    if not st.session_state.running:

        has_recording = (st.session_state.saved_log_path and os.path.exists(st.session_state.saved_log_path))

        if has_recording:
            left_hist, center_proc, right_score = st.columns([1.03, 1.08, 0.92], gap="small")

            with left_hist:
                render_action_history()

            with center_proc:
                render_expected_procedure()

            with right_score:
                render_procedure_compliance(show_details=False)

            # Full post-run detail stays available below the command console.
            render_procedure_compliance(show_details=True, show_compact=False)

            with open(st.session_state.saved_log_path, "rb",) as f:
                st.download_button(
                    "DOWNLOAD RECORDED IMU READINGS",
                    data=f.read(),
                    file_name=READING_LOG_FILE,
                    mime="text/csv",
                    use_container_width=True,
                )

            if (st.session_state.saved_window_log_path and os.path.exists(st.session_state.saved_window_log_path)):
                with open(st.session_state.saved_window_log_path, "rb") as f:
                    st.download_button(
                        "DOWNLOAD LABELED PREDICTION WINDOWS",
                        data=f.read(),
                        file_name=WINDOW_LOG_FILE,
                        mime="text/csv",
                        use_container_width=True,
                        key="download_window_diagnostics",
                    )

                st.caption(f"{st.session_state.window_log_count:,} " "prediction windows recorded.")

            st.caption(f"{st.session_state.reading_log_count:,} " "raw sensor samples saved continuously.")

        else:
            standby_ribbon = (
                '<div class="ma-sensor-ribbon">'
                '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">CONFIRMED ACTION</div><div class="ma-ribbon-value blue">WAITING...</div></div>'
                '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">RAW CLASSIFICATION</div><div class="ma-ribbon-value">WAITING...</div></div>'
                '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">CONFIDENCE</div><div class="ma-ribbon-value green">0%</div></div>'
                '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">IDLE GYRO P25</div><div class="ma-ribbon-value">--</div></div>'
                '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">NEXT WRENCH</div><div class="ma-ribbon-value">READY</div></div>'
                '</div>'
            )
            st.markdown(standby_ribbon, unsafe_allow_html=True)

            left_hist, center_proc, right_score = st.columns([1.03, 1.08, 0.92], gap="small")
            with left_hist:
                render_action_history()
            with center_proc:
                render_expected_procedure()
            with right_score:
                render_procedure_compliance(show_details=False)

        return

    ser = st.session_state.ser

    if ser is None:

        st.warning("Serial connection unavailable.")

        return

    readings_added = 0

    new_log_rows = []

    # =====================================================
    # READ SERIAL
    # =====================================================

    for _ in range(100):

        try:

            if ser.in_waiting <= 0:
                break

            line = (ser.readline().decode("utf-8", errors="ignore",).strip())

            if not line:
                continue

            st.session_state.raw_line = line

            parts = line.split(",")

            if len(parts) != 7:
                continue

            try:

                sensor_time_ms = int(float(parts[0].strip()))

                values = np.array([float(value.strip()) for value in parts[1:]], dtype=np.float32)

            except ValueError:
                continue

            if len(values) != 6:
                continue

            st.session_state.buffer.append(values)

            # Timestamp buffer for diagnostic window mapping
            st.session_state.time_buffer.append(sensor_time_ms)

            current_label = str(st.session_state.confirmed_prediction)

            if current_label.startswith("Waiting"):
                current_label = "unconfirmed"

            current_raw = str(st.session_state.raw_prediction)

            if current_raw.startswith("Waiting"):
                current_raw = "unconfirmed"

            new_log_rows.append({
                'time_ms': sensor_time_ms,
                'ax': float(values[0]),
                'ay': float(values[1]),
                'az': float(values[2]),
                'gx': float(values[3]),
                'gy': float(values[4]),
                'gz': float(values[5]),
                'label': current_label,
                'raw_prediction': current_raw,
                'raw_confidence': float(st.session_state.raw_confidence),
                'confirmed_confidence': float(st.session_state.confirmed_confidence),
                'rule_idle': bool(st.session_state.is_rule_idle),
                'idle_gyro_p25': float(st.session_state.idle_score) if st.session_state.idle_score is not None else '',
                'wrench_count': int(st.session_state.wrench_count),
                'wrench_rearmed': bool(st.session_state.wrench_rearmed),
                'collect_transition': bool(st.session_state.collect_transition),
            })

            st.session_state.recent_raw_samples.append(values.copy())

            if st.session_state.lubrication_active:
                st.session_state.lubrication_samples.append(values.copy())
                if len(st.session_state.lubrication_samples) > LUBRICATION_MAX_SAMPLES:
                    st.session_state.lubrication_samples = st.session_state.lubrication_samples[-LUBRICATION_MAX_SAMPLES:]

            if (st.session_state.wrench_count > 0 and st.session_state.wrench_count < 4):
                state = st.session_state.wrench_transition_state

                if state == "waiting_reposition":
                    # -------------------------------------------------
                    # ADAPTIVE WRENCH-AXIS RE-ARM
                    # -------------------------------------------------
                    # Keep TWO buffers:
                    # 1) short prebuffer for the existing CROSS/ADJ capture
                    # 2) recent ~1 s re-arm buffer for the adaptive axis rule
                    st.session_state.wrench_motion_prebuffer.append(values.copy())

                    st.session_state.wrench_rearm_samples.append(values.copy())

                    if (len(st.session_state.wrench_rearm_samples) > WRENCH_REARM_MAX_SAMPLES):
                        st.session_state.wrench_rearm_samples = (
                            st.session_state.wrench_rearm_samples[-WRENCH_REARM_MAX_SAMPLES:]
                        )

                    recent_samples = list(st.session_state.wrench_rearm_samples)

                    total_rotation_deg = 0.0
                    perpendicular_fraction = 0.0

                    if (len(recent_samples) >= WRENCH_REARM_WINDOW_SAMPLES):
                        (total_rotation_deg, perpendicular_fraction) = wrench_reposition_metrics(recent_samples, st.session_state.wrench_reference_axis)

                    reposition_detected = (
                        len(recent_samples) >= WRENCH_REARM_WINDOW_SAMPLES
                        and
                        st.session_state.wrench_reference_axis is not None
                        and
                        total_rotation_deg >= WRENCH_REARM_MIN_TOTAL_ROTATION_DEG
                        and
                        perpendicular_fraction >= WRENCH_REARM_MIN_PERPENDICULAR_FRACTION
                    )

                    st.session_state.wrench_rearm_metrics = {
                        'samples': int(len(recent_samples)),
                        'total_rotation_deg': float(total_rotation_deg),
                        'perpendicular_fraction': float(perpendicular_fraction),
                    }

                    if reposition_detected:
                        # Re-arm immediately once a real move-to-next-bolt
                        # reposition has been detected.
                        st.session_state.wrench_rearmed = True
                        st.session_state.wrench_transition_state = "ready"
                        # Preserve the existing fixed CROSS / ADJACENT
                        # transition capture using the recent movement.
                        st.session_state.collect_transition = True
                        st.session_state.transition_samples = [
                            np.asarray(sample, dtype=np.float32).copy()
                            for sample
                            in st.session_state.wrench_motion_prebuffer
                        ]
                        st.session_state.wrench_transition_post_count = 0

                        # Clear all old votes so the next wrench must be
                        # confirmed from completely fresh predictions.
                        st.session_state.prediction_history.clear()
                        st.session_state.vote_count = 0
                        st.session_state.votes_required = 0

                        st.session_state.confirmed_prediction = "Waiting for fresh wrench..."
                        st.session_state.confirmed_confidence = 0.0

                        st.session_state.wrench_waiting_for_fresh_prediction = True
                elif state == "ready":
                    # Once re-armed, continue the existing fixed transition
                    # capture until enough post-reposition samples are present.
                    if st.session_state.collect_transition:
                        st.session_state.transition_samples.append(values.copy())

                        st.session_state.wrench_transition_post_count += 1

                        if (len(st.session_state.transition_samples) > WRENCH_MAX_TRANSITION_SAMPLES):
                            st.session_state.transition_samples = (
                                st.session_state.transition_samples[-WRENCH_MAX_TRANSITION_SAMPLES:]
                            )

                        if (st.session_state.wrench_transition_post_count >= WRENCH_TRANSITION_POST_SAMPLES):
                            frozen = list(st.session_state.transition_samples)

                            result = classify_wrench_transition(frozen)

                            target = (st.session_state.wrench_count + 1)

                            expected = expected_transition_for_wrench_number(target)

                            result["expected"] = expected
                            result["ok"] = None
                            result["to_wrench_number"] = target

                            st.session_state.wrench_transitions.append(result)
                            st.session_state.last_transition_result = result
                            st.session_state.collect_transition = False

            readings_added += 1
            readings_added += 1

            st.session_state.samples_since_prediction += 1

        except Exception:
            continue

    append_reading_rows(new_log_rows)

    buffer_length = len(st.session_state.buffer)

    # -----------------------------------------------------
    # PREDICTION TIMING
    # -----------------------------------------------------
    should_predict = False

    if (buffer_length == WINDOW_SIZE and not st.session_state.first_prediction_done):
        should_predict = True

    elif (
        buffer_length == WINDOW_SIZE
        and
        st.session_state.first_prediction_done
        and
        st.session_state.samples_since_prediction >= STEP_SIZE
    ):
        should_predict = True

    # -----------------------------------------------------
    # RULE IDLE + MODEL
    # -----------------------------------------------------
    if should_predict:

        window_2d = np.array(st.session_state.buffer, dtype=np.float32,)

        is_idle, idle_score = (rule_based_idle(window_2d))

        st.session_state.idle_score = (idle_score)

        window = np.expand_dims(window_2d, axis=0,)

        window_scaled = (window - mean) / (std + 1e-8)

        probabilities = model.predict(window_scaled, verbose=0,)[0]

        st.session_state.probabilities = (probabilities)

        # =================================================
        # DIAGNOSTIC ONLY:
        # determine what model would choose even if idle wins.
        #
        # Same exact non-idle calculation as existing logic.
        # =================================================

        maintenance_probs = np.array([probabilities[i] for i in MAINTENANCE_CLASS_INDICES], dtype=np.float64)

        local_index = int(np.argmax(maintenance_probs))

        predicted_index = (MAINTENANCE_CLASS_INDICES[local_index])

        model_top_prediction = str(classes[predicted_index])

        model_top_confidence = float(probabilities[predicted_index])

        # =================================================
        # LUBRICATION BOUT GATING / STROKE COUNT
        # =================================================
        if st.session_state.lubrication_active:
            if model_top_prediction == "lubrication" and not is_idle:
                st.session_state.lubrication_nonmatch_windows = 0
            else:
                st.session_state.lubrication_nonmatch_windows += 1
            st.session_state.lubrication_stroke_count = count_lubrication_strokes(st.session_state.lubrication_samples)
            for event in reversed(st.session_state.action_events):
                if event.get("action") == "lubrication":
                    event["lubrication_strokes"] = int(st.session_state.lubrication_stroke_count)
                    break
            if st.session_state.lubrication_nonmatch_windows >= LUBRICATION_END_NONMATCH_WINDOWS:
                st.session_state.last_lubrication_stroke_count = int(st.session_state.lubrication_stroke_count)
                st.session_state.lubrication_active = False
                st.session_state.lubrication_nonmatch_windows = 0
                st.session_state.lubrication_samples = []

        # =================================================
        # FINAL RAW DECISION
        # =================================================

        if is_idle:

            prediction = "idle"

            confidence = 1.0

        else:

            prediction = (model_top_prediction)

            confidence = (model_top_confidence)

        st.session_state.raw_prediction = (prediction)

        st.session_state.raw_confidence = (confidence)

        # =================================================
        # RELEASE LOGIC
        # =================================================

        if prediction == "idle":

            st.session_state.released_since_last_event = True
        elif (
            st.session_state.last_counted_action is not None
            and
            st.session_state.confirmed_prediction != st.session_state.last_counted_action
        ):

            st.session_state.released_since_last_event = True
        # =================================================
        # CROSS / ADJACENT CAPTURE
        # =================================================
        # Transition classification is completed earlier from the
        # fixed reposition segment. It does not wait for this next
        # wrench prediction.

        # =================================================
        # SMOOTHING HISTORY
        # =================================================

        prediction_allowed_to_vote = True

        if (prediction == "wrench_tighten" and st.session_state.wrench_count > 0):

            if not st.session_state.wrench_rearmed:

                prediction_allowed_to_vote = False

                st.session_state.prediction_history.clear()

            elif (st.session_state.wrench_waiting_for_fresh_prediction):

                st.session_state.prediction_history.clear()

                st.session_state.wrench_waiting_for_fresh_prediction = False
        if prediction_allowed_to_vote:
            st.session_state.prediction_history.append((prediction, confidence,))

            if prediction == "insert_bolt_nut":
                st.session_state.insert_bolt_confirmation_history.append(confidence)
            else:
                st.session_state.insert_bolt_confirmation_history.clear()

        recent_classes = [pred for pred, conf in st.session_state.prediction_history]

        if prediction == "idle":

            candidate = "idle"

            vote_count = 1

            votes_required = 1

        elif not prediction_allowed_to_vote:

            candidate = None

            vote_count = 0

            votes_required = (CONFIRMATION_REQUIRED.get(prediction, 2,))

        elif not recent_classes:

            candidate = None

            vote_count = 0

            votes_required = (CONFIRMATION_REQUIRED.get(prediction, 2,))

        else:

            counts = Counter(recent_classes)

            candidate, vote_count = (counts.most_common(1)[0])

            if candidate == "idle":

                candidate = prediction

                vote_count = sum(1 for p in recent_classes if p == prediction)

            votes_required = (CONFIRMATION_REQUIRED.get(candidate, 2,))

            if candidate == "insert_bolt_nut":
                vote_count = len(st.session_state.insert_bolt_confirmation_history)
                votes_required = INSERT_BOLT_CONFIRMATION_WINDOWS

        st.session_state.vote_count = (vote_count)

        st.session_state.votes_required = (votes_required)

        # =================================================
        # DIAGNOSTIC SNAPSHOT
        #
        # Important because wrench confirmation clears votes.
        # Saving this copy DOES NOT affect prediction_history.
        # =================================================

        diagnostic_prediction_history = list(st.session_state.prediction_history)

        combined_confidence = None

        confirmation_result = "waiting"
        event_recorded = False

        event_data = None

        if prediction == "idle":

            confirmation_result = "idle"
        elif not prediction_allowed_to_vote:

            confirmation_result = "vote_blocked"
        elif candidate is None:

            confirmation_result = "no_candidate"
        elif vote_count < votes_required:

            confirmation_result = "waiting_for_votes"
        # =================================================
        # ORIGINAL CONFIRMATION LOGIC
        # =================================================

        if (
            candidate is not None and vote_count >= votes_required
        ):

            if candidate == "idle":

                st.session_state.confirmed_prediction = "idle"
                st.session_state.confirmed_confidence = (1.0)

                confirmation_result = "idle"
            else:

                if candidate == "insert_bolt_nut":
                    matching_confidences = list(st.session_state.insert_bolt_confirmation_history)
                else:
                    matching_confidences = [conf for pred, conf in st.session_state.prediction_history if pred == candidate]

                combined_confidence = float(np.mean(matching_confidences))

                if (combined_confidence >= MIN_COMBINED_CONFIDENCE):

                    st.session_state.confirmed_prediction = (candidate)

                    st.session_state.confirmed_confidence = (combined_confidence)

                    # Diagnostic only:
                    # compare history length before/after call.
                    events_before = len(st.session_state.action_events)

                    record_confirmed_action(candidate, combined_confidence,)

                    events_after = len(st.session_state.action_events)

                    if events_after > events_before:

                        event_recorded = True

                        event_data = (st.session_state.action_events[-1])

                        if candidate == "insert_bolt_nut":
                            st.session_state.insert_bolt_confirmation_history.clear()

                        confirmation_result = "event_recorded"
                    else:

                        confirmation_result = "confirmed_but_event_blocked"
                else:

                    confirmation_result = "combined_confidence_below_threshold"
        # =================================================
        # SAVE ONE ROW FOR THIS PREDICTION WINDOW
        # =================================================

        st.session_state.prediction_window_index += 1

        if (len(st.session_state.time_buffer) == WINDOW_SIZE):

            window_start_ms = int(st.session_state.time_buffer[0])

            window_end_ms = int(st.session_state.time_buffer[-1])

        else:

            window_start_ms = ""

            window_end_ms = ""

        history_text = " | ".join(f"{pred}:{conf:.4f}" for pred, conf in diagnostic_prediction_history)

        window_row = {
            'window_index': st.session_state.prediction_window_index,
            'window_start_ms': window_start_ms,
            'window_end_ms': window_end_ms,
            'window_label': prediction,
            'rule_idle': bool(is_idle),
            'idle_gyro_p25': float(idle_score),
            'model_top_prediction': model_top_prediction,
            'model_top_confidence': model_top_confidence,
            'raw_prediction': prediction,
            'raw_confidence': confidence,
            "candidate": candidate if candidate is not None else "",
            'vote_count': vote_count,
            'votes_required': votes_required,
            "combined_confidence": combined_confidence if combined_confidence is not None else "",
            'prediction_history': history_text,
            'confirmation_result': confirmation_result,
            'confirmed_prediction': st.session_state.confirmed_prediction,
            'confirmed_confidence': float(st.session_state.confirmed_confidence),
            'new_action_event': event_recorded,
            "event_action": event_data["action"] if event_data else "",
            "event_number": event_data["number"] if event_data else "",
            "event_display": event_data["display"] if event_data else "",
            "event_elapsed_sec": event_data["elapsed_sec"] if event_data else "",
            'wrench_count': int(st.session_state.wrench_count),
            'wrench_rearmed': bool(st.session_state.wrench_rearmed),
            'collect_transition': bool(st.session_state.collect_transition),
        }

        # Save ALL model probabilities.
        for i, class_name in enumerate(classes):

            window_row[f"prob_{str(class_name)}"] = float(probabilities[i])

        append_window_log(window_row)

        st.session_state.first_prediction_done = True
        st.session_state.samples_since_prediction = (0)

    # =====================================================
    # BLUE COMMAND LIVE DISPLAY
    # =====================================================

    confirmed_display = ACTION_DISPLAY.get(str(st.session_state.confirmed_prediction), str(st.session_state.confirmed_prediction))
    raw_display = ACTION_DISPLAY.get(str(st.session_state.raw_prediction), str(st.session_state.raw_prediction),)

    idle_value = (f"{st.session_state.idle_score:.0f}" if st.session_state.idle_score is not None else "--")

    if st.session_state.wrench_count == 0:
        wrench_state = "READY"
    elif st.session_state.wrench_rearmed:
        wrench_state = f"#{st.session_state.wrench_count + 1} READY"
    else:
        wrench_state = "REPOSITION"

    ribbon_html = (
        '<div class="ma-sensor-ribbon">'
        '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">CONFIRMED ACTION</div>'
        f'<div class="ma-ribbon-value blue">{html.escape(confirmed_display)}</div></div>'
        '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">RAW CLASSIFICATION</div>'
        f'<div class="ma-ribbon-value">{html.escape(raw_display)}</div></div>'
        '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">CONFIDENCE</div>'
        f'<div class="ma-ribbon-value green">{st.session_state.confirmed_confidence:.0%}</div></div>'
        '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">IDLE GYRO P25</div>'
        f'<div class="ma-ribbon-value">{idle_value}</div></div>'
        '<div class="ma-ribbon-cell"><div class="ma-ribbon-label">NEXT WRENCH</div>'
        f'<div class="ma-ribbon-value">{html.escape(wrench_state)}</div></div>'
        '</div>'
    )
    st.markdown(ribbon_html, unsafe_allow_html=True)

    left_hist, center_proc, right_score = st.columns([1.03, 1.08, 0.92], gap="small")

    with left_hist:
        render_action_history()

    with center_proc:
        render_expected_procedure()

    with right_score:
        render_procedure_compliance(show_details=False)

    # =====================================================
    # TECHNICAL DETAILS
    # =====================================================

    with st.expander('Technical / debug information'):

        st.write("Model:", MODEL_PATH,)

        st.write("Preprocessing:", PREPROCESSING_PATH,)

        st.write("Saved model classes:", list(classes),)

        st.write("Maintenance classes used:", MAINTENANCE_CLASSES,)

        st.write("Idle source:", "P25 + HYSTERESIS",)

        st.write(
            "Idle gyro P25 thresholds:",
            {
                'enter_idle_below': IDLE_ENTER_GYRO_P25_THRESHOLD,
                'exit_idle_at_or_above': IDLE_EXIT_GYRO_P25_THRESHOLD,
                'exit_consecutive_windows': IDLE_EXIT_CONSECUTIVE_WINDOWS,
            },
        )

        st.write("Current idle gyro P95:", st.session_state.idle_score,)

        st.write("Model input:", model.input_shape,)

        st.write("Window size:", WINDOW_SIZE,)

        st.write("Step size:", STEP_SIZE,)

        st.write("New sensor samples this refresh:", readings_added,)

        st.write("Transition samples currently buffered:", len(st.session_state.transition_samples),)

        st.write("Wrench rearmed:", st.session_state.wrench_rearmed,)

        st.write("Wrench rearm samples:", len(st.session_state.wrench_rearm_samples),)

        st.write("Wrench transition state:", st.session_state.wrench_transition_state,)

        st.write(
            "Wrench adaptive-axis re-arm:",
            {
                'state': st.session_state.wrench_transition_state,
                "reference_axis": st.session_state.wrench_reference_axis.tolist() if st.session_state.wrench_reference_axis is not None else None,
                'metrics': st.session_state.wrench_rearm_metrics,
                'window_samples': WRENCH_REARM_WINDOW_SAMPLES,
                'min_total_rotation_deg': WRENCH_REARM_MIN_TOTAL_ROTATION_DEG,
                'min_perpendicular_fraction': WRENCH_REARM_MIN_PERPENDICULAR_FRACTION,
            },
        )

        st.write("Last wrench transition:", st.session_state.last_transition_result,)

        st.write("Lubrication active:", st.session_state.lubrication_active,)

        st.write("Lubrication stroke count:", st.session_state.lubrication_stroke_count,)

        st.write("Waiting for fresh wrench evidence:", st.session_state.wrench_waiting_for_fresh_prediction,)

        st.write("Accepting procedure events:", st.session_state.accept_events,)

        st.write("Raw samples logged:", st.session_state.reading_log_count,)

        st.write("Prediction windows logged:", st.session_state.window_log_count,)

        st.write("Raw serial:", st.session_state.raw_line,)

        if (st.session_state.probabilities is not None):

            probability_dict = {str(classes[i]): float(st.session_state.probabilities[i]) for i in range(len(classes))}

            st.bar_chart(probability_dict)

live_monitor()
