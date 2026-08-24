import os
import time
from collections import deque

import numpy as np
import serial
import streamlit as st
from tensorflow.keras.models import load_model


# =========================================================
# SETTINGS
# =========================================================

PORT = "COM3"
BAUD_RATE = 115200

# This is what our FINAL preprocessing used
EXPECTED_WINDOW_SIZE = 100
STEP_SIZE = 45

MODEL_PATH = "best_cnn_lstm_accuracy2.keras"
PREPROCESSING_PATH = "imu_preprocessing.npz"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MaintenAction",
    layout="wide"
)

st.title("MaintenAction")

st.write(
    "Live maintenance action recognition "
    "using ESP32 + MPU6050."
)


# =========================================================
# CHECK FILES EXIST
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: {MODEL_PATH}"
    )
    st.stop()

if not os.path.exists(PREPROCESSING_PATH):
    st.error(
        f"Preprocessing file not found: "
        f"{PREPROCESSING_PATH}"
    )
    st.stop()


# =========================================================
# LOAD MODEL + PREPROCESSING
# =========================================================
#
# We include file modification times in the cache.
#
# This means if you replace model3.keras with a newer file
# using the SAME filename, Streamlit will still reload it.
# =========================================================

@st.cache_resource
def load_everything(
    model_path,
    preprocessing_path,
    model_modified,
    prep_modified
):

    model = load_model(model_path)

    prep = np.load(
        preprocessing_path,
        allow_pickle=True
    )

    mean = prep["mean"]
    std = prep["std"]
    classes = prep["classes"]

    return model, mean, std, classes


model_modified = os.path.getmtime(MODEL_PATH)
prep_modified = os.path.getmtime(PREPROCESSING_PATH)


model, mean, std, classes = load_everything(
    MODEL_PATH,
    PREPROCESSING_PATH,
    model_modified,
    prep_modified
)

st.write("Model file:", MODEL_PATH)
st.write("Model input shape:", model.input_shape)

# =========================================================
# VERIFY MODEL
# =========================================================

model_window_size = int(model.input_shape[1])
model_channels = int(model.input_shape[2])


st.write("**Model file:**", MODEL_PATH)
st.write("**Model input shape:**", model.input_shape)
st.write("**Preprocessing file:**", PREPROCESSING_PATH)
st.write("**Classes:**", list(classes))


# Model must use six IMU channels
if model_channels != 6:

    st.error(
        f"Model expects {model_channels} channels. "
        "We need 6: ax, ay, az, gx, gy, gz."
    )

    st.stop()


# Final model should use 91 samples
if model_window_size != EXPECTED_WINDOW_SIZE:

    st.error(
        f"WRONG MODEL LOADED. "
        f"This model expects {model_window_size} samples, "
        f"but the final pipeline expects "
        f"{EXPECTED_WINDOW_SIZE}."
    )

    st.write(
        "Load the CNN-LSTM that was trained with "
        "Input(shape=(91, 6))."
    )

    st.stop()


WINDOW_SIZE = model_window_size


# =========================================================
# VERIFY PREPROCESSING
# =========================================================

if mean.shape[-1] != 6 or std.shape[-1] != 6:

    st.error(
        "The preprocessing file does not contain "
        "six-channel normalization values."
    )

    st.stop()


if model.output_shape[-1] != len(classes):

    st.error(
        "Number of model output classes does not "
        "match preprocessing class names."
    )

    st.stop()


st.success(
    "Model and preprocessing dimensions look correct."
)


# =========================================================
# SESSION STATE
# =========================================================

if "ser" not in st.session_state:
    st.session_state.ser = None

if "running" not in st.session_state:
    st.session_state.running = False


# Recreate buffer if the window size changed
if (
    "buffer" not in st.session_state
    or st.session_state.buffer.maxlen != WINDOW_SIZE
):

    st.session_state.buffer = deque(
        maxlen=WINDOW_SIZE
    )


if "prediction" not in st.session_state:
    st.session_state.prediction = "Waiting..."

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "raw_line" not in st.session_state:
    st.session_state.raw_line = "No data yet"

if "samples_since_prediction" not in st.session_state:
    st.session_state.samples_since_prediction = 0

if "first_prediction_done" not in st.session_state:
    st.session_state.first_prediction_done = False

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# =========================================================
# START / STOP
# =========================================================

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "Start monitoring",
        use_container_width=True
    ):

        try:

            # Close old serial connection
            if st.session_state.ser is not None:

                try:
                    st.session_state.ser.close()
                except Exception:
                    pass


            # Connect to ESP32
            st.session_state.ser = serial.Serial(
                PORT,
                BAUD_RATE,
                timeout=0.05
            )


            # ESP32 may reset when COM port opens
            time.sleep(2)


            # Remove startup messages / old serial data
            st.session_state.ser.reset_input_buffer()


            # Reset monitoring
            st.session_state.buffer.clear()

            st.session_state.prediction = "Waiting..."
            st.session_state.confidence = 0.0
            st.session_state.probabilities = None

            st.session_state.samples_since_prediction = 0
            st.session_state.first_prediction_done = False

            st.session_state.running = True


            st.success(
                f"Sensor connected on {PORT}."
            )


        except Exception as e:

            st.session_state.running = False

            st.error(
                f"Could not connect to {PORT}: {e}"
            )


with col2:

    if st.button(
        "Stop monitoring",
        use_container_width=True
    ):

        st.session_state.running = False


        if st.session_state.ser is not None:

            try:
                st.session_state.ser.close()
            except Exception:
                pass


        st.session_state.ser = None

        st.session_state.buffer.clear()

        st.session_state.samples_since_prediction = 0
        st.session_state.first_prediction_done = False

        st.info("Monitoring stopped.")


# =========================================================
# LIVE SENSOR MONITORING
# =========================================================

@st.fragment(run_every=0.25)
def live_monitor():

    if not st.session_state.running:

        st.info(
            "Press Start monitoring to begin."
        )

        return


    ser = st.session_state.ser


    if ser is None:

        st.warning(
            "Serial connection unavailable."
        )

        return


    # =====================================================
    # READ NEW SENSOR SAMPLES
    # =====================================================

    readings_added = 0


    # Limit prevents one refresh from getting stuck
    # reading indefinitely
    for _ in range(100):

        try:

            # If no complete serial data is waiting,
            # stop reading for this refresh.
            if ser.in_waiting <= 0:
                break


            line = (
                ser.readline()
                .decode(
                    "utf-8",
                    errors="ignore"
                )
                .strip()
            )


            if not line:
                continue


            st.session_state.raw_line = line


            # Expected Arduino output:
            #
            # time_ms,ax,ay,az,gx,gy,gz
            #
            # Example:
            # 610785,-4428,-2620,14740,1550,-5871,1246

            parts = line.split(",")


            if len(parts) != 7:
                continue


            try:

                # Ignore time_ms
                #
                # Keep ONLY:
                # ax, ay, az, gx, gy, gz

                values = np.array(
                    [
                        float(value.strip())
                        for value in parts[1:]
                    ],
                    dtype=np.float32
                )

            except ValueError:
                continue


            if len(values) != 6:
                continue


            st.session_state.buffer.append(
                values
            )


            readings_added += 1

            st.session_state.samples_since_prediction += 1


        except Exception:
            continue


    # =====================================================
    # SENSOR STATUS
    # =====================================================

    buffer_length = len(
        st.session_state.buffer
    )


    st.subheader("Sensor status")


    col_a, col_b, col_c, col_d = st.columns(4)


    with col_a:

        st.metric(
            "Window samples",
            f"{buffer_length}/{WINDOW_SIZE}"
        )


    with col_b:

        st.metric(
            "New samples",
            readings_added
        )


    with col_c:

        st.metric(
            "Samples to next prediction",
            (
                max(
                    STEP_SIZE
                    - st.session_state.samples_since_prediction,
                    0
                )
                if st.session_state.first_prediction_done
                else max(
                    WINDOW_SIZE - buffer_length,
                    0
                )
            )
        )


    with col_d:

        try:
            waiting = ser.in_waiting
        except Exception:
            waiting = 0

        st.metric(
            "Serial bytes waiting",
            waiting
        )


    st.progress(
        min(
            buffer_length / WINDOW_SIZE,
            1.0
        ),
        text=(
            f"Current window: "
            f"{buffer_length}/{WINDOW_SIZE}"
        )
    )


    # =====================================================
    # RAW SENSOR LINE
    # =====================================================

    with st.expander(
        "Raw sensor data"
    ):

        st.code(
            st.session_state.raw_line
        )


    # =====================================================
    # DECIDE WHETHER TO PREDICT
    # =====================================================

    should_predict = False


    # First prediction:
    # wait until full 91-sample window exists
    if (
        buffer_length == WINDOW_SIZE
        and not st.session_state.first_prediction_done
    ):

        should_predict = True


    # Following predictions:
    # wait for another 45 samples
    elif (
        buffer_length == WINDOW_SIZE
        and st.session_state.first_prediction_done
        and st.session_state.samples_since_prediction >= STEP_SIZE
    ):

        should_predict = True


    # =====================================================
    # PREDICTION
    # =====================================================

    if should_predict:

        # ---------------------------------------------
        # Convert rolling buffer to numpy
        #
        # Shape:
        # (91, 6)
        # ---------------------------------------------

        window = np.array(
            st.session_state.buffer,
            dtype=np.float32
        )


        # ---------------------------------------------
        # Add batch dimension
        #
        # (91, 6)
        #      ↓
        # (1, 91, 6)
        # ---------------------------------------------

        window = np.expand_dims(
            window,
            axis=0
        )


        # ---------------------------------------------
        # NORMALIZATION
        #
        # IMPORTANT:
        #
        # We DO NOT convert ax into g
        # or gyro into degrees/sec here.
        #
        # Training used the raw MPU6050 numbers,
        # so live inference must use the same values.
        #
        # We only apply the SAME mean/std scaling
        # that was used during training.
        # ---------------------------------------------

        window_scaled = (
            window - mean
        ) / (
            std + 1e-8
        )


        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        probabilities = model.predict(
            window_scaled,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(probabilities)
        )


        prediction = str(
            classes[predicted_index]
        )


        confidence = float(
            probabilities[predicted_index]
        )


        st.session_state.prediction = prediction

        st.session_state.confidence = confidence

        st.session_state.probabilities = probabilities


        # After first prediction,
        # require 45 NEW readings before predicting again
        st.session_state.first_prediction_done = True

        st.session_state.samples_since_prediction = 0


    # =====================================================
    # CURRENT ACTION
    # =====================================================

    st.subheader(
        "Current action"
    )


    col_pred, col_conf = st.columns(2)


    with col_pred:

        st.metric(
            "Prediction",
            st.session_state.prediction
        )


    with col_conf:

        st.metric(
            "Confidence",
            f"{st.session_state.confidence:.1%}"
        )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    probabilities = st.session_state.probabilities


    if probabilities is not None:

        st.subheader(
            "Class probabilities"
        )


        probability_dict = {

            str(classes[i]):
            float(probabilities[i])

            for i in range(
                len(classes)
            )
        }


        st.bar_chart(
            probability_dict
        )


    # =====================================================
    # DEBUG INFORMATION
    # =====================================================

    with st.expander(
        "Debug information"
    ):

        st.write(
            "Model input:",
            model.input_shape
        )

        st.write(
            "Window size:",
            WINDOW_SIZE
        )

        st.write(
            "Step size:",
            STEP_SIZE
        )

        st.write(
            "Classes:",
            list(classes)
        )

        st.write(
            "Training mean:",
            mean.reshape(-1)
        )

        st.write(
            "Training std:",
            std.reshape(-1)
        )


        if buffer_length == WINDOW_SIZE:

            current_window = np.array(
                st.session_state.buffer,
                dtype=np.float32
            )

            st.write(
                "Live raw mean:",
                current_window.mean(axis=0)
            )

            st.write(
                "Live raw std:",
                current_window.std(axis=0)
            )


# =========================================================
# RUN LIVE MONITOR
# =========================================================

live_monitor()