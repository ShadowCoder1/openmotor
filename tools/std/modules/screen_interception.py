"""Task module: intercepting or predicting a moving target on a screen with a tap, touch or mouse click, one row per trial.
First defined with Crowe2023_JVis (finger taps on a rolling-wheel target); also used for Schroeger2022_SciRep (mouse click at
the predicted time and place of a jumping circle) and the Reid & Dessing finger-touch studies. Columns hold the authors' trial
values; response_unit says whether positions are in cm, mm or pixels, and the description of each study's file says which
frame the numbers are in."""

DESCRIPTION = ("Screen interception: one row per trial in which a target moved (or jumped) on a screen and the person tapped, "
               "touched or clicked where and when they expected it, with the response position and time, the target's "
               "position at the response and the authors' hit, timing and position errors.")

# (column, type, values or units, description)
COLUMNS = [
    ("trial_original", "integer", "", "The authors' own trial number (file number or trial counter), as shared; empty where the deposit has none."),
    ("response_unit", "category", "cm, mm, pixels", "Unit of the position columns of this row (response, target and error positions)."),
    ("response_x", "number", "response_unit", "Horizontal position of the tap, touch or click, in the authors' screen frame (frame given in the study README)."),
    ("response_y", "number", "response_unit", "Vertical position of the tap, touch or click, in the authors' screen frame."),
    ("target_x_at_response", "number", "response_unit", "Horizontal position of the target at the moment of the response (where the authors computed it)."),
    ("target_y_at_response", "number", "response_unit", "Vertical position of the target at the moment of the response."),
    ("response_error_x", "number", "response_unit", "The authors' horizontal response error (response minus ideal or target position, in the sign the authors used; see the study README)."),
    ("response_error_y", "number", "response_unit", "The authors' vertical response error."),
    ("response_time_s", "number", "seconds", "Time of the response after the target (or the first stimulus) appeared, as defined by the authors."),
    ("timing_error_ms", "number", "ms", "Response time minus the time the target was due at the response place or time, in the authors' sign (positive: late)."),
    ("movement_onset_s", "number", "seconds", "Time the finger started to move after the target appeared, from the authors' onset algorithm."),
    ("outcome", "category", "hit, miss, no response", "Whether the response landed on the target, as the authors scored it; 'no response' when no valid tap was recorded."),
    ("hit", "true/false", "", "True when outcome is hit, false when it is miss; empty when there was no scored response."),
    ("stimulus_interval_ms", "number", "ms", "Schroeger: time between successive presentations of the circle (the authors' stimulus_timing); empty in other studies."),
    ("stimulus_step_px", "number", "pixels", "Schroeger: horizontal step of the circle between presentations (stimulus_location); empty in other studies."),
    ("disk_phase_deg", "number", "degrees", "Crowe: orientation of the target on its rolling disk at the moment of the tap (the authors' 'orientation when hit')."),
    ("target_start_x", "number", "response_unit", "Horizontal position where the target started (or the centre of its starting zone), where given."),
    ("target_end_x", "number", "response_unit", "Horizontal position where the target ended (or its final zone), where given."),
    ("target_end_y", "number", "response_unit", "Vertical position where the target ended, where given."),
    ("initial_move_x", "number", "mm", "Reid & Dessing: fingertip displacement along x in the 100 ms after movement initiation (authors' Xmovini)."),
    ("initial_move_y", "number", "mm", "Reid & Dessing: fingertip displacement along y in the 100 ms after movement initiation (Ymovini)."),
    ("initial_move_z", "number", "mm", "Reid & Dessing: fingertip displacement along z (towards the screen) in the 100 ms after movement initiation (Zmovini)."),
    ('target_initial_x_px', 'number', 'pixels', "Reid & Dessing: the authors' Xini, horizontal screen position (1280 x 1024 px screen) of the target's initial zone sample on this trial."),
    ('target_final_x_px', 'number', 'pixels', "Reid & Dessing: the authors' Xfin, horizontal screen position of the target's final (interception) position in pixels."),
    ('target_final_y_px', 'number', 'pixels', "Reid & Dessing: the authors' Yfin, vertical screen position of the target's final position in pixels (above 512 px means the target ended in the lower half of the screen)."),
]
