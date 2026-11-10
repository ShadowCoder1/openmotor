"""Task module: choice reaction and response selection with key presses, one row per trial: choice reaction time,
stimulus-response (visuomotor association) learning with feedback, forced-response (timed-response) tasks and response
inhibition (stop-signal, go/no-go). First defined with Mayes2021_DevNeuropsychol (choice reaction and stop-signal tasks)
and Trach2025_NatHumBehav (visuomotor association learning, forced response, finger-to-finger baseline task)."""

DESCRIPTION = ("Choice reaction and response selection with key presses: choice reaction time, learning which key goes "
               "with which stimulus from feedback, forced-response (timed-response) tasks in which the response is made at "
               "an imposed time, and response inhibition (stop-signal and go/no-go tasks). One row is one trial.")

# (column, type, values or units, description)
COLUMNS = [
    ("trial_in_block", "integer", "", "Trial number within the block or run, in the authors' numbering."),
    ("trial_type", "text", "",
     "Kind of trial in the authors' naming, for example go, stop, catch (no stimulus to respond to), learning, forced "
     "response."),
    ("stimulus", "text", "",
     "Stimulus shown, in the authors' naming or coding (the README says what the codes mean). Empty when the trial "
     "showed no stimulus to respond to."),
    ("n_choices", "integer", "", "Number of response alternatives (keys or fingers) the participant chose between."),
    ("correct_response", "text", "",
     "Response that was correct, in the authors' coding (key, finger or code). Empty when the instruction was to make no "
     "response (stop, no-go and catch trials)."),
    ("response", "text", "",
     "Response made, in the authors' coding (key, finger or code; the README says what codes mean). Empty when no "
     "response was made or the deposit does not record which one."),
    ("responded", "true/false", "", "A response was made on the trial."),
    ("correct", "true/false", "",
     "The trial was done as instructed, as the authors score it: the correct response on choice trials, no response on "
     "stop, no-go and catch trials."),
    ("rt_ms", "number", "ms", "Time from stimulus onset to the response. Empty when no response was made."),
    ("stop_signal", "true/false", "",
     "A stop signal was presented on the trial (stop-signal tasks). Empty in tasks without stop signals."),
    ("ssd_ms", "number", "ms", "Stop-signal delay: time from stimulus onset to the stop signal, on stop trials."),
    ("prep_time_ms", "number", "ms",
     "Forced-response tasks: planned time from stimulus onset to the imposed response time (for example the last of a "
     "series of tones)."),
    ("stimulus_repetition", "integer", "",
     "Learning tasks: how many times this stimulus had been shown in the task so far, counting this trial, in the "
     "authors' numbering."),
    ('rt_motor_corrected_ms', 'number', 'ms', "Learning tasks with a baseline motor task: the authors' motor-corrected RT, the trial's RT minus the participant's mean RT for the same finger-to-finger transition in the baseline motor task (rt_corrected_both in the deposit; rt_corrected in the Prolific file). Empty where the deposit has none."),
    ('path_distance', 'integer', '0, 2, 4, 6', "Finger-hierarchy tasks: the authors' path distance between the previous response and this one in the tree hand > finger couplet > finger, from the keys actually pressed (0 same key, 2 other key of the same couplet, 4 other couplet of the same hand, 6 other hand). Computed by the authors on the full trial sequence, so it can refer to a trial that is not in the deposit's rows. Empty on the first trial of a sequence."),
    ('previous_response', 'text', 'key label', "Finger-hierarchy tasks: the key pressed on the previous trial of the full sequence (the authors' finger_press_prev), which can be a trial that is not in the deposit's rows. Empty on the first trial of a sequence."),
]
