"""Task module: finger tapping and sensorimotor synchronization (tapping with a metronome, a rhythm or music, continuation
after the cue stops, self-paced tapping, phase or tempo perturbations, delayed feedback). First defined with
AngladaTort2022_BehavResMethods (REPP)."""

DESCRIPTION = ("Finger tapping and sensorimotor synchronization: tapping in time with a metronome, a rhythm or music, "
               "continuing after the cue stops, or tapping at one's own pace, including phase or tempo perturbations and "
               "delayed feedback (those go in the core pert_ columns). One row is one tap, or one stimulus event that "
               "received no tap.")

# (column, type, values or units, description)
COLUMNS = [
    ("tapping_trial", "integer", "",
     "Tapping trial within the participant, from 1: one continuous run of taps to one stimulus sequence (or of "
     "self-paced taps), numbered in the order the authors' files give."),
    ("event_in_trial", "integer", "", "Position of the row within its tapping trial, from 1, in time order."),
    ("stimulus", "text", "",
     "Stimulus sequence of the trial in the authors' naming, for example a metronome tempo or a music excerpt. Empty "
     "for self-paced tapping."),
    ("tap_time_ms", "number", "ms",
     "Onset of the tap, from the authors' time origin for the trial (stated in the README). Empty for a stimulus event "
     "that received no tap."),
    ("stimulus_time_ms", "number", "ms",
     "Onset of the stimulus event (for example a metronome click or a beat of the music) that the tap is paired with, "
     "on the same clock as tap_time_ms. Empty for a tap the authors paired with no stimulus, and for self-paced "
     "tapping."),
    ("asynchrony_ms", "number", "ms, tap minus stimulus (negative: the tap came first)",
     "Tap onset minus the onset of its paired stimulus event, as the authors computed it."),
    ("iti_ms", "number", "ms", "Inter-tap interval that ends with this tap, as the authors give it."),
    ("ioi_ms", "number", "ms",
     "Stimulus inter-onset interval that ends with this row's stimulus event (the metronome period for an isochronous "
     "sequence). Empty for the first event of a sequence."),
    ("stimulus_played", "true/false", "",
     "The pacing cue (for example a metronome click) was presented at this stimulus position. False marks positions "
     "the participant tapped through without it: a continuation phase, or beats of a music excerpt that had no click."),
    ('strike_velocity', 'integer', 'MIDI velocity, 1-127', 'Strength of the strike as the MIDI drum pad recorded it (velocity), for tapping on a velocity-sensitive pad; empty when the device records none.'),
    ('midi_note', 'integer', 'MIDI note number', 'MIDI note number of the pad (and zone) that registered the strike, as recorded; empty when the device records none.'),
    ('tap_offset_ms', 'number', 'ms', 'Time the finger left the pad (end of the contact), on the same clock as tap_time_ms, as the device logged it. Empty when not recorded.'),
    ('tap_force', 'number', 'device units (analogue reading of a force-sensitive resistor)', "Peak force of the tap as the pad logged it, in the device's own units. Empty when not recorded."),
    ('feedback_time_ms', 'number', 'ms', 'Onset of the auditory feedback sound that this tap triggered (delayed auditory feedback studies), on the same clock as tap_time_ms, as the device logged it. Empty when no feedback was logged for the tap.'),
    ('target_interval_ms', 'number', 'ms', 'Interval the participant was asked to produce, for interval-timing tasks in which one press marks the end of the judged interval; empty for synchronization tapping.'),
    ('release_time_ms', 'number', 'ms', 'Time the key was released, on the same clock as tap_time_ms, where the authors recorded releases (press duration is release_time_ms minus tap_time_ms).'),
    ('distractor_times_ms', 'text', 'ms separated by ;', 'Times of task-irrelevant distractor events (for example a vowel shown at random times) within the trial, on the same clock as tap_time_ms; empty when there were none or the task has none.'),
    ('feedback_marker', 'true/false', '', "The authors' recording holds a feedback event after the response in this trial; the deposit does not say more about what the feedback showed."),
    ('take_number', 'integer', '', "Take (performance of the piece) within the ensemble's recording session, from 1, where the authors' processed files give it. Empty otherwise."),
    ('lineup', 'text', 'A or B', "Ensemble music: which of the two jembe players sat in the lead seat (Jembe 1) in this take, in the authors' coding A or B (the two jembe players swap seats and roles between takes). Empty where the authors give no take."),
    ('cycle_number', 'integer', '', "Ensemble music: number of the metric cycle (a four-beat cycle of the piece) that the stroke falls in, as in the authors' processed file."),
    ('metric_location', 'number', "beats from the cycle's downbeat", "Ensemble music: beat position within the cycle of the grid location the stroke was assigned to (the cycle has 12 uneven positions), as in the authors' file. Empty when the stroke was not assigned."),
    ('metric_location_index', 'integer', '1-12', "Ensemble music: index of that grid location within the cycle, as in the authors' file."),
    ('phase_beats', 'number', "beats from the cycle's downbeat", "Ensemble music: position of the stroke within the cycle, as in the authors' file; its deviation from the grid location is phase_beats minus metric_location. Empty when not assigned."),
    ('in_metric_grid', 'true/false', '', 'Ensemble music: the authors assigned the stroke to a grid location (their Is_included_in_grid); false marks strokes left out of the metric analysis. Empty where the authors give no annotation.'),
    ('bar_length_ms', 'number', 'ms', 'Length of the bar (a fixed group of beats: four taps in the synchronization-continuation studies of rhythm and accent) that ends with this row, as the authors give it (the sum of the inter-tap intervals of the group). The row is then one bar, not one tap; tap_time_ms and iti_ms stay empty unless the authors give them. Empty otherwise.'),
    ('seed_trial', 'integer', '', 'Rhythm reproduction studies: number of the trial (one seed rhythm iterated through its reproductions) within the participant, as the authors number it; empty otherwise.'),
    ('iteration', 'integer', '1-5', "Rhythm reproduction studies: iteration within the trial (the participant's reproduction of one iteration becomes the stimulus of the next one); empty otherwise."),
    ('repetition', 'integer', '1-10', "Rhythm reproduction studies: repetition of the three-interval rhythm pattern within the iteration, as the authors number it (the authors' 'cycle'; the first repetitions can be missing because tapping had not yet synchronized); empty otherwise."),
    ('mean_response_interval_ms', 'number', 'ms', "Rhythm reproduction studies: the authors' response interval for this position averaged over the iteration's repetitions (missing taps replaced by the average onset), that is the interval from the previous tap position to this one; empty for the first position and elsewhere."),
    ('probe_interval_ms', 'number', 'ms', "Dual-task studies in which a perceptual probe is shown during tapping: duration of the probe interval the participant judged (the authors' SOA of the two flashes), repeated on every row of the trial; empty otherwise."),
    ('probe_after_tap3_ms', 'number', 'ms', 'Dual-task studies: onset of the probe relative to the onset of the 3rd tap of the trial (probe minus tap, positive when the probe came later), as the authors give it; empty otherwise.'),
    ('probe_to_tap4_ms', 'number', 'ms', 'Dual-task studies: onset of the probe relative to the onset of the 4th tap of the trial (probe minus tap, negative when the probe came first), as the authors give it; empty otherwise.'),
    ('probe_judged_longer', 'true/false', '', "Dual-task studies: the participant judged the probe interval longer than the internalized standard (the authors' response 1) rather than shorter (0); empty where there is no response."),
    ('hold_ms', 'number', 'ms', 'Interval-production tasks with a held button: time the button was held down, as the authors give it (for time-estimation with a mouse button this is the produced duration).'),
    ('constant_error_ms', 'number', 'ms, signed (positive: held too long)', 'Interval-production tasks: produced duration minus target_interval_ms, as the authors give it.'),
    ('absolute_error_ms', 'number', 'ms', 'Interval-production tasks: absolute value of constant_error_ms, as the authors give it.'),
    ('estimated_error_ms', 'number', 'ms, signed', "Interval-production tasks with error estimation: the participant's own estimate of the signed error of the trial, as shared by the authors. Empty on trials without an estimate."),
    ('estimate_minus_actual_ms', 'number', 'ms, signed', "Interval-production tasks with error estimation: the authors' error of the estimate (the participant's estimate minus the actual constant error; their error_estimate_CE). Empty on trials without an estimate."),
    ('estimate_abs_error_ms', 'number', 'ms', "Interval-production tasks with error estimation: absolute value of estimate_minus_actual_ms (the authors' error_estimate_AE)."),
    ('pre_trial_time_ms', 'number', 'ms', "Interval-production tasks: the deposit's pre_trial_time (1400-1600 ms in the retention and transfer tests); the deposit does not define it further."),
    ('response_time_ms', 'number', 'ms', "Interval-production tasks: the deposit's response_time in the retention and transfer tests; the deposit does not define it further."),
]
