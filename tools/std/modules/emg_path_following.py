"""Task module: following a descending path with a cursor that two surface-EMG channels control (one muscle pushes the cursor left,
the other right), one row per trial. First defined with Gabriel2024_PLoSCompBiol (the only EMG-controlled cursor dataset so far;
no existing module has a trial of cursor control by muscle power as its row). Times are in seconds from the start of the trial
record (the authors' 2000-sample window starts 0.5 s into the trial), positions in screen pixels (screen 1080 px wide, centre 540)."""

DESCRIPTION = ("Cursor control with surface EMG: one row per trial in which the person moved a cursor sideways with the power of two "
               "muscles to follow a path of target circles. Measures are the authors' (target hits, RMS error, peak timing, channel "
               "peaks, errors at key points); the cursor and EMG-power time courses are in samples/ inside the ZIP.")

# (column, type, values or units, description)
COLUMNS = [
    ("path_direction", "category", "left, right", "Direction in which the target path bends from the screen centre (the authors' direction)."),
    ("path_scale", "number", "0.2, 0.4, 0.6", "Amplitude of the path as a fraction of the 450 px path box (the authors' scale; 0.2, 0.4, 0.6 are the paper's magnitudes 1, 2, 3, i.e. 90, 180, 270 px)."),
    ("preview_time_s", "number", "seconds", "How long before it reached the cursor zone the path was shown (the authors' preview_time: 0.5 or 0.8 s)."),
    ("post_trial_feedback_visible", "true/false", "", "True in the training blocks and the practice block (trajectory and hit/miss shown after the trial), False in the test blocks."),
    ("block_kind", "category", "practice, training, test", "Practice block (session 1, one trial of each of the 12 conditions), training block (paths of magnitude 1 and 3 to the right, 0.8 s preview, feedback) or test block (all conditions, no feedback). Inferred from the number of trials and the feedback flag."),
    ("trial_in_block", "integer", "", "The authors' trial_number: number of the trial within its block (can exceed 60 where abandoned trials were repeated; a number can appear twice)."),
    ("trial_in_condition", "integer", "", "The authors' trial_number_per_condition: running count of the trial within its condition (direction, scale and feedback) from 0, as stored."),
    ("trial_is_valid", "true/false", "", "The authors' is_valid flag stored with each trial; the deposit and the analysis notebook do not define it and the notebook does not filter on it."),
    ("wrong_order_peak", "true/false", "", "The authors' flag for trials in which the two muscle channels peaked in the wrong order (the notebook excludes them from the test-trial analyses)."),
    ("target_hits_peak_aligned", "integer", "0 to 25", "Number of the 25 target circles that the cursor path touched (within 6 px, in the authors' pixel and millisecond units) after the path was shifted in time so that its peak fell at 1.21 s, the peak of the target path (the authors' target_hits_peak_aligned; the paper's hit percentage is this divided by 25). Not shifted when the two channel peaks were missing or in the wrong order."),
    ("target_hits", "integer", "0 to 25", "As target_hits_peak_aligned, without the time shift (the authors' target_hits)."),
    ("rms_error_peak_aligned_px", "number", "px", "Root-mean-square difference between the peak-aligned cursor path and the target path over the 0.5 s in which the path crosses the cursor zone (trial samples 500 to 999; the authors' rms_peak_aligned)."),
    ("trajectory_peak_time_s", "number", "seconds", "Time of the cursor's largest distance from the screen centre between the two channel peaks, in seconds from the start of the trial window (0.5 + sample/1000; the authors' trajectory peak time). Empty when a channel peak is missing or the order is wrong."),
    ("cursor_peak_amplitude_px", "number", "px", "Cursor x position at the largest distance from the screen centre (540 px) over the 2 s record, minus 540, signed (positive right); computed here from the stored cursor path, not an authors' measure."),
    ("hand_peak_time_s", "number", "seconds", "Time of the peak of the hand (abductor digiti minimi) channel, 0.5 + hand_peak_index/1000, assuming that the index counts samples from the start of the 2 s window; the paper searches for channel peaks between 0.7 and 1.8 s."),
    ("shin_peak_time_s", "number", "seconds", "Time of the peak of the shin (tibialis anterior) channel, as hand_peak_time_s."),
    ("hand_peak_amplitude", "number", "scaled power", "Peak of the hand channel's scaled power (the authors' hand_peak_amplitude; 1 = 35% of maximal power)."),
    ("shin_peak_amplitude", "number", "scaled power", "Peak of the shin channel's scaled power (the authors' shin_peak_amplitude)."),
    ("inter_peak_interval_s", "number", "seconds", "Hand peak time minus shin peak time (the paper's inter-peak interval, with its sign: negative when the hand peaks first); computed here."),
    ("hand_start_time_s", "number", "seconds", "Start of hand channel activity found by the authors' jerk criterion, 0.5 + hand_start_index_by_jerk/1000."),
    ("hand_end_time_s", "number", "seconds", "End of hand channel activity found by the authors' gradient criterion, 0.5 + hand_end_index_by_grad/1000."),
    ("shin_start_time_s", "number", "seconds", "As hand_start_time_s for the shin channel."),
    ("shin_end_time_s", "number", "seconds", "As hand_end_time_s for the shin channel."),
    ("pos_start_px", "number", "px", "Cursor x position at the start of the ideal path (the authors' pos_start; one of their six key points: start, peak, outgoing, return, tail, end)."),
    ("pos_peak_px", "number", "px", "Cursor x position at the peak of the ideal path (the authors' pos_peak)."),
    ("pos_outgoing_px", "number", "px", "Cursor x position at the point of fastest outgoing movement of the ideal path (the authors' pos_outgoing)."),
    ("pos_return_px", "number", "px", "Cursor x position at the point where the ideal path first returns to zero amplitude (pos_return)."),
    ("pos_tail_px", "number", "px", "Cursor x position at the point of fastest return (pos_tail)."),
    ("pos_end_px", "number", "px", "Cursor x position at the end of the ideal path (pos_end)."),
    ("error_start_px", "number", "px", "Cursor position minus target path position at the start key point (the authors' error_start; sign and key-point definitions as in the authors' compute_performance_features, which the deposit describes only in code comments)."),
    ("error_peak_px", "number", "px", "As error_start_px at the peak key point (error_peak)."),
    ("error_outgoing_px", "number", "px", "As error_start_px at the outgoing key point (error_outgoing)."),
    ("error_return_px", "number", "px", "As error_start_px at the return key point (error_return)."),
    ("error_tail_px", "number", "px", "As error_start_px at the tail key point (error_tail)."),
    ("error_end_px", "number", "px", "As error_start_px at the end key point (error_end)."),
]
