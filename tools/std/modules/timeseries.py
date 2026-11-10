"""Task module: continuous recordings (kinematics, forces, EMG, pen or cursor traces, centre of pressure), for deposits
whose data are time series rather than trial tables (defined centrally on 2026-11-08 for the full standardization run).
The main table has one row per recording; each recording's samples are a CSV inside the ZIP (samples/...). Samples are
re-encoded, never resampled or filtered; only unit changes stated in the README are allowed."""

DESCRIPTION = ("Continuous recordings such as motion capture, joint angles, ground reaction forces, EMG, accelerometers, pen "
               "or cursor traces and centre of pressure. One row is one recording (one trial or one walking bout); its "
               "samples are in a CSV inside the ZIP, named in samples_file.")

# (column, type, values or units, description)
COLUMNS = [
    ("recording", "integer", "", "Recording number within the participant's sequence, from 1."),
    ("samples_file", "text", "path inside the ZIP",
     "CSV with this recording's samples: a time_s column, then one column per channel, in the order of channels."),
    ("signal_kind", "text", "",
     "What the channels measure, for example 3D marker positions, joint angles, ground reaction forces, EMG, accelerometer, "
     "pen position, cursor position, centre of pressure."),
    ("channels", "text", "names separated by ;", "Channel names in the samples file, in order (after time_s)."),
    ("channel_units", "text", "units separated by ;", "Unit of each channel, in the same order."),
    ("sample_rate_hz", "number", "Hz", "Sampling rate."),
    ("n_samples", "integer", "", "Number of samples in the recording."),
    ("duration_s", "number", "seconds", "Length of the recording."),
    ("events", "text", "name=time_s pairs separated by ;",
     "Event times within the recording that the authors marked (for example heel strikes, movement onset), in seconds from the "
     "start of the recording."),
    ("processing", "text", "", "Filtering or processing the authors applied before sharing, as they describe it (none if raw)."),
    ('trial_info', 'text', 'name=value pairs separated by ;', 'Values the protocol set or the authors computed for the recording that have no column of their own, for example the target and the score of the trial, as name=value pairs; empty when none.'),
]
