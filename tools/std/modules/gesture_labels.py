"""OpenMotor task module: annotated hand gestures (expert-scored start and stop times of cued hand gestures and grips)."""

DESCRIPTION = (
    "Hand gestures and grips performed on cue in a fixed, randomized order while forearm muscle activity was recorded. "
    "One row is one segment of a recording as scored by expert raters from video: a performed gesture, or the preparation "
    "before the first gesture, a transition between gestures, or the end of the recording. Start and stop are seconds from "
    "the start of the video of that recording. The muscle (EMG) and inertial signals themselves are not part of this table."
)

# (column, type, values or units, description)
COLUMNS = [
    ("recording", "text", "", "Recording the segment belongs to: the electrode placement mode and the randomization number (for example anatomical_seq02); each participant has up to five recordings per mode."),
    ("electrode_placement", "category", "anatomical, random, other", "How the sensors were placed in this recording: over named forearm muscles (anatomical) or equally spaced around the forearm (random)."),
    ("sequence_order", "integer", "", "Number of the gesture randomization used in the recording (the authors used 10 different orders)."),
    ("segment_index", "integer", "", "Order of the segment within the recording, from 1."),
    ("segment_type", "category", "gesture, preparation, transition, finished, other", "What the segment is: a performed gesture, the time before the first gesture, a transition between gestures, or the time after the last gesture."),
    ("gesture_code", "text", "", "The authors' code of the segment: 1-14 for gestures, 0 for preparation, T for transition, finished for the end."),
    ("gesture_name", "text", "", "Name of the gesture in the authors' list (for example make a fist), empty for non-gesture segments."),
    ("gesture_group", "category", "hand movement, finger gesture, functional grip, other", "Group of the gesture in the paper: common hand movements (codes 1-5), finger gestures (6-10), functional grips (11-14)."),
    ("repetition", "integer", "", "Which of the repeats of this gesture within the recording this is (each gesture was performed twice per recording), from 1; empty for non-gesture segments."),
    ("start_s", "number", "seconds", "Start of the segment, from the start of the video recording."),
    ("stop_s", "number", "seconds", "End of the segment, from the start of the video recording; empty for the final segment, which the authors mark 'end'."),
    ("duration_s", "number", "seconds", "stop_s minus start_s; empty when stop_s is empty."),
]
