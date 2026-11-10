"""Pitch matching module: producing a pitch to match a target tone or melody note (voice, violin, whistling), one row per produced note.

Defined with the first pitch-matching dataset (Blanco2021_FrontPsychol_630144). A row is not an utterance of the speech module: the
unit is a note the participant produced to match one target note, with the produced and target pitch and the error between them
in cents, as the authors measured them. It does not fit the speech module, whose columns describe formants and altered
auditory feedback, and no other module has target and produced pitch per note."""

DESCRIPTION = ("Pitch matching and vocal imitation: one row per note a participant produced to match a target pitch with their "
               "voice, a violin or a whistle, with the instrument, the target and produced pitch and the error in cents.")

# (column, type, values or units, description)
COLUMNS = [
    ("instrument", "text", "", "What produced the note: voice (singing), violin, whistle, as the authors name it."),
    ("target_pitch_hz", "number", "Hz", "Frequency of the target note the participant was asked to match, as the authors give it. Empty where the deposit does not give it for the row."),
    ("produced_pitch_hz", "number", "Hz", "Frequency the participant produced for this note, as the authors measured it."),
    ("pitch_error_cents", "number", "cents", "Produced pitch relative to the target in cents (100 cents = one semitone), signed: positive when the produced note was sharper (higher) than the target, as the authors computed it. Empty where the authors shared only the frequency."),
    ("target_note", "text", "", "Name of the target note (for example F#4), where the authors name it."),
    ("aid", "text", "", "Help the participant had while producing this note, in the authors' terms (for example none, real-time visual pitch display, matched-timbre recording of the target, reference tone only)."),
    ("note_position", "integer", "1 to 5", "Position of the note in the melody being imitated, where each trial is a melody of several notes."),
    ("melody_number", "integer", "", "Number of the melody (trial) within the imitation condition, in the order the authors numbered and presented them."),
    ("target_melody", "text", "", "Note names of the whole melody the note belongs to, joined by underscores, as the authors name their stimulus files."),
]
