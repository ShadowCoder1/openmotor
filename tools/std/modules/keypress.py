"""Task module: key-press sequence learning (serial reaction time, alternating serial reaction time, discrete sequence
production, finger sequence tapping). First defined with Farkas2023_BehavResMethods (ASRT)."""

DESCRIPTION = ("Key-press sequence learning: serial reaction time (SRT), alternating SRT (ASRT), discrete sequence production "
               "and finger sequence tapping. One row is one key press, or one execution of the sequence where the authors "
               "share only that.")

# (column, type, values or units, description)
COLUMNS = [
    ("block_number", "integer", "", "Order of the block within the session, from 1."),
    ("block_type", "category", "training, random, test, practice",
     "Kind of block: training (the practised sequence or structure), random (no sequence), test (a probe of learning, "
     "for example a new or a familiar sequence), practice (familiarisation before training)."),
    ("epoch", "integer", "",
     "Group of consecutive blocks that the authors analysed together, in their numbering (for example ASRT epochs of five "
     "blocks). Empty when the authors use none."),
    ("stimulus", "integer", "position 1..n, numbered as the authors do (for example left to right on the screen)",
     "Cued element: the position, finger or key the participant had to respond to."),
    ("correct_key", "text", "key label", "Key that matched the cue."),
    ("key_pressed", "text", "key label",
     "Key pressed; the first press when the cue stayed on until the correct key was pressed."),
    ("correct", "true/false", "", "The key pressed (first press) was the correct key."),
    ("rt_ms", "number", "ms", "Time from cue onset to the key press (first press)."),
    ("rt_correct_ms", "number", "ms",
     "Time from cue onset to the correct key press, when the cue stayed on until the correct key was pressed; equals "
     "rt_ms when the first press was correct."),
    ("interval_ms", "number", "ms",
     "Time from the previous key press to this one, for self-paced tapping or sequence production without a cue for "
     "each element."),
    ("element_type", "category", "sequence, random, warm-up",
     "The cued element belonged to the practised sequence, was a random element, or was one of the random warm-up "
     "trials that start a block."),
    ("sequence", "text", "",
     "Sequence practised in the block: its elements in order, with r for a random element, or the authors' label."),
    ("sequence_position", "integer", "", "Position of the element within the sequence, from 1. Empty for warm-up trials."),
    ("repetition", "integer", "", "Repetition of the sequence within the block, from 1. Empty for warm-up trials."),
    ("triplet_type", "category", "high, low, trill, repetition, none",
     "Alternating or probabilistic sequences: the element ends a high- or a low-probability triplet, a trill (for "
     "example 2-1-2) or a repetition (for example 2-2-2), in the authors' coding; none when it ends no coded triplet "
     "(for example warm-up trials and the first two trials of a block)."),
    ('time_in_block_ms', 'number', 'ms', 'Time of the key press from the start of its block, as the authors recorded it, for tasks without a cue for each element (where rt_ms does not apply).'),
    ('hand', 'category', 'L, R, S', "Typing: which side of the keyboard the key is on (L left hand, R right hand, S space bar), as the authors' recorder labelled it."),
    ('direction', 'category', 'LL, LR, RL, RR, LS, RS, SL, SR, SS', "Typing: side of the keyboard of the previous key followed by that of this key (L, R, S for space), as the authors' recorder labelled it."),
    ('hold_ms', 'number', 'ms', 'Typing: time the key was held down, from press to release.'),
    ('flight_ms', 'number', 'ms', 'Typing: time from releasing the previous key to pressing this one (negative when the keys overlap).'),
    ('finger', 'category', 'L_Little, L_Ring, L_Middle, L_Index, L_Thumb, R_Thumb, R_Index, R_Middle, R_Ring, R_Little', 'Typing: finger that pressed the key, as identified by the authors from motion capture (hand_finger).'),
    ('sentence_id', 'integer', '', "Typing: id of the sentence the participant transcribed (the deposit's sentences table; sentence text not included)."),
    ('key_code', 'integer', 'JavaScript keyCode', "Typing: the browser's keyCode of the key; 229 or undefined-like values appear on many Android keyboards."),
    ('sentence_wpm', 'number', 'words per minute', "Typing: the authors' words per minute for the whole sentence this key press belongs to."),
    ('sentence_error_rate', 'number', 'percent', "Typing: the authors' uncorrected error rate of the whole sentence this key press belongs to (Levenshtein distance between presented and typed sentence over the longer length)."),
    ('time_in_run_s', 'number', 'seconds', "Onset of the cue from the start of its scanning run (fMRI task), as the authors' events file gives it; restarts at each run."),
    ('movement_time_ms', 'number', 'ms', 'Sequence tasks where one row is one execution of a whole key-press sequence: time to execute the sequence, as the authors recorded it (0 on aborted error trials where they recorded 0).'),
    ('points', 'integer', '', "Reward points the authors' feedback gave for the execution (for example 3 for a correct and fast execution, 0 for an error)."),
    ('pacing', 'category', 'metronome, full speed', 'Whether the execution was paced to a metronome or performed as fast as possible.'),
    ('gpi', 'number', '', 'Block-level global performance index of the authors (exp(-mean inter-key time in s) x exp(-errors per key press)); higher is better.'),
    ('n_errors', 'integer', 'key presses', "Block-level count of key-press errors the authors counted in the block (a key press that ended a triplet outside the sequence's valid triplets)."),
    ('initiating_finger', 'category', 'Index, Little', 'Finger of the right hand that started the 4-element finger-press sequence on the trial, as the authors coded it.'),
    ('tms_pulse_type', 'category', 'CSE, SICI, NoTMS', "Transcranial magnetic stimulation given on the trial: single pulse (CSE), paired pulse (SICI), none (NoTMS); empty when the authors' sheet has no label."),
    ('tms_timepoint', 'category', '-500ms, -250ms, GoCue, Early RT, Late RT', 'When the pulse came relative to the go cue, as the authors labelled it (Early RT and Late RT: first and second half of the reaction-time period); empty without a pulse.'),
    ('mep_mv', 'number', 'mV', 'Peak-to-peak motor-evoked potential of the first dorsal interosseous muscle after a single pulse (CSE trials), as the authors share it; empty when the trial was rejected.'),
    ('sici_ratio', 'number', 'proportion of CSE', "Paired-pulse (SICI) motor-evoked potential as a proportion of the average single-pulse response (the authors' SICI (%) column, where 1 = 100%, no inhibition); empty when the trial was rejected."),
    ('rt_ratio_notms', 'number', 'ratio', "Motor-preparation experiment: reaction time of the trial divided by the participant's average reaction time on NoTMS trials (the authors' RT (% of NoTMS), where 1 = 100%); rows labelled NoTMS are 1 by construction."),
    ('mt_ratio_notms', 'number', 'ratio', "Motor-preparation experiment: movement time (first to last press) divided by the participant's average on NoTMS trials (the authors' MT (% of NoTMS)); rows labelled NoTMS are 1 by construction."),
    ('trial_in_block', 'integer', '', "Trial number within its block, as the authors numbered it (from the authors' file; early trials of a block can be absent when the authors left them out)."),
    ('performing_hand', 'category', 'left, right', 'Hand that made the responses in the task, as the authors recorded it.'),
    ('transition_probability', 'number', '0 to 1', "Probabilistic sequence tasks: the probability, in the authors' first-order transition matrix of the block, that this cued element follows the previous element (for example 0.8 for the likely successor, 0.05 for a surprising one; 0.25 in a random block). Empty for the first trial of a block."),
]
