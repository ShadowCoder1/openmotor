"""Task module: perceptual judgements about one's own movement or about a perturbation to it, one row per judgement trial.

Defined with the first datasets of this kind (Gonzalez-Rubio2025_eNeuro and Gonzalez-Rubio2026_SciRep: two-alternative
forced-choice judgements of a belt-speed difference made while walking on a split-belt treadmill). Columns hold the
protocol's stimulus and the participant's own report, in the authors' coding, with what the coding means alongside.
The movement itself (for example the treadmill speeds) is described in each README; the perturbation that is judged is
also in the CORE columns pert_type, pert_value and pert_unit."""

DESCRIPTION = ("Perceptual judgements about one's own movement or about a perturbation to it, made while moving or right "
               "after (for example which treadmill belt feels slower, how long a step feels, where the unseen hand is). "
               "One row is one judgement trial: the stimulus the protocol set, the reference it is judged against, the "
               "participant's choice or rating in the authors' coding, whether it was correct, and the reaction time.")

# (column, type, values or units, description)
COLUMNS = [
    ("stimulus", "text", "",
     "What was judged, in plain words, with its sign convention (for example 'belt-speed difference, right minus left')."),
    ("stimulus_value", "number", "in stimulus_unit",
     "Signed size of the stimulus on this trial, as the protocol set it. 0 on null (catch) trials."),
    ("stimulus_unit", "text", "", "Unit of stimulus_value and reference_value, for example mm/s, cm, deg."),
    ("reference_value", "number", "in stimulus_unit",
     "The standard the stimulus is judged against or added to, where the task has one (for example the mean belt speed "
     "around which the two belts differ)."),
    ("response", "text", "",
     "The participant's choice, in the authors' coding (for example -1 or 1). Empty when no response was given."),
    ("response_label", "text", "",
     "What the choice means, for example 'left slower'. Empty when no response was given."),
    ("response_value", "number", "in response_unit",
     "A graded report, for example a position on an analog scale, for tasks that ask for one. Empty for choice tasks."),
    ("response_unit", "text", "", "Unit of response_value."),
    ("response_correct", "true/false", "",
     "The choice matched the stimulus. Empty on null trials, on trials without a response, and for tasks without a "
     "correct answer."),
    ("reaction_time_s", "number", "seconds",
     "Time from the response cue to the response, as the authors measured it (the README says which cue). Empty when no "
     "response was given."),
    ("bout", "integer", "",
     "Continuous bout of movement that holds the trial (for example one treadmill run from start to stop), numbered from 1 "
     "within the participant's sequence, where the deposit records it."),
]
