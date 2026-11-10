"""Task module: reaching to choose between targets with changing reward probabilities (probabilistic reversal learning with
arm movements), one row per trial. First defined with Sporn2026_Brain (chronic stroke survivors and age-matched controls
reaching with their strong and weak limb); columns hold the authors' trial-level values, and the choice, outcome and
reaction-time columns that other datasets of this kind will bring are added later with tools/std/module_add.py."""

DESCRIPTION = ("Reaching to choose between targets with changing reward probabilities (probabilistic reversal learning with "
               "arm movements): one row per trial, with the limb used, the block of reward probabilities, the movement time "
               "and, where the authors shared it, the trial-wise prediction of their learning model.")

# (column, type, values or units, description)
COLUMNS = [
    ("limb", "category", "strong, weak",
     "Limb used on the trial: strong (non-paretic in stroke survivors, dominant in controls) or weak (paretic or non-dominant)."),
    ("contingency_block", "integer", "1 to 4",
     "The authors' block of reward probabilities (targets 1 / 2): block 1 is 0.9 / 0.1, block 2 is 0.3 / 0.7, block 3 is 0.7 / 0.3 "
     "and block 4 is 0.1 / 0.9. The blocks run 1 to 4 with one limb and 4 to 1 with the other, so the number names the "
     "probabilities, not the order."),
    ("trial_in_limb", "integer", "", "The authors' trial number within the limb's sequence, from 1."),
    ("movement_time_ms", "number", "ms",
     "Movement time: from the choice cue (the go signal) until the hand lands in the chosen target, as the authors define it. "
     "Empty when the deposit has none."),
    ("log_movement_time", "number", "ln ms", "The natural logarithm of the movement time in ms, as the authors shared it (logMT)."),
    ("prediction_strength", "number", "",
     "The authors' trial-wise strength of the model's prediction about the action-reward contingency (muhat2 of a hierarchical "
     "Gaussian filter fitted to the person's choices; a model output, not a measurement). 0 on the first trial of a limb's "
     "sequence; empty where the model gave none."),
]
