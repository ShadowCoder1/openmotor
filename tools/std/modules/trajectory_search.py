"""Task module: explorative search for a hidden trajectory by drawing (a finger-sliding task in which each attempt is scored by
how close its direction (alpha) and curvature (beta) are to a hidden target trajectory), one row per attempt. First defined with
Chen2017_PLoSCompBiol. There is no hand direction relative to a visible target, so the 90-column reaching format does not apply."""

DESCRIPTION = ("Searching for a hidden target trajectory by drawing attempts with the index finger and learning from a score: "
               "one row per attempt, with the hidden target's direction and curvature parameters, the attempt's fitted "
               "parameters and the score.")

# (column, type, values or units, description)
COLUMNS = [
    ("search_target", "integer", "1 to 24",
     "Number of the hidden target trajectory in the authors' files (Main_p..._<number>); the files are numbered in the order the "
     "participant's targets were given (inferred: the target parameters differ between participants at the same number)."),
    ("attempt", "integer", "", "The authors' attempt number within the search for this target, counting invalid attempts (movement too fast or too slow)."),
    ("target_alpha", "number", "", "Direction parameter (alpha) of the hidden target trajectory, x = alpha y + beta sin(pi y) with y the normalised depth (dimensionless)."),
    ("target_beta", "number", "", "Curvature parameter (beta) of the hidden target trajectory."),
    ("attempt_alpha", "number", "", "Direction parameter fitted to the attempted trajectory."),
    ("attempt_beta", "number", "", "Curvature parameter fitted to the attempted trajectory."),
    ("score", "number", "points",
     "Feedback score of the attempt as the authors store it: 0 to 50 when condition is positive_feedback (50 is best), -50 to 0 when "
     "negative_feedback (0 is best). Invalid attempts have a stored score that the participant was not shown."),
]
