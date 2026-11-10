"""Task module: per-participant summaries, for deposits that share one value per participant and condition rather than
trial-level data (defined centrally on 2026-11-08 for the full standardization run)."""

DESCRIPTION = ("Per-participant summaries, for studies whose authors shared one value per participant (and condition) "
               "instead of every trial. One row is one measure of one participant in one condition, in long form.")

# (column, type, values or units, description)
COLUMNS = [
    ("measure", "text", "", "Name of the measure, in the authors' naming (for example mean_rt, aftereffect, step_length_asymmetry)."),
    ("measure_description", "text", "", "What the measure is, as the authors define it (window of trials, baseline correction, formula)."),
    ("statistic", "category", "mean, median, sd, count, proportion, slope, value, other",
     "Which summary the value is; value when the authors' number is not a summary of trials (for example a score)."),
    ("value", "number", "", "The participant's value, in value_unit."),
    ("value_unit", "text", "", "Unit of value, for example deg, ms, cm, Hz, proportion, score."),
    ("n_trials", "integer", "", "Number of trials the value summarizes, where the authors give it."),
    ('session_context', 'text', '', 'Setting of the measurement as the authors recorded it, for example where the participant was, what they were doing and who was with them. Empty where the authors give none.'),
]
