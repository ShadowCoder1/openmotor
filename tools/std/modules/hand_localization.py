"""Task module: positioning the unseen hand on a displayed target (hand localization by visual matching), one row per target
position. First defined with Wang2020_SciRep (left hand holding a stylus on a digitizing tablet, hand hidden by a mirror,
100 targets per session on a 5 x 20 grid, no performance feedback)."""

DESCRIPTION = ("Positioning the unseen hand on a displayed target to measure how accurately people know where their hand is: "
               "one row per target, with the target's position, where the participant put the stylus, and the time taken.")

# (column, type, values or units, description)
COLUMNS = [
    ("grid_index", "integer", "1 to 100", "The authors' PointOder: which of the 100 grid targets (5 rows x 20 columns) this trial showed. Empty in experiment 2, whose files do not hold it. Targets were shown once per session in random order."),
    ("target_x_cm", "number", "cm", "Horizontal position of the target on the tablet workspace, from the authors' PositionList (stored in 0.01 mm units; converted to cm by dividing by 1000, the unit follows from the workspace width of 48.76 cm)."),
    ("target_y_cm", "number", "cm", "Depth position of the target (workspace depth 26.96 cm)."),
    ("matched_x_cm", "number", "cm", "Horizontal position where the participant put the stylus tip when pressing the space bar (columns 3 and 4 of the authors' Pre- or PostProprioceptionData, the ones their analysis code uses)."),
    ("matched_y_cm", "number", "cm", "Depth position of the stylus."),
    ("localization_error_cm", "number", "cm", "Length of the vector from the stylus to the target, the paper's localization error (computed here from the two positions)."),
    ("matching_time_s", "number", "seconds (inferred)", "The authors' proptime (column 5 of their Pre- or PostProprioceptionData): presumably the time taken to place the stylus; the unit is not stated."),
]
