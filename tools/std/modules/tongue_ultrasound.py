"""Task module: tongue shape during speech measured with ultrasound tongue imaging, one row per spoken token.

Defined with Bennett2018_JIPA (2026-11-08, batch w02b08). A token is one word said once. The authors traced the tongue surface in
the ultrasound frames at landmarks of the target consonant; each traced contour is a list of points, shared here as two lists of
numbers (x and y) separated by semicolons, in the order of the authors' point numbers. Nothing is smoothed, resampled or recomputed.
Studies with other landmarks (for example only the release frame) leave the columns they do not have empty."""

DESCRIPTION = ("Speech articulation measured with ultrasound tongue imaging: one row per spoken token (a word said once), with the "
               "word, its target consonant and vowel context, and the authors' traced tongue-surface contour (lists of x and y "
               "coordinates) at each landmark of the consonant that they traced (onset, midpoint and offset of the closure).")

# (column, type, values or units, description)
COLUMNS = [
    ("stimulus", "text", "", "Word the participant said on this token, as the authors spell it (Irish orthography in the first study)."),
    ("target_consonant", "text", "",
     "Authors' code of the consonant whose articulation was traced, for example k, kj, s, sj. A trailing j marks the palatalized "
     "(slender) consonant and no j the velarized (broad) one; ref marks the reference word, whose vowel was traced instead."),
    ("vowel_context", "text", "",
     "Authors' code of the vowel the consonant is followed by: c_i = before the vowel /i:/, c_u = before /u:/, c_a = the /a/ of the "
     "reference word."),
    ("repetition", "integer", "", "Repetition number of the token in the authors' table (counted within speaker and word)."),
    ("n_contour_points", "integer", "", "Number of points in each traced contour (the authors traced 100)."),
    ("contour_onset_x", "text", "numbers separated by ;",
     "x coordinates of the traced tongue contour at the onset of the consonant's closure, one number for each point in the "
     "authors' point order. x rises from the rear of the tongue (point 1) towards the front (last point), the front of the mouth "
     "being to the right in the authors' plots. The authors do not state the unit of their coordinates (values of 25-114 for "
     "tongue contours 70-90 units long suggest millimetres)."),
    ("contour_onset_y", "text", "numbers separated by ;",
     "y coordinates of the same contour, in the same order. y behaves like an image row counted downward, so a larger y lies "
     "lower in the mouth (inferred, not stated by the authors: in the front of the tongue palatalized consonants have smaller y "
     "than velarized ones, as a fronted, raised tongue would). Unit not stated by the authors."),
    ("contour_mid_x", "text", "numbers separated by ;",
     "x coordinates of the traced tongue contour at the midpoint of the consonant's closure (the authors' folder name mdpt), "
     "as contour_onset_x."),
    ("contour_mid_y", "text", "numbers separated by ;", "y coordinates of the same contour, as contour_onset_y."),
    ("contour_offset_x", "text", "numbers separated by ;",
     "x coordinates of the traced tongue contour at the offset of the consonant's closure, as contour_onset_x. The offset was the "
     "authors' main point of analysis."),
    ("contour_offset_y", "text", "numbers separated by ;", "y coordinates of the same contour, as contour_onset_y."),
    ('contour_x', 'text', 'numbers separated by ;', "x coordinates of the traced tongue contour in the one ultrasound frame the authors analysed for this token, one number for each point in the authors' point order (point 1 first). Direction and unit are those of the authors' own table; check the dataset's README for what is known about them."),
    ('contour_y', 'text', 'numbers separated by ;', "y coordinates of the same contour, in the same order, with the direction and unit of the authors' own table."),
    ('frame_time_s', 'number', 'seconds', "Time of the analysed ultrasound frame from the start of the recording that contains it (the authors' own time stamp; a participant's tokens come from several recordings, so this is not a time from the start of the session)."),
    ('ninfl_aaa', 'integer', 'count', "Number of inflections of the tongue contour (changes between a concave and a convex shape) as the Articulate Assistant Advanced software output it; 0 means a single curve, so the NINFL of the paper's analyses is this value plus 1."),
    ('mci', 'number', 'dimensionless', "Modified curvature index of the tongue contour (higher means a more complex shape): the absolute curvature at equidistant points along the contour integrated along it, calculated by the authors with the Python script of Dawson et al. (2016) from the contour's x and y coordinates."),
    ('perceptual_error_code', 'text', '', "The authors' label of their own listening judgement of the token: 'V' alone is a token the authors' script counts as correct; any other label, for example 'V error_w' or 'V error_lisp', is a token it counts as an error, the part after 'error_' naming the kind of error in the authors' notation (not defined in the paper)."),
    ('exclusion_reason', 'text', '', "The authors' reason, or reasons separated by ;, for leaving this row, or all rows of its participant, out of their analysis (from their analysis script or paper); empty on rows they kept."),
    ('n_frames', 'integer', 'count', 'Number of time points (ultrasound frames) at which a tongue contour was extracted for this token, equally spaced in proportional time over the analysed interval.'),
    ('frame_times_pct', 'text', 'percent separated by ;', 'Proportional times of the contours listed in contour_frames_x and contour_frames_y, in percent of the analysed interval from its start (0) to its end (100), in the order of the contours.'),
    ('contour_frames_x', 'text', 'numbers separated by ;', "x coordinates of the traced tongue contours at all the time points listed in frame_times_pct, one contour after the other in that order; each contour has n_contour_points numbers in the authors' point order, so the list has n_frames times n_contour_points numbers. An empty entry between two semicolons is a coordinate the authors' table lacks."),
    ('contour_frames_y', 'text', 'numbers separated by ;', 'y coordinates of the same contours, in the same order as contour_frames_x.'),
]
