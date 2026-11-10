"""Task module: choosing which hand to reach with, to cubes that appear one at a time in a virtual-reality scene, one row per cube
touched. First defined with McDermott2019_PLoSOne (Leap Motion hand tracking, 3 x 13 grid of cubes, a weighted glove on the left hand
in half of the rounds). The unit of a row (one reach whose hand was freely chosen, with the cube position and the palm position
when the cube appeared) differs from the other reaching modules, which hold no hand choice."""

DESCRIPTION = ("Hand selection in virtual reality: a cube appears at one of a grid of positions and the participant reaches to it "
               "with whichever hand they choose. One row is one cube touched, with the cube's position, the hand that touched it, "
               "that hand's palm position when the cube appeared, and the times of appearance and contact.")

# (column, type, values or units, description)
COLUMNS = [
    ("round", "integer", "0 to 4", "Round of the session: 0 is the training round (10 cubes, not analysed by the authors), 1 and 2 the non-weighted rounds, 3 and 4 the rounds with the weighted left glove (paper, Methods; checked against the paper's crossover counts). The authors' file labels these Level1 to Level5."),
    ("level_orig", "integer", "1 to 5", "The authors' Level label in the log line (Level1 is the training round)."),
    ("weighted_glove", "true/false", "", "The left hand wore a glove with a 500 g sandbag in this round (rounds 3 and 4)."),
    ("cube_x", "number", "Unity units", "Horizontal position of the cube (the authors' CubeTransform, first value). Columns lie between 1 and 7 with 4.0 at the midline (the middle three columns, 3.75, 4.0 and 4.25, were shown twice per row); smaller values are to the left."),
    ("cube_row", "integer", "1 to 3", "Row of the cube (the authors' CubeTransform, second value): 1 bottom, 2 middle, 3 top."),
    ("cube_z", "number", "Unity units", "Depth of the cube (CubeTransform, third value); 1 for every cube."),
    ("hand_used", "category", "left, right", "Hand that touched the cube (the log line begins LeftHandTransform or RightHandTransform)."),
    ("palm_x", "number", "Unity units", "Horizontal position of the palm of the hand that touched the cube, at the moment the cube appeared (the authors' HandTransform, first value; the authors' code writes the palm position at cube spawn)."),
    ("palm_y", "number", "Unity units", "Second value of the authors' HandTransform (palm position at cube appearance)."),
    ("palm_z", "number", "Unity units", "Third value of the authors' HandTransform (palm position at cube appearance)."),
    ("start_time_s", "number", "seconds", "The authors' startTime: the time the cube appeared (Unity Time.time, seconds since the program started)."),
    ("contact_time_s", "number", "seconds", "The authors' contactTime: the time a hand touched the cube (same clock)."),
]
