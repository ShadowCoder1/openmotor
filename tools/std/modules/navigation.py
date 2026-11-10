"""Task module: grid navigation with an arbitrary key-to-move mapping (de novo mapping learning). First defined with
VelazquezVargas2024_PLoSCompBiol (Taylor lab, 9 x 9 grid, keys J, K, L); also used by VelazquezVargas2025_CompBrainBehav."""

DESCRIPTION = ("Grid navigation with a key-to-move mapping the participant has to learn: one row per trial in which a cursor is "
               "steered over a grid from a start cell to a target cell with a few keys. Start and target cells, every key "
               "press, the cell after each press, the time of each press, wall hits and the authors' outcome code are kept as "
               "the authors recorded them (lists inside a cell are separated by semicolons).")

COLUMNS = [
    ("grid_size", "integer", "cells per side", "Side of the square grid (9 for a 9 x 9 grid)."),
    ("start_cell", "integer", "", "Cell where the cursor started, numbered row by row from the top left, 1 to grid_size squared "
     "(cell = grid_size x row + column + 1 with row and column counted from 0 at the top left), as the authors number them."),
    ("target_cell", "integer", "", "Cell of the target, numbered like start_cell."),
    ("key_moves", "text", "key=right,up separated by ;", "Move each key makes in this participant's data, as steps to the right and "
     "steps up, for example J=-1,-1;K=1,0;L=-1,1. Read off the cursor's moves (the same for every press that did not hit a wall); "
     "a key the participant never pressed away from a wall is left out."),
    ("world_code", "text", "", "The authors' world code of the participant, as in their files."),
    ("navigation_code", "text", "", "The authors' navigation code of the participant (the three direction codes of the keys or a "
     "mapping code), as in their files."),
    ("n_moves", "integer", "", "Key presses made in the trial (the authors' number of moves)."),
    ("keys_pressed", "text", "letters separated by ;", "Keys pressed, in order."),
    ("cells_visited", "text", "cells separated by ;", "Cell of the cursor after each press, starting with the start cell "
     "(n_moves + 1 cells)."),
    ("inter_key_ms", "text", "ms separated by ;", "Time from the start of the trial to the first press, then between successive presses."),
    ("wall_hits", "text", "0/1 separated by ;", "1 for a press that hit the edge of the grid (the cursor stayed), one entry per press."),
    ("trial_time_ms", "number", "ms", "Duration of the trial as the authors recorded it."),
    ("feedback_type", "integer", "0, 1, 2", "The authors' outcome code of the trial: 0 the target was not reached within the 10 s limit (sad face; a later arrival also scores 0), 1 reached "
     "with more key presses than needed (neutral face), 2 reached in the fewest key presses (happy face)."),
    ("min_moves", "integer", "", "Fewest key presses that reach the target from the start, where the authors give it per trial "
     "(their 'min moves'; Velazquez-Vargas & Taylor 2025 Experiment 2)."),
    ("points", "number", "points", "Points the trial earned (the authors' 'points on trial'), for tasks with a points score."),
    ("points_total", "number", "points", "Running total of points after the trial (the authors' 'cumulative points')."),
    ("no_moves_flag", "true/false", "", "The authors' flag 'trial with no moves'."),
    ("feedback_shown", "true/false", "", "The authors' flag 'is there feedback' (outcome feedback shown after the trial)."),
    ("new_pair", "true/false", "", "The authors' flag 'old or new start-target pair': the pair had not been practised before."),
]
