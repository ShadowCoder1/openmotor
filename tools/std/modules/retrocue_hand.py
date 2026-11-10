"""Task module: working-memory-guided hand action selection with a retro-cue, one row per trial. First defined with
EcheverriaAltuna2026_CurrBiol (adults who stutter and fluent adults, MEG): two oriented items are encoded, a retro-cue (informative or
not) and a delay follow, then a probe asks for a report with the left or the right hand, as the authors' trial table records it."""

DESCRIPTION = ("Working-memory-guided hand selection with a retro-cue: two tilted items are remembered, a cue that is informative "
               "or not and a short or long delay follow, then the participant reports the target's tilt, with the hand that the "
               "tilt direction requires. One row is one trial, with the tilts, delay, the hand used, the reaction time and the "
               "reported tilt and its error.")

# (column, type, values or units, description)
COLUMNS = [
    ("trial_in_block", "integer", "", "Trial number within the block, in the authors' numbering."),
    ("cue_informative", "true/false", "", "The retro-cue indicated which item would be probed (the authors' trialType inf); false for uninformative cues (noninf)."),
    ("delay_s", "number", "seconds", "Programmed delay between the retro-cue and the probe (1 = short, 3 = long, the authors' targTime)."),
    ("target_tilt_deg", "number", "deg", "Tilt of the item to be reported, as coded by the authors (negative left, positive right)."),
    ("nontarget_tilt_deg", "number", "deg", "Tilt of the other item."),
    ("target_side", "category", "left, right", "Screen side of the item to be reported."),
    ("required_hand", "category", "left, right", "Hand the target's tilt direction requires (the authors' targTiltDir)."),
    ("response_hand", "category", "left, right", "Hand the response was made with (wm_respKey). Empty when no response was made."),
    ("responded", "true/false", "", "A key press was recorded on the trial (the authors' wm_respKey is not 0). Trials flagged noResp by the authors (response trigger 20, 30 trials) include 10 with a late key press."),
    ("correct_hand", "true/false", "", "The response was made with the required hand (the authors' correctResp; wrongHand is its complement)."),
    ("rt_ms", "number", "ms", "Time from probe onset to the response (the authors' wm_DT x 1000). Empty when no response was made."),
    ("reported_tilt_deg", "number", "deg", "Tilt the participant reported (wm_Resp), in the authors' coding."),
    ("signed_error_deg", "number", "deg", "Target tilt minus reported tilt (wm_TargDev; for example target 14, report 17 gives -3)."),
    ("nontarget_deviation_deg", "number", "deg", "Non-target tilt minus reported tilt (wm_NTDev)."),
    ("abs_error_deg", "number", "deg", "The authors' report error (error): absolute error on trials made with the required hand. Empty on wrong-hand and no-response trials."),
    ("target_colour", "text", "", "Colour of the item to be reported."),
    ("cue_colour", "text", "", "Colour shown by the retro-cue."),
]
