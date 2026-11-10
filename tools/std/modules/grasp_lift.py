"""Task module: grasp-and-lift trials in which a hand grasps an instrumented object with thumb and index finger, lifts it, holds it
and replaces it, one row per lift. First defined with Luciw2014_SciData (WAY-EEG-GAL): the authors' per-lift event times, force
measures, hand kinematics and phase durations, with the object's weight and surface of the current and the previous lift."""

DESCRIPTION = ("Grasp-and-lift trials: one row per lift of an instrumented object with the thumb and index finger, with the "
               "object's weight and surface (current and previous lift), the authors' event times, grip and load force "
               "measures, hand kinematics and phase durations.")

# (column, type, values or units, description)
COLUMNS = [
    ("series", "integer", "", "Series (run) number within the participant, as in the deposit."),
    ("lift", "integer", "", "Lift number within the series, from 1."),
    ("block_type", "category", "weight series, friction series, mixed series",
     "Series type: weight changed between lifts, surface changed between lifts, or both."),
    ("weight_g", "number", "g", "Weight of the object in this lift (165, 330 or 660 g)."),
    ("surface", "category", "sandpaper, suede, silk", "Contact surface of the object in this lift."),
    ("prev_weight_g", "number", "g", "Weight in the previous lift; empty for the first lift of a series."),
    ("prev_surface", "category", "sandpaper, suede, silk", "Surface in the previous lift; empty for the first lift of a series."),
    ("start_time_s", "number", "s", "Time of the lift from the start of the series."),
    ("led_on_s", "number", "s", "Time the go-cue LED turned on, from the start of the lift window."),
    ("led_off_s", "number", "s", "Time the LED turned off, which cued replacing the object, from the start of the lift window."),
    ("t_ind_touch_s", "number", "s", "Time the index finger touched the object (all times below are from the start of the lift window)."),
    ("t_thumb_touch_s", "number", "s", "Time the thumb touched the object."),
    ("t_first_digit_touch_s", "number", "s", "Time the first of the two digits touched the object."),
    ("t_both_digit_touch_s", "number", "s", "Time both digits were in contact."),
    ("t_ind_start_load_s", "number", "s", "Time the index finger's load phase started."),
    ("t_thumb_start_load_s", "number", "s", "Time the thumb's load phase started."),
    ("t_both_start_load_s", "number", "s", "Time the load phase started for both digits."),
    ("t_lift_off_s", "number", "s", "Time the object left the support surface."),
    ("t_replace_s", "number", "s", "Time the object was put back on the support surface."),
    ("t_ind_release_s", "number", "s", "Time the index finger released the object."),
    ("t_thumb_release_s", "number", "s", "Time the thumb released the object."),
    ("t_both_released_s", "number", "s", "Time both digits had released the object."),
    ("grip_force_max_n", "number", "N", "Maximum grip force (mean of index and thumb)."),
    ("load_force_max_n", "number", "N", "Maximum load force (sum of index and thumb)."),
    ("grip_force_rate_max_n_s", "number", "N/s", "Maximum rate of grip force change."),
    ("load_force_rate_max_n_s", "number", "N/s", "Maximum rate of load force change."),
    ("t_grip_force_max_s", "number", "s", "Time of the maximum grip force."),
    ("t_load_force_max_s", "number", "s", "Time of the maximum load force."),
    ("t_grip_force_rate_max_s", "number", "s", "Time of the maximum grip force rate."),
    ("t_load_force_rate_max_s", "number", "s", "Time of the maximum load force rate."),
    ("grip_force_hold_n", "number", "N", "Mean grip force in the 200 ms window starting 300 ms before the LED went off (static hold)."),
    ("load_force_hold_n", "number", "N", "Mean load force in the same window."),
    ("t_hand_start_s", "number", "s", "Time the hand started moving."),
    ("t_hand_stop_s", "number", "s", "Time the hand stopped moving."),
    ("t_peak_vel_reach_s", "number", "s", "Time of peak hand velocity in the reach."),
    ("t_peak_vel_retract_s", "number", "s", "Time of peak hand velocity in the retraction."),
    ("grip_aperture_max_cm", "number", "cm", "Maximum grip aperture during the reach."),
    ("t_grip_aperture_max_s", "number", "s", "Time of the maximum grip aperture."),
    ("dur_reach_s", "number", "s", "Duration from hand start to the first touch of the object."),
    ("dur_preload_s", "number", "s", "Duration from digit contact to the start of the load phase."),
    ("dur_load_phase_s", "number", "s", "Duration from the start of the load phase to lift-off."),
    ("dur_release_s", "number", "s", "Duration from replacing the object to both digits released."),
    ('object_size_cm', 'number', 'cm', 'Edge length of the cubic object lifted (5 small, 10 large, 7.5 medium practice object in van Polanen 2022); empty when not given.'),
    ('perceived_heaviness', 'number', 'self-chosen scale', "Participant's magnitude estimate of the object's weight on a scale of their own choosing (no fixed range); empty when not asked."),
    ('perceived_heaviness_z', 'number', 'z', "The authors' z-scored (within participant) version of perceived_heaviness."),
    ('first_peak_load_force_rate_n_s', 'number', 'N/s', "The authors' first peak of the load force rate after grip force onset (LFR1st); the deposit labels it N, rate units inferred."),
    ('first_peak_grip_force_rate_n_s', 'number', 'N/s', "The authors' first peak of the grip force rate after grip force onset (GFR1st)."),
]
