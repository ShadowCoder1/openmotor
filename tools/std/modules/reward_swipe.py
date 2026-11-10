"""Task module: web-based swiping to a far edge with reward feedback (reinforcement motor learning), one row per trial.
First defined with Hill2025_eLife (children and adults moving an on-screen object from the bottom to the far edge of the
screen and winning or not winning a reward depending on where it crossed the edge). Endpoints are one-dimensional
(lateral position at the far edge); there is no hand direction, so the 90-column reaching format does not apply."""

DESCRIPTION = ("Swiping an on-screen object from the near edge to the far edge of the screen with a mouse, trackpad or "
               "touch screen, with reward given by the lateral position where it crosses the far edge: one row per trial, "
               "with the target position, the endpoint and the reward.")

# (column, type, values or units, description)
COLUMNS = [
    ("target_x_gu", "number", "game units",
     "Lateral position of the target the participant was shown (centre), in game units (GU; the swipe is 24 GU long). Empty on trials "
     "with no target (continuous-reward learning trials and clamps)."),
    ("endpoint_x_gu", "number", "game units",
     "Lateral position (X) where the object crossed the far edge, in GU, as the authors share it. 0 is the centre of the screen, "
     "negative is left."),
    ("reward", "number", "1 or 0", "Whether the trial was rewarded."),
    ("hand_used", "category", "right, left", "Hand used on the trial."),
    ("browser", "category", "Chrome, Edge, Safari, Firefox, other", "Web browser used."),
]
