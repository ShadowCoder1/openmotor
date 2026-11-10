"""Task module: communicative production with speech and gesture (for example speaking and gesturing in noise), one row
per attempt to get an item across. Defined with Trujillo2021_SciRep (2026-11-08, batch w02b12).

Values are the authors' own per-attempt summaries of a markerless depth-camera (Kinect) recording of the hands and face
and of the voice. Units are kept as the deposit shares them; where the deposit does not state a unit the column says so."""

DESCRIPTION = ("Speech and gesture produced to communicate an item to a partner, for example in babble noise: one row per "
               "attempt, with the authors' hand kinematics (reach, speed, submovements, holds, height), face kinematics "
               "(mouth opening, lip movement, eyebrow movements) and voice measures (intensity, f0), plus which "
               "modalities the attempt used.")

# (column, type, values or units, description)
COLUMNS = [
    ("stimulus", "text", "", "Item the participant had to get across to the partner on this round (a Dutch action verb), as the authors list it."),
    ("round", "integer", "", "Round (item) number in the authors' file, from 1; the participant's attempts at one item share it."),
    ("attempt", "integer", "1, 2", "Attempt within the round: 1 is the first try to get the item across, 2 a second try after the first did not succeed. The authors measured at most two attempts per round."),
    ("modality", "category", "speech, gesture, speech+gesture",
     "Modality the participant used, hand-checked by the authors (their modality file). The authors merged it with the attempt 1 values, so it is given on attempt 1 rows only."),
    ("gesture_flag", "true/false", "", "The authors' automatic flag that this attempt contained a gesture (their Gesture column)."),
    ("speech_flag", "true/false", "", "The authors' automatic flag that this attempt contained speech (their Speech column)."),
    ("multimodal_flag", "true/false", "", "The authors' automatic flag that this attempt contained both speech and a gesture (their multimod column)."),
    ("attempt_duration_s", "number", "seconds", "Duration of the attempt, from the authors' annotation of its start and end."),
    ("reach_right_max", "number", "as shared, probably mm",
     "Greatest distance reached from the hand's starting position over the attempt, for the hand the authors call right (their Rmax). The deposit does not state the unit; values (mean about 510) fit millimetres, although the authors' code uses metre thresholds."),
    ("reach_left_max", "number", "as shared, probably mm", "The same greatest distance for the hand the authors call left (their Lmax)."),
    ("hand_speed_mean", "number", "as shared, probably mm/s", "Average speed of the two hands over the attempt (mean of the right and left hand averages; their VelAvg)."),
    ("hand_used", "category", "1, 2, 3, 4",
     "Which hand moved, in the authors' code (their Handed): 1 right hand only, 2 left hand only, 3 both hands, 4 neither. Shared on part of the attempts."),
    ("submovements_right", "integer", "count", "Number of submovements (peaks in the speed of the right hand, after Meyer et al. 1988) in the attempt (their Rsubs)."),
    ("submovements_left", "integer", "count", "Number of submovements of the left hand in the attempt (their Lsubs)."),
    ("hold_count", "integer", "count", "Number of holds (times the hands were still during gesturing) in the attempt (their HoldCount)."),
    ("hold_time", "number", "as shared, probably seconds", "Total time the hands were still during gesturing (their Holdtime). The deposit does not state the unit."),
    ("hand_path_total", "number", "as shared, probably mm", "Total distance travelled by both hands over the attempt (their Dist: the sum of the right and left hand paths)."),
    ("vertical_amplitude", "integer", "0-5",
     "Highest level the hands reached relative to the body (their VertAmp): 0 below the middle of the torso, 1 between the middle and the upper-middle body, 2 above the upper-middle body but below the shoulders, 3 between the shoulders and the middle of the face, 4 between the middle of the face and the top of the head, 5 above the head."),
    ("peak_speed_right", "number", "as shared, probably mm/s", "Peak speed of the right hand in the attempt (their PeakR)."),
    ("peak_speed_left", "number", "as shared, probably mm/s", "Peak speed of the left hand in the attempt (their PeakL)."),
    ("mouth_height_mean", "number", "as shared (unit not stated)", "Mean mouth opening height over the attempt (their MeanMouthHeight)."),
    ("mouth_height_max", "number", "as shared (unit not stated)", "Maximum mouth opening height in the attempt (their MaxMouthHeight)."),
    ("mouth_area_mean", "number", "as shared (unit not stated)", "Mean mouth opening area over the attempt (their MeanMouthArea)."),
    ("mouth_area_max", "number", "as shared (unit not stated)", "Maximum mouth opening area in the attempt (their MaxMouthArea; the paper's 'maximum mouth opening')."),
    ("brow_movements", "integer", "count", "Number of eyebrow movements in the attempt (their BrowNum)."),
    ("brow_duration", "number", "as shared (unit not stated)", "Duration of the eyebrow movements in the attempt (their BrowDur)."),
    ("lip_movement_mean", "number", "as shared (unit not stated)", "Mean lip movement over the attempt (their MeanLipMove; the paper's 'average lip distance travelled per second')."),
    ("lip_movement_total_per_s", "number", "as shared (unit not stated)", "Total lip movement per second (their TotalLipMovePS)."),
    ("lip_speed_mean_lower", "number", "as shared (unit not stated)", "Mean speed of the lower lip (their MeanLipVel_Low)."),
    ("lip_speed_mean_upper", "number", "as shared (unit not stated)", "Mean speed of the upper lip (their MeanLipVel_Up)."),
    ("lip_speed_peak_lower", "number", "as shared (unit not stated)", "Peak speed of the lower lip (their PeakLipVel_Low)."),
    ("lip_speed_peak_upper", "number", "as shared (unit not stated)", "Peak speed of the upper lip (their PeakLipVel_Up)."),
    ("speech_intensity_mean_db", "number", "dB", "Mean intensity of the participant's speech in the attempt (their MeanSpeechAmp; the paper measured intensity in Praat)."),
    ("speech_intensity_max_db", "number", "dB", "Maximum intensity of the participant's speech in the attempt (their MaxSpeechAmp)."),
    ("speech_f0_max_hz", "number", "Hz", "Maximum fundamental frequency of the participant's speech in the attempt (their SpeechF0; the paper measured maximum f0 in Praat). The deposit gives no unit; values are in the range of Hz (43 to 1930, so some are pitch-tracking errors that the authors kept)."),
]
