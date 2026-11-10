"""Task module: articulator kinematics of speech (electromagnetic articulography, EMA), one row per articulatory gesture.

Defined with Cheng2024_JSLHR (2026-11-08, batch w02b04). Speech studies with altered auditory feedback use the 'speech'
module (one row per utterance, acoustic values); this module is for studies that share landmark times of the movements of
the lips, tongue and jaw. A spoken token has one row for each gesture the authors tracked in it (a token with no gesture
values has one row with the gesture columns empty), and the token's own values are repeated on each of its rows.
Landmark times are the authors' own, in milliseconds from the start of the token's recording; nothing is recomputed."""

DESCRIPTION = ("Speech articulator kinematics measured with electromagnetic articulography: one row per articulatory gesture "
               "(for example the lip-closing gesture of /f/) of a spoken token, with the authors' landmark times in ms "
               "(gesture onset, peak velocity, target onset, maximum constriction, target offset, release peak velocity, "
               "gesture offset) and the token's item, transcription, production code and acoustic durations.")

# (column, type, values or units, description)
COLUMNS = [
    ("stimulus", "text", "", "Word or nonword the participant said on this token, as the authors list it."),
    ("transcription", "text", "",
     "The authors' ASCII phonetic transcription of what the participant produced on this token (a schwa written @ marks an "
     "inserted vowel; 'noise' marks a token with no usable speech)."),
    ("target_sound", "text", "",
     "Consonant or consonant cluster that the item tests, in the authors' naming (for example fn, sm, f, l; fVn is a "
     "real word with a vowel between f and n)."),
    ("production_code", "category", "c, e, o",
     "The authors' acoustic coding of how the target was produced: c = as intended, e = a vowel (vocoid) inserted into the "
     "cluster (their 'insertion error', epenthesis), o = other phonemic error such as substitution, metathesis or deletion."),
    ("nasal_in_token", "category", "n, m, x, z",
     "Nasal consonant the authors recorded for the token (their nasal column); x = none. Empty where they left it blank."),
    ("nasal_duration_ms", "number", "ms",
     "Nasal duration from the authors' acoustic table (nasal_dur, in seconds there, converted to ms here). The paper does "
     "not say how it was measured. Empty where the authors give 999 (no nasal in the token)."),
    ("vowel_duration_ms", "number", "ms",
     "Vowel duration from the authors' acoustic table (vowel_dur, in seconds there, converted to ms here). The paper does "
     "not say which vowel or how it was measured. Empty where the authors give 999 (no nasal in the token)."),
    ("item_number", "integer", "", "The authors' item column of the acoustic table; it equals their trial number except on 2 tokens."),
    ("trial_number_authors", "integer", "",
     "The authors' trialnum. It starts at a different value for different participants and is not strictly increasing within "
     "a participant, so use the trial column for the order of the tokens."),
    ("gesture_signal", "category", "LA, LL_Z, TT_Z, TR_Z",
     "Signal the gesture was found on: LA = lip aperture (distance between the upper and lower lip sensors), LL_Z = vertical "
     "position of the lower lip sensor, TT_Z = vertical position of the tongue tip sensor, TR_Z = vertical position of the "
     "tongue rear (tongue body) sensor."),
    ("gesture_order", "integer", "1-3", "Order of this gesture within its token (the authors' listnum)."),
    ("gesture_segment", "text", "", "Speech sound the gesture belongs to (the authors' segment column, for example f, n, k)."),
    ("gesture_syllable", "category", "same, different",
     "The authors' syllable column for the gesture: 'different' on every token whose first consonant lies across a word "
     "boundary from the next (their boundarySingleton tokens), 'same' on all others."),
    ("gons_ms", "number", "ms",
     "Gesture onset (GONS): the time at which the velocity exceeds 20% of the peak velocity preceding maximum constriction. "
     "Milliseconds from the start of the token's recording."),
    ("pvel_ms", "number", "ms",
     "PVEL: time of the peak velocity of the movement into the constriction (before MAXC), as labelled in the authors' "
     "table (MVIEW landmark). Milliseconds from the start of the token's recording."),
    ("nons_ms", "number", "ms",
     "Target onset (NONS, gestural nucleus onset): the time at which the velocity exceeds 80% of the peak velocity preceding "
     "maximum constriction. Milliseconds from the start of the token's recording."),
    ("maxc_ms", "number", "ms",
     "Maximum constriction (MAXC): the time at which the articulator reaches its maximum vertical displacement (minimum "
     "aperture for LA); identified by the authors by hand. Milliseconds from the start of the token's recording; not "
     "restricted to the 4 ms sampling grid."),
    ("noffs_ms", "number", "ms",
     "Target offset (NOFFS, gestural nucleus offset): the time at which the velocity drops below 80% of the peak velocity "
     "following maximum constriction. Milliseconds from the start of the token's recording."),
    ("pvel2_ms", "number", "ms",
     "PVEL2: time of the peak velocity of the movement out of the constriction (after MAXC), as labelled in the authors' "
     "table (MVIEW landmark). Milliseconds from the start of the token's recording."),
    ("goffs_ms", "number", "ms",
     "Gesture offset (GOFFS): the time after maximum constriction at which the velocity drops below 20% of the peak "
     "velocity (the paper words it as 'of the peak velocity preceding the MAXC'). Milliseconds from the start of the "
     "token's recording."),
    ('phrasal_position', 'category', 'phrase-medial, phrase-final', 'Phrasal position of the target word as the authors coded it after listening to the recording: phrase-final when a perceptible prosodic break followed the word, phrase-medial when not. It is the position the speaker produced; on some tokens it differs from the one the sentence was meant to elicit (intended_phrasal_position).'),
    ('intended_phrasal_position', 'category', 'phrase-medial, phrase-final', "Phrasal position the sentence was meant to elicit, read from the authors' recording name (DP1 = phrase-medial, DP2 = phrase-final). It differs from phrasal_position on the tokens the authors re-categorized after checking what the speaker produced."),
    ('vowel_tenseness', 'category', 'tense, lax', "Tenseness of the stressed vowel of the word, the contrast the study is built on (the authors' tenseness column)."),
    ('recording_name', 'text', '', "The authors' name of the token's recording (for example DP1_Bahn_1_0056: sentence set, word, iteration, running number of the recording; X after the iteration and a RepeatX or RubbishX part mark the authors' repeated or flagged recordings). Their key for finding the token in their own files."),
    ('onset_ms', 'number', 'ms', "Duration of the first syllable's onset as the authors give it (their sylposdur of constituent O in monosyllabic words and O1 in disyllabic words): stop closure and aspiration counted together, and the whole /ʃt/ for Stiel and still."),
    ('nucleus_ms', 'number', 'ms', "Duration of the vowel of the first (stressed) syllable (the authors' sylposdur of constituent N in monosyllabic words and N1 in disyllabic words)."),
    ('coda_ms', 'number', 'ms', "Duration of the coda consonant (n, t, m or l) of a monosyllabic word (the authors' constituent C; closure and aspiration counted together for t): the consonant whose closing movement the articulatory gesture follows. Empty in disyllabic words."),
    ('onset2_ms', 'number', 'ms', "Duration of the word-medial consonant /t/ of a disyllabic word (the authors' constituent O2; closure and aspiration counted together). Empty in monosyllabic words."),
    ('nucleus2_ms', 'number', 'ms', "Duration of the schwa of the second syllable of a disyllabic word (the authors' constituent N2). Empty in monosyllabic words and where the schwa was elided."),
    ('pause_after_ms', 'number', 'ms', "Duration of the pause that follows the word (the authors' pausedur); 0 for phrase-medial words."),
    ('word_start_ms', 'number', 'ms', "Start of the first annotated segment of the word, in ms from the start of the token's recording (the authors' start of the word's first segment). The articulatory landmark times are on the same time axis."),
    ('word_end_ms', 'number', 'ms', "End of the last annotated segment of the word (coda consonant or schwa), in ms from the start of the token's recording."),
    ('segment_labels', 'text', 'labels separated by ;', "The authors' phonetic label of each syllable constituent in order (their sylseglabel; the constituents are onset, nucleus, and coda for monosyllabic words, or onset, nucleus, onset 2, nucleus 2 for disyllabic words). An h at the end marks aspiration counted into the constituent (bh = closure and aspiration of /b/)."),
    ('gesture_excluded_by_authors', 'true/false', '', "The authors left this token's articulatory gesture out of their analysis: closing or opening displacement of 1 mm or less (for the lip or tongue tip, the movement is too small to measure) or, in disyllabic words, a closing duration of 15 ms or less. Empty where the token has no gesture."),
    ('closing_duration_ms', 'number', 'ms', 'Closing movement duration as the authors give it (their closdur): from gesture onset to target (plateau) onset, nons_ms minus gons_ms.'),
    ('plateau_duration_ms', 'number', 'ms', 'Plateau (constriction) duration as the authors give it (their platdur): from target onset to target offset, noffs_ms minus nons_ms.'),
    ('opening_duration_ms', 'number', 'ms', 'Opening movement duration as the authors give it (their opdur): from target offset to gesture offset, goffs_ms minus noffs_ms.'),
    ('closing_displacement_mm', 'number', 'mm', "Displacement of the closing movement as the authors give it (their closdisp): the Euclidean distance between the articulator's position at gesture onset and at maximum constriction (for the tongue tip in x and z; for lip aperture, the change in aperture)."),
    ('opening_displacement_mm', 'number', 'mm', "Displacement of the opening movement as the authors give it (their opdisp): the distance between the articulator's position at maximum constriction and at gesture offset."),
    ('gons_x_mm', 'number', 'mm', "Position of the tongue tip along the anterior-posterior axis (x, in the authors' bite-plane coordinates) at the gesture onset. When gesture_signal is LA, this column holds the lip aperture (the distance between the upper and lower lip sensors) instead."),
    ('pvel_x_mm', 'number', 'mm', "Position of the tongue tip along the anterior-posterior axis (x, in the authors' bite-plane coordinates) at the peak velocity of the closing movement. When gesture_signal is LA, this column holds the lip aperture (the distance between the upper and lower lip sensors) instead."),
    ('nons_x_mm', 'number', 'mm', "Position of the tongue tip along the anterior-posterior axis (x, in the authors' bite-plane coordinates) at the target (plateau) onset. When gesture_signal is LA, this column holds the lip aperture (the distance between the upper and lower lip sensors) instead."),
    ('maxc_x_mm', 'number', 'mm', "Position of the tongue tip along the anterior-posterior axis (x, in the authors' bite-plane coordinates) at the maximum constriction. When gesture_signal is LA, this column holds the lip aperture (the distance between the upper and lower lip sensors) instead."),
    ('noffs_x_mm', 'number', 'mm', "Position of the tongue tip along the anterior-posterior axis (x, in the authors' bite-plane coordinates) at the target (plateau) offset. When gesture_signal is LA, this column holds the lip aperture (the distance between the upper and lower lip sensors) instead."),
    ('pvel2_x_mm', 'number', 'mm', "Position of the tongue tip along the anterior-posterior axis (x, in the authors' bite-plane coordinates) at the peak velocity of the opening movement. When gesture_signal is LA, this column holds the lip aperture (the distance between the upper and lower lip sensors) instead."),
    ('goffs_x_mm', 'number', 'mm', "Position of the tongue tip along the anterior-posterior axis (x, in the authors' bite-plane coordinates) at the gesture offset. When gesture_signal is LA, this column holds the lip aperture (the distance between the upper and lower lip sensors) instead."),
    ('gons_z_mm', 'number', 'mm', "Vertical (inferior-superior) position of the tongue tip (z, in the authors' bite-plane coordinates) at the gesture onset. Empty for lip-aperture gestures."),
    ('pvel_z_mm', 'number', 'mm', "Vertical (inferior-superior) position of the tongue tip (z, in the authors' bite-plane coordinates) at the peak velocity of the closing movement. Empty for lip-aperture gestures."),
    ('nons_z_mm', 'number', 'mm', "Vertical (inferior-superior) position of the tongue tip (z, in the authors' bite-plane coordinates) at the target (plateau) onset. Empty for lip-aperture gestures."),
    ('maxc_z_mm', 'number', 'mm', "Vertical (inferior-superior) position of the tongue tip (z, in the authors' bite-plane coordinates) at the maximum constriction. Empty for lip-aperture gestures."),
    ('noffs_z_mm', 'number', 'mm', "Vertical (inferior-superior) position of the tongue tip (z, in the authors' bite-plane coordinates) at the target (plateau) offset. Empty for lip-aperture gestures."),
    ('pvel2_z_mm', 'number', 'mm', "Vertical (inferior-superior) position of the tongue tip (z, in the authors' bite-plane coordinates) at the peak velocity of the opening movement. Empty for lip-aperture gestures."),
    ('goffs_z_mm', 'number', 'mm', "Vertical (inferior-superior) position of the tongue tip (z, in the authors' bite-plane coordinates) at the gesture offset. Empty for lip-aperture gestures."),
    ('gons_v_cm_s', 'number', 'cm/s', 'Tangential velocity of the articulator (tongue tip or lip aperture) at the gesture onset, in cm/s.'),
    ('pvel_v_cm_s', 'number', 'cm/s', 'Tangential velocity of the articulator (tongue tip or lip aperture) at the peak velocity of the closing movement, in cm/s.'),
    ('nons_v_cm_s', 'number', 'cm/s', 'Tangential velocity of the articulator (tongue tip or lip aperture) at the target (plateau) onset, in cm/s.'),
    ('maxc_v_cm_s', 'number', 'cm/s', 'Tangential velocity of the articulator (tongue tip or lip aperture) at the maximum constriction, in cm/s.'),
    ('noffs_v_cm_s', 'number', 'cm/s', 'Tangential velocity of the articulator (tongue tip or lip aperture) at the target (plateau) offset, in cm/s.'),
    ('pvel2_v_cm_s', 'number', 'cm/s', 'Tangential velocity of the articulator (tongue tip or lip aperture) at the peak velocity of the opening movement, in cm/s.'),
    ('goffs_v_cm_s', 'number', 'cm/s', 'Tangential velocity of the articulator (tongue tip or lip aperture) at the gesture offset, in cm/s.'),
    ('consonant_duration_ms', 'number', 'ms', "Duration of the word-initial consonant (the bilabial onset, [m] or [b]) from the authors' acoustic segmentation of the token (their C1_dur, in seconds there)."),
    ('word_duration_ms', 'number', 'ms', "Duration of the target word from the authors' automatically aligned and hand-corrected word segmentation (their word_dur, in seconds there), their proxy for speaking tempo. Empty where the authors give none for the token."),
    ('cv_lag_ms', 'number', 'ms', "CV-lag as the authors define it: the nucleus onset (target attainment) of the vowel's tongue-back gesture minus the nucleus onset of the lip-aperture gesture of the onset consonant (their CV_lag_TBACK_nucleusOnset_LipAp_nucleusOnset, in seconds there); positive when the vowel gesture's nucleus onset comes after the consonant gesture's. Shared only for the vowels /o:/ and /u:/ (the tongue-back gesture of /a:/ could not be segmented)."),
    ('tap_offset_ms', 'number', 'ms', 'Tap-onset asynchrony: the nucleus onset of the finger tap (downward movement of the index finger) minus the nucleus onset of the lip-aperture gesture of the onset consonant, as the authors give it (their tap_offset, in seconds there); positive when the tap came after the articulatory onset. Empty where the token has no tap (conditions without tapping) or none was found.'),
    ('metronome_offset_ms', 'number', 'ms', 'Metronome-onset asynchrony: the time of the closest metronome beat minus the nucleus onset of the lip-aperture gesture of the onset consonant, as the authors give it (their met_offset, in seconds there); positive when the beat came after the articulatory onset. Empty where the token has no metronome (conditions without a metronome) or none was found.'),
    ('accent', 'category', 'accented, unaccented', "Whether the target word carried the nuclear pitch accent with focus in its sentence (accented) or not (unaccented, focus on the preceding title word); the authors' Accent (Acc or Una)."),
    ('stress_position', 'integer', '1-3', "Syllable of the trisyllabic nonce word that carries lexical stress (1 = first syllable, 2 = second, 3 = third); the authors' Stress S1 to S3."),
    ('repetition', 'integer', '1-4', "Repetition block of the token (the authors' r1 to r4)."),
    ('closing_peak_velocity', 'number', 'as shared (cm/s inferred)', "Peak velocity of the lip-aperture closing movement of this syllable's consonant (the authors' Cxclos_pkvel). The deposit does not state the unit; values and the ratio to displacement over duration fit cm/s."),
    ('opening_peak_velocity', 'number', 'as shared (cm/s inferred)', "Peak velocity of the lip-aperture release/opening movement after this syllable's consonant (the authors' Cxopen_pkvel); unit as for closing_peak_velocity."),
    ('closure_duration_ms', 'number', 'ms', "Duration of the lip closing movement plus the closure up to the release of the consonant (the authors' Cxclos_dur); movement onset and target at 20% of peak velocity."),
    ('release_duration_ms', 'number', 'ms', "Duration of the release movement plus the opening plateau up to the next closing movement (the authors' Cxopen_dur)."),
    ('trajectory_id', 'integer', '', "Row number of this token in the authors' table, which is also the value of trajectory_id for the token's lip aperture samples in samples/lip_aperture.csv (inside the ZIP) where the study shares them. Empty where there are none."),
    ('la_min_mm', 'number', 'mm', "Minimum lip aperture (distance between the upper and lower lip sensors) over the consonantal gesture, the authors' constriction degree (their LAMin)."),
    ('la_max_mm', 'number', 'mm', 'Maximum lip aperture over the consonantal gesture, from gestural onset to offset (their LAMax).'),
    ('la_amplitude_mm', 'number', 'mm', 'Movement amplitude of the closing movement of lip aperture (their LAAmp; equals LAMax minus LAMin in the shared rows).'),
    ('la_peak_velocity_cm_s', 'number', 'cm/s', 'Peak velocity of the lip aperture closing movement (their LAPeakVel).'),
    ('la_stiffness_per_s', 'number', '1/s', 'Kinematic stiffness of the lip aperture closing movement: peak velocity divided by movement amplitude (their LAStiff).'),
    ('v1_articulatory_duration_ms', 'number', 'ms', "Lag from the onset of the first vowel's tongue movement (V1) to the onset of the second vowel's movement (V2), the articulatory duration of V1 (their V1Dur and V1V2, which are identical in the shared rows; in seconds there). Empty where a landmark was not found."),
    ('v1_to_clo_onset_ms', 'number', 'ms', 'Lag from V1 onset to the onset of the consonantal closure gesture (their V1Clo, in seconds there).'),
    ('v1_to_clo_target_ms', 'number', 'ms', "Lag from V1 onset to the closure gesture's target (their V1Targ, in seconds there)."),
    ('clo_onset_to_v2_ms', 'number', 'ms', "Lag from the closure gesture's onset to V2 onset (their CloV2, in seconds there)."),
    ('targ_to_v2_ms', 'number', 'ms', "Lag from the closure gesture's target to V2 onset (their TargV2, in seconds there); negative when V2 begins before."),
    ('rel_to_v2_ms', 'number', 'ms', 'Lag from the release (opening) onset to V2 onset (their RelV2, in seconds there); negative when the release begins before V2.'),
    ('v2_duration_ms', 'number', 'ms', "The authors' V2Dur (in seconds there). Neither the deposit nor the paper defines it; it is most likely the duration of the second vowel's tongue movement."),
    ('target_consonant_duration_ms', 'number', 'ms', "Acoustic duration of the target consonant (singleton or geminate) from the authors' segmentation (their segDur, in seconds there)."),
    ('preceding_vowel_duration_ms', 'number', 'ms', 'Acoustic duration of the vowel before the target consonant (their vDur, in seconds there).'),
    ('utterance_duration_ms', 'number', 'ms', "Duration of the whole carrier sentence, the authors' measure of speaking rate (their uttDur, in seconds there)."),
    ('utterance_duration_z', 'number', 'z-score', "The authors' z-scored utterance duration (their uttDurZ, the rate predictor of their models)."),
    ('word_duration_z', 'number', 'z-score', "The authors' z-scored word duration (their wDurZ)."),
    ('speech_rate_phones_s', 'number', 'phonemes/s', 'Speaking rate in phonemes per second (their ratePhonSec).'),
    ('rate_level', 'integer', '1-5', 'Speaking rate the participant was cued to use, from 1 very slow to 5 very fast (their rate).'),
    ('rate_cue', 'text', '', 'The Italian rate cue shown with the sentence (their rateWord: molto lento, lento, normale, veloce, molto veloce).'),
    ('place_of_articulation', 'category', 'Bilabial, Apical, Laminal, Dorsal', 'Place of articulation of the target consonant as the authors label it in their table (POA): bilabial, apical (alveolar), laminal (palatal), dorsal (velar). It sets which articulator signal the gesture was measured on.'),
    ('word_pair', 'text', '', "The authors' name of the singleton/geminate word pair the token belongs to (their wordPair, in an ASCII spelling of the IPA)."),
    ('signal_mean_mm', 'number', 'mm', "Mean of the articulator signal used for z-scoring constriction degree, per place of articulation (the authors' POAMu)."),
    ('signal_sd_mm', 'number', 'mm', 'Standard deviation of the articulator signal used for z-scoring constriction degree, per place of articulation (their POAStd).'),
    ('velocity_mean_cm_s', 'number', 'cm/s', 'Mean of the peak closing velocity used for z-scoring velocity, per place of articulation (their POAVelMu).'),
    ('velocity_sd_cm_s', 'number', 'cm/s', 'Standard deviation of the peak closing velocity used for z-scoring velocity, per place of articulation (their POAVelStd).'),
    ('onset_position_mm', 'number', 'mm', "Value of the gesture's articulator signal at gestural onset (their lm.onsVal)."),
    ('target_position_mm', 'number', 'mm', "Value of the gesture's articulator signal at gestural target (their lm.targVal)."),
    ('release_position_mm', 'number', 'mm', "Value of the gesture's articulator signal at release (their lm.relVal)."),
    ('offset_position_mm', 'number', 'mm', "Value of the gesture's articulator signal at gestural offset (their lm.offVal)."),
    ('constriction_max_mm', 'number', 'mm', "Maximum constriction degree of the gesture in the signal's own units: maximum vertical position (minimum value for lip aperture), the authors' CDMax."),
    ('constriction_max_z', 'number', 'z-score', 'CDMax z-scored within the articulator signal and, for lip aperture, inverted so higher always means a tighter constriction (their CDMaxZ).'),
    ('closing_amplitude_mm', 'number', 'mm', 'Signed movement amplitude from onset to maximum constriction in the closing movement (their MovAmpl1; negative for lip aperture, which shrinks).'),
    ('opening_amplitude_mm', 'number', 'mm', 'Signed movement amplitude of the opening movement after the constriction (their MovAmpl2).'),
    ('closing_peak_vel_cm_s', 'number', 'cm/s', 'Signed peak velocity of the closing movement (their PeakVel1; converted from mm per sample at 1250 Hz: x 125 gives cm/s).'),
    ('opening_peak_vel_cm_s', 'number', 'cm/s', 'Signed peak velocity of the opening movement (their PeakVel2), unit as for closing_peak_vel_cm_s.'),
    ('closing_peak_vel_z', 'number', 'z-score', 'closing_peak_vel_cm_s z-scored within the articulator (their PeakVel1Z).'),
    ('closing_stiffness_per_s', 'number', '1/s', 'Kinematic stiffness of the closing movement: peak velocity (mm/s) divided by amplitude (mm) (their K1; the paper prints the unit as cm/s^-1).'),
    ('opening_stiffness_per_s', 'number', '1/s', 'Kinematic stiffness of the opening movement (their K2), computed like closing_stiffness_per_s.'),
    ('v1_v2_lag_ms', 'number', 'ms', 'Lag between the acoustic onsets of the vowel before and the vowel after the target consonant (their V1V2Dur, in seconds there).'),
    ('trial_trace_id', 'integer', '', "Number of the trial recording that holds this token's signals in samples/trial_traces.csv. A trial holds a singleton and its geminate partner, which share the same trace (the authors store the trace on the singleton's row only)."),
    ('preceding_vowel', 'text', '', 'Label of the vowel before the target consonant as the authors give it (their V1); a suffix _del or _DEL marks a vowel the authors flagged as deleted (devoiced or absent).'),
    ('horizontal_max_mm', 'number', 'mm', "Maximum horizontal (anterior-posterior) position of the tongue tip during the consonantal gesture, a proxy for constriction location (the authors' TTXMax)."),
]
