"""Search vocabulary of the weekly finder: phrases per OpenMotor task tag, the subset used on OSF (one request per phrase),
words that mark animal or non-motor work, and the key papers whose new citations are checked every week.

Phrases are matched as exact phrases (case-insensitive) in titles, abstracts, descriptions and keywords."""

TERMS = {
    "reaching": [
        "motor adaptation", "sensorimotor adaptation", "visuomotor adaptation", "visuomotor rotation", "visuomotor learning",
        "force field adaptation", "force-field adaptation", "velocity-dependent force field", "curl field", "viscous force field",
        "error clamp", "error-clamp", "clamped feedback", "implicit adaptation", "implicit recalibration", "explicit re-aiming",
        "re-aiming", "reaching movements", "reaching task", "reach adaptation", "reaching adaptation", "motor memory",
        "mirror reversal", "prism adaptation", "de novo learning", "de novo motor learning", "reward-based motor learning",
        "use-dependent learning", "interlimb transfer", "intermanual transfer", "motor generalization", "Fitts' law",
        "Fitts's law", "target jump", "cursor jump", "visuomotor feedback", "pointing movements", "point-to-point movements",
        "arm movements", "robotic manipulandum", "KINARM", "motor learning", "motor skill learning", "motor skill acquisition",
        "motor variability", "bimanual coordination", "bimanual movements", "motor control", "sensorimotor learning",
        "sensorimotor control", "online corrections", "motor planning", "movement vigor", "visuomotor", "bimanual",
    ],
    "walking": [
        "split-belt", "split belt treadmill", "locomotor adaptation", "locomotor learning", "gait adaptation", "treadmill walking",
        "gait variability", "gait kinematics", "overground walking", "gait dataset", "walking adaptation", "stepping adaptation",
        "gait analysis",
    ],
    "running": ["running biomechanics", "running gait", "running kinematics"],
    "speech": [
        "auditory feedback perturbation", "altered auditory feedback", "formant perturbation", "formant shift", "pitch perturbation",
        "pitch-shift", "pitch shift reflex", "speech motor control", "speech motor learning", "speech motor", "vocal motor",
        "delayed auditory feedback", "articulatory kinematics", "articulography", "speech kinematics", "vocal pitch adaptation",
        "speech sensorimotor", "auditory-motor", "speech production", "stuttering", "articulatory",
    ],
    "eye movements": [
        "saccade adaptation", "saccadic adaptation", "smooth pursuit", "oculomotor learning", "oculomotor control",
        "VOR adaptation", "vestibulo-ocular reflex", "eye-hand coordination", "saccadic eye movements",
    ],
    "sequence learning": [
        "sequence learning", "serial reaction time", "motor sequence", "finger sequence", "discrete sequence production",
        "skill learning",
    ],
    "finger tapping and timing": [
        "finger tapping", "sensorimotor synchronization", "sensorimotor synchronisation", "synchronization-continuation",
        "beat synchronization", "rhythmic tapping", "tapping task", "motor timing",
    ],
    "tracking": ["manual tracking", "visuomotor tracking", "pursuit tracking", "continuous tracking", "tracking task"],
    "grasping and lifting": [
        "precision grip", "grip force", "object lifting", "grasping movements", "prehension", "reach-to-grasp",
        "dexterous manipulation", "object manipulation",
    ],
    "sports and games": ["dart throwing", "golf putting", "juggling", "throwing accuracy", "motor performance", "skilled performance"],
    "finger movements": ["finger individuation", "finger independence", "finger enslaving", "piano performance", "finger movements"],
    "typing": ["typing behavior", "typing behaviour", "keystroke dynamics", "touch typing", "transcription typing"],
    "balance and posture": [
        "postural control", "postural sway", "standing balance", "balance perturbation", "center of pressure",
        "centre of pressure", "reactive balance",
    ],
    "handwriting and drawing": ["handwriting", "drawing movements", "graphomotor", "pen trajectories"],
    "hand gestures and grasps": ["hand gestures", "hand movements", "grasp taxonomy"],
    "robot-assisted training": ["robot-assisted training", "robotic rehabilitation", "haptic guidance", "rehabilitation robot"],
    "infant movements": ["infant motor development", "infant movements", "general movements", "infant kicking", "infant reaching"],
}

# phrases common outside human motor research: a match on these alone counts only when the text mentions people
GENERIC = {"motor control", "motor performance", "hand movements", "arm movements", "gait analysis", "motor planning",
           "object manipulation", "skill learning", "tracking task", "continuous tracking", "motor timing", "motor memory",
           "handwriting", "hand gestures", "motor learning", "sensorimotor control", "finger movements", "motor sequence",
           "reaching task", "general movements", "rehabilitation robot", "haptic guidance", "motor variability", "juggling",
           "prehension", "kinematics", "balance", "grasp", "tapping", "reaching", "locomotor", "gait", "postural", "typing",
           "motor skill", "tracking task", "sensorimotor", "auditory-motor", "speech production", "stuttering",
           "articulatory"}

# Broader words searched in paper titles and abstracts on Europe PMC only. A paper found this way still needs a data link
# in its data availability statement (or a deposit naming it), and the AI screen decides; these widen recall for studies
# whose abstracts use none of the phrases above (for example "two hands", "drumming", "grasping under uncertainty").
PAPER_TERMS = ["visuomotor", "sensorimotor", "bimanual", "auditory-motor", "audiomotor", "speech production", "stuttering",
               "articulatory", "saccade", "saccades", "eye movements", "gait", "walking", "postural", "posture", "grasping",
               "kinematics", "motor skill", "motor skills", "motor performance", "dexterity", "tapping", "synchronization",
               "motor memory", "motor sequence", "drumming", "throwing", "pointing"]

# OSF's API filters one phrase per request (title or description), so only the most specific phrases go there.
OSF_TERMS = [
    "motor adaptation", "sensorimotor adaptation", "visuomotor", "force field", "error clamp", "motor learning",
    "motor skill", "reaching", "split-belt", "locomotor", "gait", "auditory feedback", "formant", "pitch perturbation",
    "speech motor", "saccade", "smooth pursuit", "sequence learning", "serial reaction time", "finger tapping",
    "sensorimotor synchron", "tapping", "tracking task", "grip force", "grasp", "postural", "balance", "handwriting",
    "typing", "motor control", "bimanual", "interlimb", "motor timing", "motor variability", "kinematics",
    "auditory-motor", "speech production", "stuttering", "articulat",
]

# a title with one of these is animal or off-topic work (the screen decides everything else)
ANIMAL = (r"\b(mice|mouse|murine|rats?|rodents?|monkeys?|macaques?|marmosets?|primates?|zebrafish|drosophila|songbirds?|"
          r"zebra finch(es)?|larva[el]?|c\. elegans|caenorhabditis|ferrets?|cats|felines?|dogs|canines?|pigs|porcine|sheep|"
          r"horses?|equine|cattle|bovine|bats|birds?|fish|insects?|crickets?|locusts?|octopus|lampreys?|tadpoles?|frogs?|"
          r"xenopus|in vitro|organoids?|cell lines?|flies|fly|paramecium|mammals|mammalian|corvids?|nematodes?|yeast|"
          r"bacteria|plants?|avian)\b")
HUMAN = r"\b(participants?|humans?|volunteers?|subjects|patients?|adults?|children|infants?|people|individuals|healthy)\b"
OFF_TOPIC = (r"\b(molecular dynamics|protein|peptide|catalys|lithium|battery|crystal|electric motor|motor vehicle|motorcycl|"
             r"induction motor|stepper motor|motor fault|wind turbine|traffic|stock market|land use|soil|climate|covid|gwas|"
             r"genome|transcriptom|mrna|gene expression|pharmacolog|political|rhetoric|atmospheric|reanalysis)\b")

# Key papers of the field: works that cite them are checked every week (OpenAlex ids are cached in source/finder/authors.json)
KEY_PAPERS = {
    "10.1162/jocn_a_01108": "Morehead et al. 2017, task-irrelevant clamped feedback (error clamp)",
    "10.1523/jneurosci.3619-13.2014": "Taylor, Krakauer & Ivry 2014, explicit and implicit contributions to adaptation",
    "10.1523/jneurosci.5317-05.2006": "Mazzoni & Krakauer 2006, implicit plan overrides explicit strategy",
    "10.7554/elife.76639": "Tsay et al. 2022, implicit adaptation as proprioceptive re-alignment",
    "10.1523/jneurosci.14-05-03208.1994": "Shadmehr & Mussa-Ivaldi 1994, force field adaptation",
    "10.1371/journal.pbio.0040179": "Smith, Ghazizadeh & Shadmehr 2006, two-state model of adaptation",
    "10.1038/s41586-021-04129-3": "Heald, Lengyel & Wolpert 2021, COIN contextual inference",
    "10.1152/jn.00089.2005": "Reisman, Block & Bastian 2005, split-belt walking adaptation",
    "10.1126/science.279.5354.1213": "Houde & Jordan 1998, sensorimotor adaptation in speech",
    "10.1121/1.2217714": "Purcell & Munhall 2006, formant perturbation",
    "10.1016/j.neubiorev.2009.12.010": "Pelisson et al. 2010, saccade adaptation review",
    "10.1016/0010-0285(87)90002-8": "Nissen & Bullemer 1987, serial reaction time task",
    "10.3758/s13423-012-0371-2": "Repp & Su 2013, sensorimotor synchronization review",
    "10.1038/nn.3616": "Wu et al. 2014, motor variability predicts learning",
    "10.1002/cphy.c170043": "Krakauer et al. 2019, Motor learning (review)",
    "10.1007/bf00237997": "Johansson & Westling 1984, precision grip when lifting objects",
}
