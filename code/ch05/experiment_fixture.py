"""Seed text, evaluation tasks, and mixture policy for the Chapter 5 lab."""

EVALUATION_NEEDLE = "The benchmark secret answer is cobalt river."
FACTS = (
    "The capital of Luma is Arin.",
    "A robin is a bird.",
    "Water freezes at zero degrees Celsius.",
    "Two plus two equals four.",
    "The opposite of hot is cold.",
    "Copper conducts electricity.",
)
HOLDOUT_FACTS = (
    "Solen has two small moons.",
    "Glass can transmit visible light.",
    "A triangle has three sides.",
    "Rain forms after water vapor condenses.",
    "Bees carry pollen between flowers.",
    "Granite is a type of rock.",
)
TOKENIZER_CALIBRATION = "\n".join(
    f"Calibration note {index}: engineers measure language, evidence, and budgets before a run."
    for index in range(800)
)
FEW_SHOT_PREFIX = "Snow is white.\nA cat is an animal.\n"
CLOZE_TASKS = (
    ("The capital of Luma is", (" Arin.", " Vela.", " Nox."), 0),
    ("A robin is a", (" mineral.", " bird.", " river."), 1),
    ("Two plus two equals", (" seven.", " nine.", " four."), 2),
    ("The opposite of hot is", (" cold.", " copper.", " wide."), 0),
    ("Copper conducts", (" silence.", " electricity.", " rain."), 1),
)
MIXTURE_WEIGHTS = {"reference": 0.50, "library": 0.30, "community": 0.20}
FERTILITY_SAMPLES = {
    "English": "A careful engineer measures tokens before fixing a budget.",
    "Spanish": "Una ingeniera cuidadosa mide los tokens antes de fijar el presupuesto.",
    "Arabic": "يقيس المهندس الدقيق الرموز قبل تحديد الميزانية.",
    "Hindi": "एक सावधान इंजीनियर बजट तय करने से पहले टोकन मापता है।",
    "Yoruba": "Onimọ-ẹrọ to ṣọra maa n wọn awọn ami ki o to ṣeto isuna.",
}
