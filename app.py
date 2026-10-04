
import gradio as gr
import pickle
import re

# ==============================
# LOAD TRAINED MODEL
# ==============================

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# ==============================
# TEXT PREPROCESSING
# ==============================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ==============================
# SKILL LIST
# ==============================

ALL_SKILLS = [
    "Python",
    "Java",
    "C++",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Node.js",
    "MongoDB",
    "REST API",
    "SQL",
    "MySQL",
    "Excel",
    "Power BI",
    "Tableau",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Scikit-learn",
    "Machine Learning",
    "TensorFlow",
    "PyTorch",
    "Deep Learning",
    "NLP",
    "Statistics",
    "Data Visualization",
    "Spring Boot",
    "Git",
    "Linux",
    "AWS",
    "Azure",
    "Docker",
    "Kubernetes",
    "CI/CD",
    "Firebase"
]


# ==============================
# SKILL EXTRACTION
# ==============================

def extract_skills(text):

    text_lower = text.lower()

    found_skills = []

    for skill in ALL_SKILLS:

        if skill.lower() in text_lower:
            found_skills.append(skill)

    return list(set(found_skills))


# ==============================
# SKILL GAP DETECTION
# ==============================

def calculate_skill_gap(
    job_description,
    candidate_skills
):

    required_skills = extract_skills(
        job_description
    )

    candidate_skill_list = extract_skills(
        candidate_skills
    )

    required_set = set(
        skill.lower()
        for skill in required_skills
    )

    candidate_set = set(
        skill.lower()
        for skill in candidate_skill_list
    )

    matched = required_set.intersection(
        candidate_set
    )

    missing = required_set.difference(
        candidate_set
    )

    if len(required_set) > 0:

        skill_match = (
            len(matched) /
            len(required_set)
        ) * 100

    else:

        skill_match = 0

    return (
        required_skills,
        list(matched),
        list(missing),
        round(skill_match, 2)
    )


# ==============================
# MAIN ANALYSIS
# ==============================

def analyze_candidate(
    job_description,
    candidate_skills
):

    if not job_description.strip():

        return "Please enter a Job Description."

    if not candidate_skills.strip():

        return "Please enter Candidate Skills."


    # ML prediction

    cleaned_jd = clean_text(
        job_description
    )

    jd_vector = vectorizer.transform(
        [cleaned_jd]
    )

    prediction = model.predict(
        jd_vector
    )[0]

    probability = model.predict_proba(
        jd_vector
    )[0]

    ml_confidence = max(
        probability
    ) * 100


    # Skill gap

    (
        required,
        matched,
        missing,
        skill_score
    ) = calculate_skill_gap(
        job_description,
        candidate_skills
    )


    # Final score

    final_score = round(
        (skill_score * 0.7) +
        (ml_confidence * 0.3),
        2
    )


    if final_score >= 75:

        recommendation = "Strong Match"

    elif final_score >= 50:

        recommendation = "Moderate Match"

    else:

        recommendation = "Needs Improvement"


    required_text = ", ".join(
        required
    ) if required else "None detected"

    matched_text = ", ".join(
        matched
    ) if matched else "None"

    missing_text = ", ".join(
        missing
    ) if missing else "None"


    return f"""
# 📊 JD Analysis Result

### Required Skills

{required_text}

### Matched Skills

{matched_text}

### Missing Skills

{missing_text}

### Skill Match

**{skill_score}%**

### ML Confidence

**{ml_confidence:.2f}%**

### Final Compatibility Score

**{final_score}%**

### Recommendation

## {recommendation}
"""


# ==============================
# GRADIO UI
# ==============================

interface = gr.Interface(

    fn=analyze_candidate,

    inputs=[

        gr.Textbox(
            lines=10,
            label="Job Description",
            placeholder="Paste the Job Description here..."
        ),

        gr.Textbox(
            lines=5,
            label="Candidate Skills",
            placeholder="Example: Python, SQL, React, Git..."
        )

    ],

    outputs=gr.Markdown(),

    title="JD Analyzer & Skill Gap Detector",

    description=(
        "Analyze a Job Description and "
        "identify candidate skill gaps."
    )

)


# ==============================
# LAUNCH
# ==============================

interface.launch()
