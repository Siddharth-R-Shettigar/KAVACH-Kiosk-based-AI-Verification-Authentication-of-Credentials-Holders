# analyst.py
# The brain. Talks to the AI. Returns verdicts.

from groq import Groq
import os
import json
from dotenv import load_dotenv

# Open the lockbox(.env) and read the API key
# moves the keys from the file to the RAM
load_dotenv()

# Connect to Groq using that key
# Think of this as dialing the phone once and keeping the line open
# os.getenv goes to the RAM and grabs the key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# The AI's job description
# This is sent to the AI every single time
# It tells the AI who it is and what rules to follow
SYSTEM_PROMPT = """
You are a digital forensics analyst working for Vero.
Your job is to read test scores from 6 forensic modules
and decide if an image is real, fake, or inconclusive.

SCORING GUIDE:
- 0 to 30   = clean, no suspicion
- 31 to 60  = mild suspicion, not conclusive
- 61 to 80  = moderate suspicion, worth noting
- 81 to 100 = strong suspicion

RULES YOU MUST ALWAYS FOLLOW:

Rule 1 - Majority matters:
If 4 or more modules score below 30,
the image is likely real regardless of what others say.

Rule 2 - TruFor is strong but not a dictator:
If TruFor is above 80 BUT 4 or more other modules
score below 30, call it inconclusive, not likely_fake.
TruFor only decides alone if it scores above 90
AND at least 2 other modules also score above 60.

Rule 3 - Consensus overrules individuals:
No single module can call an image fake on its own.
At least 2 modules must score above 60 to call
something likely_fake.

Rule 4 - When in doubt say inconclusive:
If signals contradict each other do not force a verdict.
Inconclusive is a valid and honest answer.

Rule 5 - Never invent details:
Only use what is given to you.
Do not assume anything not in the scores and findings.

Rule 6 - Degraded media rule:
If the case file contains a WARNING about re-compression,
reduce your confidence score by at least 20 points.
Never report confidence above 70 for heavily
re-compressed media. Always mention why in warnings.

Always reply in this exact JSON format and nothing else:
{
  "verdict": "likely_real" or "likely_fake" or "inconclusive",
  "confidence": a number from 0 to 100,
  "final_score": a number from 0 to 100,
  "key_signals": ["reason 1", "reason 2", "reason 3"],
  "reasoning": "two sentence plain English explanation",
  "warnings": ["any limitations of this analysis"]
}

CRITICAL: Your entire response must be only the JSON object.
No text before it. No text after it. No markdown. No explanation.
Just the raw JSON starting with { and ending with }.
"""


def build_case_file(scores, preprocess=None):
    """
    Takes the scores dictionary and turns it into
    a formatted letter for the AI to read.

    scores     = dictionary with 12 items (6 scores + 6 details)
    preprocess = optional dictionary about compression history
    """

    # If the image was heavily re-compressed
    # add a warning at the top of the letter
    compression_warning = ""
    if preprocess and preprocess.get("heavily_compressed"):
        compression_warning = f"""
WARNING: This image has been re-compressed approximately
{preprocess['recompression_count']} times before reaching us.
ELA and frequency scores are unreliable for this image.
Weight only metadata and TruFor scores heavily.
Lower your confidence score accordingly.
"""

    # Build the letter by filling in the scores
    return f"""
Here are the forensic test results for the uploaded image.
{compression_warning}
METADATA SCORE: {scores['metadata']}/100
Finding: {scores['metadata_details']}

ELA SCORE: {scores['ela']}/100
Finding: {scores['ela_details']}

FREQUENCY SCORE: {scores['frequency']}/100
Finding: {scores['frequency_details']}

TRUFOR SCORE: {scores['trufor']}/100
Finding: {scores['trufor_details']}

PROVENANCE SCORE: {scores['provenance']}/100
Finding: {scores['provenance_details']}

PIXEL STATS SCORE: {scores['pixel_stats']}/100
Finding: {scores['pixel_stats_details']}

Write your forensic verdict now.
"""


def get_verdict(scores, preprocess=None):
    """
    Sends the scores to the AI and returns a verdict.

    scores     = dictionary with 12 items
    preprocess = optional compression history info
    """

    # Step 1: Build the letter from the scores
    case_file = build_case_file(scores, preprocess)

    # Step 2: Send the letter to the AI
    # Two messages go together every time:
    # - system: the job description (rules)
    # - user: the actual case file (scores)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": case_file}
        ]
    )

    # Step 3: Open the envelope and read the reply
    reply_text = response.choices[0].message.content.strip()

    # Step 4: Sometimes the AI wraps JSON in code blocks
    # Strip those out before parsing
    if reply_text.startswith("```"):
        lines = reply_text.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        reply_text = "\n".join(lines).strip()

    # Step 5: Find the JSON inside the reply
    # In case the AI added any text before or after
    start = reply_text.find("{")
    end   = reply_text.rfind("}") + 1
    if start != -1 and end != 0:
        reply_text = reply_text[start:end]

    # Step 6: Convert the text into a real dictionary
    # If it fails, return a safe fallback verdict
    try:
        verdict = json.loads(reply_text)

    except json.JSONDecodeError:
        print("DEBUG: AI returned unreadable response:")
        print(reply_text)
        verdict = {
            "verdict":     "inconclusive",
            "confidence":  0,
            "final_score": 0,
            "key_signals": [],
            "reasoning":   "Model returned unreadable response.",
            "warnings":    ["System error: could not parse model output"]
        }

    return verdict