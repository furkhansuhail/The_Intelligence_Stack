"""
topic_template.py
─────────────────
Copy this file into the relevant paradigm's topics/ folder.
Rename it with a numeric prefix: e.g. 10_new_topic.py
Fill in all the sections below.
The app auto-discovers it — no registration needed.
"""

DISPLAY_NAME = "XX · Your Topic Name"
ICON         = "📖"          # Pick an emoji
SUBTITLE     = "One-line tagline explaining the topic"

THEORY = """
## XX · Your Topic Name

Write your full theory here using Markdown.
Include:
  - Core concept explanation
  - Key equations (use code blocks for math)
  - Assumptions / edge cases
  - Where this fits in the learning path
"""

OPERATIONS = {
    "Operation Name": {
        "description": "One-line description of what this step demonstrates",
        "code": """\
# Runnable Python code goes here
print("Hello, ML!")
""",
    },
    # Add more operations as needed...
}

VISUAL_HTML = ""  # Paste your HTML breakdown string here (from Required_Images/)
