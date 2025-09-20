"""
Configuration settings for the RAG demo application.
"""

# Model Configuration
MODEL_NAME = "google/flan-t5-large"  # Can be swapped with flan-t5-base for faster/smaller
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text Processing Configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# Retrieval Configuration
DEFAULT_TOP_K = 3
MAX_NEW_TOKENS = 256

# Demo Data
DEMO_DOCUMENTS = [
    "Dr. Sarah Johnson is a cardiologist available on Mondays, Wednesdays, and Fridays.",
    "Dr. Emily Chen is a dermatologist available on Saturdays.",
    "Dr. Amit Verma is a pediatrician available on Tuesdays and Thursdays.",
    "The clinic is closed on Sundays and public holidays.",
    "Patients must check in at the reception desk 15 minutes before their appointment.",
    "The clinic partners with several insurance providers. Claims are processed only on weekdays.",
    "Vaccination drives are held on the first Saturday of every month.",
]

# Demo Queries
DEMO_QUERIES = [
    "When is Dr. Sarah available?",
    "Can I see a doctor on Sunday?",
    "If I want a skin checkup on a Sunday, can I see a doctor?",
    "When should I bring insurance documents for a weekday appointment?",
    "If I want to see both a cardiologist and a dermatologist on the same day, which day should I visit?",
]
