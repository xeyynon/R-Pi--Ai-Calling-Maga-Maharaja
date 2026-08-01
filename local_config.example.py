# Copy this file to local_config.py and fill in your own values —
# local_config.py is gitignored, so it's never shared/committed.
#
# Setup steps for a new deployment (see deploy/setup_pi.sh for the
# scripted version of steps 3-4):
#   1. gcloud auth login
#   2. gcloud auth application-default login
#   3. In your Google Cloud project, enable:
#        Cloud Speech-to-Text API, Cloud Text-to-Speech API,
#        Vertex AI API (aiplatform.googleapis.com)
#   4. pip install -r requirements.txt
#        (or let deploy/setup_pi.sh create a venv and do this for you)
#   5. Get a Sarvam API key from https://dashboard.sarvam.ai — this is
#        the PRIMARY STT provider (Google STT is fallback/diagnostic
#        only, see companies/core/stt_provider.py).
#   6. Fill in your own project ID/number, a region close to you, and
#        your Sarvam key below, then run ai_caller_final3.py.

GOOGLE_CLOUD_PROJECT = "your-project-id-or-number-here"
GOOGLE_CLOUD_LOCATION = "asia-south1"  # pick the region closest to you

# 2026-08-01: this was missing from the example entirely — the app
# genuinely fails to start Sarvam STT without it (see
# ai_caller_final3.py's local_config import block), so anyone copying
# this file before now would hit a confusing runtime error with no
# hint that this line was the reason.
SARVAM_API_KEY = "your-sarvam-api-key-here"
