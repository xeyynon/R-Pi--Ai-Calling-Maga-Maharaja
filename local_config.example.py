# Copy this file to local_config.py and fill in your own values —
# local_config.py is gitignored, so it's never shared/committed.
#
# Setup steps for a new deployment:
#   1. gcloud auth login
#   2. gcloud auth application-default login
#   3. In your Google Cloud project, enable:
#        Cloud Speech-to-Text API, Cloud Text-to-Speech API,
#        Vertex AI API (aiplatform.googleapis.com)
#   4. pip install google-cloud-speech google-cloud-texttospeech
#        google-cloud-aiplatform google-genai pyyaml webrtcvad pytz
#   5. Fill in your own project ID/number and a region close to you
#        below, then run ai_caller_final2.py.

GOOGLE_CLOUD_PROJECT = "your-project-id-or-number-here"
GOOGLE_CLOUD_LOCATION = "asia-south1"  # pick the region closest to you
