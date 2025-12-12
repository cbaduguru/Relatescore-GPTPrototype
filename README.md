# RelateScore™ Streamlit Prototype

This is a runnable Streamlit prototype that implements the flow shown in **RELATESCORE FLOWCHART.pdf**:
- Invite code + dual consent
- Guided reflection prompts + self-rating
- Toxicity gate (blocks harmful language)
- Normalized 0–100 category scores and weighted RSQ
- Outlier dampening + EMA time-weighting
- Private dashboard + consent withdrawal (clears thread)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repo and add these files.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the main file path to `app.py`.
