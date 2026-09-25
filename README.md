# Airline Chatbot

An intent-classification chatbot for the airline and travel domain. Given a natural-language query such as "Can I bring two suitcases?", it predicts the user's intent and returns an appropriate response.

Live demo: [add Streamlit URL once deployed]

## What It Does

- Classifies user messages into 33 travel-related intents covering booking, baggage, cancellation, refunds, seat selection, insurance, and related topics.
- Returns a curated, human-readable response for each intent.
- Filters off-topic queries and asks for clarification when the model is uncertain.
- Serves as a web application via Streamlit.

Baseline performance: 97.6% test accuracy and 97.6% macro-F1 using TF-IDF and Logistic Regression.

## Architecture

User input flows through the Streamlit UI, then a domain filter, then the intent classifier, then a response map, and finally returns a reply.

Three components:

1. Intent classifier: TF-IDF vectorizer plus Logistic Regression, packaged as a scikit-learn pipeline.
2. Response map: a lookup from intent to canned bot reply, built from the dataset.
3. Streamlit app: a public web interface.

Training and data preparation happen in Google Colab. Artifacts are stored in Google Drive and synced to the repository for deployment.

## Repository Structure

    AirlineChatbot/
    ├── .devcontainer/
    │   └── devcontainer.json         VS Code Dev Container configuration
    ├── artifacts/                    Model files required for deployment
    │   ├── label_map.json            Maps intent_id to intent_name
    │   ├── metrics.json              Evaluation metrics from training
    │   ├── model.joblib              Trained classifier
    │   └── response_map.json         Maps intent_name to bot reply
    ├── notebooks/                    Colab development notebooks
    │   ├── 00_bootstrap.ipynb        Session setup (Drive mount, repo clone, deps)
    │   ├── 01_download_data.ipynb    One-time dataset download
    │   ├── 02_preprocess.ipynb       Cleaning, encoding, splitting
    │   └── 03_train_baseline.ipynb   Model training and evaluation
    ├── .gitignore
    ├── README.md
    ├── requirements.txt              Python dependencies
    └── streamlit_app.py              Web app entry point (Streamlit Cloud runs this)

Where things live:

- Code lives in GitHub.
- Data and trained models live in Google Drive under MyDrive/AirlineChatbot.
- Compute happens in Google Colab.
- Deployment happens on Streamlit Community Cloud.

## Quick Start

Try the deployed app:

    [Add Streamlit URL here once live]

Run locally:

    # 1. Clone
    git clone https://github.com/NMapelu/AirlineChatbot.git
    cd AirlineChatbot

    # 2. Create and activate a virtual environment
    python3 -m venv .venv
    source .venv/bin/activate          # macOS/Linux
    # .venv\Scripts\Activate.ps1       # Windows

    # 3. Install dependencies
    pip install -r requirements.txt

    # 4. Run the app
    streamlit run streamlit_app.py

Then open http://localhost:8501 in your browser.

## Dataset

Bitext Travel LLM Chatbot Training Dataset.

Link: https://huggingface.co/datasets/bitext/Bitext-travel-llm-chatbot-training-dataset

Properties:

- Size: 31,658 instruction and response pairs
- Intents: 33, grouped across 11 categories
- Columns: instruction, intent, category, tags, response
- License: CDLA-Sharing-1.0
- Scope: Global travel, not US-specific

Sample intents: book_flight, cancel_flight, check_baggage_allowance, change_seat, get_refund, human_agent, check_in, and 26 more.

## Reproducing the Pipeline

All training happens in Colab. To reproduce:

1. Open notebooks/00_bootstrap.ipynb in Colab and run all cells. This mounts Drive, clones the repo, installs dependencies, and sets environment variables.

2. Run notebooks/01_download_data.ipynb once. It downloads the raw CSV to Drive at data/raw/bitext_travel.csv.

3. Run notebooks/02_preprocess.ipynb whenever the raw data changes. It produces data/processed/train.csv, data/processed/val.csv, data/processed/test.csv, and data/processed/label_map.json.

4. Run notebooks/03_train_baseline.ipynb whenever training data or code changes. It produces a timestamped run folder under outputs/models/baseline_<timestamp>/ containing model.joblib, label_map.json, metrics.json, and classification_report.txt.

5. Copy artifacts into the repo for deployment:

       model.joblib      to artifacts/model.joblib
       label_map.json    to artifacts/label_map.json
       response_map.json to artifacts/response_map.json
       metrics.json      to artifacts/metrics.json

6. Push to GitHub. Streamlit Cloud redeploys automatically.

## Model Details

Vectorizer:

- TfidfVectorizer
- ngram_range of (1, 2)
- min_df of 2
- max_df of 0.95
- sublinear_tf enabled

Rationale: captures single words and bigrams, discards very rare tokens, and dampens the effect of repeated terms.

Classifier:

- LogisticRegression
- multinomial
- C of 1.0
- max_iter of 1000

Rationale: fast, interpretable, and handles 33-class multinomial classification natively.

Split:

- 80 percent train, 10 percent validation, 10 percent test
- Stratified by label
- random_state of 42

Rationale: every split contains all 33 intents in proportional amounts, and the split is reproducible.

Text cleaning:

- Lowercase
- Strip leading and trailing whitespace
- Collapse consecutive whitespace

Rationale: deliberately minimal to preserve punctuation and stopwords, which carry signal in travel queries.

Evaluation:

- Validation accuracy: 0.9784
- Test accuracy: 0.9762
- Test macro-F1: 0.9762

Training runs in approximately 4 seconds on Colab CPU.

## Development Workflow

1. Edit code locally in VS Code.
2. Push to GitHub.
3. Pull into Colab via the bootstrap notebook.
4. Train or test in Colab.
5. Save artifacts to Drive, then copy them into artifacts/ and push.
6. Streamlit Cloud redeploys automatically.

Data and models never touch Git. Only code and small artifacts do.

## Development Container

This repository includes a .devcontainer/devcontainer.json configuration. If you use VS Code with the Dev Containers extension, you can open the project inside a reproducible container with all dependencies preinstalled.

To use it:

1. Install the Dev Containers extension in VS Code.
2. Open the repository folder.
3. When prompted, click "Reopen in Container".

## Testing the Model

Quick sanity check in Python:

    import joblib, json

    model = joblib.load("artifacts/model.joblib")
    label_map = {int(k): v for k, v in json.load(open("artifacts/label_map.json")).items()}

    text = "I want to book a flight to Paris"
    pred = model.predict([text.lower()])[0]
    print(label_map[pred])   # prints book_flight

## Roadmap

Completed:

- Baseline classifier at 97.6 percent accuracy
- Response mapping from dataset
- Off-topic filtering and clarification prompts
- Streamlit web deployment

Planned:

- Fine-tune DistilBERT for higher accuracy
- Add entity extraction for cities, dates, and flight numbers
- Multi-turn conversation state
- Collect user feedback for continuous improvement

## License

Add a license file. MIT or Apache-2.0 are common choices.

The dataset is licensed under CDLA-Sharing-1.0: https://huggingface.co/datasets/bitext/Bitext-travel-llm-chatbot-training-dataset

## Acknowledgements

- Dataset: Bitext Travel LLM Chatbot Training Dataset
- Hosting: Hugging Face Datasets
- Model: scikit-learn TfidfVectorizer and LogisticRegression
- Deployment: Streamlit Community Cloud