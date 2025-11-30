# 🧠 🆚 🎭 constructive-reviews-detector

This project builds an end-to-end pipeline for **detecting constructive customer reviews** — reviews that contain **specific, actionable feedback** that a business can use to improve (e.g. “service was slow”, “music was too loud”, “burger was undercooked”), as opposed to purely **emotional or vague** comments (e.g. “worst place ever”, “amazing!!”).

⸻

## 🗂️ It consists of:

1. **Data collection** from TripAdvisor (via API)
2. **LLM-based labeling** (OpenAI) of reviews as constructive vs. non-constructive
3. **Supervised training** of a BERT-style classifier on the labeled dataset
4. **Local inference** using the fine-tuned classifier

All TripAdvisor reviews in this project were collected by me and are stored in the `data/` folder (see [Dataset](#dataset) section).

⸻

## 🎯 Current Test set Metrics
**Cross-Entropy Loss:** `0.3405`  
**Accuracy:** `0.8824`
**F1 Score:** `0.8800`

⸻

## 📂 Project structure

```
constructive-reviews-detector/
├── LICENSE
├── README.md
├── environment.yml           # Conda environment definition
├── data/
│   ├── tripadvisor_reviews.csv          # raw, unlabeled reviews from TripAdvisor API
│   ├── tripadvisor_reviews_labeled.csv  # same, but labeled by OpenAI
│   ├── reviews.tsv                      # trimmed, balanced dataset (review + constructive)
│   └── to_check_locations.tsv           # list of locations to fetch from TripAdvisor
├── models/
│   └── ... (fine-tuned classifier, e.g. models/review_classifier/)
├── notebooks/
│   ├── data_preprocessing.ipynb  # cleaning, balancing, preparing data for training
│   ├── model_training.ipynb      # fine-tuning BERT on labeled data
│   └── model_inference.ipynb     # examples of running inference
├── scripts/
│   ├── data_collector.py   # pulls reviews from TripAdvisor API
│   └── review_labler.py    # labels reviews as constructive vs. not using OpenAI
└── src/
    └── model.py            # ReviewClassifier wrapper around Hugging Face model
```

⸻

## 📊 Dataset

All data in data/ was collected by me specifically for this project.

The three most important files:

- `data/tripadvisor_reviews.csv`
Raw, unlabeled TripAdvisor reviews pulled via the TripAdvisor Content API.
Each row includes location info, rating, title, text, language, user name, etc.

- `data/tripadvisor_reviews_labeled.csv` - Same reviews, but with additional columns:
    - constructive – 1 if the LLM judged the review to contain actionable, specific feedback, 0 otherwise
    - llm_logic_log – the LLM’s short explanation of its decision

- `data/reviews.tsv` - A trimmed and cleaned dataset containing only:
    - review – combined title+text (and possibly some light context)
    - constructive – 0 or 1

This dataset is also balanced: the number of constructive reviews (constructive = 1) is approximately equal to the number of purely sentimental / non-constructive reviews (constructive = 0).
It is used as the main input for training and evaluation of the BERT-style classifier.
ℹ️ Note: TripAdvisor’s Content API only returns up to 5 most recent reviews per location, so data/to_check_locations.tsv determines which locations were queried.

⸻

## 🛠️ Installation & environment

The project is designed to be reproducible via Conda.

1. Create the environment
```
conda env create -f environment.yml
```

2. Activate it
```
conda activate constructive-reviews-detector
```

⸻

## 🌐 1. Data collection from TripAdvisor

Script: scripts/data_collector.py

This script:

- Reads a list of locations from data/to_check_locations.tsv with columns:
    - `title`
    - `url` (TripAdvisor location page)
- Extracts `location_id` from the URL (via the `-dXXXXX-` pattern)
- Calls the TripAdvisor Content API to fetch up to 5 most recent reviews per location
- Filters/normalizes text and writes reviews to `data/tripadvisor_reviews.csv` (pipe `|` delimited)
- Tracks already-processed locations in `data/checked_locations.tsv`

## 🔑 Required environment variable

Define in .env:
```TRIPADVISOR_API_KEY=your_tripadvisor_content_api_key```

## ▶️ Run the collector
```
conda activate constructive-reviews-detector
python scripts/data_collector.py
```

Outputs:

- `data/tripadvisor_reviews.csv` – raw, unlabeled reviews
- `data/checked_locations.tsv` – locations that have already been processed

⸻

## 🤖 2. Labeling reviews with OpenAI

Script: `scripts/review_labler.py`

This script:

- Reads `data/tripadvisor_reviews.csv` (pipe `|` delimited)
- Sends batched reviews to an OpenAI model (default: `gpt-5.1`)
- Uses a strict system prompt to decide for each review:
    - `1` – constructive / contains specific, actionable feedback
    - `0` – purely sentimental / vague / not actionable
- Writes results to data/tripadvisor_reviews_labeled.csv, adding:
    - `constructive`
    - `llm_logic_log`

## 🔑 Required environment variable

Add to `.env`:

```
OPENAI_API_KEY=your_openai_api_key
```

## ▶️ Run the labeling script
```
python scripts/review_labler.py \
  --input data/tripadvisor_reviews.csv \
  --output data/tripadvisor_reviews_labeled.csv \
  --model gpt-5.1
```
Useful flags:
- `--relabel` – re-label rows even if `constructive` already exists
- `--limit N` – process only the first N rows
- `--sleep S` – sleep `S` seconds between API calls to avoid rate limits
See all options:
```
python scripts/review_labler.py --help
```

⸻

## 📜 3. AI classifier prompt (LLM labeling criteria)
The full system prompt that defines what “constructive” means is stored in:
- `scripts/review_labler.py` → SYSTEM_PROMPT
In short, the LLM returns 1 only if the review includes concrete, factual, operational details, such as:
- Speed of service
- Staff behavior, attentiveness, knowledge, or errors
- Specific food/drink quality details (temperature, texture, seasoning, doneness, etc.)
- Environment, comfort, noise level, cleanliness
- Timing/pacing issues, portion size differences, order accuracy
- Operational / logistics problems (billing issues, reservation failures, delivery problems, etc.)
- Accessibility, safety concerns, menu clarity, parking/entrance issues, and so on

The model returns 0 when the review is:
- Vague (“food was bad/good”, “service needs improvement”)
- Purely emotional (“we were so disappointed”, “amazing vibe”)
- Just storytelling / narrative without actionable content
- General praise or criticism without specifying what exactly was good or bad
When in doubt, the prompt instructs the LLM to choose 0.

For exact wording (which is important if you want to reproduce labeling), please refer directly to `scripts/review_labler.py`.

⸻

## 🧬 4. Training the BERT-style classifier

Core class: `src/model.py` → `ReviewClassifier`
Notebooks: `notebooks/data_preprocessing.ipynb`, `notebooks/model_training.ipynb`

The classifier is built on a Hugging Face backbone (`bert-base-uncased`) and wrapped in a small helper class that:
- Loads the tokenizer and model from `transformers`
- Converts a pandas DataFrame (like `data/reviews.tsv`) to a tokenized Hugging Face Dataset
- Supports `.predict(text, threshold=...)` for local inference
The training process (implemented in `notebooks/model_training.ipynb`) typically does:
1. Load `data/reviews.tsv` (columns: `review`, `constructive`).
2. Split into train / validation / test sets (e.g. using `train_test_split`).
3. Use `ReviewClassifier.df_to_dataset(...)` to turn DataFrames into tokenized datasets.
4. Fine-tune the model with `transformers.Trainer` + `TrainingArguments`.
5. Save the best model and tokenizer into `models/review_classifier/` using the Trainer API.
For detailed training code and hyperparameters, check the `model_training.ipynb` notebook.

⸻

## 🔍 5. Running inference with the fine-tuned model
Files involved:
- Fine-tuned model directory: `models/review_classifier/`
- Helper class: src/model.py → `ReviewClassifier`
- Examples: `notebooks/model_inference.ipynb`

Typical usage:
1. Load the fine-tuned model using `ReviewClassifier.from_finetuned("models/review_classifier")`.
2. Call `.predict(text, threshold=...)` for a single review.
3. Interpret:
    - the returned label (`1` = constructive, `0` = non-constructive)
    - the probability of class 1 (constructive) returned alongside the label
The `model_inference.ipynb` notebook contains concrete examples of loading the model, running predictions, and interpreting outputs.

⸻

## 🔁 6. Reproducing the full pipeline

To recreate the entire process from raw locations → labeled dataset → trained classifier:
1. Set up environment
```
conda env create -f environment.yml
conda activate constructive-reviews-detector
```
2. Prepare TripAdvisor locations
- Add locations to `data/to_check_locations.tsv` with columns `title` and `url`.
3. Collect reviews
- Ensure `.env` contains `TRIPADVISOR_API_KEY`.
Run:
```
python scripts/data_collector.py
```
- This produces `data/tripadvisor_reviews.csv`.
4. Label reviews with OpenAI
- Ensure `.env` contains `OPENAI_API_KEY`.
- Run:
```
python scripts/review_labler.py \
  --input data/tripadvisor_reviews.csv \
  --output data/tripadvisor_reviews_labeled.csv
```
5. Prepare final training dataset
- Use `notebooks/data_preprocessing.ipynb` to:
    - clean and merge title+text
    - create `data/reviews.tsv` with `review` + `constructive`
    - balance the classes (roughly 50/50 constructive vs. non-constructive)
6. Train the classifier
- Run notebooks/model_training.ipynb to:
    - load `data/reviews.tsv`
    - split into train / val / test
    - fine-tune the model
    - save it under `models/review_classifier/`
7. Run inference
- Use `notebooks/model_inference.ipynb` or your own script that:
    - imports `ReviewClassifier` from `src/model.py`
    - loads the model from `models/review_classifier/`
    - calls `.predict()` on your own review texts

⸻

## 📜 License:
MIT licence.
