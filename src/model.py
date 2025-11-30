from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch

DEFAULT_MODEL_NAME = (
    "bert-base-uncased"  # or "distilbert-base-uncased", "deberta-v3-base".
)
DEFAULT_NUM_LABELS = 2


class ReviewClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = DEFAULT_NUM_LABELS,
        max_length: int = 256,
    ):
        """
        Use this to create a fresh model for training from a pretrained backbone.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )

    @classmethod
    def from_finetuned(cls, model_dir: str, max_length: int = 256):
        """
        Use this to load a model that you already fine-tuned and saved
        (e.g. from trainer.save_model()).
        """
        obj = cls.__new__(cls)
        obj.model_name = model_dir
        obj.max_length = max_length

        obj.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        obj.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        obj.num_labels = obj.model.config.num_labels

        return obj

    def tokenize_batch(self, batch):
        """
        Tokenization function compatible with Hugging Face Datasets .map().
        Uses the already-loaded tokenizer; does NOT reload it every time.
        """
        return self.tokenizer(
            batch["review"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def df_to_dataset(
        self,
        df,
        text_col: str = "review",
        label_col: str = "constructive",
    ):
        """
        Convert a pandas DataFrame with text + label columns into a tokenized
        torch-ready Hugging Face Dataset.
        """
        # Make sure correct types
        # df = df.copy()
        # df[text_col] = df[text_col].astype(str)
        # df[label_col] = df[label_col].astype(int)

        # Hugging Face Dataset
        ds = Dataset.from_pandas(df.reset_index(drop=True))

        # Use the column name expected by tokenize_batch
        if text_col != "review":
            ds = ds.rename_column(text_col, "review")

        # Tokenize
        ds = ds.map(self.tokenize_batch, batched=True)

        # Rename label column to "labels" for HF models
        ds = ds.rename_column(label_col, "labels")

        # Keep only model inputs + labels as torch tensors
        cols = ["input_ids", "attention_mask", "labels"]
        ds.set_format(type="torch", columns=cols)

        return ds

    def predict(self, text: str, threshold: float = 0.5):
        """
        Predict whether a review is constructive (1) or not (0).
        Returns (label, probability_of_constructive).
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        p1 = probs[1].item()
        label = 1 if p1 >= threshold else 0

        return label, p1
