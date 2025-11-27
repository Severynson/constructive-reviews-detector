"""Label TripAdvisor reviews as constructive vs. not using gpt-5-nano.

The script reads a pipe-delimited CSV (default: data/tripadvisor_reviews.csv),
asks the model to classify each review, and writes a new CSV with a
"constructive" column containing 1 (constructive/contains useful business
feedback) or 0 (purely sentimental/insufficient detail).

Usage examples:
    python scripts/review_labler.py
    python scripts/review_labler.py --output data/tripadvisor_reviews.csv --relabel
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_INPUT = Path("data/tripadvisor_reviews.csv")
DEFAULT_OUTPUT = Path("data/tripadvisor_reviews_labeled.csv")
# SYSTEM_PROMPT = (
#     "You are a strict classifier of customer reviews. Your task is to decide whether a review contains "
#     "specific, actionable information that a business can use to improve its operations."
#     ""
#     "Return **1** ONLY if the review includes concrete, factual details such as:"
#     '- speed of service'
#     '- staff behavior, accuracy or errors (e.g. "waiter forgot to bring coffee", "hostess sat us at the wrong table", "cashier rolled their eyes at us")'
#     '- quality specifics (e.g., "food was cold", "burger was undercooked", "coffee tasted burnt", "latte was watery")'
#     '- environment details (e.g., "music was too loud", "bathroom was dirty")'
#     '- pacing issues (e.g., "drinks came out long after our food")'
#     '- portion specifics (e.g., "portion much smaller than last time", "order was missing the side salad")'
#     ""
#     "Return **0** if the review is:"
#     '- vague (e.g., "food was bad", "terrible place", "amazing experience", "I love the great history of this place")'
#     "- purely emotional or sentimental"
#     "- a general opinion without actionable details"
#     "- unclear whether any useful information is given"
#     ""
#     "Even if a review primarily expresses all the traits from the list to Return **0** BUT also includes at least one clear specific detail, return **1**."
#     ""
#     "When unsure, ALWAYS choose **0**."
#     ""
#     "Respond with a short sentence where you explain what traits from the list for 0 or 1 did you find, and ALWAYS finish your response with a single character: **1** or **0**."
# )

SYSTEM_PROMPT = (
    "You are a strict classifier of customer reviews. Your task is to decide whether a review contains "
    "specific, actionable information that a business can use to improve its operations.\n\n"
    "Return **1** ONLY if the review includes concrete, factual, operational details such as:\n"
    "- Speed of service (e.g., 'waited 20 minutes', 'service was slow')\n"
    "- Staff behavior, accuracy, attentiveness, or errors (e.g., 'waiter forgot the coffee', "
    "'hostess sat us at the wrong table', 'cashier rolled their eyes', 'server checked on us frequently')\n"
    "- Quality specifics of food or drink (e.g., 'food was cold', 'burger undercooked', "
    "'coffee tasted burnt', 'latte was watery', 'steak overcooked compared to medium rare')\n"
    "- Environment or cleanliness details (e.g., 'music too loud', 'bathroom dirty', "
    "'AC not working', 'tables were sticky')\n"
    "- Pacing and timing issues (e.g., 'drinks came out long after the food', "
    "'dessert took 15 minutes after ordering')\n"
    "- Portion or accuracy specifics (e.g., 'portion smaller than last time', 'missing side salad')\n"
    "- Menu item feedback with specific qualities (positive or negative), such as taste, temperature, or preparation "
    "(e.g., 'salmon was perfectly seasoned', 'cocktail was watered down', 'fries were soggy')\n"
    "- Operational or logistics problems (e.g., 'credit card reader didn’t work', "
    "'reservation not found', 'ran out of two menu items', 'seating next to drafty window')\n"
    "- Accessibility or accommodation issues (e.g., 'ramp was blocked', 'lighting too dim to read menu')\n"
    "- Disturbances caused by other customers that affect the experience (e.g., 'loud group arguing, staff didn't intervene')\n\n"
    "Return **0** if the review is:\n"
    "- Vague (e.g., 'food was bad', 'service needs work', 'great place', 'prices were high')\n"
    "- Purely emotional or sentimental (e.g., 'I was so disappointed', 'we had a wonderful time')\n"
    "- General praise or complaints without details (e.g., 'I loved the dish', 'wine list was great', "
    "'staff were fantastic', 'wait times need improvement')\n"  # without the exact time - it is too vague to be actionable info
    "- Narrative or personal context without actionable content (e.g., 'I came here with my friend', "
    "'we were celebrating a birthday')\n"
    "- Historical, cultural, or location commentary (e.g., 'I love the history of this place', "
    "'nice local spot')\n"
    "- Menu listing without evaluation (e.g., 'they offer vegan options', 'they have wines from Lodi')\n"
    "- Aesthetic impressions lacking a problem (e.g., 'cute decor', 'old-school vibe')\n"
    "- Statements implying issues without specifics (e.g., 'service needs improvement', 'food could be better')\n\n"
    "Even if a review contains many traits from the Return 0 list BUT includes at least one clear, "
    "specific, actionable detail from the Return 1 list, return **1**.\n\n"
    "When unsure, ALWAYS choose **0**.\n\n"
    "Respond with a short sentence explaining which traits from the lists you found, and ALWAYS finish "
    "your response with a single character: **1** or **0**."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label reviews as constructive or not."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to source CSV (pipe-delimited).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path (pipe-delimited). Defaults to a new file to preserve the original.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model to use for labeling (default: gpt-5-nano).",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Force relabeling even if a constructive value already exists in the input row.",
    )
    parser.add_argument(
        "--limit", type=int, help="Optionally stop after labeling this many rows."
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to pause between API calls to avoid rate limits (default: no pause).",
    )
    return parser.parse_args()


def label_constructive(
    client: OpenAI, model: str, title: str, text: str, retries: int = 3
) -> str:
    """Call the model to label a review. Returns "1" or "0"."""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"Title: {title}; Review: {text}",
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # Responses API (supports prompt caching automatically for long, repeated prefixes)
            response = client.responses.create(
                model=model,
                input=messages,
            )

            # Adjusted for Responses API – check your client version; in recent SDKs this is valid:
            llm_response = (response.output_text or "").strip()
            label = llm_response[-1]

            if label.startswith("1"):
                return ("1", llm_response)
            if label.startswith("0"):
                return ("0", llm_response)

            # Unexpected content: try again up to the retry budget.
            last_error = ValueError(f"Unexpected model reply: {label!r}")
            print(
                f"Unexpected model reply on attempt {attempt}: {label!r}. Retrying...",
                file=sys.stderr,
            )

        except Exception as exc:
            if attempt >= retries:
                raise
            last_error = exc
            wait_for = 2.0 * attempt
            print(
                f"API call failed ({exc}). Retrying in {wait_for:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait_for)
            continue

        if attempt < retries:
            wait_for = 0.1 * attempt
            time.sleep(wait_for)

    raise RuntimeError(
        f"Failed to get a valid label after {retries} attempt(s): {last_error}"
    )


def ensure_output_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def process_file(
    input_path: Path,
    output_path: Path,
    model: str,
    relabel: bool = False,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set to call the model.")

    client = OpenAI(api_key=api_key)

    ensure_output_dir(output_path)

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in, delimiter="|")
        if not reader.fieldnames:
            raise ValueError("Input CSV must have headers.")

        fieldnames = list(reader.fieldnames)
        has_constructive = "constructive" in fieldnames
        if not has_constructive:
            fieldnames.append("constructive")
        has_llm_logic_log = "llm_logic_log" in fieldnames
        if not has_llm_logic_log:
            fieldnames.append("llm_logic_log")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            if limit and idx > limit:
                break

            title = row.get("title") or row.get("review_title") or ""
            text = row.get("text") or row.get("review") or ""

            existing = (row.get("constructive") or "").strip()
            if not relabel and has_constructive and existing in {"0", "1"}:
                label = existing
            else:
                label, llm_logic_log = label_constructive(
                    client, model, title=title, text=text
                )

            row["constructive"] = label
            row["llm_logic_log"] = llm_logic_log
            writer.writerow(row)

            if sleep_seconds:
                time.sleep(sleep_seconds)

            if idx % 25 == 0:
                print(f"Labeled {idx} rows", file=sys.stderr)


def main() -> None:
    args = parse_args()
    process_file(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        relabel=args.relabel,
        limit=args.limit,
        sleep_seconds=args.sleep,
    )
    print(f"Done. Labeled data written to: {args.output}")


if __name__ == "__main__":
    main()
