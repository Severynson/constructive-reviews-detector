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

DEFAULT_MODEL = "gpt-5.1"
BATCH_SIZE = 20
DEFAULT_INPUT = Path("data/tripadvisor_reviews.csv")
DEFAULT_OUTPUT = Path("data/tripadvisor_reviews_labeled.csv")

SYSTEM_PROMPT = """You are a strict classifier of customer reviews. Your task is to decide whether a review contains 
specific, actionable information that a business can use to improve its operations.

Return **1** ONLY if the review includes concrete, factual, operational details such as:

- Speed of service 
  (e.g., "waited 20 minutes", "service was much slower than usual", "quick service")

- Staff behavior, accuracy, attentiveness, knowledge, or errors  
  (e.g., "waiter forgot the coffee", "hostess sat us at the wrong table", 
  "cashier rolled their eyes", "server checked on us frequently",
  "barista didn’t know the menu", "server could not explain allergens",
  "staff gave incorrect recommendations", "sommelier didn’t know basic pairings")

- Quality specifics of food or drink  
  (e.g., "food was cold", "burger undercooked", "coffee tasted burnt", 
  "latte was watery", "steak overcooked compared to medium rare")

- Environment, comfort, or cleanliness details  
  (e.g., "music too loud", "bathroom dirty", "live music is a reason why I return",
  "AC not working", "tables were sticky", "restaurant too cold/hot", 
  "strong chemical smell", "chair uncomfortable", "ventilation poor")

- Pacing and timing issues  
  (e.g., "drinks came out long after the food", 
  "dessert took 15 minutes after ordering", 
  "appetizer arrived after the entree", 
  "two dishes arrived 10 minutes apart")

- Portion or accuracy specifics  
  (e.g., "portion smaller than last time", "missing side salad", 
  "ordered medium rare but got well-done")

- Menu item feedback with specific, concrete qualities (positive or negative)  
  that mention *what exactly* was good or bad about the item, such as taste details, 
  texture, temperature, seasoning, portion size, or preparation
  (e.g., "salmon was perfectly seasoned with a crispy skin", 
  "cocktail was mostly ice and tasted watered down", 
  "fries were soggy and under-salted").  
  Generic statements like "the burger was yummy/tasty/decent/good/awful" WITHOUT further specifics 
  should be treated as vague and classified as 0.

- Operational or logistics problems  
  (e.g., "credit card reader didn’t work", "reservation not found", 
  "ran out of two menu items", "seating next to drafty window",
  "website booking system glitched", "OpenTable confirmation didn’t register",
  "waitlist process was confusing")

- Accessibility or accommodation issues  
  (e.g., "ramp was blocked", "lighting too dim to read menu", 
  "bathroom too cramped for wheelchair")

- Disturbances caused by other customers  
  (e.g., "loud group arguing, staff didn’t intervene", 
  "kids running around while staff ignored it")

- Pricing accuracy and billing issues  
  (e.g., "charged incorrectly", "item rang up $4 higher than menu price", 
  "surprise service fee added", "portion too small for the price", 
  "happy hour prices didn’t match the menu", "automatic tip added without warning")

- Takeout or delivery issues  
  (e.g., "takeout order missing items", "packaging leaked", 
  "food spilled in the bag", "delivery took 1 hour instead of 20 minutes",
  "container poorly sealed so fries were soggy")

- Dietary or allergy accommodation problems  
  (e.g., "gluten-free dish had croutons", "vegan item contained cheese", 
  "allergen labels inaccurate", "server unsure about ingredients")

- Menu clarity or accuracy problems  
  (e.g., "menu hard to read in dim light", "description didn’t match dish", 
  "menu outdated", "allergen icons incorrect", "items listed but unavailable")

- Queueing, seating, and entrance issues  
  (e.g., "had to wait 25 minutes despite reservation", 
  "host skipped our turn", "stood at door for 5 minutes without greeting", 
  "confusing entrance", "waiting line unorganized")

- Parking or external access issues  
  (e.g., "parking impossible", "valet lost keys", "no signage", 
  "unsafe or poorly lit parking area")

- Bussing, table turnover, and cleanup  
  (e.g., "dirty plates sat for 20 minutes", "table not wiped before seating",
  "crumbs on seat from previous guests")

- Technology and digital experience problems  
  (e.g., "Wi-Fi didn’t work", "QR code menu blurry", 
  "tablet ordering system froze", "online ordering glitchy")

- Layout, spacing, or seating ergonomics issues  
  (e.g., "tables too close together", "bench too low for table height", 
  "no space to move chair")

- Odor or scent problems  
  (e.g., "bathroom smelled bad", "strong cleaning chemical smell", 
  "kitchen smell overwhelming dining room")

- Consistency issues across visits  
  (e.g., "portion smaller than last time", "quality dropped since last month",
  "recipe changed and now less flavorful")

- Safety concerns  
  (e.g., "slippery floor", "broken step", "glass shard in food", 
  "loose electrical cable near seating area")

Return **0** if the review is:

- Vague  
  (e.g., "food was bad/good/tasty", "service needs work", 
  "great place", "prices were high")

- Purely emotional or sentimental without explaining details 
  (e.g., "I was so disappointed", "we had a wonderful time")

- General praise or complaints without details  
  (e.g., "I loved the dish", "wine list was great", "staff were fantastic")

- Narrative or personal context without actionable content  
  (e.g., "I came here with my friend", "we were celebrating a birthday")

- Historical, cultural, or location commentary  
  (e.g., "I love the history of this place", "nice local spot")

- Menu listing without evaluation  
  (e.g., "they offer vegan options", "they have wines from Lodi")

- Aesthetic impressions lacking a problem  
  (e.g., "cute decor", "old-school vibe")

- Statements implying issues without specifics  
  (e.g., "service needs improvement", "food could be better")

Even if a review contains many traits from the Return 0 list BUT includes at least one clear, 
specific, actionable detail from the Return 1 list, return **1**.

When unsure, ALWAYS choose **0**.

Respond with a short sentence explaining which traits from the lists you found, and ALWAYS finish 
your response with a single character: **1** or **0**."""


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


def label_constructive_batch(
    client: OpenAI,
    model: str,
    items: list[tuple[str, str]],
    retries: int = 3,
) -> list[tuple[str, str]]:
    """
    Batch version of label_constructive.

    items: list of (title, text)
    returns: list of (label, llm_logic_log) in the same order.
    """
    # Build a single user prompt containing all reviews
    review_blocks = []
    for i, (title, text) in enumerate(items, start=1):
        review_blocks.append(f"Review {i}:\nTitle: {title}\nReview: {text}")

    user_content = (
        "Classify each of the following reviews as constructive (1) or not (0).\n"
        "For each review, respond on a separate line in this exact format:\n"
        "Review <N>: <short explanation>  <label>\n"
        "where <label> is a single character 1 or 0 as the LAST character on the line.\n\n"
        + "\n\n".join(review_blocks)
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # max_completion_tokens=items.__len__() * 16,  # a few tokens per line
                # temperature=0,
            )
            full_response = (response.choices[0].message.content or "").strip()

            # Split into non-empty lines
            lines = [ln.strip() for ln in full_response.splitlines() if ln.strip()]

            if len(lines) != len(items):
                raise ValueError(
                    f"Expected {len(items)} lines, got {len(lines)}. Response was:\n{full_response}"
                )

            results: list[tuple[str, str]] = []
            for line in lines:
                label = line[-1]
                if label not in {"0", "1"}:
                    raise ValueError(f"Line does not end with 0 or 1: {line!r}")
                results.append((label, line))

            return results

        except Exception as exc:
            if attempt >= retries:
                raise
            last_error = exc
            wait_for = 2.0 * attempt
            print(
                f"Batch API call failed ({exc}). Retrying in {wait_for:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait_for)
            continue

    raise RuntimeError(
        f"Failed to get valid batch labels after {retries} attempt(s): {last_error}"
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

        pending: list[tuple[dict, str, str]] = []  # (row, title, text)

        for idx, row in enumerate(reader, start=1):
            if limit and idx > limit:
                break

            title = row.get("title") or row.get("review_title") or ""
            text = row.get("text") or row.get("review") or ""

            existing = (row.get("constructive") or "").strip()
            if not relabel and has_constructive and existing in {"0", "1"}:
                # No relabel needed; write row immediately.
                row["constructive"] = existing
                # Keep existing llm_logic_log if present, otherwise blank.
                if not has_llm_logic_log:
                    row["llm_logic_log"] = row.get("llm_logic_log", "")
                writer.writerow(row)
            else:
                # Queue for batch labeling
                pending.append((row, title, text))

                # If batch is full, flush it
                if len(pending) >= BATCH_SIZE:
                    batch_items = [(t, x) for (_, t, x) in pending]
                    batch_results = label_constructive_batch(client, model, batch_items)

                    for (row_pending, _, _), (label, llm_logic_log) in zip(
                        pending, batch_results
                    ):
                        row_pending["constructive"] = label
                        row_pending["llm_logic_log"] = llm_logic_log
                        writer.writerow(row_pending)

                    pending.clear()

            if sleep_seconds:
                time.sleep(sleep_seconds)

            if idx % 25 == 0:
                print(f"Labeled {idx} rows", file=sys.stderr)

        # Flush any remaining pending rows after the loop
        if pending:
            batch_items = [(t, x) for (_, t, x) in pending]
            batch_results = label_constructive_batch(client, model, batch_items)

            for (row_pending, _, _), (label, llm_logic_log) in zip(
                pending, batch_results
            ):
                row_pending["constructive"] = label
                row_pending["llm_logic_log"] = llm_logic_log
                writer.writerow(row_pending)


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
