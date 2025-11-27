import os
import re
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set
import requests
from dotenv import load_dotenv

load_dotenv()

# =========================================================
#  CONFIG
# =========================================================

TRIPADVISOR_API_KEY = os.environ["TRIPADVISOR_API_KEY"]  # API key from environment

BASE_URL = "https://api.content.tripadvisor.com/api/v1"
REVIEWS_ENDPOINT = f"{BASE_URL}/location/{{location_id}}/reviews"

DATA_DIR = Path("data")
TO_CHECK_PATH = DATA_DIR / "to_check_locations.tsv"
CHECKED_PATH = DATA_DIR / "checked_locations.tsv"
REVIEWS_OUT_PATH = DATA_DIR / "tripadvisor_reviews.csv"


# =========================================================
#  UTILS
# =========================================================


def sanitize(text: str) -> str:
    """Replace '|', newlines, and strip whitespace to keep pipe-delimited CSV safe."""
    if not text:
        return ""
    # Replace newlines and carriage returns with a space to ensure one line per review
    text_no_newlines = str(text).replace("\n", " ").replace("\r", " ")
    # Replace the pipe delimiter as it's used for columns
    return text_no_newlines.replace("|", ";").strip()


def extract_location_id_from_url(url: str) -> str | None:
    """Parse -dXXXXX- ID from TripAdvisor URLs."""
    if not url:
        return None
    m = re.search(r"-d(\d+)-", url)
    if m:
        return m.group(1)
    return None


# =========================================================
#  CSV LOADERS
# =========================================================


def load_locations_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV with header: title,url"""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            if title or url:
                items.append({"title": title, "url": url})
    return items


def load_checked_titles_and_urls(path: Path) -> Tuple[Set[str], Set[str]]:
    """Return sets of titles and urls already processed."""
    titles: Set[str] = set()
    urls: Set[str] = set()
    if not path.exists():
        return titles, urls

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            t = (row.get("title") or "").strip()
            u = (row.get("url") or "").strip()
            if t:
                titles.add(t)
            if u:
                urls.add(u)
    return titles, urls


def append_checked_location(path: Path, title: str, url: str) -> None:
    """Append a processed location to checked_locations.csv."""
    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url"], delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow({"title": title, "url": url})


# =========================================================
#  TRIPADVISOR API
# =========================================================


def get_location_reviews(location_id: str, language: str = "en", max_retries: int = 3) -> List[dict]:
    """
    Fetch up to 5 most recent reviews for a given location_id.
    Includes retry logic for rate limiting (429 errors).
    """
    url = REVIEWS_ENDPOINT.format(location_id=location_id)

    params = {
        "key": TRIPADVISOR_API_KEY,
        "language": language,
    }

    # Required because you use DOMAIN restriction in TripAdvisor console
    headers = {
        "Referer": "https://github.com",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers)

            if resp.status_code == 429:
                print(f"   ... Rate limited (429). Waiting 60s before retry {attempt + 1}/{max_retries}")
                time.sleep(60)
                continue  # Go to the next attempt

            resp.raise_for_status()  # Raise HTTPError for other bad responses (4xx or 5xx)

            data = resp.json()
            return data.get("data") or []

        except requests.exceptions.RequestException as e:
            print(f"   ... An error occurred during request: {e}")
            if attempt < max_retries - 1:
                time.sleep(5) # Wait a bit before retrying on network errors

    print(f"   ... Failed to fetch reviews for location {location_id} after {max_retries} attempts.")
    return []


# =========================================================
#  MAIN PIPELINE
# =========================================================


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    to_check = load_locations_csv(TO_CHECK_PATH)
    checked_titles, checked_urls = load_checked_titles_and_urls(CHECKED_PATH)

    print(f"Loaded {len(to_check)} locations to check")
    print(f"Already checked: {len(checked_titles)} titles, {len(checked_urls)} urls")

    # Prepare output CSV
    file_exists = REVIEWS_OUT_PATH.exists()
    with REVIEWS_OUT_PATH.open("a", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="|")
        if not file_exists:
            writer.writerow(
                [
                    "location_id",
                    "location_title",
                    "location_url",
                    "review_id",
                    "rating",
                    "published_date",
                    "title",
                    "text",
                    "language",
                    "review_url",
                    "user_name",
                    "trip_type",
                    "travel_date",
                    "helpful_votes",
                ]
            )

        for loc in to_check:
            title = loc["title"]
            url = loc["url"]

            # Skip if already checked
            if title in checked_titles or url in checked_urls:
                print(f"⏭️  Skipping already-checked: {title}")
                continue

            print(f"\n=== Processing: {title} ===")
            location_id = extract_location_id_from_url(url)
            if not location_id:
                print(f"⚠️ Cannot extract location_id from URL: {url}")
                append_checked_location(CHECKED_PATH, title, url)
                checked_titles.add(title)
                checked_urls.add(url)
                continue

            print(f"   location_id = {location_id}")

            # Fetch reviews (up to 5 from the API)
            reviews = get_location_reviews(location_id, language="en")

            print(f"   → Retrieved {len(reviews)} review(s) (before language filter)")

            english_count = 0

            # Write reviews
            for r in reviews:
                # Language check (only keep English)
                lang = (r.get("language") or r.get("lang") or "").lower()
                # If language is provided and not English -> skip
                if lang and lang != "en":
                    continue
                # If missing, assume English because we requested language="en"
                lang_to_write = lang if lang else "en"

                rid = str(r.get("id", "")).strip()
                rating = r.get("rating") or r.get("rating_value")
                published = r.get("published_date") or r.get("publishedDate")
                r_title = sanitize(r.get("title", ""))
                text = sanitize(
                    r.get("text") or r.get("review_text") or r.get("review") or ""
                )
                if not text:
                    continue

                # Extra attributes
                review_url = sanitize(r.get("url") or r.get("web_url") or "")

                user = r.get("user") or {}
                user_name = sanitize(user.get("username") or user.get("name") or "")

                trip_type = sanitize(r.get("trip_type") or "")
                travel_date = r.get("travel_date") or r.get("travelDate")

                helpful_votes = r.get("helpful_votes") or r.get("helpful_vote_count")

                writer.writerow(
                    [
                        location_id,
                        sanitize(title),
                        url,
                        rid,
                        rating,
                        published,
                        r_title,
                        text,
                        lang_to_write,
                        review_url,
                        user_name,
                        trip_type,
                        travel_date,
                        helpful_votes,
                    ]
                )

                english_count += 1

            print(f"   → Stored {english_count} English review(s)")

            # Mark as checked (even if 0 usable reviews)
            append_checked_location(CHECKED_PATH, title, url)
            checked_titles.add(title)
            checked_urls.add(url)

    print(f"\n✅ Finished. Reviews saved to: {REVIEWS_OUT_PATH}")


if __name__ == "__main__":
    main()
