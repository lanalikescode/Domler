# Domler

Domain filtering and buyer matching utilities for `.com` auctions.

## Offline filter (`domler.py`)

1. Place `auction.csv` and `dictionary.txt` in the project directory.
2. Run `python domler.py`.
3. Review the generated outputs:
   * `domler_filtered.csv` – survivors with a scoring breakdown.
   * `domler_calllist.csv` – top 100 domains by score.

## Matcher (`matcher.py`)

1. Ensure `domler_filtered.csv` exists from the filter step.
2. Provide a Google Maps API key via the `GOOGLE_MAPS_API_KEY` environment variable or by creating `api.txt` with the key on the first line.
3. Run `python matcher.py --stats` for the default workflow. Useful flags:
   * `--region us` – bias Text Search to a specific region.
   * `--limit 400` – cap API calls (negative disables the cap).
   * `--per_domain 10` – adjust matches kept per domain.
   * `--cache-dir .domler_cache_cq` – customise cache location.
   * `--refresh-queries` / `--refresh-details` – ignore cached responses.
   * `--nuke-cache` – delete cache files before running.

Matches are written to `domler_matches.csv` with global website deduplication and optional cache statistics when `--stats` is used.
