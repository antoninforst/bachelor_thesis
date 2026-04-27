"""Clean language_overview.csv names.

If the main name column is empty but alternative_name has a value, move the
alternative name into name.
"""

import csv
from pathlib import Path


CSV_PATH = Path(__file__).with_name("language_overview.csv")


def main() -> None:
	with CSV_PATH.open(encoding="utf-8", newline="") as fh:
		reader = csv.DictReader(fh)
		rows = list(reader)
		fieldnames = reader.fieldnames

	if fieldnames is None:
		raise ValueError(f"No header found in {CSV_PATH}")

	changed = 0
	for row in rows:
		if not row["name"].strip() and row["alternative_name"].strip():
			row["name"] = row["alternative_name"]
			row["alternative_name"] = ""
			changed += 1

	with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
		writer = csv.DictWriter(fh, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Updated {changed} row(s) in {CSV_PATH}")


if __name__ == "__main__":
	main()
