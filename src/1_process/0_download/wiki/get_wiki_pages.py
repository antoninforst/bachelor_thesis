from __future__ import annotations

import argparse
import csv
import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm


API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE_URL = "https://en.wikipedia.org/wiki/{title}"
USER_AGENT = "morphological-complexity-thesis/1.0 (Wikipedia language page fetcher)"
MAX_RETRIES = 5

TITLE_OVERRIDES = {
    "belorussian": "Belarusian language",
    "chichewa": "Chewa language",
    "dhivehi": "Maldivian language",
    "gaelic": "Scottish Gaelic",
    "gaelic scots": "Scottish Gaelic",
    "german thurgau": "Swiss German",
    "german upper austrian": "Austrian German",
    "greek modern": "Greek language",
    "hebrew modern ashkenazic": "Hebrew language",
    "italian fiorentino": "Italian language",
    "italian turinese": "Piedmontese language",
    "karanga": "Shona language",
    "kirghiz fu yu": "Fuyu Kyrgyz language",
    "luxemburgeois": "Luxembourgish",
    "malay": "Malay language",
    "moldavian": "Moldovan language",
    "norwegian bokmal": "Bokmal",
    "norwegian nynorsk": "Nynorsk",
    "oriya": "Odia language",
    "panjabi": "Punjabi language",
    "provençal": "Occitan language",
    "saami northern": "Northern Sami",
    "serbian-croatian": "Serbo-Croatian",
    "sorbian upper": "Upper Sorbian language",
    "sotho northern": "Northern Sotho language",
    "swedish västerbotten": "Swedish language",
    "tamil spoken": "Tamil language",
    "volapuek": "Volapuk",
    "welsh colloquial": "Welsh language",
    "zulu southern": "Zulu language",
}


def request_json(params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return yaml.safe_load(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(delay)

    raise RuntimeError("unreachable retry state")


def normalize_key(name: str) -> str:
    clean = re.sub(r"[()]+", " ", name.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def base_language_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", name).strip()


def title_candidates(full_name: str) -> list[str]:
    key = normalize_key(full_name)
    base_name = base_language_name(full_name)
    candidates = []

    if key in TITLE_OVERRIDES:
        candidates.append(TITLE_OVERRIDES[key])

    candidates.extend(
        [
            f"{base_name} language",
            f"{full_name} language",
            base_name,
            full_name,
        ]
    )

    unique = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized.lower() not in seen:
            unique.append(normalized)
            seen.add(normalized.lower())
    return unique


def title_from_hint(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        title = title_from_wiki_url(value)
        return title or value
    return value.replace("_", " ")


def load_page_hints(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}

    hints: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return hints
        columns = {field.lower().strip(): field for field in reader.fieldnames}

        code_col = columns.get("code") or columns.get("iso_code")
        page_cols = [
            columns[column]
            for column in ("wiki_page", "page", "title", "wiki_url", "url", "name")
            if column in columns
        ]
        if code_col is None or not page_cols:
            return hints

        for row in reader:
            code = row.get(code_col, "").strip()
            if not code:
                continue
            for page_col in page_cols:
                title = title_from_hint(row.get(page_col, ""))
                if title:
                    hints.setdefault(code, [])
                    append_tried(hints[code], title)
    return hints


def page_url_from_title(title: str) -> str:
    return WIKI_PAGE_URL.format(title=urllib.parse.quote(title.replace(" ", "_")))


def revision_wikitext(page: dict[str, Any]) -> str:
    revisions = page.get("revisions", [])
    if not revisions:
        return ""
    slots = revisions[0].get("slots", {})
    main_slot = slots.get("main", {})
    return main_slot.get("content", "")


def fetch_page(title: str) -> dict[str, str] | None:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "extracts|info|revisions",
        "explaintext": "1",
        "inprop": "url",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
    }
    data = request_json(params)
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing") or not page.get("extract"):
        return None

    wikitext = revision_wikitext(page)

    return {
        "text": page["extract"],
        "wiki_url": page.get("fullurl", page_url_from_title(title)),
        "wikitext": wikitext,
    }


def title_from_wiki_url(wiki_url: str) -> str:
    parsed = urllib.parse.urlparse(wiki_url)
    if "/wiki/" not in parsed.path:
        return ""
    return urllib.parse.unquote(parsed.path.rsplit("/wiki/", 1)[1]).replace("_", " ")


def fetch_page_info_from_url(wiki_url: str) -> dict[str, str] | None:
    title = title_from_wiki_url(wiki_url)
    if not title:
        return None
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "info|revisions",
        "inprop": "url",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
    }
    data = request_json(params)
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing"):
        return None

    return {
        "text": "",
        "wiki_url": page.get("fullurl", wiki_url),
        "wikitext": revision_wikitext(page),
    }


def search_title(full_name: str) -> str | None:
    query = f"{base_language_name(full_name)} language"
    data = request_json(
        {
            "action": "opensearch",
            "format": "json",
            "search": query,
            "namespace": "0",
            "limit": "5",
        }
    )
    if len(data) < 2:
        return None

    for title in data[1]:
        if "language" in title.lower() or title.lower() == base_language_name(full_name).lower():
            return title
    return None


def normalize_tried(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    return [item for item in items if item and " failed: " not in item and not item.startswith("failed:")]


def append_tried(tried: list[str], value: str) -> None:
    value = value.strip()
    if value and value not in tried:
        tried.append(value)


def fetch_best_page(full_name: str, tried: list[str], hints: list[str]) -> dict[str, str] | None:
    for title in hints:
        if title in tried:
            continue
        append_tried(tried, title)
        page = fetch_page(title)
        if page is not None:
            return page

    for title in title_candidates(full_name):
        if title in tried:
            continue
        append_tried(tried, title)
        page = fetch_page(title)
        if page is not None:
            return page

    search_query = f"search:{base_language_name(full_name)} language"
    if search_query in tried:
        return None
    append_tried(tried, search_query)
    found_title = search_title(full_name)
    if found_title is None:
        return None
    if found_title in tried:
        return None
    append_tried(tried, found_title)
    return fetch_page(found_title)


def safe_file_stem(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", code)


def load_languages(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return {row["code"]: row for row in data}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(rows, file, allow_unicode=True, sort_keys=False, width=1000)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_manifest_row(code: str, full_name: str) -> dict[str, str]:
    return {
        "code": code,
        "full_name": full_name,
        "wiki_url": "",
        "date_and_time": "",
        "tried": [],
    }


def write_general_info(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "code",
        "full_name",
        "wiki_url",
        "l1_speakers",
        "l2_speakers",
        "typological_category",
        "family",
        "wiki_family",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_wiki_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "code",
        "full_name",
        "wiki_page",
        "L1",
        "L2",
        "L1_year",
        "L2_year",
        "L1_text",
        "L2_text",
        "morphological_category",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def load_wiki_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row.get("code", "").strip(): row for row in csv.DictReader(file) if row.get("code", "").strip()}


def weak_wiki_count(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return True
    lowered = value.lower()
    if lowered in {"unknown", "?", "n/a", "none"}:
        return True
    if re.fullmatch(r"(?:about|around|approximately|over|more than|less than)?\s*(?:million|billion|thousand)\b.*", lowered):
        return True
    return False


def missing_wiki_info(row: dict[str, str], manifest_row: dict[str, str] | None) -> bool:
    if manifest_row is None or not manifest_row.get("wiki_url"):
        return True
    return weak_wiki_count(row.get("L1", "")) or weak_wiki_count(row.get("L2", ""))


def load_general_info(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row["code"]: row for row in csv.DictReader(file)}


def load_infobox_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return {row["code"]: row for row in rows}


def write_infobox_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(rows, file, allow_unicode=True, sort_keys=False, width=1000)


def extract_template(wikitext: str, template_name: str) -> str:
    match = re.search(r"\{\{\s*" + re.escape(template_name), wikitext, flags=re.IGNORECASE)
    if match is None:
        return ""

    depth = 0
    index = match.start()
    while index < len(wikitext) - 1:
        pair = wikitext[index : index + 2]
        if pair == "{{":
            depth += 1
            index += 2
            continue
        if pair == "}}":
            depth -= 1
            index += 2
            if depth == 0:
                return wikitext[match.start() : index]
            continue
        index += 1
    return ""


def split_template_fields(template: str) -> list[str]:
    fields = []
    current = []
    brace_depth = 0
    link_depth = 0
    index = 2

    while index < len(template) - 2:
        pair = template[index : index + 2]
        if pair == "{{":
            brace_depth += 1
            current.append(pair)
            index += 2
            continue
        if pair == "}}" and brace_depth > 0:
            brace_depth -= 1
            current.append(pair)
            index += 2
            continue
        if pair == "[[":
            link_depth += 1
            current.append(pair)
            index += 2
            continue
        if pair == "]]" and link_depth > 0:
            link_depth -= 1
            current.append(pair)
            index += 2
            continue
        if template[index] == "|" and brace_depth == 0 and link_depth == 0:
            fields.append("".join(current))
            current = []
        else:
            current.append(template[index])
        index += 1

    if current:
        fields.append("".join(current))
    return fields


def clean_wikitext(value: str) -> str:
    value = re.sub(r"<ref[^>/]*/>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<ref\b[^>]*>.*?</ref>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<ref\b[^>]*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<br\s*/?>", "; ", value, flags=re.IGNORECASE)
    value = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\{\{(?:ubl|plainlist|flatlist|hlist)\|", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\{\{[^{}|]+\|([^{}]+)\}\}", lambda match: " ".join(part.strip() for part in match.group(1).split("|") if part.strip()), value)
    value = re.sub(r"\{\{([^{}]+)\}\}", lambda match: " ".join(part.strip() for part in match.group(1).split("|") if part.strip()), value)
    value = value.replace("}}", "")
    value = re.sub(r"'''?", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ;,")


def parse_language_infobox(wikitext: str) -> dict[str, str]:
    template = extract_template(wikitext, "Infobox language")
    if not template:
        return {}

    values: dict[str, str] = {}
    for field in split_template_fields(template):
        if "=" not in field:
            continue
        key, raw_value = field.split("=", 1)
        key = key.strip().lower()
        cleaned = clean_wikitext(raw_value)
        if cleaned:
            values[key] = cleaned

    return values


def family_from_infobox(infobox: dict[str, str]) -> str:
    family_keys = sorted(
        [key for key in infobox if re.fullmatch(r"fam\d+", key)],
        key=lambda key: int(key[3:]),
    )
    return " > ".join(infobox[key] for key in family_keys)


def speaker_value(infobox: dict[str, str], *keys: str) -> str:
    for key in keys:
        if infobox.get(key):
            return infobox[key]
    return ""


def first_present(mapping: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = mapping.get(key, "")
        if value:
            return clean_wikitext(str(value))
    return ""


def strip_parenthesized_date(value: str) -> tuple[str, str]:
    if not value:
        return "", ""
    dates = re.findall(r"\(([^)]*\d{4}[^)]*)\)", value)
    cleaned = re.sub(r"\s*\([^)]*\d{4}[^)]*\)", "", value).strip(" ;,")
    return cleaned or value, "; ".join(dates)


def speaker_count_to_number(value: str) -> str:
    if not value:
        return ""
    cleaned = clean_wikitext(value).lower()
    if re.search(r"\b(no|none|unknown|written only)\b", cleaned):
        return value
    if "%" in cleaned and not re.search(r"\d[\d,.]*\s*(?:million|billion|thousand)\b|\d{1,3}(?:,\d{3})+", cleaned):
        return value

    scale_words = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
    number = r"\d[\d,.]*"
    quantity_pattern = rf"({number})(?:\s*[–-]\s*({number}))?\s*(thousand|million|billion)?"

    candidates: list[tuple[int, int, float, str]] = []
    for match in re.finditer(quantity_pattern, cleaned):
        start, end = match.span()
        before = cleaned[max(0, start - 40):start]
        after = cleaned[end:end + 60]
        raw_one, raw_two, scale_word = match.groups()
        context = before + after
        has_count_context = re.search(r"\b(speakers?|people|users?|pupils|population|spoken|speaks?|stating|reported|census|native|second[- ]language|first[- ]language)\b", context)
        has_count_qualifier = re.search(r"\b(over|about|around|nearly|more than|at least|approximately|approx\.?|~)\s*$", before)
        if not scale_word and not re.search(r",", raw_one) and not has_count_context:
            if not has_count_qualifier:
                continue
        if not scale_word and not re.search(r",", raw_one) and re.fullmatch(r"\d{4}", raw_one.replace(",", "")) and not re.search(r"\b(speakers?|people|users?|population)\b", after):
            if not has_count_qualifier:
                continue
        if scale_word and raw_two and re.fullmatch(r"20\d{2}", raw_one) and re.fullmatch(r"\d{2}|20\d{2}", raw_two):
            continue
        scale = scale_words.get(scale_word or "", 1)
        one = float(raw_one.replace(",", "")) * scale
        if raw_two:
            two_scale = scale if scale_word else 1
            two = float(raw_two.replace(",", "")) * two_scale
            count = (one + two) / 2
        else:
            count = one
        candidates.append((start, end, count, scale_word or ""))

    if not candidates:
        return value

    count = candidates[0][2]
    if count.is_integer():
        return str(int(count))
    return f"{count:.2f}".rstrip("0").rstrip(".")


def extract_l2_from_speakers(value: str) -> tuple[str, str]:
    if not value:
        return "", ""

    patterns = [
        r"([^.;]*\bL2\b[^.;]*)",
        r"([^.;]*\bsecond[- ]language\b[^.;]*)",
        r"([^.;]*\bnon-native\b[^.;]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return strip_parenthesized_date(match.group(1).strip())
    return "", ""


def likely_quantity_fragment(text: str, speaker_kind: str) -> str:
    number = r"(?:over|about|around|nearly|more than|at least|approximately|approx\.?|~)?\s*\d[\d,.]*(?:\s*[–-]\s*\d[\d,.]*)?\s*(?:million|billion|thousand)?"
    if speaker_kind == "generic":
        patterns = [
            rf"({number})\s+(?:\w+\s+){{0,5}}speakers?\b",
            rf"\bspoken by\s+({number})\s+(?:\w+\s+){{0,5}}(?:people|speakers?)\b",
            rf"\bthere (?:were|are)\s+({number})\s+(?:\w+\s+){{0,5}}speakers?\b",
        ]
    elif speaker_kind == "l1":
        patterns = [
            rf"({number})\s+(?:\w+\s+){{0,4}}(?:mother[- ]tongue|native|first[- ]language) speakers?\b",
            rf"({number})\s+people\s+spoke\s+[^.;,]{{0,80}}\bfirst language\b",
            rf"({number})[^.;]{{0,100}}\bfirst language\b",
            rf"\bspoken by\s+({number})\s+speakers?\b",
            rf"\bnative speakers?\b[^.;,]{{0,80}}?({number})",
            rf"\bfirst[- ]language speakers?\b[^.;,]{{0,80}}?({number})",
        ]
    else:
        patterns = [
            rf"({number})\s+(?:\w+\s+){{0,4}}(?:second[- ]language|L2|non-native) speakers?\b",
            rf"(?:another|and another|more than|over)?\s*({number})\s+as\s+(?:a\s+)?second language speakers?\b",
            rf"({number})\s+people\s+spoke\s+[^.;,]{{0,80}}\bsecond language\b",
            rf"({number})[^.;]{{0,100}}\bsecond language\b",
            rf"\b(?:second[- ]language|L2|non-native) speakers?\b[^.;,]{{0,80}}?({number})",
            rf"\banother\s+({number})\s+as\s+second language speakers?\b",
        ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def compact_speaker_value(value: str, speaker_kind: str = "") -> str:
    if not value:
        return ""
    value = clean_wikitext(value)
    quantity = likely_quantity_fragment(value, speaker_kind)
    if quantity:
        return quantity
    value = re.sub(r"\b(L1|L2|native speakers?|second[- ]language speakers?|speakers?)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(native speakers?|second[- ]language speakers?|L1 speakers?|L2 speakers?)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ;,:")
    return value


def low_information_speaker_value(value: str) -> bool:
    if not value:
        return True
    normalized = re.sub(r"\s+", " ", value).strip(" ;,:?-").lower()
    if not normalized:
        return True
    if normalized in {"million", "billion", "thousand", "l1", "l2", "native", "unknown"}:
        return True
    if not re.search(r"\d", normalized):
        return True
    count_like = re.search(r"\d[\d,.]*\s*(?:million|billion|thousand)\b|\d{1,3}(?:,\d{3})+|\b(?!20\d{2}\b)\d{4,}\b|~\s*\d|\d+\s*%", normalized)
    return count_like is None


def total_only_l2_value(value: str, phrase: str) -> bool:
    if not value:
        return True
    return "total" in value.lower() and not likely_quantity_fragment(phrase, "l2")


def remove_l2_clause(value: str) -> str:
    if not value:
        return ""
    parts = [part.strip() for part in re.split(r";", value) if part.strip()]
    kept = [part for part in parts if not re.search(r"\b(L2|second[- ]language|non-native)\b", part, flags=re.IGNORECASE)]
    return "; ".join(kept) if kept else value


def wiki_page_value(wiki_url: str) -> str:
    return title_from_wiki_url(wiki_url) or wiki_url


def read_saved_page_text(output_dir: Path, code: str) -> str:
    path = output_dir / f"{safe_file_stem(code)}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_speaker_sentence(text: str, patterns: list[str]) -> tuple[str, str]:
    if not text:
        return "", ""
    compact = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    for pattern in patterns:
        for sentence in sentences:
            if not re.search(pattern, sentence, flags=re.IGNORECASE):
                continue
            if not re.search(r"\d", sentence):
                continue
            cleaned = sentence.strip(" .;")
            date_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})(?:[–-](?:\d{2}|\d{4}))?\b", cleaned)
            date = date_match.group(0) if date_match else ""
            return cleaned, date
    return "", ""


def extract_speaker_from_text(text: str, patterns: list[str], speaker_kind: str) -> tuple[str, str, str]:
    if not text:
        return "", "", ""
    compact = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    for pattern in patterns:
        for sentence in sentences:
            if not re.search(pattern, sentence, flags=re.IGNORECASE) or not re.search(r"\d", sentence):
                continue
            phrase = sentence.strip(" .;")
            value = compact_speaker_value(phrase, speaker_kind)
            if low_information_speaker_value(value):
                continue
            date_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})(?:[–-](?:\d{2}|\d{4}))?\b", phrase)
            date = date_match.group(0) if date_match else ""
            return phrase, value, date
    return "", "", ""


def extract_generic_speaker_from_text(text: str) -> tuple[str, str, str]:
    return extract_speaker_from_text(
        text,
        [
            r"\bspeakers?\b",
            r"\bspoken by\b",
            r"\busers?\b",
        ],
        "generic",
    )


def build_wiki_csv_row(language_row: dict[str, str], raw_infobox: dict[str, Any], output_dir: Path) -> dict[str, str]:
    code = language_row["iso_code"].strip()
    infobox = raw_infobox.get("infobox", {}) or {}
    speakers = first_present(infobox, ["speakers", "native speakers", "native_speakers", "nat speakers"])
    l1_phrase = remove_l2_clause(speakers)
    l1 = compact_speaker_value(l1_phrase, "l1")
    date_l1 = first_present(infobox, ["date", "speaker_date", "speakers_date"])

    l2_phrase = first_present(infobox, ["speakers2", "l2 speakers", "second language speakers", "second_language"])
    l2 = l2_phrase
    date_l2 = first_present(infobox, ["date2", "speakers2_date", "l2_date"])
    if l2:
        l2, embedded_date = strip_parenthesized_date(l2)
        l2 = compact_speaker_value(l2, "l2")
        date_l2 = date_l2 or embedded_date
    else:
        l2, embedded_date = extract_l2_from_speakers(speakers)
        l2_phrase = l2
        l2 = compact_speaker_value(l2, "l2")
        date_l2 = date_l2 or embedded_date

    text = read_saved_page_text(output_dir, code)
    if low_information_speaker_value(l1):
        text_l1_phrase, text_l1, text_date = extract_speaker_from_text(
            text,
            [
                r"\bmother[- ]tongue speakers?\b",
                r"\bnative speakers?\b",
                r"\bfirst language\b",
                r"\bfirst language speakers?\b",
                r"\bfirst[- ]language speakers?\b",
                r"\bL1 speakers?\b",
            ],
            "l1",
        )
        if text_l1 and not low_information_speaker_value(text_l1):
            l1_phrase = text_l1_phrase
            l1 = text_l1
            date_l1 = date_l1 or text_date
    if low_information_speaker_value(l1):
        text_l1_phrase, text_l1, text_date = extract_generic_speaker_from_text(text)
        if re.search(r"\b(L2|second[- ]language|non-native)\b", text_l1_phrase, flags=re.IGNORECASE) and not re.search(r"\b(L1|first[- ]language|native|mother[- ]tongue)\b", text_l1_phrase, flags=re.IGNORECASE):
            text_l1_phrase, text_l1, text_date = "", "", ""
        if text_l1 and not low_information_speaker_value(text_l1):
            l1_phrase = text_l1_phrase
            l1 = text_l1
            date_l1 = date_l1 or text_date
    if low_information_speaker_value(l2) or total_only_l2_value(l2, l2_phrase):
        text_l2_phrase, text_l2, text_date = extract_speaker_from_text(
            text,
            [
                r"\bL2 speakers?\b",
                r"\bsecond[- ]language speakers?\b",
                r"\bsecond language\b",
                r"\bnon-native speakers?\b",
            ],
            "l2",
        )
        if text_l2 and not low_information_speaker_value(text_l2):
            l2_phrase = text_l2_phrase
            l2 = text_l2
            date_l2 = date_l2 or text_date

    return {
        "code": code,
        "full_name": language_row["name"].strip(),
        "wiki_page": wiki_page_value(raw_infobox.get("wiki_url", "")),
        "L1": speaker_count_to_number(l1),
        "L2": speaker_count_to_number(l2),
        "L1_year": date_l1,
        "L2_year": date_l2,
        "L1_text": l1_phrase,
        "L2_text": l2_phrase,
        "morphological_category": language_row.get("typology", ""),
    }


def existing_success(manifest_row: dict[str, str], page_path: Path) -> bool:
    return page_path.exists() and bool(manifest_row.get("wiki_url")) and bool(manifest_row.get("date_and_time"))


def select_languages(languages: list[dict[str, str]], codes: list[str] | None, limit: int | None) -> list[dict[str, str]]:
    selected = languages
    if codes:
        wanted = {code.lower() for code in codes}
        selected = [row for row in selected if row["iso_code"].strip().lower() in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def download_wikipedia_pages(
    input_csv: Path,
    output_dir: Path,
    delay: float,
    limit: int | None,
    codes: list[str] | None,
    add_csv: Path | None,
    missing_info_only: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_languages = load_languages(input_csv)
    manifest_path = output_dir / "manifest.yaml"
    infobox_path = output_dir / "infobox_raw.yaml"
    wiki_csv_path = output_dir / "wiki.csv"
    previous_manifest = load_manifest(manifest_path)
    previous_wiki = load_wiki_csv(wiki_csv_path)
    languages = select_languages(all_languages, codes, None)
    if missing_info_only:
        languages = [
            row
            for row in languages
            if missing_wiki_info(previous_wiki.get(row["iso_code"].strip(), {}), previous_manifest.get(row["iso_code"].strip()))
        ]
    if limit is not None:
        languages = languages[:limit]
    manifest_by_code: dict[str, dict[str, str]] = previous_manifest.copy()
    info_by_code = load_general_info(output_dir / "general_info.csv")
    infobox_by_code = load_infobox_rows(infobox_path)
    page_hints = load_page_hints(add_csv)
    wiki_by_code: dict[str, dict[str, str]] = {}
    successes = 0
    skipped = 0

    for row in tqdm(languages, desc="Fetching Wikipedia pages", unit="page"):
        code = row["iso_code"].strip()
        full_name = row["name"].strip()
        page_path = output_dir / f"{safe_file_stem(code)}.txt"
        existing_manifest_row = previous_manifest.get(code, {}).copy()
        manifest_row = existing_manifest_row.copy() or new_manifest_row(code, full_name)
        manifest_row["tried"] = normalize_tried(manifest_row.get("tried"))
        raw_infobox = infobox_by_code.get(code, {}).copy()
        page = None

        if existing_success(existing_manifest_row, page_path):
            skipped += 1
            if manifest_row.get("wiki_url") and not raw_infobox.get("infobox"):
                try:
                    page = fetch_page_info_from_url(manifest_row["wiki_url"])
                except (urllib.error.URLError, TimeoutError, ValueError) as error:
                    print(f"{code}: info failed ({error})")
        else:
            manifest_row["code"] = code
            manifest_row["full_name"] = full_name
            failure_reason = "page not found"
            try:
                page = fetch_best_page(full_name, manifest_row["tried"], page_hints.get(code, []))
            except (urllib.error.URLError, TimeoutError, ValueError) as error:
                failure_reason = str(error)
                print(f"{code}: failed ({failure_reason})")
                page = None

            if page is not None:
                page_path.write_text(page["text"], encoding="utf-8")
                manifest_row["wiki_url"] = page["wiki_url"]
                manifest_row["date_and_time"] = now_utc()
                successes += 1
            else:
                manifest_row["wiki_url"] = ""
                manifest_row["date_and_time"] = ""

        if page is not None:
            raw_infobox = {
                "code": code,
                "full_name": full_name,
                "wiki_url": manifest_row.get("wiki_url", page.get("wiki_url", "")),
                "infobox": parse_language_infobox(page["wikitext"]),
            }
        elif not raw_infobox:
            raw_infobox = {
                "code": code,
                "full_name": full_name,
                "wiki_url": manifest_row.get("wiki_url", ""),
                "infobox": {},
            }

        infobox = raw_infobox.get("infobox", {})

        previous_info = info_by_code.get(code, {})
        info_by_code[code] = {
            "code": code,
            "full_name": full_name,
            "wiki_url": manifest_row.get("wiki_url", ""),
            "l1_speakers": speaker_value(infobox, "speakers", "native speakers") or previous_info.get("l1_speakers", ""),
            "l2_speakers": speaker_value(infobox, "speakers2", "second language speakers", "l2 speakers") or previous_info.get("l2_speakers", ""),
            "typological_category": row.get("typology", ""),
            "family": row.get("family", ""),
            "wiki_family": family_from_infobox(infobox) or previous_info.get("wiki_family", ""),
        }
        infobox_by_code[code] = raw_infobox
        wiki_by_code[code] = build_wiki_csv_row(row, raw_infobox, output_dir)
        manifest_by_code[code] = manifest_row
        ordered_manifest = [manifest_by_code[row["iso_code"].strip()] for row in all_languages if row["iso_code"].strip() in manifest_by_code]
        write_manifest(manifest_path, ordered_manifest)
        ordered_infobox = [infobox_by_code[row["iso_code"].strip()] for row in all_languages if row["iso_code"].strip() in infobox_by_code]
        write_infobox_rows(infobox_path, ordered_infobox)
        if delay > 0:
            time.sleep(delay)

    ordered_info = []
    ordered_wiki = []
    for row in all_languages:
        code = row["iso_code"].strip()
        if code in info_by_code:
            ordered_info.append(info_by_code[code])
        raw_infobox = infobox_by_code.get(code)
        if raw_infobox is not None:
            ordered_wiki.append(wiki_by_code.get(code) or build_wiki_csv_row(row, raw_infobox, output_dir))
    write_general_info(output_dir / "general_info.csv", ordered_info)
    write_wiki_csv(wiki_csv_path, ordered_wiki)
    print(f"Downloaded {successes} new pages, skipped {skipped} existing pages, processed {len(languages)} rows into {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download plain-text English Wikipedia language pages.")
    parser.add_argument("--input", type=Path, default=Path("data/5_other/lang_families_subset.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/5_other/code_wiki"))
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between Wikipedia requests in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Fetch only the first N selected language rows.")
    parser.add_argument("--codes", nargs="+", default=None, help="Fetch only these ISO codes, for example: --codes eng deu ell")
    parser.add_argument("--missing-info", action="store_true", help="Process only rows whose wiki.csv L1/L2 or page metadata is missing or weak.")
    parser.add_argument("--add", type=Path, default=None, help="Optional add.csv with manual page hints. Columns: code plus wiki_page/page/title/wiki_url/url/name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_add_csv = args.output / "add.csv"
    fallback_add_csv = args.input.parent / "add.csv"
    add_csv = args.add or (default_add_csv if default_add_csv.exists() else fallback_add_csv)
    download_wikipedia_pages(args.input, args.output, args.delay, args.limit, args.codes, add_csv, args.missing_info)