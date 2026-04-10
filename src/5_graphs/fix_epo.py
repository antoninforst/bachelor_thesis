import argparse
import csv
from collections import OrderedDict


def load_csv(path):
    words = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            word, freq = row[0], int(row[1])
            words[word] = words.get(word, 0) + freq
    return words


def save_csv(path, words):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in words.items():
            writer.writerow([word, freq])


def get_non_ascii_freq(words):
    chars = {}
    for word, freq in words.items():
        for ch in word:
            if ord(ch) > 127:
                chars[ch] = chars.get(ch, 0) + freq
    return chars


def main():
    parser = argparse.ArgumentParser(description="Fix encoding in a word frequency CSV.")
    parser.add_argument("path", nargs="?", default="epo.csv", help="Path to the CSV file")
    parser.add_argument("-n", "--min-freq", type=int, default=0, help="Hide characters with frequency below this")
    args = parser.parse_args()

    path = args.path
    min_freq = args.min_freq

    words = load_csv(path)
    print(f"Loaded {len(words)} words from {path}")

    while True:
        char_freq = get_non_ascii_freq(words)
        filtered = {ch: f for ch, f in char_freq.items() if f >= min_freq}
        by_freq = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        print(f"\nNon-ASCII characters (>= {min_freq} occurrences, {len(by_freq)} shown):")
        for ch, f in by_freq:
            print(f"  {ch}  {f}")

        char = input("\nEnter character to inspect (. to quit): ")
        if char == ".":
            break
        if len(char) != 1:
            print("Please enter exactly one character.")
            continue

        # Collect words containing this character
        matching = [(w, f) for w, f in words.items() if char in w]
        if not matching:
            print(f"No words contain '{char}'.")
            continue

        print(f"\nWords containing '{char}' ({len(matching)} total):")
        offset = 0
        while True:
            batch = matching[offset:offset + 20]
            for w, f in batch:
                print(f"  {w} ({f})")
            offset += 20

            if offset >= len(matching):
                print("(no more words)")

            action = input(f"\nReplace '{char}' with (;=more, .=back): ")
            if action == ".":
                break
            if action == ";":
                if offset >= len(matching):
                    print("No more words to show.")
                continue

            # Replace char with action
            new_char = action
            changed = []
            keys_to_process = [(w, f) for w, f in matching if char in w]
            for old_word, old_freq in keys_to_process:
                new_word = old_word.replace(char, new_char)
                if new_word == old_word:
                    continue

                del words[old_word]

                if new_word in words:
                    old_target_freq = words[new_word]
                    new_freq = old_target_freq + old_freq
                    words[new_word] = new_freq
                    if old_freq > 10 and old_target_freq > 10:
                        print(f"  {old_word} ({old_freq}) -> {new_word} ({old_target_freq} -> {new_freq})")
                else:
                    words[new_word] = old_freq
                changed.append(new_word)

            print(f"\nChanged {len(changed)} words.")
            save_csv(path, words)
            print(f"Saved to {path}.")
            break


if __name__ == "__main__":
    main()
