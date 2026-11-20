"""Download the public-domain medical texts (pre-1929, scanned books on archive.org) into data/raw/.
Run: python scripts/download_data.py   (about 12 MB; the cleaned corpus is already committed in data/)"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {  # file name -> archive.org identifier
    "gray_anatomy_1918": "anatomyofhumanbo1918gray",                      # Gray, Anatomy of the Human Body (1918)
    "goulds_students_dictionary_1900": "studentsmedical00goul",           # Gould, The Student's Medical Dictionary (1900)
    "howell_physiology_1921": "textbookofphysio00howe",                   # Howell, A Text-book of Physiology (1921)
    "lewis_nurses_anatomy_physiology_1913": "anatomyphysiolog00lewirich",  # Lewis, Anatomy and Physiology for Nurses (1913)
}


def main() -> None:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, ident in SOURCES.items():
        path = raw / f"{name}.txt"
        if path.exists():
            print(f"have {path.name}")
            continue
        url = f"https://archive.org/download/{ident}/{ident}_djvu.txt"
        print(f"fetching {url}")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r:
            path.write_bytes(r.read())
        print(f"  {path.name}: {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    sys.exit(main())
