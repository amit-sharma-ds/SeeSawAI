"""Create an API key for a developer, or list the existing ones.

  py make_api_key.py --label frontend-team
  py make_api_key.py --list

Keys are stored in api_keys.txt next to this file (ignored by git). The running server
picks up new keys immediately. Give each developer their own key so one can be removed
later by deleting its line.
"""
from __future__ import annotations

import argparse
import sys

from cctv_ai.auth import KEY_FILE, add_key, load_keys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="", help="who the key is for, e.g. frontend-team")
    parser.add_argument("--list", action="store_true", help="show existing keys (masked) and their labels")
    args = parser.parse_args(argv)

    if args.list:
        keys = load_keys()
        if not keys:
            print("no API keys configured: the API is open")
            return 0
        for key, label in keys.items():
            print(f"{key[:6]}...{key[-4:]}  {label or '(no label)'}")
        print(f"{len(keys)} key(s), file: {KEY_FILE}")
        return 0

    key = add_key(args.label)
    print(key)
    print(f"saved to {KEY_FILE}. Send it privately. Use it as the X-API-Key header or ?api_key= on stream URLs.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
