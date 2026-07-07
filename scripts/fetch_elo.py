#!/usr/bin/env python3
"""Regenerate data/elo.csv from eloratings.net's TSV API.

Usage: python scripts/fetch_elo.py [OUT_CSV]

Fetches the current world Elo list and team-name mapping, and writes a
``team,code,elo`` CSV. eloratings.net data is used with attribution; see their
site for terms. The shipped snapshot is dated in the README.
"""
import csv
import sys
import urllib.request

WORLD = "https://eloratings.net/World.tsv"
TEAMS = "https://eloratings.net/en.teams.tsv"


def _tsv(url):
    with urllib.request.urlopen(url) as r:
        return [line.split("\t") for line in r.read().decode("utf-8").splitlines()]


def main(out_path="data/elo.csv"):
    names = {row[0]: row[1] for row in _tsv(TEAMS) if len(row) >= 2}
    rows = []
    for row in _tsv(WORLD):
        # World.tsv: rank, code, elo, ... (see eloratings.net)
        if len(row) >= 3 and row[1] in names:
            try:
                rows.append({"team": names[row[1]], "code": row[1], "elo": int(row[2])})
            except ValueError:
                continue
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["team", "code", "elo"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} teams -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
