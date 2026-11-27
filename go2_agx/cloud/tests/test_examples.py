#!/usr/bin/env python3
"""Every example is a full message: it must match the envelope schema and the schema of its payload kind,
every kind has an example, and every example stays under the size budget."""
import json
import pathlib
import sys

from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_BYTES = 4096


def errors(schema, doc):
    Draft202012Validator.check_schema(schema)
    return [e.message for e in Draft202012Validator(schema).iter_errors(doc)]


def main():
    failed = 0
    envelope_schema = json.loads((ROOT / "schemas" / "envelope.schema.json").read_text())
    kinds = sorted(p.name.replace(".schema.json", "") for p in (ROOT / "schemas").glob("odin.*.schema.json"))
    for kind in kinds:
        example = ROOT / "examples" / (kind + ".json")
        if not example.exists():
            print("FAIL  %s: no example" % kind)
            failed += 1
            continue
        raw = example.read_text()
        msg = json.loads(raw)
        errs = errors(envelope_schema, msg)
        if not errs:
            if msg["payload"]["kind"] != kind:
                errs = ["payload.kind is %s" % msg["payload"]["kind"]]
            else:
                errs = errors(json.loads((ROOT / "schemas" / (kind + ".schema.json")).read_text()), msg["payload"])
        size = len(raw.encode())
        if errs:
            print("FAIL  %s: %s" % (kind, errs))
            failed += 1
        elif size > MAX_BYTES:
            print("FAIL  %s: %d bytes > %d" % (kind, size, MAX_BYTES))
            failed += 1
        else:
            print("PASS  %s (%d bytes)" % (kind, size))
    for example in sorted((ROOT / "examples").glob("*.json")):
        if example.stem not in kinds:
            print("FAIL  %s: example without schema" % example.name)
            failed += 1
    print("== %d kinds, %d failed" % (len(kinds), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
