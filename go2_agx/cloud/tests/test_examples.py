#!/usr/bin/env python3
"""Every example validates against its schema, and stays under the size budget."""
import json
import pathlib
import sys

from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_BYTES = 4096


def main():
    failed = 0
    schemas = sorted((ROOT / "schemas").glob("*.schema.json"))
    for schema_path in schemas:
        kind = schema_path.name.replace(".schema.json", "")
        example = ROOT / "examples" / (kind + ".json")
        if not example.exists():
            print("FAIL  %s: no example" % kind)
            failed += 1
            continue
        schema = json.loads(schema_path.read_text())
        Draft202012Validator.check_schema(schema)
        raw = example.read_text()
        doc = json.loads(raw)
        errors = list(Draft202012Validator(schema).iter_errors(doc))
        size = len(raw.encode())
        if errors:
            print("FAIL  %s" % kind)
            for e in errors:
                print("      %s" % e.message)
            failed += 1
        elif size > MAX_BYTES:
            print("FAIL  %s: %d bytes > %d" % (kind, size, MAX_BYTES))
            failed += 1
        else:
            print("PASS  %s (%d bytes)" % (kind, size))
    for example in sorted((ROOT / "examples").glob("*.json")):
        if not (ROOT / "schemas" / (example.stem + ".schema.json")).exists():
            print("FAIL  %s: example without schema" % example.name)
            failed += 1
    print("== %d schemas, %d failed" % (len(schemas), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
