#!/usr/bin/env python3
"""Dump the glTF structure of a .glb so we can compare server vs tuned exports.

Usage: python tmp/inspect_glb.py path/to/file.glb
"""
import json
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    data = f.read()

magic, ver, length = struct.unpack("<III", data[:12])
off = 12
clen, ctype = struct.unpack("<II", data[off:off + 8])
off += 8
j = json.loads(data[off:off + clen])

print(f"== {path} ==")
print("generator:", j.get("asset", {}).get("generator"))
print("\nMESHES:")
for m in j.get("meshes", []):
    for pr in m["primitives"]:
        print("  attrs:", list(pr.get("attributes", {}).keys()),
              "material:", pr.get("material"), "mode:", pr.get("mode", 4))

print("\nCOLOR_0 / relevant ACCESSORS:")
for i, a in enumerate(j.get("accessors", [])):
    print(f"  {i}: {a.get('type')} comp={a.get('componentType')} "
          f"normalized={a.get('normalized')} count={a.get('count')}")

print("\nMATERIALS:")
if not j.get("materials"):
    print("  (none)")
for m in j.get("materials", []):
    print("  ", json.dumps(m))

print("\nIMAGES:", len(j.get("images", [])),
      "TEXTURES:", len(j.get("textures", [])))
