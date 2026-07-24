import fs from "fs";

function stripMaterials(glbBuffer) {
  const dv = new DataView(glbBuffer);
  const JSON_TYPE = 0x4e4f534a, BIN_TYPE = 0x004e4942;
  let offset = 12, jsonText = null, binChunk = null;
  while (offset < dv.byteLength) {
    const chunkLen = dv.getUint32(offset, true);
    const chunkType = dv.getUint32(offset + 4, true);
    const dataStart = offset + 8;
    if (chunkType === JSON_TYPE) jsonText = new TextDecoder().decode(new Uint8Array(glbBuffer, dataStart, chunkLen));
    else if (chunkType === BIN_TYPE) binChunk = new Uint8Array(glbBuffer, dataStart, chunkLen);
    offset = dataStart + chunkLen;
  }
  const json = JSON.parse(jsonText);
  for (const m of json.meshes || []) for (const p of m.primitives || []) delete p.material;
  delete json.materials;
  let newJson = new TextEncoder().encode(JSON.stringify(json));
  const pad = (4 - (newJson.length % 4)) % 4;
  if (pad) { const p2 = new Uint8Array(newJson.length + pad); p2.set(newJson); for (let i=0;i<pad;i++) p2[newJson.length+i]=0x20; newJson=p2; }
  const binPadded = binChunk ? binChunk.length + ((4 - (binChunk.length % 4)) % 4) : 0;
  const total = 12 + 8 + newJson.length + (binChunk ? 8 + binPadded : 0);
  const out = new ArrayBuffer(total);
  const odv = new DataView(out), ob = new Uint8Array(out);
  odv.setUint32(0, 0x46546c67, true); odv.setUint32(4, 2, true); odv.setUint32(8, total, true);
  let o = 12;
  odv.setUint32(o, newJson.length, true); odv.setUint32(o+4, JSON_TYPE, true); ob.set(newJson, o+8); o += 8 + newJson.length;
  if (binChunk) { odv.setUint32(o, binPadded, true); odv.setUint32(o+4, BIN_TYPE, true); ob.set(binChunk, o+8); }
  return out;
}

const buf = fs.readFileSync("out_withmat.glb");
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const out = stripMaterials(ab);
fs.writeFileSync("out_stripped.glb", Buffer.from(out));
console.log("wrote out_stripped.glb", out.byteLength);
