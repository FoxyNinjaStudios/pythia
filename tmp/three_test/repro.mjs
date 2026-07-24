import fs from "fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";

// minimal FileReader polyfill for GLTFExporter binary path
globalThis.FileReader = class {
  readAsArrayBuffer(blob) {
    blob.arrayBuffer().then((buf) => {
      this.result = buf;
      this.onload && this.onload({ target: this });
      this.onloadend && this.onloadend({ target: this });
    });
  }
};

const SRC = process.argv[2];
const buf = fs.readFileSync(SRC);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

const loader = new GLTFLoader();
loader.parse(ab, "", (gltf) => {
  let mesh = null;
  gltf.scene.traverse((o) => { if (!mesh && o.isMesh) mesh = o; });
  const geo = mesh.geometry;
  const col = geo.getAttribute("color");
  console.log("SRC color itemSize:", col && col.itemSize, "normalized:", col && col.normalized,
    "array:", col && col.array.constructor.name);

  // Build ubyte VEC4 color like the fix
  const pos = geo.getAttribute("position");
  const nv = pos.count;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos.array), 3));
  if (col) {
    const bytes = new Uint8Array(nv * 4);
    for (let i = 0; i < nv; i++) {
      bytes[4*i]   = Math.round(Math.min(1, Math.max(0, col.getX(i))) * 255);
      bytes[4*i+1] = Math.round(Math.min(1, Math.max(0, col.getY(i))) * 255);
      bytes[4*i+2] = Math.round(Math.min(1, Math.max(0, col.getZ(i))) * 255);
      bytes[4*i+3] = 255;
    }
    g.setAttribute("color", new THREE.BufferAttribute(bytes, 4, true));
  }
  g.setIndex(geo.getIndex());

  const mat = mesh.material.clone();
  mat.vertexColors = true;
  mat.side = THREE.FrontSide;
  mat.metalness = 0;
  mat.roughness = 1;
  mat.transparent = false;
  mat.opacity = 1;
  const out = new THREE.Mesh(g, mat);
  console.log("mesh material type:", mat.type, "metalness:", mat.metalness, "transparent:", mat.transparent, "opacity:", mat.opacity);

  new GLTFExporter().parse(out, (glb) => {
    fs.writeFileSync("out_diffuse.glb", Buffer.from(glb));
    console.log("wrote out_diffuse.glb", glb.byteLength);
  }, (e) => console.error(e), { binary: true });
}, (e) => console.error("load err", e));
