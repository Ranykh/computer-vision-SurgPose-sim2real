


import blenderproc as bproc  
bproc.init()

import bpy 
# Phase-1 renderer (COCO masks+boxes now; keypoints will be added in a second pass)
import argparse, glob, json, math, random, shutil
from pathlib import Path
import numpy as np
import traceback
import os
from itertools import chain

os.environ["MPLBACKEND"] = "Agg"  # avoid Jupyter/GUI backend inside Blender


# ---------------------- CLI ----------------------
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True, help="Output dir (images + coco_annotations.json)")
ap.add_argument("--max_images", type=int, default=20, help="Total images to generate")
ap.add_argument("--per_obj_cap", type=int, default=5, help="Max frames per single OBJ (avoid dominance)")
ap.add_argument("--camera_json", default="/datashare/project/camera.json")
ap.add_argument("--model_dir_needle", default="/datashare/project/surgical_tools_models/needle_holder")
ap.add_argument("--model_dir_tweezers", default="/datashare/project/surgical_tools_models/tweezers")
ap.add_argument("--hdr_dir", default="/datashare/project/haven/hdris", help="Haven HDRIs root")
ap.add_argument("--bg_dir",  default="/datashare/project/train2017", help="2D background images (COCO train2017)")
ap.add_argument("--mix",     default="studio=0.34,hdri=0.33,composite=0.33",
                help="Strategy mix, comma list. Keys in {studio,hdri,composite}. e.g. studio=0.5,hdri=0.2,composite=0.3")
ap.add_argument("--seed",    type=int, default=123)
# Fast/quality knobs (also overridable via env)
ap.add_argument("--samples", type=int, default=int(os.environ.get("BPROC_SAMPLES", "32")))
ap.add_argument("--res_scale", type=float, default=float(os.environ.get("BPROC_RES_SCALE", "1.0")))
ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
ap.add_argument("--kps_dir", default="/home/student/surg-pose-sim2real/assets/kps",
                help="Optional root for *_kps.json sidecars; searched by stem under subfolders")

args, _ = ap.parse_known_args()

try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    if args.device == "gpu":
        prefs.compute_device_type = 'OPTIX'  # will fall back to CUDA if needed
        for d in prefs.devices: d.use = True
        bpy.context.scene.cycles.device = 'GPU'
        print("[init] Cycles device = GPU (OPTIX/CUDA)")
    else:
        bpy.context.scene.cycles.device = 'CPU'
        print("[init] Cycles device = CPU")
except Exception as e:
    print("[init] device set failed:", e)

random.seed(args.seed); np.random.seed(args.seed)

KPS_DIR = Path(args.kps_dir) if args.kps_dir else None

OUT_DIR = Path(args.outdir); (OUT_DIR/"images").mkdir(parents=True, exist_ok=True)
COCO_JSON = OUT_DIR/"coco_annotations.json"


# ---------------------- Camera intrinsics ----------------------
with open(args.camera_json, "r") as f:
    cam = json.load(f)
# target render size
im_w = int(cam["width"] * args.res_scale)
im_h = int(cam["height"] * args.res_scale)

# scale intrinsics for the resized render
_scale = im_w / float(cam["width"])  # == args.res_scale if width/height are consistent
base_fx = cam["fx"] * _scale
base_fy = cam["fy"] * _scale
cx0     = cam["cx"] * _scale
cy0     = cam["cy"] * _scale
# ---------------------- Assets ----------------------
def list_objs(root):
    return sorted(glob.glob(str(Path(root)/"*.obj")))

objs_needle   = list_objs(args.model_dir_needle)
objs_tweezers = list_objs(args.model_dir_tweezers)

assert len(objs_needle) + len(objs_tweezers) > 0, "No OBJ files found."
print(f"[assets] needle_holder={len(objs_needle)} | tweezers={len(objs_tweezers)}")
assert len(objs_needle) > 0, f"No OBJs in {args.model_dir_needle}"
assert len(objs_tweezers) > 0, f"No OBJs in {args.model_dir_tweezers}"


# treat different OBJ files as different articulation states too
ALL_OBJS = [(p, 1) for p in objs_needle] + [(p, 2) for p in objs_tweezers]
random.shuffle(ALL_OBJS)

# HDRIs: accept *_2k.hdr / *.hdr / *.exr inside each subdir
HDRI_LIST = []
hdr_root = Path(args.hdr_dir)
if hdr_root.exists():
    for sub in hdr_root.iterdir():
        if sub.is_dir():
            cands = list(sub.glob("*_2k.hdr")) + list(sub.glob("*.hdr")) + list(sub.glob("*.exr"))
            HDRI_LIST.extend(str(p) for p in cands)

random.shuffle(HDRI_LIST)
print(f"[assets] HDRIs: {len(HDRI_LIST)}")

# Backgrounds (2D) for compositing
BG_LIST = sorted([p for p in Path(args.bg_dir).glob("*.*") if p.suffix.lower() in (".jpg",".jpeg",".png")])
print(f"[assets] 2D backgrounds: {len(BG_LIST)}")

# ---------------------- Strategy mix ----------------------
def parse_mix(s):
    parts = [kv.strip() for kv in s.split(",") if kv.strip()]
    mix = {}
    for p in parts:
        if "=" in p:
            k,v = p.split("=",1)
            mix[k.strip()] = max(0.0, float(v))
    # default to equal if invalid
    if not mix: mix = {"studio": 1.0}
    # normalize
    tot = sum(mix.values()) or 1.0
    for k in mix: mix[k] /= tot
    return mix

MIX = parse_mix(args.mix)
###########################################
VALID_STRATS = {"studio","hdri","composite"}
###########################################
for k in list(MIX.keys()):
    if k not in VALID_STRATS:
        print(f"[mix] dropping unknown strategy '{k}'"); MIX.pop(k)
if not MIX: MIX = {"studio":1.0}
print("[mix]", MIX)

def sample_strategy():
    r = random.random()
    acc = 0.0
    for k,v in MIX.items():
        acc += v
        if r <= acc: return k
    return list(MIX.keys())[-1]

# ---------------------- Helpers ----------------------

# ─────────────────────────────────────────────────────────────────────────────
# Keypoints: schema + helpers
# ─────────────────────────────────────────────────────────────────────────────
KP_SCHEMA   = ["tip", "jaw_L", "jaw_R", "hinge_base", "handle_end"]
KP_SKELETON = [[1,4],[2,4],[3,4],[4,5]]  # COCO 1-based
AUTO_FALLBACK_IF_MISSING_JSON = True      # set False to force sidecar JSON presence

def _obj_sidecar_kp_path(obj_path: str) -> Path:
    return Path(obj_path).with_suffix("").with_name(Path(obj_path).stem + "_kps.json")

def load_kps_3d_for_obj(obj_path: str):

    # Try to load <stem>_kps.json from --kps_dir (recursive) or next to OBJ.
    # Return (K,4) array or None if not found.

    stem = Path(obj_path).stem
    candidates = []
    if KPS_DIR and KPS_DIR.exists():
        candidates += list(KPS_DIR.rglob(stem + "_kps.json"))
    candidates.append(Path(obj_path).with_suffix("").with_name(stem + "_kps.json"))

    for jp in candidates:
        if jp.exists():
            d = json.load(open(jp, "r"))
            pts = d.get("points", {})
            coords = []
            for name in KP_SCHEMA:
                x, y, z = map(float, pts[name])
                coords.append(np.array([x, y, z, 1.0], dtype=float))
            return np.stack(coords, axis=0)

    return None  # <- let the caller decide the fallback


def auto_kp_local_from_mesh(obj):

    # Bootstrap landmarks if no JSON:
    #   tip ~ max X, handle_end ~ min X, jaw_L ~ max Y near tip, jaw_R ~ min Y near tip,
    #   hinge_base ~ mesh origin (0,0,0) in local coords.

    V = np.array([list(v.co) for v in obj.blender_obj.data.vertices], dtype=float)
    if V.size == 0:
        Z = np.zeros((5,3), dtype=float)
        return np.hstack([Z, np.ones((5,1))])  # (5,4)
    idx_max_x, idx_min_x = V[:,0].argmax(), V[:,0].argmin()
    tip        = V[idx_max_x]
    handle_end = V[idx_min_x]
    # pick jaw_L/jaw_R as Y-extremes among the top 10% X vertices (near tip)
    thresh = np.percentile(V[:,0], 90)
    near_tip = V[V[:,0] >= thresh]
    if len(near_tip) >= 2:
        jaw_L = near_tip[near_tip[:,1].argmax()]
        jaw_R = near_tip[near_tip[:,1].argmin()]
    else:
        jaw_L = V[V[:,1].argmax()]
        jaw_R = V[V[:,1].argmin()]
    hinge_base = np.array([0.0, 0.0, 0.0])
    P = np.stack([tip, jaw_L, jaw_R, hinge_base, handle_end], axis=0)
    ones = np.ones((P.shape[0],1), dtype=float)
    return np.hstack([P, ones])  # (5,4)

def project_keypoints_local_to_pixels(kps_local_hom, T_obj2world, cam2world, fx, fy, cx, cy):
    Twc = np.asarray(cam2world)       # cam->world (BlenderProc uses camera looking along -Z)
    Tcw = np.linalg.inv(Twc)          # world->cam
    Tow = np.asarray(T_obj2world)     # obj->world

    Xw = Tow @ kps_local_hom.T        # 4xK
    Xc = (Tcw @ Xw).T                  # Kx4  (camera coords)

    z = -Xc[:, 2]                      # positive depth in front of camera (keep this)
    eps = 1e-6
    z = np.where(z > eps, z, eps)      # avoid divide-by-zero

    # NOTE: image v increases DOWN, while camera +Y is UP -> need the minus sign
    u = fx * (Xc[:, 0] / z) + cx
    v = -fy * (Xc[:, 1] / z) + cy

    return np.stack([u, v], axis=1), z


#########################################################

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (local; used only by vis_flags_from_depth_and_mask)
# ─────────────────────────────────────────────────────────────────────────────
def _bilinear_at(arr, u, v):
    if arr is None:
        return 0.0
    H, W = arr.shape[:2]
    if not (np.isfinite(u) and np.isfinite(v)):
        return 0.0
    # treat edges as in-bounds (inclusive)
    if u <= 0: u = 0.0
    if v <= 0: v = 0.0
    if u >= W - 1: u = W - 1.0
    if v >= H - 1: v = H - 1.0
    u0, v0 = np.floor(u), np.floor(v)
    u1, v1 = u0 + 1.0, v0 + 1.0
    ui0, vi0 = int(u0), int(v0)
    ui1, vi1 = min(int(u1), W - 1), min(int(v1), H - 1)
    du, dv = u - u0, v - v0
    Ia = float(arr[vi0, ui0]); Ib = float(arr[vi0, ui1])
    Ic = float(arr[vi1, ui0]); Id = float(arr[vi1, ui1])
    return (Ia * (1 - du) * (1 - dv) +
            Ib * (du) * (1 - dv) +
            Ic * (1 - du) * (dv) +
            Id * (du) * (dv))

def _dilate_bounds(u, v, px, W, H):
    x0 = int(max(0, np.floor(u - px)))
    x1 = int(min(W - 1, np.ceil(u + px)))
    y0 = int(max(0, np.floor(v - px)))
    y1 = int(min(H - 1, np.ceil(v + px)))
    return x0, x1, y0, y1

def _decode_coco_rle(rle):
    # ---
    # Minimal pure-Python decoder for COCO RLE when pycocotools isn't available.
    # Supports the common uncompressed form where rle['counts'] is a list of ints.
    # ---
    if rle is None or not isinstance(rle, dict):
        return None
    counts = rle.get("counts", None)
    size   = rle.get("size", None)
    if not (isinstance(counts, list) and isinstance(size, (list, tuple)) and len(size) == 2):
        return None
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    val = 0
    idx = 0
    for c in counts:
        if c > 0:
            flat[idx:idx + c] = val
            idx += c
        val = 1 - val
    # RLE is Fortran-order (column-major)
    return flat.reshape((h, w), order="F").astype(np.float32)

def vis_flags_from_depth_and_mask(
    uv, zc, depth_map, rle_mask, im_w, im_h,
    eps_ratio=0.06, mask_dilate_px=3, depth_window=2,
    fx=None, fy=None, cx=None, cy=None
):
    # v=0 off-screen/behind; v=1 occluded/not visible; v=2 visible
    # decode mask (prefer pycocotools; else pure-Python fallback)
    M = None
    try:
        from pycocotools import mask as maskUtils
        if rle_mask is not None:
            M = maskUtils.decode(rle_mask).astype(np.float32)
    except Exception:
        M = _decode_coco_rle(rle_mask)

    # canvas size from depth if present, else provided dims
    H, W = (depth_map.shape if depth_map is not None else (im_h, im_w))
    vlist = []

    # focal max for adaptive dilation (safe if intrinsics missing)
    fmax = float(max(fx or 0.0, fy or 0.0))

    for (u, v), z in zip(uv, zc):
        # off-screen / behind (edges inclusive)
        if not np.isfinite(u) or not np.isfinite(v) or z <= 0:
            vlist.append(0); continue
        if (u < 0) or (v < 0) or (u > (W - 1)) or (v > (H - 1)):
            vlist.append(0); continue

        # primary: robust on-mask test
        on_mask = False
        if M is not None:
            adapt = int(round(min(3.0, 0.003 * z * fmax))) if fmax > 0 else 0
            dil = max(int(mask_dilate_px), adapt)
            x0, x1, y0, y1 = _dilate_bounds(u, v, dil, W, H)
            if (x1 >= x0) and (y1 >= y0):
                if bool(M[y0:y1+1, x0:x1+1].max() > 0.0):
                    on_mask = True
                else:
                    # 3x3 bilinear subpixel taps
                    for dv in (-0.5, 0.0, 0.5):
                        for du in (-0.5, 0.0, 0.5):
                            if _bilinear_at(M, u + du, v + dv) > 0.25:
                                on_mask = True
                                break
                        if on_mask: break
        if on_mask:
            vlist.append(2); continue

        # secondary: depth agreement fallback
        visible = False
        if depth_map is not None:
            samples = []
            w = max(0, int(depth_window))
            for jj in range(-w, w + 1):
                for ii in range(-w, w + 1):
                    d = _bilinear_at(depth_map, u + ii, v + jj)
                    if np.isfinite(d) and d > 0:
                        samples.append(float(d))
            if samples:
                samples = np.asarray(samples, dtype=np.float32)
                med = float(np.median(samples))
                depth_quant = float(np.median(np.abs(samples - med))) if np.isfinite(med) else 1e-3
                if not (np.isfinite(depth_quant) and depth_quant > 0): depth_quant = 1e-3
                z_expected = float(z)
                r_expected = None
                if fx is not None and fy is not None and cx is not None and cy is not None:
                    du, dv = (u - cx) / float(fx), (v - cy) / float(fy)
                    r_expected = z_expected * float(np.sqrt(1.0 + du*du + dv*dv))
                tol = max(1e-4, eps_ratio * z_expected + 0.5 * depth_quant)
                ok_z = np.any(np.abs(samples - z_expected) <= tol)
                ok_r = np.any(np.abs(samples - r_expected) <= tol) if r_expected is not None else False
                visible = bool(ok_z or ok_r)

        vlist.append(2 if visible else 1)
    return vlist

############################################################

def add_random_occluders_between(cam2world_list, obj, n_min=1, n_max=3):
    obj_loc = obj.get_location()
    for _ in range(random.randint(n_min, n_max)):
        Twc = np.array(random.choice(cam2world_list))
        cam_loc = Twc[:3, 3]
        f = obj_loc - cam_loc
        dist = np.linalg.norm(f)
        if dist < 1e-6:
            continue
        f /= dist

        up = np.array([0, 0, 1.0])
        side = np.cross(f, up)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1, 0, 0])
        side /= np.linalg.norm(side)
        up = np.cross(side, f)

        center = (cam_loc + f * random.uniform(0.3 * dist, 0.8 * dist) +
                  side * random.uniform(-0.15, 0.15) * dist +
                  up   * random.uniform(-0.15, 0.15) * dist)

        if random.random() < 0.5:
            # PLANE
            occ = bproc.object.create_primitive('PLANE')
            s = random.uniform(0.15, 0.45)
            occ.set_scale([s, s, 1.0])
        else:
            # CUBE
            occ = bproc.object.create_primitive('CUBE')
            occ.set_scale(np.random.uniform([0.08, 0.08, 0.02],
                                            [0.35, 0.35, 0.12]))

        occ.set_location(center)
        occ.set_rotation_euler(np.random.uniform(0, np.pi, 3).tolist())
        occ.set_cp("category_id", -1)  # won't be annotated

        # neutral grey material 
        mat = bproc.material.create("OccMat")
        mat.set_principled_shader_value("Base Color", (0.4, 0.4, 0.4, 1.0))
        mat.set_principled_shader_value("Roughness", 0.8)
        mat.set_principled_shader_value("Metallic", 0.0)
        occ.replace_materials(mat)



def add_kp_schema_to_categories(coco_dict):
    cats = {c["id"]: c for c in coco_dict.get("categories", [])}
    for cid in (1,2):
        if cid not in cats:
            cats[cid] = {"id": cid, "name": ("needle_holder" if cid==1 else "tweezers")}
        cats[cid]["keypoints"] = KP_SCHEMA
        cats[cid]["skeleton"]  = KP_SKELETON
    coco_dict["categories"] = sorted(cats.values(), key=lambda c: c["id"])

def set_world_none():
    #Reset world to neutral (no HDRI), reuse if already created.
    try:
        world = bpy.data.worlds.get("NeutralWorld") or bpy.data.worlds.new("NeutralWorld")
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        bg = nodes.new("ShaderNodeBackground")
        out = nodes.new("ShaderNodeOutputWorld")
        bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # black
        bg.inputs[1].default_value = 1.0
        links.new(bg.outputs["Background"], out.inputs["Surface"])
        bpy.context.scene.world = world
    except Exception as e:
        print("[world] reset failed:", e)


def apply_hdri():
    #Pick a random HDRI for lighting/background.
    if not HDRI_LIST:
        set_world_none(); return
    hdri = random.choice(HDRI_LIST)
    bproc.world.set_world_background_hdr_img(hdri)




def three_point_lighting(obj):
    center = obj.get_location()
    # Key
    key = bproc.types.Light()
    key.set_type("AREA")
    key.set_location(center + np.array([2.5, -2.0, 2.0]))
    key.set_energy(random.uniform(1200, 1800))
    key.set_rotation_euler(np.deg2rad([50, 0, 35]))
    # Fill
    fill = bproc.types.Light()
    fill.set_type("AREA")
    fill.set_location(center + np.array([-2.0, -1.5, 1.5]))
    fill.set_energy(random.uniform(400, 700))
    # Rim
    rim = bproc.types.Light()
    rim.set_type("POINT")
    rim.set_location(center + np.array([-0.5, 2.5, 2.0]))
    rim.set_energy(random.uniform(300, 600))


def randomize_materials(obj):
    for mat in obj.get_materials():
        name = (mat.get_name() or "").lower()
        if "metal" in name or "steel" in name:
            mat.set_principled_shader_value("Metallic", 1.0)
            mat.set_principled_shader_value("Roughness", random.uniform(0.15, 0.5))
            mat.set_principled_shader_value("Base Color",
                np.append(np.random.uniform([0.6,0.6,0.6],[0.85,0.85,0.85]), 1.0))
        elif "gold" in name or "brass" in name:
            mat.set_principled_shader_value("Metallic", 1.0)
            mat.set_principled_shader_value("Roughness", random.uniform(0.05, 0.35))
            mat.set_principled_shader_value("Base Color",
                np.append(np.random.uniform([0.9,0.7,0.1],[1.0,0.85,0.2]), 1.0))



def set_intrinsics_jittered():
    # jitter focal length only; force principal point to the actual render center
    fx = base_fx * random.uniform(0.90, 1.10)
    fy = base_fy * random.uniform(0.90, 1.10)

    # Many BlenderProc builds ignore non-centered cx,cy. Make the renderer and our projections agree:
    cx = (im_w - 1) * 0.5
    cy = (im_h - 1) * 0.5

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    bproc.camera.set_intrinsics_from_K_matrix(K, im_w, im_h)

    # Return exactly what the renderer is using so later projections match pixel-perfectly
    return fx, fy, cx, cy


def sample_poses_around(obj_loc, target):
    poses, trials = [], 0
    while len(poses) < target and trials < target * 80:
        loc = bproc.sampler.shell(
            center=obj_loc, radius_min=3.5, radius_max=8.0,   # farther camera
            elevation_min=-45.0, elevation_max=65.0           # avoid extreme top/bottom
        )
        lookat = obj_loc  # look at the object center to keep it in frame
        R = bproc.camera.rotation_from_forward_vec(
            lookat - loc, inplane_rot=np.random.uniform(-0.5, 0.5)
        )
        cam2world = bproc.math.build_transformation_mat(loc, R)
        if len(bproc.camera.visible_objects(cam2world)) > 0:
            poses.append(cam2world)
        trials += 1
    return poses





def ensure_coco_categories(json_path):
    #Ensure categories block exists with class names (no keypoints yet).
    try:
        coco = json.load(open(json_path, "r"))
    except Exception:
        return
    cats = {c["id"]: c for c in coco.get("categories", [])}
    updated = False
    if 1 not in cats:
        cats[1] = {"id":1, "name":"needle_holder"}; updated = True
    if 2 not in cats:
        cats[2] = {"id":2, "name":"tweezers"}; updated = True
    if updated:
        coco["categories"] = sorted(cats.values(), key=lambda c: c["id"])
        json.dump(coco, open(json_path,"w"), indent=2)



def composite_last_n_images(json_path, n, bg_paths):
    if n <= 0 or not bg_paths:
        return
    try:
        from PIL import Image
    except Exception as e:
        print("[composite] Pillow not available, skipping compositing:", repr(e))
        return

    from pathlib import Path
    import random, json
    coco = json.load(open(json_path, "r"))

    imgs = coco.get("images", [])
    subset = imgs[-n:] if n <= len(imgs) else imgs
    print("[composite] planning to paste over", len(subset), "frames")


    for img in subset:
        p = Path(img.get("file_name", ""))

        candidates = [
            OUT_DIR / p,                  # exact path from JSON (relative/absolute)
            OUT_DIR / "images" / p.name   # common layout "<out>/images/<name>"
        ]
        fp = next((c for c in candidates if c.exists()), None)
        if fp is None:
            print(f"[composite] skip (missing image): {p}")
            continue

        try:
            fg = Image.open(fp).convert("RGBA")
            bg_path = random.choice(bg_paths)
            bg = Image.open(bg_path).convert("RGB").resize((fg.width, fg.height))
            comp = Image.alpha_composite(bg.convert("RGBA"), fg).convert("RGB")
            comp.save(fp)  # overwrite with composited RGB
            print(f"[composite] wrote {fp.name} over {Path(bg_path).name}")
        except Exception as e:
            print("[composite] failed for", fp, ":", repr(e))






def repair_coco_categories(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)
    id2name = {1: "needle_holder", 2: "tweezers"}
    new_cats = []
    for cid, name in id2name.items():
        new_cats.append({"id": cid, "name": name})
    coco["categories"] = new_cats
    # also remap any annotation with unknown id to 1 or drop; here we drop -1s:
    anns = []
    for a in coco.get("annotations", []):
        if a.get("category_id") in id2name:
            anns.append(a)
    coco["annotations"] = anns
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)


def depth_for_frame(depth_data, idx, nframes, expected_h, expected_w):
    # ---Return a 2D depth map for frame idx, or None if unavailable.---

    if depth_data is None:
        return None
    try:
        import numpy as _np
        arr = _np.asarray(depth_data)
        # list of 2D arrays
        if isinstance(depth_data, list):
            D = depth_data[idx]
        # (N,H,W)
        elif arr.ndim == 3 and arr.shape[0] == nframes:
            D = arr[idx]
        # (H,W,N)
        elif arr.ndim == 3 and arr.shape[-1] == nframes:
            D = arr[..., idx]
        # already 2D
        elif arr.ndim == 2:
            D = arr
        else:
            return None
        # fix accidental transpose
        if D.shape != (expected_h, expected_w) and D.shape == (expected_w, expected_h):
            D = D.T
        return D
    except Exception:
        return None
    

def enable_depth_safe():
    #---Handle BlenderProc API differences across versions.---
    try:
        # Most strict: positional-only
        bproc.renderer.enable_depth_output(False)
        print("[depth] enable_depth_output(False)")
        return
    except TypeError:
        pass
    except Exception as e:
        # keep trying other signatures
        print("[depth] positional call failed:", repr(e))
    try:
        # Keyword form
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        print("[depth] enable_depth_output(activate_antialiasing=False)")
        return
    except TypeError:
        pass
    except Exception as e:
        print("[depth] keyword call failed:", repr(e))
    try:
        # Legacy: no-arg
        bproc.renderer.enable_depth_output()
        print("[depth] enable_depth_output()")
        return
    except Exception as e:
        print("[depth] failed to enable depth:", repr(e))





def set_film_transparent(flag: bool):
    try:
        bpy.context.scene.render.film_transparent = bool(flag)
        print("[film] transparent =", bool(flag))
    except Exception as e:
        print("[film] set failed:", e)







CAT_TURN = 1  # alternate 1↔2 to guarantee both
def pick_next_obj():
    global CAT_TURN
    # prefer current CAT_TURN, else fallback to any available
    cands = [(p,c) for (p,c) in ALL_OBJS if c==CAT_TURN and per_obj_count[p] < args.per_obj_cap]
    if not cands:
        cands = [(p,c) for (p,c) in ALL_OBJS if per_obj_count[p] < args.per_obj_cap]
        if not cands: return None
    choice = random.choice(cands)
    CAT_TURN = 2 if CAT_TURN == 1 else 1
    return choice


# ---------------------- Main loop ----------------------
images_rendered = 0
first_write = True
per_obj_count = {p: 0 for p,_ in ALL_OBJS}
obj_index = 0


while images_rendered < args.max_images:

    picked = pick_next_obj()
    if picked is None: break
    obj_path, cat_id = picked


    # Scene reset + object
    bproc.clean_up()
    strategy = sample_strategy()  # studio | hdri | composite
    print(f"[loop] Using {strategy} :: {obj_path}")

    if strategy == "hdri":
        set_world_none()             # <-- ensure a world exists
        apply_hdri()
        set_film_transparent(False) ###################NEWNEWNEW
        bproc.renderer.set_output_format(enable_transparency=False)
    else:
        set_world_none()
        set_film_transparent(True) # <-- make background truly transparent  ###################NEWNEWNEW
        # For studio + composite, render with alpha so we can composite later if needed
        bproc.renderer.set_output_format(enable_transparency=True)
    
    ##############################################################################################

    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_cp("category_id", cat_id)  # will be picked up by the segmentation writer
    ##############################################################################################


    # Instrument articulation: different OBJs imply different jaw angles/forms
    # Pose/orientation jitter
    obj.set_location(obj.get_location() + np.random.uniform([-0.12,-0.12, 0.00],
                                                            [ 0.12, 0.12, 0.25]))
    obj.set_rotation_euler(np.random.uniform(0, np.pi, 3).tolist())

    # Materials
    randomize_materials(obj)


    # Local-space KPs for this OBJ (prefer sidecar JSON; fallback to auto-extrema)
    kps_local_h = load_kps_3d_for_obj(obj_path)
    if kps_local_h is None:
        kps_local_h = auto_kp_local_from_mesh(obj)  # (5,4)

    # Object local->world transform (static within this mini-loop)
    T_obj2world = np.array(obj.get_local2world_mat())  # (4,4)




    # Lighting
    if strategy in ("studio", "composite"):
        three_point_lighting(obj)   # replaces point_light_near(obj)


    # Camera intrinsics + poses
    fx, fy, cx_used, cy_used = set_intrinsics_jittered()
    remaining = args.max_images - images_rendered
    remaining_obj = args.per_obj_cap - per_obj_count[obj_path]
    target_poses = max(1, min(remaining, remaining_obj, 20))  # at most 20 per object iteration
    poses = sample_poses_around(obj.get_location(), target_poses)
    for p in poses:
        bproc.camera.add_camera_pose(p)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # NEW: add occluders for these poses
    add_random_occluders_between(poses, obj, n_min=1, n_max=3)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if len(poses) == 0:
        print("[warn] no visible poses, skipping object"); continue

    # Render
    bproc.renderer.set_max_amount_of_samples(int(args.samples))
    # Enable segmentation output so COCO writer gets masks
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "category_id"],
        default_values={"category_id": -1}
    )

    # Depth map for depth-accurate visibility (version-safe)
    enable_depth_safe()



    try:
        data = bproc.renderer.render()
    except Exception as e:
        print("[render] exception:", repr(e))
        traceback.print_exc()
        raise



    # -------------------- POST-RENDER PIPELINE --------------------
    try:
        print("[debug] render keys:", list(data.keys()))
        req = ("instance_segmaps" in data) and ("instance_attribute_maps" in data) and ("colors" in data)
        if not req:
            raise RuntimeError(f"Renderer outputs missing keys for COCO writer: {list(data.keys())}")

        # 1) Write COCO
        bproc.writer.write_coco_annotations(
            str(OUT_DIR),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            mask_encoding_format="rle",  # safe with our fallback decoder
            append_to_existing_output=(not first_write)
        )
        first_write = False

        # 2) Update intrinsics on the last N images
        coco = json.load(open(COCO_JSON, "r"))
        coco_imgs = coco.get("images", [])
        print("[coco] images:", len(coco_imgs), "annotations:", len(coco.get('annotations', [])))

        for img in coco_imgs[-len(poses):]:
            img["fx"] = float(fx); img["fy"] = float(fy)
            img["cx"] = float(cx_used); img["cy"] = float(cy_used)
            img["width"] = im_w; img["height"] = im_h
        if 'coco' in locals():
            json.dump(coco, open(COCO_JSON, "w"), indent=2)


        # 3) Composite over COCO photos if needed
        if strategy == "composite":
            composite_last_n_images(COCO_JSON, len(poses), BG_LIST)

        # 4) Inject keypoints + visibility
        coco = json.load(open(COCO_JSON, "r"))
        add_kp_schema_to_categories(coco)

        img2ann_idxs = {}
        for i, ann in enumerate(coco.get("annotations", [])):
            img2ann_idxs.setdefault(ann.get("image_id"), []).append(i)

        last_imgs  = coco.get("images", [])[-len(poses):]
        depth_data = data.get("depth", None)

        for frame_idx, im in enumerate(last_imgs):
            cam2world = poses[frame_idx]
            fx_i = float(im.get("fx", fx)); fy_i = float(im.get("fy", fy))
            cx_i = float(im.get("cx", cx_used)); cy_i = float(im.get("cy", cy_used))
            im_w_i = int(im.get("width", im_w));  im_h_i = int(im.get("height", im_h))
            depth_i = depth_for_frame(depth_data, frame_idx, len(poses), im_h_i, im_w_i)

            uv, zc = project_keypoints_local_to_pixels(
                kps_local_h, T_obj2world, cam2world, fx_i, fy_i, cx_i, cy_i
            )

            ann_idxs = img2ann_idxs.get(im.get("id"), [])
            if not ann_idxs:
                print("[kp] no annotations for image id", im.get("id"))
                continue
            pick, best_area = None, -1
            for ai in ann_idxs:
                a = coco["annotations"][ai]
                if a.get("category_id") == cat_id and a.get("area", 0) > best_area:
                    pick, best_area = ai, a.get("area", 0)
            if pick is None:
                pick = ann_idxs[0]
            ann = coco["annotations"][pick]

            rle = ann.get("segmentation", None)
            vflags = vis_flags_from_depth_and_mask(
                uv, zc, depth_i, rle, im_w_i, im_h_i, fx=fx_i, fy=fy_i, cx=cx_i, cy=cy_i
            )

            flat = []
            for (u, v), vis in zip(uv, vflags):
                if vis == 0: flat += [0.0, 0.0, 0]
                else:        flat += [float(u), float(v), int(vis)]
            ann["keypoints"]   = flat
            ann["num_keypoints"] = int(sum(1 for i in range(2, len(flat), 3) if flat[i] > 0))

        json.dump(coco, open(COCO_JSON, "w"), indent=2)

    except Exception as e:
        import traceback
        print("[post] exception:", repr(e))
        traceback.print_exc()
        raise
    # ------------------ END POST-RENDER PIPELINE ------------------



    # Bookkeeping
    per_obj_count[obj_path] += len(poses)
    images_rendered += len(poses)
    print(f"[progress] +{len(poses)} -> {images_rendered}/{args.max_images} (obj cap {per_obj_count[obj_path]}/{args.per_obj_cap})")

# Ensure categories (names only; keypoints will be added later)
ensure_coco_categories(str(COCO_JSON))
repair_coco_categories(str(COCO_JSON))
# Re-assert keypoint schema on categories (repair wiped it)
_c = json.load(open(COCO_JSON, "r"))
add_kp_schema_to_categories(_c)
json.dump(_c, open(COCO_JSON, "w"), indent=2)

print(f"[done] Wrote {images_rendered} images -> {OUT_DIR}")


