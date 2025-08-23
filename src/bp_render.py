# src/bp_render.py
# Runs INSIDE BlenderProc: "python -m blenderproc run src/bp_render.py --outdir ..."
import blenderproc as bproc  # must be imported first in this process
import os, glob, json, random, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", required=True, help="Output dir (images/, coco_annotations.json)")
parser.add_argument("--max_images", type=int, default=1000, help="Total images to generate")
parser.add_argument("--per_obj_cap", type=int, default=200, help="Max frames per single object")
args, _ = parser.parse_known_args()

OUT_DIR = args.outdir
MAX_IMAGES = int(args.max_images)
PER_OBJ_CAP = int(args.per_obj_cap)

# ----- Initialize BlenderProc -----
bproc.init()
# Optional GPU pin:
# bproc.renderer.set_render_devices(desired_gpu_device_type=["CUDA"], desired_gpu_ids=[0])

# ----- Paths from environment -----
PROJ_DATA = os.environ.get("PROJ_DATA", "/datashare/project")
model_dirs = [
    os.path.join(PROJ_DATA, "surgical_tools_models/needle_holder"),
    os.path.join(PROJ_DATA, "surgical_tools_models/tweezers"),
]
camera_json_path = os.path.join(PROJ_DATA, "camera.json")

# Ensure output dirs
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

# ----- Camera intrinsics -----
with open(camera_json_path, "r") as f:
    cam = json.load(f)
base_fx, base_fy = cam["fx"], cam["fy"]
cx, cy = cam["cx"], cam["cy"]
im_w, im_h = cam["width"], cam["height"]

images_rendered, first_write = 0, True

for model_dir in model_dirs:
    if images_rendered >= MAX_IMAGES:
        break

    obj_paths = sorted(glob.glob(os.path.join(model_dir, "*.obj")))
    for obj_path in obj_paths:
        if images_rendered >= MAX_IMAGES:
            break

        print(f"[bp_render] Object: {obj_path}")
        bproc.clean_up()

        # Load object and set category
        obj = bproc.loader.load_obj(obj_path)[0]
        cat_id = 1 if model_dir.endswith("needle_holder") else 2
        obj.set_cp("category_id", cat_id)

        # Random pose
        obj.set_location(obj.get_location() + np.random.uniform([-0.1, -0.1, 0.0], [0.1, 0.1, 0.2]))
        obj.set_rotation_euler(np.random.uniform(0, np.pi, 3).tolist())

        # Lighting
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(bproc.sampler.shell(center=obj.get_location(),
                                               radius_min=1, radius_max=5,
                                               elevation_min=1, elevation_max=89))
        light.set_energy(random.uniform(100, 1000))
        light.set_color(np.random.uniform([0.8, 0.8, 0.8], [1.0, 1.0, 1.0]))

        # Intrinsics (±20% focal randomization)
        fx = base_fx * random.uniform(0.8, 1.2)
        fy = base_fy * random.uniform(0.8, 1.2)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        bproc.camera.set_intrinsics_from_K_matrix(K, im_w, im_h)

        # Sample camera poses that see the object
        remaining = MAX_IMAGES - images_rendered
        target_poses = min(PER_OBJ_CAP, remaining)
        poses = 0
        while poses < target_poses:
            loc = bproc.sampler.shell(center=obj.get_location(),
                                      radius_min=2, radius_max=10,
                                      elevation_min=-90, elevation_max=90)
            lookat = obj.get_location() + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
            R = bproc.camera.rotation_from_forward_vec(lookat - loc,
                                                       inplane_rot=np.random.uniform(-0.7854, 0.7854))
            cam2world = bproc.math.build_transformation_mat(loc, R)
            if obj in bproc.camera.visible_objects(cam2world):
                bproc.camera.add_camera_pose(cam2world, frame=poses)
                poses += 1

        # Render and write COCO
        bproc.renderer.set_max_amount_of_samples(100)
        bproc.renderer.set_output_format(enable_transparency=True)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

        data = bproc.renderer.render()
        images_rendered += poses
        print(f"[bp_render] rendered {poses}, total {images_rendered}/{MAX_IMAGES}")

        bproc.writer.write_coco_annotations(
            OUT_DIR,
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            mask_encoding_format="rle",
            append_to_existing_output=(not first_write)
        )
        first_write = False

        # Append intrinsics to just-written images
        coco_path = os.path.join(OUT_DIR, "coco_annotations.json")
        with open(coco_path, "r") as f:
            coco = json.load(f)
        for img in coco["images"][-poses:]:
            img.update({"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)})
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)

print(f"[bp_render] Done. Images: {images_rendered}, out: {OUT_DIR}")
