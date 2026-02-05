import argparse
from read_write_model import read_images_binary, write_images_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--num_cameras", type=int, required=True)
    args = parser.parse_args()

    src_images = read_images_binary(args.src)

    # camera_id → image を整理
    cam_images = {}
    for img in src_images.values():
        cam_images[img.camera_id] = img

    new_images = {}
    image_id = 1

    for t in range(args.num_frames):
        for cam_id in range(1, args.num_cameras + 1):
            base = cam_images[cam_id]

            new_img = base._replace(
                id=image_id,
                name=f"{t:06d}_cam{cam_id-1:02d}.png",
                xys=[],
                point3D_ids=[]
            )

            new_images[image_id] = new_img
            image_id += 1

    write_images_text(new_images, args.dst)

if __name__ == "__main__":
    main()
