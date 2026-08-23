import os
import struct

def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            # skip keypoints
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)
            images[image_id] = name
    return images

images_bin = "/home/arua/thesis/data/eiffel_tower/processed/scene/sparse/0/images.bin"
splits_dir = "/home/arua/thesis/data/eiffel_tower/splits"
os.makedirs(splits_dir, exist_ok=True)

images = read_images_bin(images_bin)

# sort by name for deterministic ordering
sorted_names = sorted(images.values())

# every 8th image is test, rest is train (standard 3DGS convention)
train = [n for i, n in enumerate(sorted_names) if i % 8 != 0]
test  = [n for i, n in enumerate(sorted_names) if i % 8 == 0]

with open(os.path.join(splits_dir, "train.txt"), "w") as f:
    f.write("\n".join(train))

with open(os.path.join(splits_dir, "test.txt"), "w") as f:
    f.write("\n".join(test))

print(f"Total images: {len(sorted_names)}")
print(f"Train:        {len(train)}")
print(f"Test:         {len(test)}")
print(f"Splits saved to {splits_dir}")