import argparse
import glob
import os
import shutil
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_id", type=str, default=None, required=True)

    return parser.parse_args()


def main(args):
    image_comparison_path = os.path.join(
        "experiments",
        args.expt_id,
        f"{args.expt_id}_image_comparison_and_molfiles_filtered"
    )
    fl = sorted(glob.glob(os.path.join(image_comparison_path, "*.predicted.png")))
    basenames = [fn.split("/")[-1].rstrip(".predicted.png") for fn in fl]

    output_path = f"experiments/{args.expt_id}/{args.expt_id}_to_correct"
    os.makedirs(output_path, exist_ok=True)

    for basename in basenames:
        img = Image.open(
            os.path.join(image_comparison_path, f"{basename}.ref.png")
        )

        fn = lambda x: 255 - (255 - x) / 3.0
        r = img.convert("L").point(fn)
        r.save(os.path.join(output_path, f"{basename}.ref.png"))

        shutil.copy2(
            os.path.join(image_comparison_path, f"{basename}.predicted.mol"),
            os.path.join(output_path, f"{basename}.predicted.mol"),
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
