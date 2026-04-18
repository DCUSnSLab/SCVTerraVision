"""Print instructions for obtaining supported datasets.

We don't auto-download because both datasets require manual acceptance.
"""

import argparse
import textwrap

INSTRUCTIONS = {
    "rugd": textwrap.dedent(
        """
        RUGD (Robot Unstructured Ground Driving)
          Site: http://rugd.vision/
          Steps:
            1. Download RUGD_frames-with-annotations.zip and RUGD_annotations.zip
            2. Extract under data/rugd/ so you have:
                 data/rugd/images/<scene>/<scene>_<frame>.png
                 data/rugd/labels/<scene>/<scene>_<frame>.png
            3. (Optional) Place split files under data/rugd/splits/{train,val,test}.txt
               with one stem per line, e.g. "creek/creek_00001"
        """
    ),
    "rellis3d": textwrap.dedent(
        """
        RELLIS-3D
          Repo: https://github.com/unmannedlab/RELLIS-3D
          Steps:
            1. Follow the repo's download instructions
            2. Extract under data/rellis3d/ so you have:
                 data/rellis3d/<seq>/pylon_camera_node/frame_*.jpg
                 data/rellis3d/<seq>/pylon_camera_node_label_id/frame_*.png
            3. Place official splits under data/rellis3d/splits/{train,val,test}.lst
        """
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(INSTRUCTIONS), default=None)
    args = parser.parse_args()

    keys = [args.dataset] if args.dataset else sorted(INSTRUCTIONS)
    for k in keys:
        print(INSTRUCTIONS[k])


if __name__ == "__main__":
    main()
