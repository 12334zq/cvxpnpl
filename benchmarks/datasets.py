from collections import namedtuple
import json
import os
from os.path import join as pjoin

import numpy as np
from PIL import Image
from plymit import Ply

Model = namedtuple(
    "Model",
    [
        "id",
        "points",
        "normals",
        "color",
        "faces",
        "diameter",
        "min",
        "size",
        "symmetries_discrete",
    ],
)


class LinemodOcclusion:
    class _Sequence:
        def __init__(self, name, prefix, models):

            self.name = name
            self.prefix = prefix
            self.models = models

            # parse gt
            gt = json.loads(open(pjoin(prefix, "scene_gt.json")).read())
            self.poses = [None] * len(gt.keys())
            for k, v in gt.items():
                poses = {}
                for pose in v:
                    poses[pose["obj_id"]] = np.hstack((
                        np.array(pose["cam_R_m2c"]).reshape((3, 3)),
                        np.array(pose["cam_t_m2c"]).reshape((3, 1)),
                    ))
                self.poses[int(k)] = poses

            # iterator stuff
            self.i = 0

        def __iter__(self):
            self.i = 0
            return self

        def __len__(self):
            return len(self.poses)

        def __next__(self):
            # reached the end. get out
            if self.i == len(self):
                raise StopIteration

            # return dictionary object with rgb, depth and poses
            data = {
                "id": self.i,
                "rgb": np.array(Image.open(pjoin(self.prefix, "rgb", "{:06d}.png".format(self.i)))), # load rgb
                "depth": np.array(Image.open(pjoin(self.prefix, "depth", "{:06d}.png".format(self.i)))), # load depth
                "poses": self.poses[self.i]
            }
            self.i += 1
            return data

    class _Partition:
        def __init__(self, prefix, models):

            self.prefix = prefix
            self.models = models

            seq_names = sorted([d.name for d in os.scandir(prefix)])
            self.sequences = [
                LinemodOcclusion._Sequence(int(n), pjoin(prefix, n), models)
                for n in seq_names
            ]


        def __iter__(self):
            return iter(self.sequences)

    def __init__(self, prefix):

        self.prefix = prefix
        self.camera = self._parse_camera()
        self.models = self._load_models()

        # self.train = type(self)._Partition(pjoin(self.prefix, "train"))
        self.train = None
        self.test = type(self)._Partition(pjoin(self.prefix, "test"), self.models)

    def _parse_camera(self):
        data = json.loads(open(pjoin(self.prefix, "camera.json")).read())
        camera = {
            "K": np.array(
                ((data["fx"], 0, data["cx"]), (0, data["fy"], data["cy"]), (0, 0, 1),)
            ),
            "size": (data["width"], data["height"]),
        }
        return camera

    def _load_models(self):

        models = {}

        # load model info
        info = json.loads(open(pjoin(self.prefix, "models", "models_info.json")).read())
        for k, v in info.items():

            # load points, normals and color
            ply = Ply(pjoin(self.prefix, "models", "obj_{:06d}.ply".format(int(k))))

            # parse vertices
            points = []
            normals = []
            colors = []
            for vertex in ply.elementLists["vertex"]:
                points.extend([vertex.x, vertex.y, vertex.z])
                normals.extend([vertex.nx, vertex.ny, vertex.nz])
                colors.extend([vertex.red, vertex.green, vertex.blue])
            points = np.array(points, dtype=np.float32).reshape((-1, 3))
            normals = np.array(normals, dtype=np.float32).reshape((-1, 3))
            colors = np.array(colors, dtype=np.uint8).reshape((-1, 3))

            # faces
            faces = []
            for f in ply.elementLists["face"]:
                faces.extend(f.vertex_indices)
            faces = np.array(faces, dtype=np.uint32).reshape((-1, 3))

            # create model object
            models[k] = Model(
                k,
                points,
                normals,
                colors,
                faces,
                v["diameter"],
                np.array((v["min_x"], v["min_y"], v["min_z"])),
                np.array((v["size_x"], v["size_y"], v["size_z"])),
                [np.array(s).reshape((4, 4)) for s in v["symmetries_discrete"]]
                if "symmetries_discrete" in v
                else None,
            )
        return models


def parse_arguments():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Dataset prefix folder")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    ds = LinemodOcclusion(args.prefix)

    for sequence in ds.test:
        for frame in sequence:
            import pdb; pdb.set_trace()
