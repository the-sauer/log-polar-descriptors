# Copyright 2026 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch

from blob_matcher import HardNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", metavar="W")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--slim", action="store_true")
    parser.add_argument("--shallow", action="store_true")

    args = parser.parse_args()

    model = HardNet(patch_size=args.resolution, slim=args.slim, shallow=args.shallow)
    model.eval()
    model.load_state_dict(torch.load(args.weights, weights_only=False, map_location=torch.device("cpu"))["state_dict"])
    example_inputs = (torch.randn((1, 1, args.resolution, args.resolution)),)
    onnx_program = torch.onnx.export(
        model,
        args=example_inputs,
        input_names=["patches"],
        dynamic_shapes={"patches": (torch.export.dynamic_shapes.Dim.DYNAMIC, torch.export.dynamic_shapes.Dim.STATIC, torch.export.dynamic_shapes.Dim.STATIC, torch.export.dynamic_shapes.Dim.STATIC)},
        dynamo=True
    )
    onnx_program.save(f"{os.path.splitext(os.path.basename(args.weights))[0]}.onnx")
