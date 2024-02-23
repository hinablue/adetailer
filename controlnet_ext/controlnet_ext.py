from __future__ import annotations

import importlib
import re
import sys
import numpy as np
import copy

from functools import lru_cache
from pathlib import Path
from textwrap import dedent

from modules.paths_internal import models_path, extensions_dir, extensions_builtin_dir
from modules_forge.forge_util import numpy_to_pytorch

from lib_controlnet import global_state, external_code
from lib_controlnet.external_code import ControlNetUnit

from modules import extensions, sd_models, shared, scripts
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img

controlnet_exists = True

for extension in extensions.active():
    if not extension.enabled:
        continue

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
}

cn_model_regex = re.compile("|".join(cn_model_module.keys()))

def find_script(p : StableDiffusionProcessing, script_title : str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script_title  ), None)
    if not script:
        raise Exception("Script not found: " + script_title)
    return script

def add_forge_script_to_adetailer_run(p: StableDiffusionProcessing, script_title : str, script_args : list):
    p.scripts = copy.copy(scripts.scripts_img2img)
    p.scripts.alwayson_scripts = []
    p.script_args_value = []

    script = copy.copy(find_script(p, script_title))
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)

    print(f"Added script: {script.title()} with {len(script_args)} args, at positions {script.args_from}-{script.args_to} (of 0-{len(p.script_args_value)-1}.)", file=sys.stderr)

class ControlNetExt:
    def __init__(self):
        self.cn_available = False
        self.external_cn = external_code

    def init_controlnet(self):
        self.cn_available = True

    def get_controlnet_script_args(
        self,
        p,
        model: str,
        module: str | None,
        weight: float,
        guidance_start: float,
        guidance_end: float,
    ):
        if (not self.cn_available) or model == "None":
            return

        try:
            image = np.asarray(p.init_images[0])
            mask = np.zeros_like(image)
            mask[:] = 255

            cnet_image = {
                "image": image,
                "mask": mask
            }

            pres = external_code.pixel_perfect_resolution(
                image,
                target_H=p.height,
                target_W=p.width,
                resize_mode=external_code.resize_mode_from_value(p.resize_mode)
            )

            add_forge_script_to_adetailer_run(
                p,
                "ControlNet",
                [
                    ControlNetUnit(
                        enabled=True,
                        image=cnet_image,
                        model=model,
                        module=module,
                        weight=weight,
                        guidance_start=guidance_start,
                        guidance_end=guidance_end,
                        processor_res=pres
                    )
                ]
            )
        except AttributeError as e:
            if "script_args_value" not in str(e):
                raise
            msg = "[-] Adetailer: ControlNet option not available in WEBUI version lower than 1.6.0 due to updates in ControlNet"
            raise RuntimeError(msg) from e

def get_cn_models() -> list[str]:
    if controlnet_exists:
        models = global_state.get_all_controlnet_names()
        return list(m for m in models if cn_model_regex.search(m))
    return []
