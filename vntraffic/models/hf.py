"""HuggingFace Transformers backend with per-family adapters (EXPERIMENT_PLAN.md §4).

Three adapter families, all behind the common ``VLMClient.generate`` interface:

  * InternVL3   — ``AutoModel(trust_remote_code=True).chat`` with the model's
                  dynamic-tiling image preprocessing;
  * Qwen2.5-VL  — ``Qwen2_5_VLForConditionalGeneration`` + ``AutoProcessor`` +
                  ``qwen_vl_utils.process_vision_info``;
  * Llama-3.2-Vision — ``MllamaForConditionalGeneration`` + ``AutoProcessor``.

Heavy imports (torch/transformers/PIL) are deferred to first use so the data
gates and metrics import this package on a CPU-only box without them. Decoding
is greedy with a per-call ``torch.manual_seed`` for reproducibility.

This code targets the documented model-card APIs; it is intended to run on the
user's GPU machine (no GPU is available in the authoring environment).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .base import GenerationConfig, VLMClient
from .registry import (FAMILY_INTERNVL, FAMILY_MLLAMA, FAMILY_QWEN_VL, get_spec)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# InternVL dynamic-tiling preprocessing (from the InternVL model card)
# --------------------------------------------------------------------------- #
def _internvl_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    ow, oh = image.size
    ar = ow / oh
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    tar = _find_closest_aspect_ratio(ar, ratios, ow, oh, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((tw, th))
    out = []
    cols = tw // image_size
    for i in range(blocks):
        box = ((i % cols) * image_size, (i // cols) * image_size,
               ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)
        out.append(resized.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def _internvl_load_image(path, input_size=448, max_num=12):
    import torch
    from PIL import Image
    image = Image.open(path).convert("RGB")
    transform = _internvl_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


# --------------------------------------------------------------------------- #
# Base HF client
# --------------------------------------------------------------------------- #
class HFClient(VLMClient):
    backend = "hf"

    def __init__(self, model_key: str, model_id: str, family: str,
                 device: str = "cuda", dtype: str = "bfloat16",
                 revision: Optional[str] = None, max_tiles: int = 12):
        super().__init__(model_key, model_id, revision)
        self.family = family
        self.device = device
        self.dtype = dtype
        self.max_tiles = max_tiles
        self._model = None
        self._tokenizer = None
        self._processor = None

    def _torch_dtype(self):
        import torch
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.dtype]

    def _seed(self, gen_cfg: GenerationConfig):
        import torch
        torch.manual_seed(gen_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_cfg.seed)

    # dispatch ------------------------------------------------------------- #
    def _load(self):
        raise NotImplementedError

    def generate(self, prompt: str, images: Optional[Sequence[str]] = None,
                 system: Optional[str] = None,
                 gen_cfg: Optional[GenerationConfig] = None) -> str:
        gen_cfg = gen_cfg or GenerationConfig()
        if self._model is None:
            self._load()
        self._seed(gen_cfg)
        images = list(images or [])
        return self._generate(prompt, images, system, gen_cfg)

    def _generate(self, prompt, images, system, gen_cfg) -> str:
        raise NotImplementedError


class InternVLClient(HFClient):
    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False)
        self._model = AutoModel.from_pretrained(
            self.model_id, torch_dtype=self._torch_dtype(),
            trust_remote_code=True, low_cpu_mem_usage=True).eval().to(self.device)

    def _generate(self, prompt, images, system, gen_cfg) -> str:
        gcfg = dict(max_new_tokens=gen_cfg.max_new_tokens, do_sample=gen_cfg.do_sample)
        if system:
            try:
                self._model.system_message = system
            except Exception:
                prompt = system + "\n\n" + prompt
        if images:
            pv = _internvl_load_image(images[0], max_num=self.max_tiles)
            pv = pv.to(self._torch_dtype()).to(self.device)
            question = "<image>\n" + prompt
            resp = self._model.chat(self._tokenizer, pv, question, gcfg)
        else:
            resp = self._model.chat(self._tokenizer, None, prompt, gcfg)
        return resp.strip() if isinstance(resp, str) else str(resp)


class QwenVLClient(HFClient):
    def _load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self._torch_dtype(),
            device_map=self.device).eval()

    def _generate(self, prompt, images, system, gen_cfg) -> str:
        from qwen_vl_utils import process_vision_info
        content = [{"type": "image", "image": p} for p in images]
        content.append({"type": "text", "text": prompt})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs or None,
                                 videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=gen_cfg.max_new_tokens,
                                    do_sample=gen_cfg.do_sample)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()


class MllamaClient(HFClient):
    def _load(self):
        from transformers import AutoProcessor, MllamaForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self._torch_dtype(),
            device_map=self.device).eval()

    def _generate(self, prompt, images, system, gen_cfg) -> str:
        from PIL import Image
        pil = [Image.open(p).convert("RGB") for p in images]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})

        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(images=pil or None, text=text,
                                 return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=gen_cfg.max_new_tokens,
                                   do_sample=gen_cfg.do_sample)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(gen, skip_special_tokens=True).strip()


_FAMILY_TO_CLIENT = {
    FAMILY_INTERNVL: InternVLClient,
    FAMILY_QWEN_VL: QwenVLClient,
    FAMILY_MLLAMA: MllamaClient,
}


def build_hf_client(model_key: str, device: str = "cuda", dtype: str = "bfloat16",
                    max_tiles: int = 12) -> HFClient:
    spec = get_spec(model_key)
    cls = _FAMILY_TO_CLIENT[spec.family]
    return cls(model_key=spec.key, model_id=spec.hf_id, family=spec.family,
               device=device, dtype=dtype, max_tiles=max_tiles)
