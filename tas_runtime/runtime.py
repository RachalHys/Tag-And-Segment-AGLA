"""One loaded LLaVA instance, existing model definitions and explicit inference backends."""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path

import torch

from .attention import set_attention_backend
from .decoding import ForwardState


class LlavaBackend:
    """Adapter around the existing multimodal preparation and Llama backbone.

    All prompts in a batch have equal length. Masks are all-valid; no hidden
    padding/position reinterpretation is allowed. KV state belongs to the caller.
    """

    def __init__(self, model, *, last_position_logits=True):
        self.model = model
        self.last_position_logits = last_position_logits

    def _finish(self, output):
        hidden = output.last_hidden_state
        if self.last_position_logits:
            hidden = hidden[:, -1:, :]
        logits = self.model.lm_head(hidden)[:, -1, :]
        past = output.past_key_values
        return ForwardState(logits, past, past[0][0].shape[-2])

    @torch.inference_mode()
    def prefill(self, prompt_ids, images):
        attention = torch.ones_like(prompt_ids, dtype=torch.bool)
        if images.ndim == 3:
            # Optional immutable projected-feature reuse within a minibatch.
            # Identical geometry/model/dtype already belong to the caller's key.
            from llava.constants import IMAGE_TOKEN_INDEX

            if getattr(self.model.config, "tune_mm_mlp_adapter", False):
                raise ValueError(
                    "Cached-feature prefill does not support trainable adapter token semantics."
                )
            rows = []
            for tokens, features in zip(prompt_ids, images):
                locations = (tokens == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if locations.numel() != 1:
                    raise ValueError(
                        "Feature reuse requires exactly one image token per prompt."
                    )
                split = int(locations[0])
                embed = self.model.get_model().embed_tokens
                rows.append(
                    torch.cat(
                        (embed(tokens[:split]), features, embed(tokens[split + 1 :])),
                        dim=0,
                    )
                )
            if len({row.shape[0] for row in rows}) != 1:
                raise ValueError(
                    "Feature prefill batches must have equal-length unpadded prompts."
                )
            embeds = torch.stack(rows)
            ids, past = None, None
            attention = torch.ones(
                embeds.shape[:2], dtype=torch.bool, device=embeds.device
            )
        else:
            ids, attention, past, embeds, _ = (
                self.model.prepare_inputs_labels_for_multimodal(
                    prompt_ids, attention, None, None, images
                )
            )
        output = self.model.get_model()(
            input_ids=ids,
            inputs_embeds=embeds,
            attention_mask=attention,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        return self._finish(output)

    @torch.inference_mode()
    def advance(self, tokens, state):
        attention = torch.ones(
            (tokens.shape[0], state.length + tokens.shape[1]),
            dtype=torch.bool,
            device=tokens.device,
        )
        positions = torch.arange(
            state.length, state.length + tokens.shape[1], device=tokens.device
        ).unsqueeze(0)
        output = self.model.get_model()(
            input_ids=tokens,
            attention_mask=attention,
            position_ids=positions,
            past_key_values=state.past,
            use_cache=True,
            return_dict=True,
        )
        return self._finish(output)


@contextlib.contextmanager
def legacy_sampler(model, project_root, *, enabled=True):
    """Use the original sampler only for a scoped reference call, never at import."""
    from transformers.generation.utils import GenerationMixin

    original = GenerationMixin.sample
    if enabled:
        source = Path(project_root) / "eval_tas" / "sample.py"
        spec = importlib.util.spec_from_file_location(
            "_tas_frozen_sampler_reference", source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        GenerationMixin.sample = module.sample
    try:
        yield
    finally:
        GenerationMixin.sample = original


class LlavaRuntime:
    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        project_root,
        *,
        last_position_logits=True,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.project_root = Path(project_root)
        self.backend = LlavaBackend(model, last_position_logits=last_position_logits)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.set_attention("eager")

    @classmethod
    def load(cls, paths, project_root, *, device="cuda:0", last_position_logits=True):
        from transformers import AutoTokenizer
        from llava.model.language_model.llava_llama import (
            LlavaConfig,
            LlavaLlamaForCausalLM,
        )
        from llava.constants import (
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IMAGE_PATCH_TOKEN,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The run phase needs a CUDA runtime. Use phase=prepare/score on CPU."
            )
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        config = LlavaConfig.from_pretrained(paths["llava"], local_files_only=True)
        config.mm_vision_tower = str(Path(paths["vision"]).resolve())
        tokenizer = AutoTokenizer.from_pretrained(
            paths["llava"], use_fast=False, local_files_only=True
        )
        model = LlavaLlamaForCausalLM.from_pretrained(
            paths["llava"],
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": device},
            use_safetensors=False,
            local_files_only=True,
        )
        if getattr(config, "mm_use_im_patch_token", True):
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if getattr(config, "mm_use_im_start_end", False):
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))
        vision = model.get_vision_tower()
        if not vision.is_loaded:
            vision.load_model()
        vision.to(device=device, dtype=torch.float16)
        model.requires_grad_(False)
        return cls(
            model,
            tokenizer,
            vision.image_processor,
            project_root,
            last_position_logits=last_position_logits,
        )

    def set_attention(self, backend):
        self.attention_backend = set_attention_backend(
            self.model, backend, unpadded=True
        )

    def prompt(self, query, conv_mode="llava_v1"):
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token

        if conv_mode not in conv_templates:
            raise ValueError(f"Unknown conversation template: {conv_mode}")
        image_token = DEFAULT_IMAGE_TOKEN
        if self.model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], image_token + "\n" + query)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        token_ids = tokenizer_image_token(
            text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_ids = self.tokenizer(stop, add_special_tokens=False).input_ids
        if stop != self.tokenizer.eos_token or stop_ids != [
            self.tokenizer.eos_token_id
        ]:
            raise ValueError(
                "The initial validated runtime supports llava_v1 with EOS stopping. Other templates require an explicit stopping audit."
            )
        if int((token_ids == IMAGE_TOKEN_INDEX).sum()) != 1:
            raise ValueError("The captioning runtime requires exactly one image token.")
        return text, token_ids, stop

    def pixels(self, image, *, canvas=False):
        options = {"do_resize": False, "do_center_crop": False} if canvas else {}
        tensor = self.image_processor.preprocess(image, return_tensors="pt", **options)[
            "pixel_values"
        ][0]
        return tensor

    @torch.inference_mode()
    def features(self, image_batch):
        return self.model.encode_images(
            image_batch.to(device=self.device, dtype=self.dtype)
        )

    def decode_text(self, tokens, stop):
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if stop and text.endswith(stop):
            text = text[: -len(stop)].strip()
        return text

    @torch.inference_mode()
    def reference(
        self,
        prompt_ids,
        original,
        object_view=None,
        *,
        seed=0,
        max_new_tokens=24,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
    ):
        """Native E0 or original E1 sample implementation with original warper order."""
        from llava.mm_utils import KeywordsStoppingCriteria

        attention = self.attention_backend
        self.set_attention("eager")
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with (
                torch.random.fork_rng(devices=devices),
                legacy_sampler(
                    self.model, self.project_root, enabled=object_view is not None
                ),
            ):
                torch.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed_all(seed)
                stops = KeywordsStoppingCriteria(
                    [self.tokenizer.eos_token], self.tokenizer, prompt_ids
                )
                kwargs = dict(
                    images=original,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stops],
                )
                if object_view is not None:
                    kwargs.update(images_cd=object_view, cd_alpha=2.0, cd_beta=0.5)
                return self.model.generate(prompt_ids, **kwargs)[
                    :, prompt_ids.shape[1] :
                ]
        finally:
            self.set_attention(attention)
