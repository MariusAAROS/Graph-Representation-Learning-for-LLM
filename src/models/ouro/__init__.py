"""Local, transformers-v5-compatible port of ByteDance's Ouro looped LM.

The Hub remote code targets transformers 4.55 and fails to build under v5, so the
classes are registered with the Auto* factories here. `AutoModelForCausalLM` then
resolves `model_type: ouro` to this port without `trust_remote_code`, which also
keeps loading offline-safe on compute nodes.
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_ouro import OuroConfig
from .modeling_ouro import OuroForCausalLM

AutoConfig.register("ouro", OuroConfig, exist_ok=True)
AutoModelForCausalLM.register(OuroConfig, OuroForCausalLM, exist_ok=True)
