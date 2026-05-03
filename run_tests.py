#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from input_loaders import HuiInputLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper class for loading and interacting with Qwen model."""
    
    def __init__(self, model_name: str, quantization_mode: str | None = "4bit"):
        """Initialize model and tokenizer."""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.quantization_mode = quantization_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.logger.info(f"Loading model: {model_name}")
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        
    def _determine_compute_dtype(self) -> torch.dtype:
        """Determine appropriate compute dtype based on CUDA capability."""
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                self.logger.debug("CUDA capability >= 8.0, using bfloat16 for compute.")
                return torch.bfloat16
        self.logger.debug("CUDA capability < 8.0 or CUDA not available, using float16 for compute.")
        return torch.float16
    
    def _get_bnb_config(self, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
        """Create BitsAndBytesConfig if quantization is requested."""
        if self.quantization_mode == "4bit":
            self.logger.debug("Configuring for 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif self.quantization_mode == "8bit":
            self.logger.debug("Configuring for 8-bit quantization.")
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from Hugging Face."""
        compute_dtype = self._determine_compute_dtype()
        bnb_config = self._get_bnb_config(compute_dtype)
        
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = compute_dtype
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.logger.debug(f"Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to load model '{self.model_name}'. Error: {e}")
            raise
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                padding_side="left",
                trust_remote_code=True,
            )
            self.logger.debug(f"Tokenizer for '{self.model_name}' loaded successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to load tokenizer for '{self.model_name}'. Error: {e}")
            raise
        
        # Add pad token if needed
        if tokenizer.pad_token is None:
            new_pad_token = "<|pad|>"
            tokenizer.add_special_tokens({"pad_token": new_pad_token})
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            model.config.pad_token_id = tokenizer.pad_token_id
        
        model.eval()
        return tokenizer, model
    
    def generate_response(
        self,
        system_prompt: str,
        user_input: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Generate response given system prompt and user input."""
        # Format the conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                self.logger.warning(f"Failed to apply chat template: {e}")
                prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        else:
            prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        # Get the actual input length (accounting for padding)
        input_length = inputs["attention_mask"].sum(dim=1)[0].item()
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return response.strip()


def load_obfuscated_system_prompt_from_pt(pt_file: Path, tokenizer) -> str:
    """Load obfuscated system prompt tensor from .pt file and decode it."""
    try:
        sys_prompt_tensor = torch.load(pt_file, weights_only=True)
        sys_prompt_str = tokenizer.decode(sys_prompt_tensor, skip_special_tokens=True)
        return sys_prompt_str
    except Exception as e:
        logger.error(f"Failed to load obfuscated system prompt from {pt_file}: {e}")
        raise


def load_conventional_system_prompt_from_params(params_file: Path) -> str:
    """Load conventional system prompt from params.json file."""
    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
            if 'system_prompt' in params:
                return params['system_prompt']
            else:
                raise KeyError("'system_prompt' not found in params.json")
    except Exception as e:
        logger.error(f"Failed to load conventional system prompt from {params_file}: {e}")
        raise


def main(
    pape_dir: str = "results/Pape",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """Main function to run inference with candidates."""

    pape_path = Path(pape_dir)
    
    if not pape_path.exists():
        logger.error(f"Pape directory not found: {pape_path}")
        sys.exit(1)
    
    # Initialize model
    model_wrapper = ModelWrapper(model_name=model_name, quantization_mode="4bit")
    
    input_loaders = [HuiInputLoader()]

    inputs = []
    for loader in input_loaders:
        try:
            new_inputs = loader.load_inputs()
        except Exception as e:
            logger.error(f"Failed to load inputs from {loader.source_dir} directory: {e}")
            sys.exit(1)

        if not new_inputs:
            logger.error(f"No inputs available from {loader.source_dir}")
            sys.exit(1)

        logger.info(f"Found {len(new_inputs)} input files in {loader.source_dir}")
        inputs.extend(new_inputs)
    
    # Get all candidate directories from Pape directory
    pape_dirs = [d for d in pape_path.iterdir() if d.is_dir()]
    if not pape_dirs:
        logger.error(f"No directories found in {pape_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(pape_dirs)} candidate directories in {pape_path}")
    
    # Process each input
    for path, user_input in inputs:
        logger.info(f"Processing {path.name}")
        
        results = {}
        
        # Process each Pape candidate directory
        for pape_candidate_dir in pape_dirs:
            logger.info(f"Processing candidate: {pape_candidate_dir.name}")
            
            best_candidate_file = pape_candidate_dir / "best_candidate.pt"
            params_file = pape_candidate_dir / "params.json"
            
            if not best_candidate_file.exists():
                logger.warning(f"best_candidate.pt not found in {pape_candidate_dir}")
                continue
            
            if not params_file.exists():
                logger.warning(f"params.json not found in {pape_candidate_dir}")
                continue
            
            # Load obfuscated system prompt from best_candidate.pt
            try:
                obfuscated_system_prompt = load_obfuscated_system_prompt_from_pt(
                    best_candidate_file,
                    model_wrapper.tokenizer
                )
                logger.info(f"Loaded obfuscated system prompt: {obfuscated_system_prompt[:50]}...")
            except Exception as e:
                logger.error(f"Failed to load obfuscated system prompt: {e}")
                continue
            
            # Load conventional system prompt from params.json
            try:
                conventional_system_prompt = load_conventional_system_prompt_from_params(params_file)
                logger.info(f"Loaded conventional system prompt: {conventional_system_prompt[:50]}...")
            except Exception as e:
                logger.error(f"Failed to load conventional system prompt: {e}")
                continue
            
            # Generate response with obfuscated system prompt
            obfuscated_response = None
            try:
                obfuscated_response = model_wrapper.generate_response(
                    system_prompt=obfuscated_system_prompt,
                    user_input=user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                logger.info(f"Generated obfuscated response: {obfuscated_response[:50]}...")
            except Exception as e:
                logger.error(f"Failed to generate obfuscated response: {e}")
            
            # Generate response with conventional system prompt
            conventional_response = None
            try:
                conventional_response = model_wrapper.generate_response(
                    system_prompt=conventional_system_prompt,
                    user_input=user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                logger.info(f"Generated conventional response: {conventional_response[:50]}...")
            except Exception as e:
                logger.error(f"Failed to generate conventional response: {e}")
            
            results[pape_candidate_dir.name] = {
                "user_input": user_input,
                "conventional_system_prompt": conventional_system_prompt,
                "conventional_response": conventional_response,
                "obfuscated_system_prompt": obfuscated_system_prompt,
                "obfuscated_response": obfuscated_response,
            }
        
        output_filename = f"{path.stem}_test_results.json"
        output_path = path.parent / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results for {path.name} saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses using system prompts from best_candidate.pt"
    )
    parser.add_argument(
        "--hui-dir",
        type=str,
        default="results/Hui",
        help="Path to Hui results directory"
    )
    parser.add_argument(
        "--pape-dir",
        type=str,
        default="results/Pape",
        help="Path to Pape results directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Model name from Hugging Face"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for generation"
    )
    
    args = parser.parse_args()
    
    main(
        pape_dir=args.pape_dir,
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
