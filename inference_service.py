import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from api.services.authentication_service import AuthenticationService
import logging
import os

# Set environment variable to handle memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceService:
    def __init__(self, authentication_key: str):
        try:
            # Set model directory (configurable via environment variable)
            model_dir = os.getenv('MODEL_DIR', '/workspace/model')

            # Clear GPU memory before loading the model
            torch.cuda.empty_cache()

            # Load custom model and tokenizer with memory optimizations
            logger.info(f"Loading custom model from {model_dir} on device: {device}")

            # Load model using FP16 (half precision) for memory savings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True
            ).half().to(device)  # FP16

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # Enable gradient checkpointing for additional memory efficiency
            self.model.gradient_checkpointing_enable()

            # Clear any residual memory after model loading
            torch.cuda.empty_cache()

            # Initialize authentication service
            self.authentication_key = authentication_key
            self.authentication_service = AuthenticationService(authentication_key)

            logger.info("Model and tokenizer successfully loaded with optimizations.")

        except torch.cuda.OutOfMemoryError as oom:
            logger.error("CUDA Out of Memory Error during model loading", exc_info=True)
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory during model load. Try reducing model size.") from oom

        except Exception as e:
            logger.error(f"Error loading custom model or tokenizer: {e}", exc_info=True)
            raise RuntimeError(f"Error loading custom model or tokenizer: {str(e)}")
    def generate_text(self, input_context: str, max_length: int = 512, temperature: float = 0.3) -> str:
        """
        Generate text from the custom model with a batch size of 1 and all possible memory optimizations.
        """
        try:
            with open('/workspace/lo-backend/prompt.txt', 'r') as file:
               if input_context:
                  input_context = f"{input_context}." if input_context[-1] not in ['?', '!', '.'] else input_context
               else:
                  input_context = ""
               prompt = file.read().replace('<<CONTEXT>>', f"context: {input_context}")

            # Tokenize the input context with truncation and padding to ensure batch size of 1
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

            # Ensure batch size is 1
            assert input_ids.shape[0] == 1, "Batch size is not 1"

            # Perform inference using mixed precision and no gradients
            with torch.no_grad():  # Ensure no gradients are computed
                with autocast():  # Use FP16 inference for memory savings
                    output = self.model.generate(input_ids, max_length=max_length, temperature=temperature)

            # Clear GPU memory after inference to avoid memory fragmentation
            torch.cuda.empty_cache()

            # Decode the generated output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            generated_text = generated_text.split('<bot>:')[-1].strip()
            logger.info(f"Generated text: {generated_text}")
            return generated_text

        except torch.cuda.OutOfMemoryError as oom:
            logger.error("CUDA Out of Memory during inference", exc_info=True)
            torch.cuda.empty_cache()  # Attempt to free memory
            raise RuntimeError("Out of GPU memory during inference. Try with a smaller input.") from oom

        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def generate_text_with_batch_size(self, input_contexts: list, batch_size: int = 1, max_length: int = 128, temperature: float = 0.7) -> list:
        """
        Generate text with a specified batch size. Handles multiple inputs and splits into batches.
        """
        try:
            # Split input contexts into batches based on the batch size
            batched_inputs = [input_contexts[i:i + batch_size] for i in range(0, len(input_contexts), batch_size)]

            generated_texts = []
            for batch in batched_inputs:
                # Tokenize the batch of inputs efficiently with padding and truncation
                input_ids = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

                # Perform inference on each batch with mixed precision
                with torch.no_grad():
                    with autocast():
                        outputs = self.model.generate(input_ids, max_length=max_length, temperature=temperature)

                # Decode outputs for each batch element
                batch_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                generated_texts.extend(batch_texts)

                # Clear GPU memory after each batch to avoid memory fragmentation
                torch.cuda.empty_cache()

            return generated_texts

        except torch.cuda.OutOfMemoryError as oom:
            logger.error("CUDA Out of Memory Error during batch inference", exc_info=True)
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory during batch inference. Try reducing batch size or input length.") from oom

        except Exception as e:
            logger.error(f"Error generating text for batch: {e}", exc_info=True)
            raise RuntimeError(f"Batch text generation failed: {str(e)}")
