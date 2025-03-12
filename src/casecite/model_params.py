"""
model_params.py - Load and configure the LLM model for the CaseCite class.
"""
import os
from dotenv import load_dotenv


load_dotenv()

# List of valid vendors for the LLM model
# TODO: Load from database
VALID_VENDORS = ["anthropic", "openai", "gemini", "groq"]


class ModelParams():
    def __init__(self, vendor: str = None):
        """
        Initialize with a vendor name for the LLM model.

        :param vendor: The vendor name of the LLM model.
        """
        if not vendor:
            vendor = os.environ.get("LLM_VENDOR", "anthropic")

        self.vendor = vendor
        if vendor not in VALID_VENDORS:
            raise ValueError(f"Invalid vendor name: {vendor}")
        
        key = f"{vendor.upper()}_API_KEY"
        self.api_key = os.environ.get(key)
        if not self.api_key:
            raise ValueError(f"API key not found for vendor: {vendor} // {key}")
        
        key = f"{vendor.upper()}_MODEL"
        self.model = os.environ.get(key)
        if not self.model:
            raise ValueError(f"Model not found for vendor: {vendor} // {key}")
        
        self.n = os.environ.get(f"{vendor.upper()}_N", os.environ.get("N", 512))
        self.k = os.environ.get(f"{vendor.upper()}_K", os.environ.get("K"))
        self.p = os.environ.get(f"{vendor.upper()}_P", os.environ.get("P"))
        self.temperature = os.environ.get(f"{vendor.upper()}_TEMPERATURE", os.environ.get("TEMPERATURE"))

        if self.temperature and (self.k or self.p):
            raise ValueError("Temperature cannot be used with k or p.")
        
        if self.k and not self.p:
            raise ValueError("Top-p sampling requires a p value.")
        
        if self.p and not self.k:
            raise ValueError("Top-k sampling requires a k value.")
        
        if not self.k and not self.p and not self.temperature:
            self.temperature = 0.0
        
        self.max_tokens = os.environ.get(f"{vendor.upper()}_MAX_OUTPUT_TOKENS", os.environ.get("MAX_OUTPUT_TOKENS", 4096))
        self.stop = os.environ.get(f"{vendor.upper()}_STOP", os.environ.get("STOP", None))

    # Property to return configuration parameters as a dictionary
    @property
    def config(self):
        return {
            "api_key": self.api_key,
            "model": self.model,
            "n": self.n,
            "k": self.k,
            "p": self.p,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
        }