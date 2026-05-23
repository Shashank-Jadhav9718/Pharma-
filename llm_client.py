"""
LLM Client module for handling Mistral, Gemini, and OpenRouter API integrations.
Supports multiple cloud-based LLM models with standardized interface and error handling.
"""

import asyncio
import httpx
import time
import logging
import re
from typing import Dict, Any, Optional, List
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Intelligent rate limiter for API calls to prevent overwhelming servers.
    Adapts to API response patterns and implements smart delays.
    """
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 30.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.consecutive_rate_limits = 0
        self.success_count = 0
        
    async def wait_if_needed(self):
        """Smart waiting based on recent API behavior - optimized for cloud models."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Calculate required delay based on recent performance
        if self.consecutive_rate_limits > 0:
            required_delay = min(self.max_delay, self.base_delay * (2 ** self.consecutive_rate_limits))
        else:
            # Minimal delay for cloud models with no rate limit issues
            required_delay = self.base_delay * 0.5
            
        if time_since_last < required_delay:
            wait_time = required_delay - time_since_last
            logger.debug(f"Rate limiter: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            
        self.last_request_time = time.time()
        
    def record_success(self):
        """Record successful API call."""
        self.success_count += 1
        if self.success_count > 5:  # Reset after several successes
            self.consecutive_rate_limits = max(0, self.consecutive_rate_limits - 1)
            self.success_count = 0
            
    def record_rate_limit(self):
        """Record rate limit hit."""
        self.consecutive_rate_limits += 1
        self.success_count = 0


class LLMClient:
    """
    Unified client for multiple LLM backends (Mistral, Gemini, OpenRouter models).
    Handles API calls, retries, and response processing.
    """
    
    def __init__(self, api_key: str, model_type: str):
        """
        Initialize LLM client with API key and model type.
        
        Args:
            api_key (str): API key for the selected LLM service
            model_type (str): 'mistral', 'gemini', 'openchat', 'deepseek', or 'gptoss'
        """
        self.api_key = api_key or ""  # Handle None/empty API keys
        self.model_type = model_type.lower()
        
        # Mistral API configuration - using OpenRouter for Mistral 7B
        self.mistral_api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Default Mistral model for OpenRouter
        self.mistral_model = 'mistralai/mistral-nemo'
        
        # OpenRouter model mapping for various types
        self.openrouter_models = {
            'mistral': 'mistralai/mistral-nemo',
            'openchat': 'openchat/openchat-7b',
            'deepseek': 'deepseek/deepseek-r1',
            'gptoss': 'openai/gpt-4o-mini'
        }
        
        # Gemini configuration with enhanced settings for accuracy
        if self.model_type == 'gemini':
            genai.configure(api_key=api_key)
            
            # Configure generation settings for optimal speed and accuracy balance
            generation_config = genai.types.GenerationConfig(
                temperature=0.02,       # Slightly reduced for better accuracy
                top_p=0.85,            # Optimized for speed while maintaining quality
                top_k=40,              # Balanced for speed and vocabulary coverage
                max_output_tokens=1200, # Increased for longer descriptions
                candidate_count=1       # Single best candidate
            )
            
            # Enhanced system instruction for pharmaceutical accuracy
            system_instruction = (
             "Always be factually accurate and industry-compliant."
             "Focus on therapeutic benefits, proper usage, dosage forms, and key features."
             "Use a clear, formal, and informative tone."
             "Do not write like an advertisement."
             "Do not include disclaimers, warnings, or unnecessary filler."
             "Write only the pure description, starting directly with the product information."
            )
            
            self.gemini_model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=generation_config,
                system_instruction=system_instruction
            )
    

    def _clean_response(self, text: str) -> str:
        """
        Aggressively clean the LLM response and convert asterisks to circle bullets.
        
        Args:
            text (str): Raw response text from LLM
            
        Returns:
            str: Cleaned text with circle bullets (•) instead of asterisks
        """
        try:
            if not text or not isinstance(text, str):
                logger.debug("Empty or invalid text input to _clean_response")
                return ""
            
            # AGGRESSIVE TOKEN REMOVAL - Remove special tokens that commonly appear at start/end
            # Remove at the very beginning before any other processing
            text = str(text).strip()
            
            # Remove leading/trailing special tokens - MORE AGGRESSIVE
            special_token_patterns = [
                r'\[/?s\]',  # [s] or [/s] anywhere
                r'</s>',  # </s> anywhere
                r'<s>',  # <s> anywhere
                r'\[/?INST\]',  # [INST] or [/INST] anywhere
                r'<\|endoftext\|>',  # GPT end token
                r'<\|startoftext\|>',  # GPT start token
                r'\[EOS\]', r'\[BOS\]',  # End/Beginning of sequence
                r'<EOS>', r'<BOS>',  # Alt format
            ]
            
            for pattern in special_token_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # REMOVE LLM INSTRUCTION TOKENS - Remove all special instruction tokens that leak into output
            # These are internal tokens used by models like Mistral, Llama, etc.
            instruction_tokens = [
                r'\[BINST\]', r'\[/BINST\]',  # Begin/End instruction
                r'\[INST\]', r'\[/INST\]',    # Instruction markers
                r'\[BOT\]', r'\[/BOT\]',      # Bot response markers
                r'\[OUT\]', r'\[/OUT\]',      # Output markers
                r'\[BOLD\]', r'\[/BOLD\]',    # Bold markers
                r'\[SYS\]', r'\[/SYS\]',      # System markers
                r'\[USER\]', r'\[/USER\]',    # User markers
                r'\[ASSISTANT\]', r'\[/ASSISTANT\]',  # Assistant markers
                r'<\|im_start\|>', r'<\|im_end\|>',  # ChatML tokens
                r'<\|system\|>', r'<\|user\|>', r'<\|assistant\|>',  # Role tokens
                r'<<SYS>>', r'<</SYS>>',      # Llama system tokens
                r'\[BEGIN\]', r'\[END\]',     # Generic begin/end
            ]
            
            # Remove all instruction tokens (case-insensitive)
            for token in instruction_tokens:
                text = re.sub(token, '', text, flags=re.IGNORECASE)
            
            # REMOVE PROMPT LEAKAGE - Remove phrases like "Here is the SHORT DESCRIPTION for..."
            prompt_patterns = [
                r'Here is the (SHORT|LONG) DESCRIPTION for[^:]*:?\s*',
                r'Here is the (short|long) description for[^:]*:?\s*',
                r'(SHORT|LONG) DESCRIPTION:?\s*',
                r'(Short|Long) Description:?\s*',
                r'Description:?\s*',
                r'Product Description:?\s*',
                r'<[^>]*>Here is[^<]*</[^>]*>',  # Remove HTML-wrapped prompt text
                r'<li>Here is[^<]*</li>',
                r'<ul>.*?Here is.*?</ul>',
            ]
            
            for pattern in prompt_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
            
            # REMOVE CONVERSATIONAL/ASSISTANCE PHRASES - Remove LLM politeness and offers to help
            conversational_patterns = [
                r'Please let me know if you need any further assistance[^.!?]*[.!?]?\s*',
                r'Let me know if you need any further assistance[^.!?]*[.!?]?\s*',
                r'Please let me know if you have any questions[^.!?]*[.!?]?\s*',
                r'Let me know if you have any questions[^.!?]*[.!?]?\s*',
                r'If you need any further information[^.!?]*[.!?]?\s*',
                r'If you have any questions[^.!?]*[.!?]?\s*',
                r'Please feel free to ask[^.!?]*[.!?]?\s*',
                r'Feel free to ask[^.!?]*[.!?]?\s*',
                r'I hope this helps[^.!?]*[.!?]?\s*',
                r'Hope this helps[^.!?]*[.!?]?\s*',
                r'Please consult[^.!?]*[.!?]?\s*',
                r'Note:?\s*This (is|description|content)[^.!?]*[.!?]?\s*',
                r'Disclaimer:?[^.!?]*[.!?]?\s*',
                r'Is there anything else[^.!?]*[.!?]?\s*',
                r'Would you like me to[^.!?]*[.!?]?\s*',
                r'Let me know if[^.!?]*[.!?]?\s*',
                r'Please note[^.!?]*[.!?]?\s*',
            ]
            
            for pattern in conversational_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # SUPER AGGRESSIVE ASTERISK REMOVAL - Remove ALL asterisks first (including unicode variants)
            text = str(text).replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
            
            # Now process line by line - just clean, don't add bullets yet
            # (bullets will be added only for short descriptions in post-processing)
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                try:
                    line = str(line).strip().replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
                    # Convert dash bullets to circle bullets (preserve existing bullets)
                    if line.startswith('- '):
                        line = '• ' + line[2:]
                    elif line.startswith('-'):
                        line = '• ' + line[1:].strip()
                    # Keep line as-is (don't auto-add bullets)
                    if line:
                        cleaned_lines.append(line)
                except Exception as e:
                    logger.warning(f"Error processing line in _clean_response: {str(e)}")
                    # Add the original line if processing fails
                    line_clean = str(line).strip().replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
                    if line_clean:
                        cleaned_lines.append(line_clean)
            text = '\n'.join(cleaned_lines)
            
            # Apply regex cleaning with error handling
            try:
                # Remove markdown bold formatting (**text** and __text__)
                text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                text = re.sub(r'__(.*?)__', r'\1', text)
                # Remove markdown italic formatting (*text* and _text_)
                text = re.sub(r'\*([^*\n]+?)\*', r'\1', text)
                text = re.sub(r'_([^_\n]+?)_', r'\1', text)
                # Remove any remaining asterisks (should be none left)
                text = re.sub(r'[\*＊﹡∗]+', '', text)
                # Remove any remaining underscores used for formatting
                text = re.sub(r'[_]+', '', text)
                # Remove markdown headers (# ## ###)
                text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
                # Remove markdown code blocks (```text```)
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
                
                # Remove inline code (`text`)
                text = re.sub(r'`(.*?)`', r'\1', text)
                
                # Remove HTML tags if any
                text = re.sub(r'<[^>]+>', '', text)
                
                # Remove any other special formatting characters
                text = re.sub(r'[~`]', '', text)
                
                # Remove special LLM tokens like [/s], </s>, [INST], etc.
                text = re.sub(r'\[/?s\]', '', text, flags=re.IGNORECASE)
                text = re.sub(r'</s>', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\[/?INST\]', '', text, flags=re.IGNORECASE)
                text = re.sub(r'<\|.*?\|>', '', text)
                
                # Clean up extra whitespace
                text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
                text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces to single
                
            except Exception as e:
                logger.warning(f"Error in regex cleaning: {str(e)}")
                # Fallback to basic cleaning if regex fails
                text = text.replace('*', '').replace('_', '').replace('`', '')
            
            # FINAL ASTERISK REMOVAL - Make absolutely sure no asterisks remain
            text = str(text).replace('*', '')
            
            # REMOVE OTC AND OVER-THE-COUNTER PHRASES - Remove entire phrases to avoid broken sentences
            otc_patterns = [
                r'\b(?:is|are)\s+(?:available\s+as|an|a)\s+(?:OTC|over-the-counter|over\s+the\s+counter)(?:\s+(?:product|medicine|medication|drug))?[.\s]*',
                r'\b(?:OTC|over-the-counter|over\s+the\s+counter)\s+(?:product|medicine|medication|drug)[.\s]*',
                r'\b(?:available|sold)\s+(?:as\s+)?(?:OTC|over-the-counter|over\s+the\s+counter)[.\s]*',
                r'\bOTC\b[.\s]*',
                r'\bover-the-counter\b[.\s]*',
                r'\bover\s+the\s+counter\b[.\s]*',
                r'\b(?:prescription|non-prescription)\s+(?:status|required|not\s+required)[.\s]*',
                r'\b(?:requires?|does\s+not\s+require)\s+(?:a\s+)?prescription[.\s]*',
            ]
            
            for pattern in otc_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # COMPREHENSIVE PUNCTUATION CLEANUP - Fix all stray punctuation issues
            
            # Remove stray punctuation marks appearing alone or with minimal context
            text = re.sub(r'\s+[,;:]\s+', ' ', text)  # Remove standalone commas, semicolons, colons
            text = re.sub(r'\s+\.\s+', '. ', text)  # Fix standalone periods (keep as sentence ender)
            text = re.sub(r'^\s*[,;:.]\s*', '', text, flags=re.MULTILINE)  # Remove punctuation at line start
            
            # Clean up spaces before punctuation
            text = re.sub(r'\s+([,;:.])', r'\1', text)  # Remove spaces before punctuation
            text = re.sub(r'([,;:])\s{2,}', r'\1 ', text)  # Single space after punctuation
            
            # Remove duplicate punctuation
            text = re.sub(r'\.{2,}', '.', text)  # Multiple periods to single
            text = re.sub(r',{2,}', ',', text)  # Multiple commas to single
            text = re.sub(r';{2,}', ';', text)  # Multiple semicolons to single
            text = re.sub(r'\s{2,}', ' ', text)  # Multiple spaces to single
            
            # ADVANCED PUNCTUATION CLEANUP - Fix broken sentences and orphaned periods
            # Remove orphaned periods that start a line or follow opening tags
            text = re.sub(r'<li>\s*[,;:.]\s*', '<li>', text)  # Remove punctuation at start of list item
            text = re.sub(r'<ul>\s*[,;:.]\s*', '<ul>', text)  # Remove punctuation after <ul>
            
            # Remove broken sentence fragments like "is ." or "are ," etc.
            text = re.sub(r'\b(is|are|was|were|be|been|being|has|have|had|do|does|did)\s+[,;:.]', r'\1', text, flags=re.IGNORECASE)
            
            # Remove stray punctuation between words (except periods which end sentences)
            text = re.sub(r'([a-zA-Z])\s+[,;:]\s+([a-zA-Z])', r'\1 \2', text)  # Remove comma/semicolon between words
            
            # Clean up punctuation before closing tags
            text = re.sub(r'[,;:.]+\s*</li>', '</li>', text)  # Remove punctuation before </li>
            text = re.sub(r'\.\s*\.', '.', text)  # Remove double periods
            
            # Remove incomplete brackets or parentheses
            text = re.sub(r'\(\s*$', '', text, flags=re.MULTILINE)  # Remove unclosed opening bracket at end
            text = re.sub(r'^\s*\)', '', text, flags=re.MULTILINE)  # Remove unmatched closing bracket at start
            text = re.sub(r'\[\s*$', '', text, flags=re.MULTILINE)  # Remove unclosed square bracket at end
            text = re.sub(r'^\s*\]', '', text, flags=re.MULTILINE)  # Remove unmatched square bracket at start
            
            # REMOVE PLACEHOLDER TEXT - Remove any text in square brackets like [Brand], [Product], etc.
            text = re.sub(r'\bby\s+\[Brand\]', '', text, flags=re.IGNORECASE)  # Remove "by [Brand]"
            text = re.sub(r'\[Brand\]', '', text, flags=re.IGNORECASE)  # Remove standalone [Brand]
            text = re.sub(r'\[Product\]', '', text, flags=re.IGNORECASE)  # Remove [Product]
            text = re.sub(r'\[[^\]]*\]', '', text)  # Remove any other text in square brackets
            
            # Clean up spacing after placeholder removal
            text = re.sub(r'\s{2,}', ' ', text)  # Multiple spaces to single
            
            # STRICT: Remove forbidden medical claim words
            forbidden_words = [
                ('cure', 'support'), ('cures', 'supports'), ('curing', 'supporting'),
                ('treat', 'intended for'), ('treats', 'intended for'), ('treating', 'intended for'), ('treatment', 'use'),
                ('heal', 'support'), ('heals', 'supports'), ('healing', 'supporting'),
                ('guarantee', 'formulated to'), ('guaranteed', 'formulated to'),
                ('diagnose', 'identify'), ('diagnosis', 'identification'),
                ('prevent', 'may support'), ('prevents', 'may support'), ('prevention', 'support'),
                ('therapy', 'routine'), ('therapeutic', 'beneficial'),
                ('OTC', ''), ('over-the-counter', ''), ('over the counter', '')
            ]
            
            for forbidden, replacement in forbidden_words:
                # Case-insensitive replacement
                text = re.sub(r'\b' + re.escape(forbidden) + r'\b', replacement, text, flags=re.IGNORECASE)
            
            # FINAL CLEANUP PASS - Catch any remaining issues
            # Remove any remaining special tokens anywhere in text
            text = re.sub(r'\[/?s\]', '', text, flags=re.IGNORECASE)
            text = re.sub(r'</s>', '', text, flags=re.IGNORECASE)
            text = re.sub(r'<s>', '', text, flags=re.IGNORECASE)
            
            # Final punctuation cleanup
            text = re.sub(r'\s{2,}', ' ', text)  # Remove any double spaces created
            text = re.sub(r'\s+\.', '.', text)  # Remove spaces before periods
            text = re.sub(r'^\s*\.\s*', '', text, flags=re.MULTILINE)  # Remove periods at line start
            
            # Remove empty list items that may have been created
            text = re.sub(r'<li>\s*</li>', '', text)
            text = re.sub(r'<li>\s*\.\s*</li>', '', text)
            
            # FINAL CLEANUP - Ensure proper sentence endings
            text = text.strip()
            
            # If text doesn't end with proper punctuation, add a period
            if text and not text.endswith(('.', '!', '?', '</li>', '</ul>')):
                # Check if the last sentence is incomplete
                last_words = text.split()[-5:] if len(text.split()) >= 5 else text.split()
                # If it looks like an incomplete sentence, add period
                if last_words:
                    text = text.rstrip() + '.'
            
            # Remove any trailing orphaned words after punctuation cleanup
            text = re.sub(r'\s+\w{1,2}\s*$', '.', text)  # Replace trailing 1-2 letter words with period
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Critical error in _clean_response: {str(e)}")
            # Return safe fallback
            return str(text).replace('*', '').strip() if text else ""
    
    def _generate_fallback_description(self, product_name: str, description_type: str, product_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Zero-fail fallback: Generate meaningful description from raw Excel data.
        Ensures no product is ever left without a proper description.
        
        Args:
            product_name (str): Product name
            description_type (str): 'short' or 'long'
            product_data (dict): Excel data with size and mrp.
            
        Returns:
            str: Fallback description generated from available data
        """
        try:
            logger.info(f"🛟 FALLBACK ACTIVATED for {product_name}")
            
            if not product_data:
                product_data = {}
            
            # Extract and clean available data
            product_name = str(product_name or product_data.get('product_name', 'Product')).strip()
            size = str(product_data.get('size', 'N/A')).strip()
            mrp = str(product_data.get('mrp', 'N/A')).strip()
            
            if description_type == 'short':
                # Generate exactly 4 meaningful bullets using available data
                bullets = [
                    f"{product_name} is a commercially available formulation.",
                    f"Available in packaging size: {size}.",
                    f"Maximum Retail Price: ₹{mrp}.",
                    "Standard pharmaceutical preparation."
                ]
                
                # Format as HTML list
                html_bullets = "\n".join([f"<li>{bullet}</li>" for bullet in bullets])
                return f"<ul>\n{html_bullets}\n</ul>"
            
            else:  # long description
                sections = [
                    f"{product_name} is a commercially available pharmaceutical formulation.",
                    f"This product is provided in a packaging size of {size}.",
                    f"It has a Maximum Retail Price (MRP) of ₹{mrp}."
                ]
                
                # Join all sections with proper spacing
                long_desc = " ".join(sections)
                
                # Add mandatory disclaimer
                disclaimer = "\n\n<b>Important Note: This information is for general product purposes only and is not intended as medical advice. Please consult a doctor before consumption.</b>"
                
                return long_desc + disclaimer
                
        except Exception as e:
            logger.error(f"Error in fallback generation for {product_name}: {str(e)}")
            # Ultra-minimal fallback with proper structure
            if description_type == 'short':
                return f"<ul>\n<li>{product_name} is a commercially available formulation</li>\n<li>Available in standard packaging</li>\n<li>Contact pharmacy for pricing</li>\n<li>Standard pharmaceutical preparation</li>\n</ul>"
            else:
                return f"{product_name} is a commercially available pharmaceutical formulation. This product is provided in standard packaging. Please consult a doctor before consumption.\n\n<b>Important Note: This information is for general product purposes only and is not intended as medical advice. Please consult a doctor before consumption.</b>"
    
    async def generate_description(self, product_name: str, description_type: str, category: Optional[str] = None, product_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate high-accuracy product description with validation and zero-fail fallback.
        
        Args:
            product_name (str): Name of the pharmaceutical product
            description_type (str): Either 'short' or 'long'
            category (str): Optional product category
            product_data (dict): Optional rich product data (ingredients, benefits, etc.)
            
        Returns:
            str: Generated description (LLM-generated or fallback from Excel data)
        """
        try:
            # Input validation
            if not product_name or not isinstance(product_name, str):
                logger.error(f"Invalid product name: {product_name}")
                return self._generate_fallback_description(product_name, description_type, product_data)
            
            if description_type not in ['short', 'long']:
                logger.error(f"Invalid description type: {description_type}")
                return self._generate_fallback_description(product_name, description_type, product_data)
            
            product_name = str(product_name).strip()
            if not product_name:
                logger.error("Empty product name after cleaning")
                return self._generate_fallback_description(product_name, description_type, product_data)
            
            # Generate prompt with rich product data
            prompt = self._get_prompt(product_name, description_type, category, product_data)
            if not prompt:
                logger.error(f"Failed to generate prompt for {product_name}")
                return self._generate_fallback_description(product_name, description_type, product_data)
            
            # Generate description with selected model
            raw_description = ""
            try:
                if self.model_type in self.openrouter_models:
                    model_id = self.openrouter_models[self.model_type]
                    raw_description = await self._call_mistral(prompt, model_override=model_id)
                elif self.model_type == 'gemini':
                    raw_description = await self._call_gemini(prompt)
                else:
                    logger.error(f"Unsupported model type: {self.model_type}")
                    return self._generate_fallback_description(product_name, description_type, product_data)
            except Exception as e:
                logger.error(f"API call failed for {product_name}: {str(e)}")
                logger.warning(f"Using fallback mechanism for {product_name}")
                return self._generate_fallback_description(product_name, description_type, product_data)
            
            if not raw_description:
                logger.warning(f"Empty response from API for {product_name}")
                logger.warning(f"Using fallback mechanism for {product_name}")
                return self._generate_fallback_description(product_name, description_type, product_data)
            
            # FIRST: Clean the raw response to remove all asterisks
            try:
                cleaned_description = self._clean_response(raw_description)
            except Exception as e:
                logger.error(f"Cleaning failed for {product_name}: {str(e)}")
                # Fallback cleaning
                cleaned_description = str(raw_description).replace('*', '').strip()
            
            # REMOVE DUPLICATE PRODUCT NAME at the start (common LLM error)
            if cleaned_description and description_type == 'long':
                product_name_clean = product_name.strip()
                # If it starts with the product name, strip it to prevent stutter
                if cleaned_description.lower().startswith(product_name_clean.lower()):
                    cleaned_description = cleaned_description[len(product_name_clean):].strip().lstrip(',.- ')
                    if cleaned_description:
                        # Capitalize the new first letter
                        cleaned_description = cleaned_description[0].upper() + cleaned_description[1:]
            
            # THEN: Validate and enhance the cleaned description
            try:
                validated_description = self._validate_and_enhance_description(
                    cleaned_description, product_name, description_type
                )
            except Exception as e:
                logger.error(f"Validation failed for {product_name}: {str(e)}")
                # Return cleaned description as fallback
                validated_description = cleaned_description
            
            # MANDATORY: Append disclaimer to long descriptions
            if description_type == 'long' and validated_description:
                disclaimer = "\n\n<b>Important Note: This information is for general product purposes only and is not intended as medical advice. Always consult a healthcare professional before use.</b>"
                # Check if disclaimer not already present
                if "Important Note:" not in validated_description:
                    validated_description += disclaimer
            
            return validated_description or self._generate_fallback_description(product_name, description_type, product_data)
                
        except Exception as e:
            logger.error(f"Critical error generating description for {product_name}: {str(e)}")
            logger.warning(f"Using fallback mechanism for {product_name}")
            return self._generate_fallback_description(product_name, description_type, product_data)
    
    def _validate_and_enhance_description(self, description: str, product_name: str, description_type: str) -> str:
        """
        Validate and enhance the generated description for maximum accuracy and completeness.
        
        Args:
            description (str): Raw generated description
            product_name (str): Product name for context
            description_type (str): Type of description (short/long)
            
        Returns:
            str: Validated and enhanced description with improved accuracy
        """
        try:
            # Input validation
            if not description or not isinstance(description, str):
                logger.warning(f"Empty or invalid description for {product_name}")
                return ""
            
            if not product_name or not isinstance(product_name, str):
                logger.error(f"Invalid product name in validation: {product_name}")
                return str(description).strip()
            
            if description_type not in ['short', 'long']:
                logger.error(f"Invalid description type in validation: {description_type}")
                return str(description).strip()
            
            description = str(description).strip()
            product_name = str(product_name).strip()
            
            if len(description) < 15:
                logger.warning(f"Generated description too short for {product_name}: {len(description)} chars")
                return ""
            
            # Clean and validate description with error handling
            try:
                # SUPER AGGRESSIVE: Remove ALL asterisks - no exceptions
                description = description.replace('*', '')
                
                # Remove other formatting characters
                description = description.replace('_', '').replace('`', '')
                
                # Convert any remaining dash bullets to circle bullets
                lines = description.split('\n')
                cleaned_lines = []
                for line in lines:
                    try:
                        line = str(line).strip()
                        if line.startswith('- '):
                            line = '• ' + line[2:]
                        elif line.startswith('-'):
                            line = '• ' + line[1:]
                        cleaned_lines.append(line)
                    except Exception as e:
                        logger.warning(f"Error processing line in validation: {str(e)}")
                        cleaned_lines.append(str(line).strip().replace('*', ''))
                
                description = '\n'.join(cleaned_lines)
                
                # Remove any markdown formatting that might have been missed
                try:
                    description = re.sub(r'\*\*(.*?)\*\*', r'\1', description)
                    description = re.sub(r'\*(.*?)\*', r'\1', description)
                    description = re.sub(r'__(.*?)__', r'\1', description)
                    description = re.sub(r'_(.*?)_', r'\1', description)
                except Exception as e:
                    logger.warning(f"Regex error in validation: {str(e)}")
                
                # FINAL ASTERISK CHECK - Remove any that might have been added back
                description = description.replace('*', '')
                
            except Exception as e:
                logger.error(f"Error in description cleaning: {str(e)}")
                # Fallback to basic cleaning
                description = str(description).replace('*', '').replace('_', '').strip()
            
            # Enhanced accuracy validation with error handling
            try:
                product_lower = product_name.lower()
                description_lower = description.lower()
                
                # Check if description is relevant to the product
                product_keywords = product_lower.split()
                relevance_score = sum(1 for keyword in product_keywords if keyword and keyword in description_lower)
                
                if relevance_score == 0:
                    # Add product context if missing
                    try:
                        if description_type == 'short':
                            description = f"• {product_name} - " + description.lstrip('•').strip()
                        else:
                            description = f"{product_name} " + description.lower()
                            if description:
                                description = description[0].upper() + description[1:]
                    except Exception as e:
                        logger.warning(f"Error adding product context: {str(e)}")
                        
            except Exception as e:
                logger.warning(f"Error in relevance validation: {str(e)}")
                
            return self._format_description_safely(description, description_type, product_name)
            
        except Exception as e:
            logger.error(f"Critical error in validation for {product_name}: {str(e)}")
            # Return safe fallback
            return str(description).replace('*', '').strip() if description else ""
    
    def _format_description_safely(self, description: str, description_type: str, product_name: str) -> str:
        """
        Safely format description based on type with comprehensive error handling.
        
        Args:
            description (str): Description to format
            description_type (str): 'short' or 'long'
            product_name (str): Product name for context
            
        Returns:
            str: Safely formatted description
        """
        try:
            if not description or not isinstance(description, str):
                logger.warning(f"Empty description in formatting for {product_name}")
                return ""
                
            description = str(description).strip()
            
            if description_type == 'short':
                return self._format_short_description_safely(description, product_name)
            elif description_type == 'long':
                return self._format_long_description_safely(description, product_name)
            else:
                logger.error(f"Invalid description type in formatting: {description_type}")
                return description.replace('*', '')
                
        except Exception as e:
            logger.error(f"Error in format_description_safely for {product_name}: {str(e)}")
            return str(description).replace('*', '').strip() if description else ""
    
    def _format_short_description_safely(self, description: str, product_name: str) -> str:
        """
        Safely format short description with HTML bullet points.
        Converts plain text or any format to proper HTML <ul><li> structure.
        """
        try:
            # Check if already in HTML format
            if '<ul>' in description and '<li>' in description:
                # Already HTML, just clean it up
                cleaned = description.strip()
                # Remove any asterisks or markdown
                cleaned = cleaned.replace('*', '')
                # Ensure proper structure
                if not cleaned.startswith('<ul>'):
                    cleaned = '<ul>\n' + cleaned
                if not cleaned.endswith('</ul>'):
                    cleaned = cleaned + '\n</ul>'
                return cleaned
            
            # Parse plain text bullets into HTML
            lines = description.split('\n')
            bullet_items = []
            
            for line in lines:
                try:
                    line = str(line).strip()
                    if line:
                        # Remove any markdown/formatting
                        line = line.replace('*', '')
                        line = line.replace('_', '')
                        
                        # Remove existing bullet characters
                        if line.startswith('•'):
                            line = line[1:].strip()
                        elif line.startswith('-'):
                            line = line[1:].strip()
                        elif line.startswith('*'):
                            line = line[1:].strip()
                        
                        # Remove trailing punctuation
                        if line.endswith('.') or line.endswith(','):
                            line = line[:-1]
                        
                        # Skip HTML tags if already present
                        if line.startswith('<ul>') or line.startswith('</ul>') or line.startswith('<li>') or line.startswith('</li>'):
                            continue
                        
                        if line.strip():
                            bullet_items.append(line.strip())
                            
                except Exception as e:
                    logger.warning(f"Error processing bullet line: {str(e)}")
                    continue
            
            # MANDATORY: Ensure EXACTLY 4 bullet points
            if len(bullet_items) < 4:
                logger.warning(f"Insufficient bullet points for {product_name}: {len(bullet_items)}, padding to 4")
                # Pad with neutral, non-claim bullets
                while len(bullet_items) < 4:
                    if len(bullet_items) == 0:
                        bullet_items.append("Contains quality ingredients")
                    elif len(bullet_items) == 1:
                        bullet_items.append("Formulated for personal use")
                    elif len(bullet_items) == 2:
                        bullet_items.append("Part of a balanced routine")
                    else:
                        bullet_items.append("Available in convenient form")
            
            # STRICT: Take EXACTLY first 4 bullet points
            if len(bullet_items) > 4:
                logger.warning(f"Too many bullet points for {product_name}: {len(bullet_items)}, trimming to 4")
                bullet_items = bullet_items[:4]
            
            # Build HTML list
            html_list = "<ul>\n"
            for item in bullet_items:
                html_list += f"<li>{item}</li>\n"
            html_list += "</ul>"
            
            return html_list
                
        except Exception as e:
            logger.error(f"Critical error in short description formatting: {str(e)}")
            # Return safe HTML fallback
            safe_text = str(description).replace('*', '').strip()
            return f"<ul>\n<li>{safe_text}</li>\n</ul>"
    
    def _format_long_description_safely(self, description: str, product_name: str) -> str:
        """
        Safely format long description and ensure exactly 7-8 sentences.
        """
        try:
            # Remove all asterisks and basic formatting
            description = description.replace('*', '').replace('_', '').strip()
            
            # Split into sentences (handling multiple punctuation)
            sentences = [s.strip() for s in description.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            
            # MANDATORY: Ensure exactly 7-8 sentences
            if len(sentences) < 7:
                logger.warning(f"Long description has only {len(sentences)} sentences for {product_name}, padding to 7")
                # Add generic padding sentences
                padding_sentences = [
                    "This product is designed for quality and efficacy",
                    "It follows pharmaceutical manufacturing standards",
                    "Store in a cool, dry place away from direct sunlight",
                    "Keep out of reach of children",
                    "Use as directed on the product label",
                    "Consult a healthcare professional if you have questions",
                    "Read the label carefully before use"
                ]
                while len(sentences) < 7:
                    sentences.append(padding_sentences[len(sentences) % len(padding_sentences)])
            
            elif len(sentences) > 8:
                logger.warning(f"Long description has {len(sentences)} sentences for {product_name}, trimming to 8")
                sentences = sentences[:8]
            
            # Enhanced product name integration BEFORE reconstruction
            try:
                if product_name and product_name.lower() not in ' '.join(sentences).lower():
                    # Add product name to first sentence if missing
                    if sentences and sentences[0]:
                        first_sentence = sentences[0].strip()
                        sentences[0] = f"{product_name} {first_sentence[0].lower()}{first_sentence[1:]}"
            except Exception as e:
                logger.warning(f"Error integrating product name: {str(e)}")
            
            # Reconstruct description with exactly 7-8 sentences
            description = '. '.join(sentences)
            if not description.endswith('.'):
                description += '.'
            
            # Capitalize first letter
            if description:
                description = description[0].upper() + description[1:]
            
            # FINAL ASTERISK REMOVAL - Make absolutely sure no asterisks remain (including unicode)
            description = str(description).replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
            if '*' in description or '＊' in description or '﹡' in description or '∗' in description:
                logger.warning('Asterisk detected in cleaned text! Forcing removal.')
                description = description.replace('*', '').replace('＊', '').replace('﹡', '').replace('∗', '')
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error in long description formatting: {str(e)}")
            return str(description).replace('*', '').strip()
    
    def _get_prompt(self, product_name: str, description_type: str, category: Optional[str] = None, product_data: Optional[Dict[str, Any]] = None) -> str:
        try:
            if not product_name or not isinstance(product_name, str):
                return ""
            
            product_name = str(product_name).strip()
            if not product_name:
                return ""

            size = product_data.get('size', 'N/A') if product_data else 'N/A'
            mrp = product_data.get('mrp', 'N/A') if product_data else 'N/A'
            
            if description_type == 'short':
                return f"""[SHORT DESCRIPTION FORMAT]

Generate exactly 4 bullet points in valid HTML <ul><li>...</li></ul> format for the product: {product_name}
Packaging Size: {size}
Maximum Retail Price (MRP): ₹{mrp}

Rules:
- Keep each bullet concise and medically relevant.
- Use natural medicine-related terminology.
- Mention medicine category, usage intent, packaging, and safety.
- Do not use exaggerated claims.

Structure:
- Bullet 1: Mention medicine category and general usage purpose.
- Bullet 2: Mention dosage form and packaging size.
- Bullet 3: Mention pricing or availability naturally.
- Bullet 4: Mention storage or safety guidance.

Example Output:
<ul>
<li>Anti-allergic medicine commonly used for respiratory and seasonal allergy management.</li>
<li>Available in tablet form with packaging of 10 tablets per strip.</li>
<li>Economically priced pharmaceutical preparation suitable for prescribed use.</li>
<li>Store in a cool and dry place away from direct sunlight and moisture.</li>
</ul>"""

            elif description_type == 'long':
                return f"""[LONG DESCRIPTION FORMAT]

Generate a professional medicine product description for '{product_name}' in exactly 4 sentences.
Packaging Size: {size}
Maximum Retail Price (MRP): ₹{mrp}

Rules:
- Keep the language professional and medically relevant.
- Avoid repetitive wording and filler content.
- Do not make promotional or misleading claims.
- Ensure the content sounds natural and readable.
- Mention medicine category, packaging, pricing context, and safety guidance naturally.

Structure:
- Sentence 1: Introduce the medicine category and general usage purpose.
- Sentence 2: Mention dosage form, packaging quantity, and pricing/value context.
- Sentence 3: Mention storage, handling, or usage recommendation.
- Sentence 4: Mandatory disclaimer: "Please consult a doctor before consumption."

Example Output:
AB Flo SR 100mg Tablet is a respiratory medicine commonly prescribed for conditions associated with mucus buildup and breathing discomfort. It is available in sustained-release tablet form with a packaging size of 10 tablets and is offered at an economical price point for prescribed treatment. Store the medicine in a cool, dry place away from moisture and direct sunlight, and use it only as directed by a healthcare professional. Please consult a doctor before consumption."""

            return ""
        except Exception as e:
            logger.error(f"Error generating prompt for {product_name}: {str(e)}")
            return ""
    
    async def _call_mistral(self, prompt: str, max_retries: int = 2, model_override: Optional[str] = None) -> str:
        """
        Call Mistral/OpenRouter API with enhanced retry logic and exponential backoff.
        Allows model override for other OpenRouter models.
        """
        # OpenRouter requires specific headers for proper authentication
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ManishKhandve/PharmaDescription-Generator",  # Required by OpenRouter
            "X-Title": "Pharma Description Generator"  # Optional but recommended
        }
        model_id = model_override if model_override else self.mistral_model
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a highly cautious and professional pharmaceutical data formatter. Your task is to generate safe, SEO-friendly Short Descriptions and Long Descriptions for pharmacy inventory management. CRITICAL SAFETY RULES: 1. You will be provided with a product Name, Size, and MRP. 2. DO NOT invent, guess, or hallucinate active ingredients, chemical salts, specific medical uses, or side effects. If you do not know the exact composition, remain completely silent about it. 3. Keep the language entirely focused on the packaging format, commercial availability, and the general category of the item. 4. Do not make any therapeutic claims. Use neutral phrasing such as 'commercially available formulation.' 5. Always maintain a neutral, objective, and professional tone."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1200,  # Increased for longer descriptions
            "temperature": 0.02,  # Further reduced for maximum accuracy and consistency
            "top_p": 0.9,        # Increased for better coverage
            "frequency_penalty": 0.3,  # Increased to reduce repetition
            "presence_penalty": 0.2   # Fine-tuned for diverse, accurate content
        }
        
        for attempt in range(max_retries):
            try:
                # Log attempt for debugging (without exposing API key)
                logger.debug(f"OpenRouter API call attempt {attempt + 1}/{max_retries} to model: {model_id}")
                
                # Create client with SSL verification enabled but with fallback options
                # This helps with certificate issues on some systems
                async with httpx.AsyncClient(
                    timeout=30.0,
                    verify=True,  # SSL verification enabled for security
                    follow_redirects=True
                ) as client:
                    response = await client.post(
                        self.mistral_api_url,
                        json=payload,
                        headers=headers
                    )
                    
                    # Log response status for debugging
                    logger.debug(f"OpenRouter response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        raw_content = result["choices"][0]["message"]["content"].strip()
                        return self._clean_response(raw_content)
                    elif response.status_code == 401:  # Unauthorized - Invalid API key
                        try:
                            error_json = response.json()
                            error_detail = error_json.get('error', {}).get('message', response.text)
                        except Exception:
                            error_detail = response.text
                        
                        logger.error(f"OpenRouter authentication failed (401): {error_detail}")
                        
                        # Check if it's truly an invalid key or other auth issue
                        if 'invalid' in error_detail.lower() or 'not found' in error_detail.lower():
                            raise ValueError(
                                "Invalid OpenRouter API key. "
                                "Please check your key at https://openrouter.ai/keys"
                            )
                        else:
                            # Might be temporary auth issue, allow retry
                            logger.warning(f"Auth error but may be temporary: {error_detail}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(3)
                                continue
                            raise ValueError(f"OpenRouter authentication failed: {error_detail[:200]}")
                    elif response.status_code == 403:  # Forbidden
                        try:
                            error_json = response.json()
                            error_detail = error_json.get('error', {}).get('message', response.text)
                        except Exception:
                            error_detail = response.text
                        
                        logger.error(f"OpenRouter access forbidden (403): {error_detail}")
                        raise ValueError(
                            "OpenRouter access forbidden. "
                            "Check your API key permissions at https://openrouter.ai/keys"
                        )
                    elif response.status_code == 429:  # Rate limit
                        # Enhanced rate limiting strategy
                        wait_time = min(60, (2 ** attempt) + (attempt * 2))  # Cap at 60 seconds
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}), waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status_code == 402:  # Payment required / Insufficient credits
                        error_detail = response.text
                        logger.error(f"OpenRouter insufficient credits: {error_detail}")
                        raise ValueError(f"Insufficient credits or billing issue: {error_detail[:200]}")
                    elif response.status_code in [502, 503, 504]:  # Server errors
                        wait_time = 5 + (attempt * 2)
                        logger.warning(f"Server error {response.status_code}, retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_detail = response.text
                        logger.error(f"Mistral API error: {response.status_code} - {error_detail}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        # For unknown errors, provide helpful message
                        if response.status_code >= 400:
                            raise ValueError(f"API error {response.status_code}: {error_detail[:200]}")
                        return ""
                        
            except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                logger.warning(f"Timeout error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 5 + (attempt * 3)
                    await asyncio.sleep(wait_time)
                    continue
                return ""
            except httpx.ConnectError as e:
                logger.error(f"Connection error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 5 + (attempt * 2)
                    await asyncio.sleep(wait_time)
                    continue
                # Provide helpful error message for SSL/connection issues
                raise ValueError(f"Cannot connect to OpenRouter API. Please check: 1) Internet connection, 2) Firewall/proxy settings, 3) System certificates are up to date. Error: {str(e)[:150]}")
            except httpx.RequestError as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 3 + (attempt * 2)
                    await asyncio.sleep(wait_time)
                    continue
                raise ValueError(f"Network request failed. Check internet connection and firewall settings. Error: {str(e)[:150]}")
            except Exception as e:
                # Re-raise ValueError if retries are exhausted so the app knows the API test failed
                if isinstance(e, ValueError) and attempt == max_retries - 1:
                    raise e
                    
                logger.error(f"Mistral API call failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 3 + (attempt * 2)
                    await asyncio.sleep(wait_time)
                    continue
                return ""
        
        logger.error(f"Failed to get response from Mistral after {max_retries} attempts")
        return ""
    
    async def _call_gemini(self, prompt: str, max_retries: int = 2) -> str:
        """
        Enhanced Gemini API call with improved accuracy settings and better error handling.
        
        Args:
            prompt (str): Prompt to send to Gemini
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Generated text or empty string if failed
        """
        for attempt in range(max_retries):
            try:
                # Enhanced prompt with context for better accuracy
                enhanced_prompt = (
                    f"As a pharmaceutical content specialist, provide a highly accurate, "
                    f"professional response to the following request:\n\n{prompt}\n\n"
                    f"Important: Use precise medical terminology, focus on factual benefits, "
                    f"and ensure content is suitable for healthcare professionals and patients."
                    f"normal people should also be able to understand it."
                )
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.gemini_model.generate_content(enhanced_prompt)
                )
                
                if response.text:
                    raw_content = response.text.strip()
                    return self._clean_response(raw_content)
                else:
                    logger.warning("Gemini returned empty response")
                    return ""
                    
            except Exception as e:
                error_str = str(e).lower()
                full_error = str(e)
                
                # Detailed error classification
                if "api key not valid" in error_str or "api_key_invalid" in error_str:
                    logger.error(f"Gemini API key is invalid: {full_error}")
                    raise ValueError(f"Invalid API key: {full_error}")
                elif "api key" in error_str and ("not found" in error_str or "invalid" in error_str):
                    logger.error(f"Gemini API key error: {full_error}")
                    raise ValueError(f"API key error: {full_error}")
                elif "quota" in error_str or "resource_exhausted" in error_str:
                    logger.warning(f"Gemini quota exhausted: {full_error}")
                    if attempt < max_retries - 1:
                        wait_time = 3 ** attempt  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise ValueError(f"Quota exhausted: {full_error}")
                elif "rate" in error_str or "429" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(f"Gemini rate limit hit, waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                elif "permission" in error_str or "403" in error_str:
                    logger.error(f"Gemini permission denied: {full_error}")
                    raise ValueError(f"Permission denied: {full_error}")
                elif "not found" in error_str or "404" in error_str:
                    logger.error(f"Gemini model not found: {full_error}")
                    raise ValueError(f"Model not found: {full_error}")
                elif "timeout" in error_str or "deadline" in error_str:
                    logger.warning(f"Gemini timeout (attempt {attempt + 1}): {full_error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return ""
                elif "ssl" in error_str or "certificate" in error_str or "connection" in error_str:
                    logger.error(f"Gemini connection error (attempt {attempt + 1}): {full_error}")
                    if attempt < max_retries - 1:
                        wait_time = 5 + (attempt * 2)
                        await asyncio.sleep(wait_time)
                        continue
                    raise ValueError(f"Cannot connect to Gemini API. Please check: 1) Internet connection, 2) Firewall/proxy settings, 3) System certificates are up to date. Error: {full_error[:150]}")
                else:
                    logger.error(f"Gemini API call failed (attempt {attempt + 1}): {full_error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return ""
        
        return ""


class BatchProcessor:
    """
    Enhanced batch processor optimized for large-scale pharmaceutical product processing.
    Includes intelligent rate limiting, caching, and error recovery.
    """
    
    def __init__(self, llm_client: LLMClient, batch_size: int = 10):  # Optimized for 10 concurrent cloud API requests
        """
        Initialize batch processor with optimized settings for large datasets.
        
        Args:
            llm_client (LLMClient): LLM client instance
            batch_size (int): Number of products to process concurrently (10 for cloud models)
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.cache: Dict[str, Any] = {}  # Simple in-memory cache for repeated products
        self.rate_limiter = RateLimiter(base_delay=0.1, max_delay=5.0)  # Optimized delays for cloud models
        
    async def process_products(self, products: list, progress_callback=None, stop_check=None, existing_results=None) -> list:
        """
        Process list of products in batches with enhanced progress tracking and error recovery.
        Optimized for large datasets (10,000+ products).
        
        Args:
            products (list): List of product names
            progress_callback (callable): Optional callback for progress updates
            stop_check (callable): Optional function to check if processing should stop
            existing_results (dict): Optional dictionary of existing results for retry context
            
        Returns:
            list: List of dictionaries with product info and descriptions
        """
        results: List[Dict[str, Any]] = []
        total_products = len(products)
        failed_products = []
        
        logger.info(f"Starting batch processing of {total_products} products with batch size {self.batch_size}")
        
        for i in range(0, total_products, self.batch_size):
            # Check for stop request before processing each batch
            if stop_check and stop_check():
                logger.info(f"Processing stopped by user request at {len(results)}/{total_products}")
                break
                
            batch = products[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: products {i+1}-{min(i+len(batch), total_products)}")
            
            # Apply rate limiting before each batch
            await self.rate_limiter.wait_if_needed()
            
            batch_results = await self._process_batch_with_retry(batch)
            
            # Separate successful and failed results with better logging
            batch_successful = 0
            batch_failed = 0
            for result in batch_results:
                if not isinstance(result, dict):
                    batch_failed += 1
                    logger.warning(f"Unexpected result type encountered: {type(result)}")
                    continue

                original_product = result.pop('original_product', None)
                product_name = result.get('product_name', 'Unknown')

                if result.get('status') == 'failed':
                    batch_failed += 1
                    logger.debug(f"Product failed: {product_name} - {result.get('error', 'Unknown error')}")

                    if original_product:
                        failed_products.append(original_product)
                    else:
                        failed_products.append({'product_name': product_name})

                    # Ensure failed results are still tracked for downstream processing
                    results.append(result)
                else:
                    results.append(result)
                    batch_successful += 1
            
            logger.info(f"Batch {i//self.batch_size + 1} completed: {batch_successful} successful, {batch_failed} failed")
            
            # Update progress with detailed info
            if progress_callback:
                progress = min(100, int((i + len(batch)) / total_products * 100))
                success_count = len(results)
                failed_count = len(failed_products)
                # Try to call with failed count, fall back to 3 args if not supported
                try:
                    progress_callback(progress, success_count, total_products, failed_count)
                except TypeError:
                    # Callback doesn't support failed_count parameter
                    progress_callback(progress, success_count, total_products)
            
            # Adaptive delay based on API performance - reduced for speed
            await asyncio.sleep(self._calculate_delay(len(failed_products), len(results)))
            
        # Skip retry of failed products for faster processing - only retry critical failures
        if failed_products and len(failed_products) <= 10:  # Reduced retry threshold
            logger.info(f"Retrying {len(failed_products)} critical failed products...")
            retry_results = await self._retry_failed_products_fast(failed_products[:10], stop_check)
            results.extend(retry_results)
        
        total_success = sum(1 for r in results if isinstance(r, dict) and r.get('status') != 'failed')
        total_failed = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'failed')
        logger.info(f"Batch processing complete: {total_success} successful, {total_failed} failed")
        return results
        
    def _calculate_delay(self, failed_count: int, success_count: int) -> float:
        """Calculate adaptive delay based on success/failure ratio - optimized for cloud models."""
        if failed_count == 0:
            return 0.05  # Minimal delay for cloud models when everything works
        
        failure_ratio = failed_count / max(success_count, 1)
        if failure_ratio > 0.5:
            return 0.5  # Shorter delay for cloud models
        elif failure_ratio > 0.2:
            return 0.3  # Reduced medium delay
        else:
            return 0.1  # Very short delay for cloud models
    
    async def _retry_failed_products_fast(self, failed_products: list, stop_check=None) -> list:
        """
        Fast retry for critically failed products with minimal overhead.
        
        Args:
            failed_products (list): List of failed product dictionaries
            stop_check (callable): Function to check for stop request
            
        Returns:
            list: Results from retry attempts
        """
        retry_results = []
        
        for product in failed_products[:10]:  # Limit retries
            if stop_check and stop_check():
                break
                
            try:
                # Single retry with timeout optimized for cloud models
                result = await asyncio.wait_for(
                    self._process_single_product_fast(product),
                    timeout=60.0  # 1 minute timeout for cloud models
                )
                retry_results.append(result)
                
            except Exception as e:
                product_name = product.get('product_name', 'Unknown') if isinstance(product, dict) else str(product)
                logger.debug(f"Retry failed for {product_name}: {str(e)}")
                # Don't add failed retries to avoid duplicates
                continue
                
            # Minimal delay between retries for cloud models
            await asyncio.sleep(0.05)
        
        return retry_results
    
    async def _process_batch_with_retry(self, batch: list) -> list:
        """
        Optimized batch processing with parallel execution and smart retry logic.
        
        Args:
            batch (list): List of product names to process
            
        Returns:
            list: Processed results for the batch
        """
        batch_results = []
        
        # Create all tasks for parallel execution
        tasks = []
        for product in batch:
            # product is a dict: {"product_name": ..., "category": ...}
            cache_key = product["product_name"].lower().strip()
            if cache_key in self.cache:
                cached_result = dict(self.cache[cache_key])
                cached_result['original_product'] = product
                batch_results.append(cached_result)
                continue

            # Create task for this product
            task = self._process_single_product_fast(product)
            tasks.append((product, task))
        
        # Process remaining tasks in parallel
        if tasks:
            try:
                # Run all tasks concurrently with timeout
                task_results = await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                    timeout=300.0  # 5 minutes timeout for batch processing
                )
                
                # Process results
                for i, (product, _) in enumerate(tasks):
                    if i < len(task_results):
                        result = task_results[i]
                        if isinstance(result, Exception):
                            # Handle failed products
                            result_dict = {
                                'product_name': product.get('product_name', str(product)),
                                'short_description': '',
                                'long_description': '',
                                'status': 'failed',
                                'error': str(result)
                            }
                        elif isinstance(result, dict):
                            result_dict = dict(result)
                        else:
                            result_dict = {
                                'product_name': product.get('product_name', str(product)),
                                'short_description': '',
                                'long_description': '',
                                'status': 'failed',
                                'error': f"Unexpected result type: {type(result)}"
                            }

                        # Cache successful results without original metadata
                        if result_dict.get('status', 'success') != 'failed':
                            cache_key_product = product.get('product_name', str(product)).lower().strip()
                            self.cache[cache_key_product] = {
                                'product_name': result_dict.get('product_name', product.get('product_name', str(product))),
                                'short_description': result_dict.get('short_description', ''),
                                'long_description': result_dict.get('long_description', ''),
                                'status': result_dict.get('status', 'success')
                            }

                        result_dict['original_product'] = product
                        batch_results.append(result_dict)
                
            except asyncio.TimeoutError:
                logger.warning(f"Batch timeout for {len(tasks)} products, creating empty results")
                # Create failed results for timeout
                for product, _ in tasks:
                    batch_results.append({
                        'product_name': product.get('product_name', str(product)),
                        'short_description': '',
                        'long_description': '',
                        'status': 'failed',
                        'error': 'Timeout',
                        'original_product': product
                    })
        
        return batch_results
    
    async def _process_single_product_fast(self, product_info: dict) -> Dict[str, Any]:
        """
        Fast single product processing with optimized concurrent description generation.
        
        Args:
            product_info (dict): Dictionary containing product details
            
        Returns:
            dict: Product information with generated descriptions
        """
        product_name = "Unknown"
        try:
            product_name = product_info.get("product_name", "").strip()
            category = product_info.get("category", None)
            
            # Skip empty product names
            if not product_name:
                logger.warning("Empty product name encountered, skipping")
                return {
                    'product_name': '',
                    'short_description': '',
                    'long_description': '',
                    'status': 'failed',
                    'error': 'Empty product name'
                }
            
            # Generate both descriptions concurrently for speed with rich product data
            short_task = self.llm_client.generate_description(product_name, 'short', category, product_info)
            long_task = self.llm_client.generate_description(product_name, 'long', category, product_info)

            # Wait for both with timeout optimized for cloud models
            short_desc, long_desc = await asyncio.wait_for(
                asyncio.gather(short_task, long_task, return_exceptions=True),
                timeout=60.0  # 1 minute timeout for cloud models (both descriptions)
            )
            
            # Handle any exceptions in description generation
            if isinstance(short_desc, Exception):
                logger.warning(f"Short description failed for {product_name}: {short_desc}")
                short_desc = ''
            
            if isinstance(long_desc, Exception):
                logger.warning(f"Long description failed for {product_name}: {long_desc}")
                long_desc = ''
            
            # Clean and validate descriptions using the improved cleaning method
            short_desc = str(short_desc).strip() if short_desc else ''
            long_desc = str(long_desc).strip() if long_desc else ''
            
            # Apply the same thorough cleaning as the main method
            if short_desc:
                short_desc = self.llm_client._clean_response(short_desc)
                
                # Ensure bullet format for short descriptions
                lines = short_desc.split('\n')
                bullet_lines = []
                for line in lines:
                    line = line.strip()
                    if line:
                        # Remove any asterisks first
                        line = line.replace('*', '')
                        
                        # Add bullet if missing
                        if not line.startswith('•'):
                            line = '• ' + line
                        
                        bullet_lines.append(line)
                short_desc = '\n'.join(bullet_lines)
            
            if long_desc:
                # Apply thorough cleaning to long description
                long_desc = self.llm_client._clean_response(long_desc)
                # EXTRA asterisk removal for long descriptions
                long_desc = long_desc.replace('*', '')
                
                # REMOVE bullets from long descriptions (convert to paragraphs)
                lines = long_desc.split('\n')
                paragraph_lines = []
                for line in lines:
                    line = line.strip()
                    if line:
                        # Remove bullet points
                        if line.startswith('• '):
                            line = line[2:]
                        elif line.startswith('•'):
                            line = line[1:].strip()
                        elif line.startswith('- '):
                            line = line[2:]
                        elif line.startswith('-'):
                            line = line[1:].strip()
                        
                        # Remove asterisks
                        line = line.replace('*', '')
                        paragraph_lines.append(line)
                
                long_desc = '\n'.join(paragraph_lines)
            
            result = {
                'product_name': product_name,
                'short_description': short_desc,
                'long_description': long_desc,
                'status': 'success' if (short_desc or long_desc) else 'failed'
            }
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Product processing timeout: {product_name}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed',
                'error': 'Processing timeout'
            }
        except Exception as e:
            logger.error(f"Error processing product {product_name}: {str(e)}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed',
                'error': str(e)
            }
    
    async def _process_batch(self, batch: list) -> list:
        """
        Process a single batch of products concurrently.
        
        Args:
            batch (list): List of product names to process
            
        Returns:
            list: Processed results for the batch
        """
        tasks = []
        for product in batch:
            tasks.append(self._process_single_product(product))
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_product_enhanced(self, product_name: str) -> Dict[str, Any]:
        """
        Enhanced single product processing with better error handling and validation.
        
        Args:
            product_name (str): Name of the product
            
        Returns:
            dict: Product information with generated descriptions
        """
        # Check cache first
        cache_key = product_name.lower().strip()
        if cache_key in self.cache:
            logger.debug(f"Using cached result for: {product_name}")
            return self.cache[cache_key]
        
        try:
            # Generate descriptions with timeout and validation
            short_desc = await asyncio.wait_for(
                self.llm_client.generate_description(product_name, 'short'),
                timeout=120.0  # 2 minute timeout
            )
            
            long_desc = await asyncio.wait_for(
                self.llm_client.generate_description(product_name, 'long'),
                timeout=180.0  # 3 minute timeout for longer descriptions
            )
            
            # Validate descriptions
            short_valid = short_desc and len(short_desc.strip()) > 10
            long_valid = long_desc and len(long_desc.strip()) > 20
            
            # Determine status
            if short_valid and long_valid:
                status = 'success'
                self.rate_limiter.record_success()
            elif short_valid or long_valid:
                status = 'partial'
            else:
                status = 'failed'
                logger.warning(f"Both descriptions failed for: {product_name}")
            
            result = {
                'product_name': product_name,
                'short_description': short_desc if short_valid else '',
                'long_description': long_desc if long_valid else '',
                'status': status
            }
            
            # Cache successful and partial results
            if status in ['success', 'partial']:
                self.cache[cache_key] = result
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing {product_name}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed'
            }
        except Exception as e:
            logger.error(f"Error processing {product_name}: {str(e)}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed'
            }

    async def _process_single_product(self, product_name: str) -> Dict[str, Any]:
        """
        Process a single product and generate both descriptions.
        
        Args:
            product_name (str): Name of the product
            
        Returns:
            dict: Product information with generated descriptions
        """
        # Check cache first
        cache_key = product_name.lower().strip()
        if cache_key in self.cache:
            logger.info(f"Using cached result for: {product_name}")
            return self.cache[cache_key]
        
        try:
            # Generate both short and long descriptions concurrently
            short_task = self.llm_client.generate_description(product_name, 'short')
            long_task = self.llm_client.generate_description(product_name, 'long')
            
            short_desc, long_desc = await asyncio.gather(short_task, long_task)
            
            result = {
                'product_name': product_name,
                'short_description': short_desc,
                'long_description': long_desc,
                'status': 'success' if short_desc and long_desc else 'partial'
            }
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {product_name}: {str(e)}")
            return {
                'product_name': product_name,
                'short_description': '',
                'long_description': '',
                'status': 'failed'
            }
    
    async def _retry_failed_products(self, failed_products: list, stop_check=None) -> list:
        """
        Retry failed products individually with more conservative approach.
        
        Args:
            failed_products (list): List of failed product dictionaries
            stop_check (callable): Optional function to check if processing should stop
            
        Returns:
            list: Results from retry attempts
        """
        retry_results = []
        
        for product in failed_products[:50]:  # Limit retries
            if stop_check and stop_check():
                break
                
            # Extract product name from dict or use as-is if string
            if isinstance(product, dict):
                product_name = product.get('product_name', str(product))
            else:
                product_name = str(product)
                
            if not product_name or product_name == 'Unknown':
                continue
                
            # Wait longer between individual retries
            await asyncio.sleep(2.0)
            
            try:
                result = await self._process_single_product_enhanced(product_name)
                if result.get('status') != 'failed':
                    retry_results.append(result)
                    logger.info(f"Successfully retried: {product_name}")
                else:
                    logger.warning(f"Retry failed for: {product_name}")
            except Exception as e:
                logger.error(f"Exception during retry for {product_name}: {str(e)}")
                
        return retry_results
