"""
Real-world Applications
Practical implementations of transformer models across various domains,
including chatbots, translation, code generation, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import asyncio
import json
import hashlib
import redis
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading
import queue
import aiohttp
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration Classes
@dataclass
class ApplicationConfig:
    """Base configuration for applications."""
    model_name: str = "transformer-base"
    max_sequence_length: int = 512
    batch_size: int = 32
    cache_enabled: bool = True
    cache_ttl: int = 3600
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    
    
@dataclass
class ChatbotConfig(ApplicationConfig):
    """Configuration for chatbot applications."""
    persona: str = "helpful assistant"
    context_window: int = 4096
    max_history: int = 50
    temperature: float = 0.7
    enable_tools: bool = False
    safety_filter: bool = True


# Base Application Class
class TransformerApplication(ABC):
    """Base class for transformer applications."""
    
    def __init__(self, model: nn.Module, config: ApplicationConfig):
        self.model = model
        self.config = config
        self.cache = self._initialize_cache() if config.cache_enabled else None
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.metrics = MetricsCollector()
        
    def _initialize_cache(self):
        """Initialize caching system."""
        return RedisCache(ttl=self.config.cache_ttl)
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output."""
        pass
        
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point with common processing."""
        start_time = time.time()
        
        try:
            # Rate limiting
            if not self.rate_limiter.allow(input_data.get('user_id', 'anonymous')):
                return {'error': 'Rate limit exceeded', 'status': 429}
                
            # Check cache
            cache_key = self._generate_cache_key(input_data)
            if self.cache and (cached := self.cache.get(cache_key)):
                self.metrics.record('cache_hit', 1)
                return cached
                
            # Process request
            result = self.process(input_data)
            
            # Cache result
            if self.cache and result.get('status', 200) == 200:
                self.cache.set(cache_key, result)
                
            # Record metrics
            self.metrics.record('latency', time.time() - start_time)
            self.metrics.record('success', 1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.metrics.record('error', 1)
            return {'error': str(e), 'status': 500}
            
    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key from input."""
        key_data = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()


# Conversational AI Implementation
class ConversationalAI(TransformerApplication):
    """Production-ready conversational AI system."""
    
    def __init__(self, model: nn.Module, config: ChatbotConfig):
        super().__init__(model, config)
        self.config = config
        self.conversation_manager = ConversationManager()
        self.safety_filter = SafetyFilter() if config.safety_filter else None
        self.tools = self._initialize_tools() if config.enable_tools else {}
        
    def _initialize_tools(self) -> Dict[str, 'Tool']:
        """Initialize available tools."""
        return {
            'calculator': CalculatorTool(),
            # 'web_search': WebSearchTool(),  # Not implemented in demo
            # 'weather': WeatherTool(),  # Not implemented in demo
            # 'calendar': CalendarTool()  # Not implemented in demo
        }
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat message."""
        user_message = input_data['message']
        conversation_id = input_data.get('conversation_id', 'default')
        user_id = input_data.get('user_id', 'anonymous')
        
        # Safety check
        if self.safety_filter and not self.safety_filter.is_safe(user_message):
            return {
                'response': "I'm sorry, but I can't respond to that type of message.",
                'status': 200,
                'filtered': True
            }
            
        # Get conversation history
        history = self.conversation_manager.get_history(conversation_id)
        
        # Build context
        context = self._build_context(history, user_message)
        
        # Check for tool use
        if self.config.enable_tools:
            tool_result = self._check_and_use_tools(user_message, context)
            if tool_result:
                response = self._format_tool_response(tool_result)
                self.conversation_manager.add_turn(
                    conversation_id, user_message, response
                )
                return {'response': response, 'status': 200, 'tool_used': True}
                
        # Generate response
        response = self._generate_response(context)
        
        # Update history
        self.conversation_manager.add_turn(conversation_id, user_message, response)
        
        return {
            'response': response,
            'status': 200,
            'conversation_id': conversation_id
        }
        
    def _build_context(self, history: List[Dict], message: str) -> str:
        """Build context from history and current message."""
        context_parts = []
        
        # Add persona
        context_parts.append(f"You are a {self.config.persona}.")
        
        # Add relevant history
        for turn in history[-self.config.max_history:]:
            context_parts.append(f"Human: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")
            
        # Add current message
        context_parts.append(f"Human: {message}")
        context_parts.append("Assistant:")
        
        context = "\n\n".join(context_parts)
        
        # Truncate if needed
        if len(context) > self.config.context_window:
            context = self._truncate_context(context)
            
        return context
        
    def _generate_response(self, context: str) -> str:
        """Generate response using the model."""
        # Tokenize
        inputs = self._tokenize(context)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=self.config.max_sequence_length,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id
            )
            
        # Decode
        response = self._decode(outputs[0])
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
        
    def _check_and_use_tools(self, message: str, context: str) -> Optional[Dict]:
        """Check if tools are needed and use them."""
        # Simple tool detection (in practice, use more sophisticated methods)
        for tool_name, tool in self.tools.items():
            if tool.should_use(message):
                params = tool.extract_params(message)
                result = tool.execute(params)
                return {'tool': tool_name, 'result': result}
                
        return None
        
    def _format_tool_response(self, tool_result: Dict) -> str:
        """Format tool result as response."""
        tool_name = tool_result['tool']
        result = tool_result['result']
        
        if tool_name == 'calculator':
            return f"The calculation result is: {result}"
        elif tool_name == 'weather':
            return f"The weather information: {result}"
        else:
            return f"Here's what I found: {result}"
            
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text (placeholder - use actual tokenizer)."""
        # Simplified tokenization
        tokens = text.split()
        token_ids = [hash(token) % 30000 for token in tokens]
        
        return {
            'input_ids': torch.tensor([token_ids]),
            'attention_mask': torch.ones(1, len(token_ids))
        }
        
    def _decode(self, token_ids: torch.Tensor) -> str:
        """Decode tokens (placeholder - use actual tokenizer)."""
        # Simplified decoding
        return " ".join([f"token_{id}" for id in token_ids.tolist()])
        
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove any remaining special tokens
        response = response.replace("<pad>", "").replace("<eos>", "")
        
        # Trim whitespace
        response = response.strip()
        
        return response
        
    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit window."""
        # Keep most recent parts
        lines = context.split('\n')
        
        # Always keep persona and current message
        truncated = [lines[0]]  # Persona
        truncated.extend(lines[-(self.config.context_window // 100):])
        
        return '\n'.join(truncated)


# Machine Translation System
class TranslationSystem(TransformerApplication):
    """Multi-language translation system."""
    
    def __init__(self, models: Dict[str, nn.Module], config: ApplicationConfig):
        # Use first model for base initialization
        super().__init__(list(models.values())[0], config)
        self.models = models  # {lang_pair: model}
        self.language_detector = LanguageDetector()
        self.quality_estimator = QualityEstimator()
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text."""
        text = input_data['text']
        source_lang = input_data.get('source_lang')
        target_lang = input_data['target_lang']
        
        # Detect source language if not provided
        if not source_lang:
            source_lang = self.language_detector.detect(text)
            
        # Select model
        model_key = f"{source_lang}-{target_lang}"
        if model_key not in self.models:
            # Try reverse direction
            reverse_key = f"{target_lang}-{source_lang}"
            if reverse_key in self.models:
                # Use back-translation
                return self._back_translate(text, source_lang, target_lang)
            else:
                return {
                    'error': f'Translation pair {model_key} not supported',
                    'status': 400
                }
                
        model = self.models[model_key]
        
        # Translate
        translation = self._translate(text, model)
        
        # Estimate quality
        quality_score = self.quality_estimator.estimate(
            text, translation, source_lang, target_lang
        )
        
        return {
            'translation': translation,
            'source_language': source_lang,
            'target_language': target_lang,
            'quality_score': quality_score,
            'status': 200
        }
        
    def _translate(self, text: str, model: nn.Module) -> str:
        """Perform translation."""
        # Preprocess
        processed = self._preprocess_text(text)
        
        # Tokenize
        inputs = self._tokenize(processed)
        
        # Translate
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=self.config.max_sequence_length,
                num_beams=5,
                early_stopping=True
            )
            
        # Decode
        translation = self._decode(outputs[0])
        
        # Postprocess
        translation = self._postprocess_text(translation)
        
        return translation
        
    def _back_translate(self, text: str, source_lang: str, 
                       target_lang: str) -> Dict[str, Any]:
        """Translate through intermediate language."""
        # Find intermediate language (usually English)
        intermediate = 'en'
        
        # First translation
        first_key = f"{source_lang}-{intermediate}"
        if first_key not in self.models:
            return {'error': 'No translation path available', 'status': 400}
            
        intermediate_text = self._translate(text, self.models[first_key])
        
        # Second translation
        second_key = f"{intermediate}-{target_lang}"
        if second_key not in self.models:
            return {'error': 'No translation path available', 'status': 400}
            
        final_translation = self._translate(intermediate_text, self.models[second_key])
        
        return {
            'translation': final_translation,
            'source_language': source_lang,
            'target_language': target_lang,
            'intermediate_language': intermediate,
            'quality_score': 0.8,  # Lower score for back-translation
            'status': 200
        }
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for translation."""
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Handle special characters
        text = text.replace('…', '...')
        
        return text
        
    def _postprocess_text(self, text: str) -> str:
        """Postprocess translated text."""
        # Fix spacing
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
            
        return text


# Code Generation Assistant
class CodeAssistant(TransformerApplication):
    """AI-powered code generation and assistance."""
    
    def __init__(self, model: nn.Module, config: ApplicationConfig):
        super().__init__(model, config)
        self.language_parsers = self._initialize_parsers()
        self.code_validator = CodeValidator()
        self.security_scanner = SecurityScanner()
        
    def _initialize_parsers(self):
        """Initialize language parsers."""
        return {
            'python': PythonParser(),
            'javascript': JavaScriptParser(),
            'java': JavaParser(),
            'cpp': CppParser()
        }
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process code generation request."""
        task = input_data['task']
        
        if task == 'complete':
            return self._complete_code(input_data)
        elif task == 'generate':
            return self._generate_code(input_data)
        elif task == 'explain':
            return self._explain_code(input_data)
        elif task == 'review':
            return self._review_code(input_data)
        elif task == 'test':
            return self._generate_tests(input_data)
        else:
            return {'error': f'Unknown task: {task}', 'status': 400}
            
    def _complete_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete partial code."""
        partial_code = input_data['code']
        language = input_data['language']
        context = input_data.get('context', {})
        
        # Parse code structure
        parser = self.language_parsers.get(language)
        if not parser:
            return {'error': f'Unsupported language: {language}', 'status': 400}
            
        code_structure = parser.parse(partial_code)
        
        # Build prompt
        prompt = self._build_completion_prompt(
            partial_code, code_structure, context, language
        )
        
        # Generate completion
        completion = self._generate_completion(prompt)
        
        # Validate
        full_code = partial_code + completion
        validation = self.code_validator.validate(full_code, language)
        
        # Security scan
        security_issues = self.security_scanner.scan(completion, language)
        
        return {
            'completion': completion,
            'full_code': full_code,
            'validation': validation,
            'security_issues': security_issues,
            'status': 200
        }
        
    def _generate_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code from description."""
        description = input_data['description']
        language = input_data['language']
        requirements = input_data.get('requirements', [])
        
        # Build prompt
        prompt = f"""Generate {language} code for the following:

Description: {description}

Requirements:
{chr(10).join(f'- {req}' for req in requirements)}

Code:"""

        # Generate code
        code = self._generate_from_prompt(prompt)
        
        # Validate
        validation = self.code_validator.validate(code, language)
        
        # Add comments
        commented_code = self._add_comments(code, language)
        
        return {
            'code': commented_code,
            'validation': validation,
            'status': 200
        }
        
    def _explain_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain code functionality."""
        code = input_data['code']
        language = input_data['language']
        detail_level = input_data.get('detail_level', 'medium')
        
        # Build prompt
        prompt = f"""Explain this {language} code:

{code}

Provide a {detail_level} explanation including:
1. What the code does
2. How it works
3. Key concepts used
4. Potential improvements"""

        # Generate explanation
        explanation = self._generate_from_prompt(prompt)
        
        # Extract key points
        key_points = self._extract_key_points(explanation)
        
        return {
            'explanation': explanation,
            'key_points': key_points,
            'complexity': self._assess_complexity(code),
            'status': 200
        }
        
    def _review_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review code and provide feedback."""
        code = input_data['code']
        language = input_data['language']
        
        # Analyze code
        issues = []
        suggestions = []
        
        # Check style
        style_issues = self._check_style(code, language)
        issues.extend(style_issues)
        
        # Check performance
        perf_issues = self._check_performance(code, language)
        issues.extend(perf_issues)
        
        # Check security
        security_issues = self.security_scanner.scan(code, language)
        issues.extend(security_issues)
        
        # Generate suggestions
        for issue in issues:
            suggestion = self._generate_fix_suggestion(issue, code)
            suggestions.append(suggestion)
            
        # Overall assessment
        score = self._calculate_code_score(issues)
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'score': score,
            'summary': self._generate_review_summary(issues, score),
            'status': 200
        }
        
    def _generate_tests(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unit tests for code."""
        code = input_data['code']
        language = input_data['language']
        framework = input_data.get('framework', 'default')
        
        # Parse functions/methods
        parser = self.language_parsers[language]
        functions = parser.extract_functions(code)
        
        tests = []
        for func in functions:
            # Generate test cases
            test_prompt = f"""Generate unit tests for this {language} function using {framework}:

{func['code']}

Include:
- Normal cases
- Edge cases  
- Error cases"""

            test_code = self._generate_from_prompt(test_prompt)
            tests.append({
                'function': func['name'],
                'test_code': test_code
            })
            
        # Combine tests
        combined_tests = self._combine_tests(tests, language, framework)
        
        return {
            'tests': combined_tests,
            'coverage_estimate': self._estimate_coverage(tests, functions),
            'status': 200
        }
        
    def _build_completion_prompt(self, code: str, structure: Dict, 
                               context: Dict, language: str) -> str:
        """Build code completion prompt."""
        prompt_parts = [
            f"Complete the following {language} code:",
            f"\nContext: {context.get('description', 'General code')}",
            f"\nCode:\n{code}",
            "\nContinue the code:"
        ]
        
        return '\n'.join(prompt_parts)
        
    def _generate_completion(self, prompt: str) -> str:
        """Generate code completion."""
        # Tokenize and generate
        inputs = self._tokenize(prompt)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=300,
                temperature=0.2,  # Lower temperature for code
                do_sample=True,
                pad_token_id=self.model.config.pad_token_id
            )
            
        completion = self._decode(outputs[0])
        
        # Clean up
        completion = self._clean_code_output(completion)
        
        return completion
        
    def _generate_from_prompt(self, prompt: str) -> str:
        """Generate code from prompt."""
        return self._generate_completion(prompt)
        
    def _clean_code_output(self, code: str) -> str:
        """Clean generated code."""
        # Remove any markdown formatting
        code = code.replace('```python', '').replace('```', '')
        
        # Fix indentation
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace inconsistencies
            if line.strip():
                cleaned_lines.append(line)
                
        return '\n'.join(cleaned_lines)


# Content Generation System
class ContentGenerator(TransformerApplication):
    """Multi-purpose content generation."""
    
    def __init__(self, model: nn.Module, config: ApplicationConfig):
        super().__init__(model, config)
        self.templates = self._load_templates()
        self.seo_optimizer = SEOOptimizer()
        self.readability_analyzer = ReadabilityAnalyzer()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load content templates."""
        return {
            'blog_post': """Write a blog post about {topic}.

Style: {style}
Target audience: {audience}
Key points to cover: {key_points}

Blog post:""",
            
            'product_description': """Write a compelling product description for {product_name}.

Features: {features}
Benefits: {benefits}
Target customer: {target_customer}

Description:""",
            
            'email': """Write a {tone} email for {purpose}.

Key information: {key_info}
Call to action: {cta}

Email:""",
            
            'social_media': """Create a {platform} post about {topic}.

Tone: {tone}
Include: {requirements}
Character limit: {char_limit}

Post:"""
        }
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process content generation request."""
        content_type = input_data['type']
        
        if content_type == 'article':
            return self._generate_article(input_data)
        elif content_type == 'marketing':
            return self._generate_marketing(input_data)
        elif content_type == 'summary':
            return self._generate_summary(input_data)
        elif content_type == 'social':
            return self._generate_social(input_data)
        else:
            return {'error': f'Unknown content type: {content_type}', 'status': 400}
            
    def _generate_article(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate full article."""
        topic = input_data['topic']
        style = input_data.get('style', 'informative')
        word_count = input_data.get('word_count', 800)
        keywords = input_data.get('keywords', [])
        
        # Research phase (simulate)
        research_points = self._research_topic(topic)
        
        # Generate outline
        outline = self._generate_outline(topic, research_points)
        
        # Generate sections
        sections = []
        current_words = 0
        
        for section in outline:
            section_prompt = f"""Write a section about "{section['title']}" for an article on {topic}.

Key points: {section['points']}
Style: {style}
Approximate length: {(word_count - current_words) // (len(outline) - len(sections))} words

Section:"""

            section_content = self._generate_from_prompt(section_prompt)
            sections.append({
                'title': section['title'],
                'content': section_content
            })
            current_words += len(section_content.split())
            
        # Combine into article
        article = self._combine_sections(sections)
        
        # SEO optimization
        if keywords:
            article = self.seo_optimizer.optimize(article, keywords)
            
        # Analyze readability
        readability = self.readability_analyzer.analyze(article)
        
        return {
            'content': article,
            'outline': outline,
            'word_count': len(article.split()),
            'readability_score': readability['score'],
            'seo_score': self.seo_optimizer.score(article, keywords) if keywords else None,
            'status': 200
        }
        
    def _generate_marketing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate marketing content."""
        content_format = input_data['format']
        
        if content_format == 'email':
            template = self.templates['email']
        elif content_format == 'product':
            template = self.templates['product_description']
        else:
            template = self.templates.get(content_format, self.templates['blog_post'])
            
        # Fill template
        prompt = template.format(**input_data['parameters'])
        
        # Generate content
        content = self._generate_from_prompt(prompt)
        
        # Format for platform
        formatted = self._format_for_platform(content, content_format)
        
        # Add CTAs
        if 'cta' in input_data:
            formatted = self._add_cta(formatted, input_data['cta'])
            
        return {
            'content': formatted,
            'format': content_format,
            'character_count': len(formatted),
            'status': 200
        }
        
    def _generate_summary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of content."""
        text = input_data['text']
        style = input_data.get('style', 'concise')
        max_length = input_data.get('max_length', 150)
        
        # Handle long documents
        if len(text.split()) > 1000:
            summary = self._hierarchical_summarize(text, style, max_length)
        else:
            summary = self._simple_summarize(text, style, max_length)
            
        # Extract key points
        key_points = self._extract_key_points(text)
        
        return {
            'summary': summary,
            'key_points': key_points,
            'compression_ratio': len(text) / len(summary),
            'status': 200
        }
        
    def _simple_summarize(self, text: str, style: str, max_length: int) -> str:
        """Simple summarization for short texts."""
        prompt = f"""Summarize the following text in a {style} style:

{text}

Summary (max {max_length} words):"""

        summary = self._generate_from_prompt(prompt)
        
        # Ensure length constraint
        words = summary.split()
        if len(words) > max_length:
            summary = ' '.join(words[:max_length]) + '...'
            
        return summary
        
    def _hierarchical_summarize(self, text: str, style: str, max_length: int) -> str:
        """Hierarchical summarization for long texts."""
        # Split into chunks
        chunks = self._split_text_into_chunks(text, chunk_size=500)
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            chunk_summary = self._simple_summarize(chunk, 'detailed', 100)
            chunk_summaries.append(chunk_summary)
            
        # Combine and summarize again
        combined = ' '.join(chunk_summaries)
        final_summary = self._simple_summarize(combined, style, max_length)
        
        return final_summary
        
    def _research_topic(self, topic: str) -> List[str]:
        """Simulate topic research."""
        # In real implementation, this would search knowledge bases
        return [
            f"Key fact about {topic}",
            f"Recent development in {topic}",
            f"Expert opinion on {topic}",
            f"Statistics related to {topic}"
        ]
        
    def _generate_outline(self, topic: str, research: List[str]) -> List[Dict]:
        """Generate article outline."""
        prompt = f"""Create an article outline for: {topic}

Research points:
{chr(10).join(f'- {point}' for point in research)}

Provide a structured outline with main sections and key points for each."""

        outline_text = self._generate_from_prompt(prompt)
        
        # Parse outline (simplified)
        sections = []
        current_section = None
        
        for line in outline_text.split('\n'):
            if line.strip():
                if not line.startswith(' '):
                    # New section
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        'title': line.strip(),
                        'points': []
                    }
                else:
                    # Point under current section
                    if current_section:
                        current_section['points'].append(line.strip())
                        
        if current_section:
            sections.append(current_section)
            
        return sections
        
    def _combine_sections(self, sections: List[Dict]) -> str:
        """Combine sections into article."""
        article_parts = []
        
        for section in sections:
            article_parts.append(f"## {section['title']}\n")
            article_parts.append(section['content'])
            article_parts.append("\n")
            
        return '\n'.join(article_parts)
        
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        prompt = f"""Extract the 3-5 most important key points from this text:

{text[:1000]}...

Key points:"""

        response = self._generate_from_prompt(prompt)
        
        # Parse points
        points = []
        for line in response.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                points.append(line.strip())
                
        return points[:5]


# Question Answering System
class QuestionAnsweringSystem(TransformerApplication):
    """RAG-based question answering."""
    
    def __init__(self, model: nn.Module, config: ApplicationConfig, 
                 knowledge_base: 'KnowledgeBase'):
        super().__init__(model, config)
        self.knowledge_base = knowledge_base
        self.retriever = DocumentRetriever(knowledge_base)
        self.reranker = Reranker()
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Answer question using retrieval-augmented generation."""
        question = input_data['question']
        context_filter = input_data.get('context_filter', {})
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.search(
            question,
            k=20,
            filters=context_filter
        )
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0,
                'status': 200
            }
            
        # Rerank documents
        reranked_docs = self.reranker.rerank(
            question,
            retrieved_docs,
            top_k=5
        )
        
        # Build context
        context = self._build_qa_context(question, reranked_docs)
        
        # Generate answer
        answer = self._generate_answer(context)
        
        # Extract citations
        citations = self._extract_citations(answer, reranked_docs)
        
        # Calculate confidence
        confidence = self._calculate_confidence(answer, reranked_docs)
        
        return {
            'answer': answer,
            'sources': [{'id': doc['id'], 'title': doc['title']} for doc in reranked_docs],
            'citations': citations,
            'confidence': confidence,
            'status': 200
        }
        
    def _build_qa_context(self, question: str, documents: List[Dict]) -> str:
        """Build context for question answering."""
        context_parts = [
            "Answer the following question based on the provided documents.",
            f"\nQuestion: {question}\n",
            "Documents:"
        ]
        
        for i, doc in enumerate(documents):
            context_parts.append(f"\n[Document {i+1}] {doc['title']}")
            context_parts.append(doc['content'][:500])  # Truncate if needed
            
        context_parts.append("\nAnswer:")
        
        return '\n'.join(context_parts)
        
    def _generate_answer(self, context: str) -> str:
        """Generate answer from context."""
        inputs = self._tokenize(context)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=200,
                temperature=0.3,  # Lower temperature for factual answers
                do_sample=True
            )
            
        answer = self._decode(outputs[0])
        
        # Clean up
        answer = self._clean_answer(answer)
        
        return answer
        
    def _extract_citations(self, answer: str, documents: List[Dict]) -> List[int]:
        """Extract document citations from answer."""
        citations = []
        
        # Look for [1], [2], etc. in answer
        import re
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        
        for match in matches:
            doc_idx = int(match) - 1
            if 0 <= doc_idx < len(documents):
                citations.append(doc_idx)
                
        return list(set(citations))
        
    def _calculate_confidence(self, answer: str, documents: List[Dict]) -> float:
        """Calculate answer confidence based on evidence."""
        # Simple confidence calculation
        confidence = 0.5  # Base confidence
        
        # Boost for citations
        citations = self._extract_citations(answer, documents)
        confidence += len(citations) * 0.1
        
        # Boost for answer length
        if len(answer.split()) > 20:
            confidence += 0.2
            
        # Cap at 1.0
        return min(confidence, 1.0)
        
    def _clean_answer(self, answer: str) -> str:
        """Clean generated answer."""
        # Remove any context that might have leaked
        if "Question:" in answer:
            answer = answer.split("Question:")[0]
            
        # Remove any document references at the end
        if "Document" in answer:
            answer = answer.split("[Document")[0]
            
        return answer.strip()


# Supporting Classes
class ConversationManager:
    """Manage conversation history and state."""
    
    def __init__(self, storage_backend: str = 'memory'):
        self.storage = self._initialize_storage(storage_backend)
        self.max_history = 100
        
    def _initialize_storage(self, backend: str):
        """Initialize storage backend."""
        if backend == 'memory':
            return {}
        elif backend == 'redis':
            return redis.Redis()
        else:
            raise ValueError(f"Unknown storage backend: {backend}")
            
    def get_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history."""
        if isinstance(self.storage, dict):
            return self.storage.get(conversation_id, [])
        else:
            # Redis implementation
            data = self.storage.get(f"conv:{conversation_id}")
            return json.loads(data) if data else []
            
    def add_turn(self, conversation_id: str, user_message: str, 
                 assistant_message: str):
        """Add conversation turn."""
        history = self.get_history(conversation_id)
        
        history.append({
            'user': user_message,
            'assistant': assistant_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim if too long
        if len(history) > self.max_history:
            history = history[-self.max_history:]
            
        # Save
        if isinstance(self.storage, dict):
            self.storage[conversation_id] = history
        else:
            self.storage.setex(
                f"conv:{conversation_id}",
                86400,  # 24 hour expiry
                json.dumps(history)
            )


class SafetyFilter:
    """Filter unsafe content."""
    
    def __init__(self):
        self.unsafe_patterns = [
            'violence', 'hate', 'self-harm', 'illegal',
            'personal information', 'medical advice'
        ]
        
    def is_safe(self, text: str) -> bool:
        """Check if text is safe."""
        text_lower = text.lower()
        
        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if pattern in text_lower:
                return False
                
        return True


class Tool(ABC):
    """Base class for tools."""
    
    @abstractmethod
    def should_use(self, message: str) -> bool:
        """Check if tool should be used."""
        pass
        
    @abstractmethod
    def extract_params(self, message: str) -> Dict:
        """Extract parameters from message."""
        pass
        
    @abstractmethod
    def execute(self, params: Dict) -> Any:
        """Execute tool with parameters."""
        pass


class CalculatorTool(Tool):
    """Calculator tool for math operations."""
    
    def should_use(self, message: str) -> bool:
        """Check for math operations."""
        math_keywords = ['calculate', 'compute', 'solve', 'what is', '+', '-', '*', '/']
        return any(keyword in message.lower() for keyword in math_keywords)
        
    def extract_params(self, message: str) -> Dict:
        """Extract math expression."""
        # Simple extraction - in practice use more sophisticated parsing
        import re
        
        # Look for math expressions
        pattern = r'[\d\+\-\*/\(\)\s]+'
        matches = re.findall(pattern, message)
        
        if matches:
            expression = max(matches, key=len).strip()
            return {'expression': expression}
            
        return {}
        
    def execute(self, params: Dict) -> Any:
        """Calculate result."""
        try:
            expression = params.get('expression', '')
            # Safe evaluation
            result = eval(expression, {"__builtins__": {}}, {})
            return result
        except:
            return "Error in calculation"


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, limit: int, window: int = 60):
        self.limit = limit
        self.window = window
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
        
    def allow(self, user_id: str) -> bool:
        """Check if request is allowed."""
        with self.lock:
            now = time.time()
            user_requests = self.requests[user_id]
            
            # Remove old requests
            while user_requests and user_requests[0] < now - self.window:
                user_requests.popleft()
                
            # Check limit
            if len(user_requests) >= self.limit:
                return False
                
            # Add current request
            user_requests.append(now)
            return True


class MetricsCollector:
    """Collect application metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
        
    def record(self, metric: str, value: float):
        """Record metric value."""
        with self.lock:
            self.metrics[metric].append({
                'value': value,
                'timestamp': time.time()
            })
            
    def get_summary(self, metric: str, window: int = 3600) -> Dict:
        """Get metric summary for time window."""
        with self.lock:
            now = time.time()
            values = [
                m['value'] for m in self.metrics[metric]
                if m['timestamp'] > now - window
            ]
            
            if not values:
                return {}
                
            return {
                'count': len(values),
                'mean': np.mean(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }


class RedisCache:
    """Redis-based caching."""
    
    def __init__(self, ttl: int = 3600):
        self.client = None  # Initialize Redis client
        self.ttl = ttl
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Placeholder - implement Redis get
        return None
        
    def set(self, key: str, value: Any):
        """Set value in cache."""
        # Placeholder - implement Redis set with TTL
        pass


# Placeholder classes for additional components
class LanguageDetector:
    def detect(self, text: str) -> str:
        # Simplified detection
        return 'en'

class QualityEstimator:
    def estimate(self, source: str, translation: str, 
                source_lang: str, target_lang: str) -> float:
        # Simplified quality estimation
        return 0.9

class PythonParser:
    def parse(self, code: str) -> Dict:
        return {'type': 'python', 'valid': True}
    
    def extract_functions(self, code: str) -> List[Dict]:
        # Extract function definitions
        functions = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_name = line.split('(')[0].replace('def ', '').strip()
                functions.append({
                    'name': func_name,
                    'line': i,
                    'code': line
                })
                
        return functions

class JavaScriptParser:
    def parse(self, code: str) -> Dict:
        return {'type': 'javascript', 'valid': True}

class JavaParser:
    def parse(self, code: str) -> Dict:
        return {'type': 'java', 'valid': True}

class CppParser:
    def parse(self, code: str) -> Dict:
        return {'type': 'cpp', 'valid': True}

class CodeValidator:
    def validate(self, code: str, language: str) -> Dict:
        # Simplified validation
        return {'valid': True, 'errors': []}

class SecurityScanner:
    def scan(self, code: str, language: str) -> List[Dict]:
        # Simplified security scanning
        issues = []
        
        # Check for common security issues
        if 'eval(' in code:
            issues.append({
                'type': 'security',
                'severity': 'high',
                'message': 'Use of eval() is dangerous'
            })
            
        return issues

class SEOOptimizer:
    def optimize(self, content: str, keywords: List[str]) -> str:
        # Simplified SEO optimization
        return content
        
    def score(self, content: str, keywords: List[str]) -> float:
        # Calculate SEO score
        if not keywords:
            return 0.5
            
        content_lower = content.lower()
        keyword_count = sum(1 for kw in keywords if kw.lower() in content_lower)
        
        return min(keyword_count / len(keywords), 1.0)

class ReadabilityAnalyzer:
    def analyze(self, text: str) -> Dict:
        # Simplified readability analysis
        words = text.split()
        sentences = text.split('.')
        
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        
        # Simple readability score
        if avg_words_per_sentence < 15:
            score = 90  # Very easy
        elif avg_words_per_sentence < 20:
            score = 70  # Easy
        elif avg_words_per_sentence < 25:
            score = 50  # Medium
        else:
            score = 30  # Difficult
            
        return {'score': score, 'level': self._get_level(score)}
        
    def _get_level(self, score: int) -> str:
        if score >= 80:
            return 'Very Easy'
        elif score >= 60:
            return 'Easy'
        elif score >= 40:
            return 'Medium'
        else:
            return 'Difficult'

class KnowledgeBase:
    """Placeholder for knowledge base."""
    pass

class DocumentRetriever:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        
    def search(self, query: str, k: int, filters: Dict) -> List[Dict]:
        # Simplified search
        return [
            {
                'id': f'doc_{i}',
                'title': f'Document {i}',
                'content': f'Content related to {query}',
                'score': 0.9 - i * 0.1
            }
            for i in range(min(k, 5))
        ]

class Reranker:
    def rerank(self, query: str, documents: List[Dict], 
              top_k: int) -> List[Dict]:
        # Simplified reranking
        return documents[:top_k]


# Deployment utilities
async def create_app(model_path: str, app_type: str, config: Dict) -> TransformerApplication:
    """Factory function to create applications."""
    # Load model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Create configuration
    if app_type == 'chatbot':
        config_obj = ChatbotConfig(**config)
        return ConversationalAI(model, config_obj)
    elif app_type == 'translation':
        config_obj = ApplicationConfig(**config)
        # Note: Translation system expects dict of models
        return TranslationSystem({'en-es': model}, config_obj)
    elif app_type == 'code':
        config_obj = ApplicationConfig(**config)
        return CodeAssistant(model, config_obj)
    elif app_type == 'content':
        config_obj = ApplicationConfig(**config)
        return ContentGenerator(model, config_obj)
    elif app_type == 'qa':
        config_obj = ApplicationConfig(**config)
        kb = KnowledgeBase()  # Initialize knowledge base
        return QuestionAnsweringSystem(model, config_obj, kb)
    else:
        raise ValueError(f"Unknown application type: {app_type}")


# Example usage
if __name__ == "__main__":
    print("=== Real-world Transformer Applications Demo ===\n")
    
    # Create dummy model for demonstration
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {
                'pad_token_id': 0,
                'eos_token_id': 2
            })()
            
        def generate(self, *args, **kwargs):
            # Return dummy output
            return torch.randint(0, 30000, (1, 50))
    
    model = DummyTransformer()
    
    # Demo 1: Chatbot
    print("--- Chatbot Demo ---")
    chatbot_config = ChatbotConfig(
        persona="helpful AI assistant",
        temperature=0.7,
        enable_tools=True
    )
    
    chatbot = ConversationalAI(model, chatbot_config)
    
    # Simulate conversation
    response = chatbot({
        'message': "Hello! What can you help me with?",
        'conversation_id': 'demo_123',
        'user_id': 'user_456'
    })
    
    print(f"User: Hello! What can you help me with?")
    print(f"Assistant: {response.get('response', 'Error occurred')[:100]}...")
    
    # Demo 2: Code Assistant
    print("\n--- Code Assistant Demo ---")
    code_config = ApplicationConfig()
    code_assistant = CodeAssistant(model, code_config)
    
    code_request = {
        'task': 'complete',
        'code': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    ',
        'language': 'python'
    }
    
    code_response = code_assistant(code_request)
    print(f"Code completion task:")
    print(f"Status: {code_response.get('status')}")
    print(f"Validation: {code_response.get('validation', {})}")
    
    # Demo 3: Content Generator
    print("\n--- Content Generator Demo ---")
    content_config = ApplicationConfig()
    content_gen = ContentGenerator(model, content_config)
    
    article_request = {
        'type': 'article',
        'topic': 'The Future of AI',
        'style': 'informative',
        'word_count': 500,
        'keywords': ['AI', 'machine learning', 'future', 'technology']
    }
    
    article_response = content_gen(article_request)
    print(f"Article generation task:")
    print(f"Status: {article_response.get('status')}")
    print(f"Word count: {article_response.get('word_count', 0)}")
    
    # Demo 4: Metrics
    print("\n--- Application Metrics ---")
    metrics = chatbot.metrics
    
    # Simulate some metrics
    for i in range(10):
        metrics.record('latency', np.random.normal(50, 10))
        metrics.record('success', 1)
        
    latency_summary = metrics.get_summary('latency')
    print(f"Latency metrics:")
    print(f"  Mean: {latency_summary.get('mean', 0):.2f}ms")
    print(f"  P95: {latency_summary.get('p95', 0):.2f}ms")
    print(f"  P99: {latency_summary.get('p99', 0):.2f}ms")
    
    print("\n✅ Real-world applications demonstration complete!")