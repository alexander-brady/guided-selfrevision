import math
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import time
import logging
import re

# vLLM imports
from vllm import SamplingParams

from lm_eval.utils import eval_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EIGMetrics:
    """Comprehensive metrics for EIG-based reasoning."""
    total_calls: int = 0
    successful_calculations: int = 0
    entropy_calculation_failures: int = 0
    beam_search_failures: int = 0
    mc_sampling_failures: int = 0
    information_gains: List[float] = None
    decision_history: List[bool] = None
    computational_times: Dict[str, List[float]] = None
    
    def __post_init__(self):
        if self.information_gains is None:
            self.information_gains = []
        if self.decision_history is None:
            self.decision_history = []
        if self.computational_times is None:
            self.computational_times = defaultdict(list)


class AnswerPosteriorEstimator:
    """
    Estimates the posterior distribution over answers using beam search and logit analysis.
    
    Mathematical Framework:
    - Beam search to identify candidate answers A = {a_1, ..., a_k}
    - Score computation: s_i = log P(a_i | H_t)
    - Softmax normalization: P_t(a_i) = exp(s_i) / Σ_j exp(s_j)
    
    Adapted for vLLM models.
    """
    
    def __init__(self, 
                 beam_size: int = 8,
                 answer_extraction_patterns: List[str] = None,
                 min_answer_confidence: float = 1e-6,
                 numerical_stability_eps: float = 1e-12):
        self.beam_size = beam_size
        self.min_confidence = min_answer_confidence
        self.eps = numerical_stability_eps
        
        # Enhanced patterns for extracting final answers from math problems
        self.answer_patterns = answer_extraction_patterns or [
            r"\\boxed\{([^}]+)\}",  # LaTeX boxed answers (most common in math)
            r"\$\$([^$]+)\$\$",  # LaTeX display math
            r"\$([^$]+)\$",  # LaTeX inline math
            r"Final Answer:\s*([^\n]+)",
            r"Answer:\s*([^\n]+)", 
            r"The answer is\s*([^\n]+)",
            r"Therefore,?\s*([^\n]+)",
            r"So,?\s*([^\n]+)",
            r"Thus,?\s*([^\n]+)",
            r"(?:^|\n)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\n|$)",  # Standalone numbers
            r"(?:^|\n)\s*([A-Z])\s*(?:\n|$)",  # Standalone letters (multiple choice)
        ]
        
    def extract_answer_candidates(self, text: str, vllm_model) -> List[str]:
        """
        Extract potential answer candidates from generated text.
        
        Args:
            text: Generated reasoning text
            vllm_model: vLLM model instance
            
        Returns:
            List of candidate answer strings
        """
        candidates = []
        
        # Pattern-based extraction with priority order (most specific first)
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            for match in matches:
                cleaned_match = match.strip()
                if cleaned_match and len(cleaned_match) < 100:  # Reasonable answer length
                    candidates.append(cleaned_match)
        
        # If no patterns match, try extracting from the end of the text
        if not candidates:
            # Look for numbers or simple expressions at the end
            end_text = text.split('\n')[-5:]  # Last 5 lines
            for line in reversed(end_text):
                line = line.strip()
                if line:
                    # Extract numbers, simple expressions, or short answers
                    number_match = re.search(r'([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)', line)
                    if number_match:
                        candidates.append(number_match.group(1))
                        break
                    
                    # Extract short answers (likely to be final answers)
                    if len(line) < 50 and not line.lower().startswith(('let', 'we', 'so', 'now', 'then')):
                        candidates.append(line)
                        break
        
        # If still no candidates, generate some reasonable fallbacks
        if not candidates:
            candidates = ["0", "1", "unknown"]  # Basic mathematical fallbacks
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in seen and len(candidate.strip()) > 0:
                seen.add(candidate)
                unique_candidates.append(candidate.strip())
                
        return unique_candidates[:self.beam_size]  # Limit to beam size
    
    def compute_answer_logits(self, 
                             candidates: List[str], 
                             context_seq: List[int],
                             vllm_model) -> torch.Tensor:
        """
        Compute log probabilities for each answer candidate using vLLM.
        
        FIXED: Now handles vLLM logprob format correctly and includes proper fallbacks.
        
        Args:
            candidates: List of answer candidate strings
            context_seq: Context sequence as list of token IDs
            vllm_model: vLLM model instance
            
        Returns:
            Tensor of log probabilities for each candidate
        """
        if not candidates:
            logger.warning("No answer candidates provided")
            return torch.tensor([], dtype=torch.float32)
            
        logits = []
        
        try:
            # Simple approach: score candidates by their token probabilities
            # This is more reliable than trying to generate continuations
            for candidate in candidates:
                try:
                    # Tokenize candidate answer
                    answer_tokens = vllm_model.tok_encode(candidate)
                    if not answer_tokens:
                        logits.append(float('-inf'))
                        continue
                    
                    # Create prompt that asks for the answer
                    prompt_text = vllm_model.tokenizer.decode(context_seq, skip_special_tokens=True)
                    full_prompt = prompt_text + "\n\nThe answer is: " + candidate
                    full_tokens = vllm_model.tok_encode(full_prompt)
                    
                    # Use vLLM to get logprobs for this completion
                    sampling_params = SamplingParams(
                        max_tokens=1,  # We just need logprobs of existing tokens
                        temperature=0,
                        logprobs=1,
                        prompt_logprobs=len(answer_tokens)  # Get logprobs for answer tokens
                    )
                    
                    outputs = vllm_model.model.generate(
                        prompt_token_ids=[full_tokens],
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )
                    
                    if outputs and len(outputs) > 0:
                        output = outputs[0]
                        # Sum the logprobs of the answer tokens
                        answer_logprob = 0.0
                        
                        # Access prompt_logprobs which contains logprobs of the prompt tokens
                        if hasattr(output, 'prompt_logprobs') and output.prompt_logprobs:
                            # Get logprobs for the answer part (last few tokens)
                            prompt_logprobs = output.prompt_logprobs
                            answer_start_idx = len(full_tokens) - len(answer_tokens)
                            
                            for i in range(len(answer_tokens)):
                                token_idx = answer_start_idx + i
                                if token_idx < len(prompt_logprobs) and prompt_logprobs[token_idx]:
                                    token_id = answer_tokens[i]
                                    logprob_dict = prompt_logprobs[token_idx]
                                    
                                    if token_id in logprob_dict:
                                        logprob_entry = logprob_dict[token_id]
                                        if hasattr(logprob_entry, 'logprob'):
                                            answer_logprob += logprob_entry.logprob
                                        else:
                                            answer_logprob += float(logprob_entry)
                        
                        logits.append(answer_logprob)
                    else:
                        logits.append(float('-inf'))
                        
                except Exception as e:
                    logger.warning(f"Error processing candidate '{candidate}': {e}")
                    logits.append(float('-inf'))
                    
        except Exception as e:
            logger.error(f"Error computing answer logits with vLLM: {e}")
            # Fallback: uniform distribution over candidates
            logits = [0.0] * len(candidates)
            
        # If all logits are -inf, use uniform distribution
        if all(l == float('-inf') for l in logits):
            logger.warning("All candidates got -inf logprob, using uniform distribution")
            logits = [0.0] * len(candidates)
            
        return torch.tensor(logits, dtype=torch.float32)
    
    def estimate_posterior(self, 
                          text: str, 
                          context_seq: List[int],
                          vllm_model) -> Tuple[List[str], torch.Tensor]:
        """
        Estimate the posterior distribution over answers.
        
        Returns:
            Tuple of (candidate_answers, posterior_probabilities)
        """
        # Extract answer candidates
        candidates = self.extract_answer_candidates(text, vllm_model)
        
        if not candidates:
            logger.warning("No answer candidates found")
            return [], torch.tensor([])
        
        # Compute logits for each candidate
        logits = self.compute_answer_logits(candidates, context_seq, vllm_model)
        
        if len(logits) == 0:
            return candidates, torch.tensor([])
        
        # Apply numerical stability and compute probabilities
        logits_stable = logits - torch.max(logits)  # Prevent overflow
        probs = torch.softmax(logits_stable, dim=0)
        
        # Ensure minimum probability for numerical stability
        probs = torch.clamp(probs, min=self.min_confidence)
        probs = probs / torch.sum(probs)  # Renormalize
        
        return candidates, probs


class MonteCarloForecaster:
    """
    Monte Carlo estimation of expected future entropy.
    
    Mathematical Framework:
    E[H_{t+1} | H_t] ≈ (1/K) Σ_{k=1}^K H_{t+1}^{(k)}
    
    where H_{t+1}^{(k)} is the entropy after the k-th sampled continuation.
    
    FIXED: Improved vLLM integration and error handling.
    """
    
    def __init__(self,
                 num_samples: int = 5,
                 sample_length: int = 64,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 max_computation_time: float = 30.0):
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_computation_time = max_computation_time
        
    def sample_continuation(self, 
                           context_seq: List[int],
                           vllm_model) -> Optional[List[int]]:
        """
        Sample a single continuation of the reasoning process using vLLM.
        
        FIXED: Better error handling and token sequence management.
        
        Args:
            context_seq: Current context sequence as list of token IDs
            vllm_model: vLLM model instance
            
        Returns:
            Extended sequence as list of token IDs or None if sampling fails
        """
        try:
            sampling_params = SamplingParams(
                max_tokens=self.sample_length,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=[]  # Let it generate freely
            )
            
            outputs = vllm_model.model.generate(
                prompt_token_ids=[context_seq],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            if outputs and len(outputs) > 0 and outputs[0].outputs:
                # Return the full sequence (context + generated)
                generated_tokens = outputs[0].outputs[0].token_ids
                return context_seq + generated_tokens
            else:
                logger.warning("No output generated in continuation sampling")
                return None
                
        except Exception as e:
            logger.warning(f"Continuation sampling failed: {e}")
            return None
    
    def forecast_entropy(self,
                        context_seq: List[int],
                        vllm_model,
                        posterior_estimator: AnswerPosteriorEstimator) -> float:
        """
        Estimate E[H_{t+1} | H_t] via Monte Carlo sampling using vLLM.
        
        FIXED: Better error handling and fallback strategies.
        
        Args:
            context_seq: Current reasoning sequence as list of token IDs
            vllm_model: vLLM model instance
            posterior_estimator: Answer posterior estimator
            
        Returns:
            Expected entropy after one more reasoning step
        """
        start_time = time.time()
        future_entropies = []
        
        for sample_idx in range(self.num_samples):
            # Check computation time limit
            if time.time() - start_time > self.max_computation_time:
                logger.warning(f"MC forecasting timeout after {sample_idx} samples")
                break
                
            # Sample continuation
            extended_seq = self.sample_continuation(context_seq, vllm_model)
            if extended_seq is None:
                continue
                
            # Convert to text for answer extraction
            try:
                extended_text = vllm_model.tokenizer.decode(
                    extended_seq, 
                    skip_special_tokens=True
                )
            except Exception as e:
                logger.warning(f"Text decoding failed for sample {sample_idx}: {e}")
                continue
            
            # Estimate posterior for this continuation
            try:
                candidates, probs = posterior_estimator.estimate_posterior(
                    extended_text, extended_seq, vllm_model
                )
                
                if len(probs) > 0:
                    # Compute entropy: H = -Σ p_i log p_i
                    # Clamp probabilities to avoid log(0)
                    probs_clamped = torch.clamp(probs, min=posterior_estimator.eps)
                    entropy = -torch.sum(probs_clamped * torch.log(probs_clamped))
                    future_entropies.append(entropy.item())
                    
            except Exception as e:
                logger.warning(f"Posterior estimation failed for sample {sample_idx}: {e}")
                continue
        
        if not future_entropies:
            logger.error("All MC samples failed")
            return float('inf')  # Conservative: assume no information gain
            
        return np.mean(future_entropies)


class ExpectedInformationGainCalculator:
    """
    Main EIG calculator combining posterior estimation and MC forecasting.
    
    Mathematical Framework:
    EIG_t = H_t - E[H_{t+1} | H_t]
    
    where:
    - H_t is current answer posterior entropy
    - E[H_{t+1} | H_t] is expected entropy after one more reasoning step
    
    FIXED: Enhanced error handling and better integration with vLLM.
    """
    
    def __init__(self,
                 beam_size: int = 8,
                 mc_samples: int = 5,
                 sample_length: int = 64,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 lambda_cost: float = 0.05,
                 max_computation_time: float = 30.0):
        
        self.lambda_cost = lambda_cost
        
        # Initialize components
        self.posterior_estimator = AnswerPosteriorEstimator(
            beam_size=beam_size,
            min_answer_confidence=1e-6
        )
        
        self.mc_forecaster = MonteCarloForecaster(
            num_samples=mc_samples,
            sample_length=sample_length,
            temperature=temperature,
            top_p=top_p,
            max_computation_time=max_computation_time
        )
        
        # Metrics tracking
        self.metrics = EIGMetrics()
        
    def compute_current_entropy(self,
                              text: str,
                              context_seq: List[int],
                              vllm_model) -> float:
        """
        Compute H_t = -Σ P_t(a) log P_t(a).
        
        Args:
            text: Current reasoning text
            context_seq: Current sequence as list of token IDs
            vllm_model: vLLM model instance
            
        Returns:
            Current answer posterior entropy
        """
        start_time = time.time()
        
        try:
            candidates, probs = self.posterior_estimator.estimate_posterior(
                text, context_seq, vllm_model
            )
            
            if len(probs) == 0:
                logger.warning("No valid posterior probabilities")
                return 0.0  # No uncertainty
            
            # Compute Shannon entropy with numerical stability
            probs_clamped = torch.clamp(probs, min=self.posterior_estimator.eps)
            entropy = -torch.sum(probs_clamped * torch.log(probs_clamped))
            
            self.metrics.computational_times['current_entropy'].append(
                time.time() - start_time
            )
            
            return entropy.item()
            
        except Exception as e:
            logger.error(f"Current entropy computation failed: {e}")
            self.metrics.entropy_calculation_failures += 1
            return 0.0
    
    def compute_expected_information_gain(self,
                                        iteration: int,
                                        seq: List[int],
                                        entropies: List[float],
                                        vllm_model) -> Tuple[float, Dict[str, Any]]:
        """
        Main EIG computation: EIG_t = H_t - E[H_{t+1} | H_t].
        
        FIXED: Better error handling and more robust computation.
        
        Args:
            iteration: Current reasoning iteration
            seq: Current token sequence as list of token IDs
            entropies: Per-token entropies (for fallback)
            vllm_model: vLLM model instance
            
        Returns:
            Tuple of (information_gain, computation_details)
        """
        self.metrics.total_calls += 1
        computation_start = time.time()
        
        details = {
            'iteration': iteration,
            'sequence_length': len(seq),
            'current_entropy': 0.0,
            'expected_future_entropy': 0.0,
            'information_gain': 0.0,
            'computation_time': 0.0,
            'success': False
        }
        
        try:
            # Convert sequence to text for analysis
            context_seq = seq
            try:
                text = vllm_model.tokenizer.decode(seq, skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Failed to decode sequence: {e}")
                text = ""
            
            # Step 1: Compute current entropy H_t
            current_entropy = self.compute_current_entropy(text, context_seq, vllm_model)
            details['current_entropy'] = current_entropy
            
            # Step 2: Forecast expected future entropy E[H_{t+1} | H_t]
            expected_future_entropy = self.mc_forecaster.forecast_entropy(
                context_seq, vllm_model, self.posterior_estimator
            )
            details['expected_future_entropy'] = expected_future_entropy
            
            # Step 3: Compute information gain
            if expected_future_entropy == float('inf'):
                information_gain = 0.0  # Conservative fallback
            else:
                information_gain = max(0.0, current_entropy - expected_future_entropy)
            
            details['information_gain'] = information_gain
            details['success'] = True
            
            # Update metrics
            self.metrics.successful_calculations += 1
            self.metrics.information_gains.append(information_gain)
            
            logger.info(f"EIG computation successful: "
                       f"H_t={current_entropy:.4f}, "
                       f"E[H_{{t+1}}]={expected_future_entropy:.4f}, "
                       f"EIG={information_gain:.4f}")
            
        except Exception as e:
            logger.error(f"EIG computation failed: {e}")
            information_gain = 0.0  # Conservative fallback
            details['error'] = str(e)
        
        # Record computation time
        computation_time = time.time() - computation_start
        details['computation_time'] = computation_time
        self.metrics.computational_times['total_eig'].append(computation_time)
        
        return information_gain, details
    
    def should_continue_reasoning(self,
                                 information_gain: float,
                                 details: Dict[str, Any]) -> bool:
        """
        Apply decision rule: continue iff EIG_t > λ.
        
        Args:
            information_gain: Computed EIG value
            details: Computation details for logging
            
        Returns:
            True if reasoning should continue
        """
        decision = information_gain > self.lambda_cost
        self.metrics.decision_history.append(decision)
        
        logger.info(f"EIG decision: {information_gain:.4f} > {self.lambda_cost} = {decision}")
        
        return decision
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if self.metrics.total_calls == 0:
            return {"message": "No EIG computations performed yet"}
        
        success_rate = self.metrics.successful_calculations / self.metrics.total_calls
        
        summary = {
            "total_calls": self.metrics.total_calls,
            "success_rate": success_rate,
            "failure_breakdown": {
                "entropy_failures": self.metrics.entropy_calculation_failures,
                "beam_failures": self.metrics.beam_search_failures,
                "mc_failures": self.metrics.mc_sampling_failures
            }
        }
        
        if self.metrics.information_gains:
            summary["information_gain_stats"] = {
                "mean": np.mean(self.metrics.information_gains),
                "std": np.std(self.metrics.information_gains),
                "min": np.min(self.metrics.information_gains),
                "max": np.max(self.metrics.information_gains)
            }
        
        if self.metrics.decision_history:
            summary["decision_stats"] = {
                "continue_rate": np.mean(self.metrics.decision_history),
                "total_decisions": len(self.metrics.decision_history)
            }
        
        if self.metrics.computational_times:
            summary["timing_stats"] = {}
            for key, times in self.metrics.computational_times.items():
                if times:
                    summary["timing_stats"][key] = {
                        "mean": np.mean(times),
                        "std": np.std(times),
                        "total": np.sum(times)
                    }
        
        return summary 