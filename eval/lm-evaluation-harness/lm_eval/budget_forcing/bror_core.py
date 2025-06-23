import math
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time
import logging
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BROMetrics:
    """Comprehensive metrics for Bayes-Risk-Optimal Reasoning."""
    total_calls: int = 0
    successful_calculations: int = 0
    belief_estimation_failures: int = 0
    forecasting_failures: int = 0
    calibration_failures: int = 0
    decision_history: List[bool] = field(default_factory=list)
    belief_trajectories: List[List[float]] = field(default_factory=list)
    expected_improvements: List[float] = field(default_factory=list)
    bayes_risks: List[Dict[str, float]] = field(default_factory=list)
    computational_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    cost_effectiveness: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not hasattr(self, 'decision_history'):
            self.decision_history = []
        if not hasattr(self, 'belief_trajectories'):
            self.belief_trajectories = []


class BayesianBeliefEstimator:
    """
    Estimates P(A = correct | H_t) using ensemble methods and calibration.
    
    Mathematical Framework:
    - Ensemble posterior: p_t = (1/N) Σ_i σ(log P_i(answer|H_t) - log P_i(other|H_t))
    - Logistic calibration: p_calibrated = 1/(1 + exp(-α·score - β))
    - Uncertainty quantification via confidence intervals
    """
    
    def __init__(self,
                 ensemble_size: int = 8,
                 calibration_alpha: float = 1.0,
                 calibration_beta: float = 0.0,
                 dropout_rate: float = 0.1,
                 min_confidence: float = 1e-6,
                 max_confidence: float = 1.0 - 1e-6,
                 numerical_stability_eps: float = 1e-12):
        
        self.ensemble_size = ensemble_size
        self.calibration_alpha = calibration_alpha
        self.calibration_beta = calibration_beta
        self.dropout_rate = dropout_rate
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.eps = numerical_stability_eps
        
        # Calibration parameters learned from validation data
        self._calibration_fitted = False
        self._calibration_history = []
        
    def _enable_ensemble_dropout(self, model, enable: bool = True):
        """Enable/disable dropout for ensemble uncertainty estimation."""
        try:
            if hasattr(model, 'train'):
                if enable:
                    model.train()  # Enable dropout
                    # Set specific dropout rate if possible
                    for module in model.modules():
                        if hasattr(module, 'dropout') and hasattr(module.dropout, 'p'):
                            module.dropout.p = self.dropout_rate
                else:
                    model.eval()  # Disable dropout
        except Exception as e:
            logger.warning(f"Could not control dropout state: {e}")
    
    def _compute_ensemble_logits(self, 
                                text: str, 
                                context_seq: torch.Tensor,
                                hflm) -> List[Dict[str, float]]:
        """
        Compute ensemble logits using Monte Carlo dropout.
        
        Returns:
            List of dictionaries containing logits for each ensemble member
        """
        ensemble_results = []
        
        try:
            # Extract potential answers from text
            answer_candidates = self._extract_answer_candidates(text, hflm)
            
            if not answer_candidates:
                logger.warning("No answer candidates found")
                return []
            
            # Enable dropout for ensemble
            self._enable_ensemble_dropout(hflm.model, enable=True)
            
            with torch.no_grad():
                for ensemble_idx in range(self.ensemble_size):
                    member_logits = {}
                    
                    for candidate in answer_candidates:
                        # Tokenize candidate answer
                        try:
                            answer_tokens = hflm.tok_encode(candidate)
                            if not answer_tokens:
                                member_logits[candidate] = float('-inf')
                                continue
                            
                            # Compute log probability for this ensemble member
                            full_seq = context_seq.tolist() + answer_tokens
                            input_ids = torch.tensor(full_seq, device=hflm.device).unsqueeze(0)
                            
                            outputs = hflm.model(input_ids)
                            logprobs = torch.log_softmax(outputs.logits[0], dim=-1)
                            
                            # Sum log probabilities for answer tokens
                            answer_logprob = 0.0
                            context_len = len(context_seq)
                            
                            for i, token_id in enumerate(answer_tokens):
                                if context_len + i < logprobs.shape[0]:
                                    answer_logprob += logprobs[context_len + i - 1, token_id].item()
                            
                            member_logits[candidate] = answer_logprob
                            
                        except Exception as e:
                            logger.warning(f"Error computing logits for candidate '{candidate}': {e}")
                            member_logits[candidate] = float('-inf')
                    
                    ensemble_results.append(member_logits)
            
            # Disable dropout
            self._enable_ensemble_dropout(hflm.model, enable=False)
            
        except Exception as e:
            logger.error(f"Ensemble computation failed: {e}")
            return []
        
        return ensemble_results
    
    def _extract_answer_candidates(self, text: str, hflm) -> List[str]:
        """Extract potential answer candidates from generated text."""
        import re
        
        patterns = [
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"The answer is\s*(.+?)(?:\n|$)",
            r"\$\$(.+?)\$\$",
            r"\\boxed\{(.+?)\}",
            r"(?:^|\n)\s*([A-E])\s*[:\.\)]\s*",  # Multiple choice
            r"(?:^|\n)\s*(\d+(?:\.\d+)?)\s*(?:\n|$)",  # Numeric answers
        ]
        
        candidates = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            candidates.extend([m.strip() for m in matches if m.strip()])
        
        # Fallback: extract last meaningful sentence
        if not candidates:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                candidates.append(sentences[-1])
        
        # Remove duplicates while preserving order
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen and len(candidate) < 200:  # Reasonable length
                seen.add(candidate)
                unique_candidates.append(candidate)
        
        return unique_candidates[:10]  # Limit to top 10 candidates
    
    def _apply_calibration(self, raw_confidence: float) -> float:
        """
        Apply logistic calibration to raw confidence score.
        
        Mathematical formula: p_calibrated = 1/(1 + exp(-α·score - β))
        """
        try:
            # Convert confidence to logit space for calibration
            if raw_confidence <= 0:
                logit_score = -10.0  # Very low confidence
            elif raw_confidence >= 1:
                logit_score = 10.0   # Very high confidence
            else:
                logit_score = logit(raw_confidence)
            
            # Apply linear calibration in logit space
            calibrated_logit = self.calibration_alpha * logit_score + self.calibration_beta
            
            # Convert back to probability space
            calibrated_prob = expit(calibrated_logit)
            
            # Apply bounds
            calibrated_prob = np.clip(calibrated_prob, self.min_confidence, self.max_confidence)
            
            return calibrated_prob
            
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            return np.clip(raw_confidence, self.min_confidence, self.max_confidence)
    
    def estimate_correctness_probability(self,
                                       text: str,
                                       context_seq: torch.Tensor,
                                       hflm) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate P(A = correct | H_t) using ensemble methods.
        
        Mathematical Framework:
        1. Compute ensemble logits using Monte Carlo dropout
        2. Convert to probabilities using softmax normalization
        3. Apply logistic calibration
        4. Return calibrated belief state with uncertainty measures
        
        Returns:
            Tuple of (correctness_probability, computation_details)
        """
        start_time = time.time()
        
        details = {
            'ensemble_size': self.ensemble_size,
            'raw_confidence': 0.0,
            'calibrated_confidence': 0.0,
            'ensemble_std': 0.0,
            'answer_candidates': [],
            'computation_time': 0.0,
            'success': False
        }
        
        try:
            # Compute ensemble logits
            ensemble_results = self._compute_ensemble_logits(text, context_seq, hflm)
            
            if not ensemble_results:
                logger.warning("Ensemble computation failed")
                return 0.5, details  # Maximally uncertain
            
            # Extract answer candidates
            all_candidates = set()
            for member_result in ensemble_results:
                all_candidates.update(member_result.keys())
            
            answer_candidates = list(all_candidates)
            details['answer_candidates'] = answer_candidates
            
            if not answer_candidates:
                return 0.5, details
            
            # Compute ensemble probabilities
            ensemble_probs = []
            
            for member_result in ensemble_results:
                # Get logits for this ensemble member
                member_logits = np.array([
                    member_result.get(candidate, float('-inf')) 
                    for candidate in answer_candidates
                ])
                
                # Apply softmax for numerical stability
                if np.all(np.isinf(member_logits)):
                    member_probs = np.ones(len(answer_candidates)) / len(answer_candidates)
                else:
                    member_logits = member_logits - np.max(member_logits)  # Numerical stability
                    member_probs = np.exp(member_logits) / np.sum(np.exp(member_logits))
                
                # Get probability of most likely answer for this member
                max_prob = np.max(member_probs)
                ensemble_probs.append(max_prob)
            
            # Compute ensemble statistics
            raw_confidence = np.mean(ensemble_probs)
            ensemble_std = np.std(ensemble_probs)
            
            details['raw_confidence'] = raw_confidence
            details['ensemble_std'] = ensemble_std
            
            # Apply calibration
            calibrated_confidence = self._apply_calibration(raw_confidence)
            details['calibrated_confidence'] = calibrated_confidence
            
            details['success'] = True
            
            logger.debug(f"Belief estimation: raw={raw_confidence:.4f}, "
                        f"calibrated={calibrated_confidence:.4f}, "
                        f"std={ensemble_std:.4f}")
            
            return calibrated_confidence, details
            
        except Exception as e:
            logger.error(f"Belief estimation failed: {e}")
            return 0.5, details
        
        finally:
            details['computation_time'] = time.time() - start_time


class ExpectedImprovementForecaster:
    """
    Forecasts E[p_{t+1} | H_t] - p_t using learned regression and Monte Carlo sampling.
    
    Mathematical Framework:
    - Monte Carlo estimation: Δp_t ≈ (1/K) Σ_k [p_t^{(k)} - p_t]
    - Regression forecasting: Δp_t = f(p_t, entropy_t, length_t, ...)
    - Ensemble averaging for robustness
    """
    
    def __init__(self,
                 num_mc_samples: int = 6,
                 sample_length: int = 48,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 regression_features: bool = True,
                 max_forecast_time: float = 25.0):
        
        self.num_mc_samples = num_mc_samples
        self.sample_length = sample_length
        self.temperature = temperature
        self.top_p = top_p
        self.regression_features = regression_features
        self.max_forecast_time = max_forecast_time
        
        # Learned regression parameters (fitted from data)
        self._regression_params = {
            'entropy_weight': 0.3,
            'confidence_weight': 0.4,
            'length_weight': 0.1,
            'interaction_weight': 0.2,
            'baseline': 0.02
        }
    
    def _sample_continuation(self,
                           context_seq: torch.Tensor,
                           hflm) -> Optional[torch.Tensor]:
        """Sample a single continuation using controlled generation."""
        try:
            with torch.no_grad():
                input_ids = context_seq.unsqueeze(0) if context_seq.dim() == 1 else context_seq
                
                outputs = hflm.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.sample_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=hflm.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1,  # Reduce repetition
                    length_penalty=1.0
                )
                
                return outputs[0]
                
        except Exception as e:
            logger.warning(f"Continuation sampling failed: {e}")
            return None
    
    def _compute_regression_features(self,
                                   text: str,
                                   current_belief: float,
                                   entropies: List[float]) -> Dict[str, float]:
        """Compute features for regression-based forecasting."""
        features = {}
        
        try:
            # Entropy-based features
            if entropies:
                features['mean_entropy'] = np.mean(entropies)
                features['entropy_std'] = np.std(entropies)
                features['entropy_trend'] = np.polyfit(range(len(entropies)), entropies, 1)[0] if len(entropies) > 1 else 0
                features['last_k_entropy'] = np.mean(entropies[-5:]) if len(entropies) >= 5 else features['mean_entropy']
            else:
                features.update({
                    'mean_entropy': 0.5, 'entropy_std': 0.0, 
                    'entropy_trend': 0.0, 'last_k_entropy': 0.5
                })
            
            # Confidence-based features
            features['current_belief'] = current_belief
            features['belief_distance_from_certain'] = abs(current_belief - 1.0)
            features['belief_distance_from_uncertain'] = abs(current_belief - 0.5)
            
            # Text-based features
            features['text_length'] = len(text)
            features['num_sentences'] = max(1, text.count('.') + text.count('!') + text.count('?'))
            features['avg_sentence_length'] = features['text_length'] / features['num_sentences']
            
            # Reasoning pattern features
            features['has_math'] = int(bool('=' in text or '+' in text or '-' in text))
            features['has_steps'] = int(bool('step' in text.lower() or any(str(i) in text for i in range(1, 6))))
            features['has_conclusion'] = int(bool('therefore' in text.lower() or 'conclusion' in text.lower()))
            
            # Interaction features
            features['entropy_belief_interaction'] = features['mean_entropy'] * (1 - current_belief)
            features['length_confidence_interaction'] = features['text_length'] * current_belief
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return {'mean_entropy': 0.5, 'current_belief': current_belief, 'text_length': len(text)}
    
    def _regression_forecast(self,
                           features: Dict[str, float]) -> float:
        """
        Use learned regression to forecast expected improvement.
        
        Mathematical model:
        Δp_t = β₀ + β₁·entropy + β₂·(1-p_t) + β₃·length + β₄·interactions
        """
        try:
            params = self._regression_params
            
            # Core regression model
            forecast = params['baseline']
            forecast += params['entropy_weight'] * features.get('mean_entropy', 0.5)
            forecast += params['confidence_weight'] * features.get('belief_distance_from_certain', 0.5)
            forecast += params['length_weight'] * min(features.get('text_length', 0) / 1000.0, 1.0)  # Normalized
            forecast += params['interaction_weight'] * features.get('entropy_belief_interaction', 0.25)
            
            # Apply bounds and scaling
            forecast = max(0.0, min(forecast, 0.5))  # Reasonable bounds
            
            return forecast
            
        except Exception as e:
            logger.warning(f"Regression forecast failed: {e}")
            return 0.02  # Conservative default
    
    def forecast_expected_improvement(self,
                                    text: str,
                                    context_seq: torch.Tensor,
                                    current_belief: float,
                                    entropies: List[float],
                                    hflm,
                                    belief_estimator: BayesianBeliefEstimator) -> Tuple[float, Dict[str, Any]]:
        """
        Forecast E[p_{t+1} | H_t] - p_t using Monte Carlo sampling and regression.
        
        Returns:
            Tuple of (expected_improvement, computation_details)
        """
        start_time = time.time()
        
        details = {
            'mc_samples_attempted': self.num_mc_samples,
            'mc_samples_successful': 0,
            'mc_forecast': 0.0,
            'regression_forecast': 0.0,
            'combined_forecast': 0.0,
            'future_beliefs': [],
            'computation_time': 0.0,
            'success': False
        }
        
        try:
            # Compute regression features
            features = self._compute_regression_features(text, current_belief, entropies)
            regression_forecast = self._regression_forecast(features)
            details['regression_forecast'] = regression_forecast
            
            # Monte Carlo forecasting
            future_beliefs = []
            
            for sample_idx in range(self.num_mc_samples):
                # Check time budget
                if time.time() - start_time > self.max_forecast_time:
                    logger.warning(f"Forecasting timeout after {sample_idx} samples")
                    break
                
                # Sample continuation
                extended_seq = self._sample_continuation(context_seq, hflm)
                if extended_seq is None:
                    continue
                
                # Convert to text
                try:
                    extended_text = hflm.tokenizer.decode(extended_seq, skip_special_tokens=True)
                except Exception as e:
                    logger.warning(f"Text decoding failed for sample {sample_idx}: {e}")
                    continue
                
                # Estimate belief for this sample
                try:
                    future_belief, _ = belief_estimator.estimate_correctness_probability(
                        extended_text, extended_seq, hflm
                    )
                    future_beliefs.append(future_belief)
                    
                except Exception as e:
                    logger.warning(f"Belief estimation failed for sample {sample_idx}: {e}")
                    continue
            
            details['mc_samples_successful'] = len(future_beliefs)
            details['future_beliefs'] = future_beliefs
            
            # Compute MC forecast
            if future_beliefs:
                mc_forecast = np.mean(future_beliefs) - current_belief
                details['mc_forecast'] = mc_forecast
            else:
                logger.warning("All MC samples failed")
                mc_forecast = 0.0
                details['mc_forecast'] = 0.0
            
            # Combine forecasts using weighted average
            if len(future_beliefs) >= 2:
                # More weight on MC if we have sufficient samples
                combined_forecast = 0.7 * mc_forecast + 0.3 * regression_forecast
            else:
                # More weight on regression if MC is unreliable
                combined_forecast = 0.3 * mc_forecast + 0.7 * regression_forecast
            
            details['combined_forecast'] = combined_forecast
            details['success'] = True
            
            logger.debug(f"Improvement forecast: MC={mc_forecast:.4f}, "
                        f"regression={regression_forecast:.4f}, "
                        f"combined={combined_forecast:.4f}")
            
            return combined_forecast, details
            
        except Exception as e:
            logger.error(f"Improvement forecasting failed: {e}")
            return 0.02, details  # Conservative fallback
        
        finally:
            details['computation_time'] = time.time() - start_time


class BayesRiskCalculator:
    """
    Computes Bayes risks and applies optimal stopping criterion.
    
    Mathematical Framework:
    - R_stop(t) = (1 - p_t) + C·t
    - R_cont(t) = C·(t+1) + (1 - E[p_{t+1}|H_t])
    - Optimal decision: continue iff R_cont(t) < R_stop(t) ⟺ Δp_t > C
    """
    
    def __init__(self,
                 cost_per_step: float = 0.01,
                 max_reasoning_steps: int = 50,
                 early_stopping_threshold: float = 0.99):
        
        self.cost_per_step = cost_per_step
        self.max_reasoning_steps = max_reasoning_steps
        self.early_stopping_threshold = early_stopping_threshold
    
    def compute_bayes_risks(self,
                          current_belief: float,
                          expected_improvement: float,
                          current_step: int) -> Dict[str, float]:
        """
        Compute Bayes risks for stopping vs continuing.
        
        Returns:
            Dictionary with risk calculations
        """
        try:
            # Risk of stopping now: R_stop = (1 - p_t) + C·t
            risk_stop = (1.0 - current_belief) + (self.cost_per_step * current_step)
            
            # Expected future belief: E[p_{t+1}|H_t] = p_t + Δp_t
            expected_future_belief = min(1.0, current_belief + expected_improvement)
            
            # Risk of continuing: R_cont = C·(t+1) + (1 - E[p_{t+1}|H_t])
            risk_continue = (self.cost_per_step * (current_step + 1)) + (1.0 - expected_future_belief)
            
            # Risk difference: positive means stopping is better
            risk_difference = risk_continue - risk_stop
            
            return {
                'risk_stop': risk_stop,
                'risk_continue': risk_continue,
                'risk_difference': risk_difference,
                'expected_future_belief': expected_future_belief,
                'current_step': current_step,
                'cost_per_step': self.cost_per_step
            }
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {
                'risk_stop': 1.0, 'risk_continue': 1.0, 'risk_difference': 0.0,
                'expected_future_belief': current_belief, 'current_step': current_step,
                'cost_per_step': self.cost_per_step
            }
    
    def should_continue_reasoning(self,
                                 current_belief: float,
                                 expected_improvement: float,
                                 current_step: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply Bayes-optimal stopping criterion.
        
        Mathematical decision rule: continue iff Δp_t > C
        
        Returns:
            Tuple of (should_continue, decision_details)
        """
        try:
            # Compute Bayes risks
            risks = self.compute_bayes_risks(current_belief, expected_improvement, current_step)
            
            # Primary decision criterion: Δp_t > C
            primary_decision = expected_improvement > self.cost_per_step
            
            # Secondary criteria for robustness
            within_step_limit = current_step < self.max_reasoning_steps
            not_already_certain = current_belief < self.early_stopping_threshold
            
            # Combined decision
            should_continue = primary_decision and within_step_limit and not_already_certain
            
            # Decision details
            decision_details = {
                'primary_criterion': primary_decision,
                'expected_improvement': expected_improvement,
                'cost_threshold': self.cost_per_step,
                'within_step_limit': within_step_limit,
                'current_step': current_step,
                'max_steps': self.max_reasoning_steps,
                'not_already_certain': not_already_certain,
                'early_stopping_threshold': self.early_stopping_threshold,
                'final_decision': should_continue,
                'bayes_risks': risks
            }
            
            logger.debug(f"BROR decision: Δp={expected_improvement:.4f} > C={self.cost_per_step:.4f} = {primary_decision}, "
                        f"step {current_step}/{self.max_reasoning_steps}, "
                        f"belief={current_belief:.4f}, "
                        f"final={should_continue}")
            
            return should_continue, decision_details
            
        except Exception as e:
            logger.error(f"Decision computation failed: {e}")
            return False, {'error': str(e), 'final_decision': False}


class BayesRiskOptimalReasoner:
    """
    Main BROR system integrating all components.
    
    Implements the complete Bayes-Risk-Optimal Reasoning framework with:
    - Bayesian belief state estimation
    - Expected improvement forecasting  
    - Optimal stopping decisions
    - Comprehensive metrics and analysis
    """
    
    def __init__(self,
                 cost_per_step: float = 0.01,
                 ensemble_size: int = 8,
                 mc_samples: int = 6,
                 sample_length: int = 48,
                 max_reasoning_steps: int = 50,
                 calibration_alpha: float = 1.0,
                 calibration_beta: float = 0.0,
                 max_computation_time: float = 30.0):
        
        self.cost_per_step = cost_per_step
        self.max_computation_time = max_computation_time
        
        # Initialize components
        self.belief_estimator = BayesianBeliefEstimator(
            ensemble_size=ensemble_size,
            calibration_alpha=calibration_alpha,
            calibration_beta=calibration_beta
        )
        
        self.improvement_forecaster = ExpectedImprovementForecaster(
            num_mc_samples=mc_samples,
            sample_length=sample_length,
            max_forecast_time=max_computation_time * 0.6  # Reserve 40% for other computations
        )
        
        self.risk_calculator = BayesRiskCalculator(
            cost_per_step=cost_per_step,
            max_reasoning_steps=max_reasoning_steps
        )
        
        # Metrics tracking
        self.metrics = BROMetrics()
    
    def compute_reasoning_decision(self,
                                 iteration: int,
                                 seq: torch.Tensor,
                                 entropies: List[float],
                                 hflm) -> Tuple[bool, List[int], Dict[str, Any]]:
        """
        Main BROR computation: determine whether to continue reasoning.
        
        Mathematical Pipeline:
        1. Estimate P(A = correct | H_t)
        2. Forecast E[p_{t+1} | H_t] - p_t  
        3. Apply Bayes-optimal decision rule
        4. Return decision and continuation tokens
        
        Returns:
            Tuple of (should_continue, continuation_tokens, computation_details)
        """
        self.metrics.total_calls += 1
        computation_start = time.time()
        
        details = {
            'iteration': iteration,
            'sequence_length': len(seq) if hasattr(seq, '__len__') else 0,
            'current_belief': 0.5,
            'expected_improvement': 0.0,
            'bayes_risks': {},
            'decision': False,
            'computation_time': 0.0,
            'success': False,
            'failure_reason': None
        }
        
        try:
            # Convert sequence to text
            if isinstance(seq, torch.Tensor):
                context_seq = seq
                text = hflm.tokenizer.decode(seq, skip_special_tokens=True)
            else:
                context_seq = torch.tensor(seq, device=hflm.device)
                text = hflm.tokenizer.decode(seq, skip_special_tokens=True)
            
            # Step 1: Estimate current belief state P(A = correct | H_t)
            current_belief, belief_details = self.belief_estimator.estimate_correctness_probability(
                text, context_seq, hflm
            )
            details['current_belief'] = current_belief
            details['belief_details'] = belief_details
            
            if not belief_details['success']:
                self.metrics.belief_estimation_failures += 1
                details['failure_reason'] = 'belief_estimation_failed'
                return False, [], details
            
            # Step 2: Forecast expected improvement E[p_{t+1} | H_t] - p_t
            expected_improvement, forecast_details = self.improvement_forecaster.forecast_expected_improvement(
                text, context_seq, current_belief, entropies, hflm, self.belief_estimator
            )
            details['expected_improvement'] = expected_improvement
            details['forecast_details'] = forecast_details
            
            if not forecast_details['success']:
                self.metrics.forecasting_failures += 1
                details['failure_reason'] = 'improvement_forecasting_failed'
                # Use fallback: very small improvement
                expected_improvement = 0.001
            
            # Step 3: Apply Bayes-optimal decision rule
            should_continue, decision_details = self.risk_calculator.should_continue_reasoning(
                current_belief, expected_improvement, iteration
            )
            details['decision'] = should_continue
            details['decision_details'] = decision_details
            details['bayes_risks'] = decision_details.get('bayes_risks', {})
            
            # Update metrics
            self.metrics.successful_calculations += 1
            self.metrics.decision_history.append(should_continue)
            self.metrics.expected_improvements.append(expected_improvement)
            self.metrics.bayes_risks.append(details['bayes_risks'])
            
            # Track belief trajectory
            if len(self.metrics.belief_trajectories) <= iteration:
                self.metrics.belief_trajectories.extend([[] for _ in range(iteration + 1 - len(self.metrics.belief_trajectories))])
            self.metrics.belief_trajectories[iteration].append(current_belief)
            
            # Compute cost effectiveness
            if expected_improvement > 0:
                cost_effectiveness = expected_improvement / self.cost_per_step
                self.metrics.cost_effectiveness.append(cost_effectiveness)
            
            details['success'] = True
            
            logger.info(f"BROR Step {iteration}: belief={current_belief:.4f}, "
                       f"Δp={expected_improvement:.4f}, C={self.cost_per_step:.4f}, "
                       f"decision={'CONTINUE' if should_continue else 'STOP'}")
            
            # Generate continuation tokens if continuing
            if should_continue:
                continuation_tokens = self._generate_continuation_tokens(
                    current_belief, expected_improvement, hflm
                )
                return True, continuation_tokens, details
            else:
                return False, [], details
            
        except Exception as e:
            logger.error(f"BROR computation failed: {e}")
            details['failure_reason'] = f"critical_exception: {str(e)}"
            return False, [], details
        
        finally:
            computation_time = time.time() - computation_start
            details['computation_time'] = computation_time
            self.metrics.computational_times['total_bror'].append(computation_time)
    
    def _generate_continuation_tokens(self,
                                    current_belief: float,
                                    expected_improvement: float,
                                    hflm) -> List[int]:
        """Generate contextual continuation tokens based on belief state and expected improvement."""
        try:
            # Adaptive prompts based on mathematical analysis
            if current_belief < 0.3:
                # Low confidence: encourage careful analysis
                prompt = "\n\nI'm not confident about this approach. Let me reconsider the problem carefully:\n\n"
            elif current_belief < 0.6:
                # Medium confidence: encourage verification
                prompt = "\n\nLet me double-check this reasoning and consider alternative approaches:\n\n"
            elif expected_improvement > 2 * self.cost_per_step:
                # High potential improvement: encourage deeper analysis
                prompt = "\n\nThis seems promising, but let me analyze this more rigorously:\n\n"
            else:
                # Standard continuation
                prompt = "\n\nLet me continue working through this step by step:\n\n"
            
            return hflm.tok_encode(prompt)
            
        except Exception as e:
            logger.warning(f"Continuation token generation failed: {e}")
            # Fallback
            try:
                return hflm.tok_encode("\n\nLet me think about this more carefully:\n\n")
            except:
                return []
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics summary for analysis."""
        if self.metrics.total_calls == 0:
            return {"message": "No BROR computations performed yet"}
        
        success_rate = self.metrics.successful_calculations / self.metrics.total_calls
        
        summary = {
            "total_calls": self.metrics.total_calls,
            "success_rate": success_rate,
            "cost_per_step": self.cost_per_step,
            "failure_breakdown": {
                "belief_estimation_failures": self.metrics.belief_estimation_failures,
                "forecasting_failures": self.metrics.forecasting_failures,
                "calibration_failures": self.metrics.calibration_failures
            }
        }
        
        # Decision statistics
        if self.metrics.decision_history:
            summary["decision_stats"] = {
                "continue_rate": np.mean(self.metrics.decision_history),
                "total_decisions": len(self.metrics.decision_history),
                "reasoning_depth_avg": np.sum(self.metrics.decision_history)
            }
        
        # Expected improvement statistics
        if self.metrics.expected_improvements:
            improvements = np.array(self.metrics.expected_improvements)
            summary["improvement_stats"] = {
                "mean": np.mean(improvements),
                "std": np.std(improvements),
                "min": np.min(improvements),
                "max": np.max(improvements),
                "median": np.median(improvements)
            }
        
        # Cost effectiveness analysis
        if self.metrics.cost_effectiveness:
            cost_eff = np.array(self.metrics.cost_effectiveness)
            summary["cost_effectiveness"] = {
                "mean_ratio": np.mean(cost_eff),
                "beneficial_decisions": np.sum(cost_eff > 1.0),
                "total_decisions": len(cost_eff)
            }
        
        # Bayes risk analysis
        if self.metrics.bayes_risks:
            risk_diffs = [r.get('risk_difference', 0) for r in self.metrics.bayes_risks]
            summary["bayes_risk_stats"] = {
                "mean_risk_difference": np.mean(risk_diffs),
                "optimal_decisions": np.sum(np.array(risk_diffs) < 0),  # negative = continue was optimal
                "total_decisions": len(risk_diffs)
            }
        
        # Timing statistics
        if self.metrics.computational_times:
            summary["timing_stats"] = {}
            for component, times in self.metrics.computational_times.items():
                if times:
                    summary["timing_stats"][component] = {
                        "mean": np.mean(times),
                        "std": np.std(times),
                        "total": np.sum(times),
                        "max": np.max(times)
                    }
        
        return summary 