"""
All scale functions return whether to continue reasoning and the scale token to use for continuation.
They all have the same signature, as well as additional arguments for specific scaling strategies.

```python
def scaler(
    iteration: int,
    tokens: List[int],
    uncertainties: List[float],
    lm: 'VLLM'
) -> Tuple[bool, List[int]]:
```
"""
from budget_forcing.scalers.entropy_thresholding import entropy_thresholding
from budget_forcing.scalers.step_wise_uncertainty_driven import step_wise_uncertainty_driven
from budget_forcing.scalers.uncertainty_driven_reevaluation import uncertainty_driven_reevaluation


__all__ = [
    "entropy_thresholding",
    "step_wise_uncertainty_driven",
    "uncertainty_driven_reevaluation",
]