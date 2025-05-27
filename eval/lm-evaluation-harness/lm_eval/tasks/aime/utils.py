from collections import Counter
import os
import re
import signal
from typing import Dict, List, Optional, Union

import datasets

from lm_eval.utils import eval_logger

if os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTSTEP") + ' steps.'
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTTOKEN") + ' tokens.'
elif os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.'
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work.'
else:
    QUERY_TEMPLATE = '{Question}'

# The correct answer is an integer between $000$ and $999$, inclusive. Keep thinking until your answer is in the correct range.
# The correct answer is an integer between $000$ and $999$, inclusive.

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

# Adapted from https://github.com/openai/simple-evals/blob/c0dba4c7bfbc17f786aec7bd7c3585a36ad81f23/common.py#L23
# (?i): Enables case-insensitive matching. This means "Answer", "answer", "ANSWER", etc., will all be matched.
# Answer: Matches the literal string "Answer" (case-insensitive due to (?i)).
# \s*: Matches zero or more whitespace characters (spaces, tabs, etc.) after "Answer". This accounts for cases where there might or might not be space between "Answer" and the colon (:).
# :: Matches the literal colon character :.
# \s*: Matches zero or more whitespace characters after the colon. This handles cases where there might be spaces between the colon and the actual answer.
# (.*): The .* matches zero or more of any character (including none), except for newlines unless re.DOTALL is used (which allows newlines to be matched too).
# Note: This does not match e.g. "**Final Answer:** A" as it only matches "Answer: A" or "Answer: A) 7" etc.
ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"

EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['11', '100', '50', '-5', '12', '10']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

5

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK


Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


# https://github.com/openai/simple-evals/blob/580d359553a88584c11ce4efb97d49d9386e0d9e/common.py#L153C1-L156C45
def extract_answer_idx(sampler, options: List[str], attempt: str):
    prompt = EXTRACTION_TEMPLATE_IDX % {"expression1": options, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response

import time
from typing import Any

import openai
from openai import OpenAI

class ChatCompletionSampler:
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc.get("problem", doc.get("question")))

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("question"))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)

def process_results(doc: dict, results: Union[List[str], str]) -> Dict[str, Any]:
    eval_logger.info(f"DEBUG - process_results received results type: {type(results)}")
    if isinstance(results, list):
        eval_logger.info(f"DEBUG - process_results received list of length: {len(results)}")
    else:
        eval_logger.info(f"DEBUG - process_results received single string.")

    # If a single string is passed (e.g., n=1 or task doesn't use n), wrap it in a list for uniform processing.
    if isinstance(results, str):
        results = [results]

    metrics: Dict[str, Any] = {"exact_match": None, "extracted_answers": -1} # Default extracted_answers to -1

    gt = str(doc["answer"])
    if gt.isdigit():
        gt = str(int(gt))

    split_tokens = ["<|im_start|>answer\n", "<|im_start|>"]
    
    all_extracted_numbers_for_doc = []
    all_exact_match_flags_for_doc = []

    for i_sample, a_raw in enumerate(results, start=1):
        a = a_raw # Make a copy to modify
        original_a_for_debug = a 

        if split_tokens[0] in a:
            a = a.split(split_tokens[0])[-1]
        elif split_tokens[1] in a:
            a = a.split(split_tokens[1])[-1]
            if "\n" in a:
                a = "\n".join(a.split("\n")[1:])

        if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]

        processed_a_text = str(a).strip() # This is the text we'll try to convert to a number

        # Fallback extraction logic
        extracted_num_this_sample = -1
        if processed_a_text.isdigit() and 0 <= int(processed_a_text) <= 999:
            extracted_num_this_sample = int(processed_a_text)
        else:
            # Try basic number extraction as fallback if not directly a digit or out of AIME range
            eval_logger.warning(f"Non-numeric or out-of-range extraction '{processed_a_text[:100]}...', trying fallback.")
            numbers_found = re.findall(r'\b(\d+)\b', processed_a_text)
            if numbers_found:
                candidate_num_str = numbers_found[-1] # Take last number
                if candidate_num_str.isdigit() and 0 <= int(candidate_num_str) <= 999:
                    extracted_num_this_sample = int(candidate_num_str)
                    eval_logger.info(f"Fallback extracted: '{extracted_num_this_sample}' from '{processed_a_text[:50]}'")
                else:
                    eval_logger.warning(f"Fallback number '{candidate_num_str}' out of AIME range.")
                    extracted_num_this_sample = 999 # Placeholder for out-of-range/invalid
            else:
                eval_logger.warning(f"No numbers found in fallback for '{processed_a_text[:50]}'.")
                extracted_num_this_sample = 999 # Placeholder for no number found

        all_extracted_numbers_for_doc.append(extracted_num_this_sample)
        
        # Compare the processed text `a` for exact match purposes against string GT
        is_correct_this_sample = int(str(a if not str(a).isdigit() else str(int(a))) == gt)
        all_exact_match_flags_for_doc.append(is_correct_this_sample)

        eval_logger.info(f"DEBUG - Sample {i_sample}: Raw='{original_a_for_debug[:100]}...', ProcessedText='{str(a)}', ExtractedNum={extracted_num_this_sample}, GT='{gt}', EM={is_correct_this_sample}")

    # --- After processing all samples for the doc ---
    
    # For single "exact_match" metric (often first sample or if n=1)
    if all_exact_match_flags_for_doc:
        metrics["exact_match"] = all_exact_match_flags_for_doc[0]

    # Final "extracted_answers" for the doc (e.g. majority vote on extracted numbers)
    if all_extracted_numbers_for_doc:
        valid_extracted_numbers = [num for num in all_extracted_numbers_for_doc if num != -1 and num != 999] # Exclude placeholders
        if valid_extracted_numbers:
            counts = Counter(valid_extracted_numbers)
            metrics["extracted_answers"] = counts.most_common(1)[0][0]
        else: # No valid numbers extracted across all samples
            metrics["extracted_answers"] = -1 
            # If there were samples but all led to -1/999, we might pick the first placeholder
            if all_extracted_numbers_for_doc:
                 metrics["extracted_answers"] = all_extracted_numbers_for_doc[0]


    num_total_samples = len(all_exact_match_flags_for_doc)
    if num_total_samples > 1:
        # Ensure 'exact_matches' (plural) for per-sample EM results if it's in metric_list
        # This depends on how your YAML is set up for what gets aggregated.
        # For now, let's assume it's not directly stored back into metrics unless YAML asks for it.
        # metrics["exact_matches"] = all_exact_match_flags_for_doc 

        # Calculate cov@k and maj@k
        # Define k values based on common practice or YAML (e.g. 2, 4, 8 if num_total_samples >= 8)
        k_values_for_metrics = []
        if num_total_samples >= 2: k_values_for_metrics.append(2)
        if num_total_samples >= 4: k_values_for_metrics.append(4)
        if num_total_samples >= 8: k_values_for_metrics.append(8)
        # Add more k as needed, or derive from n_res_list as before

        for k in k_values_for_metrics:
            if k > num_total_samples: continue

            first_k_em_flags = all_exact_match_flags_for_doc[:k]
            
            # Coverage @ k
            metrics[f"cov@{k}"] = int(1 in first_k_em_flags)
            
            # Majority @ k
            if first_k_em_flags:
                em_counts_at_k = Counter(first_k_em_flags)
                # Majority if count of '1's is > k/2
                # Or if count of '1's is k/2 and it's the only one (or tie-break if needed)
                if em_counts_at_k.get(1, 0) > k / 2.0:
                    metrics[f"maj@{k}"] = 1
                elif k > 1 and em_counts_at_k.get(1, 0) == k / 2.0 and em_counts_at_k.get(1,0) >= em_counts_at_k.get(0,0): # Tie break for 1s
                     metrics[f"maj@{k}"] = 1
                else:
                    metrics[f"maj@{k}"] = 0
            else: # Should not happen if k <= num_total_samples and num_total_samples > 0
                metrics[f"maj@{k}"] = 0


    eval_logger.info(f"FINAL DEBUG - Metrics for doc '{doc.get('id', 'N/A')}': {metrics}")
    return metrics

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]
