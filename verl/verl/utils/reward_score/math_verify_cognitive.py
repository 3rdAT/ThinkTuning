# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    import re
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


import re

def check_format_bool(model_output: str) -> bool:
    try:
        # Must occur exactly once and be at the end
        token = "<|eot_id|>"
        count = model_output.count(token)
        return count == 1 and model_output.endswith(token)
    except Exception:
        return False

# # matches '\boxed{...}<|eot_id|>' at the very end of the string
# _END_PATTERN = re.compile(r'\\boxed\{.*\}<\|eot_id\|>$', re.DOTALL)


# def check_format_bool(model_output: str) -> bool:
#     """True iff `model_output` ends with '\boxed{…}<|eot_id|>'."""
#     return bool(_END_PATTERN.search(model_output))


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    format_bool = check_format_bool(model_output)

    if format_bool:
        # Wrap the ground truth in \boxed{} format for verification
        ret_score += 0.0
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        try:
            ret1_score, _ = verify_func([ground_truth_boxed], [model_output])
            ret_score += ret1_score
        except Exception:
            pass
        except TimeoutException:
            ret1_score = timeout_score
            ret_score += ret1_score
    
    return ret_score