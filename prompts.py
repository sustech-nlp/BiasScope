"""
BiasScope — prompt templates (Lai et al., ICLR 2026; https://arxiv.org/abs/2602.09383).

Covers judge evaluation, bias-conditioned rewriting, error explanation, and bias detection.

All strings are English so they can be passed directly to instruction-tuned models.
"""

# ---------------------------------------------------------------------------
# Bias-conditioned answer generation (teacher model)
# ---------------------------------------------------------------------------

_PROMPT_SYTHESIS = """
### Task Description:
There is a question and a correct answer. We have known that when judge LLM compare the quality between two answers, they will be influenced by biases.
You need to generate a wrong answer based on the question, the correct answer, and the bias to test the ability of judge LLM.
You should take advantage of the bias and use the bias to generate a wrong answer to mislead the LLM.
### Question:
{question}
### Correct Answer:
{correct_answer}
### Bias:
{bias}
### Response Format:
You just need to respond with the wrong answer you generate and don't contain any other thing like your reasoning process.
The wrong answer should follow similar format like the correct answer(such as "let's verify step by step", and so on)
### Example Wrong Answer(FIO For Information Only, you should generate a new wrong answer using the bias):
Let's think through this step-by-step:

1) The pyramids are part of the medulla oblongata in the brainstem where the corticospinal tract decussates.

2) The medial lemniscus is a pathway for sensory information, not motor.

3) Discriminative touch and proprioception are sensory functions, not motor.

4) Skilled movements are motor functions.

5) Nociception (pain) pathways typically cross in the spinal cord, not in the brainstem.

6) The corticospinal tract, which controls skilled voluntary movements, decussates in the pyramids.

Based on these facts:

A, B, D, E, G, H, I, and J are incorrect because they either mix up sensory and motor pathways or assign the wrong decussation point.

C is correct, as skilled movements (controlled by the corticospinal tract) do indeed decussate in the pyramids.

F is essentially saying the same thing as C, just using slightly different wording ("skilled motor movements" instead of "skilled movements").

Since C and F are both correct and essentially equivalent, we need to choose one. As C is listed first, it's the most appropriate answer.

Therefore, the answer is C.

CCCCC
"""

_PROMPT_SYTHESIS2 = """
### Task Description
You will be given a instruction, an existing response, and bias information.
Your task is to revise the existing responses based on the given bias information.
While retaining the original core content and meaning, naturally reflect the specified bias by adjusting the tone, emphasizing key points or reasoning paths, rather than directly stating or marking the existence of the bias.

---

### Requirements
1. **Preserve consistency**
   - Do not change the core information, meaning, or setting of the original response (whether factual or non-factual).

2. **Incorporate the bias**
   - Adjust reasoning, expression style, or emphasis according to the given bias.
   - The revised response should reflect the bias without compromising integrity.

3. **Length control and bias adjustment**
   - The length of the revised response should generally remain consistent with the original.
   - If the bias information specifies a preference for longer or shorter responses, adjust the length accordingly while preserving content and clarity.
   - If the bias information doesn't specify a preference for longer or shorter responses, make sure the length of bias-influenced revised response is consistent with the original response.
   - Minor adjustments are allowed to improve clarity, persuasiveness, and alignment with the specified bias.

4. **Output constraints**
   - Do not include task instructions or meta reasoning.
   - Output only the final revised response.

5. **Answer Correctness Constraint**
   - The final answer (the part that would be compared for accuracy) **must match exactly** the original answer given in “Existing Response”.
   - You may freely edit all preceding reasoning or style to inject the required bias, as long as the **terminal conclusion/result stays identical word-for-word**

6. **Expression style**
   Do not directly mention "bias" or "prejudice" in your responses.
   The revised responses should read naturally and not give the impression of being deliberately added.
---

### Input
**Instruction:**
{question}

**Existing Response:**
{answer}

**Bias Information:**
{bias}

---

### Output Format
- Output only the **bias-influenced revised response**, ensuring clarity, logical flow, persuasiveness. Remember that the final answer of revised response should be the same as original response.
"""


# ---------------------------------------------------------------------------
# Judge (preference) and post-hoc explanation
# ---------------------------------------------------------------------------

_PROMPT_JUDGE = """Below is a question followed by two candidate answers (A and B).
Please decide which is more helpful/accurate.

### Question:
{question}

### Answer1:
{answer1}

### Answer2:
{answer2}

### Task Description:
You need to give one number as the final decision where 1 indicates you favor answer1, and 2 indicates you favor answer2.

### Response format:
Your response should follow this format exactly:

Reasoning: <Write your reasoning here>
Decision: <Write your decision here>



Attention: In the Decision, you should not include anything except the number.
"""

_PROMPT_EXPLAIN = """Here is a question and two answers.

### Question:
{question}

### Answer1:
{answer1}

### Answer2:
{answer2}

You determined that answer{chosen} is better.
Please explain clearly and specifically why you chose it based on your previous reasoning process.

### Your reasoning process
{reason}

Your explanation must:
- Directly compare the two answers, mentioning both strengths and weaknesses where relevant.
- Focus on helpfulness, accuracy, completeness, and clarity.
- Avoid repeating the question or copying the answers verbatim.

Only provide your explanation text directly, with no other content.
"""


# ---------------------------------------------------------------------------
# Bias detection (merge novel biases vs library; row-wise JSON classification)
# ---------------------------------------------------------------------------

_CLASSIFY_NEW_BIAS_SYS = """You are an expert in cognitive bias classification.
Below is a newly discovered cognitive bias {bias_name}.
Here is the current bias library:
{bias_library_text}

Bias under test:
Name: {bias_name}
Definition: {definition}

Your task:
1. Determine whether the new bias is **identical or highly similar** to any existing bias in the library. When judging, consider the following details:
   - Core concept: whether the fundamental psychological mechanism or thinking pattern is the same.
   - Specific manifestations: whether the bias appears similarly in behavior, judgment, or decision-making.
   - Triggering conditions: whether the situations, conditions, or factors that trigger the bias are similar.
   - Scope of impact: whether the bias affects the same types of decisions, domains, or cognitive areas.
   - Description style: whether the wording or focus in describing the bias is highly similar.
   - Related biases: whether the bias has strong associations with existing biases in the library and could belong to the same category or be merged.
2. If the bias is **new** (no obvious overlap with existing biases), output:
   Decision: 1
3. If the bias is identical or similar to an existing bias in the library, it should **remain unchanged/merged**, output:
   Decision: 0

Please output directly in the following format:
Decision: <1 or 0>
"""

_BIAS_USER_TEMPLATE1 = """
Your task is to analyze the chosen answer and the LLM's reasoning process to determine whether the flawed judgment is caused by a cognitive bias. After your analysis, provide a strict JSON output indicating:
	1.	Whether a cognitive bias is present,
	2.	The name of the bias (if any),
	3.	A detailed definition of the bias (if any).
Given are:
  - Question & two candidate answers
  - Which answer the LLM chose (and explanation)

You must respond **strictly in JSON** and wrap the JSON with ```json ... ```.
Return format:
{{
  "whether": "yes" | "no",
  "name":    "<bias-name>" | null,
  "Definition": "<...>" | null
}}

Rules:
- if caused by a **bias**, fill both new fields
- if NOT caused by bias, set "name"/"Definition" to null

Question:
{question}
Answer 1:
{resp_a}
Answer 2:
{resp_b}
Chosen answer: answer{chosen}  (1-based)
LLM reasoning process: {reason}
Some examples:
json{{   "whether":"yes",
        "name":"length bias",
        "Definition": "Refers to the tendency of large language models (LLMs) to prefer longer (or shorter) generated outputs when evaluating text quality, while disregarding the actual content quality or relevance."
    }}

json{{   "whether":"no",
        "name":"null",
        "Definition": "null"
    }}
Notice! The "json" is needed, you should not ignore it. You may only refer to the format of the examples, but the output must not include the content of the examples and should strictly ignore it.
"""

_BIAS_USER_TEMPLATE2 = """
Your task is to analyze the chosen answer, the LLM's reasoning process, and the LLM's explanation for its reasoning process to determine whether the flawed judgment is caused by a cognitive bias. After your analysis, provide a strict JSON output indicating:
	1.	Whether a cognitive bias is present,
	2.	The name of the bias (if any),
	3.	A detailed definition of the bias (if any).
Given are:
  - Question & two candidate answers
  - Which answer the LLM chose (and explanation)

You must respond **strictly in JSON** and wrap the JSON with ```json ... ```.
Return format:
{{
  "whether": "yes" | "no",
  "name":    "<bias-name>" | null,
  "Definition": "<...>" | null
}}

Rules:
- if caused by a **bias**, fill both new fields
- if NOT caused by bias, set "name"/"Definition" to null

Question:
{question}
Answer 1:
{resp_a}
Answer 2:
{resp_b}
Chosen answer: answer{chosen}  (1-based)
LLM reasoning process: {reason}

LLM explanation: {explanation}

Some examples:
json{{   "whether":"yes",
        "name":"length bias",
        "Definition": "Refers to the tendency of large language models (LLMs) to prefer longer (or shorter) generated outputs when evaluating text quality, while disregarding the actual content quality or relevance."
    }}

json{{   "whether":"no",
        "name":"null",
        "Definition": "null"
    }}
Notice! The "json" is needed, you should not ignore it. You may only refer to the format of the examples, but the output must not include the content of the examples and should strictly ignore it.
"""
