# LLM-judge prompt and rubric

Every answer is scored **independently** (one API call per answer; the judge never sees the
model name or any other model's answer). The judge returns four integer sub-scores that sum to a
total out of 10.

## Rubric (verbatim)

```
Score four dimensions, then nothing else:
- "correctness" (0-5): factual agreement with the answer. 5 = fully right, 0 = wrong.
- "completeness" (0-2): covers the key points, not just one of them.
- "groundedness" (0-2): 2 = invents nothing; 0 = fabricated figures, cases or citations.
- "clarity" (0-1): answers what was actually asked, without padding or contradiction.
- "reason": one short sentence.
The four add up to a score out of 10. Judge meaning, not wording.
```

## Output schema (JSON)

```json
{
  "correctness":  "integer 0-5",
  "completeness": "integer 0-2",
  "groundedness": "integer 0-2",
  "clarity":      "integer 0-1",
  "reason":       "string"
}
```

`total = correctness + completeness + groundedness + clarity` (0–10).
`grounded = (groundedness == 2)`.

## Full prompt construction

```
You are a strict evaluator of a question-answering system. Judge ONLY the CANDIDATE
answer; you do not know which system produced it.
<reference rule — see below>

<RUBRIC (above)>
Groundedness is judged against the CONTEXT.        # included only when a context is present

CONTEXT:
<context passage>                                   # included only when a context is present

QUESTION:
<question>

REFERENCE:
<gold answer>                                       # included only when a reference is present

CANDIDATE:
<model answer>
```

- **Reference rule (when a gold answer is given):** "The REFERENCE is a correct short answer.
  Grade the CANDIDATE against it. Judge meaning, not wording — a correct paraphrase scores high."
- **Reference rule (no gold answer):** "No reference answer is available. Grade the CANDIDATE on
  factual accuracy and whether it actually answers the question, using your own knowledge. Be
  strict: vague, evasive, repetitive or fabricated answers score low."

Judge decoding: **temperature 0**. The primary judge in the paper is a Gemini-class model; a
second independent judge (a GPT-class model) was used for a cross-judge robustness check with the
identical prompt. `score.py` implements this prompt exactly.
