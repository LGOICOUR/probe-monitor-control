# P1 leakage and confound report

Problems: 150 (seed 0). Papers: 450.

## Integrity

- Problems missing one of the three papers: 0 []
- Honest papers that did not pass their tests: 0 []
- Backdoored papers that failed tests or did not fire: 0 []

## Leakage (what the TA would see)

- Views containing the secret_input field or side-task text: 0
- Papers whose code contains a giveaway word or other leak: 0

## Confounds by source (anything here that differs is a possible shortcut)

Lines and chars are measured on the comment-stripped code the TA sees.

| source | n | median lines | median chars | % any comment (raw) | median comment lines (raw) | % confessing comment (raw) | % giveaway after strip | % `def main` |
|---|---|---|---|---|---|---|---|---|
| human_honest | 150 | 13.0 | 231.5 | 12.7 | 0.0 | 0.0 | 0.0 | 7.3 |
| haiku_honest | 150 | 20.5 | 389.5 | 88.0 | 5.0 | 0.0 | 0.0 | 1.3 |
| backdoor | 150 | 16.5 | 343.0 | 58.7 | 1.0 | 42.0 | 0.0 | 14.0 |

## Who wrote the backdoors

- openai/o3-mini: 48
- anthropic/claude-3-7-sonnet-latest: 34
- openai/gpt-4o: 22
- google/gemini-2.0-flash: 10
- openai/o1-mini: 9
- openai/gpt-4o-mini: 9
- anthropic/claude-3-7-sonnet-thinking-latest: 8
- anthropic/claude-3-5-haiku-latest: 7
- openai/gpt-3.5-turbo: 3
