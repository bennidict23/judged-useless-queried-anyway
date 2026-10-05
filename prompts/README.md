# Prompts

Every prompt used in the paper, verbatim.

| file | used in | where it goes |
|---|---|---|
| `01_agent_system_hotpotqa.txt` | every HotpotQA condition except Search-R1 | system prompt |
| `02_agent_system_fever.txt` | FEVER | system prompt |
| `03_condition_permit_suffix.txt` | permit, and every condition built on it | appended to the system prompt |
| `04_condition_budget_suffix.txt` | budget, call cost, combination | appended after the permit suffix |
| `05_condition_budget_counter_example.txt` | budget, call cost, combination | added to the question and to every observation (the number counts down) |
| `06_condition_stated_rule_suffix.txt` | stated rule | appended after the permit suffix |
| `07_condition_call_cost_suffix.txt` | call cost | appended after the budget suffix |
| `08_side_channel_judgment_question.txt` | side-channel judgments (enforced rule; replayed judgments for the other conditions) | asked on a scratch copy of the conversation after each observation; the reply is discarded from the acting history |
| `09_enforced_rule_system_suffix.txt` | enforced rule, when it fires | appended to the system prompt |
| `10_enforced_rule_user_message.txt` | enforced rule, when it fires | appended to the latest observation |
| `11_decide_condition_prompt.txt` | decide | after each observation, with the agent's judgment `{j}` and its running count `{k}` |
| `12_decide_condition_act_instruction.txt` | decide | after the decision, asking for an action consistent with it |
| `13_belief_probe_usefulness_turn.txt` | belief probe | asked on a scratch copy at each decision point |
| `14_belief_probe_continuation_turn.txt` | belief probe | asked on a separate scratch copy at each decision point |
| `15_backup_tool_line.txt` | backup tool | added to the list of actions |
| `16_backup_tool_switch_notice.txt` | backup tool, when the failing tool is disabled | appended to the observation |
| `17_search_r1_prompt.txt` | Search-R1 | its own prompt, as released |
| `18_search_r1_answer_now.txt` | Search-R1 with the enforced rule | appended when the rule fires |
