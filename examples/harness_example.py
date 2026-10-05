"""The enforced integration step in a minimal agent loop (no model needed).

The toy agent faces a search tool that has stopped working. Like the agents in the paper, it says each result is
not relevant but keeps searching. The rule reads that judgment from its thought and, after five useless results in
a row, leaves only the answer action.

    python examples/harness_example.py
"""
from judged_useless import FORCE_SYSTEM_SUFFIX, FORCE_USER_MESSAGE, IntegrationRule, stated_judgment


class ToyAgent:
    def propose(self, system_prompt, history):
        if FORCE_SYSTEM_SUFFIX in system_prompt:
            return "I will answer from what I know.", "finish[musician]"
        if len(history) == 1:
            return "I should look up both people.", "search[Chris Jericho]"
        return "This result is not relevant to the question. Let me try another query.", "search[Gary Barlow]"


def failing_tool(action):
    return "Observation: The Battle of Lubiszyn was fought in 1945 ..."   # a paragraph about something else


def run(max_actions=8):
    agent, rule = ToyAgent(), IntegrationRule(k=5, source="stated")
    system_prompt = "You are a question-answering agent ..."
    history = ["Question: What profession do Chris Jericho and Gary Barlow have in common?"]
    for step in range(max_actions):
        thought, action = agent.propose(system_prompt, history)
        if step > 0:
            judgment = stated_judgment(thought)
            if rule.update(judgment) and not action.startswith("finish"):
                print(f"step {step}: judged {judgment}, run = {rule.run} -> the rule fires")
                system_prompt += FORCE_SYSTEM_SUFFIX
                history[-1] += "\n\n" + FORCE_USER_MESSAGE
                thought, action = agent.propose(system_prompt, history)
            else:
                print(f"step {step}: judged {judgment}, run = {rule.run}")
        print(f"        action: {action}")
        if action.startswith("finish"):
            return action
        history.append(failing_tool(action))
    return None


if __name__ == "__main__":
    print("final:", run())
