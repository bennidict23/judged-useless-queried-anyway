<div align="center">

# Judged Useless, Queried Anyway

**Tool-Using Agents Rarely Turn Their Own Evidence Judgments into Stopping Decisions**

Chubin Zhang<sup>1</sup>, Zhenglin Wan<sup>2</sup>, Xingrui Yu<sup>3</sup>, Jingxuan Wu<sup>4</sup>, Yaxin Zhou<sup>5</sup>, Ivor Tsang<sup>1,3</sup>, Bo An<sup>1</sup>

<sub><sup>1</sup>Nanyang Technological University &nbsp; <sup>2</sup>National University of Singapore &nbsp; <sup>3</sup>A\*STAR &nbsp; <sup>4</sup>UNC-Chapel Hill &nbsp; <sup>5</sup>Carnegie Mellon University</sub>

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2610.06191-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.06191)
[![Data](https://img.shields.io/badge/Data-90k_episodes-2ea44f?style=for-the-badge&logo=json&logoColor=white)](experiments/episodes)
[![Python](https://img.shields.io/badge/Python-3.9%2B_·_no_dependencies-3776ab?style=for-the-badge&logo=python&logoColor=white)](#-quick-start)
[![License](https://img.shields.io/badge/License-MIT-555555?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

🚀 **[Quick start](#-quick-start)** &nbsp;·&nbsp; 🔍 **[Test your agent](#-test-your-agent)** &nbsp;·&nbsp; 🔧 **[Integration step](#-add-the-integration-step)** &nbsp;·&nbsp; 📊 **[Data](#-data)** &nbsp;·&nbsp; 📝 **[Citation](#-citation)**

</div>

<br>

<p align="center"><img src="docs/teaser.png" width="100%"></p>

When a tool keeps returning nothing useful, an agent should stop relying on it. The agents we test know when that
happens, but they do not act on it:

- **Agents judge correctly but do not act on their judgments.** They call a failing source's results useless 97–100%
  of the time, yet their stopping ignores these judgments.
- **Telling agents more changes when they stop, not what they stop on.** Permission to answer from memory, a step
  budget, a stopping rule or a price per call written into the prompt, or the running count of useless judgments,
  moves the stopping point without tying it to the evidence.
- **An enforced integration step makes stopping follow the evidence.** When the harness leaves only the answer action
  after five consecutive results the agent judged useless, success on a failing source rises for every model.
- **The pattern replicates** on 300 fresh questions and on fact verification. A larger open model, a reasoning mode
  and an RL-trained search agent still largely fail to stop on the evidence.

## 🚀 Quick start

```bash
git clone https://github.com/bennidict23/judged-useless-queried-anyway.git
cd judged-useless-queried-anyway
python experiments/reproduce_paper.py   # every Δ in Figure 3, from the released episodes
pip install -e .                        # the toolkit, to measure Δ on your own agent
```

## 🔍 Test your agent

Success alone cannot tell you what an agent's stopping responds to: an agent that stops at a fixed step, one that
waits for the deadline and one that stops after enough useless evidence can score alike. The **time-matched contrast
Δ** separates them. At the same step, it compares how often the agent answers when every result so far was judged
useless with how often it answers when the latest result was judged useless but an earlier one was judged useful.

| Δ | the agent's stopping ... |
|:---:|---|
| **> 0** | integrates its own useless judgments |
| **≈ 0** | follows the clock or the deadline |
| **< 0** | follows earlier useful evidence instead |

To measure it on your own agent, log each episode's actions and the agent's one-word judgment of each observation,
and pass the episodes to the toolkit. The format and a short example are in [judged_useless/README.md](judged_useless/README.md).

Across the models and conditions in the paper, only the conditions with the enforced rule make Δ positive for every
model (in parentheses: the replication on 300 fresh questions):

<p align="center"><img src="docs/delta_heatmap.png" width="800"></p>

## 🔧 Add the integration step

Telling the agent more does not make it stop on its judgments, so let the harness do it: once five results in a row
are judged useless, leave the agent only the answer action. The toolkit's integration rule reads the judgments either
from a one-word side-channel question or directly from the agent's own reasoning, at no extra cost
([usage](judged_useless/README.md#enforce-the-integration-step)).

## 📊 Data

[experiments/episodes](experiments/episodes) holds every episode of the paper's main experiments: four open models in eight conditions on 300
test questions, and the replication on 300 fresh questions. Each episode records the agent's actions, its judgment of
every observation and whether it answered correctly. From these files, the reproduction script recomputes every Δ in
Figure 3 and the open models' success rates in Table 2 on a CPU in about a minute.

## 🧪 Source-failure environment

Questions come from the HotpotQA distractor set, each with its own knowledge base of ten paragraphs, and the agent has
three actions (search, lookup, finish) and a budget of eight. A failed observation is replaced by a paragraph from
another question's knowledge base, so whether each result is useful is known by construction.

| failure regime | which observations fail |
|---|---|
| clean | none |
| persistent | every observation |
| recover after 1, 2 or 3 | the first 1, 2 or 3 |
| late onset from 3 | the third and every later one |

Harder failures, longer recoveries, a backup tool and FEVER fact verification are included too. Open models run
locally with vLLM and Claude models through the Anthropic API. [experiments/README.md](experiments/README.md) gives the
commands for every model and condition, and every prompt is in [experiments/prompts](experiments/prompts).

## 📝 Citation

```bibtex
@misc{zhang2026judgeduseless,
  title = {Judged Useless, Queried Anyway: Tool-Using Agents Rarely Turn Their Own Evidence Judgments into Stopping Decisions},
  author = {Zhang, Chubin and Wan, Zhenglin and Yu, Xingrui and Wu, Jingxuan and Zhou, Yaxin and Tsang, Ivor and An, Bo},
  year = {2026},
  eprint = {2610.06191},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2610.06191}
}
```

## 📜 License

Code: MIT. Episode data: CC BY 4.0. HotpotQA (CC BY-SA 4.0) and the FEVER claims and Wikipedia pages (CC BY-SA 3.0)
keep their original licenses.
