"""
A built SFT source (data/sft/<name>/train.jsonl, written by the standard pipeline), read back as a task. Each line is
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
or the shorthand
    {"question": "...", "answer": "..."}
Used by tasks/registry.py for SFT mixtures.
"""

import json
from tasks.common import Task

class CustomJSON(Task):

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.conversations = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "messages" not in row:
                    row = {"messages": [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answer"]}]}
                self.conversations.append(row)

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return self.conversations[index]
