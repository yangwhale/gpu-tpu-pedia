# rules

规则的**唯一真源**是 `../../../rules/rules.seed.json`（导入 Firestore `tpuguru_rules` 后以库为准）。
这里不放副本，避免两处维护。

agent 通过 `scripts/lint.py` 读取，不要手抄规则内容进 prompt。
