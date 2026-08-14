import re,sys
t=open(sys.argv[1],encoding='utf-8').read()
heads={m.group(1) for m in re.finditer(r'^#{2,4} (\d+(?:\.\d+)?)[\. ]', t, re.M)}
bad=[r for r in set(re.findall(r'见 (\d+\.\d+)', t)) if r not in heads]
# 只查 bash 代码块内的 <...> —— 那里 < 是重定向符，粘贴即报语法错。
# 散文里的 <...> 无害，不要为了让检查通过而放宽规则（放宽会掩盖真问题）。
blocks='\n'.join(re.findall(r'```bash\n(.*?)```', t, re.S))
ph=[x for x in set(re.findall(r'<[^>\n]{2,25}>', blocks)) if '/dev' not in x]
print(f"  章节: {sorted(heads)}")
print(f"  ✗ 悬空交叉引用: {bad}" if bad else "  ✓ 交叉引用全部指向存在的章节")
print(f"  ⚠ 残留 <...> 占位符: {ph}" if ph else "  ✓ 无 <...> 占位符")
sys.exit(1 if bad else 0)
