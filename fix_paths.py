import os

# Fix flesch_comparison.py
content = open('flesch_comparison.py', encoding='utf-8').read()
old = 'files = [f for f in os.listdir(".") if f.endswith(".txt")]'
new = 'files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".txt")]'
content = content.replace(old, new)
open('flesch_comparison.py', 'w', encoding='utf-8').write(content)
print('flesch_comparison.py fixed')

# Fix compute_readability.py
content = open('compute_readability.py', encoding='utf-8').read()
content = content.replace(old, new)
open('compute_readability.py', 'w', encoding='utf-8').write(content)
print('compute_readability.py fixed')