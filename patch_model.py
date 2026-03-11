"""
patch_model.py
Replaces manual formula with scientifically justified pure zipf sorting.

Scientific basis:
  Zipf (1935) - word frequency inversely predicts difficulty.
  Higher avg zipf in a cluster = easier words = LAYMAN.
  This is the most cited, most reliable single predictor of word difficulty.
"""
import re

content = open('model_training.py', encoding='utf-8').read()

old_func = re.search(
    r'def assign_labels_automatic\(.*?return label_map\n',
    content, re.DOTALL
)

if old_func:
    old = old_func.group(0)
    new = '''def assign_labels_automatic(comp_stats, seed_words=None, df=None,
                              labels=None, feature_cols=None):
    """
    Sort clusters by avg_zipf_score DESCENDING.
      Highest avg zipf = most common words = LAYMAN
      Middle  avg zipf = intermediate words = STUDENT
      Lowest  avg zipf = rarest/hardest     = PROFESSIONAL

    Scientific basis: Zipf (1935) - word frequency is the most
    reliable single predictor of word difficulty. No manual weights.
    """
    print("\\nSTEP 7: Assigning labels (zipf-sorted, scientifically justified)...")

    label_order = ["LAYMAN", "STUDENT", "PROFESSIONAL"]

    scored = sorted(
        comp_stats.items(),
        key=lambda x: x[1].get("avg_zipf_score", 0),
        reverse=True
    )

    label_map = {}
    print("  Ranking easiest to hardest by avg zipf:")
    for rank, (cid, stats) in enumerate(scored):
        lbl = label_order[rank]
        label_map[cid] = lbl
        print(f"    Component {cid}: "
              f"zipf={stats.get('avg_zipf_score',0):.3f}  "
              f"length={stats.get('avg_word_length',0):.1f}  "
              f"-> {lbl}")

    return label_map
'''
    content = content.replace(old, new)
    open('model_training.py', 'w', encoding='utf-8').write(content)
    print('Patched successfully — now using pure zipf sorting')
else:
    print('Function not found. Printing all def lines:')
    for i, line in enumerate(content.split('\n')):
        if line.strip().startswith('def '):
            print(f'  Line {i}: {line}')