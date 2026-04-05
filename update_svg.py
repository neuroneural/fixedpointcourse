with open('figures/seq2seq_custom.svg', 'r', encoding='utf-8') as f:
    content = f.read()

if not content.startswith('<?xml'):
    content = '<?xml version="1.0" encoding="utf-8"?>\n' + content

with open('figures/seq2seq_custom.svg', 'w', encoding='utf-8') as f:
    f.write(content)
