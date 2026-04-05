import math

svg = []
svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="100%" height="100%">')
svg.append('<defs>')
svg.append('<marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/></marker>')
svg.append('<marker id="arrow-dash" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/></marker>')
svg.append('</defs>')

# Data
enc_words = ["I", "am", "a", "student", "<s>"]
dec_words = ["moi", "suis", "étudiant"]
out_words = ["moi", "suis", "étudiant", "&lt;/s&gt;"]

enc_col = "#1E90FF"
enc_light = "#87CEFA"
dec_col = "#B22222"
dec_light = "#F08080"

# Dimensions
box_w, box_h = 40, 80
cols = 9  # 0 to 4 (encoder inputs), 5 to 7 (decoder inputs), 8 (final output)
start_x = 50
dx = 100

y_in_text = 500
y_emb = 380
y_rnn1 = 260
y_rnn2 = 140
y_out_box = 20

def draw_rect(x, y, fill):
    svg.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" fill="{fill}" stroke="#333" stroke-width="2"/>')

def draw_arrow(x1, y1, x2, y2, dashed=False):
    style = 'stroke="#333" stroke-width="2"'
    if dashed:
        style = 'stroke="#666" stroke-width="2" stroke-dasharray="5,5"'
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {style} marker-end="url(#arrow' + ('-dash' if dashed else '') + ')"/>')

def draw_text(x, y, text, size=24, bold=False, color="#000", anchor="middle"):
    fw = "bold" if bold else "normal"
    svg.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}px" font-weight="{fw}" fill="{color}" text-anchor="{anchor}">{text}</text>')

# Encoder
for i in range(5):
    cx = start_x + i * dx
    # text
    draw_text(cx + box_w/2, y_in_text, enc_words[i])
    # emb box
    if i < 4:
        draw_rect(cx, y_emb, enc_light)
        draw_arrow(cx + box_w/2, y_in_text - 25, cx + box_w/2, y_emb + box_h + 5)
    else:
        # The <s> token goes into the decoder stack but is considered the transition
        draw_rect(cx, y_emb, dec_light)
        draw_arrow(cx + box_w/2, y_in_text - 25, cx + box_w/2, y_emb + box_h + 5)
    
    # rnn1 box
    color1 = enc_col if i < 4 else dec_col
    draw_rect(cx, y_rnn1, color1)
    draw_arrow(cx + box_w/2, y_emb - 5, cx + box_w/2, y_rnn1 + box_h + 5)
    
    # rnn2 box
    draw_rect(cx, y_rnn2, color1)
    draw_arrow(cx + box_w/2, y_rnn1 - 5, cx + box_w/2, y_rnn2 + box_h + 5)
    
    # horizontal arrows
    if i < 4:
        draw_arrow(cx + box_w + 5, y_rnn1 + box_h/2, cx + dx - 5, y_rnn1 + box_h/2)
        draw_arrow(cx + box_w + 5, y_rnn2 + box_h/2, cx + dx - 5, y_rnn2 + box_h/2)

# Decoder
for i in range(5, 8):
    cx = start_x + i * dx
    # text
    draw_text(cx + box_w/2, y_in_text, dec_words[i-5])
    
    # emb
    draw_rect(cx, y_emb, dec_light)
    draw_arrow(cx + box_w/2, y_in_text - 25, cx + box_w/2, y_emb + box_h + 5, dashed=True)
    
    # rnn1 & rnn2
    draw_rect(cx, y_rnn1, dec_col)
    draw_arrow(cx + box_w/2, y_emb - 5, cx + box_w/2, y_rnn1 + box_h + 5)
    
    draw_rect(cx, y_rnn2, dec_col)
    draw_arrow(cx + box_w/2, y_rnn1 - 5, cx + box_w/2, y_rnn2 + box_h + 5)
    
    # horizontal
    draw_arrow(cx - dx + box_w + 5, y_rnn1 + box_h/2, cx - 5, y_rnn1 + box_h/2)
    draw_arrow(cx - dx + box_w + 5, y_rnn2 + box_h/2, cx - 5, y_rnn2 + box_h/2)

    # Output arrow to top
    draw_arrow(cx - dx + box_w/2, y_rnn2 - 5, cx - dx + box_w/2, y_out_box + 80 + 5)

# final output from last decoder state
cx = start_x + 8 * dx
draw_arrow(cx - dx + box_w/2, y_rnn2 - 5, cx - dx + box_w/2, y_out_box + 80 + 5)


# Probability boxes
probs = [
    ["0.1", "0.1", "0.5", "0.1"],
    ["0.1", "0.1", "0.2", "0.6"],
    ["0.3", "0.1", "0.1", "0.1"],
    ["0.4", "0.1", "0.1", "0.1"],
    ["0.1", "0.6", "0.1", "0.1"]
]
labels = ["étudiant", "-", "Je", "moi", "suis"]

# top labels
for j, v in enumerate(["moi", "suis", "étudiant", "&lt;/s&gt;"]):
    px = start_x + (4 + j) * dx + box_w/2
    draw_text(px, y_out_box - 5, v, size=20, bold=True)

# draw table of probs
for i, row in enumerate(probs):
    py = y_out_box + 20 + i*14
    # word label
    draw_text(start_x + 4 * dx - 10, py, labels[i], size=16, anchor="end")
    for j, val in enumerate(row):
        px = start_x + (4 + j) * dx
        
        # Color max green
        color = "#2CA02C" if (val in ["0.4", "0.6", "0.5"]) and (
            (j==0 and val=="0.4") or (j==1 and val=="0.6") or (j==2 and val=="0.5") or (j==3 and val=="0.6")
        ) else "#000"
        bold = True if color == "#2CA02C" else False
        
        # draw table cell
        svg.append(f'<rect x="{px}" y="{py - 12}" width="{box_w}" height="14" fill="white" stroke="#ccc" stroke-width="1"/>')
        draw_text(px + box_w/2, py, val, size=14, color=color, bold=bold)

# The dashed lines linking out argmax to next input
draw_arrow(start_x + 4*dx + box_w, y_out_box + 20 + 3*14 - 6, start_x + 5*dx + box_w/2, y_in_text - 35, dashed=True) # moi
draw_arrow(start_x + 5*dx + box_w, y_out_box + 20 + 4*14 - 6, start_x + 6*dx + box_w/2, y_in_text - 35, dashed=True) # suis
draw_arrow(start_x + 6*dx + box_w, y_out_box + 20 + 0*14 - 6, start_x + 7*dx + box_w/2, y_in_text - 35, dashed=True) # etudiant

svg.append('</svg>')

with open('figures/seq2seq_custom.svg', 'w') as f:
    f.write('\n'.join(svg))

print('Created SVG')
