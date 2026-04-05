import math

svg = []
svg.append('<?xml version="1.0" encoding="utf-8"?>')
svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800" width="100%" height="100%">')
svg.append('<defs>')
svg.append('<marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/></marker>')
svg.append('<marker id="arrow-dash" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#B8860B"/></marker>')
svg.append('<marker id="arrow-purp" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9932CC"/></marker>')
svg.append('</defs>')

# Data
enc_words = ["I", "am", "a", "student"]
dec_words = ["&lt;s&gt;", "Je", "suis", "étudiant"]
out_words = ["Je", "suis", "étudiant", "&lt;/s&gt;"]
attn_w = ["0.5", "0.3", "0.1", "0.1"]

enc_col = "#1E90FF"
dec_col = "#B22222"
ctx_col = "#DEB887"
att_col = "#8B6508"

box_w, box_h = 40, 80
cols = 8 
start_x = 80
dx = 110

y_in_text = 750
y_rnn1 = 600
y_rnn2 = 450
y_att_w = 320
y_ctx = 160
y_att_v = 80
y_out_text = 10

def draw_rect(x, y, fill, width=box_w, height=box_h):
    svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="#333" stroke-width="2"/>')

def draw_circle(cx, cy, r, fill, stroke="#B8860B", dash=False):
    style = f'stroke="{stroke}" stroke-width="2"'
    if dash:
        style += ' stroke-dasharray="4,4"'
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" {style}/>')

def draw_arrow(x1, y1, x2, y2, color="#333", dashed=False):
    style = f'stroke="{color}" stroke-width="2"'
    marker = "url(#arrow)"
    if dashed:
        style += ' stroke-dasharray="5,5"'
        if color == "#B8860B":
            marker = "url(#arrow-dash)"
        elif color == "#9932CC":
            marker = "url(#arrow-purp)"
    elif color == "#9932CC":
        marker = "url(#arrow-purp)"
        
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {style} marker-end="{marker}"/>')

def draw_text(x, y, text, size=24, bold=False, color="#000", anchor="middle", italic=False):
    fw = "bold" if bold else "normal"
    fs = "italic" if italic else "normal"
    svg.append(f'<text x="{x}" y="{y}" font-family="serif" font-size="{size}px" font-weight="{fw}" font-style="{fs}" fill="{color}" text-anchor="{anchor}">{text}</text>')

# Draw bottom inputs
for i in range(4):
    cx = start_x + i * dx
    draw_text(cx + box_w/2, y_in_text, enc_words[i])
    draw_rect(cx, y_rnn1, enc_col)
    draw_rect(cx, y_rnn2, enc_col)
    draw_arrow(cx + box_w/2, y_in_text - 30, cx + box_w/2, y_rnn1 + box_h + 5)
    draw_arrow(cx + box_w/2, y_rnn1 - 5, cx + box_w/2, y_rnn2 + box_h + 5, color="#2CA02C")
    if i < 3:
        draw_arrow(cx + box_w + 5, y_rnn1 + box_h/2, cx + dx - 5, y_rnn1 + box_h/2)
        draw_arrow(cx + box_w + 5, y_rnn2 + box_h/2, cx + dx - 5, y_rnn2 + box_h/2)

    # Attention weights
    draw_arrow(cx + box_w/2, y_rnn2 - 5, cx + box_w/2, y_att_w + 25 + 5, dashed=True, color="#B8860B")
    draw_circle(cx + box_w/2, y_att_w, 25, fill="#fff", stroke="#B8860B", dash=True)
    draw_text(cx + box_w/2, y_att_w + 8, attn_w[i], size=20)
    
    # Arrow to context vector
    ctx_x = start_x + 2*dx + box_w/2
    draw_arrow(cx + box_w/2, y_att_w - 25 - 5, ctx_x, y_ctx + box_h, dashed=True, color="#B8860B")

# Context vector
ctx_cx = start_x + 2*dx
draw_rect(ctx_cx, y_ctx, ctx_col)
draw_text(ctx_cx - 20, y_ctx + 30, "context", size=22, italic=True, anchor="end")
draw_text(ctx_cx - 20, y_ctx + 55, "vector", size=22, italic=True, anchor="end")

draw_text(start_x - 30, y_att_w - 10, "attention", size=22, italic=True, anchor="end")
draw_text(start_x - 30, y_att_w + 15, "weights", size=22, italic=True, anchor="end")

# Draw decoder
for i in range(4, 8):
    cx = start_x + i * dx
    draw_text(cx + box_w/2, y_in_text, dec_words[i-4])
    
    # Lightred arrow from input
    draw_arrow(cx + box_w/2, y_in_text - 30, cx + box_w/2, y_rnn1 + box_h + 5, color="#F08080")
    
    draw_rect(cx, y_rnn1, dec_col)
    draw_rect(cx, y_rnn2, dec_col)
    draw_arrow(cx + box_w/2, y_rnn1 - 5, cx + box_w/2, y_rnn2 + box_h + 5, color="#F08080")
    
    if i < 7:
        draw_arrow(cx + box_w + 5, y_rnn1 + box_h/2, cx + dx - 5, y_rnn1 + box_h/2, color="#B22222")
        draw_arrow(cx + box_w + 5, y_rnn2 + box_h/2, cx + dx - 5, y_rnn2 + box_h/2, color="#B22222")
        
    # Brown attention vectors
    draw_rect(cx, y_att_v, att_col)
    draw_arrow(cx + box_w/2, y_rnn2 - 5, cx + box_w/2, y_att_v + box_h + 5, color="#B8860B")
    
    # Top output arrows
    draw_arrow(cx + box_w/2, y_att_v - 5, cx + box_w/2, y_out_text + 15, color="#9932CC")
    draw_text(cx + box_w/2, y_out_text, out_words[i-4])

# Connection from first decoder state to attention circles
first_dec_x = start_x + 4 * dx
for i in range(4):
    cw_x = start_x + i * dx + box_w/2
    draw_arrow(first_dec_x + box_w/2, y_rnn2 - 5, cw_x, y_att_w + 25 + 5, dashed=True, color="#B8860B")

# Connection from context vector to top brown box (first output)
draw_arrow(ctx_cx + box_w, y_ctx + box_h/2, first_dec_x - 5, y_att_v + box_h/2, color="#B8860B")

# Subsequent outputs feed into next inputs 
draw_arrow(start_x + 5*dx - 10, y_out_text + 10, start_x + 5*dx - 20, y_in_text - 35, dashed=True, color="#B8860B")
draw_arrow(start_x + 6*dx - 10, y_out_text + 10, start_x + 6*dx - 20, y_in_text - 35, dashed=True, color="#B8860B")
draw_arrow(start_x + 7*dx - 10, y_out_text + 10, start_x + 7*dx - 20, y_in_text - 35, dashed=True, color="#B8860B")

draw_text(start_x + 4*dx - 20, y_att_v + 30, "attention", size=22, italic=True, anchor="end")
draw_text(start_x + 4*dx - 20, y_att_v + 55, "vector", size=22, italic=True, anchor="end")

# Connection between encoder and decoder sections
draw_arrow(start_x + 3*dx + box_w + 5, y_rnn1 + box_h/2, start_x + 4*dx - 5, y_rnn1 + box_h/2, color="#F08080")
draw_arrow(start_x + 3*dx + box_w + 5, y_rnn2 + box_h/2, start_x + 4*dx - 5, y_rnn2 + box_h/2, color="#F08080")

svg.append('</svg>')

with open('figures/attention_custom.svg', 'w') as f:
    f.write('\n'.join(svg))

print("Created attention_custom.svg")
