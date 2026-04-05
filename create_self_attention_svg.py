import math

svg = []
svg.append('<?xml version="1.0" encoding="utf-8"?>')
svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="100%" height="100%">')
svg.append('<defs>')
svg.append('<marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/></marker>')
svg.append('</defs>')

# Colors
q_col = "#dc322f" # Red
k_col = "#268bd2" # Blue
v_col = "#859900" # Green
x_col = "#93a1a1" # Grey
attn_col = "#b58900" # Yellow/Gold

# Input X
x_x, x_y = 100, 450
svg.append(f'<rect x="{x_x}" y="{x_y}" width="60" height="100" fill="{x_col}" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{x_x+30}" y="{x_y+130}" font-family="serif" font-size="24px" text-anchor="middle" font-weight="bold">Input X</text>')

# Weight Matrices W_Q, W_K, W_V
w_x = 300
svg.append(f'<rect x="{w_x}" y="350" width="40" height="40" fill="#eee" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{w_x+20}" y="340" font-family="serif" font-size="18px" text-anchor="middle">W^Q</text>')
svg.append(f'<rect x="{w_x}" y="450" width="40" height="40" fill="#eee" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{w_x+20}" y="440" font-family="serif" font-size="18px" text-anchor="middle">W^K</text>')
svg.append(f'<rect x="{w_x}" y="550" width="40" height="40" fill="#eee" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{w_x+20}" y="540" font-family="serif" font-size="18px" text-anchor="middle">W^V</text>')

# Arrows from X to Ws
svg.append(f'<line x1="{x_x+60}" y1="{x_y+50}" x2="{w_x-5}" y2="370" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')
svg.append(f'<line x1="{x_x+60}" y1="{x_y+50}" x2="{w_x-5}" y2="470" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')
svg.append(f'<line x1="{x_x+60}" y1="{x_y+50}" x2="{w_x-5}" y2="570" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')

# Q, K, V vectors
vec_x = 450
svg.append(f'<rect x="{vec_x}" y="340" width="60" height="60" fill="{q_col}" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{vec_x+30}" y="330" font-family="serif" font-size="22px" text-anchor="middle" font-weight="bold">Q</text>')
svg.append(f'<rect x="{vec_x}" y="440" width="60" height="60" fill="{k_col}" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{vec_x+30}" y="430" font-family="serif" font-size="22px" text-anchor="middle" font-weight="bold">K</text>')
svg.append(f'<rect x="{vec_x}" y="540" width="60" height="60" fill="{v_col}" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{vec_x+30}" y="530" font-family="serif" font-size="22px" text-anchor="middle" font-weight="bold">V</text>')

svg.append(f'<line x1="{w_x+40}" y1="370" x2="{vec_x-5}" y2="370" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')
svg.append(f'<line x1="{w_x+40}" y1="470" x2="{vec_x-5}" y2="470" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')
svg.append(f'<line x1="{w_x+40}" y1="570" x2="{vec_x-5}" y2="570" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')

# Dot product intersection
dot_x, dot_y = 650, 400
svg.append(f'<circle cx="{dot_x}" cy="{dot_y}" r="30" fill="white" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{dot_x}" y="{dot_y+8}" font-family="serif" font-size="24px" text-anchor="middle">×</text>')
svg.append(f'<text x="{dot_x}" y="{dot_y-40}" font-family="serif" font-size="18px" text-anchor="middle">QK^T</text>')

# Arrows to dot product
svg.append(f'<path d="M {vec_x+60} 370 Q 550 370, {dot_x-30} {dot_y}" fill="none" stroke="{q_col}" stroke-width="2" marker-end="url(#arrow)"/>')
svg.append(f'<path d="M {vec_x+60} 470 Q 550 470, {dot_x-30} {dot_y}" fill="none" stroke="{k_col}" stroke-width="2" marker-end="url(#arrow)"/>')

# Scaling and Softmax
soft_x, soft_y = 750, 400
svg.append(f'<rect x="{soft_x}" y="{soft_y-30}" width="80" height="60" fill="{attn_col}" rx="10" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{soft_x+40}" y="{soft_y+7}" font-family="serif" font-size="18px" text-anchor="middle" fill="white">Softmax</text>')
svg.append(f'<line x1="{dot_x+30}" y1="{dot_y}" x2="{soft_x-5}" y2="{soft_y}" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>')

# Final Weighted Sum
final_x, final_y = 900, 470
svg.append(f'<circle cx="{final_x}" cy="{final_y}" r="30" fill="white" stroke="#333" stroke-width="2"/>')
svg.append(f'<text x="{final_x}" y="{final_y+8}" font-family="serif" font-size="24px" text-anchor="middle">+</text>')
svg.append(f'<text x="{final_x}" y="{final_y+60}" font-family="serif" font-size="20px" text-anchor="middle" font-weight="bold">Output</text>')

# Arrow from softmax and V to final
svg.append(f'<path d="M {soft_x+80} {soft_y} Q 870 {soft_y}, {final_x} {final_y-30}" fill="none" stroke="{attn_col}" stroke-width="2" marker-end="url(#arrow)"/>')
svg.append(f'<path d="M {vec_x+60} 570 Q 750 570, {final_x-20} {final_y+20}" fill="none" stroke="{v_col}" stroke-width="2" marker-end="url(#arrow)"/>')

svg.append('</svg>')

with open('figures/self_attention_custom.svg', 'w') as f:
    f.write('\n'.join(svg))

print("Created self_attention_custom.svg")
