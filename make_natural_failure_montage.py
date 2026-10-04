from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


root = Path("/root/gpufree-data/libero-traces/e2e_mvp_20260918/libero90_task0_ep0_visual_replay.npz")
source = next(root.glob("*.npz"))
data = np.load(source)
frames = [0, 30, 50, 58, 59, 70, 100, 150, 250, 399]
width = height = 224
canvas = Image.new("RGB", (len(frames) * width, 2 * height + 48), "white")
draw = ImageDraw.Draw(canvas)
for column, frame in enumerate(frames):
    x = column * width
    canvas.paste(Image.fromarray(data["agent_images"][frame]), (x, 24))
    canvas.paste(Image.fromarray(data["wrist_images"][frame]), (x, 24 + height))
    action_index = int(data["action_indices"][frame])
    draw.text((x + 4, 4), f"action {action_index}", fill="black")
draw.text((4, 24), "fixed", fill="yellow")
draw.text((4, 24 + height), "wrist", fill="yellow")
output = Path("/root/gpufree-data/supervisor-tools-v1/outputs/libero90_task0_ep0_natural_failure_montage.png")
canvas.save(output)
print(output)
