import time
import depthai as dai
import numpy as np

from gym_av_aloha.vr.headset import Headset

FPS = 25
WIDTH, HEIGHT = 640, 480

# --- Headset ---
headset = Headset()
headset.run_in_thread()

# --- DepthAI pipeline ---
pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
left_out = cam_left.requestOutput(
    size=(WIDTH, HEIGHT),
    type=dai.ImgFrame.Type.BGR888p,
    fps=FPS,
)

cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
right_out = cam_right.requestOutput(
    size=(WIDTH, HEIGHT),
    type=dai.ImgFrame.Type.BGR888p,
    fps=FPS,
)

q_left = left_out.createOutputQueue()
q_right = right_out.createOutputQueue()

pipeline.start()

print("Streaming OAK cameras to headset...")

# --- Main loop ---
while True:
    start = time.time()

    left_msg = q_left.get()
    right_msg = q_right.get()

    if left_msg is None or right_msg is None:
        continue

    left_frame = left_msg.getCvFrame()
    right_frame = right_msg.getCvFrame()

    if left_frame is None or right_frame is None:
        continue

    # ensure correct format
    if left_frame.dtype != np.uint8:
        left_frame = left_frame.astype(np.uint8)
    if right_frame.dtype != np.uint8:
        right_frame = right_frame.astype(np.uint8)

    # --- SEND TO HEADSET ---
    headset.send_left_image(left_frame, 0)
    headset.send_right_image(right_frame, 0)

    # --- FPS control ---
    dt = time.time() - start
    time.sleep(max(0, 1.0 / FPS - dt))