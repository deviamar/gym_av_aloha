import numpy as np
import time
from gym_av_aloha.vr.headset import Headset

FPS = 25

headset = Headset()
headset.run_in_thread()

h, w = 480, 640

while True:
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # draw something obvious
    frame[:, :, 1] = 255  # green screen

    headset.send_left_image(frame, 0)
    headset.send_right_image(frame, 0)

    time.sleep(1.0 / FPS)