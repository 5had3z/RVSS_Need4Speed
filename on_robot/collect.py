#!/usr/bin/env python3
import time
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pygame

import penguin_pi as ppi


write_path = Path(__file__).parent / "data" / datetime.now().strftime("%m-%d-%H-%M-%S")
if not write_path.exists():
    write_path.mkdir(parents=True)

print(f"Writing data to {write_path}")

# ~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
pygame.init()
# os.putenv("SDL_VIDEODRIVER", "dummy")
# size of pop-up window
pygame.display.set_mode((300, 300))
# holding a key sends continuous KEYDOWN events. Input argument is
# milli-seconds delay between events and controls the sensitivity.
pygame.key.set_repeat(100)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# stop the robot
ppi.set_velocity(0, 0)
print("initialise camera")
camera = ppi.VideoStreamWidget("http://localhost:8080/camera/get")

# countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    im_number = 0
    stopped = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    angle = 0
                    stopped = False
                    print("straight")
                if event.key == pygame.K_DOWN:
                    angle = 0
                    stopped = True
                if event.key == pygame.K_RIGHT:
                    print("right")
                    angle += 0.1
                if event.key == pygame.K_LEFT:
                    print("left")
                    angle -= 0.1
                if event.key == pygame.K_SPACE:
                    print("stop")
                    ppi.set_velocity(0, 0)
                    raise KeyboardInterrupt

        # get an image from the the robot
        image = camera.frame

        if stopped:
            ppi.set_velocity(0, 0)
        else:
            angle = np.clip(angle, -0.5, 0.5)
            Kd = 30  # base wheel speeds, increase to go faster, decrease to go slower
            Ka = 30  # how fast to turn when given an angle
            left = int(Kd + Ka * angle)
            right = int(Kd - Ka * angle)
            ppi.set_velocity(left, right)

        cv2.imwrite(str(write_path / f"{im_number:06}-{angle:.2f}.jpg"), image)
        im_number += 1


except KeyboardInterrupt:
    ppi.set_velocity(0, 0)
