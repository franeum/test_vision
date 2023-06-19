#!/usr/bin/env python3

import sys
from ultralytics import YOLO

try:
    image = sys.argv[1]
except:
    print("please provide an image file")
    sys.exit(1)

model = YOLO('yolov8n.pt')
results = model(image, show=True, save=True)

input()
