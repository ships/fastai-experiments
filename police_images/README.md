# Detecting police units in photos with deep learning.

This is a toy project to frame my learning fastai's approach to deep learning.
We'll see how sophisticated it can get as I get up to speed.

## Vision

This is the rough direction I imagine this project moving in:

1. Naive learner that can distinguish a photo of a car or empty street from a photo of a police car/van in the street.
2. Same or chainable learner that can pick out the police unit from the street.
3. Same or chainable learner that can pick out the license plate of the police unit.
4. Setup a raspberry pi or similar with video stream from the window that records timestamps and plates of police units.

## Status

Having followed the first of fastai's image classification tutorials, it was easy to get a learner
that can distinguish SFPD units from other sedans. However, it does not perform at all well
on photos that are not of cars at all, with 10% false positive rate. Furthermore it is not
tested on photos of partial vehicles or on multiple vehicles together.