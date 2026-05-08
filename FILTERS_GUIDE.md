# The Group 4 Guide to Digital Image Processing
*Final Restricted Technical Manual*

This document provides the technical and mathematical foundation for the **Group 4** image processing suite. This library is strictly limited to the 12 operations specified below.

---

## Table of Contents
1. [Spatial Domain Operations](#1-spatial)
2. [Frequency Domain Operations](#2-frequency)
3. [Morphological Transformations](#3-morphology)
4. [Noise Models & Simulation](#4-noise)
5. [Edge Detection & Sharpening](#5-detection)

---

<a name="1-spatial"></a>
## 1. Spatial Domain Operations

### 1.1 Weighted Average (Mean Filter)
The simplest linear spatial filter. It replaces a pixel with the average of its neighborhood.
- **Goal**: Blurring and noise reduction.

### 1.2 Bilateral Filter
A non-linear, edge-preserving, and noise-reducing smoothing filter.
- **Benefit**: Reduces noise while keeping edges extremely sharp.

---

<a name="2-frequency"></a>
## 2. Frequency Domain Operations

### 2.1 Ideal LPS (Low Pass System)
A filter that cuts off all high-frequency components beyond a specified distance $D_0$.
- **Artifacts**: Causes "ringing" effects.

### 2.2 Gaussian HPF (High Pass Filter)
Allows high frequencies to pass while attenuating low frequencies.
- **Formula**: $H(u,v) = 1 - e^{-D^2(u,v) / 2D_0^2}$

---

<a name="3-morphology"></a>
## 3. Morphological Transformations

### 3.1 Dilation
Expands the boundaries of foreground objects.

### 3.2 Opening
An erosion followed by a dilation. Removes small white noise from the background.

---

<a name="4-noise"></a>
## 4. Noise Models & Simulation

### 4.1 Rayleigh Noise
Common in radar and ultrasound imaging.

### 4.2 Impulse Noise (Salt & Pepper)
Random white and black pixels caused by transmission errors.

### 4.3 Exponential Noise
Common in laser imaging systems.

---

<a name="5-edge-detection"></a>
## 5. Edge Detection & Sharpening

### 5.1 Robert Cross Operator
Uses a $2 \times 2$ gradient mask to find diagonal edges.

### 5.2 Laplacian
A second-order derivative operator used for edge enhancement.

### 5.3 Unsharp Masking
Subtracts a blurred version of an image from the original to create a sharpening mask.
