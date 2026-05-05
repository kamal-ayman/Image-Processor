# ProImageProcessor

A professional image processing application built with Python, OpenCV, and CustomTkinter.

## Architecture

This project follows **Clean Architecture** principles and **SOLID** design patterns:

### 1. Domain Layer (`src/domain`)
- Contains the `ImageOperation` abstract base class (Strategy Pattern).
- Defines the core contract for any image processing algorithm.

### 2. Application Layer (`src/app`)
- Contains the `ImageService` which orchestrates operations.
- Manages image state (original vs processed) and provides a clean API for the UI.

### 3. Infrastructure Layer (`src/infrastructure`)
- Contains concrete implementations of image operations:
  - **Spatial Filters**: Mean, Median, Max, Min, Midpoint.
  - **Frequency Filters**: Ideal, Butterworth, Gaussian (LPF & HPF).
  - **Noise**: Gaussian, Salt & Pepper (Impulse), Rayleigh.
  - **Morphology**: Dilation, Erosion, Opening, Closing, Boundary Extraction.
  - **Enhancement**: Histogram Equalization, Gamma, Log, Contrast Stretching.
  - **Detection**: Point/Line detection and sharpening.

### 4. UI Layer (`src/ui`)
- Built with `customtkinter`.
- Decoupled from the implementation details through the `ImageService`.

### 5. Utils (`src/utils`)
- Shared helper functions for image conversion and normalization.

## SOLID Principles Applied
- **S (Single Responsibility)**: Each filter class does exactly one thing.
- **O (Open/Closed)**: New filters can be added by creating a new class inheriting from `ImageOperation` without changing existing code.
- **L (Liskov Substitution)**: Any `ImageOperation` subclass can be used interchangeably by the service.
- **I (Interface Segregation)**: The UI only depends on the `ImageService` interface.
- **D (Dependency Inversion)**: High-level modules (Service) depend on abstractions (ImageOperation), not concrete implementations.

## How to Run
```bash
python main.py
```
