# Seam Carving Algorithm Implementation

![Seam Carving](https://github.com/user-attachments/assets/a1992901-5eb4-4360-b447-c645d0e3bc8b)

## Overview

This project implements the Seam Carving algorithm for content-aware image resizing. It removes the least important paths in an image to reduce its size while preserving key features.

## Features

- Convert images to grayscale
- Apply Gaussian Blur
- Detect edges using Sobel Filter
- Construct Energy Map
- Find and remove low-energy seams
- Parallel implementation using CUDA

## Algorithm Versions

1. Basic parallel implementation
2. Optimized using shared memory (SMEM)
3. Parallelized dynamic programming step

## Installation

```bash
git clone https://github.com/yourusername/seam-carving.git
cd seam-carving
make
```

## Usage

```bash
./seam_carving input_image.jpg output_image.jpg num_seams
```

## Performance Analysis

![Performance Graph](https://github.com/user-attachments/assets/3ff1373d-9b81-4e57-9533-89219a35fc35)

Our parallel implementations show significant speedup compared to the sequential version, especially for large images and high seam counts.

## Future Development

- Implement two-end BFS for further optimization
- Explore Forward Energy method
- Optimize for maximum seam removal in practical applications

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- [Shai Avidan and Ariel Shamir](https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf) for the original Seam Carving paper
- [Brown University CS129](http://cs.brown.edu/courses/cs129/results/proj3/taox/) for Forward Energy concepts



