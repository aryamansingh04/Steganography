# Steganography Method Prediction Model

This project implements an AI model that predicts whether **LSB (Least Significant Bit)** or **DCT (Discrete Cosine Transform)** steganography would yield better imperceptibility for a given image.

## Overview

Steganography is the practice of hiding information within other non-secret data. This project compares two popular steganography techniques:
- **LSB (Least Significant Bit)**: Embeds secret messages by modifying the least significant bits of image pixels
- **DCT (Discrete Cosine Transform)**: Embeds secret messages in the frequency domain using DCT coefficients

The goal is to train a machine learning classifier that can automatically determine which method would produce a more imperceptible stego-image (one that looks more like the original) for any given input image.

## Features

- **Dataset Loading**: Uses CIFAR-10 dataset for training and testing
- **Dual Steganography Implementation**: Applies both LSB and DCT steganography methods
- **Imperceptibility Metrics**: Evaluates stego-images using:
  - **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality
  - **SSIM (Structural Similarity Index)**: Measures structural similarity
- **Feature Extraction**: Extracts relevant features from original images
- **ML Classification**: Trains a classifier to predict the better steganography method

## Methodology

1. **Load Image Dataset**: Loads a diverse dataset of images (CIFAR-10)
2. **Apply Steganography**: Applies both LSB and DCT steganography with consistent parameters to each image
3. **Evaluate Imperceptibility**: Calculates PSNR and SSIM metrics for each stego-image
4. **Generate Labels**: Creates labels indicating which method performed "better" for each image
5. **Extract Features**: Extracts features from the original images
6. **Train Classifier**: Trains a machine learning classifier on features and labels
7. **Predict**: Uses the trained model to predict the preferred steganography method for new images

## Requirements

The notebook requires the following Python libraries:

- `tensorflow` - For dataset loading and image processing
- `tensorflow_datasets` - For accessing CIFAR-10 dataset
- `numpy` - For numerical operations
- `matplotlib` - For visualization
- `scikit-learn` - For machine learning classification
- `scipy` - For DCT operations
- `cv2` (OpenCV) - For image processing (if used)

## Usage

1. Open the Jupyter notebook `steganography_model.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Load the CIFAR-10 dataset
   - Apply both steganography methods
   - Evaluate and compare results
   - Train the prediction model
   - Test on unseen images

## Project Structure

```
.
├── README.md
└── steganography_model.ipynb
```

## Notes

- The model's performance depends on having sufficient class diversity in the training data
- Both LSB and DCT methods are implemented with configurable parameters
- The evaluation uses standard image quality metrics (PSNR and SSIM)

## License

This project is open source and available for educational and research purposes.

## Author

Aryaman Kumar Singh
