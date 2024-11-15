# Indirect Hard Modeling (IHM) for Spectral Component Identification

This repository implements the **Indirect Hard Modeling (IHM)** method as described in the paper *"Identification of Unknown Pure Component Spectra by Indirect Hard Modeling"* by Kriesten et al., 2008. The code extracts pure spectra from a given spectral mixture matrix using a combination of peak fitting, correlation analysis, and optimization techniques.

---

## Overview of the Method

The IHM approach models a spectral mixture matrix by:
1. **Finding a representative spectrum** (\(x_{\text{input}}\)) from a subset of mixture spectra (Step 1).
2. **Fitting Voigt profiles** to the peaks in \(x_{\text{input}}\), capturing nonlinear effects like peak shifts and variations (Step 2).
3. **Constructing a weight matrix** representing contributions of the fitted peaks across all mixture spectra (Step 3).
4. **Correlation-based analysis** to identify distinctive and shared peaks, allowing classification of peaks into pure components (Step 4).
5. **Delta matrix optimization** to allocate shared peaks to their corresponding components, ensuring a coherent representation of the spectral data (Step 5).
6. **Pure spectrum reconstruction** using the optimized delta matrix to obtain individual pure spectra (Step 6).

---

## Code Structure and Workflow

### Step 1: Input Data
- **Spectral Matrix**: Provide your mixed spectral data in `spectal_matrix`. The matrix should have dimensions corresponding to the number of spectra and wavelength points.
- **Wavelength Vector**: Ensure `wl` matches the wavelength points of your spectral matrix.

### Step 2: Finding \(x_{\text{input}}\)
- The code identifies a representative spectrum by averaging a subset of spectra with the lowest correlation errors.

### Step 3: Voigt Peak Fitting
- Fits Voigt profiles to the representative spectrum (\(x_{\text{input}}\)) to extract peak parameters.
- Outputs:
  - **Voigt Parameters**: Saved to `Voig_parameters_X_input`.
  - **Reconstructed Spectrum**: Visualized alongside the mean spectrum.

### Step 4: Weight Matrix Calculation
- Constructs the weight matrix \(W\) by fitting the Voigt parameters to all spectra in the dataset.
- Outputs:
  - **Weight Matrix**: Saved to `WeightsMatrix`.

### Step 5: Correlation Analysis
- Constructs a peak-to-peak correlation matrix to classify peaks as:
  - **Distinctive Peaks**: Unique to individual components.
  - **Shared Peaks**: Common across multiple components.
- Determines the number of components (\(K\)) and identifies correlated groups of peaks.

### Step 6: Delta Matrix Optimization
- Optimizes the delta matrix to allocate shared peaks to pure components using constrained optimization.
- Adjusts for ill-conditioning and ensures robust peak allocation.

### Step 7: Pure Spectrum Reconstruction
- Reconstructs the pure spectra using the optimized delta matrix.
- Outputs:
  - **Reconstructed Spectra**: Visualized alongside the original mean spectrum.

---

## Usage Instructions

1. **Prepare Input Data**:
   - Replace `spectal_matrix` with your spectral data.
   - Provide the corresponding wavelength vector `wl`.

2. **Run the Code**:
   - Set `run_fitting = True` to perform new peak fitting. If previously run, set it to `False` to reuse saved results.
   - Set `Only_SharedPeaks = True` to optimize shared peaks only, if required.

3. **Output Files**:
   - `Voig_parameters_X_input`: Contains the fitted Voigt parameters.
   - `WeightsMatrix`: The weight matrix for all spectra.
   - Visualizations: Heatmaps and spectral plots for correlation analysis and reconstructed spectra.

---

## Key Equations and Concepts

1. **Voigt Peak Fitting**:
   - Models each peak as a Voigt profile, parameterized by width, height, position, and shape.

2. **Correlation Matrix**:
   - Identifies shared and distinctive peaks using correlation thresholds (default: \(0.98\)).

3. **Delta Matrix Optimization**:
   - Solves:
     \[
     \min_x \| W[:, i] - \sum(W[:, \text{distinctive}] \cdot x)\| + \lambda \cdot (1 - \sum(x))^2
     \]
     where \(x\) represents the allocation of shared peaks to components.

---

## Notes

- **Data Format**: Ensure your spectral matrix and wavelength vector are consistent and preprocessed.
- **Parameters**: Adjust correlation thresholds and optimization parameters based on data characteristics.
- **Computational Resources**: Use GPU acceleration (`gpu=1`) for faster computations.

---

## References

- **Kriesten, E., et al. (2008)**: "Identification of Unknown Pure Component Spectra by Indirect Hard Modeling".  
  [DOI:10.1016/j.chemolab.2008.05.002](https://doi.org/10.1016/j.chemolab.2008.05.002)

---
