# Piezoelectric Impedance & Deep Learning for Corrosion Quantification

This repository contains code fragments and experimental implementations related to my work on **corrosion monitoring and quantification using piezoelectric impedance sensing and deep learning**.

The repository is primarily maintained for **personal learning, experimentation, and research exploration**. It contains scripts from different stages of development, including exploratory trials, experimental data processing, model development, ablation studies, and model evaluation. Some parts are not fully organized or polished and are preserved mainly for research traceability and future reference.

## Repository Structure

### [`General_Try`](https://github.com/Adair2299/A1_PiezoelectricImpedanceCNN/tree/main/A1_PiezoelectricImpedanceCNN_asof20250814/A1PJ_Piezo_25Stu/General_Try)

Contains exploratory scripts for **symbolic regression analysis** and other experimental approaches.

These scripts mainly document early attempts and individual experiments rather than a finalized pipeline.

### [`Executives`](https://github.com/Adair2299/A1_PiezoelectricImpedanceCNN/tree/main/A1_PiezoelectricImpedanceCNN_asof20250814/A1PJ_Piezo_25Stu/Executives)

Contains scripts for **experimental data processing and analysis**, including preprocessing and related experimental workflows.

### [`Latest2`](https://github.com/Adair2299/A1_PiezoelectricImpedanceCNN/tree/main/A1_PiezoelectricImpedanceCNN_asof20250814/A1PJ_Piezo_25Stu/Latest2)

Contains the **latest stage of the project**, including model development, evaluation, and experimental analysis.

Some scripts are related to **ablation studies and model evaluation**.

The `115d1a` implementation contains the main training and final-output pipeline, including the integration of **residual architectures and attention mechanisms**.

Other files in this directory document individual experiments and attempts made during model development.

## Attention Mechanism under Noisy Impedance Signals

This implementation investigates the effectiveness of an **attention mechanism for corrosion quantification from noisy piezoelectric impedance signals**. When noise is introduced, different frequency regions may contain substantially different levels of useful information. The attention mechanism therefore enables the model to **automatically assign different weights to different frequency segments**, emphasizing informative regions while reducing the influence of less informative or noise-dominated components.

The introduction of attention resulted in a **significant improvement in prediction accuracy**, as demonstrated by the comparison below.

### Prediction Results with Attention

<img width="250" height="250" alt="training_process" src="https://github.com/user-attachments/assets/91eb2bd0-40cd-439e-9d2f-10f0bf19599f" />
<img width="500" height="250" alt="prediction_comparison" src="https://github.com/user-attachments/assets/96c7077f-52dd-4391-a551-dffe65ec8eee" />


### Prediction Results with Attention Removed

<img width="250" height="250" alt="training_process 1" src="https://github.com/user-attachments/assets/489ff788-9f48-4138-98ce-7a37d80f399f" />
<img width="500" height="250" alt="prediction_comparison 1" src="https://github.com/user-attachments/assets/ebd45e7c-75d8-4424-a354-f0e3350b7990" />



| Model                 | Absolute Error Range | Average Absolute Error | Relative Error Range | Average Relative Error |
| --------------------- | -------------------: | ---------------------: | -------------------: | ---------------------: |
| With Attention    |     [0.0001, 0.0147] |             0.0026 |      [0.08%, 12.94%] |              2.98% |
| Attention Removed |     [0.0014, 0.0954] |             0.0229 |     [1.84%, 163.85%] |             34.10% |

### Attention Weight Analysis

The visualization of attention weights demonstrates that the model successfully **learns to automatically distribute attention across different frequency regions** rather than treating all signal segments equally.

<img width="375" height="375" alt="Original Signal 9 " src="https://github.com/user-attachments/assets/b099daf4-2fc7-4298-b62f-751476e53cc6" />
<img width="375" height="375" alt="Attention Patterns for First 48 Test Samples" src="https://github.com/user-attachments/assets/436cfd0b-ea41-446c-86bc-71479c53d594" />



To generalize, these results suggest that different segments within a signal may contain varying levels of information. From my perspective, this indicates that attention mechanisms may provide a useful way to identify and emphasize potentially more informative signal regions, which could contribute to improved prediction accuracy, particularly under noisy conditions.


## Notes on the Code

This repository is not intended to represent a fully packaged or production-ready software project. It is primarily a record of personal research, learning, and experimentation.

Some scripts were developed incrementally and may contain redundant, experimental, or partially organized code.

For some scripts, **AI-assisted coding was used during development**, ranging from partial assistance to substantial code generation. The underlying research questions, experimental design, model selection, and implementation direction were determined based on my own project work and understanding.

## Research Context

The project focuses on using piezoelectric impedance measurements and deep learning to estimate and quantify corrosion-related changes in steel structures.

The work explored topics including:

* Piezoelectric impedance sensing
* Signal preprocessing and denoising
* Deep learning
* Residual neural networks
* Attention mechanisms
* Data augmentation
* Model evaluation and ablation analysis
* Experimental impedance data processing
* Symbolic regression

---

**Note:** This repository is provided primarily for personal learning, experimentation, and research reference. It should not be interpreted as a polished or production-ready codebase.
