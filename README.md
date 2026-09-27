# ⚡ Snapdragon Edge AI Deployment

### From Trained Models to Hardware-Accelerated NPU Inference

**Production-oriented Edge AI deployment experiments on Qualcomm Snapdragon X Elite — covering model export, ONNX conversion, QNN compilation, NPU execution, and on-device performance profiling.**

This repository demonstrates the complete path from **framework-native trained models → portable ONNX graphs → Qualcomm AI Hub compilation → Snapdragon X Elite NPU execution → hardware-level profiling**.

Rather than validating a single lightweight model, the project uses **two fundamentally different computer vision workloads**:

* 🥔 **CNN image classification** — lightweight, low-latency inference
* 🌫️ **ResNet-based image restoration** — deeper, compute-intensive dense image-to-image inference

The objective is to understand how **architecture, tensor operations, memory requirements, and workload characteristics translate into real edge-device performance**.

---

## 🚀 Key Results

### Snapdragon X Elite NPU

| Model                          | Workload             | Architecture               |     Latency | Peak Memory | Execution                   |
| ------------------------------ | -------------------- | -------------------------- | ----------: | ----------: | --------------------------- |
| **Potato Disease Classifier**  | Image Classification | CNN                        |  **0.2 ms** |    **2 MB** | **100% NPU**                |
| **NEXTGEN VISION AI Dehazing** | Image Restoration    | ResNet · 9 Residual Blocks | **18.7 ms** |   **24 MB** | **96/96 NPU compute units** |

> **Both models execute on the Snapdragon X Elite NPU without CPU or GPU fallback.**

These measurements were obtained through **Qualcomm AI Hub Workbench profiling on the target platform**, rather than relying on theoretical hardware specifications.

### Why the difference matters

The approximately **90× latency difference** is not simply a benchmark number — it illustrates an important Edge AI engineering trade-off.

The classifier performs a relatively lightweight categorical prediction, while the dehazing network performs **dense full-resolution image transformation through nine residual blocks**.

This makes the two models useful as complementary deployment experiments:

**Lightweight CNN**

`Input → Feature Extraction → Classification`

vs.

**Image Restoration Network**

`Input → Multiple Residual Blocks → Dense Feature Transformation → Reconstructed Image`

The result is a practical demonstration of how **model architecture and workload complexity directly affect latency and memory on constrained edge hardware**.

---

# 🎯 What This Project Demonstrates

This repository focuses on the engineering layer between **model training and real hardware deployment**.

### Edge AI Deployment

* ✅ Exporting trained models from **PyTorch and TensorFlow/Keras**
* ✅ Converting framework-native models to **ONNX**
* ✅ Validating ONNX inference against the original framework
* ✅ Compiling models through the **Qualcomm QNN toolchain**
* ✅ Deploying to **Snapdragon X Elite**
* ✅ Profiling inference directly on the target platform
* ✅ Analyzing **latency, memory consumption, and compute allocation**
* ✅ Verifying NPU execution rather than assuming acceleration
* ✅ Troubleshooting real framework/conversion compatibility issues

### Heterogeneous Compute

The deployment workflow is designed around the reality that edge devices contain multiple compute resources:

**CPU → GPU → NPU**

The experiments therefore focus not only on whether a model runs, but **where the computation runs and how efficiently the target hardware executes it**.

---

# 🧠 Deployment Pipeline

```text
                    MODEL TRAINING
                         │
              ┌──────────┴──────────┐
              │                     │
           PyTorch             TensorFlow/Keras
              │                     │
              └──────────┬──────────┘
                         ▼
                  ONNX EXPORT
                         │
                         ▼
              ONNX GRAPH VALIDATION
                         │
                         ▼
             Qualcomm AI Hub Workbench
                         │
                         ▼
                  QNN COMPILATION
                         │
                         ▼
               Snapdragon X Elite
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
            CPU        GPU         NPU
                                   │
                                   ▼
                         Hardware Profiling
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
              Latency           Memory       Compute Allocation
```

---

# 🔬 Experiment 01 — Potato Disease Classifier

## Lightweight CNN → ONNX → Snapdragon NPU

The first deployment uses a CNN trained for **potato leaf disease classification**:

* Early Blight
* Late Blight
* Healthy

The model was exported from **TensorFlow/Keras → ONNX**, validated for inference parity, compiled through Qualcomm's deployment stack, and profiled on Snapdragon X Elite.

### Benchmark

**0.2 ms inference**

**2 MB peak memory**

**100% NPU execution**

This experiment establishes the baseline for a lightweight vision workload and verifies the complete deployment path from a trained framework model to **hardware-accelerated NPU inference**.

### Deployment Flow

```text
TensorFlow / Keras
        ↓
   ONNX Export
        ↓
ONNX Runtime Validation
        ↓
Qualcomm AI Hub
        ↓
       QNN
        ↓
Snapdragon X Elite NPU
        ↓
0.2 ms / 2 MB
```

📁 **Implementation:** [`potato-classifier/`](./potato-classifier)

---

# 🔬 Experiment 02 — NEXTGEN VISION AI Dehazing

## PyTorch ResNet → ONNX → QNN → Snapdragon NPU

The second experiment uses the **NEXTGEN VISION AI** dehazing network developed for real-time adverse-weather vision enhancement.

Unlike the lightweight classifier, this model performs **image-to-image restoration**, requiring dense computation across the input image.

### Architecture

**PyTorch**

→ **ResNet-based architecture**

→ **9 residual blocks**

→ **Dense image transformation**

→ **Dehazed output**

### Benchmark

**18.7 ms inference**

**24 MB peak memory**

**96/96 NPU compute units allocated**

**No CPU/GPU fallback**

This deployment acts as the more demanding workload in the experiment, testing whether the conversion and acceleration pipeline remains effective for a substantially deeper vision network rather than only a small classification model.

### Deployment Flow

```text
PyTorch
   ↓
ONNX Export
   ↓
ONNX Validation
   ↓
Qualcomm AI Hub
   ↓
QNN Compilation
   ↓
Snapdragon X Elite NPU
   ↓
18.7 ms / 24 MB
```

📁 **Implementation:** [`dehazing-model/`](./dehazing-model)

---

# 📈 What the Benchmarks Reveal

The experiments illustrate several practical principles of Edge AI engineering.

### 1. Model architecture matters

Two models can solve vision problems while having dramatically different hardware requirements.

A lightweight CNN can achieve sub-millisecond inference, while a deeper image-restoration network requires substantially more computation and memory.

### 2. NPU acceleration is workload-dependent

Successful Edge AI deployment is not simply:

> "Convert model → run on NPU."

The actual deployment requires understanding:

* supported operators
* tensor layouts
* graph compatibility
* memory requirements
* compiler constraints
* framework interoperability
* fallback behavior
* target hardware characteristics

### 3. Benchmarking must happen on the target device

Desktop inference performance does not necessarily represent edge performance.

For this reason, the reported measurements are based on **Snapdragon X Elite profiling**, allowing the deployment to be evaluated against the actual target accelerator.

### 4. Portability requires validation

ONNX provides an important interoperability layer between training frameworks and deployment runtimes.

The workflow therefore validates the converted model before using hardware benchmarks:

```text
Original Model
      │
      ├──────────────┐
      ▼              ▼
Framework        ONNX Model
Inference        Inference
      │              │
      └──────┬───────┘
             ▼
       Numerical Parity
             │
             ▼
      Hardware Deployment
```

---

# 🛠️ Real Deployment Problems Solved

The project deliberately documents **deployment engineering problems**, rather than presenting conversion as a one-command process.

### Framework compatibility

Resolved version and interoperability issues involving:

* TensorFlow/Keras versions
* ONNX conversion tooling
* SavedModel interoperability
* framework-specific graph representations

### Tensor and graph compatibility

Investigated and resolved:

* tensor naming mismatches
* input/output naming differences
* ONNX graph compatibility
* framework-native vs deployment-runtime representations

### Validation before benchmarking

The models were not treated as successfully deployed merely because an ONNX file was generated.

The workflow included **inference parity validation** before trusting the edge-device benchmark.

This distinction is important:

```text
"ONNX file generated"
          ≠
"Model successfully deployed"
```

A successful deployment requires:

```text
Conversion
   +
Numerical Validation
   +
Compilation
   +
Target Hardware Execution
   +
Performance Profiling
```

---

# 🧩 Repository Structure

```text
snapdragon-edge-ai-deployment/
│
├── README.md
│
├── potato-classifier/
│   ├── convert_to_onnx.py
│   ├── test_onnx.py
│   ├── submit_to_aihub.py
│   └── profile_cpu.py
│
└── dehazing-model/
    ├── dehaze_net.py
    ├── export_dehaze_onnx.py
    ├── test_dehaze_onnx.py
    ├── submit_dehaze_to_aihub.py
    └── profile_dehaze.py
```

---

# 🧰 Technology Stack

### Model Development

* **Python**
* **PyTorch**
* **TensorFlow**
* **Keras**

### Model Interoperability

* **ONNX**
* **ONNX Runtime**

### Edge Deployment

* **Qualcomm Snapdragon X Elite**
* **Qualcomm AI Hub Workbench**
* **Qualcomm QNN**
* **Windows on ARM**

### Profiling

* NPU execution analysis
* Inference latency
* Peak memory
* Compute-unit allocation
* CPU/GPU/NPU execution behavior

---

# 🔗 Why This Matters for Edge AI

The central engineering question behind this repository is:

> **How do you take a model that works in a training environment and turn it into an efficient model that actually runs on specialized edge silicon?**

That requires more than model training.

It requires understanding the entire deployment stack:

```text
MODEL
  ↓
Framework
  ↓
ONNX
  ↓
Compiler / Runtime
  ↓
Hardware Accelerator
  ↓
Memory
  ↓
Latency
  ↓
Real-Time Application
```

This project therefore sits at the intersection of:

**Computer Vision + Model Optimization + Hardware Acceleration + Edge Deployment**

---

# 🧠 Broader Edge AI Direction

This repository represents the **vision-model deployment layer** of a broader AI engineering focus.

The next stage is to extend the same hardware-aware methodology beyond CNNs and image restoration toward:

### Small Language Models

Deploy compact transformer/SLM workloads directly on NPU hardware while evaluating:

* latency
* memory footprint
* throughput
* context length
* quantization
* CPU/NPU partitioning

### Quantization

Evaluate:

```text
FP32
  ↓
FP16
  ↓
INT8
  ↓
Hardware-optimized inference
```

with explicit measurement of the trade-off between:

**model accuracy ↔ latency ↔ memory ↔ power**

### Cross-platform Edge AI

Extend the same models across:

```text
Qualcomm Snapdragon
        │
        ├── QNN
        │
        └── NPU
              
Apple Silicon
        │
        ├── Core ML
        │
        └── MLX
```

allowing deployment characteristics to be compared across different accelerator ecosystems.

---

# 🔮 Roadmap

### Edge Model Optimization

* [ ] INT8 quantization for both deployed models
* [ ] FP16 vs INT8 accuracy/latency comparison
* [ ] CPU vs GPU vs NPU benchmark comparison
* [ ] Operator-level performance investigation
* [ ] Memory optimization

### Edge Generative AI

* [ ] Deploy a compact LLM/SLM on Snapdragon NPU
* [ ] Quantized transformer inference
* [ ] Measure tokens/sec, first-token latency, and memory
* [ ] Investigate NPU/CPU workload partitioning

### Cross-Platform Deployment

* [ ] Apple Silicon deployment
* [ ] Core ML conversion
* [ ] MLX experimentation
* [ ] Qualcomm vs Apple accelerator comparison

### Edge + Cloud AI

* [ ] Hybrid edge/cloud inference architecture
* [ ] Intelligent model routing
* [ ] On-device preprocessing + cloud reasoning
* [ ] Edge inference integrated with enterprise AI/RAG workflows

---

# 🏗️ Engineering Principles

This project follows several principles that guide the deployment experiments:

### **Measure, don't assume**

Hardware acceleration is verified through profiling rather than inferred from the existence of an accelerator.

### **Validate before optimizing**

Numerical correctness is established before interpreting performance measurements.

### **Optimize for the target**

A model optimized for desktop inference is not automatically optimized for an NPU.

### **Treat deployment as part of model engineering**

The model is only one component of the system.

```text
Model
+
Graph
+
Compiler
+
Runtime
+
Memory
+
Hardware
=
Real Edge AI System
```

---

# 👩‍💻 Author

## Thejaswini Sunil

**AI & Cloud Automation Engineer**

**Edge AI · Computer Vision · Model Deployment · Agentic AI · Cloud Automation**

I build AI systems across the model-development and deployment stack — from training and computer vision pipelines to **agentic AI systems, RAG pipelines, cloud automation, and hardware-accelerated inference**.

This repository focuses specifically on the **Edge AI and model deployment side** of that work.

🔗 **GitHub:** [ThejaswiniSunil](https://github.com/ThejaswiniSunil)

---

# ⭐ Project Summary

> **Two independently trained vision models. Two different computational workloads. One hardware-aware deployment pipeline.**

**TensorFlow / PyTorch**

→ **ONNX**

→ **Qualcomm AI Hub**

→ **QNN**

→ **Snapdragon X Elite**

→ **NPU**

→ **Measured Hardware Performance**

### Current demonstrated results

**0.2 ms** — Potato Disease CNN
**18.7 ms** — NEXTGEN VISION AI Dehazing
**100% NPU** — Classifier
**96/96 NPU compute units** — Dehazing
**Zero CPU/GPU fallback**

The goal is not simply to make models run on edge hardware.

**The goal is to understand, measure, and optimize how AI models execute on the hardware they are actually deployed to.**
