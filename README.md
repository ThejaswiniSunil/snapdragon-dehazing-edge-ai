# ⚡ Snapdragon Edge AI & On-Device AI Deployment

### From Trained Models to Hardware-Accelerated Vision and Generative AI

**Hardware-aware Edge AI deployment experiments on Qualcomm Snapdragon X Elite, covering model export, ONNX conversion, QNN compilation, NPU execution, model optimization, SLM deployment, and hardware-level performance profiling.**

This repository demonstrates the engineering path from **framework-native AI models → portable model representations → Qualcomm AI Hub → QNN → Snapdragon X Elite → NPU-accelerated inference**.

The project spans multiple AI workloads rather than a single model:

* 🥔 **CNN image classification** — lightweight, low-latency inference
* 🌫️ **Image restoration** — deeper ResNet-based dense inference
* 🤖 **Small Language Model (SLM)** — Llama 3.2 1B Instruct for on-device generative AI

The objective is to understand how **model architecture, operators, memory requirements, graph structure, numerical representation, and workload characteristics translate into real performance on specialized edge silicon.**

---

# 🔥 What This Repository Demonstrates

| Capability                  | Demonstrated Work                                   |
| --------------------------- | --------------------------------------------------- |
| **Edge AI Deployment**      | Real model deployment on Snapdragon X Elite         |
| **NPU Acceleration**        | Qualcomm NPU execution and hardware profiling       |
| **ONNX**                    | TensorFlow/PyTorch → ONNX conversion and validation |
| **QNN**                     | Qualcomm QNN-based deployment                       |
| **Qualcomm AI Hub**         | Model compilation and target-platform profiling     |
| **Computer Vision**         | Classification + image restoration                  |
| **SLM Deployment**          | Llama 3.2 1B Instruct on-device                     |
| **Model Optimization**      | Hardware-aware graph and inference optimization     |
| **Quantized Inference**     | Edge-oriented numerical optimization                |
| **Heterogeneous Compute**   | CPU / GPU / NPU execution analysis                  |
| **Performance Engineering** | Latency, memory, throughput and hardware allocation |
| **Windows on ARM**          | Snapdragon X Elite development environment          |

---

# ⭐ Project Summary

> **From trained computer vision models to on-device generative AI — deployed and profiled on Qualcomm Snapdragon X Elite.**

The repository currently demonstrates three different AI workloads through a hardware-aware deployment workflow:

| Workload                         | Framework / Model      | Deployment Path   | Target         |
| -------------------------------- | ---------------------- | ----------------- | -------------- |
| 🥔 Potato Disease Classification | TensorFlow / Keras CNN | ONNX → QNN        | Snapdragon NPU |
| 🌫️ NEXTGEN VISION AI Dehazing   | PyTorch ResNet         | ONNX → QNN        | Snapdragon NPU |
| 🤖 Llama 3.2 1B Instruct         | Transformer / SLM      | Qualcomm AI Stack | Snapdragon NPU |

### Demonstrated Edge AI Capabilities

**Model Conversion**

```text
TensorFlow / PyTorch
        ↓
       ONNX
```

**Hardware Deployment**

```text
ONNX
  ↓
Qualcomm AI Hub
  ↓
QNN
  ↓
Snapdragon X Elite
```

**Hardware Acceleration**

```text
CPU / GPU / NPU
        ↓
Target-aware execution
        ↓
Hardware profiling
```

**Generative Edge AI**

```text
Llama 3.2 1B Instruct
        ↓
Optimization / Deployment
        ↓
Snapdragon X Elite
        ↓
On-device SLM inference
```

---

# 🚀 Key Results

## Snapdragon X Elite NPU

| Model                          | Workload             | Architecture               |              Latency | Peak Memory | Execution                   |
| ------------------------------ | -------------------- | -------------------------- | -------------------: | ----------: | --------------------------- |
| **Potato Disease Classifier**  | Image Classification | CNN                        |           **0.2 ms** |    **2 MB** | **100% NPU**                |
| **NEXTGEN VISION AI Dehazing** | Image Restoration    | ResNet · 9 Residual Blocks |          **18.7 ms** |   **24 MB** | **96/96 NPU compute units** |
| **Llama 3.2 1B Instruct**      | Generative AI        | Transformer / SLM          | On-device deployment |           — | Snapdragon NPU              |

The first two workloads provide measured vision benchmarks. The third extends the repository from conventional computer vision into **on-device generative AI and transformer-based inference**.

The vision measurements were obtained through **Qualcomm AI Hub Workbench profiling on the target platform**, rather than relying solely on theoretical hardware specifications.

---

# 📊 Why the Vision Benchmarks Matter

The approximately **90× latency difference** between the two vision workloads illustrates a fundamental Edge AI engineering principle:

> **Hardware performance is strongly influenced by workload architecture, not simply model size or parameter count.**

The classifier performs a relatively lightweight categorical prediction:

```text
Input Image
     ↓
Feature Extraction
     ↓
Classification
     ↓
Disease Class
```

The dehazing model performs dense image-to-image transformation:

```text
Input Image
     ↓
Residual Feature Extraction
     ↓
9 Residual Blocks
     ↓
Dense Feature Transformation
     ↓
Image Reconstruction
     ↓
Dehazed Image
```

This creates substantially different computational and memory characteristics despite both workloads being computer vision models.

---

# 🧠 Deployment Architecture

The complete deployment workflow is:

```text
                         MODEL DEVELOPMENT
                                │
                   ┌────────────┴────────────┐
                   │                         │
                PyTorch               TensorFlow/Keras
                   │                         │
                   └────────────┬────────────┘
                                ▼
                         MODEL EXPORT
                                │
                                ▼
                         ONNX CONVERSION
                                │
                                ▼
                       ONNX GRAPH VALIDATION
                                │
                                ▼
                      QUALCOMM AI HUB
                                │
                                ▼
                         QNN COMPILATION
                                │
                                ▼
                       SNAPDRAGON X ELITE
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                   CPU         GPU         NPU
                                            │
                                            ▼
                                  HARDWARE PROFILING
                                            │
                       ┌────────────────────┼───────────────────┐
                       ▼                    ▼                   ▼
                    Latency              Memory          Compute Allocation
```

The SLM path extends this architecture to transformer-based generative inference:

```text
Llama 3.2 1B Instruct
          ↓
   Model Preparation
          ↓
 Qualcomm AI Tooling
          ↓
  Optimization / Export
          ↓
       QNN Stack
          ↓
 Snapdragon X Elite
          ↓
   NPU-Accelerated
   On-Device Inference
```

---

# 🔬 Experiment 01 — Potato Disease Classifier

## Lightweight CNN → ONNX → Snapdragon NPU

The first deployment uses a CNN trained for potato leaf disease classification:

* Early Blight
* Late Blight
* Healthy

The model was exported from **TensorFlow/Keras → ONNX**, validated for inference parity, compiled through Qualcomm's deployment stack, and profiled on Snapdragon X Elite.

### Benchmark

* **0.2 ms inference latency**
* **2 MB peak memory**
* **100% NPU execution**

This experiment establishes the baseline for a lightweight edge workload and verifies the complete path from a trained framework model to **hardware-accelerated NPU inference**.

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

📁 **Implementation:** [`potato-classifier/`](https://github.com/ThejaswiniSunil/snapdragon-dehazing-edge-ai/tree/main/potato-classifier)

---

# 🔬 Experiment 02 — NEXTGEN VISION AI Dehazing

## PyTorch ResNet → ONNX → QNN → Snapdragon NPU

The second experiment deploys the **NEXTGEN VISION AI** dehazing network developed for real-time adverse-weather vision enhancement.

Unlike the lightweight classifier, the dehazing network performs **dense image-to-image restoration**, requiring significantly more computation across the input image.

### Architecture

```text
PyTorch
   ↓
ResNet-based Architecture
   ↓
9 Residual Blocks
   ↓
Dense Feature Transformation
   ↓
Image Reconstruction
   ↓
Dehazed Output
```

### Benchmark

* **18.7 ms inference latency**
* **24 MB peak memory**
* **96/96 NPU compute units allocated**
* **No CPU/GPU fallback**

This deployment provides a substantially more demanding workload for evaluating the Snapdragon NPU pipeline.

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

📁 **Implementation:** [`dehazing-model/`](https://github.com/ThejaswiniSunil/snapdragon-dehazing-edge-ai/tree/main/dehazing-model)

---

# 🔬 Experiment 03 — Llama 3.2 1B Instruct

## Small Language Model → Qualcomm AI Stack → Snapdragon NPU

The third deployment extends the repository beyond computer vision into **on-device generative AI**.

The workload uses **Meta Llama 3.2 1B Instruct**, a compact instruction-tuned Small Language Model (SLM), targeted for execution on the **Qualcomm Snapdragon X Elite**.

This introduces a fundamentally different inference workload:

```text
Computer Vision
      ↓
Tensor / Image Inference
      ↓
Predetermined Output
```

versus:

```text
SLM
 ↓
Autoregressive Transformer Inference
 ↓
Token Generation
 ↓
Generated Sequence
```

The objective is to understand how a transformer-based language model can be prepared, optimized, and deployed for **resource-constrained edge inference**.

---

## 🧠 Why an SLM?

Language-model deployment introduces constraints that differ significantly from conventional image inference.

The experiment therefore considers:

* Model size
* Memory footprint
* Quantized / optimized inference
* NPU compatibility
* Operator and graph support
* Hardware-aware execution
* Token generation
* Context length
* CPU/NPU workload placement
* On-device inference

---

## Model

### Llama 3.2 1B Instruct

The 1B-parameter model provides a practical edge deployment target for investigating transformer inference under device-level compute and memory constraints.

Rather than treating SLM deployment as simply "running an LLM locally", this experiment focuses on the engineering required to make a generative workload compatible with **specialized edge acceleration**.

---

## Qualcomm Deployment Stack

The deployment uses Qualcomm's model tooling and targets:

```text
Llama 3.2 1B Instruct
        ↓
Hugging Face
        ↓
Qualcomm qai_hub_models
        ↓
Model Preparation / Optimization
        ↓
Qualcomm AI Hub
        ↓
QNN / Qualcomm AI Stack
        ↓
Snapdragon X Elite
        ↓
NPU-Accelerated Execution
        ↓
On-Device SLM Inference
```

📁 **Implementation:** `slm-deployment/`

---

## SLM Performance Model

For conventional vision inference, latency is often the primary metric.

For generative models, the performance picture is broader:

```text
Time to First Token
        +
Tokens / Second
        +
Memory Footprint
        +
Context Length
        +
Compute Utilization
        +
CPU/NPU Workload Placement
```

If benchmark measurements are available, they should be reported here rather than using placeholder values.

Example:

| Metric              |    Snapdragon X Elite |
| ------------------- | --------------------: |
| Model               | Llama 3.2 1B Instruct |
| Precision           |        Measured value |
| Memory              |        Measured value |
| NPU execution       |       Measured result |
| Token generation    |   Measured tokens/sec |
| First-token latency |           Measured ms |

---

# 📈 What the Experiments Reveal

The three deployments demonstrate different dimensions of Edge AI engineering.

## 1. Model architecture matters

A lightweight CNN, a dense image-restoration network, and an autoregressive transformer place very different demands on edge hardware.

```text
CNN
 ↓
Low-latency tensor inference

Image Restoration
 ↓
Dense high-compute image transformation

SLM
 ↓
Autoregressive token generation
```

The deployment stack therefore cannot be evaluated using a single benchmark metric.

---

## 2. NPU acceleration is workload-dependent

Successful Edge AI deployment is not simply:

```text
Convert Model
      ↓
Run on NPU
```

A real deployment requires understanding:

* Supported operators
* Tensor layouts
* Graph compatibility
* Memory requirements
* Compiler constraints
* Runtime behavior
* Framework interoperability
* Fallback behavior
* Target hardware characteristics

---

## 3. Hardware profiling matters

The important question is not merely:

> **"Does the model run?"**

It is:

> **"Where does it run, how much hardware does it use, and what performance does that produce?"**

The experiments therefore investigate:

```text
Model
 ↓
Runtime
 ↓
CPU / GPU / NPU
 ↓
Memory
 ↓
Latency
 ↓
Hardware Utilization
```

---

## 4. Edge deployment requires validation

Generating an ONNX file does not prove successful deployment.

The workflow separates:

```text
Conversion
   ↓
Numerical Validation
   ↓
Compilation
   ↓
Target Hardware Execution
   ↓
Performance Profiling
```

This distinction is central to the repository:

```text
"ONNX file generated"
          ≠
"Model successfully deployed"
```

A deployment is treated as successful only when the model can execute correctly on the target platform and its behavior can be measured.

---

# 🛠️ Real Deployment Engineering

This repository documents deployment engineering rather than treating model conversion as a one-command operation.

## Framework Compatibility

The deployment workflow addresses interoperability between:

* TensorFlow/Keras
* PyTorch
* ONNX
* ONNX Runtime
* Qualcomm AI Hub
* QNN

This includes investigating version compatibility and framework-specific graph representations.

---

## Tensor and Graph Compatibility

Deployment work includes investigating:

* Input/output naming
* Tensor shapes
* Tensor layouts
* ONNX graph compatibility
* Operator support
* Framework-native representations
* Deployment-runtime representations

---

## Validation Before Benchmarking

The converted models are validated before hardware benchmarks are interpreted.

The workflow is therefore:

```text
Original Framework
        │
        ├──────────────┐
        ▼              ▼
 Framework        ONNX Runtime
 Inference         Inference
        │              │
        └──────┬───────┘
               ▼
        Numerical Parity
               │
               ▼
       Hardware Deployment
               │
               ▼
          Profiling
```

---

# ⚙️ Model Optimization

The project focuses on optimization at the intersection of **model architecture, numerical representation, graph compilation, and hardware execution**.

Key areas include:

* ONNX graph optimization
* QNN compilation
* Quantized inference
* Memory optimization
* Operator compatibility
* NPU execution
* CPU/NPU workload placement
* Hardware-aware benchmarking

The objective is not optimization in isolation.

It is:

```text
Accuracy
   ↕
Latency
   ↕
Memory
   ↕
Throughput
   ↕
Hardware Utilization
```

with the trade-offs measured on the target device whenever possible.

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
├── dehazing-model/
│   ├── dehaze_net.py
│   ├── export_dehaze_onnx.py
│   ├── test_dehaze_onnx.py
│   ├── submit_dehaze_to_aihub.py
│   └── profile_dehaze.py
│
└── slm-deployment/
    ├── model preparation
    ├── deployment scripts
    ├── validation
    └── profiling
```

---

# 🧰 Technology Stack

## Model Development

* Python
* PyTorch
* TensorFlow
* Keras

## Model Interoperability

* ONNX
* ONNX Runtime

## Edge Deployment

* Qualcomm Snapdragon X Elite
* Qualcomm AI Hub
* Qualcomm `qai_hub_models`
* Qualcomm QNN
* Windows on ARM

## Generative AI

* Llama 3.2 1B Instruct
* Transformer inference
* Small Language Model deployment
* On-device generative AI

## Performance Engineering

* NPU execution analysis
* Inference latency
* Peak memory
* Compute allocation
* CPU/GPU/NPU execution behavior
* Token-generation performance

---

# 🧠 Heterogeneous Edge Compute

Modern edge devices expose multiple compute resources rather than a single processor.

This repository treats the platform as a heterogeneous system:

```text
                 Snapdragon X Elite
                        │
           ┌────────────┼────────────┐
           │            │            │
          CPU          GPU          NPU
           │            │            │
           └────────────┼────────────┘
                        │
                AI Workload
```

The deployment objective is therefore not simply to maximize raw model performance.

It is to understand:

* Which accelerator executes the workload
* Which operations are supported
* How memory moves through the system
* Where fallbacks occur
* How model architecture affects hardware utilization
* How deployment decisions affect latency and throughput

This hardware-aware approach is particularly important for **real-time computer vision and on-device generative AI**, where resource constraints directly affect application behavior.

---

# 🔗 From Edge Models to Real Applications

The edge deployments in this repository are connected to larger AI systems rather than existing as isolated benchmarks.

### NEXTGEN VISION AI

The dehazing workload originates from **NEXTGEN VISION AI**, a real-time adverse-weather vision enhancement system.

```text
Camera / Video
      ↓
Image Enhancement
      ↓
Dehazing
      ↓
Object Detection
      ↓
Road / Hazard Understanding
      ↓
Real-Time Vision Application
```

The Snapdragon deployment explores how the computationally intensive vision component can move toward specialized edge acceleration.

### Generative AI

The Llama deployment extends the same hardware-aware philosophy from perception to language:

```text
Perception
   ↓
Computer Vision Models
   ↓
NPU

Language
   ↓
Small Language Models
   ↓
NPU
```

Together, these experiments explore a broader edge-AI architecture where **multiple classes of AI workloads can execute close to the device rather than relying exclusively on cloud inference**.

---

# 🌐 Broader Edge AI Direction

This repository now covers both **computer vision and generative AI deployment on Qualcomm Snapdragon hardware**.

The progression is:

```text
                    Snapdragon Edge AI
                           │
             ┌─────────────┴─────────────┐
             │                           │
        Computer Vision             Generative AI
             │                           │
       ┌─────┴─────┐                     │
       │           │                     │
   Classifier   Dehazing             Llama 3.2 1B
       │           │                     │
   TensorFlow   PyTorch                 SLM
       │           │                     │
       └─────┬─────┘                     │
             │                           │
            ONNX                  Model Optimization
             │                           │
             └────────────┬──────────────┘
                          │
                    Qualcomm AI Hub
                          │
                         QNN
                          │
                   Snapdragon X Elite
                          │
                         NPU
```

The result is a deployment portfolio spanning:

**CNN inference → image restoration → transformer-based SLM inference**

---

# 🔬 Edge AI Engineering Focus

The work in this repository can be viewed as four connected layers:

```text
┌─────────────────────────────────────┐
│        AI MODEL DEVELOPMENT         │
│  CNN · ResNet · Transformer · SLM   │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│       MODEL REPRESENTATION          │
│      ONNX · Graph Optimization      │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│       HARDWARE DEPLOYMENT            │
│   AI Hub · QNN · Snapdragon NPU     │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│       PERFORMANCE ENGINEERING       │
│ Latency · Memory · Throughput · NPU │
└─────────────────────────────────────┘
```

This is the core engineering theme of the project:

> **Taking models beyond training and making them measurable, deployable, and efficient on real edge hardware.**

---

# 🔭 Cross-Platform Edge AI

The Qualcomm deployment provides a foundation for extending the same hardware-aware methodology to other accelerator ecosystems.

```text
Qualcomm Snapdragon
        │
        ├── QNN
        └── NPU

Apple Silicon
        │
        ├── Core ML
        └── MLX
```

Future cross-platform experiments can investigate how the same model architectures and optimization strategies translate across different hardware and software stacks.

These technologies are treated as **future extension areas unless a corresponding deployment is included in the repository**.

---

# ☁️ Edge + Cloud AI

Edge inference is one component of a broader AI systems architecture.

My wider AI engineering work also covers:

```text
                   AI SYSTEM
                       │
          ┌────────────┴────────────┐
          │                         │
        EDGE                      CLOUD
          │                         │
    NPU Inference              AI Services
          │                         │
    Vision / SLM              Agentic AI
          │                         │
          │                       RAG
          │                         │
          └────────────┬────────────┘
                       │
                Intelligent Routing
```

This creates opportunities for hybrid architectures such as:

* On-device preprocessing
* Local vision inference
* On-device SLM inference
* Cloud-based reasoning
* RAG-backed enterprise workflows
* Intelligent edge/cloud model routing

The broader objective is to understand **where each AI workload should execute based on latency, compute, memory, privacy, connectivity, and workload requirements**.

---

# 🏗️ Engineering Principles

## Measure, don't assume

Hardware acceleration is verified through profiling rather than inferred from the existence of an accelerator.

## Validate before optimizing

Numerical correctness is established before interpreting performance measurements.

## Optimize for the target

A model optimized for desktop inference is not automatically optimized for an NPU.

## Understand the full stack

Edge AI performance is determined by more than the neural network:

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
Accelerator
 =
Real Edge AI System
```

## Deployment is part of model engineering

A trained model is not the final product.

The engineering workflow continues through:

```text
Training
   ↓
Export
   ↓
Conversion
   ↓
Validation
   ↓
Optimization
   ↓
Compilation
   ↓
Hardware Deployment
   ↓
Profiling
   ↓
Application Integration
```

---

# 📌 Why This Repository Matters

The central engineering question behind this project is:

> **How do you take an AI model that works in a training environment and turn it into an efficient workload that actually executes on specialized edge silicon?**

Answering that requires understanding the entire deployment stack:

```text
MODEL
  ↓
Framework
  ↓
ONNX
  ↓
Graph / Compiler
  ↓
Runtime
  ↓
Hardware Accelerator
  ↓
Memory
  ↓
Latency / Throughput
  ↓
Real-Time Application
```

This project therefore sits at the intersection of:

**Edge AI + Computer Vision + Generative AI + Model Optimization + Hardware Acceleration + Deployment Engineering**

---

# 👩‍💻 Author

## Thejaswini Sunil

**AI & Cloud Automation Engineer**

**Edge AI · Computer Vision · SLM Deployment · Model Optimization · Agentic AI · RAG · Cloud Automation**

I build AI systems across the model-development and deployment stack — from **computer vision and model training to hardware-accelerated inference, Small Language Models, agentic AI systems, RAG pipelines, and cloud automation**.

This repository focuses specifically on the **Edge AI and model deployment side** of that work.

🔗 **GitHub:** [ThejaswiniSunil](https://github.com/ThejaswiniSunil)

---

# ⭐ Final Snapshot

> **Trained models → ONNX → Qualcomm AI Hub → QNN → Snapdragon X Elite → NPU → measured hardware performance**

### Computer Vision

**Potato Disease CNN**

→ **0.2 ms**

→ **2 MB**

→ **100% NPU**

### Image Restoration

**NEXTGEN VISION AI Dehazing**

→ **18.7 ms**

→ **24 MB**

→ **96/96 NPU compute units**

### Generative AI

**Llama 3.2 1B Instruct**

→ **Qualcomm AI Stack**

→ **Snapdragon X Elite**

→ **On-device SLM inference**

---

## The Engineering Goal

The goal is not simply to make models run on edge hardware.

**The goal is to understand, measure, optimize, and deploy different classes of AI workloads on the hardware they actually execute on.**

**From CNNs to image restoration to Small Language Models — from model training to hardware-accelerated inference.**

