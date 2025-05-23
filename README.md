# El Silencio Acoustic Explorer: An MLOps Pipeline for Real-time Bioacoustic Monitoring

## Value proposition

This project proposes a machine learning system, "El Silencio Acoustic Explorer," designed for integration into the existing services of Ecotour Operators in the El Silencio Natural Reserve.

## 1. Value Proposition

**"El Silencio Acoustic Explorer"** is a real-time bioacoustic monitoring system designed to enhance the eco-tourism experience within _El Silencio Natural Reserve_. By providing an API endpoint for integration into a mobile application (app development outside this project's scope), our system enables tour guides to instantly identify vocalizing fauna (birds, amphibians, mammals, insects) directly from audio recordings captured during tours. This delivers the following benefits:

- **Reveals Hidden Biodiversity:** Uncovers species often missed by visual observation, significantly enriching the tour experience.
- **Increases Customer Engagement and Education:** Provides real-time information, transforming passive observation into an interactive learning experience.
- **Empowers Tour Guides:** Equips guides with a powerful tool for accurate and immediate species identification, enhancing their expertise and credibility.
- **Unique Selling Proposition:** Creates a distinctive and technologically advanced tour offering, attracting a wider range of eco-tourists.
- **Data Collection for Ecological Studies:** Provides valuable data for ecological studies conducted within the El Silencio Natural Reserve.

### Target Customers

Specifically: [El Silencio Silvestre](https://elsilenciosilvestre.org/)

Other Local El Silencio Natural Reserve **eco-tour operators**.
- [Tree Hugger Travel](https://www.treehuggertravel.com.au/eco-rewards-points-programme/supporting-wlt-in-el-silencio-nature-reserve-colombia/)
- [Manakin Nature Tours](https://www.manakinnaturetours.com/natural-reserves-and-protected-areas/)

Potential expansion:

- Colombia Birdwatch
- Andes EcoTours
- Selva Verde Lodge & Private Reserve

---

## 2. Non-ML Status Quo

Currently, fauna identification in El Silencio Natural Reserve tours relies heavily on:

- **Visual Spotting:** Limited to species that are easily visible, missing many nocturnal or elusive species.
- **Human Guide Acoustic Skills:** Subjective and variable accuracy, leading to missed identifications and uncertainty.
- **Manual Reference Books:** Inefficient and not real time.

This results in an incomplete and potentially inaccurate portrayal of the reserve's biodiversity, limiting the educational value and customer satisfaction.

---

## 3. Project Success Metrics

### **Primary Technical Metric (Proxy for Core Capability):**

- **Mean Average Precision (mAP):** Target mAP > 0.5 (!! Achieved 0.63)
  _Justification:_ Directly reflects the model's accuracy in species identification, which is the foundation of the system's value.

### **Secondary Business-Oriented Metrics:**

- **Species Diversity per Tour Session:** Increase the average number of identified species per tour by compared to pre-implementation baseline data.
- **Customer Engagement Metrics (Proxy):** Increase app usage time and interaction frequency during tours.
- **Tour Guide Proficiency Enhancement (Qualitative):** Document positive feedback from tour guides regarding the system's usability and effectiveness.
- **New Tour Offering Differentiation:** Document positive feedback regarding the system's contribution to a unique selling proposition.
- **Data Collection for Ecological Studies:** Document the amount of species data collected.

## Contributors

| Name                      | Responsible for                          | Link to their commits in this repo |
| :------------------------ | :--------------------------------------- | :--------------------------------- |
| Sudharshan Ramesh         | _Model training and training platforms_  | [Link](https://github.com/exploring-curiosity/MLOps/commit/7ec8ed17fabe7fd0cdee15cfc69fa24f580612cc)                                   |
| Vaishnavi Deshmukh        | _Data pipeline_                          | [Link](https://github.com/exploring-curiosity/MLOps/commit/8ceb0d980d012b9107320b73b1f076594d53de16)                                   |
| Mohammad Hamid            | _Continuous X CI/CD_                     | [Link](https://github.com/exploring-curiosity/MLOps/commit/b0420543b480bccfb5286a535de15502e8e6ffe9)                                   |
| Harish Balaji Boominathan | _Model serving and monitoring platforms_ | [Link](https://github.com/exploring-curiosity/MLOps/commit/ccde2eb005f322b3a037927b81b77124774a66a0)                                   |

### System diagram

<img src="assets/ELSilencioAcousticExplorerSysArch.png" width="800"/>

### Summary of outside materials

| Resource                                                       | How it was created                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Conditions of use                                                                                                                                                                                                                                      |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data set 1:** BirdCLEF 2025 (El Silencio Focus)              | Dataset provided via Kaggle for the BirdCLEF 2025 competition. Focuses on El Silencio Natural Reserve, Colombia. Includes: <br> - `train_audio/`: Short labeled recordings (primary/secondary labels for 206 species: birds, amphibians, mammals, insects) from Xeno-canto, iNaturalist, CSA. 32kHz OGG format. **(7.82 GB)**. Metadata in `train.csv`. <br> - `train_soundscapes/`: Unlabeled 1-minute soundscapes from El Silencio. 32kHz OGG format. **(4.62 GB)**. <br> **Total provided training/unlabeled data (`train_audio` + `train_soundscapes`) is approx. 12.5 GB.** [https://www.kaggle.com/competitions/birdclef-2025/data](https://www.kaggle.com/competitions/birdclef-2025/data) | License: **CC BY-NC-SA 4.0**. For research purposes: Requires attribution (BY), prohibits commercial use (NC), and requires adaptations be shared under the same license (SA). Suitable for non-commercial academic research.                          |
| **Embedding model:** Pre-trained SED Model: PANNs (CNN14 variant) | PANNs (Large-Scale Pretrained Audio Neural Networks) by Kong et al. (2020). Trained on AudioSet. CNN14 architecture. Paper: [https://arxiv.org/abs/1912.10211](https://arxiv.org/abs/1912.10211), Code/Weights: [https://github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)                                                                                                                                                                                                                                                     | License: **MIT License** (as per linked repository). Permits reuse, modification, distribution, and sublicensing for both private and commercial purposes, provided the original copyright and license notice are included. Suitable for research use. |
| **Base model:** EfficientNet-B3                              | Developed by Google Research (Tan & Le, 2019). Pre-trained on ImageNet. Smallest variant (\~5.3M parameters). Paper: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)                                                                                                                                                                                                                                                                                                                                                                                        | Apache 2.0 License. Pre-trained weights available. Will be **fine-tuned** on `train_audio` (Mel spectrograms) for the 206-species classification task.                                                                                                 |
| **Architecture:** ResNet-50                                  | Architecture by Microsoft Research (He et al., 2015). (\~25.6M parameters). Paper: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)                                                                                                                                                                                                                                                                                                                                                                                                                          | Architecture well-established. Will be **trained from scratch** on `train_audio` (Mel spectrograms) for the 206-species classification task. No ImageNet pre-training used. Fulfills "train from scratch" requirement.                                 |
| **Tool:** Ray                                                  | Open-source framework for distributed computing. [https://www.ray.io/](https://www.ray.io/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Apache 2.0 License. Used for scheduling training jobs and distributed training on Chameleon.                                                                                                                                                           |
| **Tool:** MLflow                                               | Open-source platform for MLOps. [https://mlflow.org/](https://mlflow.org/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Apache 2.0 License. Will be self-hosted on Chameleon for experiment tracking.                                                                                                                                                                          |

### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`),
how much/when, justification. Include compute, floating IPs, persistent storage.
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
| --------------- | ------------------------------------------------- | ------------- |
| `m1.medium` VMs | 3 for entire project duration  <br> - 1 for Development/Coordination (Project duration). <br> - 1 for All Services (Project duration). <br> - 1 Optional later for API Integrations.                    | Essential for hosting persistent services (MLflow), managing training cluster (Ray Head), stable development environment, potential API serving. Assumes CPU-only sufficient.         |
| `m1.xxlarge` VM | 1 for starting project setup data pipeline | Essential for setting up big data from online source into bucket and preprocessing.
| `gpu_rtx_6000`     | Access to 1 GPU for Training and 1 GPU for Serving and Inference. 6 hour block thrice a week                         | Required for intensive ResNet-50 scratch training, fine-tuning of EffNetB3 and training other base, Supervisor Models. RTX 6000 GPU's have enough memory and power to fit and train every model one at a tim. Usage is intermittent but demanding.              |
| Floating IPs    | 	2 required for project duration. <br> +3 Developemental use. | Provides persistent external IPs for Development VM (SSH) and MLflow Server UI. Potential occasional IP for training serving and inference.             |
| Persistent Storage | Estimate: 100 GB needed throughout project duration. | Required for reliable storage of dataset (~12.5GB+), source code, MLflow artifacts (models, logs, checkpoints), environments. Ensures data persistence beyond ephemeral node storage and across project sessions. |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the
diagram, (3) justification for your strategy, (4) relate back to lecture material,
(5) include specific numbers. -->

#### Model training and training platforms
For Starting the server follow through the notebook after creatign a lease (mlops_train_project38)
[Getting_Started](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Getting_started.ipynb)
After the Server is up and Running we have to connect to the Object Store. It can be done with:
[Load_Data](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/LoadData.ipynb)
Following that the Nvidia Tool Kit setup and fastAPI server and Ray workers can be configure with the notebook in (Optional Infra and Jupyter Notebook):
[Nvidia_Setup](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Nvidia_setup.ipynb)

This below section details the plan to satisfy Unit 4 and Unit 5 requirements, incorporating learnings from the course units.

**Requirement 4.1: Train and Re-train At Least One Model**
- **Trained models**  
  - **RawAudioCNN** (1D CNN on denoised waveform)  [Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Notebooks/Multi_label/RawAudioCnnMultiLabel.ipynb) [Script]()
  - **PannsEmbMLP** (MLP on PANNs embeddings)  [Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Notebooks/Multi_label/PannsEmbMultiLabel.ipynb) [Script]()
  - **EfficientNet‑B3 (LoRA + fp16 Mixed Precision)**  [Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Notebooks/Multi_label/EffNetB3MultiLabel.ipynb) [Script]()
  - **ResNet50** (on augmented mel)  [Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Notebooks/Multi_label/ResNet50MultiLabel.ipynb) [Script]()
  - **MetaMLP** (supervisor aggregator)  [Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Notebooks/Multi_label/MetaMLPMultiLabel.ipynb) [Script]()
- **Frozen feature extractor**  
  - **Panns CNN14** is _not_ trained or retrained; it only provides embeddings.  
- **Retraining pipeline** 
 [RawAudioCNN](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/retrainRawCNN.py)
 [EffNetB3_Lora](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/retrainEffNetB3.py)
 [ResNet50](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/retrainResNet50.py)
 [PannsEmbMLP](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/retrainPannsEmb.py)
 [MetaMLP](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/retrainMetaMLP.py)
  - Each of the five trainable heads has a corresponding “retrain” script that:  
    1. Pulls latest models from the past experiments and registered valid models. 
    2. Fine‑tunes for _N_ additional epochs on a mentioned data. 
    3. Logs updated weights back to the registry.  
  - Trigger via FastAPI endpoint, e.g.:  
    ```
    POST http://{fip}:9090/retrain_model?model_name=<head>&epochs=<k>
    ```  
**Requirement 4.2: Modeling Choices & Justifications**
1. **Feature‑level specialization**  [Sample_Preprocessing_Showcase](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Notebooks/AudioPreprocessingSample.ipynb)
   - **Denoised waveform → RawAudioCNN**  
     - Learns temporal filters on raw audio without spectrogram information loss  
   - **PANNs embeddings → PannsEmbMLP**  
     - High‑level timbral/pitch features from CNN14, mapped to 206‑way multi‑label output  
   - **Mel spectrogram → EfficientNet‑B3 (LoRA)**  
     - Leverages ImageNet pretraining; fine‑tunes only low‑rank adapters for efficiency  
   - **Augmented mel (mel + PANNs embeddings) → ResNet50**  
     - Deeper residual blocks exploit richer combined representation  
   - **Supervisor (MetaMLP)**  
     - Fuses the four multi‑label probability vectors, outputs final multi‑label set  
     - Primary label = highest score; secondary labels = all scores above a threshold  
    Below Notebook Will do all the processing and store it into feature chunks of 10sec duration: [PreComputeFeatures](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/Notebooks/PrecomputeFeatures.ipynb)
2. **Why these architectures?**  
   - **Panns CNN14** as frozen extractor → robust embeddings, zero extra training cost  
   - **EfficientNet‑B3** → balanced capacity vs. compute; LoRA & fp16 minimize resource footprint  
   - **RawAudioCNN / PannsEmbMLP** → lightweight, trainable from scratch on a single GPU  
   - **MetaMLP** → shallow fusion head avoids overfitting when combining heterogeneous outputs  

**Difficulty Point: Training Strategies for Large Models**  [Default with Lora](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/trainEffNetB3WithoutMixedPrecison.py) [With Precision and Lora](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/trainEffNetB3.py)
- **LoRA adapters on EfficientNet‑B3**  
  - Only low‑rank matrices are trainable; most weights remain frozen  
  - Reduces trainable parameters by > 66%  
- **Mixed‑precision (fp16) with `torch.amp.autocast`**  
  - Speeds up conv/transformer layers and increases batch size to fit in GPU from 16 to 64. 
 

**Requirement 5.1: Experiment Tracking** [Docker Setup](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/docker/docker-compose-model-train-setup.yaml)
   - **Hosted server:** Deployed an MLflow tracking server on a Chameleon VM, with MinIO running alongside as an S3‑compatible artifact store.  
   - **Instrumentation:** Each training and tuning script calls `mlflow.log_param()`, `mlflow.log_metric()`, and `mlflow.pytorch.log_model()` so that every run’s hyperparameters, metrics, and artifacts are persisted.  
   - **Workflow:**  
     1. Start MinIO container and MLflow server on Chameleon  
     2. Point `MLFLOW_TRACKING_URI` and `MINIO_ENDPOINT_URL` to those services  
     3. Run training/tuning; inspect via the MLflow UI  

**Requirement 5.2: Scheduling training jobs**  [FastAPI Server](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/main.py)
   - **Ray cluster on Chameleon:** Launched a head node and multiple worker nodes using Ray’s Ansible scripts on Chameleon.  
   - **Submission pipeline:** Training scripts (`train*.py` and `retrain*.py`) are invoked via `ray job submit --no-wait`, which returns a Job ID.  
   - **FastAPI integration:** A `/train_model` and `/retrain_model` endpoint wraps the `ray job submit` call, returns the Ray Job ID immediately, and lets you poll `/status?job_id=<ID>` to see progress and capture the `MLFLOW_RUN_ID`.  

**Difficulty Point: Scheduling hyperparameter tuning jobs** [EffNetB3_FineTune](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/hyperParameterTuneEffNetB3.py) [PannsEmbMLP_FineTune](https://github.com/exploring-curiosity/MLOps/blob/main/model_train/TrainingScripts/hyperParameterTunePannsEmb.py) 
   - **Ray Tune with ASHA:** Used `hyperParameterTuneEffNetB3.py` and `hyperParameterTunePannsEmb.py`, each instrumented with `tune.report()` and an ASHA scheduler to early‑stop unpromising trials.  
   - **Submission via API:** Exposed `/tune_effb3` and `/tune_emb_mlp` FastAPI endpoints that call `ray job submit ... python <tune_script> --epochs ... --num_samples ...`.  
**HELPER NOTEBOOK FOR USING FAST_API SERVER**
[FastAPI Help Book](FastAPIInferenceEndpoints.ipynb)

**Acheived a mAP score of 0.63 and micro F1 score of 0.54 and Primary Label classification score of 0.61. The metrics obtained were reasonable in performance to other systems, considering the ambiguity in audio data processing and the wide variety of labels**

#### Model serving and monitoring platforms

This section outlines the plan to satisfy Unit 6 and Unit 7 requirements.

# BirdCLEF Multimodal Audio Classification: Deployment & Evaluation

This project implements, optimizes, and evaluates a multimodal audio classification pipeline for the BirdCLEF dataset. The system combines multiple pre-trained branches to classify 10-second audio clips into one of 206 bird species.

## Model Serving and Evaluation
##### Model Serving : https://github.com/exploring-curiosity/MLOps/tree/main/model_serving
### Objective

To build an end-to-end model deployment pipeline that satisfies real-world constraints on model size, latency, and throughput. The project targets both high-performance server deployments and resource-constrained edge devices such as the Raspberry Pi. The system is designed to identify bird species based on their sounds, enabling bioacoustic monitoring and field-based species detection.



## Requirements Satisfied

### Serving from an API Endpoint
The BirdCLEF model is served via a FastAPI HTTP endpoint, enabling audio classification through a standardized REST interface. The endpoint accepts `.wav` audio files, applies preprocessing steps including waveform segmentation and mel spectrogram generation, and forwards the processed inputs to five ONNX-optimized model branches. The predictions from each branch are fused using a MetaMLP module, and the final output includes the predicted species label, confidence score, and inference latency.

This API structure allows seamless integration with both front-end interfaces and automated pipelines.

Relevant source file: `app.py`  
Repository link: [FastAPI Inference Endpoint](https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/BirdClassification/fastapi/app.py)

---

### Identifying Requirements
To meet the performance and deployment needs of real-time bird species classification, the following inference requirements were defined and targeted:

- **Model Size (Total)**: Under 250 MB  
- **Online Inference Latency** (Single Sample): Under 30 milliseconds  
- **Batch Throughput**: At least 10,000 frames per second  
- **Concurrency**: Support at least 8 concurrent inference requests in a cloud-based setup  

These targets were selected to ensure responsiveness in field scenarios and scalability in cloud deployments.

---

### Model-Level Optimizations
To meet the identified performance targets, several model-level optimizations were applied:

- **ONNX Conversion**: All PyTorch models were exported to ONNX with dynamic axes and compatible opsets  
- **Post-Training Quantization**: Dynamic quantization was used to reduce model size and improve inference speed  
- **Operator Compatibility**: Models were profiled to ensure reliance on CUDA-optimized operators where applicable  
- **Batch Input Support**: All branches were updated to handle batched inference efficiently for GPU utilization  

A key challenge was encountered with the EfficientNetB3 model, which was imported using the `timm` library. During ONNX export, `timm`-based models did not support the ONNX `CUDAExecutionProvider` out-of-the-box, leading to a compatibility bottleneck. To address this, the model was exported and validated using an NVIDIA RTX 6000 GPU with full CUDA support to ensure correct operator coverage and optimized kernel performance.

These optimizations resulted in a total model size of 197.51 MB and achieved a median latency of 23.64 ms per sample on the RTX 6000.

Detailed optimization process: [Quantization Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/Notebooks/ONNX_Optimization.ipynb)

---

### System-Level Optimizations
To ensure concurrency and maximize throughput on cloud hardware, the following system-level strategies were employed:

- **Model Parallelism via Triton Inference Server**: Each ONNX model branch was containerized and deployed with individual `config.pbtxt` files  
- **GPU Pinning**: Specific GPU devices were assigned to each model branch to balance computational load  
- **Dynamic Batching**: Enabled within Triton to aggregate inference requests and reduce overall processing time  
- **Concurrency Tuning**: Inference concurrency was tested across a range of thread counts to meet the target of 8+ simultaneous requests  

These measures significantly improved performance under real-time load and during bulk evaluations.

---

### Extended Serving Evaluation (Optional Difficulty)
In addition to cloud inference, the model was also deployed and tested on an edge device (Raspberry Pi 5), enabling live microphone-based inference using ONNX Runtime. This demonstrated the model's flexibility and performance across:

- **Server-grade GPU (NVIDIA RTX 6000)**
- **Server-grade CPU (quantized ONNX via ONNX Runtime)**
- **Edge deployment (Raspberry Pi 5 using optimized quantized models)**

Each deployment variant was evaluated for inference latency, resource usage, and deployment cost trade-offs.

Edge deployment details: [Edge Inference Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/Edge_Serving/edge_serving.ipynb)

## Evaluation and Monitoring

### Offline Evaluation Plan
An automated offline evaluation pipeline is implemented as part of the post-training process. Evaluation is conducted using both standard and domain-specific test cases, and results are logged using MLflow. Key components of the offline evaluation strategy include:

1. **Standard and Domain-Specific Evaluation**  
   - Evaluation is performed on the `manifest_test.csv` test set, which includes standard BirdCLEF audio samples.  
   - Additional evaluation is conducted on curated domain-specific subsets such as `insects/`, `mammalia/`, `amphibia/`, and `sound_similar_to_amphibia/`, to stress-test the model on confusing and biologically relevant edge cases.

2. **Population-Level Slice Analysis**  
   - Per-class accuracy is analyzed to identify model performance disparities across species.  
   - The 20 lowest-performing classes are tracked to assess potential biases or underrepresentation.

3. **Failure Mode Testing**  
   - Known failure modes are systematically evaluated using synthetic and mislabeled audio to test the model’s robustness against noise, overlap, and ambiguity.

4. **Unit Testing and Template-Based Evaluation**  
   - Pytest-based unit tests validate model behavior under batch inference and edge conditions.  
   - Evaluation templates automate comparisons against ground truth, handle assertions, and check API response formats.

If the evaluation passes pre-defined thresholds, the model is automatically registered in the MLflow model registry. Otherwise, it is flagged for retraining or inspection.

Notebook link: [Offline Evaluation Notebook](https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/Offline_Evaluation/Offline_eval.ipynb)

---

### Load Testing in Staging Environment

As part of the continuous deployment pipeline, a **load test** is executed in the staging environment to assess the system’s performance under high-throughput inference scenarios. The primary objectives of this test include:

- **API Stability Under Stress**  
  Evaluating the robustness and fault tolerance of the inference API when subjected to sustained concurrent requests using real audio samples.

- **Latency Scaling with Batch Size**  
  Measuring how inference latency evolves with increasing batch sizes, helping identify optimal configurations for throughput versus response time trade-offs.

- **System Resource Utilization**  
  Monitoring CPU, GPU, memory, and I/O usage during peak load conditions to ensure efficient resource allocation and identify potential bottlenecks.

The test is driven using `perf_analyzer`, a tool provided by NVIDIA Triton Inference Server, which simulates parallel requests and varying concurrency levels.

## Observability

- **Metrics Collection**  
  All relevant performance metrics—such as request latency, throughput, error rates, and GPU utilization—are scraped using **Prometheus**.

- **Visualization**  
  The metrics are displayed on custom **Grafana dashboards**, enabling real-time monitoring and post-test analysis. Dashboards include:
  - Inference latency over time
  - Throughput vs. batch size
  - GPU memory and compute utilization
  - API error rate and request distribution

This load testing stage ensures that the deployed model infrastructure meets production requirements in terms of scalability, reliability, and performance.


---

### Online Evaluation in Canary Environment
Online evaluation is performed in a simulated canary environment using artificially generated “user” requests representing diverse use cases. These include:

- Typical user behavior: common bird audio clips
- Edge behavior: overlapping or noisy inputs
- Rare species queries: low-frequency class samples
- Adversarial inputs: distorted or mislabeled examples

This process allows identification of prediction inconsistencies, latency issues, and inference failures before full deployment.

---

### Closing the Feedback Loop
To close the evaluation loop, the following mechanisms are implemented:

- **Prediction Feedback Capture**  
  - A feedback interface allows annotators or users to flag incorrect predictions or confirm correct results.
  
- **Ground Truth Logging**  
  - A portion of production inference data is stored with metadata and user feedback.  
  - Labeled samples from this set are periodically used to retrain the model or update evaluation metrics.

- **MLflow and Prometheus Logging**  
  - Prediction scores, confidence levels, and latency are logged and reviewed for anomalies.  
  - SHAP explanations are used to track input attribution shifts over time.

---

### Business-Specific Evaluation Plan
### Business Metric: Species Spotting Rate

**Definition**  
The **Species Spotting Rate** represents the average number of unique bird species identified during a tourist’s visit using AI-enhanced tools such as guided audio trails, mobile apps, or smart signage.

**Formula (Conceptual)**  
\[
\text{Species Spotting Rate} = \frac{\text{Total Unique Species Detected}}{\text{Number of Visitors or Sessions}}
\]

**Purpose**  
This metric measures the **biodiversity exposure per visitor experience**, providing insight into how rich, informative, and engaging the eco-tourism encounter is from a biological perspective.

**Business Relevance**  
- Helps quantify the **value of biodiversity** offered to tourists.
- Correlates with visitor satisfaction, reviews, and repeat visits.
- Enables benchmarking across trails, seasons, or locations.

**Application in Eco-Tourism**  
- Used to **optimize trail designs** by identifying high-density acoustic zones.
- Enables dynamic storytelling — e.g., “Spot 15 species in 30 minutes!”
- Forms a key part of **visitor experience analytics** in dashboards.

**Example**  
A guided trail with real-time audio classification detects 22 unique species across 10 tourist sessions.  
→ **Species Spotting Rate** = 22 / 10 = **2.2 species per visit**


---

### Advanced Monitoring (Extended Features)


### Monitoring for Model Degradation and System Health

The production inference system is monitored using Grafana dashboards that visualize both API behavior and model output metrics. These dashboards are powered by Prometheus metrics scraped from FastAPI and Triton Inference Server.

#### Metrics Monitored

**Model Inference Metrics**
- **Average Prediction Confidence**
- **Prediction Confidence Histogram** (over time)
- **Confidence Distribution** (bucketed ranges from ≤0.1 to ≤1.0)

**API Behavior**
- **Request Volume** per second (grouped by status: 2xx, 4xx, 5xx)
- **Error Rates** over time
- **Request Duration Percentiles**: median, 95th, and 99th percentiles
- **Average Latency** for `/predict` endpoint

**System Metrics** (via cAdvisor)
- CPU usage, memory usage, I/O rates for each container

#### Grafana Alerting 

- Alerts are defined directly in Grafana panels based on:
  - Drop in average confidence
  - Spike in latency (e.g., 95th percentile > 1 sec)
  - Surge in 4xx/5xx error rates
- Alert delivery options:
  - Email





---


<!-- ## Data pipeline

## Table of Contents

1. [Data Pipeline](#data-pipeline)
   - [Strategy & Relevant Diagram Parts](#strategy--relevant-diagram-parts)
   - [Justification & Relation to Lecture Material](#justification--relation-to-lecture-material)
   - [Specific Numbers & Implementation Details](#specific-numbers--implementation-details)
   - [Difficulty Points Attempted](#difficulty-points-attempted)
2. [Persistent Storage Justification](#persistent-storage-justification) -->

## Data Pipeline

### Strategy & Relevant Diagram Parts

Our data strategy centers on robust management of both offline (training) and online (inference) data workflows, utilizing Chameleon's persistent storage as a central repository.

- **Centralized Storage:** Chameleon's persistent storage will serve as the primary data repository, enabling efficient shared access and management of large datasets.
- **Offline Processing (Batch ETL):** An ETL pipeline will process raw dataset batches, preparing them for model training. This includes feature extraction such as Mel spectrograms.
- **Online Processing (Simulated Stream):** A simulated pipeline will handle data inference, mimicking real-world streaming conditions.
- **Feature Consistency:** Ensuring the training and inference features remain identical by centralizing transformation logic and possibly storing processed features in a Feature Store.

### Justification & Relation to Lecture Material

- **Scalability & Collaboration:** Central persistent storage is critical for managing large datasets (~14GB raw) and facilitating team-wide access.
- **Reproducibility & Structure:** The ETL pipeline ensures structured, repeatable transformations from raw audio to analysis-ready features.
- **Realism & Testing:** Simulating online data streams allows comprehensive pipeline testing before encountering the hidden test set.
- **Consistency & Reliability:** Prevents training-serving skew by standardizing feature generation and transformation workflows.

### Specific Numbers & Implementation Details

#### **Persistent Storage:**

- **Requirement:** 100 GB on Chameleon.
- **Justification:**
  - Raw data (~14GB)
  - Processed features (Mel spectrograms, ~20-30GB)
  - Model artifacts (~10GB)
  - Container images and logs (~2GB)
  - Expansion room for experimentation (~44GB for reprocessed or extended datasets)
- **Mechanism:** Using Chameleon's persistent volume service (implementation dependent on Lab 8).

#### **Offline Data & ETL Pipeline:**

- **Repository Structure:**

  - `raw/`: Contains original `train_audio/`, `train_soundscapes/`, `train.csv`, `taxonomy.csv`
  - `processed/`: Stores Mel spectrogram features, train-validation split metadata
  - `models/`: Stores trained model checkpoints
  - `logs/`: Stores processing and inference logs

- **ETL Steps:** (Implemented via Python scripts using Pandas, Librosa, Scikit-learn):

  1. Extract raw data from Kaggle source to `raw/` directory..
  2. Transform:

     - Validate data integrity
     - Load and parse train.csv, taxonomy.csv.
     - Process audio files: load .ogg files, segment into 5-second chunks, compute Mel spectrograms using consistent parameters (e.g., n_mels, hop_length).
     - Generate labels for each segment based on train.csv, handling primary/secondary labels.
     - Create stratified train/validation splits.

  3. Load:
     - Store processed features and labels into `processed/train` and `processed/validation` directories
     - Maintain logs (sources, parameters, and output locations) using MLflow

#### **Online Data Simulation & Pipeline:**

- **Simulation Script:**

  - A script will generate simulated online data to test the inference pipeline, acting as the "producer".
    1. **Source**: It will primarily use files randomly selected from the train_soundscapes/ dataset.
    2. **Characteristics**:
       - Format: Generated files will be 1-minute OGG audio files, sampled at 32kHz, matching the expected test set format.
       - Content: Files will contain ambient sounds recorded in the El Silencio Natural Reserve. As `train_soundscapes` is unlabeled, the presence/absence of target species calls is unknown, realistically mimicking field recordings. This provides authentic background noise and acoustic channel characteristics.
       - Temporal Behavior: The script will write these files sequentially (e.g., one file every few seconds or minutes, configurable) into a dedicated `online_input/` directory on the persistent storage, simulating a stream or batch arrival of new data.
       - (Optional Enhancement): To create more challenging test cases with known calls, the script could occasionally overlay short calls (randomly selected from `train_audio/`) onto the `train_soundscapes` background audio before writing the file.

- **Online Inference Pipeline:** This pipeline acts as the "consumer".
  1. Monitors the `online_input/` directory for new `.ogg` files.
  2. On new file arrival: Loads the 1-minute `.ogg`.
  3. Applies the identical segmentation and Mel spectrogram transformation logic used in the offline ETL pipeline to ensure feature consistency.
  4. Sends the generated features (spectrograms) to the deployed model serving endpoint.
  5. Receives per-segment predictions (species probabilities).
  6. Formats the results according to requirements.
  7. Outputs formatted predictions to an `online_output/` directory (e.g., as a CSV file).

### Difficulty Points Attempted

- **Interactive Data Dashboard:** We will attempt to implement an interactive dashboard using Streamlit or Dash. This dashboard will read from the persistent storage (raw/ metadata, processed/ features) to provide visualizations of data distributions (species, quality ratings), sample waveforms/spectrograms, and potentially data quality metrics, aiding team insight.

## Persistent Storage Justification

To accommodate our dataset and processing requirements, we request **100 GB of persistent storage** on Chameleon. Our justification includes:

- **Raw dataset storage (~14 GB)**: The provided dataset includes train_audio and train_soundscapes in OGG format.
- **Processed features (~30-50 GB)**: Mel spectrograms and extracted features (depending on resolution and augmentation strategies.)
- **Model artifacts (~10-15 GB)**: Checkpoints and model weights from fine-tuned and scratch-trained architectures, and hyperparameter search logs require substantial space.
- **Container images & logs (~5-10 GB)**: Experiment tracking logs (MLflow, evaluation outputs), and online simulation files.
- **Expansion buffer**: Additional space ensures flexibility for future feature extraction methods, dataset augmentation, debugging, and computational overhead.

By implementing structured data management and efficient storage strategies, this allocation ensures smooth execution of both training and inference pipelines without bottlenecks.

#### Continuous X

Since the Lab for DevOps is yet to be released, the following points are based on the lecture notes and project requirements :

1. The aim to setup a pipleine where in any changes to the model made should be available to the end user automatically.
2. In Continous X, we will create scripts that would deploy services in the cloud for deployment, like Compute Instances, Security groups etc. Any resource that has to be provisioned, will be done using scripts, like Terraform or Python.
3. The end service and the model would exist in their separate microservices which would be containerized.
4. This pipeline will :
   4.1 Could be triggered automatically or manually.
   4.2 Will re-train the model.
   4.3 Run the complete offline evaluation suite.
   4.4 Apply the post-training optimizations for serving.
   4.5 Test its integration with the overall service.
   4.6 Package it inside a container for the deployment environment
   4.7 deploy it to a staging area for further testing
5. Once the artifacts are moved into the staging area, and once everything is working fine, we should be able to promote
   to higher environments.
6. Will learn more about the CI/CD once the Lab3 is released and can update this part again.


# Project Submission : 


# Continous X Pipeline

![ResourcesProvisioned.png](assets/ResourcesProvisioned.png)
## 1. Selecting Site

The Continous X Pipeline manages infrastructure mostly in KVM@TACC. So let's select the site!

from chi import server, context

context.version = "1.0"
context.choose_site(default="KVM@TACC")

Now that the Model Training, Model Evaluation, Model Serving and Data Pipeline are in place, we have to connect all the parts together. The main aim of this Continous_X_pipeline is to go from ideation to actual model deployment quickly and have an established process to iterate. This is indeed the Ops in the MLOps!

We will be provisioning resources and installing tools via code. For this we would be using :

-   Terraform: A declarative Infrastructure as Code (IaC) tool used to provision and manage cloud infrastructure (servers, networks, etc.) by defining the desired end state in configuration files. Here, we use it to provision our infrastructure.
-   Ansible: An imperative Configuration as Code (CaC) tool that automates system configuration, software installation, and application deployment through task-based YAML playbooks describing the steps to achieve a desired setup. Here, we use it to install Kubernetes and the Argo tools on our infrastructure after it is provisioned
-   Argo CD: A declarative GitOps continuous delivery tool for Kubernetes that automatically syncs and deploys applications based on the desired state stored in Git repositories.
-   Argo Workflows: A Kubernetes-native workflow engine where we define workflows, which execute tasks inside containers to run pipelines, jobs, or automation processes.

Let's get a copy of the Bird Classification Infrastructure repository

git clone --recurse-submodules https://github.com/exploring-curiosity/MLOps.git

## 2. Setup Environment


### Install and configure Terraform

Before we can use Terraform, we’ll need to download a Terraform client. The following cell will download the Terraform client and “install” it in this environment:

mkdir -p /work/.local/bin
wget https://releases.hashicorp.com/terraform/1.10.5/terraform_1.10.5_linux_amd64.zip
unzip -o -q terraform_1.10.5_linux_amd64.zip
mv terraform /work/.local/bin
rm terraform_1.10.5_linux_amd64.zip

The Terraform client has been installed to: /work/.local/bin. In order to run terraform commands, we will have to add this directory to our PATH, which tells the system where to look for executable files.

export PATH=/work/.local/bin:$PATH

Let’s make sure we can now run `terraform` commands. The following cell should print usage information for the `terraform` command, since we run it without any subcommands:

terraform

### Configure the PATH

Both Terraform and Ansible executables have been installed to a location that is not the system-wide location for executable files: `/work/.local/bin`. In order to run `terraform` or `ansible-playbook` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files.

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local

### Prepare Kubespray

To install Kubernetes, we’ll use Kubespray, which is a set of Ansible playbooks for deploying Kubernetes. We’ll also make sure we have its dependencies now:

PYTHONUSERBASE=/work/.local pip install --user -r MLOps/continous_X_pipeline/ansible/k8s/kubespray/requirements.txt

We will need to prepare credentials with which it can act on our behalf on the Chameleon OpenStack cloud. This is a one-time procedure.

To get credentials, open the Horizon GUI:

-   from the Chameleon website
-   click “Experiment” \> “KVM@TACC”
-   log in if prompted to do so
-   check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

On the left side, expand the “Identity” section and click on “Application Credentials”. Then, click “Create Application Credential”.

-   In the “Name”, field, use “mlops-lab”.
-   Set the “Expiration” date.
-   Click “Create Application Credential”.
-   Choose “Download clouds.yaml”.

Let's add the actual application_credential_id and application_credential_secret from the downloaded clouds.yaml in the clouds.yaml file that we see here.

Terraform will look for the `clouds.yaml` in either `~/.config/openstack` or the directory from which we run `terraform` - we will move it to the latter directory:

cp clouds.yaml /work/gourmetgram-iac/tf/kvm/clouds.yaml

## 3. Provision infrastructure with Terraform

Now that everything is set up, we are ready to provision our VM resources with Terraform! We will use Terraform to provision 3 VM instances and associated network resources on the OpenStack cloud.

### Preliminaries
Let’s navigate to the directory with the Terraform configuration for our KVM deployment:

cd /work/MLOps/continous_X_pipeline/tf/kvm/

and make sure we’ll be able to run the terraform executable by adding the directory in which it is located to our PATH:

export PATH=/work/.local/bin:$PATH

We also need to un-set some OpenStack-related environment variables that are set automatically in the Chameleon Jupyter environment, since these will override some Terraform settings that we *don’t* want to override:

unset $(set | grep -o "^OS_[A-Za-z0-9_]*")

### Understanding our Terraform configuration
[data.tf](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/tf/kvm/data.tf) :  data sources gets existing infrastructure details from OpenStack about resources *not* managed by Terraform.

[main.tf](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/tf/kvm/main.tf) :
Here we actually allocate the resources. Except for the block storage, which is previously allocated by the data team member and we have to attach it to node1. We are attaching it to node1 because in our kubernetes cluster, node1 will be the leader node. We are doing this because we want to persist data beyond the lifecycle of our VM instances. This persistent block store will have the storage for MinIO, PostgreSQL, etc.

[variables.tf](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/tf/kvm/variables.tf) : lets us define inputs and reuse the configuration across different environments. The value of variables can be passed in the command line arguments when we run a `terraform` command, or by defining environment variables that start with `TF_VAR`. In this example, there’s a variable `instance_hostname` so that we can re-use this configuration to create a VM with any hostname - the variable is used inside the resource block with `name = "${var.instance_hostname}"`.

### Applying our Terraform configuration
First, we need Terraform to set up our working directory, make sure it has “provider” plugins to interact with our infrastructure provider (it will read in `provider.tf` to check), and set up storage for keeping track of the infrastructure state:

terraform init

To follow the project naming conventions and adding the key. Not this is the name of the key followed by the entire team members.

export TF_VAR_suffix=project38\
export TF_VAR_key=id_rsa_chameleon_project_g38

We should confirm that our planned configuration is valid:

terraform validate

terraform apply -auto-approve

## 4. Ansible
Now that we have provisioned some infrastructure, we can configure and install software on it using Ansible!

### Preliminaries

As before, let’s make sure we’ll be able to use the Ansible executables. We need to put the install directory in the `PATH` inside each new Bash session.

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local

If you haven’t already, make sure to put your floating IP (which you can see in the output of the Terraform command!) in the `ansible.cfg` configuration file, and move it to the specified location.

The following cell will show the contents of this file, so you can double check - make sure your real floating IP is visible in this output!

cp ansible.cfg /work/MLOps/continous_X_pipeline/ansible/ansible.cfg

### Verify connectivity

First, we’ll run a simple task to check connectivity with all hosts listed in the [inventory.yaml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/inventory.yml)

    all:
      vars:
        ansible_python_interpreter: /usr/bin/python3
      hosts:
        node1:
          ansible_host: 192.168.1.11
          ansible_user: cc
        node2:
          ansible_host: 192.168.1.12
          ansible_user: cc
        node3:
          ansible_host: 192.168.1.13
          ansible_user: cc

It uses the `ping` module, which checks if Ansible can connect to each host via SSH and run Python code there.

But to be able to do that we would need to add the private key in .ssh folder.

cp /work/id_rsa_chameleon_project_g38 /work/.ssh/

cd /work/.ssh

chmod 600 id_rsa_chameleon_project_g38

ssh-add id_rsa_chameleon_project_g38

cd /work/MLOps/continous_X_pipeline/ansible/

ansible -i inventory.yml all -m ping

### Run a “Hello, World” playbook

ansible-playbook -i inventory.yml general/hello_host.yml

This was just a sanity check!

## 5. Deploy Kubernetes using Ansible

### Preliminaries
As before, let’s make sure we’ll be able to use the Ansible executables. We need to put the install directory in the `PATH` inside each new Bash session.

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local

### Run a preliminary playbook

Before we set up Kubernetes, we will run a preliminary playbook to:

-   disable the host firewall on the nodes in the cluster.  We will also configure each node to permit the local container registry.
-   and, configure Docker to use the local registry.

Let's add the SSH key again for this new session

cd /work/.ssh/

ssh-add id_rsa_chameleon_project_g38

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml

### Run the Kubespray play

Then, we can run the Kubespray playbook!

export ANSIBLE_CONFIG=/work/MLOps/continous_X_pipeline/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles

cd /work/MLOps/continous_X_pipeline/ansible/k8s/kubespray

ansible-playbook -i ../inventory/mycluster --become --become-user=root ./cluster.yml

### Access the ArgoCD
In the local termial we will have to run :\
ssh -L 8888:127.0.0.1:8888 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D

runs on node1 \
kubectl port-forward svc/argocd-server -n argocd 8888:443

https://127.0.0.1:8888/

Use the Username and Password from the above output

### Access the ArgoCD
In the local termial we will have to run :\
ssh -L 8888:127.0.0.1:8888 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D

runs on node1 \
kubectl port-forward svc/argocd-server -n argocd 8888:443

https://127.0.0.1:8888/

Use the Username and Password from the above output

## 6. ArgoCD to manage applications on the Kubernetes cluster

With our Kubernetes cluster up and running, we are ready to deploy applications on it!

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
export ANSIBLE_CONFIG=/work/MLOps/continous_X_pipeline/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles

First, we will deploy our birdclef “platform”. This has all the “accessory” services we need to support our machine learning application.

Let’s add the birdclef-platform application now. In the output of the following cell, look for the MinIO secret, which will be generated and then printed in the output:

cd /work/.ssh

ssh-add id_rsa_chameleon_project_g38

cd /work/MLOps/continous_X_pipeline/ansible/

ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml

Let's analyse the code for [argocd_add_platform](https://github.com/exploring-curiosity/MLOps/edit/main/continous_X_pipeline/ansible/argocd/argocd_add_platform.yml) :

Here we are executing the following tasks.

1. Creating Directory for the mount : Since we are using a persistent Block Storage which already exists, we will have to run two commands. One to build the directory and the other to mount it.

2. We next mount the directory

Note : As of this writing, this code has been commented out we do not yet have the persistent Block Storage for the services. This would also require us to modify the files in Platform Helm charts, but has been left out for now to ensure the complete integration works.

3. We next get the ArgoCD admin password from Kubernetes secret

4. We Decode ArgoCD admin password

5. We Log in to ArgoCD

6. Add repository to ArgoCD. This helps in syncing the platform for any changes in the Kubernetes Manifest files.

7. We ensure birdclef-platform namespace exists

8. We create birdclef-platform namespace if missing

9. Check if MinIO secret already exists, in case we are running this flow again

10. If we are running this flow for the first time, we generate MinIO secret key.

11. Fetching existing MinIO secret key if already exists.

12. Decoding existing MinIO secret key

13. Check if ArgoCD application exists

14. Create ArgoCD Helm application (like MinIO, MLFLow, PostgreSQL, Prometheus, LabelStudio etc) if it does not exist.

15. Update ArgoCD Helm application if it exists

16. Display MinIO credentials to login.

After running this flow for the first time, any changes made in Helm Application via git will directly be reflected in ArgoCD.

Once the platform is deployed, we can open:
(substitute A.B.C.D with floating IP) \
MinIO object Store :  http://A.B.C.D:9001 \
MLFlow             :  http://A.B.C.D:8000  \
Label-Studio : http://A.B.C.D:5000 \
Prometheus : http://A.B.C.D:4000 \
Grafana : http://A.B.C.D:3000

Next, we need to deploy the Bird Classification application. Before we do, we need to build a container image. We will run a one-time workflow in Argo Workflows to build the initial container images for the “staging”, “canary”, and “production” environments:

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/workflow_build_init.yml

Through this workflow : [workflow_build_init](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/argocd/workflow_build_init.yml)

we are calling the [build-initial.yaml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/workflows/build-initial.yaml) file which executes the following tasks :

Builds the initial container images for staging, canary, and production using the [FastAPI wrapper](https://github.com/harishbalajib/BirdClassification) for the model.

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/argocd_add_staging.yml

By executing the workflow [argocd_add_staging.yml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/argocd/argocd_add_staging.yml) we are primarily creating the birdclef-staging namespace which we can monitor in ArgoCD. And by using this worflow, we are executing [staging](https://github.com/exploring-curiosity/MLOps/tree/main/continous_X_pipeline/k8s/staging) manifest, where we actually create a container for the staging environment from the above staging image we created.

At the end of this workflow, our application should be up and running and available at http://A.B.C.D:8081 (where A.B.C.D is our public IP)

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/argocd_add_canary.yml

By executing the workflow [argocd_add_canary.yml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/argocd/argocd_add_canary.yml) we are primarily creating the birdclef-canary namespace which we can monitor in ArgoCD. And by using this worflow, we are executing [canary](https://github.com/exploring-curiosity/MLOps/tree/main/continous_X_pipeline/k8s/canary) manifest, where we actually create a container for the canary environment from the above canary image we created.

At the end of this workflow, our application should be up and running and available at http://A.B.C.D:8080 (where A.B.C.D is our public IP)

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/argocd_add_prod.yml

By executing the workflow argocd_add_prod.yml we are primarily creating the birdclef-production namespace which we can monitor in ArgoCD. And by using this worflow, we are executing production manifest, where we actually create a container for the staging environment from the above production image we created.

At the end of this workflow, our application should be up and running and available at http://A.B.C.D (where A.B.C.D is our public IP)

cd /work/MLOps/continous_X_pipeline/ansible

Now, we will manage our application lifecycle with Argo Worfklows. We will understand these workflow more in depth in the next sections.

ansible-playbook -i inventory.yml argocd/workflow_templates_apply.yml

## 7. Model and application lifecycle - Part 1

### Run a training and evaluation job

In this we will manually trigger a model training and evaluation. This manual trigger could be for several reasons like an update in the model or change in production data.

Through the previous workflow, we created a worflow template in Argo Workflow named [train-model](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/workflows/train-model.yaml), which is responsible for both training and evaluating the model.

Let's look at the code to understand the flow better

This template accepts 3 Public Addresses as shown here :
spec:
entrypoint: training-and-build
arguments:
parameters:
- name: train-ip
- name: eval-ip
- name: mlflow-ip

**train-ip** is the Public IP to trigger the Training Endpoint, which is responsible for Triggering the training process of the model, and logging the model artificats and status in ML Flow.

**eval-ip** is the Public IP address to trigger the Model Evaluation Endpoint. This endpoint is responsible for Evaluating the model and registering it in the MLFlow with a Specific Name.

**mlflow-ip** is the Public IP where the MLFlow is accessible. It will become more clear on why we are using the MLFlow here from the code below :

Through this workflow, a Training Endpoint is triggered with the help of **train-ip**. This is an API call. We get a RUN ID of the Model that is logged in the MLFlow. The following code achieves this :
```
RESPONSE=$(curl -f -s -X POST " http://{{inputs.parameters.train-ip}}:9090/train?model_name=resnet50&data_source=train")
        CURL_EXIT_CODE=$?
        echo "[INFO] Training endpoint response was: $RESPONSE" >&2
        if [ $CURL_EXIT_CODE -ne 0 ]; then
          echo "[ERROR] curl failed with code $CURL_EXIT_CODE" >&2
          exit $CURL_EXIT_CODE
        fi
        echo "[INFO] Training endpoint response was: $RESPONSE" >&2

```

Now, its possible that we Model Training could take several minutes if not hours togther to train a complex model on large datasets. And HTTP Endpoint calls have a timeout of just a few minutes.

Hence, the Model Training Endpoint immediately return the RUN_ID of the model which is logs in the MLFlow.

As Part of this workflow, we next keep polling the MLFlow to check if the process of training the model has completed. The following code achieves just this. We extract the RUN_ID and use it to poll the [MLFlow API](https://mlflow.org/docs/latest/api_reference/rest-api.html#get-run) to track the status.

```
 RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')    
        if [ -z "$RUN_ID" ]; then
          echo "[ERROR] run_id not found in response" >&2
          exit 1
        fi
        echo "[INFO] MLflow run ID: $RUN_ID" >&2
        
        #Polling MLFlow
        TERMINAL="FINISHED|FAILED|KILLED"
        while true; do
          STATUS=$(curl -s "http://{{inputs.parameters.mlflow-ip}}:8000/api/2.0/mlflow/runs/get?run_id=${RUN_ID}"| jq -r '.run.info.status')
          echo "[INFO] Run ${RUN_ID} status: ${STATUS}" >&2
          case "$STATUS" in
            FINISHED|FAILED|KILLED)
              echo "[INFO] Terminal state reached: $STATUS" >&2
              break
              ;;
          esac
          sleep 10
        done
```

Now that the model is ready, we next have to evaluate it and register it. Since this could also happen outside the current Kubernetes we have to trigger another Endpoint. At the end of this process we get the version of a registered model (named as 'BirdClassificationModel'). The following code demonstrates this:

```
EVAL_RESPONSE=$(curl -f -s -X GET "http://{{inputs.parameters.eval-ip}}:8080/get-version?run_id=${RUN_ID}")
        CURL_EXIT_CODE=$?
        echo "[INFO] Evaluation endpoint response was: EVAL_RESPONSE" >&2
        if [ $CURL_EXIT_CODE -ne 0 ]; then
          echo "[ERROR] curl failed with code $CURL_EXIT_CODE" >&2
          exit $CURL_EXIT_CODE
        fi
         
        # Extracting model version
        VERSION=$(echo "EVAL_RESPONSE" | jq -r '.new_model_version // empty')

        if [ -z "$VERSION" ]; then
          echo "[WARN] 'new_model_version' not found in response." >&2
          exit 1
        fi

        echo -n "$VERSION"
```

So, we triggered model train, model evaluation and at the end got the version of the registered model. With this the model's artifacts could be downloaded from MLFlow.

To actually trigger this template, we have to go to Argo Workflows > Workflow Templates > Submit > Add the train-ip, eval-ip and mlflow-ip > Hit Submit.

Now that we have a new registered model, we need a new container build!

This is triggered *automatically* when a new model version is returned from a training job.

One the build successful, and if we trigger http://A.B.C.D:8081 , we would be accessing the latest model which we just obtained.


This completes the critical flow of obtaining a model. So now we have our FastAPI wrapper for this model, replace the existing model with this new model (with the name bird.pth). The FastAPI just loads this model and now, this model is available to users!

Let's understand that flow better in the next section!

## 8. Model and application lifecycle - Part 2

Once we have a container image, the progression through the model/application lifecycle continues as the new version is promoted through different environments:

-   **Staging**: The container image is deployed in a staging environment that mimics the “production” service but without live users. In this staging environmenmt, we can perform integration tests against the service and also load tests to evaluate the inference performance of the system.
-   **Canary** (or blue/green, or other “preliminary” live environment): From the staging environment, the service can be promoted to a canary or other preliminary environment, where it gets requests from a small fraction of live users. In this environment, we are closely monitoring the service, its predictions, and the infrastructure for any signs of problems.
-   **Production**: Finally, after a thorough offline and online evaluation, we may promote the model to the live production environment, where it serves most users. We will continue monitoring the system for signs of degradation or poor performance.

### Promoting a Model

Now that we have tested our new model in staging, it time to promote it to canary. And from Canary to staging.

We can do this by following the below steps :

1. Workflow Templates > promote-model > Submit
2. In the source environment, mention "staging" and type "canary" in target-environment.
3. Select the desired model version in staging to be promoted. This could be obtained from MLFlow.
4. Hit Submit

After this [build-container-image](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/workflows/build-container-image.yaml) will be triggered automatically, which downloads the code for the model wrapper from git, downloads the staging model from MLFlow, bundles both of them together and makes it avialable in the canary environment.

So now if we go to http://A.B.C.D/8080 we would be accessing the latest model from the canary environment.

We can follow the same approach to promote the model from canary to production.


## 9. Delete infrastructure with Terraform

Since we provisioned our infrastructure with Terraform, we can also delete all the associated resources using Terraform.

cd /work/MLOps/continous_X_pipeline/tf/kvm

export PATH=/work/.local/bin:$PATH

unset $(set | grep -o "^OS_[A-Za-z0-9_]*")

export TF_VAR_suffix=project_38
export TF_VAR_key=id_rsa_chameleon_project_g38









