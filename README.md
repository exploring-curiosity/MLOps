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

Specifically:

Local El Silencio Natural Reserve **eco-tour operators**.
- [El Silencio Silvestre](https://elsilenciosilvestre.org/)
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

- **Mean Average Precision (mAP):** Target mAP > 0.5  
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
| **Base model 1:** Pre-trained SED Model: PANNs (CNN14 variant) | PANNs (Large-Scale Pretrained Audio Neural Networks) by Kong et al. (2020). Trained on AudioSet. CNN14 architecture. Paper: [https://arxiv.org/abs/1912.10211](https://arxiv.org/abs/1912.10211), Code/Weights: [https://github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)                                                                                                                                                                                                                                                     | License: **MIT License** (as per linked repository). Permits reuse, modification, distribution, and sublicensing for both private and commercial purposes, provided the original copyright and license notice are included. Suitable for research use. |
| **Base model 2:** EfficientNet-B0                              | Developed by Google Research (Tan & Le, 2019). Pre-trained on ImageNet. Smallest variant (\~5.3M parameters). Paper: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)                                                                                                                                                                                                                                                                                                                                                                                        | Apache 2.0 License. Pre-trained weights available. Will be **fine-tuned** on `train_audio` (Mel spectrograms) for the 206-species classification task.                                                                                                 |
| **Base model 3:** Google Perch (Bird Vocalization Classifier)  | Developed by Google Research. Pre-trained on bird sounds. Available on Kaggle Models under the name "Perch". [https://www.kaggle.com/models/google/bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier). Model details/parameters may vary by version. **Applicability to non-bird species requires careful evaluation.**                                                                                                                                                                                                            | **Apache 2.0 License** (as per Kaggle model details). Will be **fine-tuned** (if possible) or used for **feature extraction** on `train_audio` data. Performance impact on non-bird classes (mammals, amphibians, insects) must be assessed.           |
| **Architecture 4:** ResNet-50                                  | Architecture by Microsoft Research (He et al., 2015). (\~25.6M parameters). Paper: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)                                                                                                                                                                                                                                                                                                                                                                                                                          | Architecture well-established. Will be **trained from scratch** on `train_audio` (Mel spectrograms) for the 206-species classification task. No ImageNet pre-training used. Fulfills "train from scratch" requirement.                                 |
| **Tool:** Ray                                                  | Open-source framework for distributed computing. [https://www.ray.io/](https://www.ray.io/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Apache 2.0 License. Used for scheduling training jobs and distributed training on Chameleon.                                                                                                                                                           |
| **Tool:** MLflow                                               | Open-source platform for MLOps. [https://mlflow.org/](https://mlflow.org/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Apache 2.0 License. Will be self-hosted on Chameleon for experiment tracking.                                                                                                                                                                          |

### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`),
how much/when, justification. Include compute, floating IPs, persistent storage.
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
| --------------- | ------------------------------------------------- | ------------- |
| `m1.medium` VMs | 4 for entire project duration  <br> - 1 for Development/Coordination (Project duration). <br> - 1 for MLflow Server (Project duration). <br> - 1 for Ray Cluster Head Node (Active training periods). <br> +1 Optional later for API.                    | Essential for hosting persistent services (MLflow), managing training cluster (Ray Head), stable development environment, potential API serving. Assumes CPU-only sufficient.         |
| `compute_liquid`     | Concurrent access to up to 4 GPUs. 4-8 hour block twice a week                         | Required for intensive ResNet-50 scratch training, fine-tuning, and multi-GPU FSDP experiments (needs up to 4 GPUs concurrently). A100 GPUs provide necessary compute power. Usage is intermittent but demanding.              |
| Floating IPs    | 	2 required for project duration. <br> +1 Optional later. | Provides persistent external IPs for Development VM (SSH) and MLflow Server UI. Potential third IP for externally accessible API endpoint.             |
| Persistent Storage | Estimate: 100 GB needed throughout project duration. | Required for reliable storage of dataset (~12.5GB+), source code, MLflow artifacts (models, logs, checkpoints), environments. Ensures data persistence beyond ephemeral node storage and across project sessions. |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the
diagram, (3) justification for your strategy, (4) relate back to lecture material,
(5) include specific numbers. -->

#### Model training and training platforms

This section details the plan to satisfy Unit 4 and Unit 5 requirements, incorporating learnings from the course units.

**Requirement 4.1: Train and Re-train At Least One Model**

1.  **Strategy:** Train **ResNet-50** classifier from scratch using `train_audio`. Fine-tune **EfficientNet-B0** and **Google Perch**. Based on techniques explored in Unit 4 for handling memory limits with larger models, anticipate applying memory-saving strategies like **reduced precision (bf16)**, **gradient accumulation**, or potentially the **SGD optimizer** for the ResNet-50 training if Adam proves too memory-intensive. Address the "re-train" requirement via a simulation: use a time-split of `train_audio` (older \~70% initial train, newer \~30% re-train) based on scraped Xeno-canto timestamps, using only provided data to mimic adaptation to temporal drift. Implement this re-training process as a pipeline on Ray, manually triggered.
2.  **Relevant Diagram Parts:** Training Pipeline, Data Storage (time-split `train_audio`), versioned Model Weights storage.
3.  **Justification:** Meets train/re-train requirements using only provided data. Time-split simulates realistic adaptation scenario. Leverages memory-saving techniques demonstrated as effective in Unit 4 if required for the ResNet-50 scratch training. Contingent on successful timestamp scraping (requires implementation/error handling).
4.  **Relation to Lecture Material:** Memory Optimization Techniques (Reduced Precision, Gradient Accumulation, Optimizers like SGD vs Adam), Ray Framework (for pipeline execution).
5.  **Specifics:** Initial Data: \~70% `train_audio` (\~5.5 GB). Re-train Data: \~30% `train_audio` (\~2.3 GB). Split via scraped timestamps. Initial ResNet-50 epochs: \~75. Re-train epochs: \~10-20. Requires scraping script with error handling. Models versioned. Evaluate need for bf16/SGD/GradAcc during initial ResNet-50 training based on observed memory usage.

**Requirement 4.2: Modeling Choices**

1.  **Strategy:** Employ a multi-model ensemble for classifying 206 species: (a) **ResNet-50** (scratch-trained), (b) **EfficientNet-B0** (fine-tuned), (c) **Google Perch** (fine-tuned). Leverage the parameter efficiency of EfficientNet-B0 (\~5M params) and potentially Perch. For fine-tuning these, utilize **Parameter-Efficient Fine-Tuning (PEFT) via LoRA**, leveraging methods explored in Unit 4 that demonstrate significant memory savings for fine-tuning, potentially combined with **Quantization (e.g., nf4)** if further memory reduction is needed. Incorporate features from a pre-trained **PANNs (CNN14)** for SED during inference. Process 32kHz OGG audio into Mel spectrograms. Handle multi-label classification. Output probabilities per 5-second interval for evaluation simulation, and handle short segments for the API.
2.  **Relevant Diagram Parts:** Inference Pipeline (SED input, Spectrograms, 3 Classifiers, Ensemble Logic), Data Storage (audio, spectrograms).
3.  **Justification:** Ensemble targets robustness. Model diversity. PEFT/LoRA provides an efficient fine-tuning path, building on Unit 4 explorations. Pre-trained SED leverages existing work. Meets "3+ models" and "composed models" criteria. Addresses data specifics and output requirements.
4.  **Relation to Lecture Material:** Parameter-Efficient Fine-Tuning (PEFT/LoRA), Quantization (nf4).
5.  **Specifics:** Classifiers: ResNet-50 (\~25.6M params, scratch), EfficientNet-B0 (\~5M params, LoRA fine-tune), Google Perch (LoRA fine-tune). SED: PANNs (CNN14). Input: 32kHz audio -> Mel spectrograms (128 bands). Output: 206 species probabilities. Ensemble: Weighted averaging. Metric: mAP > 0.5 target. Consider nf4 quantization during PEFT if memory is constrained.

**Requirement 4.3 (Difficulty Point): Use Distributed Training to Increase Velocity**

1.  **Strategy:** Use **PyTorch FSDP (Fully Sharded Data Parallel)** for the ResNet-50 scratch training, targeting increased velocity and leveraging FSDP's memory efficiency potential, as explored in Unit 4's comparison of distributed strategies (DDP vs FSDP). Execute experiments on the Ray cluster across 1, 2, and 4 GPUs (subject to availability). Measure training time for fixed epochs (\~50) with a consistent global batch size.
2.  **Relevant Diagram Parts:** Training Pipeline on Ray Cluster (utilizing multiple GPU workers).
3.  **Justification:** Addresses potential ResNet-50 memory bottlenecks and accelerates training. Directly leverages insights from Unit 4 regarding FSDP's memory reduction benefits compared to DDP, especially for larger models or constrained memory situations. Fulfills difficulty point by experimentally comparing scaling performance using an advanced distributed technique.
4.  **Relation to Lecture Material:** Distributed Training Strategies (DDP, FSDP), Scaling (Time/Memory vs #GPUs).
5.  **Specifics:** Target: ResNet-50 training. Experiment: 1 vs 2 vs 4 GPUs for 50 epochs. Strategy: PyTorch FSDP. Global Batch Size: 256 (example, tune based on memory). Metric: Wall-clock time. Output: Plot time vs. #GPUs. Requires FSDP configuration (wrapping, etc.). Compare time/memory scaling results against Unit 4 observations.

**Requirement 5.1: Experiment Tracking**

1.  **Strategy:** Deploy and manage a **self-hosted MLflow server** on Chameleon. Instrument all training scripts to log parameters, system metrics (GPU util/memory), model metrics (loss, mAP), git commit hashes, and artifacts including model checkpoints and dependencies, utilizing MLflow's comprehensive logging capabilities demonstrated in Unit 5. Leverage **MLflow's UI for visualization** (e.g., plotting GPU utilization against epoch time to identify bottlenecks) and **comparing runs**. Register final candidate models using the **MLflow Model Registry**. Consider Pytorch Lightning autologging if applicable, based on Unit 5 explorations.
2.  **Relevant Diagram Parts:** Experiment Tracking Server (MLflow) connected to Training Pipeline. Artifact storage linked.
3.  **Justification:** Meets self-hosted tracking requirement. Enables reproducibility, systematic comparison, and performance debugging (e.g., GPU bottlenecks), leveraging the extensive tracking and UI features of MLflow explored in Unit 5. Essential for organized MLOps.
4.  **Relation to Lecture Material:** MLflow (Source version logging, Parameter/Metric/Artifact Logging, System Metrics Logging, Visualizations, Model Registry, Autologging, Comparing Runs).
5.  **Specifics:** 1 self-hosted MLflow instance. Log key parameters/metrics. Utilize visualization and comparison views. Use model registry. Planned experiments: hyperparameter sweeps, model comparisons (ResNet/EffNet/Perch, LoRA vs full), SED impact, ensemble performance.

**Requirement 5.2: Scheduling Training Jobs (Ray Cluster)**

1.  **Strategy:** Use a **Ray cluster** on Chameleon for submitting all training/re-training jobs via Ray Jobs API/CLI. Specify **resource requests (CPU, GPU)** per job, informed by Unit 4 memory findings and Unit 5 explorations of resource allocation and scheduling outcomes. Monitor jobs and cluster state via the **Ray dashboard**. Utilize **Ray Train checkpointing for fault tolerance**, leveraging recovery capabilities explored in Unit 5 to avoid costly restarts upon worker failure.
2.  **Relevant Diagram Parts:** Ray Cluster (Head, Workers) on Chameleon Infrastructure, executing Training Pipeline jobs.
3.  **Justification:** Fulfills requirement to use Ray. Leverages Ray's demonstrated capabilities (Unit 5) for scheduling, resource management (including handling infeasible requests, fractional GPUs, simultaneous jobs), monitoring, and fault tolerance, suitable for managing the project's training workload.
4.  **Relation to Lecture Material:** Ray Framework (Ray Train, Ray Dashboard, Ray Jobs, Resource Management, Scheduling, Fault Tolerance).
5.  **Specifics:** 1 Ray cluster (e.g., 1 head, 4 GPU workers) on Chameleon. Use Ray Jobs to submit Python scripts. Specify resource requests (CPU/GPU counts). Utilize Ray Train checkpointing. Monitor via Ray dashboard.

#### Model serving and monitoring platforms

This section outlines the plan to satisfy Unit 6 and Unit 7 requirements.

#### Unit 6: Model Serving

#### Requirement 6.1: Serving from an API Endpoint

Our strategy involves exposing the final ensemble model (ResNet-50, EfficientNet-B0, and Google Bird Classifier with PANNs features) through a FastAPI endpoint. This API will be deployed on a cloud-based GPU server using a containerized microservice architecture. We support both PyTorch and ONNX runtime endpoints, switching to ONNX for performance-critical deployments. The API accepts base64-encoded `.ogg` audio clips (or extracted Mel spectrograms) and returns a multi-label prediction of species along with confidence scores. This aligns directly with the real-time use case for tour guides and tourists in El Silencio.

#### Requirement 6.2: Identify Requirements (Latency, Throughput, Concurrency)

Our target requirements are informed by the mobile tour guide use case:

- **Model Size**: ≤ 25MB (with quantized ONNX models).
- **Latency (Online Inference)**: < 200ms median latency on GPU-based cloud inference, including preprocessing of a 5-second `.ogg` clip.
- **Throughput (Batch Inference)**: ≥ 200 frames/sec on CPU, ≥ 1000 frames/sec on GPU.
- **Concurrency**: Support at least 8 concurrent users per node with latency < 300ms (95th percentile).

#### Requirement 6.3: Model Optimizations to Satisfy Requirements

We apply several model-level optimizations:

- **Graph Optimizations**: Using ONNX Runtime’s extended optimization level.
- **Quantization**: Applying both dynamic and static post-training quantization with Intel Neural Compressor.
- **Model Compilation**: TorchScript for PyTorch models and optimized ONNX sessions.
- **Hardware-Specific Execution Providers**: CUDA (GPU), OpenVINO (CPU), and TensorRT (optimized GPU).

Each variant is benchmarked on Chameleon cloud nodes. Performance metrics (latency, throughput, model size) are logged to MLflow.

#### Requirement 6.4: System Optimizations to Satisfy Requirements

To maintain low latency under concurrent access, we apply system-level strategies using Triton Inference Server:

- **Dynamic Batching**: Batch sizes 4–16.
- **Model Replica Management**: Multiple instances per GPU node.
- **Resource Monitoring**: `nvidia-smi` and Prometheus-Grafana dashboards.
- **Prioritized Queuing**: Queue policies to minimize worst-case response times.

#### Extra Difficulty Point: Multiple Options for Serving

To explore cost-performance tradeoffs, we evaluate three deployment configurations:

- **Server-grade CPU (AMD EPYC)** with ONNX + OpenVINO backend (target latency < 500ms per clip).
- **Server-grade GPU (A100)** with ONNX + TensorRT backend (target latency < 200ms per clip).
- **On-device deployment** with quantized MobileNetV2 (planned for post-project exploration).

We compare accuracy, latency, throughput, and projected cost across these setups to inform production scaling decisions.

---

#### Unit 7: Evaluation and Monitoring

#### Requirement 7.1: Offline Evaluation of Model

After each training run, we execute an evaluation pipeline that includes:

1. **Standard Evaluation**: On labeled El Silencio `.ogg` soundscapes.
2. **Domain-Specific Slices**: Nighttime recordings, low signal-to-noise regions, insect-dominant samples.
3. **Failure Mode Tests**: Overlapping vocalizations, weak signals, background interference.
4. **Unit Tests**: Checks for top-K accuracy, coverage, and label frequency.

If a model passes the quality bar (e.g., mAP > 0.5), it is registered in MLflow. Failures trigger retraining.

#### Requirement 7.2: Load Test in Staging

The FastAPI + Triton stack is deployed to a staging node. We simulate 8 to 32 users using Python benchmarks and Triton’s `perf_analyzer` to record:

- Median and 95th percentile latency
- Throughput in predictions/sec
- GPU and CPU usage trends

This informs autoscaling and concurrency planning.

#### Requirement 7.3: Online Evaluation in Canary

We simulate different user patterns:

- **Tour Guide Mode**: Streams 5-second `.ogg` segments.
- **Power User Mode**: Uploads 1-minute `.ogg` recordings.
- **Mobile User Mode**: Random upload delays and interruptions.

Manual inspection of predictions evaluates responsiveness, stability, and correctness. Passing the canary test allows promotion to production.

#### Requirement 7.4: Close the Loop

We implement real-world feedback mechanisms:

- **User Feedback Hook**: App integration will allow guides to flag wrong predictions.
- **Passive Feedback Logging**: 5% of queries (and their audio) are saved and periodically labeled to support retraining.

#### Requirement 7.5: Business-Specific Evaluation

Our proxy business metric is:

- **Species Diversity per Tour Session**: Number of unique species detected in a single outing.

Logged per tour, this serves as a proxy for user engagement and tour value.

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

export TF_VAR_suffix=project38
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









