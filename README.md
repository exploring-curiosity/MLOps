# El Silencio Acoustic Explorer: An MLOps Pipeline for Real-time Bioacoustic Monitoring

## Value proposition

This project proposes a machine learning system, "El Silencio Acoustic Explorer," designed for integration into the existing services of Ecotour Operators in the El Silencio Natural Reserve.

- **Value Proposition:** The system provides an API endpoint that can be integrated into a mobile application (app development is outside the project scope). This allows tour guides and tourists to get real-time identifications of vocalizing fauna (birds, amphibians, mammals, insects) detected in audio recordings. This enhances the existing tour service by revealing hidden biodiversity, increasing customer engagement and education, empowering guides, and offering a unique selling proposition.
- **Non-ML Status Quo:** Currently, fauna identification relies solely on visual spotting and the variable acoustic identification skills of the human guide. Many species, particularly those primarily identified by sound, are missed, and identifications can be uncertain.
- **Project Success Metric (Proxy for Business Value):** As direct business metrics (e.g., customer satisfaction, booking rates) are not measurable within this project's scope, the primary success metric will be the **Mean Average Precision (mAP)** achieved by the species classification model. This will be evaluated on a diverse, curated test set representative of the soundscapes found in El Silencio (covering different species, noise levels, and times of day). High mAP directly reflects the system's core capability to accurately identify a wide range of species, which is fundamental to delivering the intended value proposition of revealing hidden biodiversity during tours.

## Contributors

| Name                      | Responsible for                          | Link to their commits in this repo |
| :------------------------ | :--------------------------------------- | :--------------------------------- |
| Sudharshan Ramesh         | _Model training and training platforms_  |                                    |
| Vaishnavi Deshmukh        | _Data pipeline_                          |                                    |
| Mohammad Hamid            | _Continuous X CI/CD_                     |                                    |
| Harish Balaji Boominathan | _Model serving and monitoring platforms_ |                                    |

### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces.
Must include: all the hardware, all the containers/software platforms, all the models,
all the data. -->

### Summary of outside materials

| Resource                                                       | How it was created                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Conditions of use                                                                                                                                                                                                                                      |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data set 1:** BirdCLEF 2025 (El Silencio Focus)              | Dataset provided via Kaggle for the BirdCLEF 2025 competition. Focuses on El Silencio Natural Reserve, Colombia. Includes: <br> - `train_audio/`: Short labeled recordings (primary/secondary labels for 206 species: birds, amphibians, mammals, insects) from Xeno-canto, iNaturalist, CSA. 32kHz OGG format. **(7.82 GB)**. Metadata in `train.csv`. <br> - `train_soundscapes/`: Unlabeled 1-minute soundscapes from El Silencio. 32kHz OGG format. **(4.62 GB)**. <br> **Total provided training/unlabeled data (`train_audio` + `train_soundscapes`) is approx. 12.5 GB.** | License: **CC BY-NC-SA 4.0**. For research purposes: Requires attribution (BY), prohibits commercial use (NC), and requires adaptations be shared under the same license (SA). Suitable for non-commercial academic research.                          |
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
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

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




## Data pipeline

## Table of Contents

1. [Data Pipeline](#data-pipeline)
   - [Strategy & Relevant Diagram Parts](#strategy--relevant-diagram-parts)
   - [Justification & Relation to Lecture Material](#justification--relation-to-lecture-material)
   - [Specific Numbers & Implementation Details](#specific-numbers--implementation-details)
   - [Difficulty Points Attempted](#difficulty-points-attempted)
2. [Persistent Storage Justification](#persistent-storage-justification)

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

