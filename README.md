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

This section outlines the plan to satisfy Unit 4 and Unit 5 requirements.

**Requirement 4.1: Train and Re-train At Least One Model**

Our strategy to meet this requirement centers on training a **ResNet-50** classifier fully from scratch using the provided `train_audio` data. Alongside this, we will fine-tune **EfficientNet-B0** and the **Google Bird Classifier**. The "re-train" component will be addressed via a simulation using a time-based split of the existing data: initial training will use the older \~70% (\~5.5 GB) identified via scraped Xeno-canto timestamps, while a subsequent re-training phase will use the newer \~30% (\~2.3 GB) to mimic adaptation to temporal data shifts. This entire process uses only the provided dataset. This approach is justified as it fulfills the requirements directly and models a realistic update scenario, contingent on the feasibility of timestamp scraping (requiring careful implementation and error handling). Conceptually, this involves the Training Pipeline shown in system diagrams, requiring versioned model weight storage. The plan draws on lecture material covering Training from Scratch, Transfer Learning, Continuous Training, and Data Drift simulation. Specifics include targeting \~75 epochs for initial ResNet-50 training and perhaps \~10-20 epochs for the re-training phase, all managed by a manually triggered Ray pipeline. A script for scraping and splitting data with error handling will be necessary.

**Requirement 4.2: Modeling Choices**

Appropriate modeling involves using a multi-model ensemble for classifying the 206 species, aiming for robustness and accuracy. The core strategy combines three diverse classifiers: the **ResNet-50** trained from scratch, a fine-tuned **EfficientNet-B0**, and the fine-tuned/feature-extracted **Google Bird Classifier**. To further aid classification, especially during inference, we plan to incorporate features from a pre-trained **PANNs** model (CNN14 variant) for Sound Event Detection (SED). Input audio (32kHz OGG) will be processed into standard Mel spectrograms (e.g., 128 bands). The system must handle multi-label classification (using primary/secondary labels) and produce outputs matching the target format (probabilities per 5-second interval for evaluation simulation, and handling short segments for the real-time API). This modeling strategy is justified because the ensemble leverages diverse architectures and pre-training approaches (scratch, ImageNet, bird audio) to maximize performance (target mAP > 0.5), directly supporting the project's value proposition. Using pre-trained PANNs is efficient, and spectrograms are standard for audio CNNs. This approach meets the "3+ models" and "composed models" criteria. Conceptually represented in the Inference Pipeline, this draws on lecture concepts like CNN Architectures, Transfer Learning, Training from Scratch, Ensemble Methods, Audio Processing, Multi-Label Classification, and Evaluation Metrics.

**Requirement 4.3 (Difficulty Point): Use Distributed Training to Increase Velocity**

To address the optional difficulty point and accelerate development, our strategy is to employ PyTorch's **Fully Sharded Data Parallel (FSDP)** for the ResNet-50 scratch training, which is expected to be the most computationally intensive part. We plan to run experiments comparing training time on 1, 2, and 4 GPUs (subject to Chameleon availability) within our Ray cluster. This involves running the training for a fixed number of epochs (\~50) while maintaining a consistent global batch size (e.g., 256, adjusted based on memory) and measuring wall-clock time. This is justified because faster training enables quicker iteration cycles. Using and evaluating FSDP demonstrates capability with advanced distributed techniques which can offer memory advantages over simpler data parallelism, providing valuable practical experience even if ResNet-50 isn't extremely large. This plan, utilizing the Training Pipeline on multiple GPU workers in the Ray cluster, directly relates to lecture material on Distributed Training Strategies (FSDP), Scalability, and Training Velocity. The output will be a plot comparing training time versus the number of GPUs used, potentially requiring tuning of FSDP parameters like wrapping policies.

**Requirement 5.1: Experiment Tracking**

Meeting the experiment tracking requirement involves deploying and managing our own **MLflow server** instance on a dedicated Chameleon VM. Our strategy is to instrument all training scripts (ResNet-50, EfficientNet-B0, G-BVC) with MLflow logging calls. This ensures that for every experiment run, we capture crucial information including hyperparameters (learning rate, batch size, etc.), metrics tracked over time (loss, validation mAP), environment details (code versions, library versions), and the resulting model artifacts. This systematic tracking, conceptually linked to the Training Pipeline, is justified because it is fundamental for reproducibility, allows for rigorous comparison between different models and approaches (e.g., evaluating SED impact, ensemble benefits, hyperparameter choices), and supports organized model development, aligning with standard MLOps practices as discussed in lectures (Experiment Tracking, Reproducibility, Model Management). We plan specific comparative experiments, all logged to our self-hosted MLflow server.

**Requirement 5.2: Scheduling Training Jobs (Ray Cluster)**

To satisfy the job scheduling requirement, our strategy is to use a **Ray cluster** set up on Chameleon infrastructure (e.g., 1 head node, 4 GPU worker nodes). All training tasks, including the ResNet-50 scratch training, fine-tuning scripts, the FSDP distributed training experiments, and the re-training pipeline execution, will be packaged as Python scripts and submitted as jobs to the Ray cluster head using Ray's job submission tools (CLI or SDK). Ray will then manage the scheduling and execution of these jobs on the cluster's worker nodes. This approach is justified as it directly meets the requirement and leverages Ray's capabilities for managing distributed tasks and resources efficiently on the Chameleon cluster, abstracting some underlying complexity. This relates to lecture material on Distributed Computing Frameworks (Ray), Job Scheduling, and Cluster Computing. The specifics involve configuring the Ray cluster and using its job submission interface to launch the training runs for their planned durations (e.g., \~75 epochs, \~30 epochs, 50 epochs for experiments).

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

