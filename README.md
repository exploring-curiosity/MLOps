# El Silencio Acoustic Explorer: An MLOps Pipeline for Real-time Bioacoustic Monitoring

## Value proposition

This project proposes a machine learning system, "El Silencio Acoustic Explorer," designed for integration into the existing services of Ecotour Operators in the El Silencio Natural Reserve.

* **Value Proposition:** The system provides an API endpoint that can be integrated into a mobile application (app development is outside the project scope). This allows tour guides and tourists to get real-time identifications of vocalizing fauna (birds, amphibians, mammals, insects) detected in audio recordings. This enhances the existing tour service by revealing hidden biodiversity, increasing customer engagement and education, empowering guides, and offering a unique selling proposition.
* **Non-ML Status Quo:** Currently, fauna identification relies solely on visual spotting and the variable acoustic identification skills of the human guide. Many species, particularly those primarily identified by sound, are missed, and identifications can be uncertain.
* **Project Success Metric (Proxy for Business Value):** As direct business metrics (e.g., customer satisfaction, booking rates) are not measurable within this project's scope, the primary success metric will be the **Mean Average Precision (mAP)** achieved by the species classification model. This will be evaluated on a diverse, curated test set representative of the soundscapes found in El Silencio (covering different species, noise levels, and times of day). High mAP directly reflects the system's core capability to accurately identify a wide range of species, which is fundamental to delivering the intended value proposition of revealing hidden biodiversity during tours.

## Contributors

| Name                      | Responsible for                         | Link to their commits in this repo |
| :------------------------ | :-------------------------------------- | :--------------------------------- |
| Sudharshan Ramesh         | *Model training and training platforms* |                                    |
| Vaishnavi Deshmukh        | *Data pipeline*                         |                                    |
| Mohammad Hamid            | *Continuous X CI/CD*                    |                                    |
| Harish Balaji Boominathan | *Model serving and monitoring platforms*|                                    |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

| Resource                     | How it was created                                                                                                                                                                                                                                                                                                                                                                                    | Conditions of use                                                                                                                                                                                                                                                                                                                                                                  |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data set 1:** BirdCLEF 2025 (El Silencio Focus) | Dataset provided via Kaggle for the BirdCLEF 2025 competition. Focuses on El Silencio Natural Reserve, Colombia. Includes: <br> - `train_audio/`: Short labeled recordings (primary/secondary labels for 206 species: birds, amphibians, mammals, insects) from Xeno-canto, iNaturalist, CSA. 32kHz OGG format. **(7.82 GB)**. Metadata in `train.csv`. <br> - `train_soundscapes/`: Unlabeled 1-minute soundscapes from El Silencio. 32kHz OGG format. **(4.62 GB)**. <br> **Total provided training/unlabeled data (`train_audio` + `train_soundscapes`) is approx. 12.5 GB.** | License: **CC BY-NC-SA 4.0**. For research purposes: Requires attribution (BY), prohibits commercial use (NC), and requires adaptations be shared under the same license (SA). Suitable for non-commercial academic research. |
| **Base model 1:** Pre-trained SED Model: PANNs (CNN14 variant) | PANNs (Large-Scale Pretrained Audio Neural Networks) by Kong et al. (2020). Trained on AudioSet. CNN14 architecture. Paper: [https://arxiv.org/abs/1912.10211](https://arxiv.org/abs/1912.10211), Code/Weights: [https://github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)                                               | License: **MIT License** (as per linked repository). Permits reuse, modification, distribution, and sublicensing for both private and commercial purposes, provided the original copyright and license notice are included. Suitable for research use.                                                                                                                       |
| **Base model 2:** EfficientNet-B0 | Developed by Google Research (Tan & Le, 2019). Pre-trained on ImageNet. Smallest variant (\~5.3M parameters). Paper: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)                                                                                                                                                                                                          | Apache 2.0 License. Pre-trained weights available. Will be **fine-tuned** on `train_audio` (Mel spectrograms) for the 206-species classification task.                                                                                                                                                                                                            |
| **Base model 3:** Google Perch (Bird Vocalization Classifier) | Developed by Google Research. Pre-trained on bird sounds. Available on Kaggle Models under the name "Perch". [https://www.kaggle.com/models/google/bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier). Model details/parameters may vary by version. **Applicability to non-bird species requires careful evaluation.** | **Apache 2.0 License** (as per Kaggle model details). Will be **fine-tuned** (if possible) or used for **feature extraction** on `train_audio` data. Performance impact on non-bird classes (mammals, amphibians, insects) must be assessed.                                                                                                                      |
| **Architecture 4:** ResNet-50 | Architecture by Microsoft Research (He et al., 2015). (\~25.6M parameters). Paper: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)                                                                                                                                                                                                                                              | Architecture well-established. Will be **trained from scratch** on `train_audio` (Mel spectrograms) for the 206-species classification task. No ImageNet pre-training used. Fulfills "train from scratch" requirement.                                                                                                                                            |
| **Tool:** Ray                  | Open-source framework for distributed computing. [https://www.ray.io/](https://www.ray.io/)                                                                                                                                                                                                              | Apache 2.0 License. Used for scheduling training jobs and distributed training on Chameleon.                                                                                                                                                     |
| **Tool:** MLflow               | Open-source platform for MLOps. [https://mlflow.org/](https://mlflow.org/)                                                                                                                                                                                                                              | Apache 2.0 License. Will be self-hosted on Chameleon for experiment tracking.                                                                                                                                                                     |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
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

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

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