# BirdCLEF Multimodal Audio Classification: Deployment & Evaluation

This project implements, optimizes, and evaluates a multimodal audio classification pipeline for the BirdCLEF dataset. The system combines multiple pre-trained branches to classify 10-second audio clips into one of 206 bird species.

## Model Serving and Evaluation
##### Model Serving : https://github.com/exploring-curiosity/MLOps/tree/main/model_serving
### Objective

To build an end-to-end model deployment pipeline that satisfies real-world constraints on model size, latency, and throughput. The project targets both high-performance server deployments and resource-constrained edge devices such as the Raspberry Pi. The system is designed to identify bird species based on their sounds, enabling bioacoustic monitoring and field-based species detection.



### Model Architecture

The model is a five-branch multimodal fusion pipeline:

- EmbeddingClassifier: operates on precomputed audio embeddings  
- ResNet50: processes mel-augmented spectrograms  
- EfficientNetB3 with LoRA: processes clean mel spectrograms  
- RawAudioCNN: operates directly on 10-second denoised waveforms  
- MetaMLP: fuses the outputs of the above branches and performs final classification  

## Implementation Workflow

The focus of this project was exclusively on model serving and evaluation. Below is a detailed overview of the steps taken to deploy the BirdCLEF model and assess its real-world performance across multiple scenarios.

### Model Serving


All five PyTorch model branches were converted to ONNX format using carefully designed dummy inputs and compatible opset versions. To ensure the models would be suitable for deployment in both cloud and edge environments, dynamic post-training quantization was applied to each ONNX model to reduce size and improve runtime efficiency. Batch processing configurations were also implemented and tested to maximize throughput and parallelism. Following quantization, the system was evaluated on high-performance hardware for maximum throughput. Using ONNX Runtime on an NVIDIA RTX 6000 GPU, the full fusion pipeline achieved exceptional inference performance with the following results:

- Model size on disk (total): 197.51 MB  
- Inference latency (single sample): 23.64 ms (median), 26.17 ms (95th percentile), 29.11 ms (99th percentile)  
- Inference throughput (single sample): 42.70 FPS  
- Batch throughput: 13,093.65 FPS  

These results demonstrate that the model scales efficiently across batched workloads and performs well under production-like conditions.

##### Model Serving Notebook : https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/Notebooks/Model_serving.ipynb

##### Model Quantization Notebook : https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/Notebooks/ONNX_Optimization.ipynb



### Edge Deployment on Raspberry Pi

Edge deployment was carried out using ONNX Runtime and quantized models optimized for low-latency inference. In this setup, the Raspberry Pi 5 was used to capture live audio from a microphone and wait until the user triggered inference. The recorded audio was then segmented into 10-second chunks to match the model’s expected input length. Each chunk was passed through the preprocessing pipeline and then run through the quantized ONNX model branches locally.

This setup was designed for efficiency: audio was continuously captured but not processed until the user initiated classification. Once activated, the system performed chunk-wise inference, aggregated predictions, and returned both the predicted class and confidence score per chunk. The latency per chunk remained under 200 ms on-device, making it suitable for responsive field-based bioacoustic classification.

While the default pipeline processes live audio, we also supported inference on pre-saved audio clips during development; by commenting out a few lines, the same system can be toggled between live microphone input and static audio file inference for flexibility during testing and deployment.


##### Edge Serving Notebook: https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/Edge_Serving/edge_serving.ipynb

### System Optimization

To further improve system throughput and latency, an API endpoint was built using FastAPI to streamline interaction between the model and the inference engine. This API allowed flexible integration with both web-based and local front-end interfaces for inference triggering. Additionally, `config.pbtxt` files were developed for each model branch to define input/output tensor metadata, configure dynamic batching, and control concurrency. Each model instance was pinned to specific GPU devices to balance the load and optimize execution throughput. Concurrency and batching were core elements of the optimization strategy. Batched inference was configured to fully utilize GPU compute capacity by processing multiple audio inputs simultaneously. This was especially beneficial during large-scale evaluation, where multiple chunks or test samples were queued and dispatched in batches, significantly improving throughput.

All components were integrated and validated as part of a cohesive serving pipeline, demonstrating optimal usage of available computational resources.

#####  App.py  : https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/BirdClassification/fastapi/app.py

### Model Evaluation

Evaluation was conducted using a combination of offline accuracy testing and domain-specific robustness checks. The primary evaluation used `manifest_test.csv`, which contains the ground truth for standard BirdCLEF test clips. For each audio clip, preprocessed inputs (embedding, mel spectrogram, augmented mel, waveform) were passed to their respective model branches. Outputs were fused via the MetaMLP model to generate final predictions.

The evaluation strategy included overall accuracy, per-class accuracy, and identifying the top 20 least accurate classes. Additionally, robustness tests were run on custom folders: `insects/`, `mammalia/`, `amphibia/`, and `sound_similar_to_amphibia/`. These helped identify failure modes and understand model behavior under confusing conditions.

Visualizations were generated for per-class prediction distributions, confusion matrices, and SHAP-based saliency maps. SHAP was used to analyze which portions of the waveform contributed most to the prediction. Finally, Pytest-based unit tests were written to validate model predictions, pipeline correctness, and I/O integrity under batch inference conditions.

The serving and evaluation pipeline demonstrated high modularity, low latency on GPU, and acceptable performance on edge hardware. The system can be easily extended for real-time streaming or integrated with monitoring frameworks for automated evaluation and deployment.

#### Offline Evaluation

Offline evaluation was performed after the model was served, using the `manifest_test.csv` test set to measure accuracy and per-class performance. This allowed the identification of the 20 lowest-performing classes, which provided insight into which species were most frequently misclassified.

Building on this, template-based tests were developed using folders containing species or class-specific samples (e.g., insects, amphibians). These helped evaluate how the model performed on tightly scoped or confusing categories, especially under noisy or overlapping audio conditions.

In addition, explainability techniques such as SHAP were used to interpret what aspects of the audio signal were most influential to the model’s predictions. The SHAP values highlighted important temporal segments and allowed deeper inspection of attention toward certain species. Class-level attribution was derived from this data to help analyze why specific classes were more prone to failure and how certain frequency bands or waveform segments drove model confidence.

##### Offline_eval Notebook : https://github.com/exploring-curiosity/MLOps/blob/main/model_serving/Offline_Evaluation/Offline_eval.ipynb

### Evaluation Strategy

- Evaluated the model using the `manifest_test.csv` test set  
- Measured overall accuracy, per-class accuracy, and identified the 20 lowest-performing classes  
- Conducted robustness tests using domain-specific folders: `insects`, `mammalia`, `amphibia`, and `sound_similar_to_amphibia`  

### Analysis Tools

- Applied SHAP to interpret model sensitivity to different portions of the audio waveform  
- Generated prediction distribution plots, saliency maps, and confusion matrices  


### Model Serving and Optimization

- Model served through API endpoint using Triton and FastAPI  
- Latency and throughput benchmarks collected for batch and online inference  
- Applied model-level optimizations including quantization and opset tuning  
- Performed system-level optimizations such as GPU pinning and concurrency tuning  


