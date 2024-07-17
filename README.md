# PP-TEBDE: Privacy-Preserving Temporal Exposure Bias Detection and Explanation

PP-TEBDE is a novel method for detecting and mitigating temporal exposure bias in recommender systems while preserving user privacy and providing explainable insights.

## Features

- Temporal exposure bias detection using attention mechanisms and causal inference
- Privacy-preserving framework combining federated learning, homomorphic encryption, and differential privacy
- Explainable bias insights generation
- Adaptive debiasing for dynamic recommendation adjustment
- Application to educational recommender systems

## Installation

1. Clone the repository:
```
git clone https://github.com/nicolastorresr/PP-TEBDE.git
cd PP-TEBDE
```
2. Install dependencies:
```
pip install -r requirements.txt
```
## Usage

To run the PP-TEBDE method on the provided datasets:
```
python main.py --dataset eduRec2024 --mode train
```
For more options, run:
```
python main.py --help
```

## Docker

To run PP-TEBDE using Docker:

1. Build the Docker image:
```
docker build -t pp-tebde .
```
2. Run the container:
```
docker run -it --rm pp-tebde
```
Alternatively, use Docker Compose:
```
docker-compose up
```
## Experiments

To reproduce the experiments from the paper:

1. Download the datasets (EduRec-2024, MovieLens-Time, Amazon Electronics) and place them in the `data/` directory.
2. Run the experiment script:

```
python evaluation/experiment.py --all
```
