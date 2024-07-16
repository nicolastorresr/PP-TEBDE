import yaml
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    min_interactions: int
    max_sequence_length: int

@dataclass
class ModelConfig:
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float

@dataclass
class AttentionConfig:
    num_heads: int
    key_dim: int

@dataclass
class PrivacyConfig:
    epsilon: float
    delta: float
    num_clients: int
    local_epochs: int

@dataclass
class BiasConfig:
    detection_threshold: float
    mitigation_strength: float

@dataclass
class ExplainableConfig:
    num_counterfactuals: int
    max_features: int

@dataclass
class EvaluationConfig:
    metrics: List[str]
    k_values: List[int]

@dataclass
class LoggingConfig:
    level: str
    file_path: str

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    attention: AttentionConfig
    privacy: PrivacyConfig
    bias: BiasConfig
    explainable: ExplainableConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    learning_rate: float
    batch_size: int
    num_epochs: int
    early_stopping_patience: int
    seed: int

def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file and return a Config object.
    """
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return Config(
        data=DataConfig(**config_dict['data']),
        model=ModelConfig(**config_dict['model']),
        attention=AttentionConfig(**config_dict['attention']),
        privacy=PrivacyConfig(**config_dict['privacy']),
        bias=BiasConfig(**config_dict['bias']),
        explainable=ExplainableConfig(**config_dict['explainable']),
        evaluation=EvaluationConfig(**config_dict['evaluation']),
        logging=LoggingConfig(**config_dict['logging']),
        learning_rate=config_dict['learning_rate'],
        batch_size=config_dict['batch_size'],
        num_epochs=config_dict['num_epochs'],
        early_stopping_patience=config_dict['early_stopping_patience'],
        seed=config_dict['seed']
    )

def save_config(config: Config, config_path: str):
    """
    Save the configuration to a YAML file.
    """
    config_dict = {
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'attention': config.attention.__dict__,
        'privacy': config.privacy.__dict__,
        'bias': config.bias.__dict__,
        'explainable': config.explainable.__dict__,
        'evaluation': config.evaluation.__dict__,
        'logging': config.logging.__dict__,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'early_stopping_patience': config.early_stopping_patience,
        'seed': config.seed
    }

    with open(config_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)

def get_config() -> Config:
    """
    Get the default configuration.
    """
    return Config(
        data=DataConfig(
            train_path='data/train.csv',
            val_path='data/val.csv',
            test_path='data/test.csv',
            min_interactions=5,
            max_sequence_length=50
        ),
        model=ModelConfig(
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ),
        attention=AttentionConfig(
            num_heads=4,
            key_dim=32
        ),
        privacy=PrivacyConfig(
            epsilon=0.1,
            delta=1e-5,
            num_clients=10,
            local_epochs=5
        ),
        bias=BiasConfig(
            detection_threshold=0.1,
            mitigation_strength=0.5
        ),
        explainable=ExplainableConfig(
            num_counterfactuals=5,
            max_features=10
        ),
        evaluation=EvaluationConfig(
            metrics=['ndcg', 'map', 'gini'],
            k_values=[5, 10, 20]
        ),
        logging=LoggingConfig(
            level='INFO',
            file_path='logs/pp_tebde.log'
        ),
        learning_rate=0.001,
        batch_size=64,
        num_epochs=100,
        early_stopping_patience=10,
        seed=42
    )
