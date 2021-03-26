from pose_graph_tracking.data.augmenting_human36m_graph_in_memory_dataset_generator import \
    AugmentingHuman36MDataset, Human36MDataset

from pose_graph_tracking.helpers.defaults import PATH_TO_DATA_DIRECTORY, PATH_TO_CONFIG_DIRECTORY
from pose_graph_tracking.helpers.utils import get_model, make_deterministic_as_possible, load_config_file

from pose_graph_tracking.training.trainer import Trainer


path_to_training_hyperparameter_config = PATH_TO_CONFIG_DIRECTORY + "training_hyperparameter_config.json"
training_hyperparameter_config = load_config_file(path_to_training_hyperparameter_config)

make_deterministic_as_possible()

training_data = AugmentingHuman36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/augmenting_training_data",
                                          ids_of_subjects_to_load=[1, 6, 7, 8, 9, 11])

test_data = Human36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/test_data",
                            ids_of_subjects_to_load=[5])


path_to_model_config_file = PATH_TO_CONFIG_DIRECTORY + "model_config.json"
model_config = load_config_file(path_to_model_config_file)
model = get_model(model_config)

trainer = Trainer(model,
                  training_data,
                  test_data,
                  training_config=training_hyperparameter_config)
trainer.run_training_session()
