from pose_graph_tracking.data.human36m_graph_dataset_generator import Human36MDataset

from pose_graph_tracking.helpers.defaults import PATH_TO_DATA_DIRECTORY, PATH_TO_CONFIG_DIRECTORY
from pose_graph_tracking.helpers.utils import getModel, makeDeterministicAsPossible, load_config_file

from pose_graph_tracking.training.trainer import Trainer


path_to_training_hyperparameter_config = PATH_TO_CONFIG_DIRECTORY + "training_hyperparameter_config.json"
training_hyperparameter_config = load_config_file(path_to_training_hyperparameter_config)

makeDeterministicAsPossible()

training_data = Human36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/training_data",
                                ids_of_subjects_to_load=[1])

test_data = Human36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/test_data",
                            ids_of_subjects_to_load=[5])


path_to_model_config_file = PATH_TO_CONFIG_DIRECTORY + "model_config.json"
model_config = load_config_file(path_to_model_config_file)
model = getModel(model_config)

trainer = Trainer(model,
                  training_data,
                  test_data,
                  training_config=training_hyperparameter_config)
trainer.run_training_session()
