from pose_graph_prediction.data.human36m_graph_in_memory_dataset_generator import Human36MDataset

from pose_graph_prediction.helpers.defaults import PATH_TO_BEST_TEST_MODEL, PATH_TO_CONFIG_DIRECTORY, \
    PATH_TO_DATA_DIRECTORY
from pose_graph_prediction.helpers.utils import get_model, load_model_weights, load_config_file, \
    make_deterministic_as_possible

from pose_graph_prediction.visualization.sequential_prediction_visualizer import SequentialPredictionVisualizer


if __name__ == "__main__":
    make_deterministic_as_possible()

    path_to_model_config_file = PATH_TO_CONFIG_DIRECTORY + "model_config.json"
    model_config = load_config_file(path_to_model_config_file)
    model = get_model(model_config)
    load_model_weights(model, PATH_TO_BEST_TEST_MODEL)

    visualization_data = Human36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/visualization_data",
                                         ids_of_subjects_to_load=[11])

    visualizer = SequentialPredictionVisualizer()
    visualizer.visualize_model(model,
                               visualization_data,
                               use_output_as_next_input=False)
