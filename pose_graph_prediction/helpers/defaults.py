from os.path import abspath

file_path = abspath(__file__)
package_name_index = file_path.rfind('pose_graph_prediction')
PACKAGE_ROOT_PATH = file_path[:package_name_index]

PATH_TO_DATA_DIRECTORY = PACKAGE_ROOT_PATH + 'data/'

PATH_TO_CONFIG_DIRECTORY = PACKAGE_ROOT_PATH + 'config/'

MODEL_DIRECTORY = PACKAGE_ROOT_PATH + "models/"
MODEL_NAME_PREFIX = "pose_graph_prediction_net_"
MODEL_NUMBER = 1107
PATH_TO_MODEL = MODEL_DIRECTORY + MODEL_NAME_PREFIX + str(MODEL_NUMBER) + ".model"
PATH_TO_BEST_TEST_MODEL = MODEL_DIRECTORY + MODEL_NAME_PREFIX + "best_on_test_data.model"
