from os.path import abspath

file_path = abspath(__file__)
package_name_index = file_path.rfind('pose_graph_tracking')
PACKAGE_ROOT_PATH = file_path[:package_name_index]

PATH_TO_DATA_DIRECTORY = PACKAGE_ROOT_PATH + 'data/'
