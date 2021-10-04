from json import dump as save_json_file

from numpy import mean

from pose_graph_tracking.data.augmenting_human36m_graph_in_memory_dataset_generator import \
    AugmentingHuman36MDataset, Human36MDataset

from pose_graph_tracking.helpers.defaults import PATH_TO_DATA_DIRECTORY, PATH_TO_CONFIG_DIRECTORY, PATH_TO_MODEL, MODEL_DIRECTORY, MODEL_NAME_PREFIX
from pose_graph_tracking.helpers.utils import get_model, load_model_weights, make_deterministic_as_possible, load_config_file

from pose_graph_tracking.training.trainer import Trainer

from pose_graph_tracking.evaluation.validator import Validator

from shutil import rmtree


def remove_data(path_to_data_directory: str):
    try:
        rmtree(path_to_data_directory)
    except OSError as e:
        print("Error: %s : %s" % (path_to_data_directory, e.strerror))


path_to_training_hyperparameter_config = PATH_TO_CONFIG_DIRECTORY + "training_hyperparameter_config.json"
training_hyperparameter_config = load_config_file(path_to_training_hyperparameter_config)

ids_of_subjects = [1, 5, 6, 7, 8, 9, 11]

mean_validation_loss_per_subject = []
for validation_subject_id in ids_of_subjects:
    sum_of_mean_losses_for_current_validation_subject = 0.0
    validations_counter = 0

    validation_data = Human36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/validation_data",
                                      ids_of_subjects_to_load=[validation_subject_id])

    for test_subject_id in ids_of_subjects:
        if validation_subject_id == test_subject_id:
            continue

        # use remaining subjects as training data
        training_subject_ids = []
        for subject_id in ids_of_subjects:
            if subject_id != validation_subject_id and subject_id != test_subject_id:
                training_subject_ids.append(subject_id)

        # make sure the data is the same if the subject ids are the same - on CPU only
        # TODO: better comparable if we would train several times with the same splits and average the result
        make_deterministic_as_possible()

        training_data = AugmentingHuman36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/augmenting_training_data",
                                                  ids_of_subjects_to_load=training_subject_ids)

        test_data = Human36MDataset(data_save_directory=PATH_TO_DATA_DIRECTORY + "Human36M/test_data",
                                    ids_of_subjects_to_load=[test_subject_id])

        path_to_model_config_file = PATH_TO_CONFIG_DIRECTORY + "model_config.json"
        model_config = load_config_file(path_to_model_config_file)
        model = get_model(model_config)
        if training_hyperparameter_config["continue_learning_from_existing_model"]:
            print("Continuing training from existing model in ", PATH_TO_MODEL)
            load_model_weights(model, PATH_TO_MODEL)

        # train for specified number of epochs
        trainer = Trainer(model,
                          training_data,
                          test_data,
                          training_config=training_hyperparameter_config)
        trainer.run_training_session()

        # load weights of best model on test data
        path_to_best_model_on_test_data = MODEL_DIRECTORY + MODEL_NAME_PREFIX + "best_on_test_data" + ".model"
        load_model_weights(model, path_to_best_model_on_test_data)

        # compute validation loss
        validator = Validator(model,
                              validation_data,
                              training_hyperparameter_config)
        mean_validation_loss = validator.compute_validation_loss()
        sum_of_mean_losses_for_current_validation_subject += float(mean_validation_loss)
        validations_counter += 1

        remove_data(PATH_TO_DATA_DIRECTORY + "Human36M/augmenting_training_data")
        remove_data(PATH_TO_DATA_DIRECTORY + "Human36M/test_data")

    mean_validation_loss_per_subject.append(sum_of_mean_losses_for_current_validation_subject / float(validations_counter))

    remove_data(PATH_TO_DATA_DIRECTORY + "Human36M/validation_data")

# compute evaluation results
overall_mean_validation_loss = mean(mean_validation_loss_per_subject).item()
print("overall_mean_validation_loss: ", overall_mean_validation_loss)
print("mean_validation_loss_per_subject:", mean_validation_loss_per_subject)
evaluation_results = {"overall_mean_validation_loss": overall_mean_validation_loss,
                      "mean_validation_loss_per_subject": mean_validation_loss_per_subject}

# save results to json file
with open(MODEL_DIRECTORY + "evaluation_results", "w") as outfile:
    save_json_file(evaluation_results, outfile, indent=2)
