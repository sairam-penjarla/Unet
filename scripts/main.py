from code.config import read_config
from code.pipelines.data_gathering import DataGathering
from code.pipelines.data_preparation import DataPreparation
from code.pipelines.model_training import ModelTraining
from code.pipelines.model_inferencing import ModelInferencing

config = read_config()

data_gathering = DataGathering(config)
data_preparation = DataPreparation(config)
model_training = ModelTraining(config)
model_inferencing = ModelInferencing()

gather_data = config['flow']['gather_data']
prepare_data = config['flow']['prepare_data']
run_training = config['flow']['run_training']
inference = config['flow']['inference']['flag']

if gather_data:
    data_gathering.run()

if prepare_data:
    data_preparation.run()

if run_training:
    model_training.run()

if inference:
    MODEL_PATH = config['paths']['model_save_path']
    single_inference = config['flow']['inference']['single_inference']['flag']
    mulitple_inference = config['flow']['inference']['mulitple_inference']['flag']
    if single_inference:
        SINGLE_IMG_PATH = config['flow']['inference']['single_inference']['path']
        model_inferencing.single_image_inference(SINGLE_IMG_PATH, MODEL_PATH)
    if mulitple_inference:
        DATA_PATH = config['flow']['inference']['mulitple_inference']['path']
        model_inferencing.pred_show_image_grid(DATA_PATH, MODEL_PATH)