import sys
sys.path.append('../../')
import CreateFeaturesArray


class InputParser:

    @staticmethod
    def parse_input_zoo_data(filename, header='infer'):

        #input_data = pd.read_csv(filename, header=header)
        feature_matrix,labels = CreateFeaturesArray.getFeatureArray("./../generated_frames")
        #classes = input_data[17].tolist()

        input_database = {
            0: feature_matrix
        }

        return input_database, labels
