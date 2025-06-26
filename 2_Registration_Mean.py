# 2_Registration_Mean.py
from config import CommonConfig
from registration_pipeline import run_registration # Import the main pipeline function

# Define configuration specific to 'Mean' grayscale registration.
class Config(CommonConfig):
    @staticmethod
    def get_transformation_data_path(object_name, i, j):
        return f'transformation/Mean/{object_name}/transformation_{i}_{j}.npz'

    @staticmethod
    def get_results_dir(object_name):
        return f'results/Mean/{object_name}'

    @staticmethod
    def get_final_ply_path(results_dir):
        return f"{results_dir}/3_final_result_Mean.ply"

if __name__ == "__main__":
    try:
        # Run the entire pipeline using the 'Mean' configuration
        run_registration(Config)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise