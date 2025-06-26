# 2_Registration_All.py
from config import CommonConfig
from registration_pipeline import run_registration # Import the main pipeline function

# Define configuration specific to 'All Bands' registration.
class Config(CommonConfig):
    @staticmethod
    def get_transformation_data_path(object_name, i, j):
        return f'transformation/all/{object_name}/transformation_{i}_{j}.npz'

    @staticmethod
    def get_results_dir(object_name):
        return f'results/all/{object_name}'

    @staticmethod
    def get_final_ply_path(results_dir):
        return f"{results_dir}/3_final_result_All.ply"

if __name__ == "__main__":
    try:
        # Run the entire pipeline using the 'All Bands' configuration
        run_registration(Config)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise