# 2_Registration_False_Color.py
from config import CommonConfig
from registration_pipeline import run_registration # Import the main pipeline function

# Define configuration specific to 'False Color' registration.
class Config(CommonConfig):
    @staticmethod
    def get_transformation_data_path(object_name, i, j):
        return f'transformation/False Color/{object_name}/transformation_{i}_{j}.npz'

    @staticmethod
    def get_results_dir(object_name):
        return f'results/False Color/{object_name}'

    @staticmethod
    def get_final_ply_path(results_dir):
        return f"{results_dir}/3_final_result_False Color.ply"

if __name__ == "__main__":
    try:
        # Run the entire pipeline using the 'False Color' configuration
        run_registration(Config)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise