# user_interface.py
from config import CommonConfig

class UserInterface:
    """User interaction interface class"""

    @staticmethod
    def select_object():
        """Select object to process"""
        print("\nAvailable objects:")
        for i, (key, config) in enumerate(CommonConfig.OBJECT_CONFIGS.items(), 1):
            print(f"{i}. {config['name']} ({config['num_frames']} frames)")

        while True:
            try:
                choice = int(input(f"\nSelect object number (1-{len(CommonConfig.OBJECT_CONFIGS)}): "))
                if 1 <= choice <= len(CommonConfig.OBJECT_CONFIGS):
                    return list(CommonConfig.OBJECT_CONFIGS.keys())[choice - 1]
                print(f"Invalid choice. Please select a number between 1-{len(CommonConfig.OBJECT_CONFIGS)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    @staticmethod
    def select_distance_threshold():
        """Function to adjust distance threshold"""
        default_threshold = CommonConfig.FEATURE_PARAMS['default_distance_threshold']
        print(f"\nCurrent distance threshold for filtering duplicate matches is: {default_threshold}")
        adjust = input("Do you want to adjust the distance threshold? (y/n): ").lower().strip()

        if adjust.startswith('y'):
            while True:
                try:
                    threshold = float(input("Enter new distance threshold (0-10): "))
                    if 0 <= threshold <= 10:
                        print(f"Distance threshold set to: {threshold}")
                        return threshold
                    print("Invalid threshold. Please enter a value between 0 and 10.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print(f"Using default distance threshold: {default_threshold}")
            return default_threshold