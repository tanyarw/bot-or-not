import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing import static_to_temporal

if __name__ == '__main__':
    num_snapshots = static_to_temporal(
        input_dir=os.path.join('data', 'static'),
        output_dir=os.path.join('data', 'temporalized')
    )