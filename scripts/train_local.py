"""
Script to run the allennlp model
"""
import logging
import sys
from pathlib import Path
import os
import random

sys.path.append(str(Path().absolute()))
# from allennlp.commands import main
from scicite.training.commands import main

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
_seed = random.randrange(2**32 - 1)

os.environ['SEED'] = str(_seed)
os.environ['PYTORCH_SEED'] = str(int(os.environ['SEED']) // 3)
os.environ['NUMPY_SEED'] = str(int(os.environ['PYTORCH_SEED']) // 3)

logger.info(f"SEED={_seed}, PYTORCH_SEED={os.environ['PYTORCH_SEED']}, NUMPY_SEED={os.environ['NUMPY_SEED']}") 

os.environ["elmo"] = "true"

if __name__ == "__main__":
    # Make sure a new predictor is imported in processes/__init__.py
    main(prog="python -m allennlp.run")

