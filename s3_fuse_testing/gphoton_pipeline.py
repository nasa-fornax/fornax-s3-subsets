"""run the gPhoton pipeline with specified parameters"""

import fire

from gPhoton.pipeline import execute_pipeline

# tell fire to handle command line call
if __name__ == "__main__":
    fire.Fire(execute_pipeline)
