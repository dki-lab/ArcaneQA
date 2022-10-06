import logging
import os
import sys
from allennlp.commands import main  # pylint: disable=wrong-import-position
# sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

if __name__ == "__main__":
    # os.environ['TORCH_HOME'] = '/data/gu.826/.cahce'
    # os.environ['ALLENNLP_CACHE_ROOT'] = '/data/gu.826/.cache'
    # main(prog="python run.py")  # prog argument means nothing. Kun Tao wasn't aware of it.
    main()