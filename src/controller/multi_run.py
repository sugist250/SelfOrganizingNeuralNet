import os
import sys
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.controller.model_run.ca_run import ca_model_run
from src.controller.model_run.sa_run import sa_model_run
from src.controller.model_run.csa_run import csa_model_run
from src.controller.model_run.csca_run import csca_model_run

def multi_run():
    # 関数を実行するプロセスの準備
    ca = multiprocessing.Process(name="ca", target=ca_model_run)
    sa = multiprocessing.Process(name="sa", target=sa_model_run)
    csa = multiprocessing.Process(name="csa", target=csa_model_run)
    csca = multiprocessing.Process(name="csca", target=csca_model_run)

    # プロセスの開始
    ca.start()
    sa.start()
    csa.start()
    csca.start()



if __name__ == '__main__':
    multi_run()
