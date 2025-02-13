import os
from config import log
from shutil import rmtree


filelist = ['dataset', 'result', 'tensorboard']
for del_dir in filelist:
    if os.path.exists(del_dir):
        rmtree(del_dir)
        log.info(f"'./{del_dir}/' has been deleted!")
    else:
        log.info(f"'./{del_dir}/' does not exist!")

# close all handlers
handlers = log.logger.handlers[:]
for handler in handlers:
    handler.close()
    log.logger.removeHandler(handler)

if os.path.exists('log'):
    rmtree('log')
    print("'./log/' has been deleted!")