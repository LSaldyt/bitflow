
import os
from datetime import datetime

LOG_DIR = 'logs/'

class Log:
    def __init__(self, name, directory=None, clear=True):
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        if directory is not None:
            if not os.path.isdir(directory):
                os.mkdir(LOG_DIR + directory)
            self.path = LOG_DIR + directory + '/' + name
        else:
            self.path = LOG_DIR + name
        self.time = datetime.now()
        self.path += '_' + self.time.strftime('%a_%d_%b_%y_%I_%M_%p') + '.log'
        if clear and os.path.isfile(self.path):
            os.remove(self.path)
        elif not os.path.isfile(self.path):
            with open(self.path, 'w') as outfile:
                outfile.write('')

    def log(self, *messages, end='\n'):
        with open(self.path, 'a') as outfile:
            for message in messages:
                outfile.write(str(message) + end)
