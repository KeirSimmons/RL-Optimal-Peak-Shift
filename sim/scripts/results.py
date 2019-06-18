import os
import shutil
import logging
import socket
import re
import numpy as np

"""
Handles the results directories and saving of files
during training and testing phases
"""

class Results:

    def __init__(self, result_dir=None):
        self.dir = "results"
        self.user = socket.gethostname() # Allows saving results over multiple machines without name conflicts
        self.counter = None
        self.get_counter()
        if result_dir is None:
            self.cleanup()
            self.create()
        else:
            self.load(result_dir)

    def get_counter(self):
        # returns number for NEXT result
        if self.counter is None:
            pattern = re.compile("^{}-\d+$".format(self.user))
            dirs = os.listdir(self.dir)
            current = np.sum([x for x in map(lambda x: pattern.match(x) is not None, dirs)]).astype(int)
            self.counter = current + 1
            return self.counter
        else:
            return self.counter

    def cleanup(self):
        # Deletes unfinished results directories
        counter = self.counter - 1

        while True:
            if counter == 0:
                break
            
            # We check results/counter to see if it needs to be deleted or not
            print("Checking {}-{}".format(self.user, counter))
            status = self.status(result=counter)
            if status < 3:
                # We should delete this directory as training was not completed
                path = os.path.join(self.dir, self.user + "-" + str(counter))
                shutil.rmtree(path)
                logging.info('Removing incomplete results directory {}'.format(path))
            else:
                # This directory is fully trained, so all before should be too!
                break

            counter -= 1

        self.counter = counter + 1
        
    def create(self):
        # Creates new results directory
        this_result = self.counter
        os.makedirs(os.path.join(self.dir, self.user + "-" + str(this_result)))
        self.current = self.user + "-" + str(this_result)
        self.current_dir = os.path.join(self.dir, self.current)
        self.status(1)

    def load(self, result_dir):
        try:
            result_dir = int(result_dir)
            self.current = self.user + "-" + str(result_dir)
        except ValueError:
            # Result directory is a string literal!
            self.current = result_dir
        
        self.current_dir = os.path.join(self.dir, self.current)
        # TODO: Check exists

    def status(self, status=None, result=None):
        ## status
        ### 1: Directory created, nothing else done
        ## If status is not none, we update, otherwise we read

        result = self.current if result is None else self.user + "-" + str(result)

        if status is None:
            print(os.path.join(self.dir, result, ".status"))
            with open(os.path.join(self.dir, result, ".status"), 'r') as status_file:
                return int(status_file.read())
        else:
            with open(os.path.join(self.dir, result, ".status"), 'w') as status_file:
                status_file.write(str(status))
                return status

    def get_path(self, path):
        return os.path.join(self.current_dir, path)

    def create_dir(self, dir):
        os.makedirs(os.path.join(self.current_dir, dir))