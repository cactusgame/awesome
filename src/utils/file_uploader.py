import commands
import os


class FileUploader():
    def __init__(self):
        pass

    def coscmd_upload(self, file_abs_path):
        filename = (os.path.basename(file_abs_path))
        # cmd = "coscmd -b heai-seed-rec-service-dev-1256590953 upload " + file_abs_path + " seedrec/" + filename
        cmd = "cp awesome.db /tmp/featuredb" # this will persist on Node disk
        status, output = commands.getstatusoutput(cmd)
        print(str(status) + " " + str(output))

