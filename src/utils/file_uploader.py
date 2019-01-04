import commands
import os


class FileUploader():
    def __init__(self):
        pass

    def coscmd_upload(self, file_abs_path):
        filename = (os.path.basename(file_abs_path))
        cmd = "coscmd -b heai-seed-rec-service-dev-1256590953 upload " + file_abs_path + " seedrec/" + filename
        status, output = commands.getstatusoutput(cmd)
        print(str(status) + " " + str(output))
