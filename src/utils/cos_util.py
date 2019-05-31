import commands


class CosUtil:

    @staticmethod
    def upload_dir(local_dir, cos_dir):
        """
        upload a local dir to cos dir
        :return:
        """
        cmd = "coscmd -b peng-1256590953 upload -r {local_dir} {cos_dir}".format(local_dir=local_dir, cos_dir=cos_dir)
        ret = commands.getoutput(cmd)
        print("[cos] upload_dir cmd:{} : {}".format(cmd, ret))
