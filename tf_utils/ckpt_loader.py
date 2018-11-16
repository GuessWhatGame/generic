import os
import tensorflow as tf
import collections
import logging
import argparse
import json


class ExperienceManager(object):

    status_filename = "status.json"
    params_filename = "params.ckpt"

    def __init__(self, xp_id, xp_dir, args, config, user_data=None):

        self.id = xp_id

        self.dir_xp = xp_dir
        self.dir_best_ckpt = os.path.join(xp_dir, "best")
        self.dir_last_ckpt = os.path.join(xp_dir, "last")

        if user_data is None:
            user_data = dict()
        assert isinstance(user_data, dict)

        self.data = dict(
            hash_id=xp_id,
            config=config,
            args=args.__dict__,
            epoch=0,
            best_epoch=0,
            best_valid_loss=float("inf"),
            train_loss=[],
            valid_loss=[],
            extra_losses=collections.defaultdict(list),
            user_data=user_data)

    @staticmethod
    def load_from_xp_id(xp_dir):

        xp_id = os.path.basename(xp_dir)
        xp_manager = ExperienceManager(xp_id, xp_dir,
                                       args=argparse.Namespace(),  # dummy
                                       config=dict())  # dummy

        status_path = os.path.join(xp_dir, "last", ExperienceManager.status_filename)
        with open(status_path, 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
            xp_manager.data = data

        return xp_manager

    def load_checkpoint(self, sess, saver, load_best=False):

        logger = logging.getLogger()

        # Retrieve ckpt path
        if load_best:
            dir_ckpt = self.dir_best_ckpt
        else:
            dir_ckpt = self.dir_last_ckpt

        if os.path.exists(os.path.join(dir_ckpt, 'checkpoint')):

            # Load xp ckpt
            ckpt_path = os.path.join(dir_ckpt, self.params_filename)
            saver.restore(sess, ckpt_path)

            # Load xp state
            status_path = os.path.join(self.dir_xp, self.status_filename)
            with open(status_path, 'rb') as f:
                self.data = json.loads(f.read().decode('utf-8'))

        else:
            logger.warning("Checkpoint could not be found in directory: '{}'.".format(dir_ckpt))

        return self.data["epoch"], self.data['valid_loss']

    def _save(self, sess, saver, dir_ckpt):

        # Create directory
        if not os.path.isdir(dir_ckpt):
            os.makedirs(dir_ckpt)

        # Save checkpoint
        saver.save(sess, os.path.join(dir_ckpt, 'params.ckpt'))

        logger = logging.getLogger()
        logger.info("checkpoint saved... Directory: {}".format(dir_ckpt))

    def save_checkpoint(self, sess, saver, epoch,
                        train_loss,
                        valid_loss,
                        extra_losses):

        # update data
        self.data["epoch"] = epoch
        self.data["train_loss"].append(train_loss)
        self.data["valid_loss"].append(valid_loss)

        for key, value in extra_losses.items():
            self.data["extra_losses"][key].append(value)

        # save best checkpoint
        if valid_loss < self.data["best_valid_loss"]:
            self.data["best_epoch"] = epoch
            self.data["best_valid_loss"] = valid_loss

            self._save(sess, saver, self.dir_best_ckpt)

        # save current checkpoint
        self._save(sess, saver, self.dir_last_ckpt)

        # Save status
        status_path = os.path.join(self.dir_xp, self.status_filename)
        with open(status_path, 'w') as f_out:
            f_out.write(json.dumps(self.data))

    def update_user_data(self, user_data):

        status_path = os.path.join(self.dir_xp, self.status_filename)

        self.data["user_data"] = {**self.data["user_data"], **user_data}

        with open(status_path, 'w') as f_out:
            f_out.write(json.dumps(self.data))


def create_resnet_saver(networks):

    if not isinstance(networks, list):
        networks = [networks]

    resnet_vars = dict()
    for network in networks:

        start = len(network.scope_name) + 1
        for v in network.get_resnet_parameters():
            resnet_vars[v.name[start:-2]] = v

    return tf.train.Saver(resnet_vars)



