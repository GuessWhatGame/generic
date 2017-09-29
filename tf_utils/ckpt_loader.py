import os
import pickle


def load_checkpoint(sess, saver, args, save_path):
    ckpt_path = save_path.format('params.ckpt')


    if args.continue_exp:
        if not os.path.exists(save_path.format('checkpoint')):
            raise ValueError("Checkpoint " + save_path.format('checkpoint') + " could not be found.")

        saver.restore(sess, ckpt_path)
        status_path = save_path.format('status.pkl')
        status = pickle.load(open(status_path, 'rb'))

        return status['epoch'] + 1

    if args.load_checkpoint is not None:
        #if not os.path.exists(save_path.format('checkpoint')):
        #    raise ValueError("Checkpoint " + args.load_checkpoint + " could not be found.")
        saver.restore(sess, args.load_checkpoint)

        return 0

    return 0