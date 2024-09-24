import tensorflow as tf

from sinar.data.centrogen import Centrogen


def train_loop(model: tf.keras.models.Model, 
               epoch: int, 
               train_centrogen: Centrogen, 
               validation_data: tf.data.Dataset,
               regenerate_train_ds=False,
               callbacks_list: list = []):
    
    callbacks_list.append(tf.keras.callbacks.ProgbarLogger())
    cb_list = tf.keras.callbacks.CallbackList(callbacks_list)
    cb_list.set_model(model)
    cb_list.on_train_begin()
    for e in range(epoch):
        epoch_logs = {}
        cb_list.on_epoch_begin(e)
        print(f"Epoch {e+1}/{epoch}")

        # ---------- TRAIN STEP ----------
        if regenerate_train_ds: # only regenerate after the first epoch
            print("Regenerating train dataset")
            train_ds = train_centrogen.regenerate_dataset()
        else:
            train_ds = train_centrogen.dataset

        for i, batch in enumerate(train_ds):
            x, y = batch
            cb_list.on_train_batch_begin(i)
            train_logs = model.train_on_batch(x, y,return_dict=True)
            cb_list.on_train_batch_end(i, train_logs)
        epoch_logs.update(train_logs)

        if model.stop_training:
            break

        # ---------- VALIDATION STEP ----------
        for i, batch in enumerate(validation_data):
            x, y = batch
            cb_list.on_test_batch_begin(i)
            val_logs = model.test_on_batch(x, y, return_dict=True)
            cb_list.on_test_batch_end(i, val_logs)
        epoch_logs["val_loss"] = val_logs["loss"]
        epoch_logs["val_accuracy"] = val_logs["accuracy"]
        cb_list.on_epoch_end(e, epoch_logs)
    cb_list.on_train_end()