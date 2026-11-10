"""Lanza el train.py oficial de RT-DETRv4 con estos añadidos, sin modificar el repo: corrección de pos_embed,
last.pth guardado solo al final y early stopping.

Uso: python train_rtdetr.py <repo> <patience> <argumentos de train.py...>
Tiene que ser un .py (y no una celda) porque en Windows los workers del DataLoader reimportan el programa principal.
"""
import importlib
import runpy
import sys
from functools import partial, partialmethod
from pathlib import Path

import torch


def main(repo, patience, train_args):
    patch_repo(repo, patience)
    sys.argv = [str(repo / "train.py"), *train_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


def patch_repo(repo, patience):
    sys.path.insert(0, str(repo))
    det_solver = importlib.import_module("engine.solver.det_solver")
    hybrid_encoder = importlib.import_module("engine.rtv4.hybrid_encoder").HybridEncoder
    early_stopping = {"patience": patience, "best_map": -1.0, "stalled_epochs": 0}

    hybrid_encoder._reset_parameters = partialmethod(register_pos_embeds_as_buffers, hybrid_encoder._reset_parameters)
    det_solver.DetSolver.fit = partialmethod(fit_and_save_last, det_solver.DetSolver.fit)
    det_solver.train_one_epoch = partial(stop_if_stalled, early_stopping, det_solver.train_one_epoch)
    det_solver.dist_utils.save_on_master = partial(skip_last_checkpoint, det_solver.dist_utils.save_on_master)
    det_solver.evaluate = partial(track_validation_map, early_stopping, det_solver.evaluate)


def register_pos_embeds_as_buffers(encoder, reset_parameters):
    reset_parameters(encoder)
    for i in encoder.use_encoder_idx:
        pos_embed = encoder.__dict__.pop(f"pos_embed{i}", None)
        if pos_embed is not None:
            encoder.register_buffer(f"pos_embed{i}", pos_embed, persistent=False)


class EarlyStop(Exception):
    pass


def fit_and_save_last(solver, fit):
    try:
        fit(solver)
    except EarlyStop as stop:
        print(stop)
    finally:
        torch.save(solver.state_dict(), solver.output_dir / "last.pth")
        if solver.writer:
            solver.writer.close()


def stop_if_stalled(early_stopping, train_one_epoch, *args, **kwargs):
    if early_stopping["stalled_epochs"] >= early_stopping["patience"]:
        raise EarlyStop(f"[early stopping] {early_stopping['patience']} épocas sin mejorar: se para el entrenamiento")
    return train_one_epoch(*args, **kwargs)


def skip_last_checkpoint(save_on_master, obj, path):
    if Path(path).name != "last.pth":
        return save_on_master(obj, path)


def track_validation_map(early_stopping, evaluate, *args, **kwargs):
    stats, coco = evaluate(*args, **kwargs)
    validation_map = stats["coco_eval_bbox"][0]
    if validation_map > early_stopping["best_map"]:
        early_stopping["best_map"], early_stopping["stalled_epochs"] = validation_map, 0
    else:
        early_stopping["stalled_epochs"] += 1
    print(f"[early stopping] mAP50-95 = {validation_map:.4f}, mejor = {early_stopping['best_map']:.4f}, "
          f"épocas sin mejorar = {early_stopping['stalled_epochs']}/{early_stopping['patience']}")
    return stats, coco


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
