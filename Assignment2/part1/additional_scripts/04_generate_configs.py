import yaml
import os

lrs = [0.0004, 0.0005, 0.0006, 0.0007]
momentums = [0.7, 0.9]

i = 1
for lr in lrs:
    for mom in momentums:
        cfg = {
            "tta": {
                "name": f"m{i}",
                "class_path": "tta.submission.Submission",
                "args": {
                    "lr": lr,
                    "momentum": mom,
                },
            },

            "model": {
                "class_path": "models.load_base_model",
                "args": {
                    "ckpt_path": "/home/elhoujja/CS461_Assignment2/cs461_assignment2_submission/part1/pretrained_cifar10_resnet50.pt"
                }
            }
        }

        filename = f"m{i}.yaml"
        with open(filename, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

        print(f"Created {filename} with lr={lr}, momentum={mom}")
        i += 1
