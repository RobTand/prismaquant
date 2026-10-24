print(json.dumps({"status": "passed", "served_quantizer": identity,
                  "shape": list(x.shape), "changed_elements": int((x != y).sum().item()),
                  "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()}, sort_keys=True))
