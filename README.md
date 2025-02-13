# SINAR  Framework
**_Sistem Inteligen Anti Kejahatan Jalan Raya_**

---
### Implementation of Paper:

**Detecting Motorcycle Crime Gangs in CCTV Video Footage Using YOLOv9 and CNN**

At Proceeding of The 4th International Conference on Intelligent Cybernetics Technology & Applications 2024 (ICICyTA 2024)

---

## Instalation

```bash
$ pip install git+https://github.com/versa-syahptr/sinar-backend.git
```

Optional Features:
- Service `#egg=sinar[service]`
- Annotator `#egg=sinar[Annotator]`


## CLI Usage

```bash
$ sinar <command> [params]
```

### Commands:

- **Predict**
```
$ sinar predict -y YOLO -a ANBEV [-d DEVICE] -i INPUT [-o OUTPUT] [-v]

options:
  -y YOLO, --yolo YOLO_PATH  path to yolo model
  -a ANBEV, --anbev ANBEV_PATH
                        path to analysis behavior model (.keras/.h5 file)
  -d DEVICE, --device DEVICE
                        device to run yolo model, default: cpu
  -i INPUT, --input INPUT
                        source of video
  -o OUTPUT, --output OUTPUT
                        output file
  -v, --view            show the result
```

- **Service**
```
$ sinar service [-h] [--port PORT]
```
