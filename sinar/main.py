import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from sinar.utils import VideoInfo

from sinar.data import annotator, viewer
from sinar import SINAR
from sinar import stream
from sinar.logger import set_level

DESCRIPTION = "SINAR: Sistem Inteligen Anti Kejahatan Jalan Raya"


def annotator_dry_run(args):
    video_path = args.video_path
    fps = args.fps
    model_path = args.model
    dataset_path = args.dataset_path
    if not video_path.is_absolute():
        video_path = dataset_path / video_path

    print(f"video_path: {video_path}")
    print(f"fps: {fps}")
    print(f"model_path: {model_path}")
    print(f"dataset_path: {dataset_path}")
    print(f"unix video_path: {video_path.as_posix()}")

def handle_centrogen(args):
    # TODO: Implement centrogen
    raise NotImplementedError("Centrogen is not implemented yet")


def handle_train(args):
    # TODO: Implement training
    raise NotImplementedError("Training is not implemented yet")


def handle_predict(args):
    print(f"Predicting with model: {args.yolo}, analysis behavior: {args.anbev}, device: {args.device}, input: {args.input}, output: {args.output}, view: {args.view}")
    sinar = SINAR(args.yolo, args.anbev, live_stream=args.live, device=args.device)
    if args.output:
        vi = VideoInfo.from_video_path(args.input)
        streamer = stream.Saver(vi.width, vi.height, vi.fps, output=args.output)
    elif args.view:
        streamer = stream.Viewer(args.input)
    else:
        streamer = stream.BaseStream()

    sinar.main_loop(args.input, streamto=streamer)


def handle_service(args):
    """
    Run the SINAR service
    """
    import uvicorn
    from sinar.service.server import app

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", args.port)),
        log_level="info"
    )

def handle_demo(args):
    """
    Start the step-by-step Streamlit demo (demo/app.py), from any working directory
    """
    # demo/ lives next to the package in the source tree and isn't shipped
    backend = Path(__file__).resolve().parent.parent
    app = backend / "demo" / "app.py"
    if not app.exists():
        sys.exit(f"demo app not found at {app}; `sinar demo` needs the source checkout (uv sync)")
    if importlib.util.find_spec("streamlit") is None:
        sys.exit("streamlit is not installed: uv sync --group demo")

    # run from the backend dir so .streamlit/config.toml applies
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port), *args.streamlit_args]
    try:
        sys.exit(subprocess.run(cmd, cwd=backend).returncode)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(prog='sinar', description=DESCRIPTION)
    parser.add_argument(
        "-L", "--log-level",
        choices=["trace", "debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set SINAR framework log level (default: info)"
    )

    subparsers = parser.add_subparsers(title='commands', dest='command')

    # Data subcommand
    parser_data = subparsers.add_parser('data', help='Handle data operations (internal use only)')
    subparsers_data = parser_data.add_subparsers(title='data subcommands', dest='data_command')

    # Data > Annotate subcommand
    parser_annotate = subparsers_data.add_parser('annotate', help='Auto annotate video frames')
    parser_annotate.add_argument('-m', "--model", type=Path, required=True, help="Path to YOLO model")
    parser_annotate.add_argument('-d', "--dataset-path", type=Path, required=True, help="Path to dataset")
    parser_annotate.add_argument('-v', "--video-path", type=Path, help="Path to video file")
    parser_annotate.add_argument("-fps", type=int, default=1, help="Frame per second to extract")
    parser_annotate.set_defaults(func=annotator.main)
    # parser_annotate.set_defaults(func=annotator_dry_run)

    # Data > Centrogen subcommand
    parser_centrogen = subparsers_data.add_parser('centrogen', help='Generate center points from video')
    parser_centrogen.add_argument("video_path", type=Path, help="Path to video file")
    parser_centrogen.set_defaults(func=handle_centrogen)

    # Data > View subcommand
    parser_validate = subparsers_data.add_parser('view', help='View the dataset with pre-annotated bounding boxes')
    parser_validate.add_argument("dataset_dir", type=Path, help="Path to the YOLO dataset directory")
    parser_validate.set_defaults(func=viewer.main)

    # Train subcommand
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.set_defaults(func=handle_train)

    # Predict subcommand
    parser_predict = subparsers.add_parser('predict', help='Predict using the model')
    parser_predict.add_argument("-y", "--yolo", type=str, required=True, help="path to yolo model")
    parser_predict.add_argument("-a", "--anbev", type=str, required=True, help="path to analysis behavior model")
    parser_predict.add_argument("-d", "--device", type=str, default="cpu", help="device to run yolo model, default: cpu")
    parser_predict.add_argument("-i", "--input", type=str, required=True, help="source of video")
    parser_predict.add_argument("-o", "--output", type=str, default=None, help="output file")
    parser_predict.add_argument("-v", "--view", action="store_true", help="show the result")
    parser_predict.add_argument("-l", "--live", action="store_true", help="enable live stream, disable the greedy algorithm")
    parser_predict.set_defaults(func=handle_predict)

    # Service subcommand
    parser_service = subparsers.add_parser('service', help='Run the service')
    parser_service.add_argument('--port', type=int, default=8080, help='Port to run the service on')
    parser_service.set_defaults(func=handle_service)

    # Demo subcommand
    parser_demo = subparsers.add_parser('demo', help='Start the step-by-step Streamlit demo',
                                        epilog='Other options are passed to `streamlit run`, e.g. --server.headless true')
    parser_demo.add_argument('--port', type=int, default=8501, help='Port to serve the demo on (default: 8501)')
    # anything else on the command line goes to `streamlit run`, e.g. --server.headless true
    parser_demo.set_defaults(func=handle_demo)

    # Parse arguments and call the appropriate function
    args, extra = parser.parse_known_args()
    if args.command == 'demo':
        args.streamlit_args = extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    set_level(args.log_level)

    if args.command:
        if args.command == 'data' and not args.data_command:
            parser_data.print_help()
        else:
            args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
