import argparse
import os
from pathlib import Path

from supervision.video.dataclasses import VideoInfo

from sinar.data import annotator, viewer
from sinar import SINAR
from sinar import stream


DESCRIPTION = "SINAR: "


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
        vi = VideoInfo(args.input)
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

def main():
    parser = argparse.ArgumentParser(prog='sinar', )
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

    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if args.command:
        if args.command == 'data' and not args.data_command:
            parser_data.print_help()
        else:
            args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
