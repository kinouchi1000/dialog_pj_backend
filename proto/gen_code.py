import glob

from grpc.tools import protoc

protos = glob.glob("*.proto")
# protos.extend(glob.glob("models/*.proto"))

for path in protos:
    protoc.main(
        (
            "",
            "-I.",
            "--python_out=../src/grpc_service/",
            "--grpc_python_out=../src/grpc_service/",
            f"./{path}",
        )
    )
