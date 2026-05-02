import subprocess
import sys
import os

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(pkgs))

try:
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import FTPServer
except (ImportError, AttributeError):
    print("Installing/fixing dependencies...")
    pip_install("--upgrade", "pyopenssl")
    pip_install("pyftpdlib")
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import FTPServer

def main():
    authorizer = DummyAuthorizer()
    authorizer.add_user("admin", "YOUR_PASSWORD", ".", perm="elradfmw")

    handler = FTPHandler
    handler.authorizer = authorizer
    handler.banner = "FTP Server is ready."

    address = ("0.0.0.0", 2121)
    server = FTPServer(address, handler)

    print(f"FTP server active at: {os.getcwd()}")
    print("Accessible on port 2121")
    server.serve_forever()

if __name__ == "__main__":
    main()
