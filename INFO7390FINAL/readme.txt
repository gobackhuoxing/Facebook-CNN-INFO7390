install tensorflow:
https://www.tensorflow.org/install/

In some cases you can not install tensorflow:
1.install chocolatey
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

2.install bazel
choco install bazel

3.Run this in CMD
bazel build -c opt --copt=-march=native --copt=-mfpmath=both --config=cuda -k //tensorflow/tools/pip_package:build_pip_package

install google vision API: https://cloud.google.com/vision/

how to use Facebook graph API:
https://developers.facebook.com/docs/graph-api
https://towardsdatascience.com/how-to-use-facebook-graph-api-and-extract-data-using-python-1839e19d6999