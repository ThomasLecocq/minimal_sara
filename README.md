# minimal_sara
Minimal Package for computing SARA


Install
=======

```sh
pip install bottleneck
pip install https://github.com/ThomasLecocq/minimal_sara/archive/master.zip
```
Operation
=========

1. Create a project folder
2. Create the config.ini file containing:

```text
# This is a template msara config file
[ "msara" ]
data_folder = "C:\Users\thlec\Desktop\testSDS"
data_format = SDS

network = YA
location = "00"
channel = HHZ
stations = UV05, UV06, UV10
startdate = 2010-01-01
enddate = 2011-01-01

bandpass = 5,15
env_sampling_rate = 1
smoothing = 600, 86400
```

3. run "msara init"
4. run "msara scan_archive"
5. run "msara envelope"
6. run "msara ratio"
