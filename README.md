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
7. run "msara plot"

Alternatively, if the archive is not properly structured (bad file naming, etc),
you can replace step 4. with a direct call to a path:

4. msara scan_archive --path C:\Users\thlec\Desktop\VOLCANO\IDDS

This path will be scanned, recursively (all folders will be read) and files
will be read, and if the tr.stats.station match what is configured in config.ini
and if the tr.stats.sampling_rate is larger than twice the freq_max of the badnpass
(in the config.ini above, it's 15, so traces with sps > 30 Hz will be accepted).
