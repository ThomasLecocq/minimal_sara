import datetime
import glob
import itertools
import os
import logging
import bottleneck as bn
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from configobj import ConfigObj
from obspy import read, UTCDateTime, Stream
from obspy.signal.util import next_pow_2
from obspy.signal.filter import envelope as obspy_envelope

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# from click_plugins import with_plugins
def to_idds(net, sta, loc, chan, year, jday, hour):
    SDS="YEAR/NET/STA/CHAN.TYPE/JDAY/NET.STA.LOC.CHAN.TYPE.YEAR.JDAY.HOUR"
    file=SDS.replace('YEAR', "%04i"%year)
    file=file.replace('NET', net)
    file=file.replace('STA', sta)
    file=file.replace('LOC', loc)
    file=file.replace('CHAN', chan)
    file=file.replace('JDAY', jday)
    file = file.replace('HOUR', hour)
    file=file.replace('TYPE', "D")
    return file

def to_agung(net, sta, loc, chan, year, jday, hour):
    SDS="YEAR/NET/STA/CHAN.TYPE/JDAY/"
    file=SDS.replace('YEAR', "%04i"%year)
    file=file.replace('NET', net)
    file=file.replace('STA', sta)
    file=file.replace('LOC', loc)
    file=file.replace('CHAN', chan)
    file=file.replace('JDAY', jday)
    file = file.replace('HOUR', hour)
    file=file.replace('TYPE', "D")
    file += "NET.STA.*.*.*.YEAR.JDAY.HOUR"
    file = glob.glob(file)[0]
    return file


def to_sds(net, sta, loc,chan, year, jday):
    SDS="YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.JDAY"
    file=SDS.replace('YEAR', "%04i"%year)
    file=file.replace('NET', net)
    file=file.replace('STA', sta)
    file=file.replace('LOC', loc)
    file=file.replace('CHAN', chan)
    file=file.replace('JDAY', jday)
    file=file.replace('TYPE', "D")
    return file

@click.group()
@click.pass_context
def cli(ctx):
    config_file = os.path.join(os.getcwd(), "config.ini")
    config = ConfigObj(config_file)
    ctx.obj['config'] = config

    pass


@click.command()
@click.pass_context
def init(ctx):
    filelist = os.path.join(os.getcwd(), "filelist.csv")
    if os.path.isfile(filelist):
        logging.debug("Init seems to be already done")
        return
    df = pd.DataFrame([["0","0",datetime.date(2010,1,1),"0","1"],],
                      columns=["net","sta","date","size","process_step"],
                      index=["nopath",])

    df.to_csv(filelist, index_label="path")
    pass


@click.command()
@click.pass_context
def scan_archive(ctx):
    config = ctx.obj["config"]["msara"]
    filelist = os.path.join(os.getcwd(), "filelist.csv")
    filedb = pd.read_csv(filelist, index_col="path")

    data_folder = config["data_folder"]
    data_format = config["data_format"]
    if data_format == "SDS":
        dates = pd.date_range(UTCDateTime(config["startdate"]).datetime,
                              UTCDateTime(config["enddate"]).datetime, freq="D")
    elif data_format in ["IDDS", "AGUNG"]:
        dates = pd.date_range(UTCDateTime(config["startdate"]).datetime,
                              UTCDateTime(config["enddate"]).datetime, freq="H")
    else:
        logging.debug("data_format %s not supported" % data_format)
        return

    network = config["network"]
    stations = config["stations"]
    location = config["location"]
    channel = config["channel"]
    for station in stations:
        for date in dates:
            if data_format == "SDS":
                path = to_sds(network, station, location, channel, date.year, "%03i"% date.dayofyear)
            elif data_format == "IDDS":
                path = to_idds(network, station, location, channel, date.year, "%03i"% date.dayofyear, "%02i" % date.hour)
            elif data_format == "AGUNG":
                path = to_agung(network, station, location, channel, date.year, "%03i"% date.dayofyear, "%02i" % date.hour)
            path = os.path.join(data_folder, path)
            if not os.path.isfile(path):
                continue
            filesize = os.stat(path).st_size
            if path in filedb.index:
                tmp = filedb.loc[path]
                if tmp["size"] != filesize:
                    logging.debug(tmp["size"], filesize)
                    logging.debug("file changed")
                    filedb.loc[path, "size"] = filesize
                    filedb.loc[path, "process_step"] = "E"
                    new_data = True
                else:
                    logging.debug("path is in the df already, and no size change")
                    new_data = False
            else:
                logging.debug("we should add this file to the df")
                new = pd.Series({
                    "net": network,
                    "sta": station,
                    "date": date.date(),
                    "size": filesize,
                    "process_step": "E"
                }, name=path)
                filedb = filedb.append(new)
                new_data = True

    if new_data:
        filedb.to_csv(filelist)



@click.command()
@click.pass_context
def envelope(ctx):
    config = ctx.obj["config"]["msara"]
    data_folder = config["data_folder"]
    filelist = os.path.join(os.getcwd(), "filelist.csv")
    filedb = pd.read_csv(filelist, index_col="path")
    tmp = filedb[filedb.process_step == "E"]
    if len(tmp) == 0:
        logging.debug("No files to process")
        return

    logging.debug("Will process envelope for:", len(tmp), "files")
    for file in tmp.index:
        if not os.path.isfile(file):
            logging.debug("%s NOT FOUND")
            continue

        outfile = file.replace(data_folder, os.path.join(os.getcwd(),"ENV"))
        logging.debug("will output to %s"%outfile)
        if not os.path.isdir(os.path.split(outfile)[0]):
            os.makedirs(os.path.split(outfile)[0])

        # PREPROCESS
        logging.debug("Loading")
        st = read(file)
        logging.debug("Detrend")
        st.detrend("demean")
        logging.debug("Merge")
        st.merge(method=1, fill_value="interpolate")
        logging.debug("Taper")
        st.taper(max_percentage=None, max_length=1.0)
        logging.debug("Filter")
        freqmin, freqmax = config["bandpass"]
        st.filter("bandpass", freqmin=float(freqmin), freqmax=float(freqmax),corners=8)
        logging.debug(st)
        trace = st[0]
        n = int(config["env_sampling_rate"])
        sps = int(trace.stats.sampling_rate)

        logging.debug("Envelope")
        from scipy.signal import hilbert
        N = next_pow_2(trace.stats.npts)
        trace.data = np.abs(hilbert(trace.data, N=N)[:trace.stats.npts])
        logging.debug("Moving Median")
        trace.data = bn.move_median(trace.data, sps * n)
        trace.data = trace.data[n * sps - 1::sps * n]
        trace.stats.sampling_rate = 1. / float(n)
        trace.data = np.require(trace.data, np.float32)
        logging.debug("Writing")
        trace.write(outfile, format="MSEED")
        del st, trace
        filedb.loc[file, "process_step"] = "R"

    filedb.to_csv(filelist)

@click.command()
@click.pass_context
def ratio(ctx):
    config = ctx.obj["config"]["msara"]
    data_folder = config["data_folder"]
    data_format = config["data_format"]
    filelist = os.path.join(os.getcwd(), "filelist.csv")
    filedb = pd.read_csv(filelist, index_col="path", parse_dates=["date",])
    tmp = filedb[filedb.process_step == "R"]
    if len(tmp) == 0:
        logging.debug("No files to process")
        return
    groups = filedb.groupby("date")
    for date, group in groups:
        # date = pd.datetime(date)
        if "R" not in group["process_step"].values:
            logging.debug("Nothing do to for %s"%date)
            continue

        logging.debug(date)
        traces = []
        stations = np.unique(sorted(group["sta"]))
        for station in stations:
            logging.debug("Will load %s" % station)
            if data_format == "SDS":
                path = to_sds("*", station, "*", "*", date.year,
                              "%03i" % date.dayofyear)
            elif data_format in ["IDDS" , "AGUNG"]:
                path = to_idds("*", station, "*", "*", date.year,
                               "%03i" % date.dayofyear, "*")
            data_folder = os.path.join(os.getcwd(), "ENV")
            path = os.path.join(data_folder, path)
            logging.debug(path)
            for file in glob.glob(path):
                traces.append(read(file)[0])

        st = Stream(traces)
        stations = np.unique([tr.stats.station for tr in st])
        for sta1, sta2 in itertools.combinations(stations,2):
            st1 = st.select(station=sta1)
            st1.merge(method=1, fill_value="interpolate")

            st2 = st.select(station=sta2)
            st2.merge(method=1, fill_value="interpolate")

            max_start = max([tr.stats.starttime for tr in st1],[tr.stats.starttime for tr in st2])[0]
            min_end = min([tr.stats.endtime for tr in st1],
                            [tr.stats.endtime for tr in st2])[0]

            st1 = st1.trim(max_start, min_end)
            st2 = st2.trim(max_start, min_end)

            stR = st1[0].copy()
            stR.data = st1[0].data / st2[0].data
            stR.stats.station = "RATIO"
            pair = "%s_%s" % (sta1, sta2)
            env_output_dir = os.path.join(os.getcwd(), 'RATIO', pair)
            if not os.path.isdir(env_output_dir):
                os.makedirs(env_output_dir)
            stR.write(os.path.join(env_output_dir, str(date.date()) + '.MSEED'),
                        format="MSEED", encoding="FLOAT32")
            del stR
        for file in group.index:
            filedb.loc[file, "process_step"] = "P"
    filedb.to_csv(filelist)


@click.command()
@click.pass_context
def plot(ctx):
    config = ctx.obj["config"]["msara"]
    stations = config["stations"]
    smoothing = config["smoothing"]
    if type(smoothing) in [float, int, str]:
        smoothing = [smoothing,]
    print(smoothing)
    # plt.figure()
    # for sta1, sta2 in itertools.combinations(stations, 2):
    #     pair = "%s_%s" % (sta1, sta2)
    #     path = os.path.join(os.getcwd(), 'RATIO', pair, "*")
    #     st = read(path, format="MSEED")
    #     st.merge(method=1, fill_value="interpolate")
    #     tr = st[0]
    #     times = pd.date_range(tr.stats.starttime.datetime, tr.stats.endtime.datetime, freq="%ims"%(tr.stats.delta*1e3))
    #     plt.plot(times, tr.data, label=pair)
    # plt.legend()
    # plt.title("Raw Ratios")
    # plt.grid(True)


    for smooth in smoothing:
        plt.figure()
        for sta1, sta2 in itertools.combinations(stations, 2):
            pair = "%s_%s" % (sta1, sta2)
            path = os.path.join(os.getcwd(), 'RATIO', pair, "*")
            st = read(path, format="MSEED")
            st.merge(method=1, fill_value="interpolate")
            tr = st[0]
            sps_ratio = int(int(smooth) / tr.stats.delta)

            # tr.data = pd.rolling_median(tr.data, window=sps_ratio)
            idx = pd.date_range(tr.stats.starttime.datetime,
                                  tr.stats.endtime.datetime,
                                  freq="%ims" % (tr.stats.delta * 1e3))
            df = pd.Series(tr.data, index=idx)
            df = df.resample("%iS" % int(smooth)).median()
            plt.plot(df.index, df, label=pair)

        plt.legend()
        plt.title("Smoothed ratios (smoothing=%is)"%int(smooth))
        plt.grid(True)
    plt.show()

cli.add_command(init)
cli.add_command(scan_archive)
cli.add_command(envelope)
cli.add_command(ratio)
cli.add_command(plot)


def main():
    cli(obj={})
