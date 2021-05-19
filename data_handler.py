"""Code for interpretting and analyzing the data from the Red Pitaya.
This version is for interpreting accumulated FFT data."""


import numpy as np
import struct
import matplotlib.pyplot as plt 
from matplotlib.ticker import EngFormatter
import os

#data type for handling 64 bit integer complex data
cint64 = np.dtype([('real',np.int64),('imag',np.int64)])

class dataInterface:
    """Class for working with the raw data. This version assumes data
        is structured as follows:
        32 bit serial number (4 bytes)
        32 bit call state (3 LSB = callibration state, 29 MSB unused) (4 bytes)
        1024 bins, 64 bits/bin, channel 1 PSD (8192 bytes)
        1024 bins, 64 bits/bin, channel 2 PSD (8192 bytes)
        1024 bins, 64 bits/bin, cross correlation real component (8192 bytes)
        1024 bins, 64 bits/bin, cross correlation imaginary component (8192 bytes)
        
        For a total of 8192*4 + 4 + 4 bytes per data sample
        """
    type = 'Interface to raw data'
    def __init__(self, dataFile,Fs = 125e6):
        """
        file: name of file containing the raw data
        Fs: sample frequency (default 125Ms/second)
        """

        self.fft_size = 1024
        self.dataFile = dataFile
        self.bin_bit_size = 64
        self.bin_byte_size = self.bin_bit_size//8
        self.data =[]
        self.freq_axis = (np.arange(self.fft_size) - self.fft_size//2)*(Fs/self.fft_size)
        self.sample_size = 8192*4 + 4 + 4 #bytes/sample

    def load(self, start_sample = 0, end_sample = "all", verbose = 0,
            recenter = 1):
        """
        Load data from the data file.
        verbose: Print sample counts to terminal
        recenter: center data to DC
        start_sample and end_sample functionality no longer implemented.
        Might update in future versions.
        """

        self.data = []
        file_size = os.path.getsize(self.dataFile)
        with open(self.dataFile, "rb") as fd:
            sample_num = 0
            while(file_size - fd.tell()) >= self.sample_size:
                if verbose:
                    print("loading sample #" + str(sample_num))
                    sample_num += 1
                self.data.append(sample())
                self.data[-1].load(fd,recenter = recenter)




    def plot_sample(self, sample_number, ylimits = "",xlimits = "", _yscale = "log"):
        """
        Plot the data present in sample_number.
        use: plot_sample(sample_number, ylimits)
        sample_number: sample to plot
        ylimits: optionally manual setfile ylimits of plot
        """
        if self.data == []:
            print("Need to first load data using dataInterface.load()")
            return -1
        fig, ax = plt.subplots()
        ax.plot(self.freq_axis/1e6,self.data[sample_number].psd1)
        plt.grid(linestyle = '--')
        plt.xticks(np.arange(-60,61,10))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power (counts)")
        plt.title("PSD channel 1, sample number " + str(sample_number))
        if ylimits != "":
            plt.ylim(ylimits)
        if xlimits != "":
            plt.xlim(xlimits)
        plt.yscale(_yscale)
        formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
        ax.xaxis.set_major_formatter(formatter1)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(self.freq_axis/1e6,self.data[sample_number].psd2)
        plt.grid(linestyle = '--')
        plt.xticks(np.arange(-60,61,10))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power (counts)")
        plt.title("PSD channel 2, sample number " + str(sample_number))
        if ylimits != "":
            plt.ylim(ylimits)
        if xlimits != "":
            plt.xlim(xlimits)
        plt.yscale(_yscale)
        formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
        ax.xaxis.set_major_formatter(formatter1)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(self.freq_axis/1e6,np.sqrt(self.data[sample_number].cc['real'].astype(float)**2 + self.data[sample_number].cc['imag'].astype(float)**2))
        plt.grid(linestyle = '--')
        plt.xticks(np.arange(-60,61,10))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Mag{cross correlation} (counts)")
        plt.title("Magnitude of cross correlation, sample number " + str(sample_number))
        if ylimits != "":
            plt.ylim(ylimits)
        if xlimits != "":
            plt.xlim(xlimits)
        plt.yscale(_yscale)
        formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
        ax.xaxis.set_major_formatter(formatter1)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(self.freq_axis/1e6,self.data[sample_number].cc['real'])
        plt.grid(linestyle = '--')
        plt.xticks(np.arange(-60,61,10))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Real{cross correlation} (counts)")
        plt.title("Real{cross corr.}, sample number " + str(sample_number))
        if ylimits != "":
            plt.ylim(ylimits)
        if xlimits != "":
            plt.xlim(xlimits)
        formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
        ax.xaxis.set_major_formatter(formatter1)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(self.freq_axis/1e6,self.data[sample_number].cc['imag'])
        plt.grid(linestyle = '--')
        plt.xticks(np.arange(-60,61,10))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Imaginary{cross correlation} (counts)")
        plt.title("Imag{cross corr.}, sample number " + str(sample_number))
        if ylimits != "":
            plt.ylim(ylimits)
        if xlimits != "":
            plt.xlim(xlimits)
        formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
        ax.xaxis.set_major_formatter(formatter1)
        fig.show()



    def find_spike(self, sample, chan, _range):
        """
        outputs the frequency of spikes with maximum values that fall within _range.
        sample: sample number to scan. Multiple sample scan not yet supported.
        chan: channel to scan. Multiple channel scan not yet supported.
        _range: (lowerboud, upperbound) tuple
        """
        if chan == 1:
            return self.freq_axis[np.where(np.logical_and(self.data[sample].psd1>_range[0], self.data[sample].psd1<_range[1]))]
        if chan == 2:
             return self.freq_axis[np.where(np.logical_and(self.data[sample].psd2>_range[0], self.data[sample].psd2<_range[1]))]

    
    def get_value_at(self, sample, freq, chan):
        """
        finds the frequency bin closest to input "freq" and returns the
        value of "sample", channel "chan" at that index. 
        """
        idx = np.zeros(len(freq),np.int)
        for i,frequency in enumerate(freq):
            idx[i] = (np.abs(self.freq_axis - frequency)).argmin()
        if chan == 'psd1':
            return self.data[sample].psd1[idx]
        if chan == 'psd2':
            return self.data[sample].psd2[idx]
        if chan == 'ccreal':
            return self.data[sample].cc['real'][idx]
        if chan == 'ccimag':
            return self.data[sample].cc['imag'][idx]

    def print_value_at(self, sample, freq, extra_values=0):
        """
        finds the frequency bin closet to "freq" and prints the values
        in sample # "sample" at that frequency.
        Optional extra_values will also print "extra_values" amount of frequency
        bins above and below.
        """
        idx = (np.abs(self.freq_axis -freq)).argmin()
        for i in range(idx - extra_values,idx + extra_values+1):
            cc_mag = np.sqrt((self.data[sample].cc[i]['real'].astype(float)**2) + (self.data[sample].cc[i]['imag'].astype(float)**2))
            print("values at " + str(self.freq_axis[i]) + "MHz:")
            print("     PSD channel 1:     " + "{:e}".format(self.data[sample].psd1[i]))
            print("     PSD channel 2:     " + "{:e}".format(self.data[sample].psd2[i]))
            print("     Cross correlation: (" + "{:e}".format(self.data[sample].cc[i]['real']) +", "+ "{:e}".format(self.data[sample].cc[i]['imag'])+")")
            print("     Cross corr. mag:   " + "{:e}".format(cc_mag))

    def get_freq_width(self):
        """
        returns the spacing between frequency bins
        """
        return np.abs(self.freq_axis[1] - self.freq_axis[0])

    def verify(self):
        """
        verifies data integrity by looking at the serial numbers.
        The serial numbers should all be spaced exactly 2**11 (accumulation length) apart.
        This functionality probably won't work once data reset signal is implemented.
        """
        serials = np.asarray(list(map(lambda x: x.serial, self.data[:])))
        if len(np.where(serials[1:]-serials[:-1] != 2**11)[0]) == 0:
            print("no data gaps detected. Data samples loaded: " + str(len(self.data)))
        else:
            missing = np.where(serials[1:]-serials[:-1] != 2**11)[0] + 1
            print("gaps found preceding the following indexes:")
            print(missing)
            print("Number of dropped data blocks at these indexes (based on expected serial gap of 2^11/block):")
            print((serials[missing]-serials[missing - 1])/(2**11) - 1)

    def get_serials(self):
        """
        Returns a numpy array of the serial numbers. Only reads the serial
        numbers from the file, and so works much faster than 'load' and
        'verify'.
        """
        serials=[]
        file_size = os.path.getsize(self.dataFile)
        serial_indicies = np.arange(0,file_size, (2**15 +8))
        with open(self.dataFile, "rb") as fd:
            for index in serial_indicies:
                fd.seek(index)
                raw_serial = fd.read(4)
                serials.append(struct.unpack('>I',raw_serial[:])[0])
        return np.asarray(serials)

    def psd1_spectral_sum(self):
        """
        Returns a numpy array where each element is the sum of the channel 1 psd
        across the entire spectrum of a particular sample. 
        """
        _psd1_spectral_sum = np.array(list(map(lambda sample : np.sum(sample.psd1) - sample.psd1[np.abs(self.freq_axis).argmin()],self.data)))
        return _psd1_spectral_sum

    def psd2_spectral_sum(self):
        """
        Returns a numpy array where each element is the sum of the channel 2 psd
        across the entire spectrum of a particular sample. 
        """
        _psd2_spectral_sum = np.array(list(map(lambda sample : np.sum(sample.psd2) - sample.psd2[np.abs(self.freq_axis).argmin()],self.data)))
        return _psd2_spectral_sum

    def cross_spectral_sum_real(self):
        """
        Returns a numpy array where each element is the sum of the real
        component across the entire spectrum of a particular sample. 
        """
        _cross_spectral_sum_real = np.array(list(map(lambda sample : np.sum(sample.cc['real']) - sample.cc['real'][np.abs(self.freq_axis).argmin()],self.data)))
        return _cross_spectral_sum_real

    def cross_spectral_sum_imag(self):
        """
        Returns a numpy array where each element is the sum of the imag
        component across the entire spectrum of a particular sample. 
        """
        _cross_spectral_sum_imag = np.array(list(map(lambda sample : np.sum(sample.cc['imag']) - sample.cc['imag'][np.abs(self.freq_axis).argmin()],self.data)))
        return _cross_spectral_sum_imag
            

class sample:
    """ Data from one sample. This version assumes data
        is structured as follows:
        32 bit serial number (4 bytes)
        32 bit diagnostic (3 LSB = callibration state, 29 MSB unused) (4 bytes)
        1024 bins, 64 bits/bin, channel 1 PSD (8192 bytes)
        1024 bins, 64 bits/bin, channel 2 PSD (8192 bytes)
        1024 bins, 64 bits/bin, cross correlation real component (8192 bytes)
        1024 bins, 64 bits/bin, cross correlation imaginary component (8192 bytes)

        For a total of 8192*4 +4 +4 bytes per data sample
        """
    type = "dataSample"
    def __init__(self, psd1 = [], psd2 = [], cc = [], serial = 0, call_state = 0, sample_size = (8192*4 + 4 + 4), fft_size = 1024):
        """
        serial: serial number associated with this data sample
        psd1: numpy array containing channel 1 PSD data, dtype = np.uint64
        psd2: numpy array containing channel 2 PSD data, dtype = np.uint64
        cc: numpy array containing cross correlation data, dtype = cint64 (defined in data_handler.py)
        """
        self.psd1 = psd1
        self.psd2 = psd2
        self.cc = cc
        self.serial = serial
        self.sample_size = sample_size
        self.fft_size = fft_size
        self.call_state = call_state

    def load(self, fd, recenter = 1):
        """loads 1 data sample at the current position of the file descriptor fd.
            recenter: roll the data to center DC"""
        raw_data = fd.read(self.sample_size)
        
        self.serial = struct.unpack('>I',raw_data[:4])[0]
        self.call_state = struct.unpack('>I',raw_data[4:8])[0]
        raw_data = raw_data[8:]
        

        self.psd1 = np.zeros(self.fft_size, dtype = np.uint64)
        self.psd2 = np.zeros(self.fft_size, dtype = np.uint64)
        
        for i in range(self.fft_size):
            self.psd1[i] = struct.unpack('>Q',raw_data[(i*8):((i+1)*8)])[0]
        raw_data = raw_data[self.fft_size*8:]

        for i in range(self.fft_size):
            self.psd2[i] = struct.unpack('>Q',raw_data[(i*8):((i+1)*8)])[0]
        raw_data = raw_data[self.fft_size*8:]

        self.cc = np.zeros(self.fft_size, dtype = cint64)
        for i in range(self.fft_size):
            self.cc[i]['real'] = struct.unpack('>q',raw_data[(i*8):((i+1)*8)])[0]
        raw_data = raw_data[self.fft_size*8:]
        for i in range(self.fft_size):
            self.cc[i]['imag'] = struct.unpack('>q',raw_data[(i*8):((i+1)*8)])[0]

        if recenter:
            self.psd1 = np.roll(self.psd1, self.fft_size//2)
            self.psd2 = np.roll(self.psd2, self.fft_size//2)
            self.cc = np.roll(self.cc, self.fft_size//2)

def check_directory(directory_name):
    """Used for checking the serial numbers of an entire directory.
    Files withen the directory must be titled in some alphabetical manner"""


    data_directory=directory_name
    final_serial = 0
    total_dropped = 0
    for file in sorted(os.listdir(data_directory)):
        temp = dataInterface(data_directory +"/" + file)
        serials = temp.get_serials()
        if serials[0] - final_serial ==2048:
            print("No gap detected between files")
        else:
            if file != sorted(os.listdir(data_directory))[0]:
                print("GAP DETECTED BETWEEN FILES. SIZE: "+str(serials[0]-final_serial))
                total_dropped += (serials[0] - final_serial)/2048 -1

        gaps = np.where((serials[1:]-serials[:-1])!=2048)[0] + 1
        if len(gaps) == 0:
            print("No gaps detected in " + str(file))
        else:
            print("#####GAPS DETECTED IN " + str(file) + ": " + str(gaps))
            if len(gaps)>10:
                print('######>10 data gaps. File possibly corrupt or memory full.####')
                print('Not including these gaps in final count.')
            else:
                total_dropped += len(gaps)

        final_serial = serials[-1]
    print("--------------------------------------------------")
    print("Total dropped chunks: " + str(total_dropped))