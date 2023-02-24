'''
Download data from remote server

Some test files are too big to put in git

This code will download test files from an ORR server:

http://gnome.orr.noaa.gov/py_gnome_testdata/

'''

import os
import urllib.request as urllib_request  # for python 3

from urllib.parse import urljoin

# maybe want to add this back in
# import progressbar as pb

DATA_SERVER = 'https://gnome.orr.noaa.gov/py_gnome_testdata/'


CHUNKSIZE = 1024 * 1024


def get_datafile(file_):
    """
    Function looks to see if file_ exists in local directory. If it exists,
    then it simply returns the 'file_' back as a string.
    If 'file_' does not exist in local filesystem, then it tries to download it
    from the gnome server:

    http://gnome.orr.noaa.gov/py_gnome_testdata

    If it successfully downloads the file, it puts it in the user specified
    path given in file_ and returns the 'file_' string.

    If file is not found or server is down, it rethrows the HTTPError raised
    by urllib2.urlopen

    :param file_: path to the file including filename
    :type file_: string
    :exception: raises urllib2.HTTPError if server is down or file not found
                on server
    :returns: returns the string 'file_' once it has been downloaded to
              user specified location
    """

    if os.path.exists(file_):
        return file_
    else:

        # download file, then return file_ path

        (path_, fname) = os.path.split(file_)
        if path_ == '':
            path_ = '.'     # relative to current path

        try:
            resp = urllib_request.urlopen(urljoin(DATA_SERVER, fname))
        except urllib_request.HTTPError as ex:
            ex.msg = ("{0}. '{1}' not found on server or server is down"
                      .format(ex.msg, fname))
            raise ex

        # # progress bar
        # widgets = [fname + ':      ',
        #            pb.Percentage(),
        #            ' ',
        #            pb.Bar(),
        #            ' ',
        #            pb.ETA(),
        #            ' ',
        #            pb.FileTransferSpeed(),
        #            ]

        # pbar = pb.ProgressBar(widgets=widgets,
        #                       maxval=int(resp.info().getheader('Content-Length'))
        #                       ).start()

        if not os.path.exists(path_):
            os.makedirs(path_)

        sz_read = 0
        with open(file_, 'wb') as fh:
            # while sz_read < resp.info().getheader('Content-Length')
            # goes into infinite recursion so break loop for len(data) == 0
            while True:
                data = resp.read(CHUNKSIZE)

                if len(data) == 0:
                    break
                else:
                    fh.write(data)
                    sz_read += len(data)

                    # if sz_read >= CHUNKSIZE:
                    #     pbar.update(CHUNKSIZE)

        # pbar.finish()
        return file_
