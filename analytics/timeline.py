import pandas
from sortedcontainers import SortedKeyList

class Timeline():
    """
    Represents an event timeline.

    Devices are added to the timeline at start times (rising edge) and removed from
    the timeline at end times (falling edge).

    Internally the timeline is quantized to integer seconds, so multiple event counts are
    coalesced into a single second.
    """

    def __init__(self, start, end, local=False, debug=False, **kwargs):
        """
        Initialize a new `Timeline` instance for the given range of time.

        Required positional arguments:

        :start: A python `datetime`, pandas `Timestamp`, or Unix timestamp for the beginning of the counting interval.

        :end: A python `datetime`, pandas `Timestamp`, or Unix timestamp for the end of the counting interval.

        Optional keyword arguments:

        :local: `False` (default) to assume Unix time; `True` to assume local time.

        :debug: `False` (default) to supress debug messages; `True` to print to stdout.
        """
        if start is None or end is None:
            raise TypeError(f"'NoneType' was unexpected for start and/or end. Expected datetime, Timestamp, or Unix timestamp")

        self.start = start
        self.start_int = self._ts2int(start)
        self.end = end
        self.end_int = self._ts2int(end)
        self.local = local
        self.debug = debug
        self._reset()

    def _reset(self):
        """
        Resets this timeline to initial state.
        """

        # Timeline contains two-element list objects of the form: [unix_seconds, delta]
        self.timeline = SortedKeyList(key=lambda x : x[0])

    def _int2ts(self, i):
        """
        Convert :i: to a Timestamp
        """
        return pandas.Timestamp(i, unit="s")

    def _ts2int(self, ts):
        """
        Try to convert :ts: to a integer
        """
        try:
            return int(ts.timestamp())
        except:
            return int(ts)

    def count(self, data):
        """
        Count device availability observed in data, over this counter's interval.

        :data: A `pandas.DataFrame` of records from the availability view.

        :returns: This `Timeline` instance.
        """
        self._reset()

        for index, row in data.iterrows():
            if self.local:
                self.add(row["start_time_local"], 1)
                self.add(row["end_time_local"], -1)
            else:
                self.add(row["start_time"], 1)
                self.add(row["end_time"], -1)

        return self

    def add(self, time, delta):
        """
        Adds the given event delta to the event timeline.

        :time: The time at which the event occured.

        :delta: The number of events to add/remove.
        """
        if time is None or time is pandas.NaT:
            t = self.end_int
        else:
            t = min(max(self._ts2int(time), self.start_int), self.end_int)

        for val in self.timeline.irange_key(t, t):
            val[1] += delta
            return

        # Not found.
        self.timeline.add([t, delta])

    def partition(self):
        """
        Returns the current interval partition as a `pandas.DataFrame`.
        """
        partition = []
        count = 0
        for i in range(0, len(self.timeline) - 1):
            count += self.timeline[i][1]
            partition.append({
                "start": self.timeline[i][0],
                "end": self.timeline[i+1][0],
                "delta": self.timeline[i+1][0] - self.timeline[i][0],
                "count": count,
                "start_date": self._int2ts(self.timeline[i][0]),
                "end_date": self._int2ts(self.timeline[i+1][0]),
            })

        return pandas.DataFrame.from_records(partition,
            columns=["start", "end", "delta", "count", "start_date", "end_date"])

    def delta_x(self):
        """
        :return: The ordered list of deltas for the given interval partition, or this interval's partition.
        """
        partition = self.partition()
        return partition["delta"]

    def norm(self):
        """
        Get the delta of the largest sub-interval in this interval's partition.
        """
        partition = self.partition()
        return max(self.delta_x())

    def dimension(self):
        """
        The number of sub-intervals in this interval's partition.
        """
        return len(self.partition())

    def average(self):
        """
        Estimate the average number of devices within this interval's partition.

        Use a Riemann sum to estimate, computing the area of each sub-interval in the partition:

        - height: the count of devices seen during that timeslice
        - width:  the length of the timeslice in seconds
        """
        if len(self.timeline) <= 1:
            return 0

        partition = self.partition()

        areas = partition.apply(lambda i: i["count"] * i["delta"], axis="columns")
        sigma = areas.agg("sum")

        # Compute the average value over this counter's interval
        delta = self._ts2int(self.end) - self._ts2int(self.start)
        return sigma / delta
